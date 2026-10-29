"""Background pipeline: video + photos in, finished clip with the original sound out.

State lives in runs/<id>/job.json and is written after every step, so a restart can resume
polling. A paid request is never sent twice automatically: once a request id is saved, only
polling is repeated.

A job can ask for several takes of the same generation; every finished take is scored for
how much the person looks like the reference photos and the best one becomes the result.
"""
import json
import math
import secrets
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import audio, faces, hosting, masking, media, photos, prompts, providers, templates
from .config import RUNS, has

MAX_SECONDS = 30
MIN_PROVIDER_SECONDS = 4
OVERLAP = 1.0  # seconds shared by neighbouring parts of a long clip, blended when joined
MAX_TAKES = 3
POLL_EVERY = 8
POLL_LIMIT = 3 * 3600

# Jobs spend most of their time waiting on the provider, so a few can run side by side.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="job")
_locks = {}
_cancel = set()


class Canceled(Exception):
    pass


# ---------- storage ----------

def folder(job_id):
    return RUNS / job_id


def load(job_id):
    path = folder(job_id) / "job.json"
    if not path.exists():
        raise KeyError(job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def save(job):
    path = folder(job["id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _lock(job_id):
    return _locks.setdefault(job_id, threading.Lock())


def update(job_id, **fields):
    with _lock(job_id):
        job = load(job_id)
        job.update(fields)
        save(job)
        return job


def mutate(job_id, fn):
    """Read-modify-write under the job lock (for nested fields such as takes)."""
    with _lock(job_id):
        job = load(job_id)
        fn(job)
        save(job)
        return job


def log(job_id, message, progress=None, stage=None):
    with _lock(job_id):
        job = load(job_id)
        if message:
            job["log"].append({"t": _now(), "msg": message})
            job["log"] = job["log"][-200:]
            job["stage"] = message
        if stage:
            job["stage_key"] = stage
        if progress is not None:
            job["progress"] = max(job.get("progress", 0), progress)
        save(job)
    if job_id in _cancel:
        raise Canceled()


def notice(job_id, text):
    mutate(job_id, lambda j: j["notices"].append(text) if text not in j["notices"] else None)


def list_jobs():
    jobs = []
    if RUNS.exists():
        for p in RUNS.glob("*/job.json"):
            try:
                jobs.append(json.loads(p.read_text(encoding="utf-8")))
            except ValueError:
                pass
    return sorted(jobs, key=lambda j: j["created"], reverse=True)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------- creation ----------

def create(video_file, video_name, photo_files, options):
    """photo_files: [(role, file-like)], role in body/face/extra/object. Validates before anything paid."""
    options = templates.apply(options.get("template"), options)
    provider = providers.get(options.get("provider") or None)
    ok, missing = provider.ready()
    if not ok:
        raise ValueError(f"Не задан {missing}. Откройте «Настройки» и вставьте ключ.")
    roles = [r for r, _ in photo_files]
    if options["mode"] == "object":
        if "object" not in roles:
            raise ValueError("Загрузите фото объекта, которым заменить.")
        photo_files = [(r, f) for r, f in photo_files if r == "object"]
    elif "body" not in roles and "face" not in roles:
        raise ValueError("Загрузите хотя бы одно фото: лицо или в полный рост.")
    else:
        photo_files = [(r, f) for r, f in photo_files if r != "object"]
    if len(photo_files) > provider.max_images:
        raise ValueError(f"{provider.label} принимает до {provider.max_images} фото.")
    model = (options.get("model") or "").strip()
    if model and model not in getattr(provider, "models", {}):
        raise ValueError(f"{provider.label} не знает модель «{model}».")
    takes = int(options.get("takes") or 1)
    if not 1 <= takes <= MAX_TAKES:
        raise ValueError(f"Дублей можно от 1 до {MAX_TAKES}.")

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    d = folder(job_id)
    d.mkdir(parents=True)
    try:
        ext = Path(video_name).suffix.lower() or ".mp4"
        source = d / f"source{ext if ext in ('.mp4', '.mov', '.webm', '.m4v', '.mkv') else '.mp4'}"
        with source.open("wb") as f:
            shutil.copyfileobj(video_file, f)
        info = media.probe(source)
        if info["duration"] < 1:
            raise ValueError("Видео короче 1 секунды.")

        refs = []
        counters = {}
        for role, file in photo_files:
            counters[role] = counters.get(role, 0) + 1
            name = f"photo-{role}{counters[role] if role == 'extra' else ''}.jpg"
            raw = d / (name + ".upload")
            with raw.open("wb") as f:
                shutil.copyfileobj(file, f)
            photos.prepare(raw, d / name)
            raw.unlink()
            refs.append({"role": role, "file": name, "send": name})
        refs.sort(key=_ref_order)
        photos.sheet([d / r["file"] for r in refs], d / "sheet.jpg")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise

    job = {
        "id": job_id,
        "created": _now(),
        "status": "queued",
        "stage": "В очереди",
        "stage_key": "queued",
        "progress": 0,
        "log": [],
        "source": source.name,
        "source_info": info,
        "refs": refs,
        "options": {
            "mode": options.get("mode", "swap"),
            "who": (options.get("who") or "person").strip()[:120],
            "scene": (options.get("scene") or "").strip()[:500],
            "extra": (options.get("extra") or "").strip()[:800],
            # An object is never hidden: the generator must see what it replaces.
            "masking": "none" if options["mode"] == "object" else options.get("masking", "auto"),
            "audio": options.get("audio", "auto"),
            "quality": options.get("quality", "draft"),
            "review": options.get("review", "auto"),
            "takes": takes,
            "enhance": options.get("enhance", "auto"),
            "face_mesh": options.get("face_mesh", "auto"),
            "provider": provider.name,
            "model": model,
            "template": options["template"],
            "style": options.get("style", ""),
            "photoreal": options.get("photoreal", True),
        },
        "notices": [],
        "takes": [],
        "chosen": None,
        "hd": None,
        "result": None,
        "error": None,
    }
    if info["duration"] > MAX_SECONDS + 0.05:
        job["notices"].append(f"Видео длиннее {MAX_SECONDS} с — беру первые {MAX_SECONDS} с.")
    save(job)
    start(job_id)
    return job


def _ref_order(ref):
    # Body first, then face, then extras: the prompt's image labels follow this order.
    return {"body": 0, "face": 1}.get(ref["role"], 2)


def start(job_id):
    _cancel.discard(job_id)
    _executor.submit(_guard, job_id, _prepare_and_generate)


def _guard(job_id, step):
    try:
        step(job_id)
    except Canceled:
        update(job_id, status="canceled", stage="Отменено")
    except Exception as exc:
        traceback.print_exc()
        job = load(job_id)
        message = str(exc) or type(exc).__name__
        if job.get("hd") and job["hd"].get("status") == "running":
            update(job_id, hd=dict(job["hd"], status="failed", error=message), stage="1080p: ошибка")
        else:
            update(job_id, status="failed", error=message, stage="Ошибка")


# ---------- preparation ----------

def _masking_method(choice):
    if choice != "auto":
        return choice
    if has("REPLICATE_API_TOKEN"):
        return "cloud"
    return "local" if faces.available() else "none"


def _audio_mode(choice, has_audio):
    if not has_audio:
        return "none"
    if choice != "auto":
        return choice
    return "vocals" if has("REPLICATE_API_TOKEN") else "full"


def _prepare_photos(job_id, max_images):
    """Check every photo, give the provider a clean full-body shot and a tight face close-up."""
    job = load(job_id)
    d = folder(job_id)
    refs = job["refs"]
    labels = {"face": "Фото лица", "body": "Фото в полный рост", "extra": "Доп. ракурс"}
    for ref in refs:
        try:
            ref["check"] = faces.analyze(d / ref["file"], ref["role"])
        except Exception as exc:
            ref["check"] = {"warnings": [], "error": str(exc)}
            continue
        for w in ref["check"]["warnings"]:
            notice(job_id, f"{labels[ref['role']]}: {w}")
        if ref["role"] == "body":
            try:
                clean = faces.clean_background(d / ref["file"], d / "photo-body-clean.jpg")
                if clean:
                    ref["send"] = clean.name
            except Exception as exc:
                notice(job_id, f"Фон фото в полный рост не очищен: {exc}")

    face_refs = [r for r in refs if r["role"] == "face"]
    tight = any(r.get("check", {}).get("face_ratio", 0) >= 0.3 for r in face_refs)
    if not tight and len(refs) < max_images:
        # The best source is the photo with the biggest face.
        candidates = sorted((r for r in refs if r.get("check", {}).get("faces")), key=lambda r: r["check"].get("face_px", 0), reverse=True)
        for src in candidates:
            crop = faces.face_closeup(d / src["file"], d / "photo-face-auto.jpg", src["check"])
            if crop:
                for r in face_refs:
                    r["role"] = "extra"  # the wide "face" photo still helps as another view
                refs.append({"role": "face", "file": crop.name, "send": crop.name, "auto": True})
                break
    refs.sort(key=_ref_order)
    photos.sheet([d / r["send"] for r in refs], d / "sheet.jpg")
    update(job_id, refs=refs)


def _prepare_and_generate(job_id):
    update(job_id, status="running", error=None, progress=0)
    job = load(job_id)
    d = folder(job_id)
    opts = job["options"]
    provider = providers.get(opts["provider"])
    status = lambda msg: log(job_id, msg)

    if opts.get("enhance", "auto") != "off" and opts["mode"] != "object" and faces.available():
        log(job_id, "Проверяю фото: лицо, резкость, свет; чищу фон", 2, "normalize")
        try:
            _prepare_photos(job_id, provider.max_images)
        except Exception as exc:
            notice(job_id, f"Подготовка фото пропущена: {exc}")

    log(job_id, "Привожу видео к 24 кадрам/с", 5, "normalize")
    meta = media.normalize_video(d / job["source"], d / "original.mp4", max_seconds=MAX_SECONDS)
    meta["frames"] = max(1, round(meta["duration"] * media.FPS))
    source_wav = None
    if job["source_info"]["has_audio"]:
        source_wav = d / "source-audio.wav"
        media.extract_audio(d / job["source"], source_wav, meta["duration"])
    update(job_id, meta=meta)

    method = _masking_method(opts["masking"])
    audio_mode = _audio_mode(opts["audio"], source_wav is not None)
    update(job_id, used={"masking": method, "audio": audio_mode, "face_mesh": False})

    # Audio separation and masking are independent remote calls: run them side by side.
    with ThreadPoolExecutor(max_workers=2) as pool:
        guide_audio = pool.submit(audio.guide_track, d, source_wav, audio_mode, status)
        log(job_id, "Скрываю человека в кадре", 10, "mask")
        qa = _mask(job_id, d, meta, method, opts, status)
        guide_audio = guide_audio.result()

    guide = d / "guidance-silent.mp4"
    if guide_audio:
        media.mux(guide, guide_audio, d / "guidance.mp4")
        guide = d / "guidance.mp4"
    update(job_id, qa=qa, guide=guide.name, lipsync=bool(guide_audio))

    needs_review = opts["review"] == "always" or (qa and not qa.get("ok") and opts["review"] == "auto")
    if needs_review:
        update(job_id, status="review", stage="Проверьте маску и подтвердите отправку", stage_key="review", progress=45)
        return
    _generate(job_id)


def _mask(job_id, d, meta, method, opts, status):
    if method == "none":
        shutil.copyfile(d / "original.mp4", d / "guidance-silent.mp4")
        media.thumbnail(d / "original.mp4", d / "preview.jpg", min(1.0, meta["duration"] / 2))
        return None
    depth = None
    if method == "cloud":
        with ThreadPoolExecutor(max_workers=2) as pool:
            masks = pool.submit(masking.segment_cloud, d / "original.mp4", d, opts["who"], status)
            depth_raw = pool.submit(masking.depth_cloud, d / "original.mp4", d, status)
            mask_dir = masks.result()
            depth = masking.align_depth(depth_raw.result(), d / "depth.mp4", meta["duration"])
    elif method == "local":
        mask_dir = masking.segment_local(d / "original.mp4", d, meta["width"], meta["height"], status)
    else:
        raise ValueError(f"Неизвестный режим маски «{method}».")
    log(job_id, None, 30)
    if opts.get("face_mesh", "auto") != "off" and faces.available():
        try:
            share = faces.track(d / "original.mp4", mask_dir, meta["width"], meta["height"], meta["frames"], status)
            mutate(job_id, lambda j: j["used"].update(face_mesh=share >= 0.3))
            if share < 0.3:
                (Path(mask_dir) / "face.json").unlink(missing_ok=True)
        except Exception as exc:
            notice(job_id, f"Сетка лица не построена: {exc}")
    log(job_id, None, 38)
    qa = masking.compose(d / "original.mp4", mask_dir, d, meta, depth=depth, status=status)
    masking.save_stats(d, qa)
    return qa


def approve(job_id):
    job = load(job_id)
    if job["status"] != "review":
        raise ValueError("Задача не ждёт подтверждения.")
    _cancel.discard(job_id)
    update(job_id, status="running")
    _executor.submit(_guard, job_id, _generate)


# ---------- generation ----------

def plan_segments(duration, limit, overlap=OVERLAP):
    """Parts no longer than `limit`, neighbours sharing `overlap` seconds for a blended join."""
    if duration <= limit + 1e-6:
        return [{"index": 0, "start": 0.0, "duration": round(duration, 3)}]
    count = math.ceil((duration - overlap) / (limit - overlap) - 1e-6)
    length = (duration + (count - 1) * overlap) / count
    return [{"index": i, "start": round(i * (length - overlap), 3), "duration": round(length, 3)} for i in range(count)]


def _inputs(job_id, hd=False):
    """Prompt, hosted reference images and hosted guide parts, shared by every take."""
    job = load(job_id)
    d = folder(job_id)
    opts = job["options"]
    provider = providers.get(opts["provider"])
    meta = job["meta"]
    local = getattr(provider, "local_files", False)
    used = job.get("used", {})

    guide = d / job["guide"]
    if meta["duration"] < MIN_PROVIDER_SECONDS:
        # Seedance needs ≥4 s: hold the last frame, the final file is cut back to the real length.
        padded = d / "guidance-padded.mp4"
        media.run(["-i", guide, "-vf", f"tpad=stop_mode=clone:stop_duration={MIN_PROVIDER_SECONDS - meta['duration'] + 0.1:.2f}",
                   "-af", "apad", "-t", str(MIN_PROVIDER_SECONDS), *media.venc(16), padded], "дополнение до 4 с")
        guide = padded
    plan = plan_segments(max(meta["duration"], MIN_PROVIDER_SECONDS), provider.max_seconds)

    def host(path):
        return str(path) if local else hosting.host(path)

    log(job_id, "Готовлю ссылки на файлы для Seedance", None if hd else 50, "upload")
    images = [providers.image_payload(provider, d / r["send"], host) for r in job["refs"]]
    parts = []
    for seg in plan:
        part = guide
        if len(plan) > 1:
            part = d / f"guide-part{seg['index']}.mp4"
            if not part.exists():
                media.cut(guide, part, seg["start"], seg["duration"])
        parts.append(host(part))
    prompt = prompts.build(
        opts["mode"], provider.tags, [r["role"] for r in job["refs"]], who=opts["who"],
        masked=used.get("masking") != "none", lipsync=job.get("lipsync", False),
        scene=opts["scene"], extra=opts["extra"], pose_lines=used.get("masking") == "local", face_mesh=used.get("face_mesh", False),
        style=opts.get("style", ""), photoreal=opts.get("photoreal", True),
    )
    webhook = "" if local or provider.name != "enhancor" else hosting.webhook_url()
    return {"plan": plan, "images": images, "parts": parts, "prompt": prompt, "webhook": webhook}


def _submit_takes(job_id, count, quality, hd=False):
    job = load(job_id)
    provider = providers.get(job["options"]["provider"])
    inp = _inputs(job_id, hd)
    update(job_id, prompt=inp["prompt"])
    if not hd:
        update(job_id, status="generating", takes=[], chosen=None)
    takes = []
    for t in range(count):
        take = {"index": t, "state": "pending", "segments": [dict(s, state=None) for s in inp["plan"]]}
        takes.append(take)
        _store_takes(job_id, takes, hd)
        for seg, url in zip(take["segments"], inp["parts"]):
            label = f"Отправляю в {provider.label}"
            if count > 1:
                label += f" · дубль {t + 1}/{count}"
            if len(inp["plan"]) > 1:
                label += f" · часть {seg['index'] + 1}/{len(inp['plan'])}"
            log(job_id, label, None if hd else 55, "generate")
            try:
                request_id, request = provider.submit(url, inp["images"], inp["prompt"], dict(job["meta"], duration=seg["duration"], model=job["options"].get("model")), quality, inp["webhook"])
            except Exception as exc:
                if t == 0:
                    raise
                # Earlier takes are already paid for: keep them, drop this one and stop sending more.
                take.update(state="failed", error=f"Не отправлен: {exc}")
                _store_takes(job_id, takes, hd)
                notice(job_id, f"Дубль {t + 1} не отправлен: {exc}")
                return
            seg.update(request_id=request_id, state="pending")
            # Saved right away: from here on this request is only ever polled.
            _store_takes(job_id, takes, hd)
            (folder(job_id) / f"request{'-hd' if hd else ''}-t{t}-p{seg['index']}.json").write_text(json.dumps(request, indent=2, ensure_ascii=False))


def _store_takes(job_id, takes, hd):
    def apply(job):
        if hd:
            job["hd"] = dict(job["hd"] or {}, takes=takes)
        else:
            job["takes"] = takes
    mutate(job_id, apply)


def _generate(job_id, hd=False):
    job = load(job_id)
    quality = "hd" if hd else job["options"]["quality"]
    _submit_takes(job_id, 1 if hd else job["options"].get("takes", 1), quality, hd)
    _collect(job_id, hd)


def _collect(job_id, hd=False):
    """Poll every saved request; assemble each take as soon as its parts are in; pick the best."""
    job = load(job_id)
    provider = providers.get(job["options"]["provider"])
    takes = (job["hd"] or {}).get("takes") if hd else job["takes"]
    for take in takes:
        # A take with a part that never got a request id can never finish.
        if take["state"] == "pending" and any(not s.get("request_id") for s in take["segments"]):
            take.update(state="failed", error="Часть дубля не была отправлена.")
    started = time.time()
    log(job_id, "Seedance генерирует видео… обычно 2–10 минут", None if hd else 60, "generate")
    while True:
        pending = [(t, s) for t in takes if t["state"] == "pending" for s in t["segments"] if s.get("state") == "pending"]
        for take, seg in pending:
            try:
                res = provider.poll(seg["request_id"])
            except Exception as exc:
                # Network hiccups must not lose a paid request: keep polling.
                seg["last_error"] = str(exc)[:200]
                continue
            if res["state"] == "done":
                path = folder(job_id) / f"gen{'-hd' if hd else ''}-t{take['index']}-p{seg['index']}.mp4"
                providers.download(res["url"], path)
                seg.update(state="done", file=path.name)
            elif res["state"] == "failed":
                seg.update(state="failed", error=res.get("error"))
                take.update(state="failed", error="Seedance: " + str(res.get("error")))
        for take in takes:
            if take["state"] == "pending" and all(s.get("state") == "done" for s in take["segments"]):
                try:
                    _assemble(job_id, take, hd)
                except Canceled:
                    raise
                except Exception as exc:
                    # One broken take must not sink the others.
                    traceback.print_exc()
                    take.update(state="failed", error=f"Сборка дубля: {exc}")
        _store_takes(job_id, takes, hd)
        done = sum(t["state"] in ("done", "failed") for t in takes)
        if not hd:
            log(job_id, None, 60 + int(30 * done / len(takes)))
        if all(t["state"] != "pending" for t in takes):
            break
        if time.time() - started > POLL_LIMIT:
            raise TimeoutError("Seedance не ответил за 3 часа. Запросы сохранены — нажмите «Повторить», новых списаний не будет.")
        time.sleep(POLL_EVERY)

    good = [t for t in takes if t["state"] == "done"]
    if not good:
        raise providers.ProviderError(takes[0].get("error") or "Seedance не вернул видео.")
    if hd:
        update(job_id, hd=dict(load(job_id)["hd"], status="done", result=good[0]["result"], takes=takes), stage="1080p готово")
        return
    best = max(good, key=lambda t: (t.get("score") or {}).get("score", -1))
    _choose(job_id, best["index"])
    if len(good) > 1:
        log(job_id, f"Лучший дубль: №{best['index'] + 1}" + (f" (сходство {best['score']['score']}%)" if best.get("score") else ""))
    update(job_id, status="done", progress=100, stage="Готово", stage_key="done")


def _assemble(job_id, take, hd):
    """Parts → one clip → original soundtrack → identity score."""
    d = folder(job_id)
    job = load(job_id)
    meta = job["meta"]
    tag = f"{'-hd' if hd else ''}-t{take['index']}"
    log(job_id, "Собираю видео и возвращаю оригинальный звук" + (f" · дубль {take['index'] + 1}" if len(job["takes"] or []) > 1 else ""), None if hd else 90, "finish")
    segs = sorted(take["segments"], key=lambda s: s["index"])
    generated = d / segs[0]["file"]
    if len(segs) > 1:
        generated = d / f"generated{tag}.mp4"
        media.crossfade_concat([d / s["file"] for s in segs], [s["duration"] for s in segs], OVERLAP, generated)
    result = d / f"result{tag}.mp4"
    media.restore_audio(generated, d / job["source"], result, meta["duration"])
    media.thumbnail(result, d / f"thumb{tag}.jpg", min(1.0, meta["duration"] / 2))
    take.update(state="done", result=result.name, thumb=f"thumb{tag}.jpg")
    if not hd and job["options"]["mode"] != "object" and faces.available():
        try:
            log(job_id, "Сравниваю лицо в видео с вашими фото")
            refs = [d / r["file"] for r in job["refs"] if r["role"] == "face"] or [d / r["file"] for r in job["refs"]]
            take["score"] = faces.score_video(result, refs)
        except faces.FaceError as exc:
            log(job_id, f"Оценка сходства: {exc}")
            notice(job_id, "Оценка сходства недоступна: не скачалась модель распознавания лиц (нужен интернет при первом запуске). Подробности в журнале.")
        except Exception as exc:
            log(job_id, f"Оценка сходства: {exc}")
            notice(job_id, "Оценка сходства недоступна — подробности в журнале.")


def _choose(job_id, index):
    def apply(job):
        take = next(t for t in job["takes"] if t["index"] == index)
        job["chosen"] = index
        job["result"] = take["result"]
        if job.get("hd") and job["hd"].get("take") != index and job["hd"].get("status") != "running":
            job["hd"] = None
    mutate(job_id, apply)
    shutil.copyfile(folder(job_id) / load(job_id)["takes"][index]["thumb"], folder(job_id) / "thumb.jpg")


def choose(job_id, index):
    job = load(job_id)
    if job["status"] != "done":
        raise ValueError("Выбирать можно, когда все дубли готовы.")
    if not any(t["index"] == index and t["state"] == "done" for t in job["takes"]):
        raise ValueError("Такого готового дубля нет.")
    if (job.get("hd") or {}).get("status") == "running":
        raise ValueError("Дождитесь окончания 1080p.")
    _choose(job_id, index)


# ---------- HD, retry, resume ----------

def request_hd(job_id):
    job = load(job_id)
    if job["status"] != "done":
        raise ValueError("Сначала дождитесь черновика.")
    if job["options"]["quality"] == "hd":
        raise ValueError("Это видео уже в 1080p.")
    if job.get("hd") and job["hd"].get("status") in ("running", "done"):
        raise ValueError("1080p уже запущено.")
    update(job_id, hd={"status": "running", "take": job["chosen"], "takes": [], "error": None}, stage="1080p: отправляю")
    _executor.submit(_guard, job_id, _hd)


def _hd(job_id):
    job = load(job_id)
    provider = providers.get(job["options"]["provider"])
    if not hasattr(provider, "complete_draft"):
        _generate(job_id, hd=True)
        return
    # The provider finishes the chosen draft with the same settings: no new creative run.
    local = getattr(provider, "local_files", False)
    webhook = "" if local or provider.name != "enhancor" else hosting.webhook_url()
    draft = next(t for t in job["takes"] if t["index"] == job["chosen"])
    take = {"index": 0, "state": "pending", "segments": []}
    for seg in draft["segments"]:
        request_id, request = provider.complete_draft(seg["request_id"], webhook)
        take["segments"].append({"index": seg["index"], "duration": seg["duration"], "request_id": request_id, "state": "pending"})
        _store_takes(job_id, [take], True)
        (folder(job_id) / f"request-hd-p{seg['index']}.json").write_text(json.dumps(request, indent=2))
    _collect(job_id, hd=True)


def _has_pending(takes):
    return any(t["state"] == "pending" and any(s.get("state") == "pending" for s in t["segments"]) for t in takes or [])


def retry(job_id):
    job = load(job_id)
    if job.get("kind") in ("photo", "image"):
        from . import photo

        return photo.retry(job_id)
    hd = job.get("hd") or {}
    if hd.get("status") == "failed":
        if _has_pending(hd.get("takes")):
            update(job_id, hd=dict(hd, status="running", error=None))
            _executor.submit(_guard, job_id, lambda j: _collect(j, hd=True))
            return
        update(job_id, hd=None)
        return request_hd(job_id)
    if job["status"] not in ("failed", "canceled"):
        raise ValueError("Повторить можно только после ошибки или отмены.")
    # A cancel requested earlier must not kill the retry on its first step.
    _cancel.discard(job_id)
    if _has_pending(job["takes"]):
        # Requests already paid for: only resume polling.
        update(job_id, status="generating", error=None)
        _executor.submit(_guard, job_id, _collect)
    elif job.get("guide") and (folder(job_id) / job["guide"]).exists():
        # The guide video is ready (masks were paid for already): only the generation is redone.
        update(job_id, status="running", error=None, takes=[])
        _executor.submit(_guard, job_id, _generate)
    else:
        update(job_id, takes=[], log=job["log"] + [{"t": _now(), "msg": "Повтор с начала"}])
        start(job_id)


def cancel(job_id):
    job = load(job_id)
    if job["status"] == "review":
        update(job_id, status="canceled", stage="Отменено")
    elif job["status"] in ("queued", "running"):
        _cancel.add(job_id)
    else:
        raise ValueError("Отправленную в Seedance генерацию отменить нельзя — она уже оплачена.")


def delete(job_id):
    job = load(job_id)
    if job["status"] in ("running", "generating") or (job.get("hd") or {}).get("status") == "running":
        raise ValueError("Задача ещё выполняется.")
    shutil.rmtree(folder(job_id))


def _resumable(takes):
    """Every unfinished part has a saved request id (nothing was half-sent)."""
    return all(s.get("state") in ("pending", "done", "failed") for t in takes for s in t["segments"]) and _has_pending(takes)


def resume_all():
    """After a restart: keep polling saved requests, never resubmit."""
    for job in list_jobs():
        if job.get("kind") in ("photo", "image"):
            from . import photo

            photo.resume(job)
            continue
        takes = job.get("takes") or []
        if job["status"] == "generating":
            if _resumable(takes):
                _executor.submit(_guard, job["id"], _collect)
            else:
                update(job["id"], status="failed", error="Сервер был перезапущен во время отправки. Проверьте кабинет провайдера, прежде чем нажимать «Повторить», чтобы не оплатить дважды.")
        elif job["status"] in ("queued", "running"):
            update(job["id"], status="failed", error="Подготовка прервана перезапуском сервера. Нажмите «Повторить».")
        hd = job.get("hd") or {}
        if hd.get("status") == "running":
            if _resumable(hd.get("takes") or []):
                _executor.submit(_guard, job["id"], lambda j: _collect(j, hd=True))
            else:
                update(job["id"], hd=dict(hd, status="failed", error="Перезапуск сервера во время отправки 1080p."))
