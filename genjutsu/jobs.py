"""Background pipeline: video + photos in, finished clip with the original sound out.

State lives in runs/<id>/job.json and is written after every step, so a restart can resume
polling. A paid request is never sent twice automatically: once a request id is saved, only
polling is repeated.
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

from . import audio, hosting, masking, media, photos, prompts, providers
from .config import RUNS, has

MAX_SECONDS = 30
MIN_PROVIDER_SECONDS = 4
POLL_EVERY = 8
POLL_LIMIT = 3 * 3600

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="job")
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


def update(job_id, **fields):
    with _locks.setdefault(job_id, threading.Lock()):
        job = load(job_id)
        job.update(fields)
        save(job)
        return job


def log(job_id, message, progress=None, stage=None):
    with _locks.setdefault(job_id, threading.Lock()):
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
    """photo_files: [(role, file-like)], role in body/face/extra. Validates before anything paid."""
    provider = providers.get(options.get("provider") or None)
    ok, missing = provider.ready()
    if not ok:
        raise ValueError(f"Не задан {missing}. Откройте «Настройки» и вставьте ключ.")
    roles = [r for r, _ in photo_files]
    if "body" not in roles and "face" not in roles:
        raise ValueError("Загрузите хотя бы одно фото: лицо или в полный рост.")
    if len(photo_files) > provider.max_images:
        raise ValueError(f"{provider.label} принимает до {provider.max_images} фото.")

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
            refs.append({"role": role, "file": name})
        # Body first, then face: the prompt labels follow this order.
        refs.sort(key=lambda r: {"body": 0, "face": 1}.get(r["role"], 2))
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
            "masking": options.get("masking", "auto"),
            "audio": options.get("audio", "auto"),
            "quality": options.get("quality", "draft"),
            "review": options.get("review", "auto"),
            "provider": provider.name,
        },
        "notices": [],
        "segments": [],
        "hd": None,
        "result": None,
        "error": None,
    }
    if info["duration"] > MAX_SECONDS + 0.05:
        job["notices"].append(f"Видео длиннее {MAX_SECONDS} с — беру первые {MAX_SECONDS} с.")
    save(job)
    start(job_id)
    return job


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
            job["hd"].update(status="failed", error=message)
            update(job_id, hd=job["hd"], stage="1080p: ошибка")
        else:
            update(job_id, status="failed", error=message, stage="Ошибка")


# ---------- preparation ----------

def _masking_method(choice):
    if choice != "auto":
        return choice
    if has("REPLICATE_API_TOKEN"):
        return "cloud"
    try:
        import mediapipe  # noqa: F401

        return "local"
    except ImportError:
        return "none"


def _audio_mode(choice, has_audio):
    if not has_audio:
        return "none"
    if choice != "auto":
        return choice
    return "vocals" if has("REPLICATE_API_TOKEN") else "full"


def _prepare_and_generate(job_id):
    update(job_id, status="running", error=None, progress=0)
    job = load(job_id)
    d = folder(job_id)
    opts = job["options"]
    status = lambda msg: log(job_id, msg)

    log(job_id, "Привожу видео к 24 кадрам/с", 3, "normalize")
    meta = media.normalize_video(d / job["source"], d / "original.mp4", max_seconds=MAX_SECONDS)
    meta["frames"] = max(1, round(meta["duration"] * media.FPS))
    source_wav = None
    if job["source_info"]["has_audio"]:
        source_wav = d / "source-audio.wav"
        media.extract_audio(d / job["source"], source_wav, meta["duration"])
    update(job_id, meta=meta)

    method = _masking_method(opts["masking"])
    audio_mode = _audio_mode(opts["audio"], source_wav is not None)
    update(job_id, used={"masking": method, "audio": audio_mode})

    # Audio separation and masking are independent remote calls: run them side by side.
    with ThreadPoolExecutor(max_workers=2) as pool:
        guide_audio = pool.submit(audio.guide_track, d, source_wav, audio_mode, status)
        log(job_id, "Скрываю человека в кадре", 10, "mask")
        qa = _mask(job_id, d, meta, method, opts["who"], status)
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


def _mask(job_id, d, meta, method, who, status):
    if method == "none":
        shutil.copyfile(d / "original.mp4", d / "guidance-silent.mp4")
        media.thumbnail(d / "original.mp4", d / "preview.jpg", min(1.0, meta["duration"] / 2))
        return None
    depth = None
    if method == "cloud":
        with ThreadPoolExecutor(max_workers=2) as pool:
            masks = pool.submit(masking.segment_cloud, d / "original.mp4", d, who, status)
            depth_raw = pool.submit(masking.depth_cloud, d / "original.mp4", d, status)
            mask_dir = masks.result()
            depth = masking.align_depth(depth_raw.result(), d / "depth.mp4", meta["duration"])
    elif method == "local":
        mask_dir = masking.segment_local(d / "original.mp4", d, meta["width"], meta["height"], status)
    else:
        raise ValueError(f"Неизвестный режим маски «{method}».")
    log(job_id, None, 35)
    qa = masking.compose(d / "original.mp4", mask_dir, d, meta, depth=depth, status=status)
    masking.save_stats(d, qa)
    return qa


def approve(job_id):
    job = load(job_id)
    if job["status"] != "review":
        raise ValueError("Задача не ждёт подтверждения.")
    update(job_id, status="running")
    _executor.submit(_guard, job_id, _generate)


# ---------- generation ----------

def _segments(duration, limit):
    count = max(1, math.ceil(duration / limit - 1e-6))
    size = duration / count
    return [{"index": i, "start": round(i * size, 3), "duration": round(size, 3)} for i in range(count)]


def _generate(job_id, hd=False):
    job = load(job_id)
    d = folder(job_id)
    opts = job["options"]
    provider = providers.get(opts["provider"])
    meta = job["meta"]
    quality = "hd" if hd else opts["quality"]
    suffix = "-hd" if hd else ""
    local = getattr(provider, "local_files", False)

    guide = d / job["guide"]
    if meta["duration"] < MIN_PROVIDER_SECONDS:
        # Seedance needs ≥4 s: hold the last frame, the final file is cut back to the real length.
        padded = d / "guidance-padded.mp4"
        media.run(["-i", guide, "-vf", f"tpad=stop_mode=clone:stop_duration={MIN_PROVIDER_SECONDS - meta['duration'] + 0.1:.2f}",
                   "-af", "apad", "-t", str(MIN_PROVIDER_SECONDS), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", padded], "дополнение до 4 с")
        guide = padded
    segments = _segments(max(meta["duration"], MIN_PROVIDER_SECONDS), provider.max_seconds)

    def host(path):
        return str(path) if local else hosting.host(path)

    log(job_id, "Готовлю ссылки на файлы для Seedance", 50 if not hd else None, "upload")
    image_urls = [providers.image_payload(provider, d / r["file"], host) for r in job["refs"]]
    webhook = "" if local or provider.name != "enhancor" else hosting.webhook_url()
    prompt = prompts.build(
        opts["mode"], provider.tags, [r["role"] for r in job["refs"]], who=opts["who"],
        masked=job.get("used", {}).get("masking") != "none", lipsync=job.get("lipsync", False),
        scene=opts["scene"], extra=opts["extra"], pose_lines=job.get("used", {}).get("masking") == "local",
    )
    update(job_id, prompt=prompt, status="generating" if not hd else job["status"])

    for seg in segments:
        part = guide
        if len(segments) > 1:
            part = d / f"guide-part{seg['index']}{suffix}.mp4"
            media.cut(guide, part, seg["start"], seg["duration"])
        seg_meta = dict(meta, duration=seg["duration"])
        log(job_id, f"Отправляю в {provider.label}" + (f" · часть {seg['index'] + 1}/{len(segments)}" if len(segments) > 1 else ""), 55 if not hd else None, "generate")
        request_id, request = provider.submit(host(part), image_urls, prompt, seg_meta, quality, webhook)
        seg.update(request_id=request_id, state="pending")
        (d / f"request{suffix}-{seg['index']}.json").write_text(json.dumps(request, indent=2, ensure_ascii=False))
        _store_segments(job_id, segments, hd)

    _collect(job_id, hd)


def _store_segments(job_id, segments, hd):
    if hd:
        job = load(job_id)
        update(job_id, hd=dict(job["hd"] or {}, segments=segments))
    else:
        update(job_id, segments=segments)


def _collect(job_id, hd=False):
    """Poll every saved request, download, join, put back the original audio."""
    job = load(job_id)
    d = folder(job_id)
    provider = providers.get(job["options"]["provider"])
    segments = (job["hd"] or {}).get("segments") if hd else job["segments"]
    suffix = "-hd" if hd else ""
    started = time.time()
    log(job_id, "Seedance генерирует видео… обычно 2–10 минут", 60 if not hd else None, "generate")
    while True:
        pending = [s for s in segments if s.get("state") == "pending"]
        if not pending:
            break
        if time.time() - started > POLL_LIMIT:
            raise TimeoutError("Seedance не ответил за 3 часа. Запросы сохранены — нажмите «Повторить проверку», новых списаний не будет.")
        for seg in pending:
            try:
                res = provider.poll(seg["request_id"])
            except Exception as exc:
                # Network hiccups must not lose a paid request: keep polling.
                seg["last_error"] = str(exc)[:200]
                continue
            if res["state"] == "done":
                provider_file = d / f"gen{suffix}-{seg['index']}.mp4"
                providers.download(res["url"], provider_file)
                seg.update(state="done", file=provider_file.name)
            elif res["state"] == "failed":
                seg.update(state="failed", error=res.get("error"))
                _store_segments(job_id, segments, hd)
                raise providers.ProviderError("Seedance: " + str(res.get("error")))
        _store_segments(job_id, segments, hd)
        done = sum(s.get("state") == "done" for s in segments)
        if not hd:
            log(job_id, None, 60 + int(30 * done / len(segments)))
        if any(s.get("state") == "pending" for s in segments):
            time.sleep(POLL_EVERY)

    log(job_id, "Возвращаю оригинальный звук целиком", 92 if not hd else None, "finish")
    parts = [d / s["file"] for s in sorted(segments, key=lambda s: s["index"])]
    generated = parts[0]
    if len(parts) > 1:
        generated = d / f"generated{suffix}.mp4"
        media.concat(parts, generated)
    meta = load(job_id)["meta"]
    result = d / f"result{suffix}.mp4"
    media.restore_audio(generated, d / job["source"], result, meta["duration"])
    if hd:
        job = load(job_id)
        update(job_id, hd=dict(job["hd"], status="done", result=result.name), stage="1080p готово")
    else:
        media.thumbnail(result, d / "thumb.jpg", min(1.0, meta["duration"] / 2))
        update(job_id, status="done", result=result.name, progress=100, stage="Готово", stage_key="done")


# ---------- HD, retry, resume ----------

def request_hd(job_id):
    job = load(job_id)
    if job["status"] != "done":
        raise ValueError("Сначала дождитесь черновика.")
    if job["options"]["quality"] == "hd":
        raise ValueError("Это видео уже в 1080p.")
    if job.get("hd") and job["hd"].get("status") in ("running", "done"):
        raise ValueError("1080p уже запущено.")
    update(job_id, hd={"status": "running", "segments": [], "error": None}, stage="1080p: отправляю")
    _executor.submit(_guard, job_id, _hd)


def _hd(job_id):
    job = load(job_id)
    provider = providers.get(job["options"]["provider"])
    if hasattr(provider, "complete_draft"):
        # The provider finishes the accepted draft with the same settings: no new creative run.
        local = getattr(provider, "local_files", False)
        webhook = "" if local or provider.name != "enhancor" else hosting.webhook_url()
        segments = []
        for seg in job["segments"]:
            request_id, request = provider.complete_draft(seg["request_id"], webhook)
            segments.append({"index": seg["index"], "request_id": request_id, "state": "pending"})
            (folder(job_id) / f"request-hd-{seg['index']}.json").write_text(json.dumps(request, indent=2))
            _store_segments(job_id, segments, True)
        _collect(job_id, hd=True)
    else:
        _generate(job_id, hd=True)


def retry(job_id):
    job = load(job_id)
    hd = job.get("hd") or {}
    if hd.get("status") == "failed":
        if any(s.get("state") == "pending" for s in hd.get("segments", [])):
            update(job_id, hd=dict(hd, status="running", error=None))
            _executor.submit(_guard, job_id, lambda j: _collect(j, hd=True))
            return
        update(job_id, hd=None)
        return request_hd(job_id)
    if job["status"] not in ("failed", "canceled"):
        raise ValueError("Повторить можно только после ошибки или отмены.")
    pending = [s for s in job["segments"] if s.get("state") == "pending"]
    if pending:
        # Requests already paid for: only resume polling.
        update(job_id, status="generating", error=None)
        _executor.submit(_guard, job_id, _collect)
    elif job.get("guide") and (folder(job_id) / job["guide"]).exists():
        # The guide video is ready (masks were paid for already): only the generation is redone.
        update(job_id, status="running", error=None, segments=[])
        _executor.submit(_guard, job_id, _generate)
    else:
        update(job_id, segments=[], log=job["log"] + [{"t": _now(), "msg": "Повтор с начала"}])
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


def resume_all():
    """After a restart: keep polling saved requests, never resubmit."""
    for job in list_jobs():
        if job["status"] == "generating":
            pending = [s for s in job["segments"] if s.get("state") == "pending"]
            if pending and len(pending) == len([s for s in job["segments"] if s.get("state") != "done"]):
                _executor.submit(_guard, job["id"], _collect)
            else:
                update(job["id"], status="failed", error="Сервер был перезапущен во время отправки. Проверьте кабинет провайдера, прежде чем нажимать «Повторить», чтобы не оплатить дважды.")
        elif job["status"] in ("queued", "running"):
            update(job["id"], status="failed", error="Подготовка прервана перезапуском сервера. Нажмите «Повторить».")
        hd = job.get("hd") or {}
        if hd.get("status") == "running":
            if hd.get("segments") and all(s.get("state") in ("pending", "done") for s in hd["segments"]):
                _executor.submit(_guard, job["id"], lambda j: _collect(j, hd=True))
            else:
                update(job["id"], hd=dict(hd, status="failed", error="Перезапуск сервера во время отправки 1080p."))
