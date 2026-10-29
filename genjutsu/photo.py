"""Photo jobs: a scene photo + your photos in, edited photos out.

GPT (on the user's ChatGPT plan) writes the prompt from the guide; Dreamina (on the user's
Dreamina subscription, via useapi.net) draws the images; every image is scored for likeness and
the best comes first. Shares storage, logging and the worker pool with video jobs.
"""
import json
import secrets
import shutil
import time
import traceback
from datetime import datetime

from . import chatgpt, faces, jobs, masking, photo_prompt, photos, providers, templates
from .config import setting

MAX_VARIANTS = 4
POLL_EVERY = 5
POLL_LIMIT = 30 * 60


def default_model():
    return setting("DREAMINA_IMAGE_MODEL") or "gpt-image-2.5-flare"


def create(scene_file, scene_name, photo_files, options):
    options = templates.apply(options.get("template"), options)
    provider = providers.get("dreamina")
    ok, missing = provider.ready()
    if not ok:
        raise ValueError(f"Не задан {missing}. Откройте «Настройки» и вставьте токен useapi.net.")
    model = options.get("image_model") or default_model()
    if model not in provider.image_models:
        raise ValueError(f"Неизвестная модель картинок «{model}».")
    count = int(options.get("count") or 1)
    if not 1 <= count <= MAX_VARIANTS:
        raise ValueError(f"Вариантов можно от 1 до {MAX_VARIANTS}.")
    roles = [r for r, _ in photo_files]
    if options["mode"] == "object":
        if "object" not in roles:
            raise ValueError("Загрузите фото объекта, которым заменить.")
        photo_files = [(r, f) for r, f in photo_files if r == "object"]
    elif "body" not in roles and "face" not in roles:
        raise ValueError("Загрузите хотя бы одно фото: лицо или в полный рост.")
    else:
        photo_files = [(r, f) for r, f in photo_files if r != "object"]
    if len(photo_files) + 1 > provider.max_image_refs:
        raise ValueError(f"Вместе со сценой можно до {provider.max_image_refs} фото.")

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    d = jobs.folder(job_id)
    d.mkdir(parents=True)
    try:
        raw = d / "scene.upload"
        with raw.open("wb") as f:
            shutil.copyfileobj(scene_file, f)
        scene_info = photos.prepare(raw, d / "scene.jpg")
        raw.unlink()
        refs, counters = [], {}
        for role, file in photo_files:
            counters[role] = counters.get(role, 0) + 1
            name = f"photo-{role}{counters[role] if role == 'extra' else ''}.jpg"
            raw = d / (name + ".upload")
            with raw.open("wb") as f:
                shutil.copyfileobj(file, f)
            photos.prepare(raw, d / name)
            raw.unlink()
            refs.append({"role": role, "file": name, "send": name})
        refs.sort(key=jobs._ref_order)
        photos.sheet([d / r["file"] for r in refs], d / "sheet.jpg")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise

    job = {
        "id": job_id,
        "kind": "photo",
        "created": jobs._now(),
        "status": "queued",
        "stage": "В очереди",
        "stage_key": "queued",
        "progress": 0,
        "log": [],
        "scene": "scene.jpg",
        "scene_info": scene_info,
        "refs": refs,
        "options": {
            "mode": options["mode"],
            "template": options["template"],
            "style": options.get("style", ""),
            "who": (options.get("who") or "person").strip()[:120],
            "scene": (options.get("scene") or "").strip()[:500],
            "extra": (options.get("extra") or "").strip()[:800],
            "hide": options.get("hide", "auto"),
            "enhance": options.get("enhance", "auto"),
            "image_model": model,
            "count": count,
            "quality": options.get("quality", "std"),
            "provider": "dreamina",
        },
        "notices": [],
        "requests": [],
        "images": [],
        "result": None,
        "error": None,
    }
    jobs.save(job)
    start(job_id)
    return job


def start(job_id):
    jobs._cancel.discard(job_id)
    jobs._executor.submit(_guard, job_id, _pipeline)


def _guard(job_id, step):
    try:
        step(job_id)
    except jobs.Canceled:
        jobs.update(job_id, status="canceled", stage="Отменено")
    except Exception as exc:
        traceback.print_exc()
        jobs.update(job_id, status="failed", error=str(exc) or type(exc).__name__, stage="Ошибка")


def _pipeline(job_id):
    jobs.update(job_id, status="running", error=None, progress=0)
    job = jobs.load(job_id)
    d = jobs.folder(job_id)
    opts = job["options"]
    template = templates.BY_ID[opts["template"]]

    if opts["mode"] != "object" and opts.get("enhance") != "off" and faces.available():
        jobs.log(job_id, "Проверяю ваши фото, делаю кроп лица и чистый фон", 5, "photos")
        try:
            jobs._prepare_photos(job_id, providers.get("dreamina").max_image_refs - 1)
        except Exception as exc:
            jobs.notice(job_id, f"Подготовка фото пропущена: {exc}")

    scene = d / "scene.jpg"
    masked = False
    if opts["mode"] != "object" and opts.get("hide", "auto") != "off" and faces.available():
        jobs.log(job_id, "Скрываю человека на фото-референсе", 15, "photos")
        try:
            if masking.hide_person_photo(scene, d / "scene-guide.jpg"):
                scene, masked = d / "scene-guide.jpg", True
            else:
                jobs.notice(job_id, "На фото-референсе не найден человек — отправляю фото как есть.")
        except Exception as exc:
            jobs.notice(job_id, f"Не удалось скрыть человека: {exc}")
    jobs.update(job_id, guide=scene.name, masked=masked)

    _write_prompt(job_id, template, scene, masked)
    _generate(job_id)


def _write_prompt(job_id, template, scene, masked):
    job = jobs.load(job_id)
    d = jobs.folder(job_id)
    opts = job["options"]
    refs = job["refs"]
    model = opts["image_model"]
    if chatgpt.status()["connected"]:
        jobs.log(job_id, "GPT пишет промпт по гайду", 25, "prompt")
        text = photo_prompt.request_text(template, opts, refs, masked, providers.Dreamina.image_models[model])
        images = [scene] + [d / r["send"] for r in refs]
        try:
            try:
                answer = chatgpt.respond(photo_prompt.GUIDE, text, images)
            except chatgpt.ChatGPTError as exc:
                if "не поддерживает" not in str(exc):
                    raise
                # Plan usage without image input: GPT still writes from the text description.
                answer = chatgpt.respond(photo_prompt.GUIDE, text)
            result = photo_prompt.parse(answer)
            jobs.update(job_id, prompt=result["prompt"], summary=result["summary"], prompt_by="gpt")
            return
        except Exception as exc:
            jobs.notice(job_id, f"GPT не написал промпт ({exc}) — использую встроенный.")
    else:
        jobs.notice(job_id, "ChatGPT не подключён — промпт собран по встроенному шаблону. Войдите через ChatGPT, чтобы его писал GPT.")
    result = photo_prompt.builtin(template, opts, refs, masked)
    jobs.update(job_id, prompt=result["prompt"], summary="", prompt_by="builtin")


def _generate(job_id, count=None):
    job = jobs.load(job_id)
    d = jobs.folder(job_id)
    opts = job["options"]
    provider = providers.get("dreamina")
    refs = [d / job["guide"]] + [d / r["send"] for r in job["refs"]]
    info = job["scene_info"]
    count = count or opts["count"]
    jobs.update(job_id, status="generating")
    requests_ = list(job.get("requests") or [])
    for i in range(count):
        jobs.log(job_id, f"Dreamina рисует · {providers.Dreamina.image_models[opts['image_model']]}" + (f" · вариант {i + 1}/{count}" if count > 1 else ""), 40, "generate")
        try:
            jobid, payload = provider.submit_image(job["prompt"], refs, opts["image_model"], info["width"], info["height"], hd=opts.get("quality") == "4k")
        except Exception:
            if not any(r["state"] == "pending" for r in requests_):
                raise
            jobs.notice(job_id, f"Вариант {i + 1} не отправлен — довожу уже отправленные.")
            break
        requests_.append({"jobid": jobid, "state": "pending"})
        # Saved right away: from here on it is only ever polled.
        jobs.update(job_id, requests=requests_)
        (d / f"image-request-{len(requests_)}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _collect(job_id)


def _collect(job_id):
    job = jobs.load(job_id)
    d = jobs.folder(job_id)
    provider = providers.get("dreamina")
    requests_ = job["requests"]
    images = list(job.get("images") or [])
    started = time.time()
    while any(r["state"] == "pending" for r in requests_):
        for req in [r for r in requests_ if r["state"] == "pending"]:
            try:
                res = provider.poll_image(req["jobid"])
            except Exception as exc:
                req["last_error"] = str(exc)[:200]
                continue
            if res["state"] == "done":
                for url in res["urls"]:
                    ext = ".png" if ".png" in url.lower() else ".webp" if ".webp" in url.lower() else ".jpg"
                    out = d / f"out-{len(images) + 1}{ext}"
                    providers.download(url, out)
                    images.append({"file": out.name, "score": None})
                req["state"] = "done"
            elif res["state"] == "failed":
                req.update(state="failed", error=res.get("error"))
        jobs.update(job_id, requests=requests_, images=images)
        done = sum(r["state"] != "pending" for r in requests_)
        jobs.log(job_id, None, 40 + int(45 * done / len(requests_)))
        if any(r["state"] == "pending" for r in requests_):
            if time.time() - started > POLL_LIMIT:
                raise TimeoutError("Dreamina не ответил за 30 минут. Задачи сохранены — нажмите «Повторить», новых списаний не будет.")
            time.sleep(POLL_EVERY)
    if not images:
        errors = [r.get("error") for r in requests_ if r.get("error")]
        raise providers.ProviderError(errors[0] if errors else "Dreamina не вернул картинок.")
    failed = [r for r in requests_ if r["state"] == "failed"]
    if failed:
        jobs.notice(job_id, f"{len(failed)} вариант(а) не получились: {failed[0].get('error')}")
    _score(job_id, images)


def _score(job_id, images):
    job = jobs.load(job_id)
    d = jobs.folder(job_id)
    if job["options"]["mode"] != "object" and faces.available():
        jobs.log(job_id, "Сравниваю лица с вашими фото", 90, "finish")
        refs = [d / r["file"] for r in job["refs"] if r["role"] == "face"] or [d / r["file"] for r in job["refs"]]
        try:
            identity = faces.Identity()
            for im in images:
                if im.get("score") is None:
                    im["score"] = faces.score_still(d / im["file"], refs, identity=identity)
            images.sort(key=lambda im: (im.get("score") or {}).get("score", -1), reverse=True)
        except Exception as exc:
            jobs.log(job_id, f"Оценка сходства: {exc}")
            jobs.notice(job_id, "Оценка сходства недоступна — подробности в журнале.")
    jobs.update(job_id, images=images, result=images[0]["file"], status="done", progress=100, stage="Готово", stage_key="done")


def more(job_id, count=1):
    """More variants from the same prompt (a new paid generation)."""
    job = jobs.load(job_id)
    if job.get("kind") != "photo" or job["status"] != "done":
        raise ValueError("Ещё варианты можно, когда готовы текущие.")
    count = max(1, min(MAX_VARIANTS, int(count)))
    jobs.update(job_id, status="running", stage="Ещё варианты", progress=40)
    jobs._executor.submit(_guard, job_id, lambda j: _generate(j, count))


def retry(job_id):
    job = jobs.load(job_id)
    jobs._cancel.discard(job_id)
    if any(r["state"] == "pending" for r in job.get("requests") or []):
        jobs.update(job_id, status="generating", error=None)
        jobs._executor.submit(_guard, job_id, _collect)
    elif job.get("prompt") and job.get("guide"):
        jobs.update(job_id, status="running", error=None)
        jobs._executor.submit(_guard, job_id, _generate)
    else:
        jobs.update(job_id, requests=[], images=[])
        start(job_id)


def resume(job):
    if job["status"] == "generating" and any(r["state"] == "pending" for r in job.get("requests") or []):
        jobs._executor.submit(_guard, job["id"], _collect)
    elif job["status"] in ("queued", "running", "generating"):
        jobs.update(job["id"], status="failed", error="Прервано перезапуском сервера. Нажмите «Повторить».")

