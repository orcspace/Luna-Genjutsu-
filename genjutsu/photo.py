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

from PIL import Image

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


RATIOS = ["1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "21:9"]
MAX_FREE_REFS = 6


def create_free(prompt, ref_files, options):
    """Plain generation: an idea (any language) and optional reference pictures, no scene to edit."""
    provider = providers.get("dreamina")
    ok, missing = provider.ready()
    if not ok:
        raise ValueError(f"Не задан {missing}. Откройте «Настройки» и вставьте токен useapi.net.")
    prompt = (prompt or "").strip()[:4000]
    if not prompt and not ref_files:
        raise ValueError("Опишите, что нарисовать, или добавьте фото-референс.")
    model = options.get("image_model") or default_model()
    if model not in provider.image_models:
        raise ValueError(f"Неизвестная модель картинок «{model}».")
    count = int(options.get("count") or 1)
    if not 1 <= count <= MAX_VARIANTS:
        raise ValueError(f"Вариантов можно от 1 до {MAX_VARIANTS}.")
    ratio = options.get("ratio") or "auto"
    if ratio != "auto" and ratio not in RATIOS:
        raise ValueError(f"Неизвестный формат «{ratio}».")
    if ratio == "auto" and not ref_files:
        ratio = "1:1"
    if len(ref_files) > MAX_FREE_REFS:
        raise ValueError(f"Можно до {MAX_FREE_REFS} фото-референсов.")

    job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    d = jobs.folder(job_id)
    d.mkdir(parents=True)
    try:
        refs = []
        for i, file in enumerate(ref_files, start=1):
            raw = d / f"ref-{i}.upload"
            with raw.open("wb") as f:
                shutil.copyfileobj(file, f)
            info = photos.prepare(raw, d / f"photo-ref{i}.jpg")
            raw.unlink()
            refs.append({"role": "ref", "file": f"photo-ref{i}.jpg", "send": f"photo-ref{i}.jpg", **info})
        if refs:
            photos.sheet([d / r["file"] for r in refs], d / "sheet.jpg")
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise
    job = {
        "id": job_id,
        "kind": "image",
        "created": jobs._now(),
        "status": "queued",
        "stage": "В очереди",
        "stage_key": "queued",
        "progress": 0,
        "log": [],
        "idea": prompt,
        "refs": refs,
        "options": {"mode": "free", "template": "free", "image_model": model, "count": count,
                    "quality": options.get("quality", "std"), "ratio": ratio, "enhance": options.get("enhance", "auto"), "provider": "dreamina"},
        "notices": [],
        "requests": [],
        "images": [],
        "result": None,
        "error": None,
    }
    jobs.save(job)
    jobs._cancel.discard(job_id)
    jobs._executor.submit(_guard, job_id, _pipeline_free)
    return job


def _pipeline_free(job_id):
    jobs.update(job_id, status="running", error=None, progress=0)
    job = jobs.load(job_id)
    d = jobs.folder(job_id)
    opts = job["options"]
    idea = job["idea"]
    if opts.get("enhance") != "off" and chatgpt.status()["connected"]:
        jobs.log(job_id, "GPT превращает описание в промпт", 20, "prompt")
        text = photo_prompt.free_request(idea, len(job["refs"]), providers.Dreamina.image_models[opts["image_model"]])
        images = [d / r["send"] for r in job["refs"]]
        try:
            try:
                answer = chatgpt.respond(photo_prompt.FREE_GUIDE, text, images)
            except chatgpt.ChatGPTError as exc:
                if "не поддерживает" not in str(exc) or not images:
                    raise
                answer = chatgpt.respond(photo_prompt.FREE_GUIDE, text)
            result = photo_prompt.parse(answer)
            jobs.update(job_id, prompt=result["prompt"], summary=result["summary"], prompt_by="gpt")
        except Exception as exc:
            jobs.notice(job_id, f"GPT не улучшил промпт ({exc}) — отправляю ваше описание как есть.")
    job = jobs.load(job_id)
    if not job.get("prompt"):
        if opts.get("enhance") != "off" and not chatgpt.status()["connected"]:
            jobs.notice(job_id, "ChatGPT не подключён — отправляю описание как есть. Войдите через ChatGPT, чтобы GPT дописывал промпт.")
        jobs.update(job_id, prompt=idea or "A high-quality picture based on the reference images.", prompt_by="user", summary="")
    _generate(job_id)


def _canvas(job):
    """Width/height to pick the output ratio from: the explicit ratio, else the scene or first reference."""
    ratio = job["options"].get("ratio")
    if ratio and ratio != "auto":
        w, h = ratio.split(":")
        return int(w), int(h)
    info = job.get("scene_info") or (job["refs"][0] if job.get("refs") else {})
    return info.get("width", 1), info.get("height", 1)


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
    refs = ([d / job["guide"]] if job.get("guide") else []) + [d / r["send"] for r in job["refs"]]
    width, height = _canvas(job)
    count = count or opts["count"]
    jobs.update(job_id, status="generating")
    requests_ = list(job.get("requests") or [])
    start = len(requests_)
    for i in range(count):
        jobs.log(job_id, f"Dreamina рисует · {providers.Dreamina.image_models[opts['image_model']]}" + (f" · вариант {i + 1}/{count}" if count > 1 else ""), 40, "generate")
        try:
            jobid, payload = provider.submit_image(job["prompt"], refs, opts["image_model"], width, height, hd=opts.get("quality") == "4k")
        except Exception as exc:
            if len(requests_) > start or any(r["state"] == "pending" for r in requests_):
                jobs.notice(job_id, f"Вариант {i + 1} не отправлен — довожу уже отправленные.")
                break
            if job.get("images"):
                # A later round failed to start: the pictures already made stay, the job stays done.
                jobs.notice(job_id, f"Новый вариант не отправлен: {exc}")
                jobs.update(job_id, status="done", stage="Готово", stage_key="done", progress=100)
                return
            raise
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
    if job["options"]["mode"] not in ("object", "free") and faces.available():
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
    _thumbnail(d / images[0]["file"], d / "thumb.jpg")
    jobs.update(job_id, images=images, result=images[0]["file"], status="done", progress=100, stage="Готово", stage_key="done")


def _thumbnail(src, dst, size=640):
    """Small preview for the gallery, so cards don't pull 4K originals."""
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            im.save(dst, "JPEG", quality=85)
    except OSError:
        shutil.copyfile(src, dst)


def more(job_id, count=1):
    """More variants from the same prompt (a new paid generation)."""
    job = jobs.load(job_id)
    if job.get("kind") not in ("photo", "image") or job["status"] != "done":
        raise ValueError("Ещё варианты можно, когда готовы текущие.")
    count = max(1, min(MAX_VARIANTS, int(count)))
    jobs.update(job_id, status="running", stage="Ещё варианты", progress=40)
    jobs._executor.submit(_guard, job_id, lambda j: _generate(j, count))


def _remember_prompt(job_id, prompt, by, summary=""):
    def apply(job):
        job.setdefault("prompts", [])
        if job.get("prompt"):
            job["prompts"].append({"prompt": job["prompt"], "by": job.get("prompt_by"), "t": jobs._now()})
            job["prompts"] = job["prompts"][-20:]
        job.update(prompt=prompt, prompt_by=by, summary=summary or job.get("summary", ""))
    jobs.mutate(job_id, apply)


def regenerate(job_id, prompt, count=1):
    """Draw again with a prompt the user edited (a new paid generation)."""
    job = jobs.load(job_id)
    prompt = " ".join((prompt or "").split())
    if job.get("kind") not in ("photo", "image") or job["status"] != "done":
        raise ValueError("Перерисовать можно, когда готовы текущие варианты.")
    if len(prompt) < 20:
        raise ValueError("Промпт слишком короткий.")
    _remember_prompt(job_id, prompt[:5000], "user")
    jobs.update(job_id, status="running", stage="Перерисовываю с вашим промптом", progress=40)
    jobs._executor.submit(_guard, job_id, lambda j: _generate(j, max(1, min(MAX_VARIANTS, int(count)))))


def refine(job_id, image=None, count=1):
    """GPT looks at a result next to the references and rewrites the prompt to fix what is off."""
    job = jobs.load(job_id)
    if job.get("kind") not in ("photo", "image") or job["status"] != "done":
        raise ValueError("Улучшать можно, когда готовы текущие варианты.")
    if not chatgpt.status()["connected"]:
        raise ValueError("Для улучшения войдите через ChatGPT (кнопка вверху справа).")
    files = [im["file"] for im in job["images"]]
    image = image if image in files else files[0]
    jobs.update(job_id, status="running", stage="GPT сравнивает результат с вашими фото", progress=30, stage_key="prompt")
    jobs._executor.submit(_guard, job_id, lambda j: _refine(j, image, max(1, min(MAX_VARIANTS, int(count)))))


def _refine(job_id, image, count):
    job = jobs.load(job_id)
    d = jobs.folder(job_id)
    if job.get("kind") == "image":
        refs = job["refs"]
        text = "\n".join([
            photo_prompt.free_request(job.get("idea", ""), len(refs), providers.Dreamina.image_models[job["options"]["image_model"]]),
            f"Image {len(refs) + 1} is the GENERATED picture to review.",
            f"Prompt that produced it: {job['prompt']}",
        ])
        answer = chatgpt.respond(photo_prompt.FREE_REVIEW_GUIDE, text, [d / r["send"] for r in refs] + [d / image])
        result = photo_prompt.parse(answer)
        _remember_prompt(job_id, result["prompt"], "gpt-review", result["summary"])
        jobs.log(job_id, "GPT исправил промпт: " + (result["summary"] or "готово"))
        _generate(job_id, count)
        return
    template = templates.BY_ID[job["options"]["template"]]
    refs = job["refs"]
    roles = photo_prompt.image_roles(refs, job.get("masked"))
    score = next((im.get("score") for im in job["images"] if im["file"] == image), None)
    text = "\n".join([
        photo_prompt.request_text(template, job["options"], refs, job.get("masked"), providers.Dreamina.image_models[job["options"]["image_model"]]),
        f"- image {len(roles) + 1}: the GENERATED picture to review"
        + (f" (face likeness score {score['score']}/100)" if score and score.get("face") else ""),
        f"Prompt that produced it: {job['prompt']}",
    ])
    images = ([d / job["guide"]] if job.get("guide") else []) + [d / r["send"] for r in refs] + [d / image]
    try:
        answer = chatgpt.respond(photo_prompt.REVIEW_GUIDE, text, images)
    except chatgpt.ChatGPTError as exc:
        if "не поддерживает" in str(exc):
            raise chatgpt.ChatGPTError("Подписка ChatGPT не принимает картинки в сторонних приложениях, поэтому GPT не может сравнить результат. Отредактируйте промпт вручную.")
        raise
    result = photo_prompt.parse(answer)
    _remember_prompt(job_id, result["prompt"], "gpt-review", result["summary"])
    jobs.log(job_id, "GPT исправил промпт: " + (result["summary"] or "готово"))
    _generate(job_id, count)


def retry(job_id):
    job = jobs.load(job_id)
    jobs._cancel.discard(job_id)
    if any(r["state"] == "pending" for r in job.get("requests") or []):
        jobs.update(job_id, status="generating", error=None)
        jobs._executor.submit(_guard, job_id, _collect)
    elif job.get("prompt") and (job.get("guide") or job.get("kind") == "image"):
        jobs.update(job_id, status="running", error=None)
        jobs._executor.submit(_guard, job_id, _generate)
    else:
        jobs.update(job_id, requests=[], images=[])
        if job.get("kind") == "image":
            jobs._executor.submit(_guard, job_id, _pipeline_free)
        else:
            start(job_id)


def resume(job):
    if job["status"] == "generating" and any(r["state"] == "pending" for r in job.get("requests") or []):
        jobs._executor.submit(_guard, job["id"], _collect)
    elif job["status"] in ("queued", "running", "generating"):
        jobs.update(job["id"], status="failed", error="Прервано перезапуском сервера. Нажмите «Повторить».")

