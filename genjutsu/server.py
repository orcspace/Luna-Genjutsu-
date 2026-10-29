"""Local UI and JSON API. Binds to 127.0.0.1 only; this is a single-user app."""
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import faces, hosting, jobs, media, photos, providers, templates
from .config import DEFAULTS, ROOT, SECRET_KEYS, has, masked, save_settings, setting

STATIC = ROOT / "static"
SERVABLE = re.compile(r"^(source\.[a-z0-9]+|photo-[a-z]+\d*(-clean|-auto)?\.jpg|sheet\.jpg|preview\.jpg|thumb(-hd)?(-t\d)?\.jpg|original\.mp4|mask\.mp4|guidance(-silent)?\.mp4|result(-hd)?(-t\d)?\.mp4)$")


@asynccontextmanager
async def lifespan(app):
    jobs.resume_all()
    yield
    hosting.stop_tunnel()


app = FastAPI(title="Luna Genjutsu", lifespan=lifespan)


@app.exception_handler(ValueError)
async def value_error(request, exc):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(KeyError)
async def missing(request, exc):
    return JSONResponse({"detail": "Задача не найдена."}, status_code=404)


def _local_only(request: Request):
    if request.client and request.client.host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "Только с этого компьютера.")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    try:
        import mediapipe  # noqa: F401

        local_masks = True
    except ImportError:
        local_masks = False
    try:
        media.ffmpeg()
        ffmpeg_ok = True
    except media.MediaError:
        ffmpeg_ok = False
    return {
        "provider": setting("PROVIDER"),
        "providers": [
            {"name": p.name, "label": p.label, "ready": p.ready()[0], "max_seconds": p.max_seconds, "max_images": p.max_images,
             "models": getattr(p, "models", {})}
            for p in providers.PROVIDERS.values()
        ],
        "replicate": has("REPLICATE_API_TOKEN"),
        "local_masks": local_masks,
        "ffmpeg": ffmpeg_ok,
        "hosting": hosting.status(),
        "settings": {k: (masked(setting(k)) if k in SECRET_KEYS else setting(k)) for k in DEFAULTS if k not in ("PORT", "PUBLIC_PORT")},
    }


@app.post("/api/settings")
async def settings(request: Request):
    _local_only(request)
    body = await request.json()
    # Masked placeholders coming back from the form mean "unchanged".
    updates = {k: str(v).strip() for k, v in body.items() if k in DEFAULTS and "…" not in str(v) and "••" not in str(v)}
    save_settings(updates)
    return config()


@app.post("/api/jobs")
def create_job(
    request: Request,
    video: UploadFile = File(...),
    body: Optional[UploadFile] = File(None),
    face: Optional[UploadFile] = File(None),
    object: Optional[UploadFile] = File(None),
    extras: List[UploadFile] = File(default_factory=list),
    consent: str = Form(""),
    mode: str = Form("swap"),
    who: str = Form("person"),
    scene: str = Form(""),
    extra: str = Form(""),
    masking: str = Form("auto"),
    audio: str = Form("auto"),
    quality: str = Form("draft"),
    review: str = Form("auto"),
    takes: int = Form(1),
    enhance: str = Form("auto"),
    face_mesh: str = Form("auto"),
    provider: str = Form(""),
    model: str = Form(""),
    template: str = Form("hero"),
    tpl_input: str = Form(""),
):
    _local_only(request)
    if consent != "yes":
        raise ValueError("Подтвердите, что на фото вы или человек, давший согласие.")
    _check_choice(mode, ("swap", "motion"))
    _check_choice(masking, ("auto", "cloud", "local", "none"))
    _check_choice(audio, ("auto", "vocals", "full", "none"))
    _check_choice(quality, ("draft", "hd"))
    _check_choice(review, ("auto", "always", "never"))
    _check_choice(enhance, ("auto", "off"))
    _check_choice(face_mesh, ("auto", "off"))
    files = [(role, f.file) for role, f in (("body", body), ("face", face), ("object", object)) if f is not None and f.filename]
    files += [("extra", f.file) for f in extras if f.filename]
    job = jobs.create(video.file, video.filename or "video.mp4", files, {
        "mode": mode, "who": who, "scene": scene, "extra": extra, "masking": masking,
        "audio": audio, "quality": quality, "review": review, "provider": provider or None,
        "takes": takes, "enhance": enhance, "face_mesh": face_mesh, "model": model, "template": template, "tpl_input": tpl_input,
    })
    return job


def _check_choice(value, allowed):
    if value not in allowed:
        raise ValueError(f"Недопустимое значение «{value}».")


@app.get("/api/jobs")
def list_jobs():
    return [{k: j.get(k) for k in ("id", "created", "status", "stage", "progress", "result", "hd", "options", "error", "chosen")} for j in jobs.list_jobs()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return jobs.load(_id(job_id))


@app.post("/api/jobs/{job_id}/approve")
def approve(job_id: str, request: Request):
    _local_only(request)
    jobs.approve(_id(job_id))
    return jobs.load(job_id)


@app.get("/api/templates")
def list_templates():
    return templates.public()


@app.post("/api/jobs/{job_id}/choose/{index}")
def choose(job_id: str, index: int, request: Request):
    _local_only(request)
    jobs.choose(_id(job_id), index)
    return jobs.load(job_id)


@app.post("/api/check-photo")
def check_photo(request: Request, photo: UploadFile = File(...), role: str = Form("face")):
    """Instant feedback while the user picks photos, before anything is paid for."""
    _local_only(request)
    _check_choice(role, ("face", "body", "extra", "object"))
    if role == "object" or not faces.available():
        return {"available": False, "warnings": []}
    with tempfile.TemporaryDirectory() as tmp:
        raw, prepared = Path(tmp) / "upload", Path(tmp) / "photo.jpg"
        raw.write_bytes(photo.file.read())
        photos.prepare(raw, prepared)
        try:
            report = faces.analyze(prepared, role)
        except faces.FaceError as exc:
            return {"available": False, "warnings": [str(exc)]}
    report.pop("box", None)
    return dict(report, available=True)


@app.post("/api/jobs/{job_id}/hd")
def hd(job_id: str, request: Request):
    _local_only(request)
    jobs.request_hd(_id(job_id))
    return jobs.load(job_id)


@app.post("/api/jobs/{job_id}/retry")
def retry(job_id: str, request: Request):
    _local_only(request)
    jobs.retry(_id(job_id))
    return jobs.load(job_id)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str, request: Request):
    _local_only(request)
    jobs.cancel(_id(job_id))
    return jobs.load(job_id)


@app.delete("/api/jobs/{job_id}")
def delete(job_id: str, request: Request):
    _local_only(request)
    jobs.delete(_id(job_id))
    return {"ok": True}


@app.get("/files/{job_id}/{name}")
def files(job_id: str, name: str, download: int = 0):
    if not SERVABLE.match(name):
        raise HTTPException(404)
    path = jobs.folder(_id(job_id)) / name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, filename=f"genjutsu-{job_id}-{name}" if download else None)


def _id(job_id):
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", job_id):
        raise KeyError(job_id)
    return job_id


app.mount("/static", StaticFiles(directory=STATIC), name="static")
