"""Seedance backends. Every paid call is made once; polling is the only thing that repeats."""
import base64
import math
import shutil
import time
from pathlib import Path

import requests

from . import media
from .config import setting

RATIOS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}


class ProviderError(RuntimeError):
    pass


def nearest_ratio(width, height):
    return min(RATIOS, key=lambda r: abs(RATIOS[r] - width / height))


class Enhancor:
    """Seedance through Enhancor: accepts human-face references (pass_faces), draft → 1080p."""

    name = "enhancor"
    label = "Enhancor · Seedance"
    max_seconds = 30
    max_images = 4
    images_as_urls = True
    tags = {"video": "@Video1", "image": lambda i: f"@Image{i}"}

    def _base(self):
        return setting("ENHANCOR_BASE_URL").rstrip("/") + "/"

    def _headers(self):
        key = setting("ENHANCOR_API_KEY")
        if not key:
            raise ProviderError("Нужен ENHANCOR_API_KEY (настройки → ключи).")
        return {"x-api-key": key}

    def ready(self):
        return bool(setting("ENHANCOR_API_KEY")), "ENHANCOR_API_KEY"

    def submit(self, video_url, images, prompt, meta, quality, webhook):
        payload = {
            "mode": "edit",
            "prompt": prompt,
            "videos": [video_url],
            "images": images,
            "audios": [],
            "resolution": "1080p",
            "aspect_ratio": "adaptive",
            "output_format": "mp4",
            "is_draft": quality == "draft",
            "pass_faces": True,
            "webhook_url": webhook,
        }
        return self._queue(payload), payload

    def complete_draft(self, request_id, webhook):
        payload = {"draft_id": request_id, "webhook_url": webhook}
        return self._queue(payload), payload

    def _queue(self, payload):
        r = requests.post(self._base() + "queue", headers=self._headers(), json=payload, timeout=90)
        if not r.ok:
            raise ProviderError(f"Enhancor отклонил запрос ({r.status_code}): {r.text[:500]}")
        ident = r.json().get("requestId")
        if not ident:
            raise ProviderError("Enhancor не вернул requestId: " + r.text[:300])
        return ident

    def poll(self, request_id):
        r = requests.post(self._base() + "status", headers=self._headers(), json={"request_id": request_id}, timeout=30)
        r.raise_for_status()
        j = r.json()
        state = j.get("status")
        if state == "COMPLETED":
            url = j.get("result")
            if not isinstance(url, str) or not url.startswith("http"):
                raise ProviderError("Enhancor завершил задачу без ссылки на видео.")
            return {"state": "done", "url": url, "raw": j}
        if state in ("FAILED", "CANCELED"):
            return {"state": "failed", "error": str(j.get("error") or state), "raw": j}
        return {"state": "pending", "raw": j}


class BytePlus:
    """Official BytePlus ModelArk. Its Seedance 2.x rejects real human faces in references
    unless they come from the private portrait library (asset://…)."""

    name = "byteplus"
    label = "BytePlus ModelArk · Seedance 2.0"
    max_seconds = 15
    max_images = 9
    images_as_urls = False  # sent inline as base64, so photos never sit on a public host
    tags = {"video": "video 1", "image": lambda i: f"image {i}"}

    def _base(self):
        return setting("BYTEPLUS_BASE_URL").rstrip("/")

    def _headers(self):
        key = setting("BYTEPLUS_API_KEY")
        if not key:
            raise ProviderError("Нужен BYTEPLUS_API_KEY (настройки → ключи).")
        return {"Authorization": f"Bearer {key}"}

    def ready(self):
        return bool(setting("BYTEPLUS_API_KEY")), "BYTEPLUS_API_KEY"

    def submit(self, video_url, images, prompt, meta, quality, webhook):
        content = [{"type": "text", "text": prompt}, {"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"}]
        content += [{"type": "image_url", "image_url": {"url": img}, "role": "reference_image"} for img in images]
        payload = {
            "model": setting("BYTEPLUS_MODEL"),
            "content": content,
            "ratio": nearest_ratio(meta["width"], meta["height"]),
            "duration": int(max(4, min(self.max_seconds, math.ceil(meta["duration"] - 0.05)))),
            "resolution": "480p" if quality == "draft" else "1080p",
            "generate_audio": False,
            "watermark": False,
        }
        r = requests.post(self._base() + "/contents/generations/tasks", headers=self._headers(), json=payload, timeout=90)
        if not r.ok:
            raise ProviderError(f"BytePlus отклонил запрос ({r.status_code}): {r.text[:500]}")
        ident = r.json().get("id")
        if not ident:
            raise ProviderError("BytePlus не вернул id задачи: " + r.text[:300])
        return ident, _redact(payload)

    def poll(self, request_id):
        r = requests.get(f"{self._base()}/contents/generations/tasks/{request_id}", headers=self._headers(), timeout=30)
        r.raise_for_status()
        j = r.json()
        state = j.get("status")
        if state == "succeeded":
            url = (j.get("content") or {}).get("video_url")
            if not url:
                raise ProviderError("BytePlus завершил задачу без ссылки на видео.")
            return {"state": "done", "url": url, "raw": j}
        if state in ("failed", "cancelled", "expired"):
            err = j.get("error") or {}
            return {"state": "failed", "error": err.get("message") or err.get("code") or state, "raw": j}
        return {"state": "pending", "raw": j}


class Mock:
    """Free dry run: returns the guide video recoloured after a few seconds. For testing the pipeline."""

    name = "mock"
    label = "Тест без оплаты (mock)"
    max_seconds = 30
    max_images = 9
    images_as_urls = False
    tags = Enhancor.tags
    local_files = True  # skips public hosting entirely
    _jobs = {}

    def ready(self):
        return True, ""

    def submit(self, video_url, images, prompt, meta, quality, webhook):
        ident = f"mock-{int(time.time() * 1000)}"
        self._jobs[ident] = {"video": video_url, "at": time.time()}
        return ident, {"video": video_url, "images": len(images), "prompt": prompt, "quality": quality}

    def complete_draft(self, request_id, webhook):
        ident = request_id + "-hd"
        self._jobs[ident] = dict(self._jobs.get(request_id) or {"video": None}, at=time.time())
        return ident, {"draft_id": request_id}

    def poll(self, request_id):
        job = self._jobs.get(request_id)
        if not job or not job["video"]:
            return {"state": "failed", "error": "mock: задача потерялась после перезапуска"}
        if time.time() - job["at"] < 3:
            return {"state": "pending"}
        src = Path(job["video"])
        out = src.with_name(f"mock-{request_id}.mp4")
        if not out.exists():
            media.run(["-i", src, "-an", "-vf", "hue=h=40:s=1.4,eq=contrast=1.05", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", out], "mock")
        return {"state": "done", "url": out.as_uri()}


PROVIDERS = {p.name: p for p in (Enhancor(), BytePlus(), Mock())}


def get(name=None):
    name = name or setting("PROVIDER")
    if name not in PROVIDERS:
        raise ProviderError(f"Неизвестный провайдер «{name}».")
    return PROVIDERS[name]


def image_payload(provider, path, host):
    if provider.images_as_urls:
        return host(path)
    if getattr(provider, "local_files", False):
        return str(path)
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def download(url, dst):
    if url.startswith("file://"):
        from urllib.parse import unquote, urlparse

        shutil.copyfile(unquote(urlparse(url).path), dst)
        return dst
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dst


def _redact(payload):
    """Request copy for job.json without megabytes of base64."""
    copy = dict(payload)
    copy["content"] = [
        dict(c, image_url={"url": c["image_url"]["url"][:40] + "…"}) if c.get("type") == "image_url" and c["image_url"]["url"].startswith("data:") else c
        for c in payload.get("content", [])
    ]
    return copy
