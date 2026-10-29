"""Seedance backends. Every paid call is made once; polling is the only thing that repeats."""
import base64
import hashlib
import math
import shutil
import time
from pathlib import Path
from urllib.parse import quote

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


class Dreamina:
    """Seedance on your Dreamina (CapCut) subscription through useapi.net.

    Files go up as account assets (no public links or tunnel needed); credits come from the
    Dreamina subscription. API from useapi.net's official examples (github.com/useapi/dreamina-api).
    """

    name = "dreamina"
    label = "Dreamina · Seedance (useapi.net)"
    api = "https://api.useapi.net/v1/dreamina"
    max_seconds = 15  # reference videos on the Seedance 2.0 family are capped at 15.4 s
    max_images = 9
    images_as_urls = False
    local_files = True  # receives local paths and uploads them itself
    tags = {"video": "@Video1", "image": lambda i: f"@Image{i}"}
    models = {
        "seedance-2.5": "Seedance 2.5 — лучшее качество, принимает реальные лица",
        "seedance-2.0": "Seedance 2.0",
        "seedance-2.0-fast": "Seedance 2.0 Fast — быстрее и дешевле",
        "seedance-2.0-mini": "Seedance 2.0 Mini — самый дешёвый, только 720p",
    }
    busy_retries = 6
    busy_wait = 10
    _assets = {}  # (account, sha1) -> assetRef, so takes and parts reuse uploads
    _account = None

    def _headers(self, **extra):
        token = setting("USEAPI_TOKEN")
        if not token:
            raise ProviderError("Нужен USEAPI_TOKEN (настройки → ключи).")
        return {"Authorization": f"Bearer {token}", "Accept": "application/json", **extra}

    def ready(self):
        return bool(setting("USEAPI_TOKEN")), "USEAPI_TOKEN"

    def account(self):
        """REGION:email of the connected account to use (assets and jobs must share it)."""
        wanted = setting("DREAMINA_ACCOUNT").strip()
        if self._account and (not wanted or wanted in (self._account, self._account.split(":", 1)[-1])):
            return self._account
        r = requests.get(f"{self.api}/accounts", headers=self._headers(), timeout=30)
        if not r.ok:
            raise ProviderError(f"useapi.net: не удалось получить аккаунты Dreamina ({r.status_code}): {r.text[:300]}")
        accounts = list((r.json() or {}).values())
        if not accounts:
            raise ProviderError("В useapi.net не подключён ни один аккаунт Dreamina: https://useapi.net/docs/start-here/setup-dreamina")
        if wanted:
            match = next((a for a in accounts if wanted in (a.get("account"), a.get("email"))), None)
            if not match:
                raise ProviderError(f"Аккаунт Dreamina «{wanted}» не найден в useapi.net.")
        else:
            match = next((a for a in accounts if not a.get("error")), accounts[0])
        if match.get("error"):
            raise ProviderError(f"У аккаунта {match.get('account')} ошибка: {match['error']}. Переподключите его в useapi.net.")
        Dreamina._account = match["account"]
        return self._account

    def upload(self, path, account):
        path = Path(path)
        digest = hashlib.sha1(path.read_bytes()).hexdigest()
        key = (account, digest)
        if key not in self._assets:
            ctype = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".mp4": "video/mp4"}.get(path.suffix.lower())
            if not ctype:
                raise ProviderError(f"Dreamina не принимает файлы {path.suffix}.")
            r = requests.post(f"{self.api}/assets/{quote(account, safe='')}", headers=self._headers(**{"Content-Type": ctype}), data=path.read_bytes(), timeout=600)
            if not r.ok:
                raise ProviderError(f"Dreamina не принял файл {path.name} ({r.status_code}): {r.text[:300]}")
            ref = r.json().get("assetRef")
            if not ref:
                raise ProviderError("Dreamina не вернул assetRef: " + r.text[:300])
            self._assets[key] = ref
        return self._assets[key]

    def submit(self, video_url, images, prompt, meta, quality, webhook):
        account = self.account()
        model = meta.get("model") or setting("DREAMINA_MODEL") or "seedance-2.5"
        if model not in self.models:
            raise ProviderError(f"Неизвестная модель Dreamina «{model}».")
        payload = {
            "account": account,
            "model": model,
            "prompt": prompt,
            "ratio": nearest_ratio(meta["width"], meta["height"]),
            "duration": int(max(4, min(self.max_seconds, math.ceil(meta["duration"] - 0.05)))),
            "resolution": "720p" if quality == "draft" or model == "seedance-2.0-mini" else "1080p",
            "omni_1_videoRef": self.upload(video_url, account),
        }
        for i, img in enumerate(images, start=1):
            payload[f"omni_{i}_imageRef"] = self.upload(img, account)
        for attempt in range(self.busy_retries + 1):
            r = requests.post(f"{self.api}/videos", headers=self._headers(**{"Content-Type": "application/json"}), json=payload, timeout=90)
            # 429 = every account is at capacity; nothing was queued or charged, so waiting is safe.
            if r.status_code != 429 or attempt == self.busy_retries:
                break
            time.sleep(self.busy_wait)
        if r.status_code == 402:
            raise ProviderError("Dreamina: закончились кредиты или подписка. Пополните на dreamina.capcut.com.")
        if r.status_code == 596:
            raise ProviderError("Dreamina: сессия аккаунта истекла — переподключите аккаунт в useapi.net.")
        if r.status_code == 429:
            raise ProviderError("Dreamina: все аккаунты заняты другими генерациями. Попробуйте позже.")
        if not r.ok:
            raise ProviderError(f"Dreamina отклонил запрос ({r.status_code}): {r.text[:500]}")
        ident = r.json().get("jobid")
        if not ident:
            raise ProviderError("Dreamina не вернул jobid: " + r.text[:300])
        return ident, payload

    def poll(self, request_id):
        # The job id goes into the path as-is, exactly like useapi.net's own example script.
        r = requests.get(f"{self.api}/videos/{request_id}", headers=self._headers(), timeout=30)
        r.raise_for_status()
        j = r.json()
        state = j.get("status")
        if state == "completed":
            result = j.get("response") or {}
            url = result.get("videoUrl") or result.get("videoUrlWatermarked")
            if not url:
                raise ProviderError("Dreamina завершил задачу без ссылки на видео.")
            return {"state": "done", "url": url, "raw": j}
        if state == "failed":
            detail = j.get("error") or "failed"
            if j.get("errorDetails"):
                detail += f" — {j['errorDetails']}"
            return {"state": "failed", "error": detail, "raw": j}
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
            media.run(["-i", src, "-an", "-vf", "hue=h=40:s=1.4,eq=contrast=1.05", *media.venc(20), out], "mock")
        return {"state": "done", "url": out.as_uri()}


PROVIDERS = {p.name: p for p in (Enhancor(), Dreamina(), BytePlus(), Mock())}


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
        from urllib.parse import urlparse
        from urllib.request import url2pathname

        # url2pathname turns /C:/Users/... back into C:\Users\... on Windows.
        shutil.copyfile(url2pathname(urlparse(url).path), dst)
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
