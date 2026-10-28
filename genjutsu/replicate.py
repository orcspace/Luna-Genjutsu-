"""Minimal Replicate HTTP client: upload a file, run a prediction, wait for it."""
import json
import time
from pathlib import Path

import requests

from .config import setting

API = "https://api.replicate.com/v1"


class ReplicateError(RuntimeError):
    pass


def _headers():
    token = setting("REPLICATE_API_TOKEN")
    if not token:
        raise ReplicateError("Нужен REPLICATE_API_TOKEN (настройки → ключи).")
    return {"Authorization": f"Bearer {token}"}


def upload(path):
    with open(path, "rb") as f:
        r = requests.post(f"{API}/files", headers=_headers(), files={"content": (Path(path).name, f)}, timeout=300)
    r.raise_for_status()
    return r.json()["urls"]["get"]


def latest_version(model):
    r = requests.get(f"{API}/models/{model}", headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()["latest_version"]


def run(version, inputs, log_file=None, status=None, timeout=1800):
    r = requests.post(f"{API}/predictions", headers=_headers(), json={"version": version, "input": inputs}, timeout=60)
    if not r.ok:
        raise ReplicateError(f"Replicate отклонил запрос ({r.status_code}): {r.text[:300]}")
    job = r.json()
    started = time.time()
    while job["status"] not in ("succeeded", "failed", "canceled"):
        if time.time() - started > timeout:
            raise ReplicateError(f"Replicate не ответил за {timeout // 60} мин (prediction {job['id']}).")
        time.sleep(4)
        r = requests.get(job["urls"]["get"], headers=_headers(), timeout=30)
        r.raise_for_status()
        job = r.json()
        if status and job["status"] == "processing":
            status(None)
    if log_file:
        Path(log_file).write_text(json.dumps({k: job.get(k) for k in ("id", "status", "error", "metrics", "version")}, indent=2))
    if job["status"] != "succeeded":
        raise ReplicateError(f"Replicate: {job.get('error') or job['status']}")
    return job["output"]


def download(url, dst):
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dst
