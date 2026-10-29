"""Give the provider links it can download, and a webhook it can call.

Default: a tiny second server on another localhost port that serves only the files registered
for a job (under random tokens) plus /hook, exposed through a temporary Cloudflare Quick Tunnel.
The UI port is never exposed. tmpfiles / litterbox are optional public temporary hosts.
"""
import mimetypes
import platform
import re
import secrets
import shutil
import stat
import subprocess
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

from .config import BIN, RUNS, setting

TOKEN_TTL = 24 * 3600


class HostingError(RuntimeError):
    pass


_files = {}  # token -> (path, expires_at)
_lock = threading.Lock()
_server = None
_tunnel = {"proc": None, "url": None, "error": None}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _file(self):
        m = re.fullmatch(r"/m/([A-Za-z0-9_-]{20,})/[^/]+", self.path.split("?")[0])
        if not m:
            return None
        with _lock:
            entry = _files.get(m.group(1))
        if not entry or entry[1] < time.time() or not Path(entry[0]).exists():
            return None
        return Path(entry[0])

    def _headers(self, path):
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()

    def do_HEAD(self):
        path = self._file()
        if not path:
            self.send_error(404)
            return
        self._headers(path)

    def do_GET(self):
        path = self._file()
        if not path:
            self.send_error(404)
            return
        self._headers(path)
        with path.open("rb") as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def do_POST(self):
        if self.path.split("?")[0] != "/hook":
            self.send_error(404)
            return
        body = self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 1 << 20))
        # Polling stays the source of truth; callbacks are only logged.
        RUNS.mkdir(exist_ok=True)
        with (RUNS / "webhooks.log").open("ab") as f:
            f.write(time.strftime("%Y-%m-%dT%H:%M:%S ").encode() + body[:4000] + b"\n")
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


def start_public_server():
    global _server
    if _server is None:
        port = int(setting("PUBLIC_PORT"))
        try:
            _server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
        except OSError:
            raise HostingError(f"Порт {port} для отдачи файлов занят. Закройте другую копию приложения или задайте другой PUBLIC_PORT в .env.")
        threading.Thread(target=_server.serve_forever, daemon=True).start()
    return _server


# ---------- cloudflared ----------

def _cloudflared_asset():
    system, machine = platform.system().lower(), platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    if system == "darwin":
        return f"cloudflared-darwin-{arch}.tgz"
    if system == "windows":
        return "cloudflared-windows-amd64.exe"
    return f"cloudflared-linux-{arch}"


def cloudflared_path():
    found = shutil.which("cloudflared")
    if found:
        return found
    target = BIN / ("cloudflared.exe" if platform.system() == "Windows" else "cloudflared")
    if target.exists():
        return str(target)
    asset = _cloudflared_asset()
    BIN.mkdir(exist_ok=True)
    url = f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}"
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    if asset.endswith(".tgz"):
        tmp = BIN / asset
        tmp.write_bytes(r.content)
        with tarfile.open(tmp) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("cloudflared"))
            target.write_bytes(tar.extractfile(member).read())
        tmp.unlink()
    else:
        target.write_bytes(r.content)
    target.chmod(target.stat().st_mode | stat.S_IEXEC)
    return str(target)


def _read_tunnel_output(proc):
    for line in proc.stderr:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m and not _tunnel["url"]:
            _tunnel["url"] = m.group(0)


def start_tunnel(timeout=45):
    if _tunnel["url"] and _tunnel["proc"] and _tunnel["proc"].poll() is None:
        return _tunnel["url"]
    start_public_server()
    try:
        binary = cloudflared_path()
    except Exception as exc:
        _tunnel["error"] = f"Не удалось скачать cloudflared: {exc}"
        raise HostingError(_tunnel["error"] + ". Установите его вручную или задайте PUBLIC_BASE_URL / MEDIA_HOST=tmpfiles.")
    _tunnel["url"] = None
    proc = subprocess.Popen([binary, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{setting('PUBLIC_PORT')}"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
    _tunnel["proc"] = proc
    threading.Thread(target=_read_tunnel_output, args=(proc,), daemon=True).start()
    started = time.time()
    while time.time() - started < timeout:
        if _tunnel["url"]:
            _tunnel["error"] = None
            return _tunnel["url"]
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    _tunnel["error"] = "cloudflared не выдал адрес туннеля."
    raise HostingError(_tunnel["error"])


def stop_tunnel():
    if _tunnel["proc"] and _tunnel["proc"].poll() is None:
        _tunnel["proc"].terminate()


def public_base():
    custom = setting("PUBLIC_BASE_URL").rstrip("/")
    if custom:
        start_public_server()
        return custom
    return start_tunnel()


def status():
    return {"mode": setting("MEDIA_HOST"), "public_url": setting("PUBLIC_BASE_URL") or _tunnel["url"], "error": _tunnel["error"]}


def webhook_url():
    return public_base() + "/hook"


# ---------- hosting ----------

def _register(path):
    token = secrets.token_urlsafe(24)
    with _lock:
        now = time.time()
        for k in [k for k, v in _files.items() if v[1] < now]:
            del _files[k]
        _files[token] = (str(Path(path).resolve()), now + TOKEN_TTL)
    return token


def _check(url, attempts=12):
    """Fresh trycloudflare names take a few seconds to resolve; retry before giving up."""
    last = None
    for _ in range(attempts):
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                if r.ok and "text/html" not in r.headers.get("Content-Type", ""):
                    first = next(r.iter_content(256), b"")
                    if first and not first.lstrip().lower().startswith((b"<!doctype", b"<html")):
                        return url
                last = f"HTTP {r.status_code}"
        except requests.RequestException as exc:
            last = type(exc).__name__
        time.sleep(3)
    raise HostingError(f"Ссылка на файл недоступна снаружи ({last}). Генерация не отправлена.")


def _tmpfiles(path):
    with open(path, "rb") as f:
        r = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": (Path(path).name, f)}, timeout=300)
    r.raise_for_status()
    url = r.json()["data"]["url"].replace("http://", "https://")
    return url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/", 1)


def _litterbox(path):
    with open(path, "rb") as f:
        r = requests.post("https://litterbox.catbox.moe/resources/internals/api.php",
                          data={"reqtype": "fileupload", "time": "24h"}, files={"fileToUpload": (Path(path).name, f)}, timeout=300)
    r.raise_for_status()
    return r.text.strip()


def host(path):
    """Public HTTPS URL for a local file, verified to download as media."""
    mode = setting("MEDIA_HOST")
    if mode == "tmpfiles":
        url = _tmpfiles(path)
    elif mode == "litterbox":
        url = _litterbox(path)
    else:
        url = f"{public_base()}/m/{_register(path)}/{Path(path).name}"
    if not url.startswith("https://"):
        raise HostingError("Хостинг не вернул HTTPS-ссылку.")
    return _check(url)
