"""Sign in with ChatGPT (plan usage / token sharing) and calls to the Responses API.

The app spends the user's ChatGPT plan instead of an API key. Protocol as implemented by the
open-source launch partners (OpenCode, OpenClaw, Pi): public client with PKCE, the first-time
client id `dynamic_agent_client`, the issued client id returned on the callback and kept for
refreshes, loopback redirect on 127.0.0.1:1455. The grant covers Responses inference only:
no image generation, no other media APIs.
"""
import base64
import hashlib
import json
import secrets
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import requests

from .config import ROOT

AUTHORIZE_URL = "https://auth.openai.com/api/accounts/authorize"
TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
RESOURCE = "https://api.openai.com/v1"
FIRST_CLIENT_ID = "dynamic_agent_client"
AGENT_NAME = "Luna Genjutsu"
CALLBACK_PORT = 1455
CALLBACK_PATH = "/auth/callback"
REDIRECT_URI = f"http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}"
DIRECT_SCOPE = "chatgpt.tokens.use.direct"
SCOPE = f"openid profile email offline_access resource.invoke {DIRECT_SCOPE}"
STORE = ROOT / ".chatgpt.json"
EXPIRY_MARGIN = 180
USER_AGENT = "LunaGenjutsu/1.0"
# Preferred text models, best first; whatever the account lists is used when none match.
PREFERRED_MODELS = ["gpt-6.1-sol", "gpt-6-sol", "gpt-6.1-terra", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.5"]

ERRORS = {
    "subscription_sharing_usage_limit_exceeded": "Лимит подписки ChatGPT для сторонних приложений исчерпан. Подождите сброса лимита или поднимите лимит для Luna Genjutsu в настройках ChatGPT.",
    "subscription_sharing_user_not_eligible": "Этот аккаунт ChatGPT не может делиться подпиской с приложениями (нужен Go, Plus или Pro; рабочие пространства могут запрещать).",
    "subscription_sharing_unsupported_capability": "Подписка ChatGPT не поддерживает эту операцию в сторонних приложениях.",
    "subscription_sharing_route_not_supported": "Подписка ChatGPT не поддерживает этот запрос в сторонних приложениях.",
    "subscription_sharing_invalid_user": "Вход в ChatGPT больше недействителен — войдите заново.",
}

_lock = threading.Lock()
_login = {"server": None, "state": None, "verifier": None, "error": None, "done": False}


class ChatGPTError(RuntimeError):
    pass


# ---------- token storage ----------

def _load():
    try:
        return json.loads(STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(data):
    STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        STORE.chmod(0o600)
    except OSError:
        pass


def host_id():
    """OpenAI identifies each installation by a stable urn:uuid."""
    data = _load()
    if not data.get("host_id"):
        data["host_id"] = f"urn:uuid:{uuid.uuid4()}"
        _save(data)
    return data["host_id"]


def status():
    data = _load()
    return {
        "connected": bool(data.get("refresh") and data.get("client_id")),
        "email": data.get("email"),
        "model": data.get("model"),
        "login_pending": bool(_login["server"]) and not _login["done"],
        "login_error": _login["error"],
    }


def logout():
    data = _load()
    _save({"host_id": data.get("host_id")} if data.get("host_id") else {})


# ---------- login ----------

def _b64url(raw):
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _email_from_id_token(id_token):
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
        return claims.get("email")
    except Exception:
        return None


def _token_request(body):
    r = requests.post(TOKEN_URL, data=body, headers={"Accept": "application/json", "User-Agent": USER_AGENT}, timeout=30)
    if not r.ok:
        raise ChatGPTError(f"OpenAI отклонил вход ({r.status_code}): {r.text[:300]}")
    return r.json()


def _store_tokens(token, client_id, keep=None):
    scopes = (token.get("scope") or "").split()
    if DIRECT_SCOPE not in scopes:
        raise ChatGPTError("Вход прошёл, но без разрешения использовать подписку. При входе включите «Использовать мой план ChatGPT».")
    data = dict(keep or _load())
    data.update(
        client_id=client_id,
        access=token["access_token"],
        refresh=token["refresh_token"],
        expires=time.time() + float(token.get("expires_in") or 3600) - EXPIRY_MARGIN,
        scopes=scopes,
    )
    if token.get("id_token"):
        data["email"] = _email_from_id_token(token["id_token"]) or data.get("email")
    _save(data)
    return data


class _Callback(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _page(self, code, text):
        body = (f"<!doctype html><meta charset='utf-8'><title>Luna Genjutsu</title>"
                f"<body style='background:#0b0b0c;color:#f2f2f3;font:16px system-ui;display:grid;place-items:center;height:100vh;margin:0'>"
                f"<div style='text-align:center'><div style='font-size:40px;color:#ffd60a'>☾</div><p>{text}</p></div>"
                f"<script>setTimeout(()=>window.close(),1500)</script>").encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path != CALLBACK_PATH:
            self._page(404, "Не та страница.")
            return
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if q.get("error"):
                raise ChatGPTError(f"ChatGPT не подключён: {q['error']}")
            if q.get("state") != _login["state"]:
                raise ChatGPTError("Ответ входа не совпал с запросом (state). Начните вход заново.")
            client_id = (q.get("client_id") or "").strip()
            if not q.get("code") or not client_id:
                raise ChatGPTError("OpenAI не вернул код входа или client_id.")
            token = _token_request({
                "grant_type": "authorization_code", "client_id": client_id, "code": q["code"],
                "code_verifier": _login["verifier"], "redirect_uri": REDIRECT_URI, "resource": RESOURCE,
            })
            _store_tokens(token, client_id)
            _login.update(done=True, error=None)
            self._page(200, "ChatGPT подключён. Окно можно закрыть.")
        except Exception as exc:
            _login.update(done=True, error=str(exc))
            self._page(400, str(exc))
        threading.Thread(target=_stop_server, daemon=True).start()


def _stop_server():
    time.sleep(0.5)
    with _lock:
        server, _login["server"] = _login["server"], None
    if server:
        server.shutdown()
        server.server_close()


def start_login():
    """Start the loopback server and return the URL to open in the browser."""
    with _lock:
        if _login["server"]:
            # A second click on "sign in": free the port before binding it again.
            old, _login["server"] = _login["server"], None
            old.shutdown()
            old.server_close()
        verifier = _b64url(secrets.token_bytes(32))
        challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
        state = _b64url(secrets.token_bytes(32))
        try:
            server = HTTPServer(("127.0.0.1", CALLBACK_PORT), _Callback)
        except OSError:
            raise ChatGPTError(f"Порт {CALLBACK_PORT} занят — вероятно, незаконченным входом в Codex CLI или другом приложении. Закройте его и попробуйте снова.")
        _login.update(server=server, state=state, verifier=verifier, error=None, done=False)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    params = {
        "client_id": FIRST_CLIENT_ID,
        "agent_name_hint": AGENT_NAME,
        "ext_agent_host_id": host_id(),
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "resource": RESOURCE,
        "scope": SCOPE,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "nonce": _b64url(secrets.token_bytes(32)),
    }
    return f"{AUTHORIZE_URL}?{urlencode(params)}"


# ---------- API ----------

def access_token():
    data = _load()
    if not data.get("refresh") or not data.get("client_id"):
        raise ChatGPTError("Войдите через ChatGPT (кнопка вверху справа).")
    if data.get("expires", 0) > time.time():
        return data["access"]
    token = _token_request({"grant_type": "refresh_token", "client_id": data["client_id"], "refresh_token": data["refresh"], "resource": RESOURCE})
    return _store_tokens(token, data["client_id"], keep=data)["access"]


def _headers(token, stream=False):
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "User-Agent": USER_AGENT,
            "Accept": "text/event-stream" if stream else "application/json"}


def _raise_for(r):
    try:
        err = (r.json() or {}).get("error") or {}
    except ValueError:
        err = {}
    code = err.get("code") if isinstance(err, dict) else None
    if code in ERRORS:
        raise ChatGPTError(ERRORS[code])
    if r.status_code == 401:
        raise ChatGPTError("Вход в ChatGPT истёк — войдите заново.")
    message = err.get("message") if isinstance(err, dict) else None
    raise ChatGPTError(f"ChatGPT ответил ошибкой ({r.status_code}): {message or r.text[:300]}")


def model():
    """The account's best listed text model (cached after the first lookup)."""
    data = _load()
    if data.get("model") and data.get("model_checked", 0) > time.time() - 86400:
        return data["model"]
    r = requests.get(f"{RESOURCE}/models", headers=_headers(access_token()), timeout=30)
    if not r.ok:
        _raise_for(r)
    listed = [m.get("slug") for m in (r.json() or {}).get("models", []) if m.get("visibility") == "list" and m.get("slug")]
    if not listed:
        raise ChatGPTError("Для этого аккаунта ChatGPT нет доступных моделей.")
    choice = next((m for m in PREFERRED_MODELS if m in listed), listed[0])
    data = _load()
    data.update(model=choice, model_checked=time.time())
    _save(data)
    return choice


def respond(instructions, text, images=(), timeout=180):
    """One Responses call; images are local paths sent as data URLs. Returns the output text.

    Streams, like the reference clients, and keeps only the final text. Sampling parameters
    are left out: the plan-usage backend ignores them."""
    content = [{"type": "input_text", "text": text}]
    for path in images:
        mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
        content.append({"type": "input_image", "image_url": f"data:{mime};base64,{base64.b64encode(Path(path).read_bytes()).decode()}", "detail": "high"})
    body = {"model": model(), "instructions": instructions, "input": [{"role": "user", "content": content}], "store": False, "stream": True}
    for attempt in range(2):
        r = requests.post(f"{RESOURCE}/responses", headers=_headers(access_token(), stream=True), json=body, stream=True, timeout=timeout)
        if r.status_code == 401 and attempt == 0:
            data = _load()
            data["expires"] = 0  # force a refresh and try once more
            _save(data)
            continue
        break
    if not r.ok:
        _raise_for(r)
    return _read_stream(r)


def _read_stream(r):
    parts, final = [], None
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "response.output_text.delta":
            parts.append(event.get("delta", ""))
        elif kind == "response.completed":
            final = event.get("response") or {}
        elif kind in ("response.failed", "error"):
            err = (event.get("response") or {}).get("error") or event.get("error") or {}
            code = err.get("code") if isinstance(err, dict) else None
            raise ChatGPTError(ERRORS.get(code) or f"ChatGPT: {err.get('message') if isinstance(err, dict) else err}")
    if final:
        texts = [c.get("text", "") for item in final.get("output", []) if item.get("type") == "message"
                 for c in item.get("content", []) if c.get("type") == "output_text"]
        if any(texts):
            return "".join(texts)
    text = "".join(parts)
    if not text:
        raise ChatGPTError("ChatGPT не вернул текст.")
    return text
