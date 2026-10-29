import base64
import json
import time
from urllib.parse import parse_qs, urlparse

import pytest
import requests

from genjutsu import chatgpt


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(chatgpt, "STORE", tmp_path / ".chatgpt.json")
    monkeypatch.setattr(chatgpt, "CALLBACK_PORT", 18455)
    monkeypatch.setattr(chatgpt, "REDIRECT_URI", "http://127.0.0.1:18455/auth/callback")
    yield tmp_path / ".chatgpt.json"
    chatgpt._stop_server()


class Resp:
    def __init__(self, data=None, status=200, lines=None):
        self.data, self.status_code, self.ok, self._lines = data, status, status < 400, lines or []
        self.text = json.dumps(data) if data is not None else ""

    def json(self):
        return self.data

    def iter_lines(self, decode_unicode=True):
        return iter(self._lines)


def id_token(email):
    payload = base64.urlsafe_b64encode(json.dumps({"email": email}).encode()).rstrip(b"=").decode()
    return f"h.{payload}.s"


def test_authorize_url_follows_the_open_source_flow(store):
    url = urlparse(chatgpt.start_login())
    q = {k: v[0] for k, v in parse_qs(url.query).items()}
    assert url.geturl().startswith("https://auth.openai.com/api/accounts/authorize?")
    assert q["client_id"] == "dynamic_agent_client" and q["agent_name_hint"] == "Luna Genjutsu"
    assert q["ext_agent_host_id"].startswith("urn:uuid:") and q["code_challenge_method"] == "S256"
    assert "chatgpt.tokens.use.direct" in q["scope"].split() and q["resource"] == "https://api.openai.com/v1"
    assert q["redirect_uri"] == "http://127.0.0.1:18455/auth/callback"
    # The host id is stable across logins.
    again = parse_qs(urlparse(chatgpt.start_login()).query)["ext_agent_host_id"][0]
    assert again == q["ext_agent_host_id"]


def test_callback_exchanges_code_and_keeps_issued_client_id(store, monkeypatch):
    sent = {}

    def post(url, data=None, headers=None, timeout=None):
        sent.update(url=url, data=data)
        return Resp({"access_token": "A", "refresh_token": "R", "expires_in": 3600, "scope": chatgpt.SCOPE, "id_token": id_token("me@x.com")})

    monkeypatch.setattr(chatgpt.requests, "post", post)
    q = parse_qs(urlparse(chatgpt.start_login()).query)
    r = requests.get(f"http://127.0.0.1:18455/auth/callback?code=C&state={q['state'][0]}&client_id=oaiapp_123", timeout=5)
    assert r.status_code == 200
    assert sent["data"]["client_id"] == "oaiapp_123" and sent["data"]["code_verifier"] and sent["data"]["grant_type"] == "authorization_code"
    st = chatgpt.status()
    assert st["connected"] and st["email"] == "me@x.com"


def test_callback_rejects_wrong_state(store):
    chatgpt.start_login()
    r = requests.get("http://127.0.0.1:18455/auth/callback?code=C&state=bad&client_id=x", timeout=5)
    assert r.status_code == 400 and not chatgpt.status()["connected"]
    assert "state" in chatgpt.status()["login_error"]


def test_grant_without_plan_scope_is_refused(store):
    with pytest.raises(chatgpt.ChatGPTError, match="подписку"):
        chatgpt._store_tokens({"access_token": "A", "refresh_token": "R", "expires_in": 60, "scope": "openid email"}, "c")


def test_refresh_uses_issued_client_id(store, monkeypatch):
    chatgpt._save({"client_id": "oaiapp_9", "access": "old", "refresh": "R", "expires": time.time() - 1})
    sent = {}
    monkeypatch.setattr(chatgpt.requests, "post", lambda url, data=None, **k: sent.update(data) or Resp({"access_token": "new", "refresh_token": "R2", "expires_in": 3600, "scope": chatgpt.SCOPE}))
    assert chatgpt.access_token() == "new"
    assert sent["grant_type"] == "refresh_token" and sent["client_id"] == "oaiapp_9"
    assert chatgpt._load()["refresh"] == "R2"


def test_respond_reads_the_stream_and_sends_images(store, monkeypatch, tmp_path):
    chatgpt._save({"client_id": "c", "access": "A", "refresh": "R", "expires": time.time() + 999, "model": "gpt-6-sol", "model_checked": time.time()})
    img = tmp_path / "p.jpg"
    img.write_bytes(b"\xff\xd8jpeg")
    seen = {}
    lines = ['data: {"type":"response.output_text.delta","delta":"{\\"prompt\\": "}', "",
             'data: {"type":"response.completed","response":{"output":[{"type":"message","content":[{"type":"output_text","text":"FINAL"}]}]}}']

    def post(url, headers=None, json=None, stream=None, timeout=None):
        seen.update(url=url, headers=headers, body=json)
        return Resp(lines=lines)

    monkeypatch.setattr(chatgpt.requests, "post", post)
    assert chatgpt.respond("GUIDE", "task", [img]) == "FINAL"
    assert seen["url"].endswith("/v1/responses") and seen["headers"]["Authorization"] == "Bearer A"
    body = seen["body"]
    assert body["model"] == "gpt-6-sol" and body["instructions"] == "GUIDE" and body["store"] is False
    assert "temperature" not in body
    content = body["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "task"} and content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_plan_errors_are_explained(store, monkeypatch):
    chatgpt._save({"client_id": "c", "access": "A", "refresh": "R", "expires": time.time() + 999, "model": "m", "model_checked": time.time()})
    monkeypatch.setattr(chatgpt.requests, "post", lambda *a, **k: Resp({"error": {"code": "subscription_sharing_usage_limit_exceeded"}}, 429))
    with pytest.raises(chatgpt.ChatGPTError, match="Лимит подписки"):
        chatgpt.respond("g", "t")


def test_model_picks_the_best_listed(store, monkeypatch):
    chatgpt._save({"client_id": "c", "access": "A", "refresh": "R", "expires": time.time() + 999})
    monkeypatch.setattr(chatgpt.requests, "get", lambda *a, **k: Resp({"models": [
        {"slug": "gpt-5.5", "visibility": "list"}, {"slug": "gpt-6-sol", "visibility": "list"}, {"slug": "gpt-6.1-sol", "visibility": "hide"}]}))
    assert chatgpt.model() == "gpt-6-sol"


def test_logout_keeps_the_host_id(store):
    hid = chatgpt.host_id()
    chatgpt._save(dict(chatgpt._load(), client_id="c", refresh="R"))
    chatgpt.logout()
    assert not chatgpt.status()["connected"] and chatgpt.host_id() == hid
