import json

import pytest

from genjutsu import prompts, providers


class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status
        self.ok = status < 400
        self.text = json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(self.status_code)


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ENHANCOR_API_KEY", "enh-key")
    monkeypatch.setenv("BYTEPLUS_API_KEY", "bp-key")


def test_enhancor_request_shape(keys, monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, json=json)
        return FakeResponse({"requestId": "R1"})

    monkeypatch.setattr(providers.requests, "post", post)
    ident, payload = providers.Enhancor().submit("https://h/v.mp4", ["https://h/b.jpg", "https://h/f.jpg"], "P", {"duration": 5, "width": 540, "height": 960}, "draft", "https://h/hook")
    assert ident == "R1"
    assert sent["url"].endswith("/queue") and sent["headers"] == {"x-api-key": "enh-key"}
    body = sent["json"]
    assert body["mode"] == "edit" and body["videos"] == ["https://h/v.mp4"] and body["images"] == ["https://h/b.jpg", "https://h/f.jpg"]
    assert body["is_draft"] is True and body["pass_faces"] is True and body["webhook_url"] == "https://h/hook"


def test_enhancor_poll_states(keys, monkeypatch):
    replies = iter([{"status": "IN_PROGRESS"}, {"status": "COMPLETED", "result": "https://cdn/x.mp4"}, {"status": "FAILED", "error": "nope"}])
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: FakeResponse(next(replies)))
    p = providers.Enhancor()
    assert p.poll("R")["state"] == "pending"
    assert p.poll("R") == {"state": "done", "url": "https://cdn/x.mp4", "raw": {"status": "COMPLETED", "result": "https://cdn/x.mp4"}}
    assert p.poll("R")["error"] == "nope"


def test_byteplus_request_shape(keys, monkeypatch):
    sent = {}

    def post(url, headers, json, timeout):
        sent.update(url=url, headers=headers, json=json)
        return FakeResponse({"id": "cgt-1"})

    monkeypatch.setattr(providers.requests, "post", post)
    ident, stored = providers.BytePlus().submit("https://h/v.mp4", ["data:image/jpeg;base64," + "A" * 500], "P", {"duration": 7.4, "width": 540, "height": 960}, "hd", "")
    body = sent["json"]
    assert ident == "cgt-1" and sent["url"].endswith("/contents/generations/tasks")
    assert sent["headers"] == {"Authorization": "Bearer bp-key"}
    roles = [c.get("role") for c in body["content"]]
    assert roles == [None, "reference_video", "reference_image"]
    assert body["duration"] == 8 and body["ratio"] == "9:16" and body["resolution"] == "1080p" and body["generate_audio"] is False
    # The copy stored in the job folder has no base64 blob.
    assert len(json.dumps(stored)) < 1000


def test_byteplus_poll(keys, monkeypatch):
    replies = iter([{"status": "running"}, {"status": "succeeded", "content": {"video_url": "https://x/v.mp4"}}, {"status": "failed", "error": {"message": "face detected"}}])
    monkeypatch.setattr(providers.requests, "get", lambda *a, **k: FakeResponse(next(replies)))
    p = providers.BytePlus()
    assert p.poll("t")["state"] == "pending"
    assert p.poll("t")["url"] == "https://x/v.mp4"
    assert p.poll("t")["error"] == "face detected"


def test_rejection_message_is_kept(keys, monkeypatch):
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: FakeResponse({"error": "bad"}, 400))
    with pytest.raises(providers.ProviderError, match="400"):
        providers.Enhancor().submit("u", [], "p", {"duration": 5, "width": 1, "height": 1}, "draft", "w")


def test_prompt_labels_follow_reference_order():
    text = prompts.build("swap", providers.Enhancor.tags, ["body", "face"], who="man in red", masked=True, lipsync=True)
    assert "Replace only the man in red in @Video1" in text
    assert "@Image1 shows the full body" in text and "@Image2 is a face close-up" in text
    assert "depth silhouette" in text and "Sync the lips" in text and "no watermark" in text
    motion = prompts.build("motion", providers.BytePlus.tags, ["face"], scene="a beach at sunset")
    assert "video 1" in motion and "a beach at sunset" in motion and "image 1 is a face close-up" in motion


def test_segments_overlap_and_cover_the_clip():
    from genjutsu.jobs import plan_segments
    assert plan_segments(15, 15) == [{"index": 0, "start": 0.0, "duration": 15}]
    segs = plan_segments(29.5, 15, overlap=1.0)
    assert len(segs) == 3 or len(segs) == 2
    assert all(s["duration"] <= 15 for s in segs)
    # Neighbours share exactly the overlap and the last part ends at the clip end.
    for a, b in zip(segs, segs[1:]):
        assert abs(a["start"] + a["duration"] - b["start"] - 1.0) < 0.01
    assert abs(segs[-1]["start"] + segs[-1]["duration"] - 29.5) < 0.01
    assert all(s["duration"] >= 4 for s in plan_segments(16, 15))


def test_prompt_mentions_guide_lines_only_when_drawn():
    plain = prompts.build("swap", providers.Enhancor.tags, ["face"], masked=True)
    assert "guidance only and must not appear" not in plain
    both = prompts.build("swap", providers.Enhancor.tags, ["face"], masked=True, pose_lines=True, face_mesh=True)
    assert "pose lines" in both and "eyebrows and lips" in both and "must not appear" in both


@pytest.fixture
def dreamina(monkeypatch):
    monkeypatch.setenv("USEAPI_TOKEN", "user:1-abc")
    monkeypatch.setenv("DREAMINA_ACCOUNT", "")
    monkeypatch.setattr(providers.Dreamina, "_account", None)
    monkeypatch.setattr(providers.Dreamina, "_assets", {})
    monkeypatch.setattr(providers.Dreamina, "busy_wait", 0)
    return providers.Dreamina()


def test_dreamina_uploads_assets_once_and_submits_omni_refs(dreamina, monkeypatch, tmp_path):
    video, body, face = tmp_path / "g.mp4", tmp_path / "b.jpg", tmp_path / "f.jpg"
    video.write_bytes(b"video")
    body.write_bytes(b"body")
    face.write_bytes(b"face")
    calls = []

    def get(url, headers, timeout):
        calls.append(("GET", url, headers))
        return FakeResponse({"US:me@x.com": {"account": "US:me@x.com", "email": "me@x.com"}})

    def post(url, headers, timeout, data=None, json=None):
        calls.append(("POST", url, headers, data, json))
        if "/assets/" in url:
            return FakeResponse({"assetRef": "asset-" + data.decode()})
        return FakeResponse({"jobid": "job-1"})

    monkeypatch.setattr(providers.requests, "get", get)
    monkeypatch.setattr(providers.requests, "post", post)
    meta = {"duration": 7.3, "width": 540, "height": 960, "model": None}
    ident, payload = dreamina.submit(str(video), [str(body), str(face)], "P", meta, "draft", "")
    assert ident == "job-1"
    assert calls[0][1].endswith("/dreamina/accounts") and calls[0][2]["Authorization"] == "Bearer user:1-abc"
    uploads = [c for c in calls if c[0] == "POST" and "/assets/" in c[1]]
    assert uploads[0][1].endswith("/assets/US%3Ame%40x.com") and uploads[0][2]["Content-Type"] == "video/mp4"
    assert payload == {
        "account": "US:me@x.com", "model": "seedance-2.5", "prompt": "P", "ratio": "9:16", "duration": 8, "resolution": "720p",
        "omni_1_videoRef": "asset-video", "omni_1_imageRef": "asset-body", "omni_2_imageRef": "asset-face",
    }
    # A second take reuses the uploaded assets and the resolved account.
    calls.clear()
    dreamina.submit(str(video), [str(body), str(face)], "P", dict(meta, model="seedance-2.0-mini"), "hd", "")
    assert [c[1].rsplit("/", 1)[-1] for c in calls] == ["videos"]
    assert calls[0][4]["resolution"] == "720p" and calls[0][4]["model"] == "seedance-2.0-mini"


def test_dreamina_waits_out_busy_accounts_then_reports_errors(dreamina, monkeypatch, tmp_path):
    video = tmp_path / "g.mp4"
    video.write_bytes(b"v")
    dreamina._account = "US:me@x.com"
    dreamina._assets[("US:me@x.com", __import__("hashlib").sha1(b"v").hexdigest())] = "a"
    replies = iter([FakeResponse({}, 429), FakeResponse({}, 429), FakeResponse({"jobid": "j"})])
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: next(replies))
    assert dreamina.submit(str(video), [], "P", {"duration": 5, "width": 16, "height": 9}, "hd", "")[0] == "j"
    for code, text in ((402, "кредиты"), (596, "сессия"), (400, "отклонил")):
        monkeypatch.setattr(providers.requests, "post", lambda *a, code=code, **k: FakeResponse({"error": "x"}, code))
        with pytest.raises(providers.ProviderError, match=text):
            dreamina.submit(str(video), [], "P", {"duration": 5, "width": 16, "height": 9}, "hd", "")


def test_dreamina_account_selection(dreamina, monkeypatch):
    accounts = {"US:a@x.com": {"account": "US:a@x.com", "email": "a@x.com", "error": "session expired"},
                "CA:b@x.com": {"account": "CA:b@x.com", "email": "b@x.com"}}
    monkeypatch.setattr(providers.requests, "get", lambda *a, **k: FakeResponse(accounts))
    assert dreamina.account() == "CA:b@x.com"  # skips the broken one
    monkeypatch.setattr(providers.Dreamina, "_account", None)
    monkeypatch.setenv("DREAMINA_ACCOUNT", "a@x.com")
    with pytest.raises(providers.ProviderError, match="ошибка"):
        dreamina.account()
    monkeypatch.setenv("DREAMINA_ACCOUNT", "nobody@x.com")
    with pytest.raises(providers.ProviderError, match="не найден"):
        dreamina.account()


def test_dreamina_poll(dreamina, monkeypatch):
    replies = iter([{"status": "started"}, {"status": "completed", "response": {"videoUrlWatermarked": "w", "videoUrl": "clean"}},
                    {"status": "failed", "error": "moderation", "errorDetails": "face"}])
    seen = []
    monkeypatch.setattr(providers.requests, "get", lambda url, **k: seen.append(url) or FakeResponse(next(replies)))
    assert dreamina.poll("user:1-dreamina:US:me@x.com-video:123")["state"] == "pending"
    assert dreamina.poll("j")["url"] == "clean"  # the clean master, not the watermarked copy
    assert dreamina.poll("j")["error"] == "moderation — face"
    assert seen[0].endswith("/videos/user:1-dreamina:US:me@x.com-video:123")
