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


def test_segments_cover_the_clip():
    from genjutsu.jobs import _segments
    segs = _segments(29.5, 15)
    assert len(segs) == 2 and abs(sum(s["duration"] for s in segs) - 29.5) < 0.01
    assert all(s["duration"] >= 4 for s in _segments(16, 15))
    assert len(_segments(15, 15)) == 1
