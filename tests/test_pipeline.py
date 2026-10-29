"""End to end through the HTTP API with the free mock provider."""
import time

import pytest
import requests
from fastapi.testclient import TestClient

from genjutsu import faces, hosting, jobs, masking, media, providers
from conftest import box_masks


@pytest.fixture
def client(runs, monkeypatch):
    monkeypatch.setenv("PROVIDER", "mock")
    monkeypatch.setenv("REPLICATE_API_TOKEN", "")
    # Stand-in for MediaPipe (its model is downloaded on first real use).
    monkeypatch.setattr(masking, "segment_local", lambda video, folder, w, h, status: box_masks(folder, round(media.probe(video)["duration"] * 24), w, h))
    from genjutsu.server import app
    with TestClient(app) as c:
        yield c


def submit(client, video, photo, **form):
    files = [("video", ("clip.mp4", open(video, "rb"), "video/mp4")), ("face", ("f.jpg", open(photo("f.jpg"), "rb"), "image/jpeg")),
             ("body", ("b.jpg", open(photo("b.jpg"), "rb"), "image/jpeg"))]
    data = {"consent": "yes", "masking": "local", "provider": "mock", **form}
    return client.post("/api/jobs", files=files, data=data)


def wait(client, job_id, until=("done", "failed", "review"), timeout=120):
    started = time.time()
    while time.time() - started < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in until and not (job.get("hd") or {}).get("status") == "running":
            return job
        time.sleep(0.3)
    raise AssertionError(f"stuck: {job['status']} / {job['stage']}")


def test_full_run_keeps_original_audio(client, make_video, photo):
    video = make_video(seconds=5)
    r = submit(client, video, photo)
    assert r.status_code == 200, r.text
    job = wait(client, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert job["used"] == {"masking": "local", "audio": "full", "face_mesh": False}
    assert job["lipsync"] and "@Image1 shows the full body" in job["prompt"] and "@Image2 is a face close-up" in job["prompt"]
    result = jobs.folder(job["id"]) / job["result"]
    info = media.probe(result)
    assert info["has_audio"] and abs(info["duration"] - 5) < 0.15
    assert client.get(f"/files/{job['id']}/{job['result']}").status_code == 200
    assert client.get(f"/files/{job['id']}/job.json").status_code == 404

    # Draft → 1080p through the provider's draft completion.
    assert client.post(f"/api/jobs/{job['id']}/hd").status_code == 200
    job = wait(client, job["id"])
    assert job["hd"]["status"] == "done" and (jobs.folder(job["id"]) / job["hd"]["result"]).exists()


def test_long_clip_is_split_and_joined(client, make_video, photo, monkeypatch):
    monkeypatch.setattr(providers.Mock, "max_seconds", 4)
    job = wait(client, submit(client, make_video(seconds=9), photo).json()["id"])
    assert job["status"] == "done", job["error"]
    # 9 s in ≤4 s parts sharing 1 s: three parts, blended back to exactly 9 s.
    assert len(job["takes"][0]["segments"]) == 3
    assert abs(media.probe(jobs.folder(job["id"]) / job["result"])["duration"] - 9) < 0.2


def test_short_clip_is_padded_for_the_provider_and_cut_back(client, make_video, photo):
    job = wait(client, submit(client, make_video(seconds=2), photo).json()["id"])
    assert job["status"] == "done", job["error"]
    assert abs(media.probe(jobs.folder(job["id"]) / job["result"])["duration"] - 2) < 0.15


def test_review_gate_then_approve(client, make_video, photo):
    job = wait(client, submit(client, make_video(seconds=4), photo, review="always").json()["id"])
    assert job["status"] == "review" and job["qa"]["ok"]
    assert not job["takes"]
    client.post(f"/api/jobs/{job['id']}/approve")
    assert wait(client, job["id"])["status"] == "done"


def test_validation(client, make_video, photo):
    video = make_video(seconds=4)
    assert "согласие" in submit(client, video, photo, consent="").json()["detail"]
    r = client.post("/api/jobs", files=[("video", ("c.mp4", open(video, "rb"), "video/mp4"))], data={"consent": "yes", "provider": "mock"})
    assert r.status_code == 400 and "фото" in r.json()["detail"]
    tiny = photo("tiny.jpg", size=(100, 100))
    r = client.post("/api/jobs", files=[("video", ("c.mp4", open(video, "rb"), "video/mp4")), ("face", ("t.jpg", open(tiny, "rb"), "image/jpeg"))], data={"consent": "yes", "provider": "mock"})
    assert r.status_code == 400 and "маленькое" in r.json()["detail"]


def test_missing_key_is_reported_before_upload(client, make_video, photo, monkeypatch):
    monkeypatch.setenv("ENHANCOR_API_KEY", "")
    monkeypatch.setattr("genjutsu.config._read_env_file", lambda: {})
    r = submit(client, make_video(seconds=4), photo, provider="enhancor")
    assert r.status_code == 400 and "ENHANCOR_API_KEY" in r.json()["detail"]


def test_resume_never_resubmits(runs, monkeypatch, make_video, photo, client):
    job = wait(client, submit(client, make_video(seconds=4), photo).json()["id"])
    jobs.update(job["id"], status="generating", takes=[{"index": 0, "state": "pending", "segments": [{"index": 0, "state": None}]}], result=None)
    jobs.resume_all()
    job = jobs.load(job["id"])
    assert job["status"] == "failed" and "дважды" in job["error"]


def test_best_take_wins_and_can_be_changed(client, make_video, photo, monkeypatch):
    scores = iter([41, 77, 63])
    monkeypatch.setattr(faces, "score_video", lambda video, refs: {"score": next(scores), "coverage": 1.0})
    job = wait(client, submit(client, make_video(seconds=4), photo, takes="3").json()["id"])
    assert job["status"] == "done", job["error"]
    assert [t["score"]["score"] for t in job["takes"]] == [41, 77, 63]
    assert job["chosen"] == 1 and job["result"] == "result-t1.mp4"
    assert client.get(f"/files/{job['id']}/result-t2.mp4").status_code == 200
    job = client.post(f"/api/jobs/{job['id']}/choose/2").json()
    assert job["chosen"] == 2 and job["result"] == "result-t2.mp4"
    # 1080p finishes the chosen take.
    client.post(f"/api/jobs/{job['id']}/hd")
    job = wait(client, job["id"])
    assert job["hd"]["status"] == "done" and job["hd"]["take"] == 2


def test_one_failed_take_does_not_sink_the_job(client, make_video, photo, monkeypatch):
    real_poll = providers.Mock.poll
    calls = {}

    def poll(self, request_id):
        first = calls.setdefault("first", request_id)
        return {"state": "failed", "error": "boom"} if request_id == first else real_poll(self, request_id)

    monkeypatch.setattr(providers.Mock, "poll", poll)
    job = wait(client, submit(client, make_video(seconds=4), photo, takes="2").json()["id"])
    assert job["status"] == "done", job["error"]
    assert [t["state"] for t in job["takes"]] == ["failed", "done"] and job["chosen"] == 1


def test_check_photo_reports_missing_face(client, photo):
    r = client.post("/api/check-photo", files={"photo": ("p.jpg", open(photo("plain.jpg"), "rb"), "image/jpeg")}, data={"role": "face"})
    assert r.status_code == 200
    body = r.json()
    assert body["available"] and body["faces"] == 0 and "Лицо не найдено" in body["warnings"][0]


def test_dreamina_end_to_end_without_tunnel(client, make_video, photo, monkeypatch, tmp_path):
    monkeypatch.setenv("USEAPI_TOKEN", "user:1-abc")
    monkeypatch.setattr(providers.Dreamina, "_account", None)
    monkeypatch.setattr(providers.Dreamina, "_assets", {})
    monkeypatch.setattr(hosting, "host", lambda path: pytest.fail("Dreamina must not need public links"))
    monkeypatch.setattr(hosting, "webhook_url", lambda: pytest.fail("Dreamina needs no webhook"))
    finished = make_video(seconds=5, audio=False, name="dreamina-out.mp4")
    sent = []

    class R:
        def __init__(self, data, code=200):
            self.data, self.status_code, self.ok, self.text = data, code, code < 400, str(data)

        def json(self):
            return self.data

        def raise_for_status(self):
            pass

    def get(url, headers=None, timeout=None, **kw):
        if url.endswith("/accounts"):
            return R({"US:me@x.com": {"account": "US:me@x.com", "email": "me@x.com"}})
        return R({"status": "completed", "response": {"videoUrl": finished.as_uri()}})

    def post(url, headers=None, timeout=None, data=None, json=None):
        if "/assets/" in url:
            return R({"assetRef": f"ref{len(sent)}"})
        sent.append(json)
        return R({"jobid": f"job{len(sent)}"})

    monkeypatch.setattr(providers.requests, "get", get)
    monkeypatch.setattr(providers.requests, "post", post)
    job = wait(client, submit(client, make_video(seconds=5), photo, provider="dreamina", model="seedance-2.0-fast").json()["id"])
    assert job["status"] == "done", job["error"]
    assert sent[0]["model"] == "seedance-2.0-fast" and sent[0]["omni_1_videoRef"] and sent[0]["omni_2_imageRef"]
    assert "@Video1" in sent[0]["prompt"] and "@Image2" in sent[0]["prompt"]
    info = media.probe(jobs.folder(job["id"]) / job["result"])
    assert info["has_audio"] and abs(info["duration"] - 5) < 0.15
    # Unknown model names are refused before anything is uploaded.
    r = submit(client, make_video(seconds=4, name="v2.mp4"), photo, provider="dreamina", model="seedance-9")
    assert r.status_code == 400 and "seedance-9" in r.json()["detail"]


def test_public_server_serves_only_registered_files(tmp_path, monkeypatch):
    monkeypatch.setenv("PUBLIC_PORT", "18771")
    hosting.start_public_server()
    f = tmp_path / "v.mp4"
    f.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"0" * 100)
    token = hosting._register(f)
    assert requests.get(f"http://127.0.0.1:18771/m/{token}/v.mp4", timeout=5).content == f.read_bytes()
    assert requests.get(f"http://127.0.0.1:18771/m/{'x' * 32}/v.mp4", timeout=5).status_code == 404
    assert requests.get("http://127.0.0.1:18771/", timeout=5).status_code == 404
    assert requests.get("http://127.0.0.1:18771/api/config", timeout=5).status_code == 404
