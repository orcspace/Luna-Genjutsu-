"""End to end through the HTTP API with the free mock provider."""
import time

import pytest
import requests
from fastapi.testclient import TestClient

from genjutsu import hosting, jobs, masking, media, providers
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
    assert job["used"] == {"masking": "local", "audio": "full"}
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
    assert len(job["segments"]) == 3
    assert abs(media.probe(jobs.folder(job["id"]) / job["result"])["duration"] - 9) < 0.2


def test_short_clip_is_padded_for_the_provider_and_cut_back(client, make_video, photo):
    job = wait(client, submit(client, make_video(seconds=2), photo).json()["id"])
    assert job["status"] == "done", job["error"]
    assert abs(media.probe(jobs.folder(job["id"]) / job["result"])["duration"] - 2) < 0.15


def test_review_gate_then_approve(client, make_video, photo):
    job = wait(client, submit(client, make_video(seconds=4), photo, review="always").json()["id"])
    assert job["status"] == "review" and job["qa"]["ok"]
    assert not job["segments"]
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
    jobs.update(job["id"], status="generating", segments=[{"index": 0, "state": None}], result=None)
    jobs.resume_all()
    job = jobs.load(job["id"])
    assert job["status"] == "failed" and "дважды" in job["error"]


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
