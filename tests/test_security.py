import io
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from genjutsu import photos


@pytest.fixture
def raw_client(runs, monkeypatch):
    monkeypatch.setenv("PROVIDER", "mock")
    from genjutsu.server import app
    with TestClient(app) as c:
        yield c


def test_changes_need_the_app_header(raw_client):
    # What another website's form or no-cors fetch would send: no custom header.
    r = raw_client.post("/api/settings", json={"PROVIDER": "mock"})
    assert r.status_code == 403
    r = raw_client.post("/api/photo-jobs", files={"scene": ("s.jpg", b"x", "image/jpeg")}, data={"consent": "yes"})
    assert r.status_code == 403
    assert raw_client.post("/api/chatgpt/logout").status_code == 403
    assert raw_client.get("/api/config").status_code == 200  # reads stay open to the page itself


def test_foreign_host_is_refused(raw_client):
    # DNS rebinding: evil.example resolving to 127.0.0.1.
    r = raw_client.get("/api/jobs", headers={"Host": "evil.example:8770"})
    assert r.status_code == 403
    assert raw_client.get("/api/jobs", headers={"Host": "127.0.0.1:8770"}).status_code == 200
    assert raw_client.get("/api/jobs", headers={"Host": "localhost:8770"}).status_code == 200
    assert raw_client.get("/api/jobs", headers={"Host": "[::1]:8770"}).status_code == 200


def test_heic_photos_open(tmp_path):
    pytest.importorskip("pillow_heif")
    src = tmp_path / "IMG_0001.HEIC"
    try:
        Image.new("RGB", (640, 480), (200, 100, 50)).save(src, format="HEIF")
    except Exception:
        pytest.skip("no HEIF encoder in this build")
    info = photos.prepare(src, tmp_path / "out.jpg")
    assert info == {"width": 640, "height": 480}


def test_delete_and_download_all(runs, monkeypatch, photo):
    monkeypatch.setenv("PROVIDER", "mock")
    from genjutsu import jobs
    from genjutsu.server import app
    job_id = "20261029-120000-abcdef"
    d = jobs.folder(job_id)
    d.mkdir(parents=True)
    for n in (1, 2):
        Image.new("RGB", (64, 64), (n * 60, 0, 0)).save(d / f"out-{n}.png")
    jobs.save({"id": job_id, "kind": "photo", "created": "2026-10-29T12:00:00+00:00", "status": "done", "log": [], "notices": [],
               "options": {"template": "hero", "mode": "swap"}, "images": [{"file": "out-1.png"}, {"file": "out-2.png"}], "result": "out-1.png"})
    with TestClient(app, headers={"X-Genjutsu": "1"}) as c:
        r = c.get(f"/api/jobs/{job_id}/download-all")
        assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
        assert sorted(zipfile.ZipFile(io.BytesIO(r.content)).namelist()) == [f"genjutsu-{job_id}-1.png", f"genjutsu-{job_id}-2.png"]
        assert c.delete(f"/api/jobs/{job_id}").status_code == 200
        assert not d.exists() and c.get(f"/api/jobs/{job_id}").status_code == 404
