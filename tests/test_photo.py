"""Photo mode end to end with fake ChatGPT and Dreamina."""
import json
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from genjutsu import chatgpt, faces, photo, photo_prompt, providers, templates


@pytest.fixture
def client(runs, monkeypatch, tmp_path):
    monkeypatch.setenv("USEAPI_TOKEN", "user:1-abc")
    monkeypatch.setattr(providers.Dreamina, "_account", "US:me@x.com")
    monkeypatch.setattr(providers.Dreamina, "upload", lambda self, path, account: f"ref:{path.name if hasattr(path, 'name') else path}")
    monkeypatch.setattr(photo, "POLL_EVERY", 0.05)
    finished = tmp_path / "gen.png"
    Image.new("RGB", (512, 768), (90, 120, 200)).save(finished)
    state = {"submitted": [], "polls": 0}

    def submit_image(self, prompt, refs, model, width, height, hd=False, strength=None):
        state["submitted"].append({"prompt": prompt, "refs": [r.name for r in refs], "model": model, "ratio": (width, height), "hd": hd})
        return f"img-{len(state['submitted'])}", {"model": model}

    def poll_image(self, jobid):
        state["polls"] += 1
        return {"state": "done", "urls": [finished.as_uri()]}

    monkeypatch.setattr(providers.Dreamina, "submit_image", submit_image)
    monkeypatch.setattr(providers.Dreamina, "poll_image", poll_image)
    from genjutsu.server import app
    with TestClient(app, headers={"X-Genjutsu": "1"}) as c:
        c.state = state
        yield c


def wait(client, job_id, timeout=60):
    started = time.time()
    while time.time() - started < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.2)
    raise AssertionError(job["stage"])


def post(client, photo_fixture, **form):
    files = [("scene", ("s.jpg", open(photo_fixture("scene.jpg", (900, 1200)), "rb"), "image/jpeg")),
             ("face", ("f.jpg", open(photo_fixture("f.jpg"), "rb"), "image/jpeg")),
             ("body", ("b.jpg", open(photo_fixture("b.jpg"), "rb"), "image/jpeg"))]
    return client.post("/api/photo-jobs", files=files, data={"consent": "yes", "count": "2", **form})


def test_gpt_writes_the_prompt_and_dreamina_draws(client, photo, monkeypatch):
    monkeypatch.setattr(chatgpt, "status", lambda: {"connected": True})
    asked = {}

    def respond(instructions, text, images=()):
        asked.update(instructions=instructions, text=text, images=[p.name for p in images])
        return 'Here: {"prompt": "Replace the person in image 1 with the person from image 2 and image 3. Keep exactly this face.", "summary_ru": "Вы в кресле"}'

    monkeypatch.setattr(chatgpt, "respond", respond)
    scores = iter([30, 80])
    monkeypatch.setattr(faces, "Identity", lambda: None)
    monkeypatch.setattr(faces, "score_still", lambda img, refs, identity=None: {"score": next(scores), "face": True})
    r = post(client, photo, image_model="nano-banana-pro")
    assert r.status_code == 200, r.text
    job = wait(client, r.json()["id"])
    assert job["status"] == "done", job["error"]
    assert job["prompt_by"] == "gpt" and job["summary"] == "Вы в кресле"
    assert asked["instructions"] == photo_prompt.GUIDE and "image 1: the scene" in asked["text"]
    assert asked["images"][0] in ("scene.jpg", "scene-guide.jpg")
    sent = client.state["submitted"]
    assert len(sent) == 2 and sent[0]["model"] == "nano-banana-pro" and sent[0]["prompt"].startswith("Replace the person in image 1")
    assert sent[0]["refs"][0] == job["guide"] and sent[0]["ratio"] == (900, 1200)
    # Best likeness first.
    assert [im["score"]["score"] for im in job["images"]] == [80, 30] and job["result"] == job["images"][0]["file"]
    assert client.get(f"/files/{job['id']}/{job['result']}").status_code == 200

    # One more variant with the same prompt.
    scores2 = iter([55])
    monkeypatch.setattr(faces, "score_still", lambda img, refs, identity=None: {"score": next(scores2), "face": True})
    client.post(f"/api/jobs/{job['id']}/more")
    job = wait(client, job["id"])
    assert len(job["images"]) == 3 and len(client.state["submitted"]) == 3
    assert [im["score"]["score"] for im in job["images"]] == [80, 55, 30]


def test_refine_and_edit_the_prompt(client, photo, monkeypatch):
    monkeypatch.setattr(chatgpt, "status", lambda: {"connected": True})
    monkeypatch.setattr(faces, "available", lambda: False)
    asked = []

    def respond(instructions, text, images=()):
        asked.append((instructions, text, [p.name for p in images]))
        if instructions == photo_prompt.REVIEW_GUIDE:
            return '{"prompt": "Replace the person in image 1 with the person from image 2. The hair is short and dark brown.", "summary_ru": "делаю волосы короче"}'
        return '{"prompt": "Replace the person in image 1 with the person from image 2 and image 3 please.", "summary_ru": "ок"}'

    monkeypatch.setattr(chatgpt, "respond", respond)
    job = wait(client, post(client, photo, count="1").json()["id"])
    first = job["prompt"]
    r = client.post(f"/api/jobs/{job['id']}/refine?image={job['images'][0]['file']}")
    assert r.status_code == 200, r.text
    job = wait(client, job["id"])
    review = asked[-1]
    assert review[0] == photo_prompt.REVIEW_GUIDE and review[2][-1] == "out-1.png" and first in review[1]
    assert job["prompt_by"] == "gpt-review" and "short and dark brown" in job["prompt"] and job["summary"] == "делаю волосы короче"
    assert job["prompts"][0]["prompt"] == first and len(job["images"]) == 2

    r = client.post(f"/api/jobs/{job['id']}/regenerate", json={"prompt": "  My own prompt:   replace the person in image 1 with image 2.  ", "count": 1})
    assert r.status_code == 200, r.text
    job = wait(client, job["id"])
    assert job["prompt_by"] == "user" and job["prompt"] == "My own prompt: replace the person in image 1 with image 2."
    assert client.state["submitted"][-1]["prompt"] == job["prompt"] and len(job["images"]) == 3
    assert client.post(f"/api/jobs/{job['id']}/regenerate", json={"prompt": "short"}).status_code == 400


def test_failed_extra_round_keeps_the_job_done(client, photo, monkeypatch):
    monkeypatch.setattr(chatgpt, "status", lambda: {"connected": False})
    monkeypatch.setattr(faces, "available", lambda: False)
    job = wait(client, post(client, photo, count="1").json()["id"])
    assert client.get(f"/files/{job['id']}/thumb.jpg").status_code == 200
    from genjutsu import jobs as jobs_module
    thumb = jobs_module.folder(job["id"]) / "thumb.jpg"
    thumb.unlink()
    assert client.get(f"/files/{job['id']}/thumb.jpg").status_code == 200 and thumb.exists()  # rebuilt for older jobs

    def broke(*a, **k):
        raise providers.ProviderError("Dreamina: закончились кредиты")

    monkeypatch.setattr(providers.Dreamina, "submit_image", broke)
    client.post(f"/api/jobs/{job['id']}/more")
    job = wait(client, job["id"])
    assert job["status"] == "done" and len(job["images"]) == 1
    assert any("закончились кредиты" in n for n in job["notices"])


def test_refine_needs_chatgpt(client, photo, monkeypatch):
    monkeypatch.setattr(chatgpt, "status", lambda: {"connected": False})
    monkeypatch.setattr(faces, "available", lambda: False)
    job = wait(client, post(client, photo, count="1").json()["id"])
    r = client.post(f"/api/jobs/{job['id']}/refine")
    assert r.status_code == 400 and "ChatGPT" in r.json()["detail"]


def test_without_chatgpt_the_builtin_prompt_is_used(client, photo, monkeypatch):
    monkeypatch.setattr(chatgpt, "status", lambda: {"connected": False})
    monkeypatch.setattr(faces, "available", lambda: False)
    job = wait(client, post(client, photo, template="anime").json()["id"])
    assert job["status"] == "done", job["error"]
    assert job["prompt_by"] == "builtin" and "1990s Japanese cel anime" in job["prompt"]
    assert any("ChatGPT не подключён" in n for n in job["notices"])


def test_gpt_failure_falls_back(client, photo, monkeypatch):
    monkeypatch.setattr(chatgpt, "status", lambda: {"connected": True})
    monkeypatch.setattr(faces, "available", lambda: False)

    def boom(*a, **k):
        raise chatgpt.ChatGPTError("Лимит подписки ChatGPT исчерпан.")

    monkeypatch.setattr(chatgpt, "respond", boom)
    job = wait(client, post(client, photo).json()["id"])
    assert job["status"] == "done" and job["prompt_by"] == "builtin"
    assert any("Лимит подписки" in n for n in job["notices"])


def test_validation(client, photo, monkeypatch):
    r = post(client, photo, consent="")
    assert r.status_code == 400
    r = post(client, photo, image_model="midjourney")
    assert r.status_code == 400 and "midjourney" in r.json()["detail"]
    r = client.post("/api/photo-jobs", files=[("scene", ("s.jpg", open(photo("s.jpg"), "rb"), "image/jpeg"))], data={"consent": "yes"})
    assert r.status_code == 400 and "фото" in r.json()["detail"]
    monkeypatch.setenv("USEAPI_TOKEN", "")
    monkeypatch.setattr("genjutsu.config._read_env_file", lambda: {})
    r = post(client, photo)
    assert r.status_code == 400 and "USEAPI_TOKEN" in r.json()["detail"]


def test_prompt_parsing_and_request_text():
    assert photo_prompt.parse('```json\n{"prompt": "  A  long enough prompt about image 1 and image 2 here.", "summary_ru": "ок"}\n```') == {
        "prompt": "A long enough prompt about image 1 and image 2 here.", "summary": "ок"}
    assert photo_prompt.parse("Replace the person in image 1 with the person from image 2, keep the face.")["prompt"].startswith("Replace")
    with pytest.raises(ValueError):
        photo_prompt.parse("no")
    t = templates.BY_ID["object"]
    text = photo_prompt.request_text(t, {"who": "the mug", "extra": "сделай чашку жёлтой"}, [{"role": "object"}], False, "GPT Image 2.5 Flare")
    assert "Replace the mug" in text and "image 2: the new object" in text and "сделай чашку жёлтой" in text


def test_builtin_prompt_names_images_in_order():
    refs = [{"role": "body"}, {"role": "face"}]
    out = photo_prompt.builtin(templates.BY_ID["hero"], {"who": "person"}, refs, True)["prompt"]
    assert "from image 3 and image 2" in out and "silhouette" in out
    obj = photo_prompt.builtin(templates.BY_ID["object"], {"who": "the phone"}, [{"role": "object"}], False)["prompt"]
    assert "Replace the phone in the scene with the object from image 2" in obj


def test_dreamina_image_request_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("USEAPI_TOKEN", "user:1-abc")
    monkeypatch.setattr(providers.Dreamina, "_account", "US:me@x.com")
    monkeypatch.setattr(providers.Dreamina, "_assets", {})
    monkeypatch.setattr(providers.Dreamina, "busy_wait", 0)
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    calls = []

    class R:
        def __init__(self, data, code=200):
            self.data, self.status_code, self.ok, self.text = data, code, code < 400, json.dumps(data)

        def json(self):
            return self.data

        def raise_for_status(self):
            pass

    def post(url, headers=None, data=None, json=None, timeout=None):
        calls.append((url, headers, json))
        return R({"assetRef": "asset-" + data.decode()} if "/assets/" in url else {"jobid": "j1"})

    monkeypatch.setattr(providers.requests, "post", post)
    monkeypatch.setattr(providers.requests, "get", lambda url, **k: calls.append((url,)) or R({"status": "completed", "response": {"images": [{"imageUrl": "https://x/1.png"}, {"imageUrl": "https://x/2.png"}]}}))
    d = providers.Dreamina()
    ident, payload = d.submit_image("P", [a, b], "seedream-5.0-lite", 1080, 1920, hd=True)
    assert ident == "j1" and payload["ratio"] == "9:16" and payload["resolution"] == "4k"
    assert payload["imageRef_1"] == "asset-a" and payload["imageRef_2"] == "asset-b"
    assert calls[-1][0].endswith("/dreamina/images") and calls[-1][1]["User-Agent"].startswith("LunaGenjutsu")
    assert "resolution" not in d.submit_image("P", [a], "gpt-image-2.5-flare", 1000, 1000)[1]
    res = d.poll_image("user:1-dreamina:US:me@x.com-image:7")
    assert res["urls"] == ["https://x/1.png", "https://x/2.png"]
    assert calls[-1][0].endswith("/images/user%3A1-dreamina%3AUS%3Ame%40x.com-image%3A7")
