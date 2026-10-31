import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import video
from app.main import app, image_rate_limit, job_rate_limit

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_IMAGE = ROOT / "samples" / "street.jpg"
SAMPLE_VIDEO = ROOT / "samples" / "crosswalk.mp4"
SESSION = {"X-Session-Id": "test-session-00000001"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def reset_rate_limits():
    image_rate_limit.calls.clear()
    job_rate_limit.calls.clear()


def test_health_and_limits(client):
    assert client.get("/api/health").json()["status"] == "ok"
    limits = client.get("/api/limits").json()
    assert limits["browser_model"] == "/models/yolo26n-320.onnx"


def test_detect_image(client):
    with SAMPLE_IMAGE.open("rb") as f:
        r = client.post("/api/detect?conf=0.4", files={"file": ("street.jpg", f, "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["counts"]["person"] >= 3
    assert body["image"].startswith("data:image/jpeg;base64,")
    d = body["detections"][0]
    assert set(d) == {"class", "class_id", "confidence", "box"} and len(d["box"]) == 4


def test_detect_rejects_bad_input(client):
    r = client.post("/api/detect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    r = client.post("/api/detect", files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")})
    assert r.status_code == 400
    r = client.post("/api/detect?conf=2", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 422


def test_image_rate_limit(client, monkeypatch):
    monkeypatch.setattr(image_rate_limit, "per_hour", 2)
    for expected in (400, 400, 429):
        r = client.post("/api/detect", files={"file": ("a.jpg", b"x", "image/jpeg")})
        assert r.status_code == expected


def test_jobs_need_a_valid_session(client):
    assert client.get("/api/jobs").status_code == 422
    assert client.get("/api/jobs", headers={"X-Session-Id": "short"}).status_code == 400


def test_model_files_are_whitelisted(client):
    r = client.get("/models/yolo26n-320.onnx")
    assert r.status_code == 200 and "max-age" in r.headers["cache-control"]
    assert client.get("/models/yolo26n-640.onnx").status_code == 404
    assert client.get("/models/..%2Fapp%2Fconfig.py").status_code == 404


@pytest.mark.skipif(not video.ffmpeg_available(), reason="ffmpeg not installed")
def test_video_job_end_to_end(client):
    with SAMPLE_VIDEO.open("rb") as f:
        r = client.post("/api/jobs", headers=SESSION, files={"file": ("crosswalk.mp4", f, "video/mp4")})
    assert r.status_code == 202
    job_id = r.json()["id"]

    for _ in range(240):
        job = client.get(f"/api/jobs/{job_id}", headers=SESSION).json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert job["status"] == "done", job
    assert job["frames"] == 60 and job["counts"]["person"] >= 5

    r = client.get(f"/api/jobs/{job_id}/video", headers=SESSION)
    assert r.status_code == 200 and r.headers["content-type"] == "video/mp4"
    assert r.content[4:8] == b"ftyp"
    # Other sessions can't see it
    other = {"X-Session-Id": "someone-else-0000001"}
    assert client.get(f"/api/jobs/{job_id}", headers=other).status_code == 404

    assert client.delete(f"/api/jobs/{job_id}", headers=SESSION).status_code == 200
    assert client.get(f"/api/jobs/{job_id}", headers=SESSION).status_code == 404


@pytest.mark.skipif(not video.ffmpeg_available(), reason="ffmpeg not installed")
def test_video_job_rejects_non_video(client):
    r = client.post("/api/jobs", headers=SESSION, files={"file": ("fake.mp4", b"\x00" * 1000, "video/mp4")})
    assert r.status_code == 202
    job_id = r.json()["id"]
    for _ in range(40):
        job = client.get(f"/api/jobs/{job_id}", headers=SESSION).json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.25)
    assert job["status"] == "error" and "video" in job["error"].lower()
    client.delete(f"/api/jobs/{job_id}", headers=SESSION)
