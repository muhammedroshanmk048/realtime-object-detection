import asyncio
import base64
import logging
import re
import shutil
import tempfile
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import jobs, video
from app.config import get_settings
from app.detector import draw, load

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vision")
settings = get_settings()
STATIC = Path(__file__).resolve().parent.parent / "static"
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_TYPES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}


def image_detector():
    return load(settings.image_model, settings.classes_file, settings.onnx_threads)


def video_detector():
    return load(settings.video_model, settings.classes_file, settings.onnx_threads)


async def cleanup_loop():
    """Drop expired results periodically: nothing is kept longer than job_ttl_minutes."""
    while True:
        try:
            removed = jobs.purge_expired()
            if removed:
                log.info("Cleanup removed %d expired jobs", removed)
        except Exception:
            log.exception("Cleanup failed")
        await asyncio.sleep(settings.cleanup_interval_minutes * 60)


def warm_up():
    # Loading a session and running it once takes seconds on a 0.1 CPU host;
    # do it in the background at startup so the first visitor doesn't pay for it.
    try:
        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        image_detector().detect(blank)
        video_detector().detect(blank)
        log.info("Models ready")
    except Exception:
        log.exception("Could not load models")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not video.ffmpeg_available():
        log.warning("ffmpeg/ffprobe not found on PATH: video jobs will fail until it is installed")
    asyncio.get_running_loop().run_in_executor(None, warm_up)
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="VisionTrack · Real-Time Object Detection", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
# CC0 sample image and clip for the "Try a sample" buttons
app.mount("/samples", StaticFiles(directory=STATIC.parent / "samples"), name="samples")


# ---------- Sessions + rate limiting ----------

SESSION_RE = re.compile(r"^[A-Za-z0-9-]{16,64}$")


def session_id(x_session_id: Annotated[str, Header()]) -> str:
    """Each browser sends a random ID; jobs are only visible to the session that created them."""
    if not SESSION_RE.match(x_session_id):
        raise HTTPException(400, "Invalid session id")
    return x_session_id


Session = Annotated[str, Depends(session_id)]


def client_ip(request: Request) -> str:
    # Hosting platforms sit behind a proxy, so the real client is the first X-Forwarded-For hop
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class RateLimit:
    """Sliding one-hour window per client IP, to keep the shared free CPU usable for everyone."""

    def __init__(self, per_hour: int, what: str):
        self.per_hour = per_hour
        self.what = what
        self.calls: dict[str, deque] = defaultdict(deque)

    def __call__(self, request: Request) -> None:
        now = time.monotonic()
        calls = self.calls[client_ip(request)]
        while calls and now - calls[0] > 3600:
            calls.popleft()
        if len(calls) >= self.per_hour:
            raise HTTPException(429, f"Limit of {self.per_hour} {self.what} per hour reached. Try again later.")
        calls.append(now)
        # Drop idle clients so the table doesn't grow forever
        for ip in [ip for ip, c in self.calls.items() if not c or now - c[-1] > 3600]:
            del self.calls[ip]


image_rate_limit = RateLimit(settings.images_per_hour_per_ip, "images")
job_rate_limit = RateLimit(settings.jobs_per_hour_per_ip, "videos")


def check_type(filename: str | None, allowed: set[str], kind: str) -> str:
    name = Path(filename or kind).name
    if Path(name).suffix.lower() not in allowed:
        raise HTTPException(400, f"Unsupported {kind} type. Use one of: {' '.join(sorted(allowed))}")
    return name


# ---------- Routes ----------

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "ffmpeg": video.ffmpeg_available()}


@app.get("/api/limits")
def limits():
    return {
        "max_image_mb": settings.max_image_mb,
        "max_video_mb": settings.max_video_mb,
        "max_video_seconds": settings.max_video_seconds,
        "video_fps": settings.video_fps,
        "job_ttl_minutes": settings.job_ttl_minutes,
        "max_jobs_per_session": settings.max_jobs_per_session,
        "image_types": sorted(IMAGE_TYPES),
        "video_types": sorted(VIDEO_TYPES),
        "default_conf": settings.default_conf,
        "browser_model": f"/models/{settings.browser_model}",
        "classes": "/models/classes.json",
    }


@app.post("/api/detect", dependencies=[Depends(image_rate_limit)])
def detect_image(file: UploadFile, conf: Annotated[float, Query(ge=0.05, le=0.95)] = settings.default_conf):
    """Detect objects in one image. Returns the detections plus the annotated image as a JPEG data URL."""
    check_type(file.filename, IMAGE_TYPES, "image")
    limit = settings.max_image_mb * 1024 * 1024
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"Image larger than {settings.max_image_mb} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise HTTPException(400, f"Could not read this image (corrupt, or over {settings.max_image_megapixels} MP)")
    h, w = img.shape[:2]
    if max(h, w) > settings.max_image_side:
        r = settings.max_image_side / max(h, w)
        img = cv2.resize(img, (round(w * r), round(h * r)), interpolation=cv2.INTER_AREA)

    detector = image_detector()
    start = time.perf_counter()
    dets = detector.detect(img, conf=conf, iou_thres=settings.iou_threshold)
    ms = (time.perf_counter() - start) * 1000

    draw(img, [d.box for d in dets], [f"{d.name} {d.confidence:.2f}" for d in dets], [d.cls for d in dets])
    ok, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    counts = Counter(d.name for d in dets)
    return {
        "width": img.shape[1],
        "height": img.shape[0],
        "model": f"{settings.image_model.stem} ({detector.size} px, ONNX Runtime CPU)",
        "inference_ms": round(ms, 1),
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "detections": [d.public() for d in dets],
        "image": "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode(),
    }


@app.post("/api/jobs", status_code=202)
def create_job(file: UploadFile, sid: Session, request: Request):
    name = check_type(file.filename, VIDEO_TYPES, "video")
    # Check session limits before counting the upload against the IP's hourly allowance
    try:
        jobs.check_limits(sid)
    except ValueError as e:
        raise HTTPException(409, str(e))
    job_rate_limit(request)

    # Stream to a private temp dir in 1 MB pieces, stopping as soon as the size limit is passed
    work_dir = Path(tempfile.mkdtemp(prefix="vision-"))
    upload = work_dir / f"upload{Path(name).suffix.lower()}"
    limit = settings.max_video_mb * 1024 * 1024
    size = 0
    try:
        with upload.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Video larger than {settings.max_video_mb} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The file is empty")
        return jobs.create(sid, name, work_dir, upload).public()
    except ValueError as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(409, str(e))
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


@app.get("/api/jobs")
def list_jobs(sid: Session):
    return [j.public() for j in jobs.session_jobs(sid)]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, sid: Session):
    job = jobs.get(sid, job_id)
    if not job:
        raise HTTPException(404, "Job not found or expired")
    return job.public()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str, sid: Session):
    if not jobs.delete(sid, job_id):
        raise HTTPException(404, "Job not found, expired or still running")
    return {"deleted": job_id}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str, sid: Session):
    job = jobs.get(sid, job_id)
    if not job or not job.result or not job.result.exists():
        raise HTTPException(404, "No video for this job")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(job.filename).stem)[:60] or "video"
    return FileResponse(job.result, media_type="video/mp4", filename=f"{stem}-tracked.mp4")


# The browser downloads the model once and runs it locally; let it cache for a day
MODEL_FILES = {settings.browser_model, "classes.json"}


@app.get("/models/{name}", include_in_schema=False)
def model_file(name: str):
    if name not in MODEL_FILES:
        raise HTTPException(404, "Not found")
    path = settings.classes_file if name == "classes.json" else settings.video_model.parent / name
    media_type = "application/json" if name.endswith(".json") else "application/octet-stream"
    return FileResponse(path, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})
