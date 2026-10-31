import logging
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app import video
from app.config import get_settings
from app.detector import draw, load
from app.tracker import Tracker

log = logging.getLogger("jobs")
settings = get_settings()

ACTIVE = {"queued", "reading", "detecting"}


@dataclass
class Job:
    id: str
    session_id: str
    filename: str
    work_dir: Path
    status: str = "queued"
    message: str = "Waiting for a free worker"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    duration: float | None = None
    frames: int = 0
    seconds: float | None = None  # processing time
    counts: dict[str, int] = field(default_factory=dict)
    result: Path | None = None
    error: str | None = None

    def public(self) -> dict:
        data = {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "message": self.message,
            "progress": round(self.progress, 3),
            "created_at": int(self.created_at),
            "expires_at": int(self.expires_at) if self.expires_at else None,
            "duration": self.duration,
            "frames": self.frames,
            "seconds": round(self.seconds, 1) if self.seconds else None,
            "counts": self.counts,
            "has_video": bool(self.result and self.result.exists()),
            "error": self.error,
        }
        if self.status == "queued":
            data["position"] = queue_position(self)
        return data


_jobs: dict[str, Job] = {}
_lock = threading.Lock()
# One job at a time: the free host has a tenth of a CPU
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")


def queue_position(job: Job) -> int:
    with _lock:
        return sum(1 for j in _jobs.values() if j.status == "queued" and j.created_at <= job.created_at)


def session_jobs(session_id: str) -> list[Job]:
    with _lock:
        return sorted((j for j in _jobs.values() if j.session_id == session_id), key=lambda j: -j.created_at)


def get(session_id: str, job_id: str) -> Job | None:
    with _lock:
        job = _jobs.get(job_id)
    return job if job and job.session_id == session_id else None


def check_limits(session_id: str) -> None:
    jobs = session_jobs(session_id)
    if any(j.status in ACTIVE for j in jobs):
        raise ValueError("You already have a video in progress. Wait for it to finish first.")
    if len(jobs) >= settings.max_jobs_per_session:
        raise ValueError(f"Limit of {settings.max_jobs_per_session} videos reached. Delete one first.")


def create(session_id: str, filename: str, work_dir: Path, upload: Path) -> Job:
    """Register a job for an already-saved upload and queue it. Raises ValueError when a limit is hit."""
    check_limits(session_id)
    job = Job(id=uuid.uuid4().hex[:12], session_id=session_id, filename=filename, work_dir=work_dir)
    with _lock:
        _jobs[job.id] = job
    _worker.submit(run, job, upload)
    return job


def _remove(job: Job) -> None:
    with _lock:
        _jobs.pop(job.id, None)
    shutil.rmtree(job.work_dir, ignore_errors=True)


def delete(session_id: str, job_id: str) -> bool:
    job = get(session_id, job_id)
    if not job or job.status in ACTIVE:
        return False
    _remove(job)
    return True


def purge_expired() -> int:
    now = time.time()
    with _lock:
        expired = [j for j in _jobs.values() if j.expires_at and j.expires_at < now]
    for job in expired:
        _remove(job)
    return len(expired)


def _set(job: Job, status: str, message: str) -> None:
    job.status, job.message = status, message
    log.info("Job %s: %s (%s) at +%.1fs", job.id, status, message, time.time() - job.created_at)


def run(job: Job, upload: Path) -> None:
    start = time.time()
    try:
        _set(job, "reading", "Reading video")
        info = video.probe(upload)
        job.duration = round(info.duration, 2)
        if info.duration > settings.max_video_seconds + 0.5:
            raise ValueError(
                f"Clip is {info.duration:.0f} s long; the limit is {settings.max_video_seconds:.0f} s. Trim it first."
            )

        detector = load(settings.video_model, settings.classes_file, settings.onnx_threads)
        tracker = Tracker(settings.track_iou, settings.track_max_age, settings.track_min_hits)

        def annotate(frame: np.ndarray) -> np.ndarray:
            dets = detector.detect(frame, conf=settings.default_conf, iou_thres=settings.iou_threshold)
            tracks = tracker.update(
                np.array([d.box for d in dets]), np.array([d.cls for d in dets]), np.array([d.confidence for d in dets])
            )
            return draw(
                frame,
                [tuple(t.box) for t in tracks],
                [f"#{t.id} {detector.names[t.cls]}" for t in tracks],
                [t.id for t in tracks],
            )

        def progress(done: int, total: int) -> None:
            job.progress = min(done / total, 1.0)
            job.message = f"Detecting and tracking: frame {done} of ~{total}"

        _set(job, "detecting", "Detecting and tracking")
        result = job.work_dir / "result.mp4"
        job.frames = video.process(
            upload, result, info, settings.video_fps, settings.video_max_side, settings.max_video_seconds,
            annotate, progress,
        )
        upload.unlink(missing_ok=True)
        job.result = result
        job.counts = dict(sorted(
            ((detector.names[c], n) for c, n in tracker.counts.items()), key=lambda kv: (-kv[1], kv[0])
        ))
        job.progress = 1.0
        _set(job, "done", "Done")
    except ValueError as e:
        job.error = str(e)
        _set(job, "error", "Failed")
    except Exception as e:
        log.exception("Job %s failed", job.id)
        job.error = f"Processing failed: {e}"
        _set(job, "error", "Failed")
    finally:
        job.seconds = time.time() - start
        # The upload is never kept; only the annotated result stays, until the job expires
        upload.unlink(missing_ok=True)
        if job.status == "error":
            shutil.rmtree(job.work_dir, ignore_errors=True)
        job.expires_at = time.time() + settings.job_ttl_minutes * 60
