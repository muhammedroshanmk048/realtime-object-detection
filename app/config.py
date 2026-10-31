from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Two exports of the same YOLO26n weights. 640 px is more accurate for single images;
    # 320 px is ~4x cheaper, so it runs video on the free host's 0.1 CPU and live in the browser.
    image_model: Path = ROOT / "models" / "yolo26n-640.onnx"
    video_model: Path = ROOT / "models" / "yolo26n-320.onnx"
    browser_model: str = "yolo26n-320.onnx"
    classes_file: Path = ROOT / "models" / "classes.json"
    # More threads only add contention on a fractional CPU
    onnx_threads: int = 1
    default_conf: float = 0.25
    iou_threshold: float = 0.7

    # Images
    max_image_mb: int = 8
    # Bigger images are downscaled after decoding; the annotated result is a JPEG of this size
    max_image_side: int = 1920
    # Decoding refuses anything above this before allocating it (guards against pixel bombs)
    max_image_megapixels: int = 40

    # Short video jobs: frames are sampled and downscaled before detection
    max_video_mb: int = 20
    max_video_seconds: float = 15
    video_fps: float = 5
    video_max_side: int = 640

    # Tracker (SORT-style: constant-velocity prediction + IoU matching)
    track_iou: float = 0.2
    track_max_age: int = 5  # frames a track survives without a match (1 s at 5 fps)
    track_min_hits: int = 2  # matches before a track gets an ID and is counted

    # Limits for a public deployment: each browser session only sees its own jobs,
    # uploads are deleted right after processing and results expire after job_ttl_minutes
    job_ttl_minutes: float = 30
    max_jobs_per_session: int = 5
    jobs_per_hour_per_ip: int = 10
    images_per_hour_per_ip: int = 60
    cleanup_interval_minutes: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
