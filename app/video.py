"""Video in and out through ffmpeg pipes.

Decoding: ffmpeg samples the clip at a fixed fps, downscales it and streams raw BGR
frames to stdout, so only one frame is in memory at a time and any codec ffmpeg reads
works. Encoding: annotated frames go back through ffmpeg as H.264 (yuv420p, faststart),
which every browser plays. OpenCV's pip wheels can't write H.264, hence ffmpeg.
"""

import json
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class VideoError(ValueError):
    pass


@dataclass
class VideoInfo:
    width: int  # display size, after applying rotation metadata
    height: int
    duration: float


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def probe(path: Path) -> VideoInfo:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration:stream_side_data=rotation:format=duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is not installed or not on PATH")
    except subprocess.TimeoutExpired:
        raise VideoError("Reading the video took too long")
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if result.returncode != 0 or not streams:
        raise VideoError("Could not find a video track in this file")
    s = streams[0]
    width, height = int(s.get("width") or 0), int(s.get("height") or 0)
    if not width or not height:
        raise VideoError("Could not read the video size")
    # Phone clips are often stored landscape with a rotation flag; ffmpeg applies it when decoding
    rotation = next((int(d["rotation"]) for d in s.get("side_data_list", []) if "rotation" in d), 0)
    if rotation % 180:
        width, height = height, width
    duration = float(s.get("duration") or data.get("format", {}).get("duration") or 0)
    if duration <= 0:
        raise VideoError("Could not read the video length")
    return VideoInfo(width, height, duration)


def output_size(info: VideoInfo, max_side: int) -> tuple[int, int]:
    """Fit inside max_side (never upscale), with even dimensions as H.264 4:2:0 requires."""
    r = min(1.0, max_side / max(info.width, info.height))
    return max(2, round(info.width * r / 2) * 2), max(2, round(info.height * r / 2) * 2)


def read_frames(path: Path, size: tuple[int, int], fps: float, max_seconds: float) -> Iterator[np.ndarray]:
    w, h = size
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-t", f"{max_seconds:.3f}", "-an",
        "-vf", f"fps={fps},scale={w}:{h}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    finished = False
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()
        finished = True
    finally:
        if not finished:  # the consumer stopped early (e.g. an error downstream): don't mask it
            proc.kill()
        proc.stdout.close()
        stderr = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        proc.wait()
    if proc.returncode != 0:
        lines = stderr.strip().splitlines()
        # Hide the server path ffmpeg puts in front of the reason
        reason = lines[-1].rsplit(": ", 1)[-1] if lines else "unknown error"
        raise VideoError(f"Could not decode this video: {reason}")


class Writer:
    """Context manager that pipes BGR frames into an H.264 MP4."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float):
        w, h = size
        self.path = path
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-nostdin", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0",
                # ultrafast: the free host's CPU belongs to detection; crf 26 keeps 15 s clips ~1 MB
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(frame.tobytes())

    def __enter__(self) -> "Writer":
        return self

    def __exit__(self, exc_type, *_) -> None:
        self.proc.stdin.close()
        stderr = self.proc.stderr.read().decode(errors="replace")
        self.proc.stderr.close()
        if self.proc.wait() != 0 and exc_type is None:
            raise RuntimeError(f"Encoding failed: {stderr.strip()[-300:]}")


def process(
    src: Path,
    dst: Path,
    info: VideoInfo,
    fps: float,
    max_side: int,
    max_seconds: float,
    annotate: Callable[[np.ndarray], np.ndarray],
    progress: Callable[[int, int], None] = lambda done, total: None,
) -> int:
    """Decode, annotate every sampled frame and encode. Returns the number of frames processed."""
    size = output_size(info, max_side)
    total = max(1, int(min(info.duration, max_seconds) * fps))
    count = 0
    with Writer(dst, size, fps) as writer:
        for frame in read_frames(src, size, fps, max_seconds):
            writer.write(annotate(frame))
            count += 1
            progress(count, total)
    if not count:
        raise VideoError("The video has no readable frames")
    return count
