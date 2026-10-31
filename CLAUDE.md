# CLAUDE.md

## What this project is

**Real-Time Object Detection System**: a mini portfolio project. A computer vision app that detects and tracks objects in images and live video streams.

Required stack: Python, Ultralytics YOLO, OpenCV, PyTorch.
Skills it must demonstrate: object detection, multi-object tracking, video processing, model export/optimization, deployment trade-offs.

Owner: Muhammed Roshan M K, backend engineer (Python, FastAPI, GCP). Public portfolio project, meant to be light and showcase-focused.
- Portfolio site: sibling repo `../project-zap` (live at https://muhammedroshanmk.vercel.app). It has a "Personal Projects" section with cards for DocChat and MeetScribe; add a card for this project in the same style once it's deployed. Résumé PDF there is built by a ReportLab script (ask the owner before changing it).
- Earlier mini projects with the same conventions, reuse their patterns (config, Dockerfile, render.yaml, session isolation, rate limiting, UI layout):
  - `../rag-document-assistant`: DocChat, https://docchat-ft5u.onrender.com
  - `../meeting-transcription-assistant`: MeetScribe, https://meetscribe-5j70.onrender.com (background-job pipeline with progress polling, ffmpeg in Docker)

## Hard constraint: everything must stay free

No paid service, no credit card, no paid API. This project needs **no API keys at all**.

## Agreed architecture (decided with the owner)

Render free gives ~512 MB RAM and 0.1 CPU, so PyTorch cannot be deployed and server-side live video can't be real-time. The stack is split by where each piece fits:

### 1. Local Python side (owner's PC, not deployed): the PyTorch / Ultralytics showcase
- Ultralytics YOLO **nano** model (use the current Ultralytics nano detector, e.g. YOLO11n or newer) with PyTorch on CPU.
- `scripts/webcam_track.py`: real-time webcam (or video file) detection + tracking in an OpenCV window using Ultralytics' built-in tracker (ByteTrack); FPS overlay; track IDs and trails; optional per-class counting / line-crossing counter.
- `scripts/export_onnx.py`: export the model to ONNX (fixed input size, e.g. 640 or 320) into `models/`. Record whether NMS is inside the graph (`nms=True` export or an NMS-free model); post-processing must match.
- Optional later: fine-tune on a small public dataset (`scripts/train.py`) to show training.
- Record a short GIF of the local real-time run for the README.
- Local-only deps go in `requirements-local.txt` (ultralytics, torch CPU, opencv-python). Never imported by the deployed app.

### 2. Deployed FastAPI app on Render (Docker, Blueprint via `render.yaml`)
- Deps: `fastapi`, `uvicorn`, `python-multipart`, `pydantic-settings`, `onnxruntime`, `opencv-python-headless`, `numpy`. **No torch, no ultralytics.** Target image well under 1 GB and peak RAM under ~350 MB; set onnxruntime `intra_op_num_threads=1`.
- Load the ONNX model once at startup. Implement preprocessing (letterbox, normalize) and post-processing (decode boxes, confidence filter, NMS if needed) in NumPy, matching the exported model.
- **Image endpoint**: upload a JPG/PNG and get back the annotated image plus JSON (`class`, `confidence`, `box`, counts per class).
- **Short video endpoint** as a background job (MeetScribe's pattern: `POST /api/jobs`, poll `GET /api/jobs/{id}`): limit clips to ~15 s / ~20 MB, sample frames (e.g. 5 fps, downscale), detect, track with a small **SORT-style tracker written in NumPy** (IoU matching + simple motion), draw boxes and IDs, and return the annotated video plus per-class unique-object counts.
  - Gotcha: OpenCV pip wheels can't write browser-playable H.264. Install ffmpeg in the Docker image (apt) and encode the output as H.264 MP4 (or use WebM/VP8), then verify it plays in Chrome.
- Public-demo safety, same as DocChat/MeetScribe: browser-generated `X-Session-Id`, upload type/size/duration limits, per-IP rate limit, one active job per session, files processed in a temp dir and deleted, results expire.

### 3. Live webcam page (runs in the visitor's browser)
- Real-time detection on the webcam using **onnxruntime-web** (WASM, WebGPU if available) loading the same ONNX model from the server (`/models/...`, cache headers). Zero server CPU.
- JS pre/post-processing mirrors the Python version; draw boxes on a canvas overlay; show FPS; optional simple tracker in JS for IDs.
- Load onnxruntime-web from a CDN with an exact pinned version. Camera access only works on HTTPS or localhost (Render is HTTPS).

## UI

Plain HTML/CSS/JS, same layout family as DocChat/MeetScribe (Inter / JetBrains Mono, dark). Give it its own accent color, distinct from DocChat (violet `#7c5cff`) and MeetScribe (emerald `#10b981`), e.g. orange `#f97316`; confirm with the owner. Pick a short brand name and confirm it with the owner. Tabs or sections: Image · Video · Live webcam. Show detections list / counts beside the media.

## Suggested layout

- `app/config.py`: pydantic-settings
- `app/detector.py`: ONNX session, preprocess, postprocess, NMS, draw
- `app/tracker.py`: NumPy SORT-style tracker
- `app/video.py`: frame sampling, encode via ffmpeg
- `app/jobs.py`: in-memory job store with TTL + worker
- `app/main.py`: routes, static files, model file serving
- `static/`: UI, including the browser inference code
- `models/`: exported ONNX model (commit it if small, ~10 MB, so the Docker build doesn't need PyTorch)
- `scripts/`: local PyTorch/Ultralytics scripts (webcam tracking, export, optional training)
- `tests/`: unit tests for pre/post-processing, NMS, tracker
- `Dockerfile`, `render.yaml`, `requirements.txt`, `requirements-local.txt`, `.gitignore`, `.dockerignore`, `README.md`, `LICENSE`

## License

Ultralytics YOLO and its weights are AGPL-3.0. Keep this repo public and add an AGPL-3.0 `LICENSE`. Mention it in the README.

## Plan / status (as of 2026-10-31)

Steps 1–7 done and committed locally (not pushed; the owner pushes). Decisions made with the owner: brand **VisionTrack**, accent orange `#f97316`.

What exists and what was decided along the way:
- **Model:** YOLO26n, exported with `nms=False` (NMS-free, output `(1,300,6)`). There are two exports: 640 px for images and 320 px for video and the browser.
- **Parity:** `scripts/check_parity.py` passes exactly against Ultralytics running the same ONNX file. PyTorch differs slightly because it uses a rectangular letterbox.
- **Docker:** base image is `python:3.13-slim`, not 3.11, because numpy 2.5 needs Python 3.12+. ffmpeg comes from the static `mwader/static-ffmpeg:9.0.2` image. Image is ~640 MB unpacked; peak RAM ~306 MB under `--memory 512m --cpus 0.1`.
- **Browser:** onnxruntime-web 1.30.0 from jsDelivr. Verified in Chrome via Playwright with a fake camera: WebGPU ~20 FPS. Video playback and the image tab also verified.
- **Tests:** 26 pytest tests pass (`requirements-dev.txt`).
- **Not done:** the optional `scripts/train.py`. The README GIF comes from `webcam_track.py` run on the sample clip, not from a webcam; the owner can re-record it from the webcam.

Deployed: https://visiontrack-45cp.onrender.com (Render service `visiontrack`), verified live in Chrome on 2026-10-31. Portfolio card added in ../project-zap (pending owner approval to push).

Steps:
1. Local env: `.venv`, install `requirements-local.txt` (CPU torch from the PyTorch CPU index; large, run in the background). Confirm the nano model runs on the webcam with tracking (`scripts/webcam_track.py`).
2. Export to ONNX; write `app/detector.py` and check its detections match Ultralytics on a few sample images.
3. Build the FastAPI image endpoint + UI; test locally.
4. Add the tracker + video job; test with a short sample clip (keep one public-domain clip in `samples/`).
5. Add the live webcam page with onnxruntime-web; test in Chrome on localhost.
6. Tests, pin versions, README (architecture diagram, why ONNX instead of PyTorch on the server, demo GIF, limits, license).
7. Docker build locally, run with `--memory 512m --cpus 0.1` to check memory and speed.
8. Commit and push (ask the owner first), deploy via Render Blueprint, verify live.
9. Add the project card to `../project-zap` and push after the owner approves.

## Environment notes

- Windows 11, Python 3.13 locally (Docker image: python:3.11-slim). Shell: PowerShell or Git Bash. ffmpeg is installed locally (winget).
- `gh` CLI is not installed; the owner pushes via VS Code or git with their saved login.
- Large pip installs are slow; run them in the background.
- Commit messages end with a `Co-Authored-By` line. Ask the owner before pushing.
