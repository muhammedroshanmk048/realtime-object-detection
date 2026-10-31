# Static ffmpeg/ffprobe build (with libx264): ~120 MB, versus ~450 MB for Debian's ffmpeg and its X/audio deps
FROM mwader/static-ffmpeg:9.0.2 AS ffmpeg

FROM python:3.13-slim

# ffmpeg decodes uploads and encodes browser-playable H.264; inference is ONNX Runtime, so no PyTorch here
COPY --from=ffmpeg /ffmpeg /ffprobe /usr/local/bin/

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
# Render (and most hosts) set PORT; default to 8000 when run locally
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
