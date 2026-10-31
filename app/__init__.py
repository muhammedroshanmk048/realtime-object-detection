import os

from app.config import get_settings

# OpenCV refuses to decode images above this many pixels, before allocating them (pixel-bomb guard).
# It is read on first decode, so set it before any image is touched.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(get_settings().max_image_megapixels * 1_000_000))
