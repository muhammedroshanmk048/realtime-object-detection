from pathlib import Path

import cv2
import numpy as np
import pytest

from app.detector import Detector, decode, letterbox, nms, preprocess, scale_boxes

ROOT = Path(__file__).resolve().parent.parent


def test_letterbox_keeps_aspect_and_pads_grey():
    img = np.full((360, 640, 3), 255, dtype=np.uint8)
    out, r, (dw, dh) = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert r == 0.5
    assert (dw, dh) == (0, 70)
    assert (out[:70] == 114).all() and (out[-70:] == 114).all()
    assert (out[70:250] == 255).all()


def test_letterbox_odd_padding_matches_ultralytics_rounding():
    # 100x99 -> 320x317: 3 px of padding, Ultralytics puts 1 on the left and 2 on the right
    img = np.zeros((100, 99, 3), dtype=np.uint8)
    out, _, (dw, _) = letterbox(img, 320)
    assert out.shape == (320, 320, 3)
    assert dw == 1.5
    assert (out[:, 0] == 114).all() and (out[:, 1] == 0).all()
    assert (out[:, -2:] == 114).all() and (out[:, -3] == 0).all()


def test_preprocess_is_rgb_nchw_unit_range():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = 255  # blue in BGR
    blob, _, _ = preprocess(img, 32)
    assert blob.shape == (1, 3, 32, 32) and blob.dtype == np.float32
    assert blob[0, 2].max() == 1.0 and blob[0, 0].max() == 0.0  # blue ends up in the last (B) channel


def test_decode_end_to_end_filters_and_sorts():
    out = np.zeros((1, 300, 6), dtype=np.float32)
    out[0, 0] = [10, 10, 50, 50, 0.4, 2]
    out[0, 1] = [60, 60, 90, 90, 0.9, 0]
    out[0, 2] = [0, 0, 5, 5, 0.1, 1]
    dets = decode(out, conf=0.25, iou_thres=0.7)
    assert dets.shape == (2, 6)
    assert dets[0, 4] == pytest.approx(0.9) and dets[1, 4] == pytest.approx(0.4)


def test_decode_raw_head_runs_nms():
    # Raw (1, 4 + classes, anchors) layout: two overlapping anchors for the same object, one elsewhere
    out = np.zeros((1, 4 + 3, 3), dtype=np.float32)
    out[0, :4, 0] = [50, 50, 40, 40]
    out[0, 4 + 1, 0] = 0.9
    out[0, :4, 1] = [52, 51, 40, 40]
    out[0, 4 + 1, 1] = 0.8
    out[0, :4, 2] = [200, 200, 20, 20]
    out[0, 4 + 2, 2] = 0.7
    dets = decode(out, conf=0.25, iou_thres=0.5)
    assert dets.shape == (2, 6)
    np.testing.assert_allclose(dets[0], [30, 30, 70, 70, 0.9, 1], rtol=1e-6)
    assert dets[1, 5] == 2


def test_nms_is_class_aware():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.array([0, 0, 1])
    keep = nms(boxes, scores, classes, iou_thres=0.5)
    assert keep.tolist() == [0, 2]  # the duplicate of class 0 goes, the class-1 box stays


def test_scale_boxes_inverts_letterbox():
    shape = (360, 640)
    _, r, pad = letterbox(np.zeros((*shape, 3), dtype=np.uint8), 320)
    original = np.array([[100, 50, 300, 200]], dtype=np.float32)
    in_input = original * r + [pad[0], pad[1], pad[0], pad[1]]
    np.testing.assert_allclose(scale_boxes(in_input, r, pad, shape), original, atol=1e-4)


def test_scale_boxes_clips_to_image():
    out = scale_boxes(np.array([[-10, -10, 400, 400]], dtype=np.float32), 1.0, (0, 0), (100, 200))
    assert out.tolist() == [[0, 0, 200, 100]]


@pytest.mark.parametrize("size", [640, 320])
def test_detector_finds_people_in_sample(size):
    det = Detector(ROOT / "models" / f"yolo26n-{size}.onnx", ROOT / "models" / "classes.json")
    assert det.size == size
    img = cv2.imread(str(ROOT / "samples" / "street.jpg"))
    dets = det.detect(img, conf=0.4)
    names = [d.name for d in dets]
    assert names.count("person") >= 3
    h, w = img.shape[:2]
    for d in dets:
        x1, y1, x2, y2 = d.box
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
