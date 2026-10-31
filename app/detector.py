"""YOLO inference with onnxruntime + NumPy only (no PyTorch on the server).

Pre-processing matches Ultralytics' LetterBox exactly, so detections line up with
`model.predict` on the same image. Two output layouts are supported:

- end-to-end (YOLO26 exported with nms=False): (1, N, 6) rows of [x1, y1, x2, y2, score, class]
  in input pixels; already de-duplicated, so only a confidence filter is needed.
- raw head (e.g. YOLO11, or YOLO26 exported with nms=None): (1, 4 + classes, anchors) with
  [cx, cy, w, h, class scores...] per anchor; needs class-aware NMS, done here in NumPy.
"""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PAD_VALUE = 114  # Ultralytics' letterbox grey


@dataclass
class Detection:
    cls: int
    name: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in original image pixels

    def public(self) -> dict:
        return {
            "class": self.name,
            "class_id": self.cls,
            "confidence": round(self.confidence, 3),
            "box": [round(v, 1) for v in self.box],
        }


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to size x size. Returns (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = round(w * r), round(h * r)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Same rounding as Ultralytics, so an odd padding puts the extra pixel in the same place
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return img, r, (dw, dh)


def preprocess(bgr: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    img, r, pad = letterbox(bgr, size)
    blob = img[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR HWC -> RGB NCHW
    return np.ascontiguousarray(blob), r, pad


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one xyxy box against an (N, 4) array."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area + areas - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thres: float) -> np.ndarray:
    """Class-aware greedy NMS. Returns kept indices, highest score first."""
    # Offsetting each class far apart means boxes of different classes never overlap
    shifted = boxes + classes[:, None].astype(np.float32) * 10_000
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        order = order[1:][iou(shifted[i], shifted[order[1:]]) <= iou_thres]
    return np.array(keep, dtype=np.int64)


def decode(output: np.ndarray, conf: float, iou_thres: float, max_det: int = 300) -> np.ndarray:
    """Model output -> (K, 6) [x1, y1, x2, y2, score, class] in letterboxed input pixels."""
    out = output[0]
    if out.ndim == 2 and out.shape[-1] == 6:  # end-to-end head
        dets = out[out[:, 4] >= conf]
        return dets[np.argsort(-dets[:, 4])][:max_det]

    preds = out.T  # (anchors, 4 + classes)
    cls_scores = preds[:, 4:]
    classes = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(preds)), classes]
    mask = scores >= conf
    if not mask.any():
        return np.zeros((0, 6), dtype=np.float32)
    xywh, scores, classes = preds[mask, :4], scores[mask], classes[mask]
    boxes = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], axis=1)
    keep = nms(boxes, scores, classes, iou_thres)[:max_det]
    return np.concatenate([boxes[keep], scores[keep, None], classes[keep, None].astype(np.float32)], axis=1)


def scale_boxes(boxes: np.ndarray, r: float, pad: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    """Undo the letterbox: input pixels -> original image pixels, clipped to the image."""
    h, w = shape
    out = boxes.copy()
    out[:, [0, 2]] = ((out[:, [0, 2]] - round(pad[0] - 0.1)) / r).clip(0, w)
    out[:, [1, 3]] = ((out[:, [1, 3]] - round(pad[1] - 0.1)) / r).clip(0, h)
    return out


class Detector:
    def __init__(self, model_path: Path, classes_path: Path, threads: int = 1):
        opts = ort.SessionOptions()
        # The free host has a fraction of one CPU: more threads only add contention
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2])
        self.names: list[str] = json.loads(Path(classes_path).read_text())

    def detect(self, bgr: np.ndarray, conf: float = 0.25, iou_thres: float = 0.7) -> list[Detection]:
        blob, r, pad = preprocess(bgr, self.size)
        output = self.session.run(None, {self.input_name: blob})[0]
        dets = decode(output, conf, iou_thres)
        if not len(dets):
            return []
        boxes = scale_boxes(dets[:, :4], r, pad, bgr.shape[:2])
        return [
            Detection(int(c), self.names[int(c)], float(s), tuple(float(v) for v in b))
            for b, s, c in zip(boxes, dets[:, 4], dets[:, 5])
        ]


@lru_cache
def load(model_path: Path, classes_path: Path, threads: int = 1) -> Detector:
    """One shared session per model file; onnxruntime sessions are safe to run from several threads."""
    return Detector(model_path, classes_path, threads)


def color_for(key: int) -> tuple[int, int, int]:
    """Stable, well-spread BGR colour per class or track id (golden-ratio hue steps)."""
    hue = int((key * 0.618034 % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def draw(img: np.ndarray, boxes: list[tuple], labels: list[str], keys: list[int]) -> np.ndarray:
    """Draw xyxy boxes with filled labels in place. Line width scales with image size."""
    lw = max(2, round(sum(img.shape[:2]) / 2 * 0.003))
    scale = lw * 0.28
    for (x1, y1, x2, y2), label, key in zip(boxes, labels, keys):
        color = color_for(key)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, lw, cv2.LINE_AA)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, max(lw - 1, 1))
        top = p1[1] - th - base - 4 >= 0
        y_text = p1[1] - base - 2 if top else p1[1] + th + 2
        cv2.rectangle(img, (p1[0], y_text - th - 2), (p1[0] + tw + 4, y_text + base), color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (p1[0] + 2, y_text), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                    max(lw - 1, 1), cv2.LINE_AA)
    return img
