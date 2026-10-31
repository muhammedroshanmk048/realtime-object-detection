"""Check that the deployed detector (onnxruntime + NumPy) matches Ultralytics.

Two references per image and size:
- strict: Ultralytics running the same .onnx file. Same weights and input, so this isolates
  our letterbox/decode/scale code; boxes and confidences should agree to float precision.
- info:   Ultralytics running the PyTorch .pt. It letterboxes to a rectangle (minimal padding)
  instead of the fixed square the ONNX export needs, so confidences drift by a few points.

    python scripts/check_parity.py                      # Ultralytics' bundled sample images
    python scripts/check_parity.py --images a.jpg b.jpg --sizes 320
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.detector import Detector, iou  # noqa: E402


def compare(ours, ref) -> tuple[int, int, float, float]:
    """Match each of our detections to the best same-class reference box.
    Returns (ours, reference count, min IoU, max |confidence difference|)."""
    ref_boxes, ref_cls, ref_conf = ref.xyxy.numpy(), ref.cls.numpy().astype(int), ref.conf.numpy()
    worst_iou, worst_dconf = 1.0, 0.0
    for d in ours:
        same = ref_cls == d.cls
        if not same.any():
            return len(ours), len(ref_boxes), 0.0, 1.0
        ious = iou(np.array(d.box), ref_boxes[same])
        j = ious.argmax()
        worst_iou = min(worst_iou, float(ious[j]))
        worst_dconf = max(worst_dconf, abs(d.confidence - float(ref_conf[same][j])))
    return len(ours), len(ref_boxes), worst_iou, worst_dconf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", nargs="*", default=[str(ASSETS / "bus.jpg"), str(ASSETS / "zidane.jpg")])
    p.add_argument("--sizes", type=int, nargs="+", default=[640, 320])
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    torch_model = YOLO(ROOT / "models" / "yolo26n.pt")
    ok = True
    for size in args.sizes:
        path = ROOT / "models" / f"yolo26n-{size}.onnx"
        ours_model = Detector(path, ROOT / "models" / "classes.json")
        ref_model = YOLO(path, task="detect")
        for image in args.images:
            img = cv2.imread(image)
            ours = ours_model.detect(img, conf=args.conf)
            strict = ref_model.predict(img, imgsz=size, conf=args.conf, verbose=False)[0].boxes
            torch = torch_model.predict(img, imgsz=size, conf=args.conf, verbose=False)[0].boxes

            n, m, min_iou, dconf = compare(ours, strict)
            passed = n == m and min_iou >= 0.999 and dconf <= 1e-3
            ok &= passed
            tn, tm, t_iou, t_dconf = compare(ours, torch)
            print(f"[{'PASS' if passed else 'FAIL'}] {Path(image).name} @{size}: {n} vs {m} detections, "
                  f"min IoU {min_iou:.4f}, max |dconf| {dconf:.4f}  |  vs PyTorch: {tn} vs {tm}, "
                  f"min IoU {t_iou:.3f}, max |dconf| {t_dconf:.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
