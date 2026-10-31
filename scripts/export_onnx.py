"""Export the Ultralytics nano model to ONNX for the deployed app and the browser.

The server and the browser run these files with onnxruntime / onnxruntime-web, so
neither needs PyTorch. Each export has a fixed square input; the app reads the size
from the model. YOLO26 is NMS-free (end-to-end), so the graph already outputs final
detections as (1, 300, 6) rows of [x1, y1, x2, y2, score, class] in input pixels and
post-processing is just a confidence filter + un-letterboxing. Older models (e.g.
YOLO11) export without NMS; the detector handles that layout too.

    python scripts/export_onnx.py                  # models/yolo26n-640.onnx + yolo26n-320.onnx
    python scripts/export_onnx.py --sizes 416
"""

import argparse
import json
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO

MODELS = Path(__file__).resolve().parent.parent / "models"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=str(MODELS / "yolo26n.pt"))
    p.add_argument("--sizes", type=int, nargs="+", default=[640, 320])
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    model = YOLO(args.model)
    stem = Path(args.model).stem
    for size in args.sizes:
        # nms=False selects YOLO26's NMS-free one-to-one head (the default, None, exports the raw head
        # for external NMS). simplify folds constants; a fixed input runs fastest in onnxruntime-web.
        exported = Path(model.export(
            format="onnx", imgsz=size, opset=args.opset, simplify=True, dynamic=False, nms=False
        ))
        dst = MODELS / f"{stem}-{size}.onnx"
        shutil.move(exported, dst)

        proto = onnx.load(dst)
        graph = proto.graph
        meta = {m.key: m.value for m in proto.metadata_props}
        inp = [d.dim_value for d in graph.input[0].type.tensor_type.shape.dim]
        out = [d.dim_value for d in graph.output[0].type.tensor_type.shape.dim]
        print(f"{dst.name}: input {inp} output {out} end2end={meta.get('end2end', 'n/a')} "
              f"{dst.stat().st_size / 1e6:.1f} MB")

    # Class names travel with the app (the browser can't read ONNX metadata cheaply)
    (MODELS / "classes.json").write_text(json.dumps([model.names[i] for i in range(len(model.names))]))


if __name__ == "__main__":
    main()
