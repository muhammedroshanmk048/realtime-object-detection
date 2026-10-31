"""Real-time detection + multi-object tracking on a webcam or video file.

Local PyTorch/Ultralytics showcase (not deployed). Uses Ultralytics' built-in
ByteTrack tracker, draws boxes with track IDs and motion trails, shows FPS, and
counts unique objects per class. An optional horizontal line counts crossings.

    python scripts/webcam_track.py                       # default webcam
    python scripts/webcam_track.py --source clip.mp4 --line 0.6 --save out.mp4

Keys: q / Esc quit, r reset counts and trails.
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

TRAIL_LEN = 30
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="webcam index or video path (default: 0)")
    p.add_argument("--model", default="models/yolo26n.pt", help="Ultralytics nano weights (default: models/yolo26n.pt)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--classes", type=int, nargs="*", help="only these COCO class ids (e.g. 0 2 for person, car)")
    p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--line", type=float, help="count crossings of a horizontal line at this height fraction (0-1)")
    p.add_argument("--save", help="write the annotated stream to this .mp4 path")
    p.add_argument("--no-show", action="store_true", help="don't open a window (useful with --save)")
    return p.parse_args()


def color_for(track_id: int) -> tuple[int, int, int]:
    # Stable, well-spread colour per track id (golden-ratio hue steps).
    hue = int((track_id * 0.618034 % 1.0) * 179)
    bgr = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def draw_label(img, text: str, x: int, y: int, color) -> None:
    (w, h), base = cv2.getTextSize(text, FONT, 0.5, 1)
    y = max(y, h + base + 2)
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - base), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_panel(img, lines: list[str]) -> None:
    pad, lh = 8, 20
    width = max(cv2.getTextSize(t, FONT, 0.55, 1)[0][0] for t in lines) + 2 * pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (width, pad * 2 + lh * len(lines)), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (pad, pad + lh * (i + 1) - 5), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    model = YOLO(args.model)
    names = model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source {args.source!r}")

    writer = None
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    seen: dict[str, set[int]] = defaultdict(set)  # class name -> unique track ids
    last_y: dict[int, float] = {}
    crossed: set[int] = set()
    crossings = {"down": 0, "up": 0}
    fps = 0.0
    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]

        result = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            classes=args.classes,
            verbose=False,
        )[0]

        line_y = int(args.line * h) if args.line is not None else None
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for (x1, y1, x2, y2), tid, cls, conf in zip(
                boxes.xyxy.int().tolist(), boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.conf.tolist()
            ):
                name = names[cls]
                color = color_for(tid)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                seen[name].add(tid)
                trails[tid].append((cx, cy))

                # Count each track once, so box jitter around the line can't inflate the total.
                if line_y is not None and tid in last_y and tid not in crossed:
                    if last_y[tid] < line_y <= cy:
                        crossings["down"] += 1
                        crossed.add(tid)
                    elif last_y[tid] >= line_y > cy:
                        crossings["up"] += 1
                        crossed.add(tid)
                last_y[tid] = cy

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, f"#{tid} {name} {conf:.2f}", x1, y1, color)
                pts = np.array(trails[tid], dtype=np.int32)
                if len(pts) > 1:
                    cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)

        if line_y is not None:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 165, 255), 2)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6)) if fps else 1.0 / max(now - prev, 1e-6)
        prev = now

        panel = [f"FPS {fps:5.1f}   {args.model}  {args.imgsz}px"]
        panel += [f"{name}: {len(ids)}" for name, ids in sorted(seen.items(), key=lambda kv: -len(kv[1]))[:8]]
        if line_y is not None:
            panel.append(f"line  down {crossings['down']}  up {crossings['up']}")
        draw_panel(frame, panel)

        if args.save:
            if writer is None:
                src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("Real-time detection + tracking (q to quit)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                trails.clear()
                seen.clear()
                last_y.clear()
                crossed.clear()
                crossings = {"down": 0, "up": 0}

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("Unique objects per class:", {k: len(v) for k, v in sorted(seen.items())})
    if args.line is not None:
        print("Line crossings:", crossings)


if __name__ == "__main__":
    main()
