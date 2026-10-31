"""Small SORT-style multi-object tracker in NumPy.

Each track predicts its next box with a constant-velocity model (a smoothed per-frame
box delta, a lightweight stand-in for SORT's Kalman filter). Detections are matched to
predictions greedily by IoU, within the same class. A track gets a visible ID only after
`min_hits` matches, so one-frame false positives never get counted, and it is dropped
after `max_age` frames without a match.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

# How much a new observed velocity replaces the old estimate (1 = no smoothing)
VELOCITY_ALPHA = 0.5


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N, 4) and (M, 4) xyxy boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


@dataclass
class Track:
    box: np.ndarray  # xyxy, last observed position
    cls: int
    score: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    hits: int = 1
    misses: int = 0  # frames since the last match
    id: int | None = None  # assigned once confirmed

    def predict(self) -> np.ndarray:
        """Expected box in the coming frame, extrapolated across any missed frames."""
        return self.box + self.velocity * (self.misses + 1)

    def update(self, box: np.ndarray, score: float) -> None:
        # Velocity is measured against the last real position, spread over any missed frames
        observed = (box - self.box) / (self.misses + 1)
        self.velocity = VELOCITY_ALPHA * observed + (1 - VELOCITY_ALPHA) * self.velocity
        self.box, self.score = box, score
        self.hits += 1
        self.misses = 0


class Tracker:
    def __init__(self, iou_threshold: float = 0.2, max_age: int = 5, min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: list[Track] = []
        self.next_id = 1
        # class id -> number of confirmed tracks ever seen: the "unique objects" count
        self.counts: dict[int, int] = defaultdict(int)

    def update(self, boxes: np.ndarray, classes: np.ndarray, scores: np.ndarray) -> list[Track]:
        """Feed one frame's detections. Returns the confirmed tracks matched in this frame."""
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(classes, dtype=np.int64).reshape(-1)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        predicted = np.array([t.predict() for t in self.tracks], dtype=np.float32).reshape(-1, 4)
        ious = iou_matrix(predicted, boxes)
        # Never match across classes
        if ious.size:
            ious[np.array([t.cls for t in self.tracks])[:, None] != classes[None, :]] = 0

        matched_tracks, matched_dets = set(), set()
        # Greedy assignment, best IoU first (Hungarian buys little at these densities)
        for flat in np.argsort(-ious, axis=None):
            ti, di = divmod(int(flat), ious.shape[1])
            if ious[ti, di] < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            self.tracks[ti].update(boxes[di], float(scores[di]))

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                # Keep coasting on the motion model so a briefly occluded object can be re-matched
                track.misses += 1

        for di in range(len(boxes)):
            if di not in matched_dets:
                self.tracks.append(Track(box=boxes[di], cls=int(classes[di]), score=float(scores[di])))

        visible = []
        for track in self.tracks:
            if track.id is None and track.hits >= self.min_hits:
                track.id = self.next_id
                self.next_id += 1
                self.counts[track.cls] += 1
            if track.id is not None and track.misses == 0:
                visible.append(track)

        self.tracks = [t for t in self.tracks if t.misses <= self.max_age and (t.id is not None or t.misses == 0)]
        return visible
