import numpy as np

from app.tracker import Tracker, iou_matrix


def frame(*objs):
    """objs: (box, cls) pairs -> arrays the tracker takes."""
    boxes = np.array([b for b, _ in objs], dtype=np.float32).reshape(-1, 4)
    classes = np.array([c for _, c in objs], dtype=np.int64)
    return boxes, classes, np.full(len(objs), 0.9, dtype=np.float32)


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]], rtol=1e-6)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (1, 0)


def test_moving_object_keeps_its_id():
    t = Tracker(iou_threshold=0.2, max_age=3, min_hits=2)
    ids = set()
    for i in range(10):
        x = 10 + i * 8  # moves 8 px per frame on a 40 px box
        visible = t.update(*frame(([x, 10, x + 40, 50], 0)))
        ids.update(tr.id for tr in visible)
    assert ids == {1}
    assert t.counts == {0: 1}


def test_needs_min_hits_before_getting_an_id():
    t = Tracker(min_hits=3)
    assert t.update(*frame(([0, 0, 10, 10], 0))) == []
    assert t.update(*frame(([0, 0, 10, 10], 0))) == []
    assert [tr.id for tr in t.update(*frame(([0, 0, 10, 10], 0)))] == [1]


def test_one_frame_false_positive_is_never_counted():
    t = Tracker(min_hits=2)
    t.update(*frame(([0, 0, 10, 10], 0)))
    for _ in range(5):
        t.update(*frame())
    assert t.counts == {} and t.tracks == []


def test_motion_model_bridges_a_short_occlusion():
    t = Tracker(iou_threshold=0.3, max_age=3, min_hits=2)
    x = 0
    for _ in range(5):
        x += 20
        t.update(*frame(([x, 0, x + 40, 40], 0)))
    for _ in range(2):  # hidden for two frames while it keeps moving
        x += 20
        assert t.update(*frame()) == []
    x += 20
    visible = t.update(*frame(([x, 0, x + 40, 40], 0)))
    # Without velocity the box would be 60 px away from its last position: zero overlap, new ID
    assert [tr.id for tr in visible] == [1]


def test_track_dropped_after_max_age():
    t = Tracker(max_age=2, min_hits=1)
    t.update(*frame(([0, 0, 10, 10], 0)))
    for _ in range(3):
        t.update(*frame())
    assert t.tracks == []
    assert [tr.id for tr in t.update(*frame(([0, 0, 10, 10], 0)))] == [2]


def test_classes_never_match_each_other():
    t = Tracker(min_hits=1)
    t.update(*frame(([0, 0, 10, 10], 0)))
    visible = t.update(*frame(([0, 0, 10, 10], 2)))
    assert [tr.cls for tr in visible] == [2] and visible[0].id == 2
    assert t.counts == {0: 1, 2: 1}


def test_two_crossing_objects_keep_separate_ids():
    t = Tracker(iou_threshold=0.2, max_age=3, min_hits=1)
    first = {}
    for i in range(6):
        a = [i * 10, 0, i * 10 + 30, 30]
        b = [200 - i * 10, 100, 230 - i * 10, 130]
        for tr in t.update(*frame((a, 0), (b, 0))):
            first.setdefault(round(float(tr.box[1])), set()).add(tr.id)
    assert first == {0: {1}, 100: {2}}
