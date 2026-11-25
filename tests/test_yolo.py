from pathlib import Path

import numpy as np

from visionguard.data.yolo import load_labels, save_labels, xywh_norm_to_xyxy


def test_label_roundtrip(tmp_path: Path):
    path = tmp_path / "img.txt"
    classes = np.array([0, 2])
    boxes = np.array([[0.5, 0.5, 0.25, 0.5], [0.2, 0.3, 0.1, 0.2]], dtype=np.float32)
    save_labels(path, classes, boxes)
    loaded_classes, loaded_boxes = load_labels(path)
    assert loaded_classes.tolist() == [0, 2]
    np.testing.assert_allclose(loaded_boxes, boxes, atol=1e-5)


def test_missing_label_is_empty(tmp_path: Path):
    classes, boxes = load_labels(tmp_path / "absent.txt")
    assert classes.shape == (0,)
    assert boxes.shape == (0, 4)


def test_bad_label_line_raises(tmp_path: Path):
    path = tmp_path / "bad.txt"
    path.write_text("0 0.5 0.5\n", encoding="utf-8")
    try:
        load_labels(path)
    except ValueError as exc:
        assert "5 fields" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_xywh_norm_to_xyxy():
    boxes = np.array([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    xyxy = xywh_norm_to_xyxy(boxes, width=100, height=100)
    assert xyxy[0].tolist() == [25, 25, 75, 75]
