import numpy as np
import pytest

from visionguard.infer.postprocess import postprocess
from visionguard.infer.preprocess import LetterboxMeta

NAMES = ("scratch", "crack", "spot")
META = LetterboxMeta(ratio=1.0, pad_w=0.0, pad_h=0.0, size=64)


def _output(rows: list[list[float]]) -> np.ndarray:
    array = np.zeros((1, 7, len(rows)), dtype=np.float32)
    for index, row in enumerate(rows):
        array[0, :, index] = row
    return array


def test_keeps_the_confident_box_and_drops_the_weak_one():
    output = _output(
        [
            [32, 32, 20, 10, 0.90, 0.05, 0.02],
            [10, 10, 8, 8, 0.01, 0.02, 0.10],
        ]
    )
    found = postprocess(output, META, (64, 64), NAMES, conf_threshold=0.25, iou_threshold=0.45)
    assert len(found) == 1
    det = found[0]
    assert det.class_name == "scratch"
    assert det.score == pytest.approx(0.9)
    assert [det.x1, det.y1, det.x2, det.y2] == pytest.approx([22, 27, 42, 37])


def test_accepts_anchor_first_layout():
    channels_first = _output([[32, 32, 20, 10, 0.9, 0.0, 0.0]])
    anchor_first = np.transpose(channels_first, (0, 2, 1))
    found = postprocess(anchor_first, META, (64, 64), NAMES)
    assert len(found) == 1
    assert found[0].class_id == 0


def test_nms_suppresses_overlapping_boxes_of_the_same_class():
    output = _output(
        [
            [32, 32, 20, 20, 0.9, 0.0, 0.0],
            [34, 32, 20, 20, 0.6, 0.0, 0.0],
        ]
    )
    found = postprocess(output, META, (64, 64), NAMES, conf_threshold=0.25, iou_threshold=0.45)
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.9)


def test_different_classes_can_share_a_region():
    output = _output(
        [
            [32, 32, 20, 20, 0.9, 0.0, 0.0],
            [32, 32, 20, 20, 0.0, 0.0, 0.8],
        ]
    )
    found = postprocess(output, META, (64, 64), NAMES)
    assert sorted(det.class_name for det in found) == ["scratch", "spot"]


def test_results_are_sorted_by_score():
    output = _output(
        [
            [10, 10, 8, 8, 0.4, 0.0, 0.0],
            [40, 40, 8, 8, 0.95, 0.0, 0.0],
        ]
    )
    found = postprocess(output, META, (80, 80), NAMES, iou_threshold=0.1)
    assert [det.score for det in found] == pytest.approx([0.95, 0.4])
