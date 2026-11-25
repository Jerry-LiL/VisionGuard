import numpy as np
import pytest

from visionguard.eval.metrics import (
    GroundTruth,
    Prediction,
    average_precision,
    confusion_matrix,
    iou,
    summarize,
)
from visionguard.eval.report import render_confusion_matrix


def _gt(image: str, class_id: int, box: tuple[float, float, float, float]) -> GroundTruth:
    return GroundTruth(image, class_id, box)


def _pred(
    image: str,
    class_id: int,
    score: float,
    box: tuple[float, float, float, float],
) -> Prediction:
    return Prediction(image, class_id, score, box)


def test_iou_of_identical_and_shifted_boxes():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (0, 0, 10, 7)) == pytest.approx(0.7)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_perfect_detection_scores_one():
    summary = summarize(
        [_pred("a", 0, 0.9, (0, 0, 10, 10))],
        [_gt("a", 0, (0, 0, 10, 10))],
        ["scratch"],
    )
    assert summary["map50"] == pytest.approx(1.0)
    assert summary["map50_95"] == pytest.approx(1.0)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)


def test_a_miss_and_a_false_positive():
    # Two ground-truth scratches. The higher-scoring box hits one of them.
    # The other prediction is elsewhere, so recall stops at 0.5 and AP is 0.5.
    summary = summarize(
        [
            _pred("a", 0, 0.9, (0, 0, 10, 10)),
            _pred("a", 0, 0.8, (30, 30, 40, 40)),
        ],
        [
            _gt("a", 0, (0, 0, 10, 10)),
            _gt("a", 0, (50, 50, 60, 60)),
        ],
        ["scratch"],
    )
    assert summary["precision"] == pytest.approx(0.5)
    assert summary["recall"] == pytest.approx(0.5)
    assert summary["map50"] == pytest.approx(0.5)
    assert summary["per_class"][0]["ap50"] == pytest.approx(0.5)


def test_boxes_do_not_match_across_images():
    summary = summarize(
        [_pred("a", 0, 0.9, (0, 0, 10, 10))],
        [_gt("b", 0, (0, 0, 10, 10))],
        ["scratch"],
    )
    assert summary["precision"] == pytest.approx(0.0)
    assert summary["recall"] == pytest.approx(0.0)
    assert summary["map50"] == pytest.approx(0.0)


def test_partial_overlap_drops_at_stricter_iou():
    pred_box = (0, 0, 10, 7)
    gt_box = (0, 0, 10, 10)
    overlap = iou(pred_box, gt_box)
    summary = summarize(
        [_pred("a", 0, 0.9, pred_box)],
        [_gt("a", 0, gt_box)],
        ["scratch"],
    )
    thresholds = [round(0.5 + 0.05 * step, 2) for step in range(10)]
    expected = sum(overlap >= threshold for threshold in thresholds) / len(thresholds)
    assert overlap == pytest.approx(0.7)
    assert summary["map50"] == pytest.approx(1.0)
    assert summary["map50_95"] == pytest.approx(expected)
    assert summary["map50_95"] < 1.0


def test_confusion_matrix_records_a_class_swap_and_background():
    matrix = confusion_matrix(
        [
            _pred("a", 1, 0.9, (0, 0, 10, 10)),
            _pred("a", 2, 0.4, (40, 40, 50, 50)),
        ],
        [
            _gt("a", 0, (0, 0, 10, 10)),
            _gt("a", 0, (70, 70, 80, 80)),
        ],
        num_classes=3,
    )
    # Row 0 (scratch) predicted as crack, plus one unmatched scratch (background column).
    assert matrix[0, 1] == 1
    assert matrix[0, 3] == 1
    # Unmatched spot prediction lands in the background row.
    assert matrix[3, 2] == 1


def test_average_precision_of_an_empty_curve_is_zero():
    assert average_precision(np.zeros(0), np.zeros(0)) == 0.0


def test_confusion_matrix_image_changes_with_the_counts():
    labels = ["scratch", "background"]
    quiet = render_confusion_matrix(np.zeros((2, 2), dtype=np.int64), labels)
    busy = render_confusion_matrix(np.array([[3, 0], [0, 1]], dtype=np.int64), labels)
    assert quiet.shape[2] == 3
    assert quiet.dtype == np.uint8
    assert not np.array_equal(quiet, busy)
