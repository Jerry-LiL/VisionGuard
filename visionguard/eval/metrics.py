"""Precision, recall, mAP, and a confusion matrix for detection boxes.

Average precision is the area under the precision-recall curve with the
all-point interpolation used by Pascal VOC. A prediction only matches a
ground-truth box of the same class, and only inside the same image. The
confusion matrix is allowed to match across classes, so a scratch predicted
as a crack shows up off the diagonal instead of as a miss plus a false alarm.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GroundTruth:
    image_id: str
    class_id: int
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    image_id: str
    class_id: int
    score: float
    box: tuple[float, float, float, float]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def match_detections(
    predictions: list[Prediction],
    ground_truth: list[GroundTruth],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy one-to-one matching. Predictions must already be limited to one class.

    Returns two float arrays, ``tp`` and ``fp``, aligned with ``predictions``.
    The caller sorts predictions by score so the greediest match takes the
    most confident box first.
    """
    by_image: dict[str, list[int]] = defaultdict(list)
    for index, gt in enumerate(ground_truth):
        by_image[gt.image_id].append(index)

    matched: set[int] = set()
    tp = np.zeros(len(predictions), dtype=np.float64)
    fp = np.zeros(len(predictions), dtype=np.float64)
    for pred_index, pred in enumerate(predictions):
        best_iou = 0.0
        best_gt = None
        for gt_index in by_image.get(pred.image_id, []):
            if gt_index in matched:
                continue
            overlap = iou(pred.box, ground_truth[gt_index].box)
            if overlap > best_iou:
                best_iou = overlap
                best_gt = gt_index
        if best_gt is not None and best_iou >= iou_threshold:
            matched.add(best_gt)
            tp[pred_index] = 1.0
        else:
            fp[pred_index] = 1.0
    return tp, fp


def precision_recall_curve(
    tp: np.ndarray,
    fp: np.ndarray,
    num_ground_truth: int,
) -> tuple[np.ndarray, np.ndarray]:
    if num_ground_truth == 0 or len(tp) == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / num_ground_truth
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return precision, recall


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the precision-recall curve, with precision made monotonic."""
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.where(mrec[1:] != mrec[:-1])[0]
    area = np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1])
    return float(area)


def _sorted_class(
    predictions: list[Prediction],
    ground_truth: list[GroundTruth],
    class_id: int,
) -> tuple[list[Prediction], list[GroundTruth]]:
    preds = sorted(
        (pred for pred in predictions if pred.class_id == class_id),
        key=lambda pred: pred.score,
        reverse=True,
    )
    truths = [gt for gt in ground_truth if gt.class_id == class_id]
    return preds, truths


def confusion_matrix(
    predictions: list[Prediction],
    ground_truth: list[GroundTruth],
    num_classes: int,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Rows are ground truth, columns are predictions. The last index is background."""
    matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    background = num_classes
    images = {pred.image_id for pred in predictions} | {gt.image_id for gt in ground_truth}
    for image_id in images:
        image_preds = sorted(
            (pred for pred in predictions if pred.image_id == image_id),
            key=lambda pred: pred.score,
            reverse=True,
        )
        image_gts = [gt for gt in ground_truth if gt.image_id == image_id]
        used: set[int] = set()
        for pred in image_preds:
            best_iou = 0.0
            best_index = None
            for gt_index, gt in enumerate(image_gts):
                if gt_index in used:
                    continue
                overlap = iou(pred.box, gt.box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = gt_index
            if best_index is not None and best_iou >= iou_threshold:
                used.add(best_index)
                matrix[image_gts[best_index].class_id, pred.class_id] += 1
            else:
                matrix[background, pred.class_id] += 1
        for gt_index, gt in enumerate(image_gts):
            if gt_index not in used:
                matrix[gt.class_id, background] += 1
    return matrix


def summarize(
    predictions: list[Prediction],
    ground_truth: list[GroundTruth],
    class_names: tuple[str, ...] | list[str],
    iou_threshold: float = 0.5,
) -> dict:
    """mAP at 0.5, mAP over 0.5:0.95, operating-point precision/recall, and the matrix.

    Classes with no ground truth are left out of the mAP average. Precision and
    recall use every prediction the caller passed in (the confidence threshold
    is applied before this function) at ``iou_threshold``.
    """
    names = tuple(class_names)
    thresholds = [round(0.5 + 0.05 * step, 2) for step in range(10)]
    per_class = []
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0

    for class_id, name in enumerate(names):
        preds, truths = _sorted_class(predictions, ground_truth, class_id)
        n_gt = len(truths)
        if n_gt or preds:
            tp, fp = match_detections(preds, truths, iou_threshold)
        else:
            tp = np.zeros(0, dtype=np.float64)
            fp = np.zeros(0, dtype=np.float64)
        tp_sum = float(tp.sum()) if len(tp) else 0.0
        fp_sum = float(fp.sum()) if len(fp) else 0.0
        precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) else 0.0
        recall = tp_sum / n_gt if n_gt else 0.0
        curve_p, curve_r = precision_recall_curve(tp, fp, n_gt)
        ap50 = average_precision(curve_r, curve_p) if n_gt else None
        # False positives still count when a class has no ground truth in the split.
        if n_gt or preds:
            tp_total += tp_sum
            fp_total += fp_sum
            fn_total += n_gt - tp_sum
        per_class.append(
            {
                "class_id": class_id,
                "name": name,
                "precision": precision,
                "recall": recall,
                "ap50": ap50,
                "ground_truth": n_gt,
            }
        )

    map_by_threshold: dict[float, float] = {}
    for threshold in thresholds:
        aps = []
        for class_id, _name in enumerate(names):
            preds, truths = _sorted_class(predictions, ground_truth, class_id)
            if not truths:
                continue
            tp, fp = match_detections(preds, truths, threshold)
            curve_p, curve_r = precision_recall_curve(tp, fp, len(truths))
            aps.append(average_precision(curve_r, curve_p))
        map_by_threshold[threshold] = float(np.mean(aps)) if aps else 0.0

    matrix = confusion_matrix(predictions, ground_truth, len(names), iou_threshold)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    return {
        "map50": map_by_threshold[0.5],
        "map50_95": float(np.mean(list(map_by_threshold.values()))),
        "precision": precision,
        "recall": recall,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_axes": {"rows": "ground_truth", "columns": "predicted"},
        "labels": list(names) + ["background"],
    }


def format_summary(summary: dict) -> str:
    lines = [
        f"mAP@0.5      {summary['map50']:.4f}",
        f"mAP@0.5:0.95 {summary['map50_95']:.4f}",
        f"precision    {summary['precision']:.4f}",
        f"recall       {summary['recall']:.4f}",
    ]
    for row in summary["per_class"]:
        ap = "n/a" if row["ap50"] is None else f"{row['ap50']:.4f}"
        lines.append(
            f"  {row['name']:<12} P {row['precision']:.4f}  R {row['recall']:.4f}  AP50 {ap}"
        )
    labels = summary["labels"]
    header = " ".join(f"{label[:8]:>8}" for label in labels)
    lines.append("confusion matrix (rows ground truth, columns predicted)")
    lines.append(f"{'':>12} {header}")
    for label, row in zip(labels, summary["confusion_matrix"], strict=True):
        cells = " ".join(f"{int(value):8d}" for value in row)
        lines.append(f"{label[:12]:>12} {cells}")
    return "\n".join(lines)
