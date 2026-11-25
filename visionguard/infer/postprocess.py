"""Decode a YOLOv8 ONNX output and run OpenCV NMS.

A standard YOLOv8 export emits ``(1, 4 + num_classes, anchors)``: cx, cy, w, h
in the letterboxed input, plus one score per class. The sigmoid lives in the
graph, so scores are already in 0..1. NMS stays here, which lets the service
change the confidence threshold without exporting again.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionguard.infer.preprocess import LetterboxMeta, scale_boxes_to_original


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def _prediction_rows(output: np.ndarray, num_classes: int) -> np.ndarray:
    pred = np.asarray(output, dtype=np.float32)
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(f"expected batch size 1, got shape {pred.shape}")
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"expected a 2D prediction tensor, got shape {output.shape}")
    channels = 4 + num_classes
    if pred.shape[0] == channels:
        pred = pred.T
    if pred.shape[1] != channels:
        raise ValueError(
            f"expected {channels} channels for {num_classes} classes, "
            f"got shape {tuple(output.shape)}"
        )
    return pred


def _nms_indices(boxes_xywh: np.ndarray, scores: np.ndarray, conf: float, iou: float) -> list[int]:
    if len(boxes_xywh) == 0:
        return []
    picked = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.astype(float).tolist(), conf, iou)
    if picked is None or len(picked) == 0:
        return []
    return [int(index) for index in np.asarray(picked).reshape(-1)]


def postprocess(
    output: np.ndarray,
    meta: LetterboxMeta,
    orig_hw: tuple[int, int],
    class_names: tuple[str, ...] | list[str],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Turn raw YOLOv8 rows into detections in the original image's pixel space.

    NMS is per class, which is the YOLOv8 default: two different defects may
    occupy the same region and both survive.
    """
    names = tuple(class_names)
    rows = _prediction_rows(output, len(names))
    boxes_xywh = rows[:, :4]
    class_scores = rows[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(rows)), class_ids]

    keep = (scores >= conf_threshold) & (boxes_xywh[:, 2] > 0) & (boxes_xywh[:, 3] > 0)
    boxes_xywh = boxes_xywh[keep]
    class_ids = class_ids[keep]
    scores = scores[keep]
    if len(boxes_xywh) == 0:
        return []

    xyxy = np.empty_like(boxes_xywh)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

    # cv2.dnn.NMSBoxes wants corner-format xywh, not centre-format.
    corner = xyxy.copy()
    corner[:, 2] = boxes_xywh[:, 2]
    corner[:, 3] = boxes_xywh[:, 3]

    chosen: list[int] = []
    for class_id in np.unique(class_ids):
        indexes = np.flatnonzero(class_ids == class_id)
        local = _nms_indices(corner[indexes], scores[indexes], conf_threshold, iou_threshold)
        chosen.extend(int(indexes[item]) for item in local)

    if not chosen:
        return []

    chosen_arr = np.asarray(chosen, dtype=np.int64)
    order = chosen_arr[np.argsort(-scores[chosen_arr])]
    scaled = scale_boxes_to_original(xyxy[order], meta, orig_hw)
    detections: list[Detection] = []
    orig_h, orig_w = orig_hw
    for box, index in zip(scaled, order, strict=True):
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(orig_w), x2), min(float(orig_h), y2)
        class_id = int(class_ids[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=names[class_id],
                score=float(scores[index]),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections
