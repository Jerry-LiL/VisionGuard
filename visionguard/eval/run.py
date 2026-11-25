"""Score a YOLO split with the ONNX detector and write metrics next to a confusion matrix."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from visionguard.data.yolo import find_images, load_labels, xywh_norm_to_xyxy
from visionguard.eval.metrics import GroundTruth, Prediction, format_summary, summarize
from visionguard.eval.report import write_confusion_matrix
from visionguard.infer.detector import OnnxDetector


def class_names_from_yaml(data_yaml: Path) -> tuple[str, ...]:
    payload = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        ordered = [names[key] for key in sorted(names, key=lambda item: int(item))]
    else:
        ordered = list(names)
    return tuple(ordered)


def _split_dirs(data_yaml: Path, split: str) -> tuple[Path, Path, tuple[str, ...]]:
    payload = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))
    root = Path(payload["path"])
    image_dir = root / payload[split]
    label_dir = root / "labels" / image_dir.name
    return image_dir, label_dir, class_names_from_yaml(data_yaml)


def collect_split(
    detector: OnnxDetector,
    data_yaml: Path,
    split: str = "val",
) -> tuple[list[Prediction], list[GroundTruth], tuple[str, ...]]:
    image_dir, label_dir, class_names = _split_dirs(data_yaml, split)
    if tuple(detector.class_names) != class_names:
        raise ValueError(
            f"detector labels {detector.class_names} do not match {data_yaml} names {class_names}"
        )
    predictions: list[Prediction] = []
    ground_truth: list[GroundTruth] = []
    images = find_images(image_dir)
    if not images:
        raise FileNotFoundError(f"no images in {image_dir}")
    for image_path in images:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"failed to read {image_path}")
        height, width = image.shape[:2]
        for det in detector.predict(image):
            predictions.append(
                Prediction(
                    image_id=image_path.stem,
                    class_id=det.class_id,
                    score=det.score,
                    box=(det.x1, det.y1, det.x2, det.y2),
                )
            )
        class_ids, boxes = load_labels(label_dir / f"{image_path.stem}.txt")
        for class_id, box in zip(class_ids, xywh_norm_to_xyxy(boxes, width, height), strict=True):
            ground_truth.append(
                GroundTruth(
                    image_id=image_path.stem,
                    class_id=int(class_id),
                    box=tuple(float(value) for value in box),
                )
            )
    return predictions, ground_truth, class_names, len(images)


def evaluate_dataset(
    detector: OnnxDetector,
    data_yaml: Path,
    split: str = "val",
    iou_threshold: float = 0.5,
    out_dir: Path | None = None,
) -> dict:
    predictions, ground_truth, class_names, image_count = collect_split(
        detector,
        data_yaml,
        split=split,
    )
    summary = summarize(predictions, ground_truth, class_names, iou_threshold=iou_threshold)
    summary["split"] = split
    summary["images"] = image_count
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_confusion_matrix(
            out_dir / "confusion_matrix.png",
            np.asarray(summary["confusion_matrix"], dtype=np.int64),
            summary["labels"],
        )
        (out_dir / "metrics.txt").write_text(format_summary(summary) + "\n", encoding="utf-8")
    return summary
