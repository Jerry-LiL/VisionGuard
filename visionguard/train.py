"""Fine-tune a PyTorch YOLOv8 detector on a YOLO-format defect dataset."""

from __future__ import annotations

import os
from pathlib import Path

from visionguard.data.augment import augmentation_hyperparams


def _yolo_class():
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "training needs PyTorch and Ultralytics: pip install -r requirements-train.txt"
        ) from exc
    return YOLO


def train(
    data: str | Path,
    model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "cpu",
    project: str = "runs/detect",
    name: str = "visionguard",
    workers: int = 0,
    seed: int = 7,
) -> str:
    """Fine-tune YOLOv8 and return the path to ``best.pt``.

    Training starts from the pretrained PyTorch checkpoint (``yolov8n.pt``
    unless another weights file is passed). The ``val`` split in ``data`` is
    held out: the trainer reports precision, recall, mAP, and a confusion
    matrix on it. Mosaic, horizontal flip, and HSV colour jitter are the
    batch augmentations. The run lasts the requested number of epochs.
    """
    if imgsz % 32 != 0:
        raise ValueError("imgsz must be a multiple of 32")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch < 1:
        raise ValueError("batch must be at least 1")

    aug = augmentation_hyperparams(epochs)
    yolo_cls = _yolo_class()
    detector = yolo_cls(model)
    detector.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        val=True,
        plots=True,
        seed=seed,
        workers=workers,
        patience=epochs,
        **aug,
    )
    best = Path(project) / name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"training finished but {best} was not written")
    return str(best)
