"""Export a fine-tuned YOLOv8 checkpoint to ONNX for CPU inference."""

from __future__ import annotations

import shutil
from pathlib import Path


def _yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "export needs Ultralytics: pip install -r requirements-train.txt"
        ) from exc
    return YOLO


def export_onnx(
    weights: str | Path,
    out: str | Path | None = None,
    imgsz: int = 640,
    opset: int = 12,
) -> str:
    """Write an ONNX graph that emits raw boxes and class scores.

    Non-maximum suppression is not baked into the graph. The service runs
    it in OpenCV, so the confidence and IoU thresholds can change at runtime.
    """
    if imgsz % 32 != 0:
        raise ValueError("imgsz must be a multiple of 32")
    yolo_cls = _yolo_class()
    model = yolo_cls(str(weights))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            dynamic=False,
            simplify=False,
            # None keeps the raw boxes and class scores. NMS runs in OpenCV.
            nms=None,
        )
    )
    if out is None:
        return str(exported)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != dest.resolve():
        shutil.copy2(exported, dest)
    return str(dest)
