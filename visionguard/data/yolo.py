"""Read and write YOLO detection labels (class cx cy w h, all normalised)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def save_labels(path: Path, class_ids: np.ndarray, boxes: np.ndarray) -> None:
    class_ids = np.asarray(class_ids, dtype=np.int64).reshape(-1)
    boxes = (
        np.zeros((0, 4), dtype=np.float32)
        if np.asarray(boxes).size == 0
        else np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    )
    if len(class_ids) != len(boxes):
        raise ValueError(f"{path}: {len(class_ids)} classes and {len(boxes)} boxes")
    lines = [
        f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
        for cls, (x, y, w, h) in zip(class_ids, boxes, strict=True)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    text = ("\n".join(lines) + "\n") if lines else ""
    path.write_text(text, encoding="utf-8")


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file() or path.stat().st_size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)
    classes: list[int] = []
    boxes: list[list[float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, got {raw!r}")
        classes.append(int(parts[0]))
        boxes.append([float(value) for value in parts[1:]])
    return np.asarray(classes, dtype=np.int64), np.asarray(boxes, dtype=np.float32)


def xywh_norm_to_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert normalised xywh boxes to pixel xyxy for an image of the given size."""
    if np.asarray(boxes).size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    src = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty_like(src)
    out[:, 0] = (src[:, 0] - src[:, 2] / 2.0) * width
    out[:, 1] = (src[:, 1] - src[:, 3] / 2.0) * height
    out[:, 2] = (src[:, 0] + src[:, 2] / 2.0) * width
    out[:, 3] = (src[:, 1] + src[:, 3] / 2.0) * height
    return out


def xyxy_to_xywh_norm(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    size: int,
) -> tuple[float, float, float, float]:
    width = (x2 - x1) / size
    height = (y2 - y1) / size
    cx = (x1 + x2) / 2.0 / size
    cy = (y1 + y2) / 2.0 / size
    return cx, cy, width, height


def find_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
