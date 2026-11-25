"""Draw synthetic inspection plates and write a YOLO dataset.

Scratches are thin dark lines, cracks are jagged polylines, spots are tinted
ellipses. The pictures exist so the pipeline can be trained without a factory
dataset. ``data.yaml`` is ordinary YOLO, and a real annotated set replaces
this directory with no other code changes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from visionguard.classes import DEFAULT_CLASS_NAMES
from visionguard.data.splits import split_pool
from visionguard.data.yolo import save_labels, xyxy_to_xywh_norm


def _plate(rng: np.random.Generator, size: int) -> np.ndarray:
    base = int(rng.integers(155, 205))
    image = np.full((size, size, 3), base, dtype=np.uint8)
    noise = rng.integers(-12, 13, size=(size, size, 1), dtype=np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (3, 3), (size - 4, size - 4), (110, 110, 110), 1)
    return image


def _region(rng: np.random.Generator, size: int) -> tuple[int, int, int, int]:
    side_w = int(rng.integers(max(12, size // 20), max(13, size // 4)))
    side_h = int(rng.integers(max(12, size // 20), max(13, size // 4)))
    x1 = int(rng.integers(0, size - side_w))
    y1 = int(rng.integers(0, size - side_h))
    return x1, y1, x1 + side_w, y1 + side_h


def _draw_scratch(
    image: np.ndarray,
    rng: np.random.Generator,
    box: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    shade = int(rng.integers(35, 80))
    cv2.line(image, (x1, y1), (x2, y2), (shade, shade, shade), 1, cv2.LINE_AA)
    if x2 - x1 > 16:
        offset = int(rng.integers(2, 5))
        cv2.line(
            image,
            (x1, min(y1 + offset, y2)),
            (x2, max(y2 - offset, y1)),
            (shade, shade, shade),
            1,
            cv2.LINE_AA,
        )


def _draw_crack(
    image: np.ndarray,
    rng: np.random.Generator,
    box: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    steps = 4
    points = []
    for step in range(steps + 1):
        x = int(x1 + (x2 - x1) * step / steps)
        y = int(rng.integers(y1, y2 + 1))
        points.append((x, y))
    cv2.polylines(image, [np.array(points, dtype=np.int32)], False, (25, 25, 25), 1, cv2.LINE_AA)


def _draw_spot(image: np.ndarray, rng: np.random.Generator, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
    colour = (
        int(rng.integers(30, 70)),
        int(rng.integers(50, 100)),
        int(rng.integers(80, 140)),
    )
    cv2.ellipse(image, center, axes, 0, 0, 360, colour, -1, cv2.LINE_AA)


_DRAWERS = (_draw_scratch, _draw_crack, _draw_spot)


def render_sample(
    rng: np.random.Generator,
    size: int = 640,
    forced_class: int | None = None,
    clean: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one BGR plate plus class ids and normalised xywh boxes."""
    image = _plate(rng, size)
    if clean:
        return image, np.zeros((0,), dtype=np.int64), np.zeros((0, 4), dtype=np.float32)

    if forced_class is None:
        chosen = [int(rng.integers(0, len(_DRAWERS))) for _ in range(int(rng.integers(1, 4)))]
    else:
        chosen = [int(forced_class)]

    classes: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    for class_id in chosen:
        rect = _region(rng, size)
        _DRAWERS[class_id](image, rng, rect)
        x1, y1, x2, y2 = rect
        boxes.append(xyxy_to_xywh_norm(x1, y1, x2, y2, size))
        classes.append(class_id)

    return (
        image,
        np.asarray(classes, dtype=np.int64),
        np.asarray(boxes, dtype=np.float32),
    )


def synthesize_dataset(
    out_dir: Path,
    count: int = 80,
    image_size: int = 640,
    val_ratio: float = 0.2,
    seed: int = 7,
    clean_ratio: float = 0.15,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> Path:
    """Write a synthetic pool, then a stratified train/val split. Returns data.yaml."""
    if count < 2:
        raise ValueError("count must be at least 2")
    if image_size < 64:
        raise ValueError("image_size must be at least 64")

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pool = out_dir / "_pool"
    image_dir = pool / "images"
    label_dir = pool / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    for index in range(count):
        forced = index if index < len(class_names) else None
        clean = False if forced is not None else bool(rng.random() < clean_ratio)
        image, classes, boxes = render_sample(rng, image_size, forced_class=forced, clean=clean)
        stem = f"img_{index:04d}"
        image_path = image_dir / f"{stem}.png"
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"failed to write {image_path}")
        save_labels(label_dir / f"{stem}.txt", classes, boxes)

    yaml_path = split_pool(
        pool,
        out_dir,
        val_ratio=val_ratio,
        seed=seed,
        class_names=class_names,
    )
    shutil.rmtree(pool)
    return yaml_path
