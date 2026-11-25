"""Write a few augmented training samples so labels can be checked by eye."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from visionguard.data.augment import colour_jitter, horizontal_flip, mosaic
from visionguard.data.yolo import find_images, load_labels, save_labels


def write_preview(data_dir: Path, out_dir: Path, seed: int = 7, out_size: int = 640) -> Path:
    """Apply mosaic, flip, and colour jitter to the training split and save the pictures."""
    train_images = find_images(Path(data_dir) / "images" / "train")
    if not train_images:
        raise FileNotFoundError(f"no training images under {data_dir}")

    label_dir = Path(data_dir) / "labels" / "train"
    rng = np.random.default_rng(seed)
    picked_idx = rng.choice(len(train_images), size=4, replace=len(train_images) < 4)
    picked = [train_images[int(index)] for index in picked_idx]

    images: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    classes: list[np.ndarray] = []
    for path in picked:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"failed to read {path}")
        class_ids, box = load_labels(label_dir / f"{path.stem}.txt")
        images.append(image)
        boxes.append(box)
        classes.append(class_ids)

    mosaic_image, mosaic_boxes, mosaic_classes = mosaic(images, boxes, classes, out_size=out_size)
    flipped, flipped_boxes = horizontal_flip(images[0], boxes[0])
    jittered = colour_jitter(images[0], rng=rng)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = (
        ("mosaic.png", mosaic_image),
        ("flip.png", flipped),
        ("colour_jitter.png", jittered),
    )
    for name, image in written:
        target = out_dir / name
        if not cv2.imwrite(str(target), image):
            raise OSError(f"failed to write {target}")
    save_labels(out_dir / "mosaic.txt", mosaic_classes, mosaic_boxes)
    save_labels(out_dir / "flip.txt", classes[0], flipped_boxes)
    return out_dir
