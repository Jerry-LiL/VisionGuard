"""Held-out train/val split for a pool of YOLO images.

The split runs before any augmentation. A mosaic is built only from the
training list, so a validation image cannot leak into a training batch.
"""

from __future__ import annotations

import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from visionguard.classes import DEFAULT_CLASS_NAMES
from visionguard.data.yolo import find_images, load_labels


def primary_class(class_ids: np.ndarray) -> int:
    """Majority label on an image. Empty images use -1. Ties take the smaller id."""
    if len(class_ids) == 0:
        return -1
    values, counts = np.unique(class_ids, return_counts=True)
    top = np.flatnonzero(counts == counts.max())
    return int(values[top].min())


def assign_splits(
    items: list[tuple[str, int]],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Stratify by primary class and return (train_stems, val_stems).

    A class with a single image stays in train, so the hold-out does not
    become the only example of that defect. Every other class contributes
    at least one validation image and keeps at least one training image.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if len(items) < 2:
        raise ValueError("need at least 2 images to hold out a validation split")

    groups: dict[int, list[str]] = defaultdict(list)
    for stem, class_id in items:
        groups[class_id].append(stem)

    rng = random.Random(seed)
    train: list[str] = []
    val: list[str] = []
    for class_id in sorted(groups):
        stems = groups[class_id][:]
        rng.shuffle(stems)
        if len(stems) == 1:
            train.extend(stems)
            continue
        n_val = int(round(len(stems) * val_ratio))
        n_val = min(max(n_val, 1), len(stems) - 1)
        val.extend(stems[:n_val])
        train.extend(stems[n_val:])
    return train, val


def split_pool(
    source: Path,
    out_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 7,
    class_names: tuple[str, ...] = DEFAULT_CLASS_NAMES,
) -> Path:
    """Copy ``source/{images,labels}`` into a YOLO tree and write ``data.yaml``.

    ``source/images`` holds the pictures. ``source/labels`` holds matching
    ``.txt`` files; a missing label file is treated as an image with no defects.
    """
    image_dir = source / "images"
    label_dir = source / "labels"
    if image_dir.resolve() == (out_dir / "images").resolve():
        raise ValueError("source and output image directories must differ")
    for split_name in ("train", "val"):
        for kind in ("images", "labels"):
            dest = out_dir / kind / split_name
            if dest.exists():
                shutil.rmtree(dest)
    images = find_images(image_dir)
    if len(images) < 2:
        raise ValueError(f"need at least 2 images in {image_dir}")

    items: list[tuple[str, int]] = []
    for image_path in images:
        class_ids, _ = load_labels(label_dir / f"{image_path.stem}.txt")
        items.append((image_path.stem, primary_class(class_ids)))

    train_stems, val_stems = assign_splits(items, val_ratio=val_ratio, seed=seed)
    by_stem = {path.stem: path for path in images}

    for split_name, stems in ("train", train_stems), ("val", val_stems):
        dest_images = out_dir / "images" / split_name
        dest_labels = out_dir / "labels" / split_name
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            image_path = by_stem[stem]
            shutil.copy2(image_path, dest_images / image_path.name)
            label_path = label_dir / f"{stem}.txt"
            target = dest_labels / f"{stem}.txt"
            if label_path.is_file():
                shutil.copy2(label_path, target)
            else:
                target.write_text("", encoding="utf-8")

    yaml_path = out_dir / "data.yaml"
    payload = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": list(class_names),
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path
