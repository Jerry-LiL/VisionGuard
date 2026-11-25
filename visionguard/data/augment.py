"""Mosaic, horizontal flip, and HSV colour jitter.

The trainer applies the same three operations online. These functions are the
geometry used by the preview command and by the tests, and
``augmentation_hyperparams`` is what ``train`` forwards to YOLOv8.
"""

from __future__ import annotations

import numpy as np

# 2x2 tile origins in normalised (x, y): top-left, top-right, bottom-left, bottom-right.
_QUADRANTS = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))


def augmentation_hyperparams(epochs: int) -> dict[str, float | int]:
    """Augmentation settings passed straight through to the YOLOv8 trainer.

    Mosaic pastes four training images onto one canvas so objects show up
    smaller and in a new context. ``fliplr`` mirrors a sample left to right.
    ``hsv_h``, ``hsv_s`` and ``hsv_v`` are colour jitter: hue, saturation
    and value move, boxes do not. Parts sit in a fixture, so vertical flip
    and the extra geometric knobs stay off.

    On a long run the last 10 epochs drop mosaic so the final weights see
    normal-scale objects. Shorter runs keep mosaic on the whole way through,
    because turning it off immediately would drop it entirely.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    return {
        "mosaic": 1.0,
        "close_mosaic": 10 if epochs > 20 else 0,
        "fliplr": 0.5,
        "flipud": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "bgr": 0.0,
        "erasing": 0.0,
    }


def horizontal_flip(image: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror an image and its normalised xywh boxes left to right."""
    import cv2

    flipped = cv2.flip(image, 1)
    if np.asarray(boxes).size == 0:
        return flipped, np.zeros((0, 4), dtype=np.float32)
    out = np.array(boxes, dtype=np.float32, copy=True).reshape(-1, 4)
    out[:, 0] = 1.0 - out[:, 0]
    return flipped, out


def colour_jitter(
    image: np.ndarray,
    hsv_h: float = 0.015,
    hsv_s: float = 0.7,
    hsv_v: float = 0.4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Jitter hue, saturation, and value. Boxes are unchanged, so they are not an argument.

    Gains follow the YOLOv8 ranges: each channel is multiplied by
    ``1 + uniform(-1, 1) * gain``. OpenCV hue is stored in 0..179.
    """
    import cv2

    rng = rng or np.random.default_rng()
    gains = rng.uniform(-1.0, 1.0, size=3) * np.array([hsv_h, hsv_s, hsv_v], dtype=np.float32) + 1.0
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    ladder = np.arange(256, dtype=np.float32)
    lut_h = np.mod(ladder * gains[0], 180).astype(np.uint8)
    lut_s = np.clip(ladder * gains[1], 0, 255).astype(np.uint8)
    lut_v = np.clip(ladder * gains[2], 0, 255).astype(np.uint8)
    jittered = cv2.merge((cv2.LUT(hue, lut_h), cv2.LUT(sat, lut_s), cv2.LUT(val, lut_v)))
    return cv2.cvtColor(jittered, cv2.COLOR_HSV2BGR)


def mosaic(
    images: list[np.ndarray],
    boxes: list[np.ndarray],
    class_ids: list[np.ndarray],
    out_size: int = 640,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tile four images into one square and rescale their boxes to match.

    Each source is resized into a quadrant (top-left, top-right, bottom-left,
    bottom-right). A normalised box keeps its shape relative to its own image,
    then shifts by half a frame into the mosaic. The trainer's online mosaic
    uses a random centre instead of a fixed grid; the box math is the same idea.
    """
    import cv2

    if len(images) != 4 or len(boxes) != 4 or len(class_ids) != 4:
        raise ValueError("mosaic expects exactly 4 images, each with boxes and class ids")
    if out_size < 2 or out_size % 2 != 0:
        raise ValueError("out_size must be a positive even number")

    tile = out_size // 2
    canvas = np.full((out_size, out_size, 3), 114, dtype=np.uint8)
    kept_boxes: list[np.ndarray] = []
    kept_classes: list[np.ndarray] = []

    for image, raw_boxes, raw_classes, (origin_x, origin_y) in zip(
        images, boxes, class_ids, _QUADRANTS, strict=True
    ):
        resized = cv2.resize(image, (tile, tile), interpolation=cv2.INTER_LINEAR)
        x0 = int(origin_x * out_size)
        y0 = int(origin_y * out_size)
        canvas[y0 : y0 + tile, x0 : x0 + tile] = resized

        if np.asarray(raw_boxes).size == 0:
            continue
        src = np.asarray(raw_boxes, dtype=np.float32).reshape(-1, 4)
        classes = np.asarray(raw_classes, dtype=np.int64).reshape(-1)
        if len(src) != len(classes):
            raise ValueError("boxes and class ids differ in length")
        adjusted = np.empty_like(src)
        adjusted[:, 0] = src[:, 0] * 0.5 + origin_x
        adjusted[:, 1] = src[:, 1] * 0.5 + origin_y
        adjusted[:, 2] = src[:, 2] * 0.5
        adjusted[:, 3] = src[:, 3] * 0.5
        kept_boxes.append(adjusted)
        kept_classes.append(classes)

    if not kept_boxes:
        return canvas, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return canvas, np.concatenate(kept_boxes, axis=0), np.concatenate(kept_classes, axis=0)
