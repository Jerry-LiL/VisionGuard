"""OpenCV letterbox and colour conversion for a fixed-size YOLOv8 input."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Grey pad used by YOLOv8 during training.
_PAD = (114, 114, 114)


@dataclass(frozen=True)
class LetterboxMeta:
    """How a letterbox was built, so boxes can be mapped back to the original image."""

    ratio: float
    pad_w: float
    pad_h: float
    size: int


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize with a preserved aspect ratio and pad to a square of ``size``.

    The exported ONNX graph takes a fixed square, so this always returns
    exactly ``size`` by ``size``. Padding matches the trainer: half on each
    side, fill value 114.
    """
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w = int(round(width * ratio))
    new_h = int(round(height * ratio))
    pad_w = (size - new_w) / 2.0
    pad_h = (size - new_h) / 2.0

    resized = image
    if (width, height) != (new_w, new_h):
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=_PAD)
    if padded.shape[0] != size or padded.shape[1] != size:
        padded = cv2.resize(padded, (size, size), interpolation=cv2.INTER_LINEAR)
    return padded, LetterboxMeta(ratio=ratio, pad_w=pad_w, pad_h=pad_h, size=size)


def preprocess(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Return a float32 NCHW batch in RGB, pixels scaled to 0..1, plus letterbox metadata."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR image with shape (H, W, 3)")
    padded, meta = letterbox(image, size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    return np.ascontiguousarray(tensor), meta


def scale_boxes_to_original(
    boxes_xyxy: np.ndarray,
    meta: LetterboxMeta,
    orig_hw: tuple[int, int],
) -> np.ndarray:
    """Undo letterbox padding and scale xyxy boxes into the original image."""
    orig_h, orig_w = orig_hw
    out = np.array(boxes_xyxy, dtype=np.float32, copy=True)
    if out.size == 0:
        return out.reshape(0, 4)
    out[:, [0, 2]] -= meta.pad_w
    out[:, [1, 3]] -= meta.pad_h
    out[:, :4] /= meta.ratio
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, orig_w)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, orig_h)
    return out
