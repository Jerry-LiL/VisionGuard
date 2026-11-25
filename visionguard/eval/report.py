"""Draw a confusion matrix with OpenCV so evaluation does not need another plotting library."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def render_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> np.ndarray:
    """Rows are ground truth, columns are predictions, including a background row and column."""
    labels = list(class_names)
    count = len(labels)
    if matrix.shape != (count, count):
        raise ValueError(f"matrix shape {matrix.shape} does not match {count} labels")

    cell = 72
    gutter = 130
    title = 36
    canvas_size = gutter + cell * count
    canvas = np.full((title + canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    peak = max(int(matrix.max()), 1)
    cv2.putText(
        canvas,
        "rows: ground truth, cols: predicted",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    for index, label in enumerate(labels):
        cv2.putText(
            canvas,
            label[:12],
            (4, title + gutter + index * cell + cell // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label[:10],
            (gutter + index * cell + 4, title + gutter - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    for row in range(count):
        for col in range(count):
            value = int(matrix[row, col])
            shade = int(255 - 180 * (value / peak))
            if row == col and row != count - 1:
                colour = (shade, shade, 255)
            else:
                colour = (shade, shade, shade)
            x0 = gutter + col * cell
            y0 = title + gutter + row * cell
            cv2.rectangle(canvas, (x0, y0), (x0 + cell - 4, y0 + cell - 4), colour, thickness=-1)
            cv2.putText(
                canvas,
                str(value),
                (x0 + 8, y0 + cell // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return canvas


def write_confusion_matrix(path: Path, matrix: np.ndarray, class_names: list[str]) -> None:
    image = render_confusion_matrix(matrix, class_names)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write {path}")
