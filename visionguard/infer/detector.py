"""CPU inference with ONNX Runtime. OpenCV does the work on either side of the session."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from visionguard.infer.postprocess import Detection, postprocess
from visionguard.infer.preprocess import preprocess


def _fixed_square(shape: list) -> int | None:
    if len(shape) != 4:
        return None
    height, width = shape[2], shape[3]
    if isinstance(height, int) and isinstance(width, int) and height == width and height > 0:
        return height
    return None


class OnnxDetector:
    """Run a YOLOv8 ONNX graph on CPU and return boxes in the source image."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: tuple[str, ...] | list[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.class_names = tuple(class_names)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        fixed = _fixed_square(list(model_input.shape))
        # A static ONNX input wins. img_size is only for a dynamic export.
        if fixed is not None:
            self.img_size = fixed
        elif img_size is not None:
            self.img_size = int(img_size)
        else:
            self.img_size = 640

    def predict(self, image: np.ndarray) -> list[Detection]:
        tensor, meta = preprocess(image, self.img_size)
        outputs = self.session.run(None, {self.input_name: tensor})
        return postprocess(
            outputs[0],
            meta,
            orig_hw=image.shape[:2],
            class_names=self.class_names,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
