"""A tiny ONNX graph with a fixed YOLOv8-style output, for inference tests."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build_constant_detector(path: Path, img_size: int = 64) -> Path:
    """Two anchors: one scratch above the confidence cutoff, one spot below it.

    The graph adds ``input * 0`` to a constant so the input is used and the
    output does not depend on the picture.
    """
    const = np.zeros((1, 7, 2), dtype=np.float32)
    const[0, :, 0] = (32, 32, 20, 10, 0.90, 0.05, 0.02)
    const[0, :, 1] = (10, 10, 8, 8, 0.01, 0.02, 0.10)

    images = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, img_size, img_size]
    )
    output = helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 7, 2])
    initializers = [
        numpy_helper.from_array(const, name="const"),
        numpy_helper.from_array(np.array(0.0, dtype=np.float32), name="zero"),
    ]
    nodes = [
        helper.make_node("ReduceSum", ["images"], ["reduced"], keepdims=0),
        helper.make_node("Mul", ["reduced", "zero"], ["nothing"]),
        helper.make_node("Add", ["const", "nothing"], ["output0"]),
    ]
    graph = helper.make_graph(nodes, "fixture", [images], [output], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)
    return path
