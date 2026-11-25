from pathlib import Path

import numpy as np
import pytest

from tests.onnx_fixture import build_constant_detector
from visionguard.infer.detector import OnnxDetector

NAMES = ("scratch", "crack", "spot")


def test_detector_reads_the_static_input_size_and_decodes_a_box(tmp_path: Path):
    model = build_constant_detector(tmp_path / "fixture.onnx", img_size=64)
    detector = OnnxDetector(model, NAMES, img_size=32)
    assert detector.img_size == 64
    assert "CPUExecutionProvider" in detector.session.get_providers()

    square = np.zeros((64, 64, 3), dtype=np.uint8)
    found = detector.predict(square)
    assert len(found) == 1
    assert found[0].class_name == "scratch"
    assert found[0].score == pytest.approx(0.9, abs=1e-5)
    assert [found[0].x1, found[0].y1, found[0].x2, found[0].y2] == pytest.approx([22, 27, 42, 37])

    wide = np.zeros((32, 64, 3), dtype=np.uint8)
    shifted = detector.predict(wide)
    assert [shifted[0].x1, shifted[0].y1, shifted[0].x2, shifted[0].y2] == pytest.approx(
        [22, 11, 42, 21]
    )
