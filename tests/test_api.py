from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from tests.onnx_fixture import build_constant_detector
from visionguard.api.app import create_app
from visionguard.infer.detector import OnnxDetector


def _png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


def test_health_and_detect_without_a_model():
    with TestClient(create_app(load_env=False)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        body = health.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False
        assert body["labels"] == ["scratch", "crack", "spot"]
        refused = client.post(
            "/detect",
            files={"file": ("part.png", _png(np.zeros((8, 8, 3), np.uint8)), "image/png")},
        )
        assert refused.status_code == 503


def test_detect_returns_boxes_and_scores(tmp_path: Path):
    model = build_constant_detector(tmp_path / "fixture.onnx", img_size=64)
    detector = OnnxDetector(model, ("scratch", "crack", "spot"))
    image = np.full((64, 48, 3), 180, dtype=np.uint8)
    with TestClient(create_app(detector)) as client:
        health = client.get("/health")
        assert health.json()["model_loaded"] is True
        assert health.json()["img_size"] == 64
        response = client.post("/detect", files={"file": ("part.png", _png(image), "image/png")})
        assert response.status_code == 200
        body = response.json()
        assert body["width"] == 48
        assert body["height"] == 64
        assert len(body["detections"]) == 1
        det = body["detections"][0]
        assert det["class_name"] == "scratch"
        assert det["score"] == pytest_approx(0.9)
        assert set(det["box"]) == {"x1", "y1", "x2", "y2"}
        empty = client.post("/detect", files={"file": ("part.png", b"", "image/png")})
        assert empty.status_code == 400
        garbage = client.post("/detect", files={"file": ("part.png", b"not-an-image", "image/png")})
        assert garbage.status_code == 400


def pytest_approx(value: float) -> float:
    import pytest

    return pytest.approx(value)
