"""FastAPI service that returns defect boxes and scores from an ONNX model."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from visionguard import __version__
from visionguard.api.schemas import (
    BoxSchema,
    DetectionSchema,
    DetectResponse,
    HealthResponse,
    LabelsResponse,
)
from visionguard.classes import parse_class_names
from visionguard.infer.detector import OnnxDetector
from visionguard.settings import env_float, load_dotenv


def detector_from_env() -> OnnxDetector | None:
    """Build a detector when MODEL_PATH points at a file. Otherwise return None."""
    path = os.environ.get("MODEL_PATH", "models/visionguard.onnx")
    if not path or not os.path.isfile(path):
        return None
    names = parse_class_names(os.environ.get("CLASS_NAMES"))
    img_size = os.environ.get("IMG_SIZE")
    return OnnxDetector(
        path,
        names,
        conf_threshold=env_float("CONF_THRESHOLD", 0.25),
        iou_threshold=env_float("IOU_THRESHOLD", 0.45),
        img_size=int(img_size) if img_size else None,
    )


def create_app(detector: OnnxDetector | None = None, *, load_env: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if detector is not None:
            app.state.detector = detector
        elif load_env:
            load_dotenv()
            app.state.detector = detector_from_env()
        else:
            app.state.detector = None
        yield

    app = FastAPI(title="VisionGuard", version=__version__, lifespan=lifespan)

    def labels() -> list[str]:
        current = app.state.detector
        if current is not None:
            return list(current.class_names)
        return list(parse_class_names(os.environ.get("CLASS_NAMES")))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current = app.state.detector
        return HealthResponse(
            status="ok",
            model_loaded=current is not None,
            img_size=None if current is None else current.img_size,
            labels=labels(),
        )

    @app.get("/labels", response_model=LabelsResponse)
    def label_list() -> LabelsResponse:
        return LabelsResponse(labels=labels())

    @app.post("/detect", response_model=DetectResponse)
    async def detect(file: UploadFile = File(...)) -> DetectResponse:
        current = app.state.detector
        if current is None:
            raise HTTPException(status_code=503, detail="model is not loaded")
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="empty file")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="could not decode image")
        height, width = image.shape[:2]
        detections = [
            DetectionSchema(
                class_id=item.class_id,
                class_name=item.class_name,
                score=item.score,
                box=BoxSchema(x1=item.x1, y1=item.y1, x2=item.x2, y2=item.y2),
            )
            for item in current.predict(image)
        ]
        return DetectResponse(width=width, height=height, detections=detections)

    return app


app = create_app()
