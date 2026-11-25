"""Response bodies for the detection service."""

from pydantic import BaseModel, Field


class BoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionSchema(BaseModel):
    class_id: int
    class_name: str
    score: float
    box: BoxSchema


class DetectResponse(BaseModel):
    width: int
    height: int
    detections: list[DetectionSchema]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    img_size: int | None = None
    labels: list[str] = Field(default_factory=list)


class LabelsResponse(BaseModel):
    labels: list[str]
