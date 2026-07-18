from __future__ import annotations

from pydantic import BaseModel, Field


class Shape(BaseModel):
    """One normalized bounding box (fractions of image w/h, 0..1)."""

    label: str
    x: float
    y: float
    w: float
    h: float


class AnnotationsSave(BaseModel):
    annotations: list[Shape] = Field(default_factory=list)
    mark_labeled: bool = True


class ImageOut(BaseModel):
    id: str
    filename: str
    url: str
    width: int
    height: int
    status: str
    box_count: int
    split: str


class ImageDetail(BaseModel):
    id: str
    filename: str
    url: str
    width: int
    height: int
    status: str
    split: str
    annotations: list[Shape]
    labels: list[dict]
    index: int
    total: int
    next_id: str | None
    prev_id: str | None


class ClassCount(BaseModel):
    name: str
    color: str
    count: int


class DatasetSummary(BaseModel):
    project_id: str
    name: str
    annotation_type: str
    mode: str
    total_images: int
    labeled: int
    unlabeled: int
    total_boxes: int
    classes: list[ClassCount]
    splits: dict[str, int]
