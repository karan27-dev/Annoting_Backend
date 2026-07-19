from __future__ import annotations

from pydantic import BaseModel, Field


class Shape(BaseModel):
    """One annotation, all coords normalized to fractions of image w/h (0..1).

    type "bbox"           — x/y/w/h is the box.
    type "polygon"        — points is the vertex list; x/y/w/h is the enclosing
                            box (kept in sync so bbox consumers keep working).
    type "classification" — whole-image label; x/y/w/h is 0,0,1,1.
    """

    label: str
    x: float = 0.0
    y: float = 0.0
    w: float = 1.0
    h: float = 1.0
    type: str = "bbox"
    points: list[list[float]] | None = None


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
    annotation_type: str = "bbox"
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
