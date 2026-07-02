from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.project import (
    AnnotationType,
    DataSource,
    DeliveryFormat,
    MediaType,
)


class LabelClass(BaseModel):
    name: str
    color: str = "#e2553d"
    attributes: list[str] = Field(default_factory=list)


class ProjectCreate(BaseModel):
    name: str
    annotation_type: AnnotationType
    label_taxonomy: list[LabelClass] = Field(default_factory=list)
    description: str | None = None
    total_images: int = 0
    turnaround_days: int | None = 14
    media_type: MediaType = MediaType.images
    data_source: DataSource = DataSource.upload
    delivery_format: DeliveryFormat = DeliveryFormat.coco


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str | None
    annotation_type: str
    status: str
    total_images: int
    images_completed: int
    quality_score: float | None
    quality_target: float
    turnaround_days: int | None
    created_at: datetime
    delivered_at: datetime | None
    media_type: str = "images"
    data_source: str = "upload"
    gdrive_link: str | None = None
    image_count: int = 0
    video_count: int = 0
    delivery_format: str = "coco"
    estimated_objects_per_image: float | None = None
    complexity_tier: str | None = None
    intake_status: str = "awaiting_data"
    intake_detail: str | None = None

    class Config:
        from_attributes = True


class ProgressOut(BaseModel):
    images_done: int
    images_total: int
    percent: float
    velocity_per_day: float
    eta_days: int | None


class StatusUpdate(BaseModel):
    status: str


class CvatSetupRequest(BaseModel):
    # 0 = full container: one job with every frame, no splitting.
    segment_size: int = 0
    gt_frame_count: int = 10
    honeypot_mode: bool = True


class PresignedUrlRequest(BaseModel):
    project_id: str
    filename: str
    file_size_bytes: int
    content_type: str = "application/zip"


class PresignedUrlResponse(BaseModel):
    upload_url: str
    r2_key: str


class UploadConfirmRequest(BaseModel):
    project_id: str
    r2_key: str
    file_count: int


class GdriveLinkRequest(BaseModel):
    project_id: str
    link: str


class QuotePublishRequest(BaseModel):
    """Admin's final numbers after reviewing the counted dataset."""

    avg_objects_per_image: float | None = None  # None = keep auto-estimate
    rate_per_label_inr: float | None = None     # None = standard rate card
    notes: str | None = None


class QuoteSummary(BaseModel):
    id: str
    rate_per_label_inr: float
    estimated_labels: int
    quoted_total_inr: float
    turnaround_premium_pct: float
    volume_discount_pct: float
    accepted_at: datetime | None

    class Config:
        from_attributes = True


class IntakeOut(BaseModel):
    """Live intake-pipeline status for a project."""

    intake_status: str
    intake_detail: str | None
    media_type: str
    image_count: int
    video_count: int
    total_files: int
    data_source: str
    gdrive_link: str | None
    estimated_objects_per_image: float | None
    complexity_tier: str | None
    delivery_format: str
    quote: QuoteSummary | None = None
