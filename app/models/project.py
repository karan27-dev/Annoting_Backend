from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.common import gen_uuid, utcnow


class AnnotationType(str, enum.Enum):
    bbox = "bbox"
    polygon = "polygon"
    segmentation = "segmentation"
    keypoint = "keypoint"
    classification = "classification"


class ProjectStatus(str, enum.Enum):
    pending_setup = "pending_setup"
    active = "active"
    paused = "paused"
    review = "review"
    delivered = "delivered"
    archived = "archived"


class ProjectMode(str, enum.Enum):
    """How the data gets labeled.

    self_serve — the owner labels their own images in the in-app editor and
                 builds their dataset directly (Roboflow-style).
    managed    — Annoting's annotators label it as a paid service (the quote →
                 accept → annotate → review → deliver flow).
    """

    self_serve = "self_serve"
    managed = "managed"


class MediaType(str, enum.Enum):
    images = "images"
    videos = "videos"
    mixed = "mixed"


class DataSource(str, enum.Enum):
    upload = "upload"
    gdrive = "gdrive"


class DeliveryFormat(str, enum.Enum):
    """Formats the client can receive their labeled data in. Each maps to a
    CVAT export format string in services/export_formats.py."""

    coco = "coco"
    yolo = "yolo"
    voc = "voc"
    cvat_xml = "cvat_xml"
    datumaro = "datumaro"


class IntakeStatus(str, enum.Enum):
    """Where the client's data sits in the intake pipeline, before annotation."""

    awaiting_data = "awaiting_data"     # project created, nothing received yet
    counting = "counting"               # archive/link received, inspection running
    counted = "counted"                 # media counted, complexity estimated
    pending_review = "pending_review"   # draft quote awaits Annoting admin review
    quoted = "quoted"                   # admin published the quote to the client
    quote_accepted = "quote_accepted"   # client accepted — ready for CVAT ingestion


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[str] = mapped_column(String(12), default=ProjectMode.managed.value)
    annotation_type: Mapped[str] = mapped_column(String(20))
    label_taxonomy: Mapped[list] = mapped_column(JSON, default=list)
    total_images: Mapped[int] = mapped_column(Integer, default=0)
    images_completed: Mapped[int] = mapped_column(Integer, default=0)
    quality_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    quality_target: Mapped[float] = mapped_column(Numeric(5, 2), default=0.85)
    turnaround_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=ProjectStatus.pending_setup.value
    )
    r2_dataset_prefix: Mapped[str | None] = mapped_column(String(500), nullable=True)
    r2_deliverable_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # ── Intake pipeline ────────────────────────────────────────────────────────
    media_type: Mapped[str] = mapped_column(String(10), default=MediaType.images.value)
    data_source: Mapped[str] = mapped_column(String(10), default=DataSource.upload.value)
    gdrive_link: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    video_count: Mapped[int] = mapped_column(Integer, default=0)
    delivery_format: Mapped[str] = mapped_column(
        String(20), default=DeliveryFormat.coco.value
    )
    estimated_objects_per_image: Mapped[float | None] = mapped_column(
        Numeric(6, 2), nullable=True
    )
    complexity_tier: Mapped[str | None] = mapped_column(String(10), nullable=True)
    intake_status: Mapped[str] = mapped_column(
        String(20), default=IntakeStatus.awaiting_data.value
    )
    intake_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    delivered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class CvatMapping(Base):
    """Maps a platform project to its CVAT entities."""

    __tablename__ = "cvat_mappings"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    cvat_project_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cvat_task_ids: Mapped[list | None] = mapped_column(JSON, nullable=True)
    gt_job_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # 0 = don't split: one CVAT job holds the entire dataset (full container).
    segment_size: Mapped[int] = mapped_column(Integer, default=0)
    export_format: Mapped[str] = mapped_column(String(50), default="COCO 1.0")
    last_synced_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
