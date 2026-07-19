"""Dataset versions and model-training jobs (self-serve).

A DatasetVersion is a frozen snapshot of the labeled set with train/valid/test
splits. A TrainingJob trains one architecture on one version. The actual GPU
work runs OUTSIDE our servers — typically the user's free Google Colab GPU —
which authenticates back to us with the job's ingest_token and streams
per-epoch metrics + final results. Nothing here is fabricated: metrics only
exist once a trainer reports them.
"""
from __future__ import annotations

import enum
import secrets
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.common import gen_uuid, utcnow


def gen_token() -> str:
    return secrets.token_urlsafe(24)


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    number: Mapped[int] = mapped_column(Integer, default=1)
    name: Mapped[str] = mapped_column(String(120), default="")
    image_count: Mapped[int] = mapped_column(Integer, default=0)
    train_count: Mapped[int] = mapped_column(Integer, default=0)
    valid_count: Mapped[int] = mapped_column(Integer, default=0)
    test_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrainingStatus(str, enum.Enum):
    awaiting_gpu = "awaiting_gpu"  # job created, waiting for Colab to connect
    running = "running"            # trainer is streaming epochs
    completed = "completed"        # final results stored
    failed = "failed"


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("dataset_versions.id"))
    engine: Mapped[str] = mapped_column(String(20), default="custom")
    architecture: Mapped[str] = mapped_column(String(30))   # yolov8 | yolo11 | rtdetr
    model_size: Mapped[str] = mapped_column(String(10))     # n | s | m …
    epochs_total: Mapped[int] = mapped_column(Integer, default=25)
    status: Mapped[str] = mapped_column(
        String(15), default=TrainingStatus.awaiting_gpu.value, index=True
    )
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    # [{epoch, train_loss, val_loss, map50, precision, recall}, …]
    metrics: Mapped[list] = mapped_column(JSON, default=list)
    # {map50, map50_95, precision, recall, f1, per_class:[{name,precision,recall,map50,count}],
    #  confusion_matrix:{labels:[...], matrix:[[...]]},
    #  confidence_curve:[{confidence, f1, precision, recall}], optimal_confidence}
    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    ingest_token: Mapped[str] = mapped_column(String(64), default=gen_token)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
