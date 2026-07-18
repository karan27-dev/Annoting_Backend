"""Self-serve dataset storage.

Each image the user uploads to a self-serve project becomes a DatasetImage row.
Annotations are stored inline as JSON — a list of normalized shapes:
    {"label": "car", "x": 0.12, "y": 0.30, "w": 0.20, "h": 0.15}
coordinates are 0..1 fractions of width/height so they're resolution-independent.
"""
from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.common import gen_uuid, utcnow


class ImageStatus(str, enum.Enum):
    unlabeled = "unlabeled"
    labeled = "labeled"
    review = "review"       # self-serve owner marked done, optional QA
    approved = "approved"


class DatasetImage(Base):
    __tablename__ = "dataset_images"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    filename: Mapped[str] = mapped_column(String(500))
    r2_key: Mapped[str] = mapped_column(String(700))
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(12), default=ImageStatus.unlabeled.value, index=True
    )
    # List of normalized shapes (see module docstring).
    annotations: Mapped[list] = mapped_column(JSON, default=list)
    split: Mapped[str] = mapped_column(String(8), default="train")  # train/valid/test
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
