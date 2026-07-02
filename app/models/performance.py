from __future__ import annotations

from datetime import date

from sqlalchemy import Date, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.common import gen_uuid


class AnnotatorPerformanceSnapshot(Base):
    """Weekly snapshot, computed by a background job (see services/quality_sync)."""

    __tablename__ = "annotator_performance_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    annotator_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    week_start: Mapped[date] = mapped_column(Date)
    jobs_completed: Mapped[int] = mapped_column(Integer, default=0)
    labels_completed: Mapped[int] = mapped_column(Integer, default=0)
    avg_iou_score: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    rework_count: Mapped[int] = mapped_column(Integer, default=0)
    avg_time_per_job_minutes: Mapped[float | None] = mapped_column(
        Numeric(8, 2), nullable=True
    )
    labels_per_hour: Mapped[float | None] = mapped_column(Numeric(8, 2), nullable=True)
    earnings_inr: Mapped[float] = mapped_column(Numeric(10, 2), default=0)
