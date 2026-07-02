from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.common import gen_uuid, utcnow


class AssignmentStatus(str, enum.Enum):
    assigned = "assigned"
    in_progress = "in_progress"
    submitted = "submitted"
    review_pending = "review_pending"
    approved = "approved"
    revision_required = "revision_required"
    rejected = "rejected"


class ReviewAction(str, enum.Enum):
    approved = "approved"
    revision_required = "revision_required"
    rejected = "rejected"


class TaskAssignment(Base):
    __tablename__ = "task_assignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    cvat_job_id: Mapped[int] = mapped_column(Integer, index=True)
    cvat_task_id: Mapped[int] = mapped_column(Integer)
    annotator_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id"), nullable=True, index=True
    )
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default=AssignmentStatus.assigned.value
    )
    iou_score: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    rework_count: Mapped[int] = mapped_column(SmallInteger, default=0)
    labels_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time_spent_minutes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    flag_reason: Mapped[str | None] = mapped_column(String(50), nullable=True)


class QualityReview(Base):
    __tablename__ = "quality_reviews"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    assignment_id: Mapped[str] = mapped_column(
        ForeignKey("task_assignments.id"), index=True
    )
    reviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(20))
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
