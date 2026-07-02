from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base
from app.models.common import gen_uuid, utcnow


class Role(str, enum.Enum):
    super_admin = "super_admin"
    ops_manager = "ops_manager"
    reviewer = "reviewer"
    annotator = "annotator"
    client = "client"


class AnnotatorStatus(str, enum.Enum):
    pending = "pending"
    active = "active"
    probation = "probation"
    suspended = "suspended"


class ClientTier(str, enum.Enum):
    standard = "standard"
    priority = "priority"
    enterprise = "enterprise"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default=Role.client.value)
    full_name: Mapped[str] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    annotator_profile: Mapped["AnnotatorProfile | None"] = relationship(
        back_populates="user", uselist=False
    )
    client_profile: Mapped["Client | None"] = relationship(
        back_populates="user", uselist=False
    )


class AnnotatorProfile(Base):
    __tablename__ = "annotator_profiles"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), unique=True)
    skills: Mapped[dict] = mapped_column(
        JSON,
        default=lambda: {
            "bbox": False,
            "polygon": False,
            "segmentation": False,
            "keypoint": False,
            "classification": False,
        },
    )
    calibration_passed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    calibration_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    rolling_accuracy: Mapped[float] = mapped_column(Numeric(5, 2), default=0)
    total_jobs_completed: Mapped[int] = mapped_column(Integer, default=0)
    total_labels_completed: Mapped[int] = mapped_column(BigInteger, default=0)
    rework_rate: Mapped[float] = mapped_column(Numeric(5, 2), default=0)
    status: Mapped[str] = mapped_column(String(20), default=AnnotatorStatus.pending.value)
    cvat_username: Mapped[str | None] = mapped_column(String(100), nullable=True)
    cvat_user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Phase-1: annotator's CVAT canvas password. TODO: encrypt at rest.
    cvat_password: Mapped[str | None] = mapped_column(String(100), nullable=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # NOTE: encrypt at rest before storing real bank details.
    bank_account_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    user: Mapped[User] = relationship(back_populates="annotator_profile")


class Client(Base):
    __tablename__ = "clients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), unique=True)
    company_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    gst_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    billing_address: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_projects: Mapped[int] = mapped_column(Integer, default=0)
    total_spend_inr: Mapped[float] = mapped_column(Numeric(14, 2), default=0)
    tier: Mapped[str] = mapped_column(String(20), default=ClientTier.standard.value)

    user: Mapped[User] = relationship(back_populates="client_profile")


class AnnotatorApplication(Base):
    """Public annotator applications, before admin approval."""

    __tablename__ = "annotator_applications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    full_name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), index=True)
    city: Mapped[str | None] = mapped_column(String(100), nullable=True)
    device: Mapped[str | None] = mapped_column(String(50), nullable=True)
    experience: Mapped[str | None] = mapped_column(Text, nullable=True)
    preferred_task_types: Mapped[list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
