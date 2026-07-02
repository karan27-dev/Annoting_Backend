from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.common import gen_uuid, utcnow


class InvoiceStatus(str, enum.Enum):
    draft = "draft"
    sent = "sent"
    paid = "paid"
    overdue = "overdue"
    cancelled = "cancelled"


class ProjectQuote(Base):
    __tablename__ = "project_quotes"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    annotation_type: Mapped[str] = mapped_column(String(50))
    rate_per_label_inr: Mapped[float] = mapped_column(Numeric(8, 4))
    estimated_labels: Mapped[int] = mapped_column(Integer)
    quoted_total_inr: Mapped[float] = mapped_column(Numeric(14, 2))
    actual_labels: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_total_inr: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    turnaround_premium_pct: Mapped[float] = mapped_column(Numeric(5, 2), default=0)
    volume_discount_pct: Mapped[float] = mapped_column(Numeric(5, 2), default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # A quote is a draft until an admin reviews the dataset and publishes it —
    # only published quotes are visible to the client.
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    admin_notes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    client_id: Mapped[str] = mapped_column(ForeignKey("clients.id"), index=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    invoice_number: Mapped[str] = mapped_column(String(50), unique=True)
    amount_inr: Mapped[float] = mapped_column(Numeric(14, 2))
    gst_amount_inr: Mapped[float] = mapped_column(Numeric(14, 2))
    total_inr: Mapped[float] = mapped_column(Numeric(14, 2))
    status: Mapped[str] = mapped_column(String(20), default=InvoiceStatus.draft.value)
    razorpay_order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    pdf_r2_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
