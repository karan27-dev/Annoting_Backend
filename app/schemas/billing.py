from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.models.project import AnnotationType


class QuoteRequest(BaseModel):
    annotation_type: AnnotationType
    image_count: int
    avg_objects_per_image: float | None = None
    turnaround_days: int = 14


class QuoteBreakdown(BaseModel):
    base_inr: float
    rush_premium_inr: float
    volume_discount_inr: float


class QuoteResponse(BaseModel):
    annotation_type: str
    rate_per_label_inr: float
    estimated_labels: int
    estimated_total_inr: float
    turnaround_premium_pct: float
    volume_discount_pct: float
    breakdown: QuoteBreakdown


class InvoiceOut(BaseModel):
    id: str
    invoice_number: str
    project_id: str
    amount_inr: float
    gst_amount_inr: float
    total_inr: float
    status: str
    issued_at: datetime | None
    due_at: datetime | None

    class Config:
        from_attributes = True


class InvoiceCreate(BaseModel):
    project_id: str
