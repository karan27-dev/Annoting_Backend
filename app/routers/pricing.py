from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import get_current_user
from app.models.user import User
from app.schemas.billing import QuoteRequest, QuoteResponse
from app.services.pricing_engine import calculate_quote

router = APIRouter(prefix="/pricing", tags=["pricing"])


@router.post("/quote", response_model=QuoteResponse)
async def quote(body: QuoteRequest, _: User = Depends(get_current_user)):
    return calculate_quote(
        body.annotation_type.value,
        body.image_count,
        body.avg_objects_per_image,
        body.turnaround_days,
    )
