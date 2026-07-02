"""Quote calculation. Single source of truth for rates — mirror this in the
frontend estimator only for instant ballparks; the firm quote comes from here."""
from __future__ import annotations

from app.schemas.billing import QuoteBreakdown, QuoteResponse

# ₹ per label, and average objects per image used to estimate label volume.
RATE_CARD: dict[str, dict[str, float]] = {
    "classification": {"rate": 1.5, "avg_objects": 1},
    "bbox": {"rate": 5.0, "avg_objects": 6},
    "keypoint": {"rate": 8.0, "avg_objects": 4},
    "polygon": {"rate": 12.0, "avg_objects": 5},
    "segmentation": {"rate": 18.0, "avg_objects": 4},
}

GST_RATE = 0.18


def _turnaround_premium(days: int) -> float:
    if days <= 3:
        return 0.45
    if days <= 7:
        return 0.20
    return 0.0


def _volume_discount(labels: int) -> float:
    if labels >= 500_000:
        return 0.20
    if labels >= 100_000:
        return 0.12
    if labels >= 25_000:
        return 0.06
    return 0.0


def calculate_quote(
    annotation_type: str,
    image_count: int,
    avg_objects_per_image: float | None,
    turnaround_days: int,
) -> QuoteResponse:
    card = RATE_CARD.get(annotation_type, RATE_CARD["bbox"])
    avg_objects = avg_objects_per_image or card["avg_objects"]
    rate = card["rate"]

    estimated_labels = max(1, round(image_count * avg_objects))
    base = estimated_labels * rate

    premium_pct = _turnaround_premium(turnaround_days)
    discount_pct = _volume_discount(estimated_labels)

    rush = base * premium_pct
    after_rush = base + rush
    discount = after_rush * discount_pct
    total = round(after_rush - discount, 2)

    return QuoteResponse(
        annotation_type=annotation_type,
        rate_per_label_inr=rate,
        estimated_labels=estimated_labels,
        estimated_total_inr=total,
        turnaround_premium_pct=round(premium_pct * 100, 2),
        volume_discount_pct=round(discount_pct * 100, 2),
        breakdown=QuoteBreakdown(
            base_inr=round(base, 2),
            rush_premium_inr=round(rush, 2),
            volume_discount_inr=round(discount, 2),
        ),
    )


def calculate_quote_custom(
    annotation_type: str,
    image_count: int,
    avg_objects_per_image: float,
    turnaround_days: int,
    rate_override: float | None = None,
) -> QuoteResponse:
    """Admin-reviewed quote: same structure as calculate_quote but the admin
    can pin the density (objects/image) and the per-label rate."""
    card = RATE_CARD.get(annotation_type, RATE_CARD["bbox"])
    rate = rate_override if rate_override and rate_override > 0 else card["rate"]

    estimated_labels = max(1, round(image_count * avg_objects_per_image))
    base = estimated_labels * rate

    premium_pct = _turnaround_premium(turnaround_days)
    discount_pct = _volume_discount(estimated_labels)

    rush = base * premium_pct
    after_rush = base + rush
    discount = after_rush * discount_pct
    total = round(after_rush - discount, 2)

    return QuoteResponse(
        annotation_type=annotation_type,
        rate_per_label_inr=rate,
        estimated_labels=estimated_labels,
        estimated_total_inr=total,
        turnaround_premium_pct=round(premium_pct * 100, 2),
        volume_discount_pct=round(discount_pct * 100, 2),
        breakdown=QuoteBreakdown(
            base_inr=round(base, 2),
            rush_premium_inr=round(rush, 2),
            volume_discount_inr=round(discount, 2),
        ),
    )


def gst_breakdown(amount: float) -> tuple[float, float]:
    """Returns (gst_amount, total_with_gst)."""
    gst = round(amount * GST_RATE, 2)
    return gst, round(amount + gst, 2)
