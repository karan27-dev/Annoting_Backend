"""Shared quote-review logic used by both the admin and reviewer dashboards.

Annoting staff (admin or reviewer role) look at a counted dataset, correct the
per-image object density if the auto-estimate is off (one image can hold 60+
objects), set the rate, and publish the quote to the client.
"""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.billing import ProjectQuote
from app.models.common import utcnow
from app.models.project import IntakeStatus, Project
from app.models.user import Client, User
from app.services.email_service import email_service
from app.services.pricing_engine import calculate_quote_custom


async def list_pending_quotes(db: AsyncSession) -> list[dict]:
    """Every counted dataset whose draft quote still needs a human price."""
    rows = (
        await db.execute(
            select(Project, Client)
            .join(Client, Client.id == Project.client_id, isouter=True)
            .where(Project.intake_status == IntakeStatus.pending_review.value)
            .order_by(Project.created_at.desc())
        )
    ).all()

    out: list[dict] = []
    for project, client in rows:
        quote = (
            await db.execute(
                select(ProjectQuote)
                .where(ProjectQuote.project_id == project.id)
                .order_by(ProjectQuote.created_at.desc())
            )
        ).scalars().first()
        out.append(
            {
                "project_id": project.id,
                "project_name": project.name,
                "client_company": client.company_name if client else None,
                "annotation_type": project.annotation_type,
                "image_count": project.image_count,
                "video_count": project.video_count,
                "total_files": project.total_images,
                "complexity_tier": project.complexity_tier,
                "estimated_objects_per_image": (
                    float(project.estimated_objects_per_image)
                    if project.estimated_objects_per_image is not None
                    else None
                ),
                "turnaround_days": project.turnaround_days,
                "delivery_format": project.delivery_format,
                "suggested": {
                    "rate_per_label_inr": float(quote.rate_per_label_inr),
                    "estimated_labels": quote.estimated_labels,
                    "quoted_total_inr": float(quote.quoted_total_inr),
                }
                if quote
                else None,
            }
        )
    return out


async def publish_project_quote(
    db: AsyncSession,
    project_id: str,
    avg_objects_per_image: float | None,
    rate_per_label_inr: float | None,
    notes: str | None,
) -> dict:
    """Set final density/rate and publish the quote for the client to accept."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.total_images:
        raise HTTPException(status_code=400, detail="No counted data to quote")

    avg = avg_objects_per_image or float(project.estimated_objects_per_image or 1)
    result = calculate_quote_custom(
        project.annotation_type,
        project.total_images,
        avg,
        project.turnaround_days or 14,
        rate_override=rate_per_label_inr,
    )

    quote = (
        await db.execute(
            select(ProjectQuote)
            .where(ProjectQuote.project_id == project.id)
            .order_by(ProjectQuote.created_at.desc())
        )
    ).scalars().first()
    if not quote:
        quote = ProjectQuote(
            project_id=project.id, annotation_type=project.annotation_type
        )
        db.add(quote)
    if quote.accepted_at is not None:
        raise HTTPException(status_code=400, detail="Quote already accepted")

    quote.rate_per_label_inr = result.rate_per_label_inr
    quote.estimated_labels = result.estimated_labels
    quote.quoted_total_inr = result.estimated_total_inr
    quote.turnaround_premium_pct = result.turnaround_premium_pct
    quote.volume_discount_pct = result.volume_discount_pct
    quote.admin_notes = notes
    quote.published_at = utcnow()

    project.estimated_objects_per_image = avg
    project.intake_status = IntakeStatus.quoted.value
    project.intake_detail = (
        f"Quote ready — {result.estimated_labels} labels at "
        f"₹{result.rate_per_label_inr}/label."
    )
    await db.commit()

    # Tell the client their reviewed quote is live (best-effort).
    try:
        client = await db.get(Client, project.client_id)
        user = await db.get(User, client.user_id) if client else None
        if user:
            email_service.send(
                user.email,
                f"Your quote for “{project.name}” is ready",
                f"<p>Our team reviewed your dataset ({project.total_images} files) "
                f"and published your quote: <b>₹{result.estimated_total_inr}</b>. "
                f"Accept it on your dashboard to start annotation.</p>",
            )
    except Exception:  # noqa: BLE001
        pass

    return {
        "message": "Quote published to client",
        "quoted_total_inr": result.estimated_total_inr,
        "estimated_labels": result.estimated_labels,
        "rate_per_label_inr": result.rate_per_label_inr,
    }
