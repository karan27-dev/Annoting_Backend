from __future__ import annotations

import hashlib
import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.models.assignment import AssignmentStatus, TaskAssignment
from app.models.common import utcnow
from app.schemas.misc import MessageResponse

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


def _valid_signature(raw: bytes, signature: str | None) -> bool:
    secret = settings.cvat_webhook_secret
    if not secret:
        # No secret configured (dev) — accept, but log-worthy in prod.
        return True
    if not signature:
        return False
    expected = hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@router.post("/cvat/job-updated", response_model=MessageResponse)
async def cvat_job_updated(
    request: Request,
    x_signature_256: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db),
):
    raw = await request.body()
    if not _valid_signature(raw, x_signature_256):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()
    job_id = payload.get("job", {}).get("id") or payload.get("job_id")
    new_status = payload.get("job", {}).get("state") or payload.get("status")
    iou = payload.get("iou_score")

    if job_id is None:
        return MessageResponse(message="ignored: no job id")

    assignment = (
        await db.execute(
            select(TaskAssignment).where(TaskAssignment.cvat_job_id == int(job_id))
        )
    ).scalar_one_or_none()
    if not assignment:
        return MessageResponse(message="ignored: unknown job")

    if iou is not None:
        assignment.iou_score = float(iou)

    # Map CVAT states into our assignment lifecycle.
    if new_status in ("completed", "submitted"):
        assignment.status = AssignmentStatus.review_pending.value
        assignment.submitted_at = utcnow()
    elif new_status == "rejected":
        assignment.status = AssignmentStatus.revision_required.value
        assignment.flag_reason = "auto-rejected"

    await db.commit()
    return MessageResponse(message="ok")
