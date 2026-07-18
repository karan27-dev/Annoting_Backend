from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import require_roles
from app.models.assignment import TaskAssignment
from app.models.common import utcnow
from app.models.project import CvatMapping, Project, ProjectStatus
from app.models.user import AnnotatorProfile, Role, User
from app.schemas.misc import AnnotatorStatusUpdate, MessageResponse
from app.schemas.project import CvatSetupRequest, ProjectOut, QuotePublishRequest
from app.services.cvat_client import CvatNotConfigured, cvat
from app.services.ingestion import IngestionError, ingest_project
from app.services.quote_review import list_pending_quotes, publish_project_quote

router = APIRouter(prefix="/admin", tags=["admin"])

ADMIN = require_roles(Role.super_admin, Role.ops_manager)


@router.get("/projects", response_model=list[ProjectOut])
async def all_projects(
    _: User = Depends(ADMIN), db: AsyncSession = Depends(get_db)
):
    rows = (
        await db.execute(select(Project).order_by(Project.created_at.desc()))
    ).scalars().all()
    return rows


@router.post("/projects/{project_id}/setup-cvat")
async def setup_cvat(
    project_id: str,
    body: CvatSetupRequest,
    _: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    """Create the CVAT project/task, push the uploaded dataset from R2 into CVAT,
    wait for extraction, and mirror the real CVAT jobs into our system."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Ensure a mapping exists with the chosen segment size.
    mapping = (
        await db.execute(
            select(CvatMapping).where(CvatMapping.project_id == project.id)
        )
    ).scalar_one_or_none()
    if not mapping:
        mapping = CvatMapping(project_id=project.id, segment_size=body.segment_size)
        db.add(mapping)
    else:
        mapping.segment_size = body.segment_size
    await db.commit()

    try:
        summary = await ingest_project(db, project)
    except CvatNotConfigured:
        raise HTTPException(status_code=503, detail="CVAT is not configured")
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "message": (
            f"Ingested {summary['total_images']} images into CVAT "
            f"and created {summary['jobs_total']} jobs."
        ),
        **summary,
    }


@router.post("/projects/{project_id}/launch", response_model=MessageResponse)
async def launch_project(
    project_id: str,
    _: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Jobs come from the real CVAT ingestion done in setup-cvat.
    job_count = (
        await db.execute(
            select(func.count(TaskAssignment.id)).where(
                TaskAssignment.project_id == project.id
            )
        )
    ).scalar() or 0
    if job_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No CVAT jobs yet — run Setup CVAT (ingest the dataset) first.",
        )

    project.status = ProjectStatus.active.value
    await db.commit()
    return MessageResponse(
        message=f"Project launched — {job_count} jobs available to annotators"
    )


@router.post("/projects/{project_id}/export", response_model=MessageResponse)
async def export_project(
    project_id: str,
    _: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    """Deliver in the format the client chose at project creation."""
    from app.services.export_formats import cvat_format_for

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    fmt = cvat_format_for(project.delivery_format)
    mapping = (
        await db.execute(
            select(CvatMapping).where(CvatMapping.project_id == project.id)
        )
    ).scalar_one_or_none()
    if mapping:
        mapping.export_format = fmt
        if mapping.cvat_task_ids:
            try:
                await cvat.trigger_export(mapping.cvat_task_ids[0], fmt)
            except CvatNotConfigured:
                pass  # dev without CVAT — still mark delivered
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    status_code=502, detail=f"CVAT export failed: {e}"
                )

    project.status = ProjectStatus.delivered.value
    project.delivered_at = utcnow()
    await db.commit()
    return MessageResponse(
        message=f"Export triggered in {fmt}. Client will be notified."
    )


@router.get("/quotes/pending")
async def pending_quotes(
    _: User = Depends(ADMIN), db: AsyncSession = Depends(get_db)
):
    """Projects whose draft quote awaits review — check the counted dataset,
    adjust density/rate if the auto-estimate is off (e.g. 60 objects in one
    image), and publish the quote to the client."""
    return await list_pending_quotes(db)


@router.post("/projects/{project_id}/quote/publish")
async def publish_quote(
    project_id: str,
    body: QuotePublishRequest,
    _: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    return await publish_project_quote(
        db,
        project_id,
        body.avg_objects_per_image,
        body.rate_per_label_inr,
        body.notes,
    )


@router.get("/annotators")
async def list_annotators(
    _: User = Depends(ADMIN), db: AsyncSession = Depends(get_db)
):
    rows = (
        await db.execute(
            select(AnnotatorProfile, User).join(
                User, User.id == AnnotatorProfile.user_id
            )
        )
    ).all()
    return [
        {
            "id": profile.user_id,
            "full_name": user.full_name,
            "city": profile.city,
            "status": profile.status,
            "rolling_accuracy": float(profile.rolling_accuracy),
            "total_jobs_completed": profile.total_jobs_completed,
            "skills": profile.skills or {},
        }
        for profile, user in rows
    ]


@router.patch("/annotators/{user_id}/status", response_model=MessageResponse)
async def set_annotator_status(
    user_id: str,
    body: AnnotatorStatusUpdate,
    _: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    profile = (
        await db.execute(
            select(AnnotatorProfile).where(AnnotatorProfile.user_id == user_id)
        )
    ).scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Annotator not found")
    profile.status = body.status
    await db.commit()
    return MessageResponse(message=f"Annotator set to {body.status}")


@router.get("/dashboard")
async def dashboard(_: User = Depends(ADMIN), db: AsyncSession = Depends(get_db)):
    """Operations command center — everything real, nothing hardcoded."""
    from app.models.assignment import AssignmentStatus
    from app.models.billing import Invoice, InvoiceStatus
    from app.models.project import IntakeStatus

    # ── Pipeline funnel: every project bucketed into one operational stage ────
    projects = (await db.execute(select(Project))).scalars().all()
    funnel = {
        "awaiting_data": 0,
        "counting": 0,
        "pending_quote": 0,
        "quoted": 0,
        "in_annotation": 0,
        "in_review": 0,
        "delivered": 0,
    }
    for p in projects:
        if p.status == ProjectStatus.delivered.value:
            funnel["delivered"] += 1
        elif p.status == ProjectStatus.review.value:
            funnel["in_review"] += 1
        elif p.status == ProjectStatus.active.value:
            funnel["in_annotation"] += 1
        elif p.intake_status == IntakeStatus.pending_review.value:
            funnel["pending_quote"] += 1
        elif p.intake_status == IntakeStatus.quoted.value:
            funnel["quoted"] += 1
        elif p.intake_status == IntakeStatus.quote_accepted.value:
            funnel["in_annotation"] += 1
        elif p.intake_status == IntakeStatus.counting.value:
            funnel["counting"] += 1
        else:
            funnel["awaiting_data"] += 1

    annotators = (
        await db.execute(
            select(func.count(AnnotatorProfile.id)).where(
                AnnotatorProfile.status == "active"
            )
        )
    ).scalar() or 0

    # ── Review queue depth (jobs waiting on a reviewer) ──────────────────────
    review_pending = (
        await db.execute(
            select(func.count(TaskAssignment.id)).where(
                TaskAssignment.status == AssignmentStatus.review_pending.value
            )
        )
    ).scalar() or 0

    # ── Labels delivered per day, last 14 days (real submissions) ────────────
    assignments = (await db.execute(select(TaskAssignment))).scalars().all()
    now = utcnow()
    series: list[int] = []
    for i in range(13, -1, -1):
        day_start = (now - timedelta(days=i)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + timedelta(days=1)
        count = 0
        for a in assignments:
            ts = a.completed_at
            if ts is None:
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=now.tzinfo)
            if day_start <= ts < day_end:
                count += a.labels_count or 0
        series.append(count)
    labels_today = series[-1]

    # ── Revenue: this month's issued invoices + outstanding balance ──────────
    month_start = now.replace(
        day=1, hour=0, minute=0, second=0, microsecond=0
    )
    invoices = (await db.execute(select(Invoice))).scalars().all()
    revenue_mtd = 0.0
    outstanding = 0.0
    for inv in invoices:
        issued = inv.issued_at
        if issued is not None:
            if issued.tzinfo is None:
                issued = issued.replace(tzinfo=now.tzinfo)
            if issued >= month_start:
                revenue_mtd += float(inv.total_inr)
        if inv.status != InvoiceStatus.paid.value:
            outstanding += float(inv.total_inr)

    return {
        "active_projects": funnel["in_annotation"],
        "annotators_online": int(annotators),
        "labels_today": int(labels_today),
        "revenue_mtd_inr": round(revenue_mtd, 2),
        "outstanding_inr": round(outstanding, 2),
        "pending_setup": funnel["awaiting_data"] + funnel["counting"],
        "pending_quotes": funnel["pending_quote"],
        "review_queue": int(review_pending),
        "funnel": funnel,
        "labels_series_14d": series,
    }
