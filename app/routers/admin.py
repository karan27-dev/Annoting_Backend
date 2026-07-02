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
    """Projects whose draft quote awaits review — the admin checks the counted
    dataset, adjusts density/rate if the auto-estimate is off (e.g. 60 objects
    in one image), and publishes the quote to the client."""
    from app.models.billing import ProjectQuote
    from app.models.project import IntakeStatus
    from app.models.user import Client

    rows = (
        await db.execute(
            select(Project, Client)
            .join(Client, Client.id == Project.client_id, isouter=True)
            .where(Project.intake_status == IntakeStatus.pending_review.value)
            .order_by(Project.created_at.desc())
        )
    ).all()

    out = []
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


@router.post("/projects/{project_id}/quote/publish")
async def publish_quote(
    project_id: str,
    body: QuotePublishRequest,
    _: User = Depends(ADMIN),
    db: AsyncSession = Depends(get_db),
):
    """Admin sets the final density/rate after eyeballing the dataset, then the
    quote goes live for the client to accept."""
    from app.models.billing import ProjectQuote
    from app.models.project import IntakeStatus
    from app.models.user import Client
    from app.services.email_service import email_service
    from app.services.pricing_engine import calculate_quote_custom

    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not project.total_images:
        raise HTTPException(status_code=400, detail="No counted data to quote")

    avg = body.avg_objects_per_image or float(
        project.estimated_objects_per_image or 1
    )
    result = calculate_quote_custom(
        project.annotation_type,
        project.total_images,
        avg,
        project.turnaround_days or 14,
        rate_override=body.rate_per_label_inr,
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
    quote.admin_notes = body.notes
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
    active = (
        await db.execute(
            select(func.count(Project.id)).where(
                Project.status == ProjectStatus.active.value
            )
        )
    ).scalar() or 0
    pending = (
        await db.execute(
            select(func.count(Project.id)).where(
                Project.status == ProjectStatus.pending_setup.value
            )
        )
    ).scalar() or 0
    annotators = (
        await db.execute(
            select(func.count(AnnotatorProfile.id)).where(
                AnnotatorProfile.status == "active"
            )
        )
    ).scalar() or 0

    today = utcnow() - timedelta(days=1)
    labels_today = (
        await db.execute(
            select(func.coalesce(func.sum(TaskAssignment.labels_count), 0)).where(
                TaskAssignment.completed_at.is_not(None),
                TaskAssignment.completed_at >= today,
            )
        )
    ).scalar() or 0

    return {
        "active_projects": int(active),
        "annotators_online": int(annotators),
        "labels_today": int(labels_today),
        "revenue_mtd_inr": 0,
        "pending_setup": int(pending),
    }
