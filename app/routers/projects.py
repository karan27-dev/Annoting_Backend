from __future__ import annotations

import secrets

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models.assignment import AssignmentStatus, TaskAssignment
from app.models.billing import ProjectQuote
from app.models.common import utcnow
from app.models.project import IntakeStatus, Project, ProjectStatus
from app.models.user import Client, Role, User
from app.schemas.misc import MessageResponse
from app.schemas.project import (
    IntakeOut,
    ProgressOut,
    ProjectCreate,
    ProjectOut,
    QuoteSummary,
    StatusUpdate,
)
from app.services.cvat_client import cvat

router = APIRouter(prefix="/projects", tags=["projects"])


async def _client_for(user: User, db: AsyncSession) -> Client:
    client = (
        await db.execute(select(Client).where(Client.user_id == user.id))
    ).scalar_one_or_none()
    if not client:
        # Self-heal: a client-role user without a profile row (e.g. created
        # before profiles existed) gets one on first use instead of a 403.
        if user.role == Role.client.value:
            client = Client(user_id=user.id, company_name=user.full_name)
            db.add(client)
            await db.commit()
            await db.refresh(client)
            return client
        raise HTTPException(status_code=403, detail="No client profile for this user")
    return client


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    client = await _client_for(user, db)
    rows = (
        await db.execute(
            select(Project)
            .where(Project.client_id == client.id)
            .order_by(Project.created_at.desc())
        )
    ).scalars().all()
    return rows


@router.post("", response_model=ProjectOut, status_code=201)
async def create_project(
    body: ProjectCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    client = await _client_for(user, db)
    project = Project(
        client_id=client.id,
        name=body.name,
        description=body.description,
        annotation_type=body.annotation_type.value,
        label_taxonomy=[lc.model_dump() for lc in body.label_taxonomy],
        total_images=body.total_images,
        turnaround_days=body.turnaround_days,
        status=ProjectStatus.pending_setup.value,
        media_type=body.media_type.value,
        data_source=body.data_source.value,
        delivery_format=body.delivery_format.value,
        mode=body.mode.value,
    )
    db.add(project)
    client.total_projects += 1
    await db.commit()
    await db.refresh(project)
    return project


async def _owned_project(
    project_id: str, user: User, db: AsyncSession
) -> Project:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if user.role not in (Role.super_admin.value, Role.ops_manager.value):
        client = await _client_for(user, db)
        if project.client_id != client.id:
            raise HTTPException(status_code=403, detail="Not your project")
    return project


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _owned_project(project_id, user, db)


async def _latest_quote(project_id: str, db: AsyncSession) -> ProjectQuote | None:
    return (
        await db.execute(
            select(ProjectQuote)
            .where(ProjectQuote.project_id == project_id)
            .order_by(ProjectQuote.created_at.desc())
        )
    ).scalars().first()


@router.get("/{project_id}/intake", response_model=IntakeOut)
async def project_intake(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Live intake status: what we detected in the client's data and the
    auto-generated quote. Polled by the dashboard while counting runs."""
    p = await _owned_project(project_id, user, db)
    quote = await _latest_quote(project_id, db)
    return IntakeOut(
        intake_status=p.intake_status,
        intake_detail=p.intake_detail,
        media_type=p.media_type,
        image_count=p.image_count,
        video_count=p.video_count,
        total_files=p.total_images,
        data_source=p.data_source,
        gdrive_link=p.gdrive_link,
        estimated_objects_per_image=(
            float(p.estimated_objects_per_image)
            if p.estimated_objects_per_image is not None
            else None
        ),
        complexity_tier=p.complexity_tier,
        delivery_format=p.delivery_format,
        # Drafts stay internal — the client only sees admin-published quotes.
        quote=(
            QuoteSummary.model_validate(quote)
            if quote and quote.published_at is not None
            else None
        ),
    )


async def _activate_for_annotation(project: Project, db: AsyncSession) -> None:
    """Make an accepted project immediately visible on the annotator dashboard.

    The whole dataset is ONE job (never split). If CVAT ingestion hasn't run
    yet we create a placeholder full-container assignment so annotators see the
    work right away; real CVAT ingestion later replaces the placeholder (its
    cleanup removes assignments with cvat_job_id >= 900_000)."""
    project.status = ProjectStatus.active.value

    existing = (
        await db.execute(
            select(func.count(TaskAssignment.id)).where(
                TaskAssignment.project_id == project.id
            )
        )
    ).scalar() or 0
    if existing == 0:
        # Placeholder full-container job — one job, all frames.
        placeholder_id = 900_000 + secrets.randbelow(90_000_000)
        db.add(
            TaskAssignment(
                project_id=project.id,
                cvat_job_id=placeholder_id,
                cvat_task_id=placeholder_id,
                frame_count=project.total_images or project.image_count or 0,
                status=AssignmentStatus.assigned.value,
            )
        )


@router.post("/{project_id}/quote/accept", response_model=MessageResponse)
async def accept_quote(
    project_id: str,
    background: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Client accepts the quote — the project goes live for annotation
    immediately (full dataset as one job), and we kick off CVAT ingestion in
    the background to wire up the real labeling canvas."""
    p = await _owned_project(project_id, user, db)
    quote = await _latest_quote(project_id, db)
    if not quote or quote.published_at is None:
        raise HTTPException(
            status_code=400,
            detail="No published quote yet — our team is still reviewing your dataset.",
        )
    if quote.accepted_at is None:
        quote.accepted_at = utcnow()
        p.intake_status = IntakeStatus.quote_accepted.value
        await _activate_for_annotation(p, db)
        await db.commit()
        # Best-effort: push data into CVAT and mirror real jobs. If CVAT is
        # down the placeholder job stays visible until an admin re-runs setup.
        from app.services.ingestion import run_ingestion_bg

        background.add_task(run_ingestion_bg, project_id)
    return MessageResponse(
        message="Quote accepted — your project is now live for annotation."
    )


@router.get("/{project_id}/progress", response_model=ProgressOut)
async def project_progress(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    p = await _owned_project(project_id, user, db)
    percent = (p.images_completed / p.total_images * 100) if p.total_images else 0
    # Velocity/ETA are refined by the quality_sync job; this is a simple estimate.
    velocity = max(1.0, p.images_completed / 7) if p.images_completed else 0.0
    remaining = p.total_images - p.images_completed
    eta = int(remaining / velocity) if velocity else None
    return ProgressOut(
        images_done=p.images_completed,
        images_total=p.total_images,
        percent=round(percent, 1),
        velocity_per_day=round(velocity, 1),
        eta_days=eta,
    )


@router.get("/{project_id}/quality")
async def project_quality(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Real quality metrics — nothing synthetic. Per-class label distribution
    comes live from CVAT, velocity from actual submissions; both are empty
    until annotators start working."""
    from datetime import timedelta

    from app.models.assignment import AssignmentStatus, TaskAssignment

    p = await _owned_project(project_id, user, db)

    assignments = (
        await db.execute(
            select(TaskAssignment).where(TaskAssignment.project_id == project_id)
        )
    ).scalars().all()

    reviewed = [
        a for a in assignments if a.status == AssignmentStatus.approved.value
    ]

    # ── Per-class distribution, live from CVAT (bounded) ─────────────────────
    per_class: dict[str, dict] = {}
    total_shapes = 0
    for a in assignments[:10]:
        try:
            labels = await cvat.get_job_labels(a.cvat_job_id)
            lm = {
                lab["id"]: (lab["name"], lab.get("color") or "#e2553d")
                for lab in labels
            }
            ann = await cvat.get_job_annotations(a.cvat_job_id)
            for s in ann.get("shapes", []):
                name, color = lm.get(s.get("label_id"), ("unknown", "#8b857c"))
                entry = per_class.setdefault(name, {"color": color, "count": 0})
                entry["count"] += 1
                total_shapes += 1
        except Exception:  # noqa: BLE001 — CVAT down/unconfigured
            continue

    classes = [
        {
            "name": name,
            "color": v["color"],
            "count": v["count"],
            "share": round(v["count"] / total_shapes * 100, 1) if total_shapes else 0,
        }
        for name, v in sorted(
            per_class.items(), key=lambda kv: -kv[1]["count"]
        )
    ]

    # ── Velocity: labels/day over the last 7 days, from real submissions ─────
    now = utcnow()
    velocity = []
    for i in range(6, -1, -1):
        day_start = (now - timedelta(days=i)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        day_end = day_start + timedelta(days=1)
        count = 0
        for a in assignments:
            ts = a.completed_at or a.submitted_at
            if ts is None:
                continue
            if ts.tzinfo is None:  # SQLite returns naive datetimes
                from datetime import timezone as _tz

                ts = ts.replace(tzinfo=_tz.utc)
            if day_start <= ts < day_end:
                count += a.labels_count or 0
        velocity.append(
            {"day": day_start.strftime("%a"), "labels": count}
        )

    return {
        "aggregate_iou": float(p.quality_score) if p.quality_score else None,
        "quality_target": float(p.quality_target),
        "reviewed_jobs": len(reviewed),
        "total_jobs": len(assignments),
        "total_shapes": total_shapes,
        "per_class": classes,
        "velocity": velocity,
    }


@router.get("/{project_id}/sample")
async def project_sample(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Finalized labeled samples for the client preview — approved frames with
    their annotation overlays, pulled live from CVAT."""
    from app.models.assignment import AssignmentStatus, TaskAssignment
    from app.services.annotation_view import frame_payload

    await _owned_project(project_id, user, db)

    approved = (
        await db.execute(
            select(TaskAssignment).where(
                TaskAssignment.project_id == project_id,
                TaskAssignment.status == AssignmentStatus.approved.value,
            )
        )
    ).scalars().all()

    samples: list[dict] = []
    for a in approved:
        try:
            meta = await cvat.get_job_meta(a.cvat_job_id)
            start = meta.get("start_frame", 0)
            stop = meta.get("stop_frame", start)
            # A few frames per approved job, up to 10 total.
            for fr in range(start, min(stop + 1, start + 5)):
                payload = await frame_payload(a.cvat_job_id, fr)
                if payload["shapes"]:  # only frames that actually have labels
                    samples.append(payload)
                if len(samples) >= 10:
                    break
        except Exception:  # noqa: BLE001
            continue
        if len(samples) >= 10:
            break
    return samples


@router.patch("/{project_id}/status", response_model=ProjectOut)
async def update_status(
    project_id: str,
    body: StatusUpdate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if user.role not in (Role.super_admin.value, Role.ops_manager.value):
        raise HTTPException(status_code=403, detail="Admin only")
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    project.status = body.status
    await db.commit()
    await db.refresh(project)
    return project
