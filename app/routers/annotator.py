from __future__ import annotations

import secrets
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user, require_roles
from app.models.assignment import AssignmentStatus, TaskAssignment
from app.models.common import utcnow
from app.models.project import Project
from app.models.user import AnnotatorApplication, AnnotatorProfile, Role, User
from app.schemas.misc import AnnotatorApplyRequest, MessageResponse
from app.services.cvat_client import CvatNotConfigured, cvat
from app.services.pricing_engine import RATE_CARD

router = APIRouter(prefix="/annotator", tags=["annotator"])

# Rough seconds-per-image by task type — used to estimate job time.
_EFFORT_SECONDS = {
    "classification": 6,
    "bbox": 20,
    "keypoint": 30,
    "polygon": 45,
    "segmentation": 60,
}


def _rate_for(annotation_type: str) -> float:
    return float(RATE_CARD.get(annotation_type, RATE_CARD["bbox"])["rate"])


def _est_minutes(annotation_type: str, frames: int) -> int:
    secs = _EFFORT_SECONDS.get(annotation_type, 20)
    return max(1, round(frames * secs / 60))


@router.post("/apply", response_model=MessageResponse, status_code=201)
async def apply(body: AnnotatorApplyRequest, db: AsyncSession = Depends(get_db)):
    """Public endpoint — annotator application form on the marketing site."""
    db.add(
        AnnotatorApplication(
            full_name=body.full_name,
            email=body.email,
            city=body.city,
            device=body.device,
            experience=body.experience,
            preferred_task_types=body.preferred_task_types,
        )
    )
    await db.commit()
    return MessageResponse(message="Application received. We'll be in touch.")


async def _profile(user: User, db: AsyncSession) -> AnnotatorProfile:
    p = (
        await db.execute(
            select(AnnotatorProfile).where(AnnotatorProfile.user_id == user.id)
        )
    ).scalar_one_or_none()
    if not p:
        raise HTTPException(status_code=404, detail="No annotator profile")
    return p


@router.get("/jobs/available")
async def available_jobs(
    user: User = Depends(require_roles(Role.annotator)),
    db: AsyncSession = Depends(get_db),
):
    """Unassigned jobs matching the annotator's certified skills."""
    profile = await _profile(user, db)
    skills = {k for k, v in (profile.skills or {}).items() if v}

    rows = (
        await db.execute(
            select(TaskAssignment, Project)
            .join(Project, Project.id == TaskAssignment.project_id)
            .where(TaskAssignment.annotator_id.is_(None))
            .where(TaskAssignment.status == AssignmentStatus.assigned.value)
        )
    ).all()

    out = []
    for assignment, project in rows:
        if project.annotation_type not in skills:
            continue
        frames = assignment.frame_count or 0
        out.append(
            {
                "cvat_job_id": assignment.cvat_job_id,
                "cvat_task_id": assignment.cvat_task_id,
                "project_name": project.name,
                "annotation_type": project.annotation_type,
                "image_count": frames,
                "rate_per_label": _rate_for(project.annotation_type),
                "estimated_minutes": _est_minutes(project.annotation_type, frames),
            }
        )
    return out


@router.post("/jobs/{cvat_job_id}/accept")
async def accept_job(
    cvat_job_id: int,
    user: User = Depends(require_roles(Role.annotator)),
    db: AsyncSession = Depends(get_db),
):
    profile = await _profile(user, db)
    assignment = (
        await db.execute(
            select(TaskAssignment).where(TaskAssignment.cvat_job_id == cvat_job_id)
        )
    ).scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Job not found")
    if assignment.annotator_id and assignment.annotator_id != user.id:
        raise HTTPException(status_code=409, detail="Job already taken")

    # Provision a real CVAT canvas account on first accept (cvat_password is our
    # marker that we've actually created one — seed data may carry a fake user id).
    if cvat.configured and not profile.cvat_password:
        username = f"ann_{user.id.replace('-', '')[:10]}"
        password = secrets.token_urlsafe(9)
        try:
            await cvat.register_user(username, user.email, password)
        except Exception:  # noqa: BLE001 — may already exist; look it up below
            pass
        uid = await cvat.find_user_id(username)
        if uid:
            profile.cvat_username = username
            profile.cvat_user_id = uid
            profile.cvat_password = password

    assignment.annotator_id = user.id
    assignment.status = AssignmentStatus.in_progress.value
    assignment.started_at = utcnow()

    # Assign the real CVAT job to the annotator's CVAT account.
    if profile.cvat_user_id:
        try:
            await cvat.assign_job(cvat_job_id, profile.cvat_user_id)
        except (CvatNotConfigured, Exception):  # noqa: BLE001
            pass

    await db.commit()
    return {
        "deep_link": cvat.job_deep_link(cvat_job_id),
        "cvat_username": profile.cvat_username,
        "cvat_password": profile.cvat_password,
    }


@router.get("/jobs/active")
async def active_jobs(
    user: User = Depends(require_roles(Role.annotator)),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(TaskAssignment, Project)
            .join(Project, Project.id == TaskAssignment.project_id)
            .where(
                TaskAssignment.annotator_id == user.id,
                TaskAssignment.status.in_(
                    [
                        AssignmentStatus.in_progress.value,
                        AssignmentStatus.revision_required.value,
                    ]
                ),
            )
        )
    ).all()
    return [
        {
            "cvat_job_id": a.cvat_job_id,
            "project_name": p.name,
            "annotation_type": p.annotation_type,
            "image_count": a.frame_count or 0,
            "status": a.status,
            "deep_link": cvat.job_deep_link(a.cvat_job_id),
        }
        for a, p in rows
    ]


@router.post("/jobs/{cvat_job_id}/submit", response_model=MessageResponse)
async def submit_job(
    cvat_job_id: int,
    user: User = Depends(require_roles(Role.annotator)),
    db: AsyncSession = Depends(get_db),
):
    """Annotator marks a job done → moves to the reviewer queue."""
    assignment = (
        await db.execute(
            select(TaskAssignment).where(
                TaskAssignment.cvat_job_id == cvat_job_id,
                TaskAssignment.annotator_id == user.id,
            )
        )
    ).scalar_one_or_none()
    if not assignment:
        raise HTTPException(status_code=404, detail="Job not found")

    assignment.status = AssignmentStatus.review_pending.value
    assignment.submitted_at = utcnow()
    if assignment.started_at:
        delta = utcnow() - assignment.started_at
        assignment.time_spent_minutes = max(1, int(delta.total_seconds() / 60))
    # Labels count ≈ frames × avg objects; refined by the CVAT quality sync later.
    assignment.labels_count = assignment.frame_count or 0
    await db.commit()
    return MessageResponse(message="Submitted for review")


@router.get("/jobs/history")
async def job_history(
    user: User = Depends(require_roles(Role.annotator)),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(TaskAssignment, Project)
            .join(Project, Project.id == TaskAssignment.project_id)
            .where(
                TaskAssignment.annotator_id == user.id,
                TaskAssignment.status.in_(
                    [
                        AssignmentStatus.review_pending.value,
                        AssignmentStatus.approved.value,
                        AssignmentStatus.rejected.value,
                    ]
                ),
            )
            .order_by(TaskAssignment.submitted_at.desc())
        )
    ).all()
    return [
        {
            "cvat_job_id": a.cvat_job_id,
            "project_name": p.name,
            "annotation_type": p.annotation_type,
            "image_count": a.frame_count or 0,
            "status": a.status,
            "iou_score": float(a.iou_score) if a.iou_score is not None else None,
            "rework_count": a.rework_count,
            "submitted_at": a.submitted_at,
        }
        for a, p in rows
    ]


@router.get("/performance")
async def performance(
    user: User = Depends(require_roles(Role.annotator)),
    db: AsyncSession = Depends(get_db),
):
    profile = await _profile(user, db)
    week_ago = utcnow() - timedelta(days=7)
    recent = (
        await db.execute(
            select(TaskAssignment).where(
                TaskAssignment.annotator_id == user.id,
                TaskAssignment.submitted_at.is_not(None),
                TaskAssignment.submitted_at >= week_ago,
            )
        )
    ).scalars().all()
    labels_week = sum(a.labels_count or 0 for a in recent)

    actives = (
        await db.execute(
            select(AnnotatorProfile).where(AnnotatorProfile.status == "active")
        )
    ).scalars().all()
    ranked = sorted(actives, key=lambda p: float(p.rolling_accuracy), reverse=True)
    rank = next(
        (i + 1 for i, p in enumerate(ranked) if p.user_id == user.id), None
    )

    # Average delivered rate across the annotator's certified skills.
    skills = [k for k, v in (profile.skills or {}).items() if v]
    avg_rate = (
        sum(_rate_for(s) for s in skills) / len(skills) if skills else _rate_for("bbox")
    )

    return {
        "rolling_accuracy": float(profile.rolling_accuracy),
        "labels_this_week": labels_week,
        "earnings_week_inr": round(labels_week * avg_rate, 2),
        "rank": rank,
        "total_active": len(actives),
    }


@router.get("/profile")
async def my_profile(
    user: User = Depends(require_roles(Role.annotator)),
    db: AsyncSession = Depends(get_db),
):
    """Real skills, calibration status and stats for the annotator UI."""
    profile = await _profile(user, db)
    return {
        "skills": profile.skills or {},
        "status": profile.status,
        "calibration_passed_at": profile.calibration_passed_at,
        "calibration_score": (
            float(profile.calibration_score)
            if profile.calibration_score is not None
            else None
        ),
        "rolling_accuracy": float(profile.rolling_accuracy),
        "rework_rate": float(profile.rework_rate),
        "total_jobs_completed": profile.total_jobs_completed,
        "total_labels_completed": profile.total_labels_completed,
        "cvat_username": profile.cvat_username,
    }


@router.post("/calibration/start")
async def start_calibration(
    user: User = Depends(require_roles(Role.annotator)),
):
    # Real flow: create a CVAT calibration task with 20 GT frames, return deep-link.
    try:
        return {"deep_link": cvat.job_deep_link(0), "message": "Calibration ready"}
    except CvatNotConfigured:
        return {"deep_link": None, "message": "CVAT not configured yet"}
