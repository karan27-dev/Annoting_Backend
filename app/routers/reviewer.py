from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import require_roles
from app.models.assignment import (
    AssignmentStatus,
    QualityReview,
    ReviewAction,
    TaskAssignment,
)
from app.models.common import hours_since, utcnow
from app.models.project import Project
from app.models.user import AnnotatorProfile, Role, User
from app.schemas.misc import MessageResponse, ReviewActionRequest
from app.schemas.project import QuotePublishRequest
from app.services.cvat_client import cvat

router = APIRouter(prefix="/reviewer", tags=["reviewer"])


def _label_map(labels: list[dict]) -> dict[int, dict]:
    return {
        lab["id"]: {"name": lab["name"], "color": lab.get("color") or "#e2553d"}
        for lab in labels
    }


@router.get("/queue")
async def queue(
    _: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    rows = (
        await db.execute(
            select(TaskAssignment, Project, User)
            .join(Project, Project.id == TaskAssignment.project_id)
            .join(User, User.id == TaskAssignment.annotator_id, isouter=True)
            .where(
                TaskAssignment.status == AssignmentStatus.review_pending.value
            )
            .order_by(TaskAssignment.submitted_at.asc())
        )
    ).all()

    out = []
    for assignment, project, annotator in rows:
        age_h = hours_since(assignment.submitted_at)
        out.append(
            {
                "assignment_id": assignment.id,
                "annotator_name": annotator.full_name if annotator else "Unassigned",
                "task_type": project.annotation_type,
                "iou_score": float(assignment.iou_score or 0),
                "reason": assignment.flag_reason or "auto-rejected",
                "age_hours": age_h,
            }
        )
    return out


@router.get("/queue/{assignment_id}")
async def review_detail(
    assignment_id: str,
    _: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    """Full review item: job info + the annotator's labeled-data summary from CVAT."""
    a = await db.get(TaskAssignment, assignment_id)
    if not a:
        raise HTTPException(status_code=404, detail="Review item not found")
    project = await db.get(Project, a.project_id)
    annotator = await db.get(User, a.annotator_id) if a.annotator_id else None

    detail = {
        "assignment_id": a.id,
        "cvat_job_id": a.cvat_job_id,
        "cvat_task_id": a.cvat_task_id,
        "annotator_name": annotator.full_name if annotator else "Unassigned",
        "project_id": a.project_id,
        "project_name": project.name if project else "",
        "annotation_type": project.annotation_type if project else "bbox",
        # The client chose this at project creation — approving here moves the
        # data toward delivery in exactly this format.
        "delivery_format": project.delivery_format if project else "coco",
        "iou_score": float(a.iou_score) if a.iou_score is not None else None,
        "reason": a.flag_reason or "spot-check",
        "status": a.status,
        "frame_count": a.frame_count or 0,
        "deep_link": cvat.job_deep_link(a.cvat_job_id),
    }

    # Pull the real annotation layers from CVAT (best-effort).
    try:
        labels = await cvat.get_job_labels(a.cvat_job_id)
        lm = _label_map(labels)
        ann = await cvat.get_job_annotations(a.cvat_job_id)
        meta = await cvat.get_job_meta(a.cvat_job_id)
        shapes = ann.get("shapes", [])
        summary: dict[str, int] = {}
        for s in shapes:
            nm = lm.get(s.get("label_id"), {}).get("name", "unknown")
            summary[nm] = summary.get(nm, 0) + 1
        detail.update(
            {
                "cvat_available": True,
                "start_frame": meta.get("start_frame", 0),
                "stop_frame": meta.get("stop_frame", (a.frame_count or 1) - 1),
                "total_shapes": len(shapes),
                "shape_summary": summary,
                "labels": [
                    {"name": lab["name"], "color": lab.get("color") or "#e2553d"}
                    for lab in labels
                ],
            }
        )
    except Exception:  # noqa: BLE001 — CVAT down/unconfigured; degrade gracefully
        detail.update(
            {
                "cvat_available": False,
                "start_frame": 0,
                "stop_frame": max(0, (a.frame_count or 1) - 1),
                "total_shapes": 0,
                "shape_summary": {},
                "labels": project.label_taxonomy if project else [],
            }
        )
    return detail


@router.get("/queue/{assignment_id}/frame/{frame}")
async def review_frame(
    assignment_id: str,
    frame: int,
    _: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    """A single frame image (base64) plus the annotator's shapes drawn on it."""
    a = await db.get(TaskAssignment, assignment_id)
    if not a:
        raise HTTPException(status_code=404, detail="Review item not found")
    try:
        labels = await cvat.get_job_labels(a.cvat_job_id)
        lm = _label_map(labels)
        ann = await cvat.get_job_annotations(a.cvat_job_id)
        meta = await cvat.get_job_meta(a.cvat_job_id)

        shapes = []
        for s in ann.get("shapes", []):
            if s.get("frame") != frame:
                continue
            lab = lm.get(s.get("label_id"), {})
            shapes.append(
                {
                    "type": s.get("type"),
                    "label": lab.get("name", "unknown"),
                    "color": lab.get("color", "#e2553d"),
                    "points": s.get("points", []),
                }
            )

        frames_meta = meta.get("frames", [])
        idx = frame - meta.get("start_frame", 0)
        dim = frames_meta[idx] if 0 <= idx < len(frames_meta) else {}

        img = await cvat.get_frame_bytes(a.cvat_job_id, frame)
        b64 = base64.b64encode(img).decode()
        return {
            "frame": frame,
            "image": f"data:image/jpeg;base64,{b64}",
            "width": dim.get("width", 1280),
            "height": dim.get("height", 720),
            "shapes": shapes,
        }
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502, detail=f"Could not load frame from CVAT: {e}"
        )


async def _act(
    assignment_id: str,
    reviewer: User,
    action: ReviewAction,
    notes: str | None,
    db: AsyncSession,
) -> None:
    assignment = await db.get(TaskAssignment, assignment_id)
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    db.add(
        QualityReview(
            assignment_id=assignment.id,
            reviewer_id=reviewer.id,
            action=action.value,
            notes=notes,
        )
    )

    if action == ReviewAction.approved:
        assignment.status = AssignmentStatus.approved.value
        assignment.completed_at = utcnow()
        # Push the finalized labeled data toward the client dashboard: advance
        # progress, refresh the aggregate quality score, and flag for delivery
        # once every job in the project is approved.
        await _propagate_to_client(assignment, db)
    elif action == ReviewAction.revision_required:
        assignment.status = AssignmentStatus.revision_required.value
        assignment.rework_count += 1
    else:  # rejected
        assignment.status = AssignmentStatus.rejected.value
        # Penalise the annotator's rolling accuracy slightly.
        if assignment.annotator_id:
            profile = (
                await db.execute(
                    select(AnnotatorProfile).where(
                        AnnotatorProfile.user_id == assignment.annotator_id
                    )
                )
            ).scalar_one_or_none()
            if profile:
                profile.rolling_accuracy = max(
                    0, float(profile.rolling_accuracy) - 0.02
                )
        assignment.annotator_id = None  # reassignable

    await db.commit()


async def _propagate_to_client(assignment: TaskAssignment, db: AsyncSession) -> None:
    from app.models.project import ProjectStatus

    project = await db.get(Project, assignment.project_id)
    if not project:
        return

    approved = (
        await db.execute(
            select(TaskAssignment).where(
                TaskAssignment.project_id == project.id,
                TaskAssignment.status == AssignmentStatus.approved.value,
            )
        )
    ).scalars().all()

    project.images_completed = sum(a.frame_count or 0 for a in approved)
    scored = [float(a.iou_score) for a in approved if a.iou_score is not None]
    if scored:
        project.quality_score = round(sum(scored) / len(scored), 4)

    total_jobs = (
        await db.execute(
            select(TaskAssignment).where(TaskAssignment.project_id == project.id)
        )
    ).scalars().all()
    if total_jobs and all(
        a.status == AssignmentStatus.approved.value for a in total_jobs
    ):
        # Everything approved → ready for the client to receive / download.
        project.status = ProjectStatus.review.value


@router.post("/queue/{assignment_id}/approve", response_model=MessageResponse)
async def approve(
    assignment_id: str,
    body: ReviewActionRequest = ReviewActionRequest(),
    reviewer: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    await _act(assignment_id, reviewer, ReviewAction.approved, body.notes, db)
    return MessageResponse(message="Approved")


@router.post("/queue/{assignment_id}/revise", response_model=MessageResponse)
async def revise(
    assignment_id: str,
    body: ReviewActionRequest = ReviewActionRequest(),
    reviewer: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    await _act(assignment_id, reviewer, ReviewAction.revision_required, body.notes, db)
    return MessageResponse(message="Sent back for revision")


@router.post("/queue/{assignment_id}/reject", response_model=MessageResponse)
async def reject(
    assignment_id: str,
    body: ReviewActionRequest = ReviewActionRequest(),
    reviewer: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    await _act(assignment_id, reviewer, ReviewAction.rejected, body.notes, db)
    return MessageResponse(message="Rejected and reassigned")


@router.get("/quotes/pending")
async def reviewer_pending_quotes(
    _: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    """Reviewers can also price counted datasets — same queue as admin."""
    from app.services.quote_review import list_pending_quotes

    return await list_pending_quotes(db)


@router.post("/projects/{project_id}/quote/publish")
async def reviewer_publish_quote(
    project_id: str,
    body: QuotePublishRequest,
    _: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
):
    from app.services.quote_review import publish_project_quote

    return await publish_project_quote(
        db,
        project_id,
        body.avg_objects_per_image,
        body.rate_per_label_inr,
        body.notes,
    )


@router.get("/annotator-scorecards")
async def scorecards(
    _: User = Depends(require_roles(Role.reviewer)),
    db: AsyncSession = Depends(get_db),
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
            "annotator_id": profile.user_id,
            "name": user.full_name,
            "rolling_accuracy": float(profile.rolling_accuracy),
            "rework_rate": float(profile.rework_rate),
            "jobs_completed": profile.total_jobs_completed,
            "trend": "flat",
        }
        for profile, user in rows
    ]
