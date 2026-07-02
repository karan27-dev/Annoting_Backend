"""Annotator-to-job matching.

Score = 0.7 * rolling_accuracy + 0.3 * (1 / (1 + rework_rate)).
Only 'active', calibrated annotators certified for the task type are considered,
and annotators at/over their concurrent-job cap are filtered out.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.assignment import AssignmentStatus, TaskAssignment
from app.models.user import AnnotatorProfile, AnnotatorStatus

MAX_CONCURRENT_JOBS = 3


def _score(accuracy: float, rework_rate: float) -> float:
    return 0.7 * float(accuracy) + 0.3 * (1.0 / (1.0 + float(rework_rate)))


async def eligible_annotators(
    db: AsyncSession, annotation_type: str
) -> list[AnnotatorProfile]:
    rows = (
        await db.execute(
            select(AnnotatorProfile).where(
                AnnotatorProfile.status == AnnotatorStatus.active.value,
                AnnotatorProfile.calibration_passed_at.is_not(None),
            )
        )
    ).scalars().all()
    return [p for p in rows if (p.skills or {}).get(annotation_type)]


async def _active_job_count(db: AsyncSession, annotator_id: str) -> int:
    rows = (
        await db.execute(
            select(TaskAssignment).where(
                TaskAssignment.annotator_id == annotator_id,
                TaskAssignment.status.in_(
                    [
                        AssignmentStatus.assigned.value,
                        AssignmentStatus.in_progress.value,
                    ]
                ),
            )
        )
    ).scalars().all()
    return len(rows)


async def pick_best_annotator(
    db: AsyncSession, project_id: str, annotation_type: str
) -> AnnotatorProfile | None:
    candidates = await eligible_annotators(db, annotation_type)

    # Annotators who already worked this project (avoid anchoring bias).
    worked = {
        a.annotator_id
        for a in (
            await db.execute(
                select(TaskAssignment).where(TaskAssignment.project_id == project_id)
            )
        ).scalars().all()
    }

    scored: list[tuple[float, AnnotatorProfile]] = []
    for p in candidates:
        if await _active_job_count(db, p.user_id) >= MAX_CONCURRENT_JOBS:
            continue
        if p.user_id in worked:
            continue
        scored.append((_score(p.rolling_accuracy, p.rework_rate), p))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]
