"""R2 → CVAT ingestion: push a project's uploaded dataset into CVAT, wait for it
to extract, then mirror the real CVAT jobs into our task_assignments table.

This is the backbone that makes uploaded images actually appear in the annotation
canvas. It talks to CVAT only via its REST API.
"""
from __future__ import annotations

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.assignment import AssignmentStatus, TaskAssignment
from app.models.common import utcnow
from app.models.project import CvatMapping, Project
from app.services.cvat_client import cvat
from app.services.r2_client import r2


class IngestionError(RuntimeError):
    pass


async def ingest_project(db: AsyncSession, project: Project) -> dict:
    """Full pipeline for one project. Returns a summary dict.

    Idempotent-ish: if assignments already exist we skip re-ingesting.
    """
    if not cvat.configured:
        raise IngestionError("CVAT is not configured")
    if not r2.configured:
        raise IngestionError("R2 is not configured")

    prefix = project.r2_dataset_prefix or f"projects/{project.id}/"
    archive_key = r2.find_dataset_archive(prefix)
    if not archive_key:
        raise IngestionError(
            "No dataset archive found in storage for this project — upload a ZIP first."
        )

    mapping = (
        await db.execute(
            select(CvatMapping).where(CvatMapping.project_id == project.id)
        )
    ).scalar_one_or_none()

    # 1. Ensure a CVAT project + task exist.
    labels = [
        {"name": lc["name"], "color": lc.get("color", "#e2553d")}
        for lc in (project.label_taxonomy or [])
    ] or [{"name": "object", "color": "#e2553d"}]

    if mapping and mapping.cvat_task_ids:
        cvat_task_id = mapping.cvat_task_ids[0]
        cvat_project_id = mapping.cvat_project_id
    else:
        cproj = await cvat.create_project(project.name, labels)
        cvat_project_id = cproj["id"]
        # Default: no splitting — the full dataset stays together as one job.
        seg = mapping.segment_size if mapping else 0
        ctask = await cvat.create_task(project.name, cvat_project_id, seg)
        cvat_task_id = ctask["id"]
        if mapping:
            mapping.cvat_project_id = cvat_project_id
            mapping.cvat_task_ids = [cvat_task_id]
        else:
            from app.services.export_formats import cvat_format_for

            mapping = CvatMapping(
                project_id=project.id,
                cvat_project_id=cvat_project_id,
                cvat_task_ids=[cvat_task_id],
                export_format=cvat_format_for(project.delivery_format),
            )
            db.add(mapping)
        await db.flush()

    # 2. Kick off data ingestion from the R2 archive (presigned URL).
    url = r2.presign_get(archive_key, expires=7200)
    await cvat.attach_data_remote(cvat_task_id, url)

    # 3. Wait for CVAT to finish extracting (bounded).
    state = "Queued"
    for _ in range(120):  # up to ~10 min
        status = await cvat.task_status(cvat_task_id)
        state = status.get("state")
        if state == "Finished":
            break
        if state == "Failed":
            raise IngestionError(
                f"CVAT failed to process the dataset: {status.get('message')}"
            )
        await asyncio.sleep(5)
    if state != "Finished":
        raise IngestionError("CVAT is still processing — try again shortly.")

    # 4. Real frame count + jobs from CVAT.
    task = await cvat.get_task(cvat_task_id)
    project.total_images = task.get("size", project.total_images)

    jobs = await cvat.list_task_jobs(cvat_task_id)

    # Remove any placeholder segments from before real CVAT ingestion existed.
    placeholders = (
        await db.execute(
            select(TaskAssignment).where(
                TaskAssignment.project_id == project.id,
                TaskAssignment.cvat_job_id >= 900_000,
            )
        )
    ).scalars().all()
    for ph in placeholders:
        await db.delete(ph)
    await db.flush()

    # 5. Mirror CVAT jobs into task_assignments (skip ones we already have).
    existing = {
        a.cvat_job_id
        for a in (
            await db.execute(
                select(TaskAssignment).where(
                    TaskAssignment.project_id == project.id
                )
            )
        ).scalars().all()
    }
    created = 0
    for j in jobs:
        if j["id"] in existing:
            continue
        frames = (j.get("stop_frame", 0) - j.get("start_frame", 0)) + 1
        db.add(
            TaskAssignment(
                project_id=project.id,
                cvat_job_id=j["id"],
                cvat_task_id=cvat_task_id,
                frame_count=frames,
                status=AssignmentStatus.assigned.value,
            )
        )
        created += 1

    mapping.last_synced_at = utcnow()
    await db.commit()

    return {
        "cvat_task_id": cvat_task_id,
        "total_images": project.total_images,
        "jobs_created": created,
        "jobs_total": len(jobs),
    }
