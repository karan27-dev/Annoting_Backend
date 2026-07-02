"""Seed demo data so every dashboard shows realistic content.

Run from backend/:  python seed.py
Logins (password for all):  annoting123
  admin@annoting.com      · super_admin
  client@annoting.com     · client (has projects + invoice)
  annotator@annoting.com  · annotator (certified, with jobs)
  reviewer@annoting.com   · reviewer (has a queue)
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

from app.core.security import hash_password
from app.database import AsyncSessionLocal, init_db
from app.models.assignment import AssignmentStatus, TaskAssignment
from app.models.billing import Invoice, InvoiceStatus
from app.models.common import utcnow
from app.models.project import Project, ProjectStatus
from app.models.user import (
    AnnotatorProfile,
    AnnotatorStatus,
    Client,
    Role,
    User,
)
from sqlalchemy import select

PW = hash_password("annoting123")


async def upsert_user(db, email, name, role, verified=True) -> User:
    user = (
        await db.execute(select(User).where(User.email == email))
    ).scalar_one_or_none()
    if user:
        return user
    user = User(
        email=email,
        password_hash=PW,
        full_name=name,
        role=role.value,
        is_verified=verified,
    )
    db.add(user)
    await db.flush()
    return user


async def main() -> None:
    await init_db()
    async with AsyncSessionLocal() as db:
        # ── Users ──────────────────────────────────────────────────────────────
        admin = await upsert_user(db, "admin@annoting.com", "Ops Admin", Role.super_admin)
        reviewer = await upsert_user(
            db, "reviewer@annoting.com", "Riya Reviewer", Role.reviewer
        )
        client_user = await upsert_user(
            db, "client@annoting.com", "Karan Mehta", Role.client
        )
        annot_user = await upsert_user(
            db, "annotator@annoting.com", "Aanya Sharma", Role.annotator
        )

        # ── Client profile ───────────────────────────────────────────────────────
        client = (
            await db.execute(select(Client).where(Client.user_id == client_user.id))
        ).scalar_one_or_none()
        if not client:
            client = Client(
                user_id=client_user.id, company_name="Acme Vision Inc.", tier="priority"
            )
            db.add(client)
            await db.flush()

        # ── Annotator profile (certified) ──────────────────────────────────────────
        profile = (
            await db.execute(
                select(AnnotatorProfile).where(
                    AnnotatorProfile.user_id == annot_user.id
                )
            )
        ).scalar_one_or_none()
        if not profile:
            db.add(
                AnnotatorProfile(
                    user_id=annot_user.id,
                    skills={
                        "bbox": True,
                        "polygon": True,
                        "segmentation": False,
                        "keypoint": False,
                        "classification": True,
                    },
                    calibration_passed_at=utcnow(),
                    calibration_score=0.88,
                    rolling_accuracy=0.91,
                    rework_rate=0.07,
                    total_jobs_completed=42,
                    total_labels_completed=12800,
                    status=AnnotatorStatus.active.value,
                    city="Pune",
                    cvat_user_id=2,
                )
            )

        # ── Projects ───────────────────────────────────────────────────────────────
        existing = (
            await db.execute(select(Project).where(Project.client_id == client.id))
        ).scalars().all()
        if not existing:
            p1 = Project(
                client_id=client.id,
                name="Street scenes — vehicle detection",
                description="Label every car, truck and pedestrian.",
                annotation_type="bbox",
                label_taxonomy=[
                    {"name": "car", "color": "#e2553d"},
                    {"name": "truck", "color": "#c98a17"},
                    {"name": "person", "color": "#2f8f5b"},
                ],
                total_images=5000,
                images_completed=3200,
                quality_score=0.91,
                status=ProjectStatus.active.value,
                turnaround_days=14,
            )
            p2 = Project(
                client_id=client.id,
                name="Retail shelf — product polygons",
                annotation_type="polygon",
                label_taxonomy=[{"name": "product", "color": "#3b6ea5"}],
                total_images=1200,
                images_completed=1200,
                quality_score=0.94,
                status=ProjectStatus.delivered.value,
                turnaround_days=7,
                delivered_at=utcnow() - timedelta(days=2),
            )
            p3 = Project(
                client_id=client.id,
                name="Drone survey — building segmentation",
                annotation_type="segmentation",
                label_taxonomy=[{"name": "building", "color": "#8b5cf6"}],
                total_images=800,
                images_completed=0,
                status=ProjectStatus.pending_setup.value,
                turnaround_days=21,
            )
            db.add_all([p1, p2, p3])
            await db.flush()

            # A couple of review-queue items + an available job.
            db.add_all(
                [
                    TaskAssignment(
                        project_id=p1.id,
                        cvat_job_id=101,
                        cvat_task_id=11,
                        annotator_id=annot_user.id,
                        status=AssignmentStatus.review_pending.value,
                        iou_score=0.72,
                        labels_count=180,
                        flag_reason="auto-rejected",
                        submitted_at=utcnow() - timedelta(hours=6),
                    ),
                    TaskAssignment(
                        project_id=p1.id,
                        cvat_job_id=102,
                        cvat_task_id=11,
                        annotator_id=None,
                        status=AssignmentStatus.assigned.value,
                    ),
                ]
            )

            # Delivered project -> invoice
            db.add(
                Invoice(
                    client_id=client.id,
                    project_id=p2.id,
                    invoice_number=f"SYL-{utcnow().year}-0001",
                    amount_inr=72000,
                    gst_amount_inr=12960,
                    total_inr=84960,
                    status=InvoiceStatus.sent.value,
                    issued_at=utcnow() - timedelta(days=2),
                    due_at=utcnow() + timedelta(days=13),
                )
            )

        await db.commit()
    print("✓ Seed complete. Login with any *@annoting.com / password: annoting123")


if __name__ == "__main__":
    asyncio.run(main())
