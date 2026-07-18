"""Self-serve dataset API — users upload images, label them in the in-app
editor, and their annotations build a dataset they can view and export.

This is the Roboflow-style path: no CVAT, no annotators, no quote. The owner
labels their own data and it's stored directly in our DB + R2.
"""
from __future__ import annotations

import asyncio
import io
import uuid

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models.dataset import DatasetImage, ImageStatus
from app.models.project import Project
from app.models.user import Client, Role, User
from app.schemas.dataset import (
    AnnotationsSave,
    ClassCount,
    DatasetSummary,
    ImageDetail,
    ImageOut,
    Shape,
)
from app.schemas.misc import MessageResponse
from app.services.r2_client import r2

router = APIRouter(prefix="/datasets", tags=["datasets"])

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


async def _owned_project(project_id: str, user: User, db: AsyncSession) -> Project:
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if user.role in (Role.super_admin.value, Role.ops_manager.value):
        return project
    client = (
        await db.execute(select(Client).where(Client.user_id == user.id))
    ).scalar_one_or_none()
    if not client or project.client_id != client.id:
        raise HTTPException(status_code=403, detail="Not your project")
    return project


def _label_defs(project: Project) -> list[dict]:
    return [
        {"name": lc["name"], "color": lc.get("color", "#e2553d")}
        for lc in (project.label_taxonomy or [])
    ] or [{"name": "object", "color": "#e2553d"}]


def _probe_size(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.width, im.height
    except Exception:  # noqa: BLE001
        return 0, 0


def _ext(name: str) -> str:
    dot = name.rfind(".")
    return name[dot:].lower() if dot != -1 else ""


@router.post("/{project_id}/images")
async def upload_images(
    project_id: str,
    files: list[UploadFile] = File(...),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload one or many images straight into the dataset (backend-proxied to
    R2, so no browser CORS). Each becomes an unlabeled DatasetImage."""
    project = await _owned_project(project_id, user, db)

    start_index = (
        await db.execute(
            select(func.count(DatasetImage.id)).where(
                DatasetImage.project_id == project.id
            )
        )
    ).scalar() or 0

    created = 0
    for i, up in enumerate(files):
        if _ext(up.filename or "") not in IMAGE_EXTS:
            continue
        data = await up.read()
        if not data:
            continue
        w, h = _probe_size(data)
        key = f"projects/{project.id}/dataset/{uuid.uuid4().hex}{_ext(up.filename or '.jpg')}"
        if r2.configured:
            try:
                await asyncio.to_thread(
                    r2.put_fileobj, key, io.BytesIO(data), up.content_type or "image/jpeg"
                )
            except Exception as e:  # noqa: BLE001
                raise HTTPException(status_code=502, detail=f"Storage failed: {e}")
        db.add(
            DatasetImage(
                project_id=project.id,
                filename=up.filename or f"image_{start_index + i}",
                r2_key=key,
                width=w,
                height=h,
                order_index=start_index + created,
                status=ImageStatus.unlabeled.value,
                annotations=[],
            )
        )
        created += 1

    # Keep the project's headline count in sync (new rows are already flushed).
    total = (
        await db.execute(
            select(func.count(DatasetImage.id)).where(
                DatasetImage.project_id == project.id
            )
        )
    ).scalar() or 0
    project.total_images = total
    project.image_count = total
    await db.commit()

    return {"uploaded": created, "total_images": total}


@router.get("/{project_id}", response_model=DatasetSummary)
async def dataset_summary(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(project_id, user, db)
    images = (
        await db.execute(
            select(DatasetImage).where(DatasetImage.project_id == project.id)
        )
    ).scalars().all()

    labels = _label_defs(project)
    color_of = {ld["name"]: ld["color"] for ld in labels}
    counts: dict[str, int] = {ld["name"]: 0 for ld in labels}
    total_boxes = 0
    labeled = 0
    splits: dict[str, int] = {"train": 0, "valid": 0, "test": 0}
    for img in images:
        if img.status in (ImageStatus.labeled.value, ImageStatus.approved.value):
            labeled += 1
        splits[img.split] = splits.get(img.split, 0) + 1
        for sh in img.annotations or []:
            total_boxes += 1
            nm = sh.get("label", "object")
            counts[nm] = counts.get(nm, 0) + 1
            color_of.setdefault(nm, "#8b857c")

    return DatasetSummary(
        project_id=project.id,
        name=project.name,
        annotation_type=project.annotation_type,
        mode=project.mode,
        total_images=len(images),
        labeled=labeled,
        unlabeled=len(images) - labeled,
        total_boxes=total_boxes,
        classes=[
            ClassCount(name=n, color=color_of.get(n, "#8b857c"), count=c)
            for n, c in sorted(counts.items(), key=lambda kv: -kv[1])
        ],
        splits=splits,
    )


@router.get("/{project_id}/images", response_model=list[ImageOut])
async def list_images(
    project_id: str,
    status: str | None = Query(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(project_id, user, db)
    q = select(DatasetImage).where(DatasetImage.project_id == project.id)
    if status in (ImageStatus.labeled.value, ImageStatus.unlabeled.value):
        q = q.where(DatasetImage.status == status)
    q = q.order_by(DatasetImage.order_index)
    images = (await db.execute(q)).scalars().all()
    return [
        ImageOut(
            id=img.id,
            filename=img.filename,
            url=r2.presign_get(img.r2_key, expires=3600),
            width=img.width,
            height=img.height,
            status=img.status,
            box_count=len(img.annotations or []),
            split=img.split,
        )
        for img in images
    ]


async def _ordered_ids(project_id: str, db: AsyncSession) -> list[str]:
    rows = (
        await db.execute(
            select(DatasetImage.id)
            .where(DatasetImage.project_id == project_id)
            .order_by(DatasetImage.order_index)
        )
    ).all()
    return [r[0] for r in rows]


@router.get("/{project_id}/images/{image_id}", response_model=ImageDetail)
async def image_detail(
    project_id: str,
    image_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(project_id, user, db)
    img = await db.get(DatasetImage, image_id)
    if not img or img.project_id != project.id:
        raise HTTPException(status_code=404, detail="Image not found")

    ids = await _ordered_ids(project.id, db)
    idx = ids.index(img.id) if img.id in ids else 0
    return ImageDetail(
        id=img.id,
        filename=img.filename,
        url=r2.presign_get(img.r2_key, expires=3600),
        width=img.width,
        height=img.height,
        status=img.status,
        split=img.split,
        annotations=[Shape(**s) for s in (img.annotations or [])],
        labels=_label_defs(project),
        index=idx,
        total=len(ids),
        next_id=ids[idx + 1] if idx + 1 < len(ids) else None,
        prev_id=ids[idx - 1] if idx > 0 else None,
    )


@router.put("/{project_id}/images/{image_id}/annotations", response_model=MessageResponse)
async def save_annotations(
    project_id: str,
    image_id: str,
    body: AnnotationsSave,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(project_id, user, db)
    img = await db.get(DatasetImage, image_id)
    if not img or img.project_id != project.id:
        raise HTTPException(status_code=404, detail="Image not found")

    img.annotations = [s.model_dump() for s in body.annotations]
    if body.mark_labeled:
        img.status = (
            ImageStatus.labeled.value
            if img.annotations
            else ImageStatus.unlabeled.value
        )
    await db.commit()
    return MessageResponse(message="Saved")


@router.delete("/{project_id}/images/{image_id}", response_model=MessageResponse)
async def delete_image(
    project_id: str,
    image_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(project_id, user, db)
    img = await db.get(DatasetImage, image_id)
    if not img or img.project_id != project.id:
        raise HTTPException(status_code=404, detail="Image not found")
    await db.delete(img)
    await db.commit()
    return MessageResponse(message="Deleted")


@router.get("/{project_id}/export")
async def export_dataset(
    project_id: str,
    fmt: str = Query("coco", pattern="^(coco|yolo)$"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export labeled images. COCO returns a single JSON; YOLO returns a map of
    filename → label lines. Denormalizes coords back to pixels for COCO."""
    project = await _owned_project(project_id, user, db)
    labels = _label_defs(project)
    label_index = {ld["name"]: i for i, ld in enumerate(labels)}

    images = (
        await db.execute(
            select(DatasetImage)
            .where(DatasetImage.project_id == project.id)
            .order_by(DatasetImage.order_index)
        )
    ).scalars().all()

    if fmt == "coco":
        coco = {
            "info": {"description": project.name, "version": "1.0"},
            "categories": [
                {"id": i, "name": ld["name"], "supercategory": "none"}
                for i, ld in enumerate(labels)
            ],
            "images": [],
            "annotations": [],
        }
        ann_id = 1
        for img in images:
            coco["images"].append(
                {
                    "id": img.order_index,
                    "file_name": img.filename,
                    "width": img.width,
                    "height": img.height,
                }
            )
            for sh in img.annotations or []:
                w = sh["w"] * img.width
                h = sh["h"] * img.height
                x = sh["x"] * img.width
                y = sh["y"] * img.height
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": img.order_index,
                        "category_id": label_index.get(sh["label"], 0),
                        "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                        "area": round(w * h, 1),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
        return coco

    # YOLO: per-image "class cx cy w h" (already normalized centre form)
    out: dict[str, str] = {}
    for img in images:
        lines = []
        for sh in img.annotations or []:
            cls = label_index.get(sh["label"], 0)
            cx = sh["x"] + sh["w"] / 2
            cy = sh["y"] + sh["h"] / 2
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {sh['w']:.6f} {sh['h']:.6f}")
        stem = img.filename.rsplit(".", 1)[0]
        out[f"{stem}.txt"] = "\n".join(lines)
    return {
        "classes": [ld["name"] for ld in labels],
        "labels": out,
    }
