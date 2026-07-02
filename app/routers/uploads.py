from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models.project import DataSource, IntakeStatus, Project
from app.models.user import User
from app.schemas.misc import MessageResponse
from app.schemas.project import (
    GdriveLinkRequest,
    PresignedUrlRequest,
    PresignedUrlResponse,
    UploadConfirmRequest,
)
from app.services.dataset_inspector import run_intake
from app.services.gdrive import GdriveError, run_gdrive_intake, validate_link
from app.services.r2_client import r2

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("/presigned-url", response_model=PresignedUrlResponse)
async def presigned_url(
    body: PresignedUrlRequest, _: User = Depends(get_current_user)
):
    # The file uploads directly browser -> R2; it never touches our server.
    key = f"projects/{body.project_id}/{body.filename}"
    url = r2.presign_put(key, body.content_type)
    return PresignedUrlResponse(upload_url=url, r2_key=key)


@router.post("/confirm", response_model=MessageResponse)
async def confirm_upload(
    body: UploadConfirmRequest,
    background: BackgroundTasks,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, body.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project.r2_dataset_prefix = f"projects/{body.project_id}/"
    project.data_source = DataSource.upload.value
    project.intake_status = IntakeStatus.counting.value
    await db.commit()

    # Count assets + estimate complexity + generate the quote, off-request.
    background.add_task(run_intake, body.project_id, body.file_count or 0)
    return MessageResponse(
        message="Upload received — counting your images and videos now."
    )


@router.post("/gdrive", response_model=MessageResponse)
async def link_gdrive(
    body: GdriveLinkRequest,
    background: BackgroundTasks,
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Client shares a Google Drive file/folder link instead of uploading."""
    project = await db.get(Project, body.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        validate_link(body.link)
    except GdriveError as e:
        raise HTTPException(status_code=400, detail=str(e))

    project.gdrive_link = body.link
    project.data_source = DataSource.gdrive.value
    project.intake_status = IntakeStatus.counting.value
    await db.commit()

    background.add_task(run_gdrive_intake, body.project_id, body.link)
    return MessageResponse(
        message="Drive link received — fetching your file list now."
    )
