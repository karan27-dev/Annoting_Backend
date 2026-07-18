from __future__ import annotations

import asyncio

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
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


@router.post("/direct", response_model=MessageResponse)
async def direct_upload(
    background: BackgroundTasks,
    project_id: str = Form(...),
    file: UploadFile = File(...),
    _: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Backend-proxied upload: the browser sends the file here and we stream it
    to R2 server-side. This avoids the browser→R2 CORS requirement entirely, so
    uploads work out of the box. FastAPI spools large files to disk, so big ZIPs
    don't blow up memory."""
    project = await db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    key = f"projects/{project_id}/{file.filename}"
    content_type = file.content_type or "application/zip"

    if r2.configured:
        try:
            # boto3 is blocking — run it off the event loop.
            await asyncio.to_thread(r2.put_fileobj, key, file.file, content_type)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(
                status_code=502, detail=f"Storage upload failed: {e}"
            )

    project.r2_dataset_prefix = f"projects/{project_id}/"
    project.data_source = DataSource.upload.value
    project.intake_status = IntakeStatus.counting.value
    await db.commit()

    background.add_task(run_intake, project_id, 0)
    return MessageResponse(
        message="Upload received — counting your images and videos now."
    )


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
