"""Google Drive intake: a client pastes a share link instead of uploading.

Supports:
  • folder links  — listed recursively via the Drive API v3 (needs GOOGLE_API_KEY;
                    the folder must be shared "anyone with the link")
  • file links    — a ZIP is streamed into R2 so the normal archive pipeline
                    (counting, sampling, CVAT ingestion) takes over

Sample images are pulled via `alt=media` for complexity estimation, same as the
upload path. No API key → the link is stored and ops are pointed at the gap.
"""
from __future__ import annotations

import asyncio
import logging
import re

import httpx

from app.config import settings
from app.database import AsyncSessionLocal
from app.models.project import IntakeStatus, Project
from app.services.dataset_inspector import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    SAMPLE_IMAGES,
    _ext,
    finalize_counts,
    run_intake,
)
from app.services.r2_client import r2

logger = logging.getLogger("annoting.gdrive")

DRIVE_API = "https://www.googleapis.com/drive/v3"

_FOLDER_RE = re.compile(r"drive\.google\.com/drive/(?:u/\d+/)?folders/([\w-]+)")
_FILE_RE = re.compile(r"drive\.google\.com/file/d/([\w-]+)")
_OPEN_RE = re.compile(r"drive\.google\.com/open\?id=([\w-]+)")


class GdriveError(RuntimeError):
    pass


def validate_link(link: str) -> tuple[str, str]:
    """Returns ("folder"|"file", drive_id) or raises GdriveError."""
    if m := _FOLDER_RE.search(link):
        return "folder", m.group(1)
    if m := _FILE_RE.search(link):
        return "file", m.group(1)
    if m := _OPEN_RE.search(link):
        return "file", m.group(1)
    raise GdriveError(
        "That doesn't look like a Google Drive link. Paste a file or folder "
        "share link (drive.google.com/…) set to “anyone with the link”."
    )


async def _list_folder(client: httpx.AsyncClient, folder_id: str) -> list[dict]:
    """All files in a public folder, recursing into subfolders."""
    files: list[dict] = []
    queue = [folder_id]
    while queue:
        fid = queue.pop()
        page_token = None
        while True:
            params = {
                "q": f"'{fid}' in parents and trashed=false",
                "key": settings.google_api_key,
                "fields": "nextPageToken,files(id,name,mimeType,size)",
                "pageSize": 1000,
            }
            if page_token:
                params["pageToken"] = page_token
            r = await client.get(f"{DRIVE_API}/files", params=params)
            if r.status_code == 403:
                raise GdriveError(
                    "Google rejected the request — check GOOGLE_API_KEY and that "
                    "the folder is shared with “anyone with the link”."
                )
            if r.status_code == 404:
                raise GdriveError("Drive folder not found — is the link correct?")
            r.raise_for_status()
            data = r.json()
            for f in data.get("files", []):
                if f.get("mimeType") == "application/vnd.google-apps.folder":
                    queue.append(f["id"])
                else:
                    files.append(f)
            page_token = data.get("nextPageToken")
            if not page_token:
                break
    return files


def _classify(files: list[dict]) -> tuple[int, int, list[dict]]:
    images = videos = 0
    image_files: list[dict] = []
    for f in files:
        mime = f.get("mimeType", "")
        ext = _ext(f.get("name", ""))
        if mime.startswith("image/") or ext in IMAGE_EXTS:
            images += 1
            image_files.append(f)
        elif mime.startswith("video/") or ext in VIDEO_EXTS:
            videos += 1
    return images, videos, image_files


async def _download_samples(
    client: httpx.AsyncClient, image_files: list[dict]
) -> list[bytes]:
    small = [
        f for f in image_files if int(f.get("size", 0) or 0) < 8 * 1024 * 1024
    ]
    step = max(1, len(small) // SAMPLE_IMAGES) if small else 1
    blobs: list[bytes] = []
    for f in small[::step][:SAMPLE_IMAGES]:
        try:
            r = await client.get(
                f"{DRIVE_API}/files/{f['id']}",
                params={"alt": "media", "key": settings.google_api_key},
            )
            if r.status_code == 200:
                blobs.append(r.content)
        except httpx.HTTPError:
            continue
    return blobs


def _transfer_zip_to_r2_sync(file_id: str, project_id: str) -> str | None:
    """Stream a public Drive ZIP into R2 so the archive pipeline takes over."""
    if not r2.configured:
        return None
    url = f"{DRIVE_API}/files/{file_id}?alt=media&key={settings.google_api_key}"
    key = f"projects/{project_id}/gdrive-dataset.zip"
    with httpx.Client(timeout=None, follow_redirects=True) as client:
        with client.stream("GET", url) as resp:
            if resp.status_code != 200:
                return None
            r2._get_client().upload_fileobj(  # noqa: SLF001 — same package
                _HttpxStreamReader(resp), settings.r2_bucket_name, key
            )
    return key


class _HttpxStreamReader:
    """Minimal file-like wrapper so boto3 can multipart-upload an HTTP stream."""

    def __init__(self, resp: httpx.Response) -> None:
        self._iter = resp.iter_bytes(chunk_size=1024 * 1024)
        self._buf = b""

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            return self._buf + b"".join(self._iter)
        while len(self._buf) < size:
            try:
                self._buf += next(self._iter)
            except StopIteration:
                break
        out, self._buf = self._buf[:size], self._buf[size:]
        return out


async def run_gdrive_intake(project_id: str, link: str) -> None:
    """Background task: resolve the Drive link, count media, quote."""
    async with AsyncSessionLocal() as db:
        project = await db.get(Project, project_id)
        if not project:
            return
        try:
            kind, drive_id = validate_link(link)

            if not settings.google_api_key:
                project.intake_status = IntakeStatus.awaiting_data.value
                project.intake_detail = (
                    "Drive link saved. Automatic counting needs GOOGLE_API_KEY "
                    "on the server — ops will process this link manually."
                )
                await db.commit()
                return

            async with httpx.AsyncClient(timeout=60) as client:
                if kind == "folder":
                    files = await _list_folder(client, drive_id)
                    images, videos, image_files = _classify(files)
                    samples = await _download_samples(client, image_files)
                    await finalize_counts(db, project, images, videos, samples)
                else:
                    # Single file: a ZIP gets streamed to R2, then the normal
                    # archive inspection runs against storage.
                    meta = await client.get(
                        f"{DRIVE_API}/files/{drive_id}",
                        params={
                            "key": settings.google_api_key,
                            "fields": "id,name,mimeType,size",
                        },
                    )
                    if meta.status_code != 200:
                        raise GdriveError(
                            "Could not read that Drive file — is it shared with "
                            "“anyone with the link”?"
                        )
                    info = meta.json()
                    name = info.get("name", "")
                    if _ext(name) == ".zip" or info.get("mimeType") in (
                        "application/zip",
                        "application/x-zip-compressed",
                    ):
                        key = await asyncio.to_thread(
                            _transfer_zip_to_r2_sync, drive_id, project.id
                        )
                        if key:
                            project.r2_dataset_prefix = f"projects/{project.id}/"
                            await db.commit()
                            await run_intake(project.id)
                            return
                        raise GdriveError(
                            "Could not transfer the ZIP from Drive to storage."
                        )
                    # A single loose media file.
                    images, videos, image_files = _classify([info])
                    samples = await _download_samples(client, image_files)
                    await finalize_counts(db, project, images, videos, samples)
        except GdriveError as e:
            project.intake_status = IntakeStatus.awaiting_data.value
            project.intake_detail = str(e)
            await db.commit()
        except Exception as e:  # noqa: BLE001 — background task must not raise
            logger.exception("gdrive intake failed for %s", project_id)
            project.intake_status = IntakeStatus.awaiting_data.value
            project.intake_detail = f"Drive intake failed: {e}"
            await db.commit()
