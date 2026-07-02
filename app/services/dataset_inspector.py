"""Automatic dataset intake: as soon as a client's data lands (R2 upload or
Google Drive link), count the images/videos, estimate per-image object density
from a sample, and generate a firm quote — all without a human touching it.

Counting reads only the ZIP central directory via ranged R2 reads (a few KB),
so a 5 GB archive is inspected in seconds without downloading it.

Everything degrades gracefully in dev: no R2 → fall back to client-declared
counts; no Pillow → fall back to the annotation type's default density.
"""
from __future__ import annotations

import asyncio
import io
import logging
import zipfile

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.models.billing import ProjectQuote
from app.models.common import utcnow
from app.models.project import IntakeStatus, MediaType, Project
from app.services.pricing_engine import calculate_quote
from app.services.r2_client import r2

logger = logging.getLogger("annoting.intake")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

# How many sample images to pull out of the archive for density estimation.
SAMPLE_IMAGES = 6
# Only sample images below this size (keeps ranged reads cheap).
SAMPLE_MAX_BYTES = 8 * 1024 * 1024


def _ext(name: str) -> str:
    dot = name.rfind(".")
    return name[dot:].lower() if dot != -1 else ""


def _is_junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return name.endswith("/") or base.startswith((".", "__MACOSX", "._"))


class _R2RangedFile(io.RawIOBase):
    """Seekable read-only file over an R2 object using HTTP Range requests.

    zipfile only reads the end-of-central-directory + per-entry headers, so
    opening a huge archive costs a handful of small ranged GETs.
    """

    def __init__(self, key: str, size: int) -> None:
        self._key = key
        self._size = size
        self._pos = 0

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        return self._pos

    def tell(self) -> int:
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            size = self._size - self._pos
        if size <= 0 or self._pos >= self._size:
            return b""
        end = min(self._pos + size, self._size) - 1
        data = r2.read_range(self._key, self._pos, end)
        self._pos += len(data)
        return data


def _estimate_density_from_images(blobs: list[bytes]) -> float | None:
    """Edge-density heuristic → multiplier on the type's default objects/image.

    More edges and texture in a frame correlates with more distinct objects to
    draw. Returns a multiplier in [0.6, 2.2], or None if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageFilter  # optional dependency
    except ImportError:
        return None

    scores: list[float] = []
    for blob in blobs:
        try:
            img = Image.open(io.BytesIO(blob)).convert("L").resize((160, 160))
            edges = img.filter(ImageFilter.FIND_EDGES)
            hist = edges.histogram()
            total = sum(hist) or 1
            # Fraction of pixels with a strong edge response.
            strong = sum(hist[48:]) / total
            scores.append(strong)
        except Exception:  # noqa: BLE001 — corrupt sample, skip it
            continue
    if not scores:
        return None
    avg = sum(scores) / len(scores)
    # avg ~0.02 = near-empty scenes, ~0.25+ = dense/cluttered scenes.
    return max(0.6, min(2.2, 0.6 + avg * 6.5))


def _tier(multiplier: float) -> str:
    if multiplier < 0.85:
        return "simple"
    if multiplier <= 1.4:
        return "moderate"
    return "dense"


def _inspect_r2_sync(prefix: str) -> dict:
    """Blocking part: list the prefix, walk archive central directory, sample
    images. Runs in a thread so the event loop stays free."""
    images = 0
    videos = 0
    sample_blobs: list[bytes] = []

    keys = r2.list_keys(prefix)
    archive_key = r2.find_dataset_archive(prefix)

    # Loose files uploaded directly (not archived).
    for key in keys:
        if key == archive_key:
            continue
        ext = _ext(key)
        if ext in IMAGE_EXTS:
            images += 1
        elif ext in VIDEO_EXTS:
            videos += 1

    if archive_key:
        size = r2.object_size(archive_key)
        if size and archive_key.lower().endswith(".zip"):
            with zipfile.ZipFile(_R2RangedFile(archive_key, size)) as zf:  # type: ignore[arg-type]
                infos = [i for i in zf.infolist() if not _is_junk(i.filename)]
                image_infos = []
                for info in infos:
                    ext = _ext(info.filename)
                    if ext in IMAGE_EXTS:
                        images += 1
                        image_infos.append(info)
                    elif ext in VIDEO_EXTS:
                        videos += 1
                # Evenly spaced sample across the archive for density estimation.
                candidates = [
                    i for i in image_infos if i.file_size <= SAMPLE_MAX_BYTES
                ]
                if candidates:
                    step = max(1, len(candidates) // SAMPLE_IMAGES)
                    for info in candidates[::step][:SAMPLE_IMAGES]:
                        try:
                            sample_blobs.append(zf.read(info))
                        except Exception:  # noqa: BLE001
                            continue

    return {"images": images, "videos": videos, "samples": sample_blobs}


async def run_intake(project_id: str, declared_count: int = 0) -> None:
    """Full intake pass for one project. Safe to call repeatedly; each run
    re-counts and refreshes the open (unaccepted) quote."""
    async with AsyncSessionLocal() as db:
        project = await db.get(Project, project_id)
        if not project:
            return
        project.intake_status = IntakeStatus.counting.value
        project.intake_detail = None
        await db.commit()

        try:
            await _run_intake_inner(db, project, declared_count)
        except Exception as e:  # noqa: BLE001 — never crash the background task
            logger.exception("intake failed for project %s", project_id)
            project.intake_status = IntakeStatus.awaiting_data.value
            project.intake_detail = f"Intake failed: {e}"
            await db.commit()


async def _run_intake_inner(db, project: Project, declared_count: int) -> None:
    prefix = project.r2_dataset_prefix or f"projects/{project.id}/"

    if r2.configured:
        result = await asyncio.to_thread(_inspect_r2_sync, prefix)
    else:
        # Dev fallback: trust what the client told us so the flow still demos.
        result = {"images": declared_count, "videos": 0, "samples": []}

    images, videos = result["images"], result["videos"]
    if images == 0 and videos == 0 and declared_count:
        images = declared_count

    await finalize_counts(db, project, images, videos, result["samples"])


async def finalize_counts(
    db,
    project: Project,
    images: int,
    videos: int,
    samples: list[bytes],
) -> None:
    """Shared tail of every intake path (R2 upload, Drive link): store counts,
    estimate complexity from samples, and generate/refresh the quote."""
    project.image_count = images
    project.video_count = videos
    if images and videos:
        project.media_type = MediaType.mixed.value
    elif videos:
        project.media_type = MediaType.videos.value
    elif images:
        project.media_type = MediaType.images.value
    project.total_images = images + videos

    # ── Complexity: sampled edge density → objects-per-image estimate ────────
    from app.services.pricing_engine import RATE_CARD

    card = RATE_CARD.get(project.annotation_type, RATE_CARD["bbox"])
    multiplier = _estimate_density_from_images(samples)
    if multiplier is not None:
        avg_objects = round(card["avg_objects"] * multiplier, 2)
        project.complexity_tier = _tier(multiplier)
    else:
        avg_objects = card["avg_objects"]
        project.complexity_tier = "moderate"
    project.estimated_objects_per_image = avg_objects
    project.intake_status = IntakeStatus.counted.value

    detail_bits = [f"{images} images" if images else "", f"{videos} videos" if videos else ""]
    project.intake_detail = (
        f"Detected {' + '.join(b for b in detail_bits if b) or 'no media'} · "
        f"~{avg_objects} objects/image ({project.complexity_tier})"
    )

    # ── Draft quote from real counts — an Annoting admin reviews the dataset
    # and publishes (possibly adjusting density/rate) before the client sees it.
    if project.total_images > 0:
        quote = calculate_quote(
            project.annotation_type,
            project.total_images,
            avg_objects,
            project.turnaround_days or 14,
        )
        # Refresh the open draft if the client hasn't accepted one yet.
        existing = (
            await db.execute(
                select(ProjectQuote)
                .where(ProjectQuote.project_id == project.id)
                .order_by(ProjectQuote.created_at.desc())
            )
        ).scalars().first()
        if existing and existing.accepted_at is None:
            existing.rate_per_label_inr = quote.rate_per_label_inr
            existing.estimated_labels = quote.estimated_labels
            existing.quoted_total_inr = quote.estimated_total_inr
            existing.turnaround_premium_pct = quote.turnaround_premium_pct
            existing.volume_discount_pct = quote.volume_discount_pct
            existing.published_at = None  # counts changed → re-review
            existing.created_at = utcnow()
        elif not existing or existing.accepted_at is None:
            db.add(
                ProjectQuote(
                    project_id=project.id,
                    annotation_type=project.annotation_type,
                    rate_per_label_inr=quote.rate_per_label_inr,
                    estimated_labels=quote.estimated_labels,
                    quoted_total_inr=quote.estimated_total_inr,
                    turnaround_premium_pct=quote.turnaround_premium_pct,
                    volume_discount_pct=quote.volume_discount_pct,
                )
            )
        project.intake_status = IntakeStatus.pending_review.value
        project.intake_detail = (
            (project.intake_detail or "")
            + " · Our team is reviewing your dataset to finalise the quote."
        )

    await db.commit()
