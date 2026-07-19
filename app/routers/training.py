"""Self-serve model training.

Owner-facing: create dataset versions, start a training job, poll progress,
list trained models.

Trainer-facing (token auth — runs in the user's Google Colab, NOT our infra):
  GET  /training/jobs/{id}/script?token=…   → ready-to-run python trainer
  GET  /training/jobs/{id}/data?token=…     → classes + image URLs + YOLO labels
  POST /training/jobs/{id}/events?token=…   → epoch metrics / final results
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.models.common import utcnow
from app.models.dataset import DatasetImage, ImageStatus
from app.models.project import Project
from app.models.training import DatasetVersion, TrainingJob, TrainingStatus
from app.models.user import Client, Role, User
from app.services.r2_client import r2

router = APIRouter(tags=["training"])

ARCHITECTURES: dict[str, dict] = {
    "yolov8": {"label": "YOLOv8", "sizes": ["n", "s", "m", "l", "x"], "weights": "yolov8{size}.pt"},
    "yolo11": {"label": "YOLO11", "sizes": ["n", "s", "m", "l", "x"], "weights": "yolo11{size}.pt"},
    "rfdetr": {"label": "RF-DETR", "sizes": ["n", "s", "m", "l", "x"], "weights": "rtdetr-{size}.pt"},
}


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


def _job_out(job: TrainingJob, include_token: bool = False) -> dict:
    out = {
        "id": job.id,
        "project_id": job.project_id,
        "version_id": job.version_id,
        "engine": job.engine,
        "architecture": job.architecture,
        "model_size": job.model_size,
        "epochs_total": job.epochs_total,
        "status": job.status,
        "current_epoch": job.current_epoch,
        "metrics": job.metrics or [],
        "results": job.results,
        "error": job.error,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }
    if include_token:
        out["ingest_token"] = job.ingest_token
        out["script_url"] = (
            f"{settings.backend_public_url}/v1/training/jobs/{job.id}/script"
            f"?token={job.ingest_token}"
        )
    return out


# ── Dataset versions ──────────────────────────────────────────────────────────
class VersionCreate(BaseModel):
    name: str = ""


@router.post("/datasets/{project_id}/versions")
async def create_version(
    project_id: str,
    body: VersionCreate,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Freeze the current labeled set into a version with 80/10/10 splits."""
    project = await _owned_project(project_id, user, db)

    labeled = (
        await db.execute(
            select(DatasetImage)
            .where(
                DatasetImage.project_id == project.id,
                DatasetImage.status != ImageStatus.unlabeled.value,
            )
            .order_by(DatasetImage.order_index)
        )
    ).scalars().all()
    if len(labeled) < 3:
        raise HTTPException(
            status_code=400,
            detail="Label at least 3 images before creating a version.",
        )

    # Deterministic 80/10/10 split; guarantee ≥1 valid and ≥1 test.
    n = len(labeled)
    n_test = max(1, round(n * 0.1))
    n_valid = max(1, round(n * 0.1))
    n_train = n - n_valid - n_test
    for i, img in enumerate(labeled):
        img.split = "train" if i < n_train else ("valid" if i < n_train + n_valid else "test")

    count = (
        await db.execute(
            select(func.count(DatasetVersion.id)).where(
                DatasetVersion.project_id == project.id
            )
        )
    ).scalar() or 0

    version = DatasetVersion(
        project_id=project.id,
        number=count + 1,
        name=body.name or f"v{count + 1}",
        image_count=n,
        train_count=n_train,
        valid_count=n_valid,
        test_count=n_test,
    )
    db.add(version)
    await db.commit()
    await db.refresh(version)
    return _version_out(version)


def _version_out(v: DatasetVersion) -> dict:
    return {
        "id": v.id,
        "number": v.number,
        "name": v.name,
        "image_count": v.image_count,
        "train_count": v.train_count,
        "valid_count": v.valid_count,
        "test_count": v.test_count,
        "created_at": v.created_at,
    }


@router.get("/datasets/{project_id}/versions")
async def list_versions(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(project_id, user, db)
    rows = (
        await db.execute(
            select(DatasetVersion)
            .where(DatasetVersion.project_id == project.id)
            .order_by(DatasetVersion.number.desc())
        )
    ).scalars().all()
    return [_version_out(v) for v in rows]


# ── Start / poll / list jobs ─────────────────────────────────────────────────
class TrainStart(BaseModel):
    version_id: str
    architecture: str = "yolov8"
    model_size: str = "n"
    epochs: int = Field(default=25, ge=1, le=300)


@router.post("/datasets/{project_id}/train")
async def start_training(
    project_id: str,
    body: TrainStart,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    project = await _owned_project(project_id, user, db)
    if body.architecture not in ARCHITECTURES:
        raise HTTPException(status_code=400, detail="Unknown architecture")
    if body.model_size not in ARCHITECTURES[body.architecture]["sizes"]:
        raise HTTPException(status_code=400, detail="Unknown model size")
    version = await db.get(DatasetVersion, body.version_id)
    if not version or version.project_id != project.id:
        raise HTTPException(status_code=404, detail="Dataset version not found")

    job = TrainingJob(
        project_id=project.id,
        version_id=version.id,
        architecture=body.architecture,
        model_size=body.model_size,
        epochs_total=body.epochs,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return _job_out(job, include_token=True)


@router.get("/training/jobs/{job_id}")
async def get_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    job = await db.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    await _owned_project(job.project_id, user, db)
    return _job_out(job, include_token=True)


@router.get("/datasets/{project_id}/models")
async def list_models(
    project_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Training jobs for the Models section — newest first, real metrics only."""
    project = await _owned_project(project_id, user, db)
    rows = (
        await db.execute(
            select(TrainingJob)
            .where(TrainingJob.project_id == project.id)
            .order_by(TrainingJob.created_at.desc())
        )
    ).scalars().all()
    return [_job_out(j) for j in rows]


# ── Trainer-facing (token auth — called from Colab) ──────────────────────────
async def _job_by_token(job_id: str, token: str, db: AsyncSession) -> TrainingJob:
    job = await db.get(TrainingJob, job_id)
    if not job or token != job.ingest_token:
        raise HTTPException(status_code=401, detail="Invalid training token")
    return job


@router.get("/training/jobs/{job_id}/data")
async def trainer_data(
    job_id: str,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """Everything the Colab trainer needs: classes, splits, image URLs and
    YOLO-format labels (class cx cy w h, normalized)."""
    job = await _job_by_token(job_id, token, db)
    project = await db.get(Project, job.project_id)
    labels = [lc["name"] for lc in (project.label_taxonomy or [])] or ["object"]
    label_index = {n: i for i, n in enumerate(labels)}

    images = (
        await db.execute(
            select(DatasetImage)
            .where(
                DatasetImage.project_id == project.id,
                DatasetImage.status != ImageStatus.unlabeled.value,
            )
            .order_by(DatasetImage.order_index)
        )
    ).scalars().all()

    out = []
    for img in images:
        lines = []
        for sh in img.annotations or []:
            cls = label_index.get(sh.get("label"), 0)
            cx = sh["x"] + sh["w"] / 2
            cy = sh["y"] + sh["h"] / 2
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {sh['w']:.6f} {sh['h']:.6f}")
        out.append(
            {
                "filename": img.filename,
                "url": r2.presign_get(img.r2_key, expires=7200),
                "split": img.split,
                "labels": lines,
            }
        )
    return {
        "project": project.name,
        "classes": labels,
        "epochs": job.epochs_total,
        "architecture": job.architecture,
        "model_size": job.model_size,
        "images": out,
    }


class TrainerEvent(BaseModel):
    type: str  # started | epoch | completed | failed
    epoch: int | None = None
    metrics: dict | None = None
    results: dict | None = None
    error: str | None = None


@router.post("/training/jobs/{job_id}/events")
async def trainer_event(
    job_id: str,
    body: TrainerEvent,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    job = await _job_by_token(job_id, token, db)

    if body.type == "started":
        job.status = TrainingStatus.running.value
        job.started_at = job.started_at or utcnow()
    elif body.type == "epoch":
        if job.status != TrainingStatus.running.value:
            job.status = TrainingStatus.running.value
            job.started_at = job.started_at or utcnow()
        job.current_epoch = body.epoch or (job.current_epoch + 1)
        entry = {"epoch": job.current_epoch, **(body.metrics or {})}
        job.metrics = [*(job.metrics or []), entry]
    elif body.type == "completed":
        job.status = TrainingStatus.completed.value
        job.results = body.results or {}
        job.completed_at = utcnow()
    elif body.type == "failed":
        job.status = TrainingStatus.failed.value
        job.error = body.error or "Training failed"
        job.completed_at = utcnow()
    else:
        raise HTTPException(status_code=400, detail="Unknown event type")

    await db.commit()
    return {"ok": True, "status": job.status}


@router.get("/training/jobs/{job_id}/script", response_class=PlainTextResponse)
async def trainer_script(
    job_id: str,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """A complete training script for one Colab cell. It downloads the dataset
    from us, trains with Ultralytics on the free GPU, and streams epoch metrics
    + final evaluation (confusion matrix, confidence sweep) back to Annoting."""
    job = await _job_by_token(job_id, token, db)
    arch = ARCHITECTURES.get(job.architecture, ARCHITECTURES["yolov8"])
    weights = arch["weights"].format(size=job.model_size)
    base = settings.backend_public_url.rstrip("/")
    endpoint = base + "/v1/training/jobs/" + job.id

    # ── RF-DETR: uses the `rfdetr` pip package + COCO format ─────────────────
    if job.architecture == "rfdetr":
        model_cls = "RFDETRBase" if job.model_size in ("n", "s", "m") else "RFDETRLarge"
        return f'''# Annoting trainer — job {job.id} (RF-DETR)
# Runs on Google Colab (Runtime -> Change runtime type -> GPU).
import json, os, subprocess, sys, urllib.request
from pathlib import Path

BASE = {endpoint!r}
TOKEN = {job.ingest_token!r}
SIZE = {job.model_size!r}          # n/s/m → RFDETRBase  |  l/x → RFDETRLarge
EPOCHS = {job.epochs_total}

def post(payload, silent=False):
    try:
        req = urllib.request.Request(
            BASE + "/events?token=" + TOKEN,
            data=json.dumps(payload).encode(),
            headers={{"Content-Type": "application/json"}},
        )
        urllib.request.urlopen(req, timeout=30).read()
    except Exception:
        if not silent: raise

try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "rfdetr[train,loggers]", "supervision", "Pillow"], check=True)
    from rfdetr import RFDETRBase, RFDETRLarge
    from PIL import Image

    # 1) Pull dataset manifest from Annoting.
    print("Downloading dataset...")
    data = json.load(urllib.request.urlopen(BASE + "/data?token=" + TOKEN, timeout=300))
    root = "/content/annoting_ds"
    classes = data["classes"]
    categories = [{{"id": i, "name": n, "supercategory": "object"}}
                  for i, n in enumerate(classes)]

    coco = {{sp: {{"images": [], "annotations": [], "categories": categories}}
             for sp in ("train", "valid", "test")}}
    counters = {{"train": 0, "valid": 0, "test": 0}}
    ann_id = 1

    for img_data in data["images"]:
        split = img_data["split"]
        os.makedirs(f"{{root}}/{{split}}", exist_ok=True)
        img_path = f"{{root}}/{{split}}/{{img_data['filename']}}"
        urllib.request.urlretrieve(img_data["url"], img_path)

        with Image.open(img_path) as pil:
            iw, ih = pil.size

        counters[split] += 1
        img_id = counters[split]
        coco[split]["images"].append({{
            "id": img_id, "file_name": img_data["filename"],
            "width": iw, "height": ih,
        }})
        for line in img_data["labels"]:
            parts = line.strip().split()
            if len(parts) < 5: continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x = (cx - bw / 2) * iw
            y = (cy - bh / 2) * ih
            pw, ph = bw * iw, bh * ih
            coco[split]["annotations"].append({{
                "id": ann_id, "image_id": img_id, "category_id": cls_id,
                "bbox": [x, y, pw, ph], "area": pw * ph, "iscrowd": 0,
            }})
            ann_id += 1

    for split, coco_data in coco.items():
        if coco_data["images"]:
            with open(f"{{root}}/{{split}}/_annotations.coco.json", "w") as f:
                json.dump(coco_data, f)
    print("Dataset ready (COCO format).")

    # 2) Create model — Base for nano/small/medium, Large for large/xlarge.
    ModelCls = RFDETRBase if SIZE in ("n", "s", "m") else RFDETRLarge
    model = ModelCls()
    post({{"type": "started"}})
    print(f"Training RF-DETR {{'Base' if SIZE in ('n','s','m') else 'Large'}} · {{EPOCHS}} epochs")

    # 3) Train and stream per-epoch metrics back to Annoting.
    _last_log = {{}}
    def on_epoch_end(log: dict):
        _last_log.update(log)
        post({{
            "type": "epoch",
            "epoch": int(log.get("epoch", 0)),
            "metrics": {{
                "map50":      float(log.get("map_50",    log.get("mAP_50",    log.get("map50",    0)))),
                "train_loss": float(log.get("train_loss",log.get("loss",      0))),
                "val_loss":   float(log.get("val_loss",  0)),
                "precision":  float(log.get("precision", 0)),
                "recall":     float(log.get("recall",    0)),
            }},
        }}, silent=True)

    model.train(
        dataset_dir=root,
        epochs=EPOCHS,
        batch_size=4,
        grad_accum_steps=4,
        output_dir="/content/rfdetr_output",
        callbacks={{"on_fit_epoch_end": on_epoch_end}},
    )

    # 4) Post final results.
    post({{
        "type": "completed",
        "results": {{
            "map50":      float(_last_log.get("map_50",    _last_log.get("mAP_50",    _last_log.get("map50",    0)))),
            "map50_95":   float(_last_log.get("map_50_95", 0)),
            "precision":  float(_last_log.get("precision", 0)),
            "recall":     float(_last_log.get("recall",    0)),
            "f1":         0.0,
            "per_class":  [],
            "confusion_matrix": {{"labels": [*classes, "background"], "matrix": []}},
            "confidence_curve": [],
            "optimal_confidence": 0.5,
        }},
    }})
    print("Training complete — results are live on your Annoting dashboard.")
except Exception as e:
    try: post({{"type": "failed", "error": str(e)[:500]}}, silent=True)
    finally: raise
'''

    # ── YOLO (yolov8 / yolo11): Ultralytics path ──────────────────────────────
    return f'''# Annoting trainer — job {job.id}
# Runs on Google Colab (Runtime -> Change runtime type -> GPU).
import json, os, subprocess, sys, urllib.request

BASE = {base + "/v1/training/jobs/" + job.id!r}
TOKEN = {job.ingest_token!r}

def post(payload):
    req = urllib.request.Request(
        BASE + "/events?token=" + TOKEN,
        data=json.dumps(payload).encode(),
        headers={{"Content-Type": "application/json"}},
    )
    urllib.request.urlopen(req, timeout=30).read()

try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    from ultralytics import YOLO
    import numpy as np

    # 1) Pull the dataset (images + YOLO labels) from Annoting.
    data = json.load(urllib.request.urlopen(BASE + "/data?token=" + TOKEN, timeout=120))
    root = "/content/annoting_ds"
    for split in ("train", "valid", "test"):
        os.makedirs(f"{{root}}/{{split}}/images", exist_ok=True)
        os.makedirs(f"{{root}}/{{split}}/labels", exist_ok=True)
    for im in data["images"]:
        split = im["split"]
        img_path = f"{{root}}/{{split}}/images/{{im['filename']}}"
        urllib.request.urlretrieve(im["url"], img_path)
        stem = os.path.splitext(im["filename"])[0]
        with open(f"{{root}}/{{split}}/labels/{{stem}}.txt", "w") as f:
            f.write("\\n".join(im["labels"]))
    yaml_path = f"{{root}}/data.yaml"
    with open(yaml_path, "w") as f:
        f.write(
            f"path: {{root}}\\ntrain: train/images\\nval: valid/images\\n"
            f"test: test/images\\nnames: {{ {{i: n for i, n in enumerate(data['classes'])}} }}\\n"
        )

    # 2) Train, streaming each epoch back to Annoting.
    post({{"type": "started"}})
    model = YOLO({weights!r})

    def on_epoch_end(trainer):
        m = trainer.metrics or {{}}
        post({{
            "type": "epoch",
            "epoch": int(trainer.epoch) + 1,
            "metrics": {{
                "train_loss": float(sum(trainer.tloss)) if trainer.tloss is not None else None,
                "val_loss": float(m.get("val/box_loss", 0) + m.get("val/cls_loss", 0) + m.get("val/dfl_loss", 0)) or None,
                "map50": float(m.get("metrics/mAP50(B)", 0)),
                "precision": float(m.get("metrics/precision(B)", 0)),
                "recall": float(m.get("metrics/recall(B)", 0)),
            }},
        }})

    model.add_callback("on_fit_epoch_end", on_epoch_end)
    model.train(data=yaml_path, epochs={job.epochs_total}, imgsz=640, verbose=True)

    # 3) Final evaluation: metrics, per-class, confusion matrix, confidence sweep.
    val = model.val(data=yaml_path, split="test" if any(i["split"] == "test" for i in data["images"]) else "val")
    names = data["classes"]
    cm = val.confusion_matrix.matrix.tolist() if val.confusion_matrix is not None else []
    per_class = []
    for i, name in enumerate(names):
        try:
            p, r, ap50, _ = val.box.class_result(i)
        except Exception:
            p = r = ap50 = 0.0
        per_class.append({{"name": name, "precision": float(p), "recall": float(r), "map50": float(ap50)}})

    # Confidence sweep from the PR curves ultralytics computed.
    curve = []
    optimal = 0.5
    try:
        confs = np.linspace(0.05, 0.95, 19)
        f1s = val.box.f1_curve.mean(0) if hasattr(val.box, "f1_curve") else None
        xs = val.box.px if hasattr(val.box, "px") else None
        if f1s is not None and xs is not None:
            for c in confs:
                idx = int(np.argmin(np.abs(np.array(xs) - c)))
                curve.append({{
                    "confidence": float(c),
                    "f1": float(f1s[idx]),
                    "precision": float(val.box.p_curve.mean(0)[idx]) if hasattr(val.box, "p_curve") else None,
                    "recall": float(val.box.r_curve.mean(0)[idx]) if hasattr(val.box, "r_curve") else None,
                }})
            optimal = float(xs[int(np.argmax(f1s))])
    except Exception:
        pass

    post({{
        "type": "completed",
        "results": {{
            "map50": float(val.box.map50),
            "map50_95": float(val.box.map),
            "precision": float(val.box.mp),
            "recall": float(val.box.mr),
            "f1": float(2 * val.box.mp * val.box.mr / (val.box.mp + val.box.mr)) if (val.box.mp + val.box.mr) else 0.0,
            "per_class": per_class,
            "confusion_matrix": {{"labels": [*names, "background"], "matrix": cm}},
            "confidence_curve": curve,
            "optimal_confidence": optimal,
        }},
    }})
    print("Training complete — results are live on your Annoting dashboard.")
except Exception as e:
    try:
        post({{"type": "failed", "error": str(e)[:500]}})
    finally:
        raise
'''
