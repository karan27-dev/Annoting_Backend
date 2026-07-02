"""Render a CVAT job's frames + shapes into a UI-friendly payload.

Shared by the reviewer panel (inspect a job) and the client dashboard (preview
finalized samples). Talks to CVAT via its REST API only.
"""
from __future__ import annotations

import base64

from app.services.cvat_client import cvat


def label_map(labels: list[dict]) -> dict[int, dict]:
    return {
        lab["id"]: {"name": lab["name"], "color": lab.get("color") or "#e2553d"}
        for lab in labels
    }


async def frame_payload(job_id: int, frame: int) -> dict:
    """One frame: image (base64 data URL) + the shapes drawn on it."""
    labels = await cvat.get_job_labels(job_id)
    lm = label_map(labels)
    ann = await cvat.get_job_annotations(job_id)
    meta = await cvat.get_job_meta(job_id)

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

    img = await cvat.get_frame_bytes(job_id, frame)
    b64 = base64.b64encode(img).decode()
    return {
        "frame": frame,
        "image": f"data:image/jpeg;base64,{b64}",
        "width": dim.get("width", 1280),
        "height": dim.get("height", 720),
        "shapes": shapes,
    }
