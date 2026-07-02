"""Shared column helpers — portable across SQLite (dev) and Postgres (prod)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone


def gen_uuid() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hours_since(dt: datetime | None) -> int:
    """Whole hours between dt and now, tolerant of naive datetimes from SQLite."""
    if dt is None:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int((utcnow() - dt).total_seconds() / 3600)
