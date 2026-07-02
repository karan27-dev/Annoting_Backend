"""Async SQLAlchemy engine + session, plus the declarative Base."""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import settings

if settings.is_postgres:
    # Serverless Postgres (Neon) scales to zero and closes idle connections, and
    # its pooler (pgbouncer) doesn't support asyncpg prepared-statement caching.
    # NullPool = fresh connection per request (robust); statement_cache_size=0 for the pooler.
    engine = create_async_engine(
        settings.resolved_database_url,
        echo=False,
        future=True,
        poolclass=NullPool,
        connect_args={"ssl": True, "statement_cache_size": 0},
    )
else:
    engine = create_async_engine(
        settings.resolved_database_url, echo=False, future=True
    )

AsyncSessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


# Columns added after a DB was first created. Poor-man's migration for Phase 1
# so existing SQLite/Neon databases pick them up without Alembic.
_ADDITIVE_COLUMNS: list[tuple[str, str, str]] = [
    ("task_assignments", "frame_count", "INTEGER"),
    ("annotator_profiles", "cvat_password", "VARCHAR(100)"),
    # Intake pipeline (data source, media counts, delivery format, complexity).
    ("projects", "media_type", "VARCHAR(10) DEFAULT 'images'"),
    ("projects", "data_source", "VARCHAR(10) DEFAULT 'upload'"),
    ("projects", "gdrive_link", "VARCHAR(1000)"),
    ("projects", "image_count", "INTEGER DEFAULT 0"),
    ("projects", "video_count", "INTEGER DEFAULT 0"),
    ("projects", "delivery_format", "VARCHAR(20) DEFAULT 'coco'"),
    ("projects", "estimated_objects_per_image", "NUMERIC(6,2)"),
    ("projects", "complexity_tier", "VARCHAR(10)"),
    ("projects", "intake_status", "VARCHAR(20) DEFAULT 'awaiting_data'"),
    ("projects", "intake_detail", "TEXT"),
    # Admin-reviewed quotes: drafts until published.
    ("project_quotes", "published_at", "TIMESTAMP"),
    ("project_quotes", "admin_notes", "VARCHAR(500)"),
]


def _apply_additive_migrations(sync_conn) -> None:
    """Add missing columns, checking first so we never trip on duplicates.
    Runs on ONE connection — a lock-contended ALTER per column made startup
    crawl when another dev server held the SQLite file."""
    from sqlalchemy import inspect as sa_inspect

    insp = sa_inspect(sync_conn)
    existing: dict[str, set[str]] = {}
    for table, col, coltype in _ADDITIVE_COLUMNS:
        if table not in existing:
            try:
                existing[table] = {c["name"] for c in insp.get_columns(table)}
            except Exception:
                continue
        if col not in existing[table]:
            sync_conn.exec_driver_sql(
                f"ALTER TABLE {table} ADD COLUMN {col} {coltype}"
            )
            existing[table].add(col)


async def init_db() -> None:
    """Create all tables. Phase-1 convenience — swap for Alembic migrations later."""
    # Import models so they register on Base.metadata before create_all.
    from app import models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_apply_additive_migrations)
