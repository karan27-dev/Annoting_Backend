from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import init_db
from app.routers import (
    admin,
    annotator,
    auth,
    billing,
    datasets,
    pricing,
    projects,
    reviewer,
    training,
    uploads,
    webhooks,
)

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Phase-1: auto-create tables on boot. Swap for Alembic migrations in prod.
    await init_db()
    yield


app = FastAPI(
    title="Annoting API",
    version="0.1.0",
    description="The operations layer around CVAT — clients, routing, QA, billing.",
    lifespan=lifespan,
)

# In development, allow any localhost port (dev servers pick varying ports).
# In production, lock to the configured frontend origin.
_cors = dict(
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if settings.environment == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
        **_cors,
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[settings.frontend_origin],
        **_cors,
    )

API = "/v1"
for r in (
    auth.router,
    projects.router,
    uploads.router,
    datasets.router,
    training.router,
    pricing.router,
    annotator.router,
    reviewer.router,
    admin.router,
    billing.router,
    webhooks.router,
):
    app.include_router(r, prefix=API)


@app.get("/")
async def root():
    return {"name": "Annoting API", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "environment": settings.environment,
        "integrations": {
            "cvat": bool(settings.cvat_api_user),
            "r2": bool(settings.r2_access_key),
            "email": bool(settings.resend_api_key),
            "payments": settings.payments_enabled,
        },
    }
