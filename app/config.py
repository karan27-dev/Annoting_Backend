"""Central configuration. All secrets come from the environment / .env file."""
from __future__ import annotations

import secrets
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    environment: str = "development"

    # ── Database ──────────────────────────────────────────────────────────────
    # Empty -> local SQLite so the app runs with zero setup. Point at Postgres for prod:
    #   postgresql+asyncpg://user:pass@localhost:5432/annoting
    database_url: str = ""

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    refresh_token_expire_days: int = 30

    # ── CVAT ──────────────────────────────────────────────────────────────────
    cvat_url: str = "http://localhost:8080"
    cvat_api_user: str = ""
    cvat_api_password: str = ""
    cvat_webhook_secret: str = ""

    # ── Cloudflare R2 ─────────────────────────────────────────────────────────
    r2_account_id: str = ""
    r2_access_key: str = ""
    r2_secret_key: str = ""
    r2_bucket_name: str = "annoting-datasets"

    # ── Google Drive intake ───────────────────────────────────────────────────
    # API key with the Drive API enabled; used to list/download files that
    # clients share via "anyone with the link".
    google_api_key: str = ""

    # ── Training bridge ───────────────────────────────────────────────────────
    # Public base URL of THIS backend, embedded into the Colab training script
    # so the notebook can call back with epoch metrics. localhost works only
    # for local trainers; for real Colab use a tunnel (ngrok/cloudflared) or a
    # deployed backend URL.
    backend_public_url: str = "http://localhost:8000"

    # ── Google Sign-In (OAuth) ────────────────────────────────────────────────
    # OAuth 2.0 Web client ID from Google Cloud Console. The same value must be
    # exposed to the frontend as NEXT_PUBLIC_GOOGLE_CLIENT_ID. Enables the
    # "Continue with Google" button; no client secret needed (ID-token flow).
    google_oauth_client_id: str = ""

    # ── Resend (email) ────────────────────────────────────────────────────────
    resend_api_key: str = ""
    resend_from_email: str = "noreply@annoting.com"

    # ── Razorpay (payments — coming soon) ─────────────────────────────────────
    payments_enabled: bool = False
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""
    razorpay_webhook_secret: str = ""

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = ""

    # ── CORS ──────────────────────────────────────────────────────────────────
    frontend_origin: str = "http://localhost:3000"

    @property
    def resolved_database_url(self) -> str:
        raw = self.database_url.strip()
        if not raw:
            # Zero-config dev default.
            return "sqlite+aiosqlite:///./annoting.db"
        # Accept a raw Postgres/Neon/Supabase URL and normalize it to the async
        # driver, stripping libpq-only query params (sslmode, channel_binding…)
        # that asyncpg rejects. SSL is enabled via connect_args in database.py.
        if raw.startswith("postgres://"):
            raw = "postgresql://" + raw[len("postgres://") :]
        if raw.startswith("postgresql://"):
            raw = "postgresql+asyncpg://" + raw[len("postgresql://") :]
        if raw.startswith("postgresql+asyncpg://") and "?" in raw:
            raw = raw.split("?", 1)[0]
        return raw

    @property
    def is_postgres(self) -> bool:
        return self.resolved_database_url.startswith("postgresql+asyncpg")

    @property
    def resolved_jwt_secret(self) -> str:
        # Stable per-process dev secret if none provided. SET THIS IN PRODUCTION.
        return self.jwt_secret or _DEV_SECRET


_DEV_SECRET = secrets.token_hex(32)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
