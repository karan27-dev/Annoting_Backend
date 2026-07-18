from __future__ import annotations

import secrets

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.security import (
    create_purpose_token,
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.database import get_db
from app.deps import get_current_user
from app.models.common import utcnow
from app.models.user import Client, Role, User
from app.schemas.auth import (
    ForgotPasswordRequest,
    GoogleAuthRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserOut,
    VerifyEmailRequest,
)
from app.schemas.misc import MessageResponse
from app.services.email_service import email_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/google", response_model=TokenResponse)
async def google_auth(body: GoogleAuthRequest, db: AsyncSession = Depends(get_db)):
    """Sign in / sign up with a Google ID token. Verifies the token with Google,
    then creates the client account on first use (no password needed)."""
    if not settings.google_oauth_client_id:
        raise HTTPException(
            status_code=503,
            detail="Google sign-in isn't configured on the server yet.",
        )

    # Verify the ID token directly with Google (no client secret needed).
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(
            "https://oauth2.googleapis.com/tokeninfo",
            params={"id_token": body.credential},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    info = r.json()

    if info.get("aud") != settings.google_oauth_client_id:
        raise HTTPException(status_code=401, detail="Google token audience mismatch")
    email = info.get("email")
    if not email or str(info.get("email_verified")).lower() != "true":
        raise HTTPException(status_code=401, detail="Google email not verified")

    name = info.get("name") or email.split("@")[0]

    user = (
        await db.execute(select(User).where(User.email == email))
    ).scalar_one_or_none()
    if not user:
        # Random password — the account is only reachable via Google.
        user = User(
            email=email,
            password_hash=hash_password(secrets.token_urlsafe(24)),
            full_name=name,
            role=Role.client.value,
            is_verified=True,
        )
        db.add(user)
        await db.flush()
        db.add(Client(user_id=user.id, company_name=name))
    user.last_login_at = utcnow()
    await db.commit()

    return TokenResponse(
        access_token=create_token(user.id, user.role, "access"),
        refresh_token=create_token(user.id, user.role, "refresh"),
    )


@router.post("/register", response_model=MessageResponse, status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    exists = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        full_name=body.full_name,
        role=Role.client.value,
        # In dev (no email service) accounts are usable immediately.
        is_verified=not email_service.configured,
    )
    db.add(user)
    await db.flush()

    db.add(Client(user_id=user.id, company_name=body.company_name))
    await db.commit()

    token = create_purpose_token(user.email, "verify")
    email_service.send_verification(user.email, token)
    return MessageResponse(message="Account created. Check your email to verify.")


@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(body: VerifyEmailRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(body.token)
    if not payload or payload.get("purpose") != "verify":
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    user = (
        await db.execute(select(User).where(User.email == payload["sub"]))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.is_verified = True
    await db.commit()
    return MessageResponse(message="Email verified. You can now log in.")


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")

    user.last_login_at = utcnow()
    await db.commit()
    return TokenResponse(
        access_token=create_token(user.id, user.role, "access"),
        refresh_token=create_token(user.id, user.role, "refresh"),
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    payload = decode_token(body.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    user = await db.get(User, payload["sub"])
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found")
    return TokenResponse(
        access_token=create_token(user.id, user.role, "access"),
        refresh_token=create_token(user.id, user.role, "refresh"),
    )


@router.post("/logout", response_model=MessageResponse)
async def logout(_: User = Depends(get_current_user)):
    # With Redis configured, add the token jti to a blocklist here.
    return MessageResponse(message="Logged out")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    body: ForgotPasswordRequest, db: AsyncSession = Depends(get_db)
):
    user = (
        await db.execute(select(User).where(User.email == body.email))
    ).scalar_one_or_none()
    if user:
        token = create_purpose_token(user.email, "reset", minutes=60)
        email_service.send_password_reset(user.email, token)
    # Always 200 to avoid email enumeration.
    return MessageResponse(message="If that email exists, a reset link was sent.")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    body: ResetPasswordRequest, db: AsyncSession = Depends(get_db)
):
    payload = decode_token(body.token)
    if not payload or payload.get("purpose") != "reset":
        raise HTTPException(status_code=400, detail="Invalid or expired token")
    user = (
        await db.execute(select(User).where(User.email == payload["sub"]))
    ).scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password_hash = hash_password(body.new_password)
    await db.commit()
    return MessageResponse(message="Password updated. You can log in now.")


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user
