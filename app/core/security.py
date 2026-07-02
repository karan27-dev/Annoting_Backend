"""Password hashing and JWT helpers."""
from __future__ import annotations

from datetime import timedelta

import bcrypt
from jose import JWTError, jwt

from app.config import settings
from app.models.common import utcnow


def _to_bytes(password: str) -> bytes:
    # bcrypt only considers the first 72 bytes; truncate explicitly to avoid errors.
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_to_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_to_bytes(plain), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_token(subject: str, role: str, kind: str = "access") -> str:
    now = utcnow()
    if kind == "refresh":
        expires = now + timedelta(days=settings.refresh_token_expire_days)
    else:
        expires = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": subject,
        "role": role,
        "type": kind,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
    }
    return jwt.encode(
        payload, settings.resolved_jwt_secret, algorithm=settings.jwt_algorithm
    )


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(
            token, settings.resolved_jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except JWTError:
        return None


def create_purpose_token(email: str, purpose: str, minutes: int = 60 * 24) -> str:
    """Short-lived token for email verification / password reset."""
    now = utcnow()
    payload = {
        "sub": email,
        "purpose": purpose,
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(
        payload, settings.resolved_jwt_secret, algorithm=settings.jwt_algorithm
    )
