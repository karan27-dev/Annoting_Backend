from __future__ import annotations

from pydantic import BaseModel, EmailStr


class AnnotatorApplyRequest(BaseModel):
    full_name: str
    email: EmailStr
    city: str | None = None
    device: str | None = None
    experience: str | None = None
    preferred_task_types: list[str] = []


class ReviewActionRequest(BaseModel):
    notes: str | None = None


class AnnotatorStatusUpdate(BaseModel):
    status: str
    reason: str | None = None


class MessageResponse(BaseModel):
    message: str
