"""Transactional email via Resend.

If RESEND_API_KEY is unset, emails are logged to the console instead of sent, so
the full flow (register -> verify, milestones, delivery) works in dev.
"""
from __future__ import annotations

import logging

from app.config import settings

logger = logging.getLogger("annoting.email")


class EmailService:
    @property
    def configured(self) -> bool:
        return bool(settings.resend_api_key)

    def send(self, to: str, subject: str, html: str) -> None:
        if not self.configured:
            logger.info("[email:dev] to=%s subject=%r", to, subject)
            return
        import resend  # lazy import

        resend.api_key = settings.resend_api_key
        resend.Emails.send(
            {
                "from": settings.resend_from_email,
                "to": [to],
                "subject": subject,
                "html": html,
            }
        )

    # ── Templated helpers ───────────────────────────────────────────────────────
    def send_verification(self, to: str, token: str) -> None:
        link = f"{settings.frontend_origin}/verify-email?token={token}"
        self.send(
            to,
            "Verify your Annoting account",
            f"<p>Welcome to Annoting!</p><p>Confirm your email: "
            f'<a href="{link}">Verify</a></p>',
        )

    def send_password_reset(self, to: str, token: str) -> None:
        link = f"{settings.frontend_origin}/reset-password?token={token}"
        self.send(
            to,
            "Reset your Annoting password",
            f'<p>Reset your password: <a href="{link}">Reset</a></p>',
        )

    def send_milestone(self, to: str, project_name: str, percent: int) -> None:
        self.send(
            to,
            f"{project_name} is {percent}% complete",
            f"<p>Your project <b>{project_name}</b> just hit {percent}%.</p>",
        )

    def send_job_assigned(self, to: str, project_name: str, deep_link: str) -> None:
        self.send(
            to,
            "A new job has been assigned to you",
            f'<p>New job on <b>{project_name}</b>. '
            f'<a href="{deep_link}">Open in canvas</a></p>',
        )


email_service = EmailService()
