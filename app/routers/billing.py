from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.deps import get_current_user, require_roles
from app.models.assignment import AssignmentStatus, TaskAssignment
from app.models.billing import Invoice, InvoiceStatus, ProjectQuote
from app.models.common import utcnow
from app.models.project import Project
from app.models.user import Client, Role, User
from app.schemas.billing import InvoiceCreate, InvoiceOut
from app.schemas.misc import MessageResponse
from app.services.pricing_engine import gst_breakdown
from app.services.r2_client import r2

router = APIRouter(tags=["billing"])


@router.get("/billing/invoices", response_model=list[InvoiceOut])
async def list_invoices(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
):
    if user.role in (Role.super_admin.value, Role.ops_manager.value):
        rows = (await db.execute(select(Invoice))).scalars().all()
    else:
        client = (
            await db.execute(select(Client).where(Client.user_id == user.id))
        ).scalar_one_or_none()
        if not client:
            return []
        rows = (
            await db.execute(select(Invoice).where(Invoice.client_id == client.id))
        ).scalars().all()
    return rows


@router.post("/billing/invoices", response_model=InvoiceOut, status_code=201)
async def create_invoice(
    body: InvoiceCreate,
    _: User = Depends(require_roles(Role.super_admin, Role.ops_manager)),
    db: AsyncSession = Depends(get_db),
):
    project = await db.get(Project, body.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Don't double-bill a project.
    existing = (
        await db.execute(
            select(Invoice).where(Invoice.project_id == project.id)
        )
    ).scalars().first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"This project already has invoice {existing.invoice_number}.",
        )

    quote = (
        await db.execute(
            select(ProjectQuote)
            .where(
                ProjectQuote.project_id == project.id,
                ProjectQuote.accepted_at.is_not(None),
            )
            .order_by(ProjectQuote.accepted_at.desc())
        )
    ).scalars().first()
    if not quote:
        raise HTTPException(
            status_code=400,
            detail="No accepted quote — the client must accept the quote first.",
        )

    rate = float(quote.rate_per_label_inr)

    # Labels the reviewer actually approved so far.
    actual_labels = (
        await db.execute(
            select(func.coalesce(func.sum(TaskAssignment.labels_count), 0)).where(
                TaskAssignment.project_id == project.id,
                TaskAssignment.status == AssignmentStatus.approved.value,
            )
        )
    ).scalar() or 0

    # Bill the price the client agreed to (the accepted quote). Once real work
    # is approved and exceeds the estimate, bill the delivered labels instead so
    # dense datasets are charged fairly.
    delivered_amount = round(float(actual_labels) * rate, 2)
    quoted_amount = float(quote.quoted_total_inr)
    amount = max(quoted_amount, delivered_amount)
    gst, total = gst_breakdown(amount)

    quote.actual_labels = int(actual_labels)
    quote.final_total_inr = amount

    count = (await db.execute(select(func.count(Invoice.id)))).scalar() or 0
    number = f"SYL-{utcnow().year}-{count + 1:04d}"

    invoice = Invoice(
        client_id=project.client_id,
        project_id=project.id,
        invoice_number=number,
        amount_inr=amount,
        gst_amount_inr=gst,
        total_inr=total,
        status=InvoiceStatus.sent.value,
        issued_at=utcnow(),
        due_at=utcnow() + timedelta(days=15),
    )
    db.add(invoice)
    await db.commit()
    await db.refresh(invoice)
    return invoice


@router.get("/billing/invoiceable")
async def invoiceable_projects(
    _: User = Depends(require_roles(Role.super_admin, Role.ops_manager)),
    db: AsyncSession = Depends(get_db),
):
    """Projects the client has accepted a quote for that don't yet have an
    invoice — the admin generates the invoice from here."""
    from app.models.user import Client

    invoiced = {
        row[0]
        for row in (
            await db.execute(select(Invoice.project_id))
        ).all()
    }

    rows = (
        await db.execute(
            select(Project, ProjectQuote, Client)
            .join(
                ProjectQuote,
                (ProjectQuote.project_id == Project.id)
                & (ProjectQuote.accepted_at.is_not(None)),
            )
            .join(Client, Client.id == Project.client_id, isouter=True)
            .order_by(Project.created_at.desc())
        )
    ).all()

    out = []
    seen: set[str] = set()
    for project, quote, client in rows:
        if project.id in invoiced or project.id in seen:
            continue
        seen.add(project.id)
        out.append(
            {
                "project_id": project.id,
                "project_name": project.name,
                "client_company": client.company_name if client else None,
                "status": project.status,
                "quoted_total_inr": float(quote.quoted_total_inr),
                "estimated_labels": quote.estimated_labels,
                "rate_per_label_inr": float(quote.rate_per_label_inr),
            }
        )
    return out


@router.get("/billing/invoices/{invoice_id}/pdf")
async def invoice_pdf(
    invoice_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    invoice = await db.get(Invoice, invoice_id)
    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    url = invoice.pdf_r2_url or r2.presign_get(f"invoices/{invoice.invoice_number}.pdf")
    return {"url": url}


@router.post("/billing/invoices/{invoice_id}/payment-link")
async def payment_link(
    invoice_id: str,
    _: User = Depends(require_roles(Role.super_admin, Role.ops_manager)),
):
    # Razorpay integration is coming soon. Surface a clear, non-error signal.
    if not settings.payments_enabled:
        raise HTTPException(
            status_code=503,
            detail="Online payments (Razorpay) are coming soon.",
        )
    # When enabled: create a Razorpay order and return the checkout link.
    return {"payment_link": None}


@router.post("/billing/webhooks/razorpay", response_model=MessageResponse)
async def razorpay_webhook():
    # Coming soon: validate HMAC-SHA256 signature, then mark the invoice paid.
    if not settings.payments_enabled:
        return MessageResponse(message="Payments not enabled")
    return MessageResponse(message="ok")
