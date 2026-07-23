"""Field templates and value builders for form-section pages."""

from __future__ import annotations

import datetime as dt
from random import Random
from typing import Dict, Tuple

from trace_tasks.tasks.pages.shared.text_generation import (
    format_currency_from_cents,
    sample_company_name,
    sample_email,
    sample_identifier,
    sample_person_name,
    sample_phone_number,
)


SUPPORTED_SECTIONED_DOCUMENT_SCENE_VARIANTS: Tuple[str, ...] = (
    "form_sheet",
    "invoice_sheet",
    "receipt_sheet",
)
SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE: Dict[str, Tuple[Dict[str, str], ...]] = {
    "form_sheet": (
        {
            "field_id": "applicant_name",
            "field_label": "Applicant Name",
            "section_id": "profile",
            "section_label": "Profile",
            "comparison_kind": "other",
        },
        {
            "field_id": "reference_id",
            "field_label": "Reference ID",
            "section_id": "profile",
            "section_label": "Profile",
            "comparison_kind": "other",
        },
        {
            "field_id": "department",
            "field_label": "Department",
            "section_id": "profile",
            "section_label": "Profile",
            "comparison_kind": "other",
        },
        {
            "field_id": "reviewer_name",
            "field_label": "Reviewer",
            "section_id": "profile",
            "section_label": "Profile",
            "comparison_kind": "other",
        },
        {
            "field_id": "contact_phone",
            "field_label": "Phone",
            "section_id": "contact",
            "section_label": "Contact",
            "comparison_kind": "other",
        },
        {
            "field_id": "contact_email",
            "field_label": "Email",
            "section_id": "contact",
            "section_label": "Contact",
            "comparison_kind": "other",
        },
        {
            "field_id": "registration_fee",
            "field_label": "Registration Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "service_fee",
            "field_label": "Service Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "discount_amount",
            "field_label": "Discount",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "processing_fee",
            "field_label": "Processing Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "materials_fee",
            "field_label": "Materials Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "facility_fee",
            "field_label": "Facility Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "record_fee",
            "field_label": "Record Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "access_fee",
            "field_label": "Access Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "priority_fee",
            "field_label": "Priority Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
        {
            "field_id": "late_fee",
            "field_label": "Late Fee",
            "section_id": "fees",
            "section_label": "Fees",
            "comparison_kind": "amount",
        },
    ),
    "invoice_sheet": (
        {
            "field_id": "vendor_name",
            "field_label": "Vendor",
            "section_id": "parties",
            "section_label": "Parties",
            "comparison_kind": "other",
        },
        {
            "field_id": "customer_name",
            "field_label": "Customer",
            "section_id": "parties",
            "section_label": "Parties",
            "comparison_kind": "other",
        },
        {
            "field_id": "invoice_number",
            "field_label": "Invoice Number",
            "section_id": "parties",
            "section_label": "Parties",
            "comparison_kind": "other",
        },
        {
            "field_id": "account_id",
            "field_label": "Account ID",
            "section_id": "account",
            "section_label": "Account",
            "comparison_kind": "other",
        },
        {
            "field_id": "contact_email",
            "field_label": "Contact Email",
            "section_id": "account",
            "section_label": "Account",
            "comparison_kind": "other",
        },
        {
            "field_id": "issue_date",
            "field_label": "Issue Date",
            "section_id": "dates",
            "section_label": "Dates",
            "comparison_kind": "other",
        },
        {
            "field_id": "service_date",
            "field_label": "Service Date",
            "section_id": "dates",
            "section_label": "Dates",
            "comparison_kind": "other",
        },
        {
            "field_id": "due_date",
            "field_label": "Due Date",
            "section_id": "dates",
            "section_label": "Dates",
            "comparison_kind": "other",
        },
        {
            "field_id": "subtotal_amount",
            "field_label": "Subtotal",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
        {
            "field_id": "tax_amount",
            "field_label": "Tax",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
        {
            "field_id": "discount_amount",
            "field_label": "Discount",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
        {
            "field_id": "shipping_amount",
            "field_label": "Shipping",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
        {
            "field_id": "handling_amount",
            "field_label": "Handling",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
        {
            "field_id": "service_credit",
            "field_label": "Service Credit",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
        {
            "field_id": "platform_amount",
            "field_label": "Platform",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
        {
            "field_id": "archive_amount",
            "field_label": "Archive",
            "section_id": "billing_summary",
            "section_label": "Billing Summary",
            "comparison_kind": "amount",
        },
    ),
    "receipt_sheet": (
        {
            "field_id": "store_name",
            "field_label": "Store Name",
            "section_id": "store_details",
            "section_label": "Store Details",
            "comparison_kind": "other",
        },
        {
            "field_id": "receipt_number",
            "field_label": "Receipt No.",
            "section_id": "store_details",
            "section_label": "Store Details",
            "comparison_kind": "other",
        },
        {
            "field_id": "cashier_name",
            "field_label": "Cashier",
            "section_id": "store_details",
            "section_label": "Store Details",
            "comparison_kind": "other",
        },
        {
            "field_id": "purchase_date",
            "field_label": "Purchase Date",
            "section_id": "purchase_info",
            "section_label": "Purchase Info",
            "comparison_kind": "other",
        },
        {
            "field_id": "member_id",
            "field_label": "Member ID",
            "section_id": "purchase_info",
            "section_label": "Purchase Info",
            "comparison_kind": "other",
        },
        {
            "field_id": "register_id",
            "field_label": "Register ID",
            "section_id": "purchase_info",
            "section_label": "Purchase Info",
            "comparison_kind": "other",
        },
        {
            "field_id": "items_total",
            "field_label": "Items Total",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "tax_amount",
            "field_label": "Tax",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "discount_amount",
            "field_label": "Discount",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "service_charge",
            "field_label": "Service Charge",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "bag_fee",
            "field_label": "Bag Fee",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "coupon_savings",
            "field_label": "Coupon Savings",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "deposit_amount",
            "field_label": "Deposit",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "rounding_amount",
            "field_label": "Rounding",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "pickup_fee",
            "field_label": "Pickup Fee",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
        {
            "field_id": "warranty_fee",
            "field_label": "Warranty Fee",
            "section_id": "totals",
            "section_label": "Totals",
            "comparison_kind": "amount",
        },
    ),
}
SECTIONED_DOCUMENT_AMOUNT_SECTION_BY_SCENE = {
    "form_sheet": ("fees", "Fees"),
    "invoice_sheet": ("billing_summary", "Billing Summary"),
    "receipt_sheet": ("totals", "Totals"),
}
SECTIONED_DOCUMENT_FIELD_COUNT_RANGE = [
    min(
        len(SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE["form_sheet"]),
        len(SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE["receipt_sheet"]),
    ),
    len(SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE["invoice_sheet"]),
]
_DEPARTMENTS: Tuple[str, ...] = (
    "Operations",
    "Finance",
    "Support",
    "Planning",
    "Enrollment",
)


def _build_form_sectioned_values(rng: Random) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Return visible values plus numeric fee amounts for form-sheet sectioned scenes."""

    applicant_name = sample_person_name(rng)
    registration_cents = rng.randint(5600, 15800)
    service_cents = rng.randint(1800, 7600)
    discount_cents = rng.randint(200, 1800)
    processing_cents = rng.randint(700, 3400)
    materials_cents = rng.randint(900, 5200)
    facility_cents = rng.randint(1100, 6800)
    record_cents = rng.randint(500, 2600)
    access_cents = rng.randint(600, 3100)
    priority_cents = rng.randint(800, 3900)
    late_cents = rng.randint(400, 2300)
    visible = {
        "applicant_name": applicant_name,
        "reference_id": sample_identifier(rng, prefix="REF", digits=5),
        "department": str(rng.choice(_DEPARTMENTS)),
        "reviewer_name": sample_person_name(rng),
        "contact_phone": sample_phone_number(rng),
        "contact_email": sample_email(rng, local_hint=applicant_name),
        "registration_fee": format_currency_from_cents(registration_cents),
        "service_fee": format_currency_from_cents(service_cents),
        "discount_amount": format_currency_from_cents(discount_cents),
        "processing_fee": format_currency_from_cents(processing_cents),
        "materials_fee": format_currency_from_cents(materials_cents),
        "facility_fee": format_currency_from_cents(facility_cents),
        "record_fee": format_currency_from_cents(record_cents),
        "access_fee": format_currency_from_cents(access_cents),
        "priority_fee": format_currency_from_cents(priority_cents),
        "late_fee": format_currency_from_cents(late_cents),
    }
    amounts = {
        "registration_fee": registration_cents,
        "service_fee": service_cents,
        "discount_amount": discount_cents,
        "processing_fee": processing_cents,
        "materials_fee": materials_cents,
        "facility_fee": facility_cents,
        "record_fee": record_cents,
        "access_fee": access_cents,
        "priority_fee": priority_cents,
        "late_fee": late_cents,
    }
    return visible, amounts


def _build_invoice_sectioned_values(rng: Random) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Return visible values plus numeric billing-summary amounts for invoice sectioned scenes."""

    vendor_name = sample_company_name(rng)
    issue_offset = rng.randint(0, 18)
    service_offset = issue_offset + rng.randint(2, 8)
    due_offset = service_offset + rng.randint(8, 24)
    issue_date = dt.date(2026, 3, 1) + dt.timedelta(days=issue_offset)
    service_date = dt.date(2026, 3, 1) + dt.timedelta(days=service_offset)
    due_date = dt.date(2026, 3, 1) + dt.timedelta(days=due_offset)
    subtotal_cents = rng.randint(12600, 33600)
    tax_cents = rng.randint(500, 2600)
    discount_cents = rng.randint(300, 2200)
    shipping_cents = rng.randint(800, 6200)
    handling_cents = rng.randint(600, 4200)
    service_credit_cents = rng.randint(400, 2800)
    platform_cents = rng.randint(700, 3600)
    archive_cents = rng.randint(500, 2400)
    visible = {
        "vendor_name": vendor_name,
        "customer_name": sample_person_name(rng),
        "invoice_number": sample_identifier(rng, prefix="INV", digits=5),
        "account_id": sample_identifier(rng, prefix="ACC", digits=4),
        "contact_email": sample_email(rng, local_hint=vendor_name),
        "issue_date": issue_date.strftime("%Y-%m-%d"),
        "service_date": service_date.strftime("%Y-%m-%d"),
        "due_date": due_date.strftime("%Y-%m-%d"),
        "subtotal_amount": format_currency_from_cents(subtotal_cents),
        "tax_amount": format_currency_from_cents(tax_cents),
        "discount_amount": format_currency_from_cents(discount_cents),
        "shipping_amount": format_currency_from_cents(shipping_cents),
        "handling_amount": format_currency_from_cents(handling_cents),
        "service_credit": format_currency_from_cents(service_credit_cents),
        "platform_amount": format_currency_from_cents(platform_cents),
        "archive_amount": format_currency_from_cents(archive_cents),
    }
    amounts = {
        "subtotal_amount": subtotal_cents,
        "tax_amount": tax_cents,
        "discount_amount": discount_cents,
        "shipping_amount": shipping_cents,
        "handling_amount": handling_cents,
        "service_credit": service_credit_cents,
        "platform_amount": platform_cents,
        "archive_amount": archive_cents,
    }
    return visible, amounts


def _build_receipt_sectioned_values(rng: Random) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Return visible values plus numeric total-section amounts for receipt sectioned scenes."""

    store_name = sample_company_name(rng)
    purchase_offset = rng.randint(0, 20)
    purchase_date = dt.date(2026, 4, 1) + dt.timedelta(days=purchase_offset)
    items_total_cents = rng.randint(2600, 9800)
    tax_cents = rng.randint(120, 980)
    discount_cents = rng.randint(100, 900)
    service_charge_cents = rng.randint(150, 1200)
    bag_fee_cents = rng.randint(50, 500)
    coupon_savings_cents = rng.randint(100, 1200)
    deposit_cents = rng.randint(100, 1000)
    rounding_cents = rng.randint(10, 150)
    pickup_fee_cents = rng.randint(75, 750)
    warranty_fee_cents = rng.randint(200, 1600)
    visible = {
        "store_name": store_name,
        "receipt_number": sample_identifier(rng, prefix="R", digits=5),
        "cashier_name": sample_person_name(rng),
        "purchase_date": purchase_date.strftime("%b %d, %Y"),
        "member_id": sample_identifier(rng, prefix="M", digits=4),
        "register_id": sample_identifier(rng, prefix="REG", digits=3),
        "items_total": format_currency_from_cents(items_total_cents),
        "tax_amount": format_currency_from_cents(tax_cents),
        "discount_amount": format_currency_from_cents(discount_cents),
        "service_charge": format_currency_from_cents(service_charge_cents),
        "bag_fee": format_currency_from_cents(bag_fee_cents),
        "coupon_savings": format_currency_from_cents(coupon_savings_cents),
        "deposit_amount": format_currency_from_cents(deposit_cents),
        "rounding_amount": format_currency_from_cents(rounding_cents),
        "pickup_fee": format_currency_from_cents(pickup_fee_cents),
        "warranty_fee": format_currency_from_cents(warranty_fee_cents),
    }
    amounts = {
        "items_total": items_total_cents,
        "tax_amount": tax_cents,
        "discount_amount": discount_cents,
        "service_charge": service_charge_cents,
        "bag_fee": bag_fee_cents,
        "coupon_savings": coupon_savings_cents,
        "deposit_amount": deposit_cents,
        "rounding_amount": rounding_cents,
        "pickup_fee": pickup_fee_cents,
        "warranty_fee": warranty_fee_cents,
    }
    return visible, amounts


_SECTIONED_DOCUMENT_VALUE_BUILDERS = {
    "form_sheet": _build_form_sectioned_values,
    "invoice_sheet": _build_invoice_sectioned_values,
    "receipt_sheet": _build_receipt_sectioned_values,
}


def build_sectioned_document_values(scene_variant: str, rng: Random) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Return visible values plus numeric amount fields for one sectioned document scene."""

    return _SECTIONED_DOCUMENT_VALUE_BUILDERS[str(scene_variant)](rng)


__all__ = [
    "SECTIONED_DOCUMENT_AMOUNT_SECTION_BY_SCENE",
    "SECTIONED_DOCUMENT_FIELD_COUNT_RANGE",
    "SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE",
    "SUPPORTED_SECTIONED_DOCUMENT_SCENE_VARIANTS",
    "build_sectioned_document_values",
]
