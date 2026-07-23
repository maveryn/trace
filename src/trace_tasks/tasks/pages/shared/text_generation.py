"""Typed text generators for synthetic structured-document fields."""

from __future__ import annotations

import datetime as dt
import re
from random import Random
from typing import Dict, Tuple

from ...shared.name_assets import load_name_manifest


_FIRST_NAMES = load_name_manifest(asset_group="pages", manifest_name="first_names.txt")
_LAST_NAMES = load_name_manifest(asset_group="pages", manifest_name="last_names.txt")
_COMPANY_ROOTS = load_name_manifest(asset_group="pages", manifest_name="company_roots.txt")
_CITY_NAMES = load_name_manifest(asset_group="pages", manifest_name="city_names.txt")

_COMPANY_SUFFIXES: Tuple[str, ...] = (
    "Co.",
    "Studio",
    "Supply",
    "Works",
    "Office",
    "Partners",
    "Market",
)
_EMAIL_DOMAINS: Tuple[str, ...] = (
    "example.com",
    "mailhub.net",
    "northpost.co",
    "brightdesk.io",
)
_DEPARTMENTS: Tuple[str, ...] = (
    "Operations",
    "Finance",
    "Support",
    "Planning",
    "Facilities",
    "Enrollment",
)


def _slugify_token(text: str) -> str:
    """Return one compact ASCII token for email-style fields."""

    token = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    return token or "field"


def sample_person_name(rng: Random) -> str:
    """Return one short visible person name."""

    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def sample_company_name(rng: Random) -> str:
    """Return one short company/vendor/store name."""

    return f"{rng.choice(_COMPANY_ROOTS)} {rng.choice(_COMPANY_SUFFIXES)}"


def sample_city_name(rng: Random) -> str:
    """Return one short city name."""

    return str(rng.choice(_CITY_NAMES))


def sample_identifier(rng: Random, *, prefix: str, digits: int) -> str:
    """Return one deterministic short identifier."""

    number = rng.randint(10 ** max(1, int(digits) - 1), (10 ** int(digits)) - 1)
    return f"{str(prefix)}-{int(number)}"


def sample_phone_number(rng: Random) -> str:
    """Return one compact US-style phone number."""

    area_code = rng.choice((212, 310, 415, 503, 617, 704, 818, 929))
    suffix = rng.randint(1000, 9999)
    exchange = rng.randint(100, 999)
    return f"({int(area_code)}) {int(exchange)}-{int(suffix):04d}"


def sample_email(rng: Random, *, local_hint: str) -> str:
    """Return one short email address derived from the provided hint."""

    local = _slugify_token(str(local_hint))
    if len(local) > 14:
        local = local[:14]
    return f"{local}@{rng.choice(_EMAIL_DOMAINS)}"


def format_currency_from_cents(cents: int) -> str:
    """Return one visible currency string from integer cents."""

    return f"${float(int(cents)) / 100.0:.2f}"


def sample_currency_amount(rng: Random, *, min_cents: int, max_cents: int) -> str:
    """Return one currency amount in `$12.34` format."""

    cents = rng.randint(int(min_cents), int(max_cents))
    return format_currency_from_cents(int(cents))


def sample_date(rng: Random, *, style: str, start: dt.date | None = None, day_offset_range: Tuple[int, int] = (0, 180)) -> str:
    """Return one visible date string in the requested style."""

    anchor = start or dt.date(2026, 1, 1)
    offset = rng.randint(int(day_offset_range[0]), int(day_offset_range[1]))
    value = anchor + dt.timedelta(days=int(offset))
    if str(style) == "iso":
        return value.strftime("%Y-%m-%d")
    if str(style) == "slash":
        return value.strftime("%m/%d/%Y")
    if str(style) == "long":
        return value.strftime("%b %d, %Y")
    raise ValueError(f"unsupported date style '{style}'")


def build_form_field_values(rng: Random) -> Dict[str, str]:
    """Return one coherent application-form field-value set."""

    applicant_name = sample_person_name(rng)
    reviewer_name = sample_person_name(rng)
    return {
        "applicant_name": applicant_name,
        "reference_id": sample_identifier(rng, prefix="REF", digits=5),
        "submission_date": sample_date(rng, style="slash"),
        "contact_phone": sample_phone_number(rng),
        "contact_email": sample_email(rng, local_hint=applicant_name),
        "department": str(rng.choice(_DEPARTMENTS)),
        "city": sample_city_name(rng),
        "fee_amount": sample_currency_amount(rng, min_cents=2500, max_cents=14900),
        "reviewer_name": reviewer_name,
    }


def build_invoice_field_values(rng: Random) -> Dict[str, str]:
    """Return one coherent invoice-style field-value set."""

    vendor_name = sample_company_name(rng)
    customer_name = sample_person_name(rng)
    issue_anchor = dt.date(2026, 2, 1)
    issue_date = sample_date(rng, style="iso", start=issue_anchor, day_offset_range=(0, 60))
    due_date_obj = dt.datetime.strptime(issue_date, "%Y-%m-%d").date() + dt.timedelta(days=rng.randint(7, 30))
    tax_cents = rng.randint(350, 2400)
    subtotal_cents = rng.randint(9800, 32800)
    total_cents = subtotal_cents + tax_cents
    return {
        "vendor_name": vendor_name,
        "customer_name": customer_name,
        "invoice_number": sample_identifier(rng, prefix="INV", digits=5),
        "account_id": sample_identifier(rng, prefix="ACC", digits=4),
        "issue_date": issue_date,
        "due_date": due_date_obj.strftime("%Y-%m-%d"),
        "contact_email": sample_email(rng, local_hint=vendor_name),
        "contact_phone": sample_phone_number(rng),
        "city": sample_city_name(rng),
        "tax_amount": f"${float(tax_cents) / 100.0:.2f}",
        "total_due": f"${float(total_cents) / 100.0:.2f}",
    }


def build_receipt_field_values(rng: Random) -> Dict[str, str]:
    """Return one coherent receipt-style field-value set."""

    store_name = sample_company_name(rng)
    cashier_name = sample_person_name(rng)
    tax_cents = rng.randint(120, 980)
    base_cents = rng.randint(1800, 9200)
    total_cents = base_cents + tax_cents
    return {
        "store_name": store_name,
        "receipt_number": sample_identifier(rng, prefix="R", digits=5),
        "purchase_date": sample_date(rng, style="long"),
        "cashier_name": cashier_name,
        "member_id": sample_identifier(rng, prefix="M", digits=4),
        "contact_phone": sample_phone_number(rng),
        "contact_email": sample_email(rng, local_hint=store_name),
        "tax_amount": f"${float(tax_cents) / 100.0:.2f}",
        "total_paid": f"${float(total_cents) / 100.0:.2f}",
    }


__all__ = [
    "build_form_field_values",
    "build_invoice_field_values",
    "build_receipt_field_values",
    "format_currency_from_cents",
    "sample_city_name",
    "sample_company_name",
    "sample_currency_amount",
    "sample_date",
    "sample_email",
    "sample_identifier",
    "sample_person_name",
    "sample_phone_number",
]
