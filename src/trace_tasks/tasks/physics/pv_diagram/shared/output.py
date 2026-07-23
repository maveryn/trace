"""Trace metadata helpers for pressure-volume diagrams."""

from __future__ import annotations

from typing import Any, Sequence

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .prompts import SCENE_PROMPT_KEY


def build_font_trace(*, font_family: str) -> dict[str, Any]:
    """Return font metadata for visible pressure-volume diagram text."""

    font_record = get_font_family_record(str(font_family))
    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": SCENE_PROMPT_KEY,
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


def object_witness(entity_ids: Sequence[str]) -> dict[str, Any]:
    """Return symbolic witness metadata for the annotated PV object region."""

    return {"type": "object_set", "ids": [str(item) for item in entity_ids]}


__all__ = ["build_font_trace", "object_witness"]
