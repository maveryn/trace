"""Trace metadata helpers for pulley outputs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .prompts import SCENE_PROMPT_KEY


def build_font_trace(*, font_family: str) -> dict[str, Any]:
    """Return font metadata for visible pulley text."""

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


def build_object_witness(key_to_entity_id: Mapping[str, str]) -> dict[str, Any]:
    """Return object-map symbolic witness metadata."""

    return {
        "type": "object_map",
        "ids": [str(value) for value in key_to_entity_id.values()],
        "key_to_entity_id": {str(key): str(value) for key, value in key_to_entity_id.items()},
    }


def cut_segment_records(cut_segments: Sequence[Any]) -> list[dict[str, Any]]:
    """Return JSON-stable records for cut non-supporting strand distractors."""

    return [
        {
            "segment_id": str(segment.segment_id),
            "attach_side": str(segment.attach_side),
            "slot_index": int(segment.x_order),
            "cut_fraction": float(segment.cut_fraction),
            "supports_moving_block": False,
        }
        for segment in cut_segments
    ]


__all__ = ["build_font_trace", "build_object_witness", "cut_segment_records"]
