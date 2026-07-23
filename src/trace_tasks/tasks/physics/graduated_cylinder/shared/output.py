"""Trace metadata helpers for graduated-cylinder outputs."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .prompts import SCENE_PROMPT_KEY
from .state import RenderedCylinderScene


def build_font_trace(*, font_family: str) -> dict[str, Any]:
    """Return font metadata for visible graduated-cylinder text."""

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


def build_render_spec(rendered: RenderedCylinderScene) -> dict[str, Any]:
    """Return renderer metadata without public task or query routing."""

    return {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "font": build_font_trace(font_family=str(rendered.font_family)),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def build_object_witness(ids: list[str]) -> dict[str, Any]:
    """Return object-map symbolic witness metadata."""

    return {
        "type": "object_map",
        "ids": list(ids),
        "key_to_entity_id": {str(key): str(key) for key in ids},
    }
