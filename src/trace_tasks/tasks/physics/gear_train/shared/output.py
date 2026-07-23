"""Trace metadata helpers for gear-train outputs."""

from __future__ import annotations

from typing import Any, Sequence

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record


def build_font_trace(*, font_family: str, prompt_scope: str) -> dict[str, Any]:
    """Return font metadata for visible gear labels."""

    font_record = get_font_family_record(str(font_family))
    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": str(prompt_scope),
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


def build_render_spec(
    *,
    rendered: Any,
    prompt_scope: str,
) -> dict[str, Any]:
    """Return renderer metadata common to gear-train objectives."""

    return {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "font": build_font_trace(font_family=str(rendered.font_family), prompt_scope=str(prompt_scope)),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def build_object_witness(*, ids: Sequence[str]) -> dict[str, Any]:
    """Return object witness ids for bbox-map annotation."""

    return {
        "type": "object_map",
        "ids": [str(item) for item in ids],
    }


__all__ = ["build_font_trace", "build_object_witness", "build_render_spec"]
