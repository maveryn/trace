"""Trace metadata helpers for free-body-force outputs."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .prompts import SCENE_PROMPT_KEY
from .state import ForceScenario, RenderedForceScene, SamplingAxes


def build_font_trace(*, font_family: str) -> dict[str, Any]:
    """Return font metadata for visible force labels and options."""

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


def build_render_spec(*, rendered: RenderedForceScene, axes: SamplingAxes) -> dict[str, Any]:
    """Return renderer metadata without public task or query identity."""

    return {
        "scene_variant": str(axes.scene_variant),
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "accent_color_name": str(axes.accent_color_name),
        "font": build_font_trace(font_family=str(rendered.font_family)),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def force_payload(scenario: ForceScenario) -> list[dict[str, Any]]:
    """Return JSON-stable visible force metadata."""

    return [
        {
            "force_id": str(spec.force_id),
            "direction": str(spec.direction),
            "magnitude_n": int(spec.magnitude_n),
            "vector": [int(spec.vector[0]), int(spec.vector[1])],
        }
        for spec in scenario.force_specs
    ]
