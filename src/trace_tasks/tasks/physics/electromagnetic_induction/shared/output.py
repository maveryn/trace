"""Trace metadata helpers for electromagnetic induction outputs."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .state import InductionScenario, RenderedInductionScene


def build_font_trace(*, font_family: str) -> dict[str, Any]:
    """Return font metadata for visible induction panel text."""

    font_record = get_font_family_record(str(font_family))
    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": "electromagnetic_induction_panels",
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


def build_render_spec(rendered: RenderedInductionScene) -> dict[str, Any]:
    """Return renderer metadata without public task or query identity."""

    return {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "font": build_font_trace(font_family=str(rendered.font_family)),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def panel_current_classes(scenario: InductionScenario) -> dict[str, str]:
    """Map panel ids to derived current classes for verifier tracing."""

    return {str(panel.panel_id): str(panel.current_class) for panel in scenario.panels}


def panel_flux_changes(scenario: InductionScenario) -> dict[str, str]:
    """Map panel ids to symbolic flux-change classes."""

    return {str(panel.panel_id): str(panel.flux_change) for panel in scenario.panels}


def panel_field_orientations(scenario: InductionScenario) -> dict[str, str]:
    """Map panel ids to visible magnetic-field orientations."""

    return {str(panel.panel_id): str(panel.field_orientation) for panel in scenario.panels}


def panel_mechanisms(scenario: InductionScenario) -> dict[str, str]:
    """Map panel ids to visible flux-change cue mechanisms."""

    return {str(panel.panel_id): str(panel.mechanism) for panel in scenario.panels}


def projected_bbox_set(annotation_value: list[list[float]]) -> dict[str, Any]:
    """Project a bbox-set annotation into the trace payload shape."""

    boxes = [list(bbox) for bbox in annotation_value]
    return {
        "type": "bbox_set",
        "bbox_set": list(boxes),
        "pixel_bbox_set": list(boxes),
    }
