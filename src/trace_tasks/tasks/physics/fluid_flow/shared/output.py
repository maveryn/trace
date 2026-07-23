"""Trace metadata helpers for fluid-flow outputs."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .prompts import SCENE_PROMPT_KEY
from .state import FlowScenario, RenderedFlowScene


def build_font_trace(*, font_family: str) -> dict[str, Any]:
    """Return font metadata for visible fluid-flow text."""

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


def build_render_spec(rendered: RenderedFlowScene) -> dict[str, Any]:
    """Return renderer metadata without public task or query routing."""

    return {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "font": build_font_trace(font_family=str(rendered.font_family)),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def flow_scenario_params(scenario: FlowScenario) -> dict[str, Any]:
    """Return JSON-stable sampled parameter metadata for a flow scenario."""

    return {
        "missing_station": str(scenario.missing_station),
        "area_1_cm2": int(scenario.area_1_cm2),
        "area_2_cm2": int(scenario.area_2_cm2),
        "speed_1_m_s": int(scenario.speed_1_m_s),
        "speed_2_m_s": int(scenario.speed_2_m_s),
        "target_answer": int(scenario.target_answer),
        "orientation": str(scenario.orientation),
        "orientation_probabilities": dict(scenario.orientation_probabilities),
        "missing_station_probabilities": dict(scenario.missing_station_probabilities),
        "area_1_cm2_probabilities": dict(scenario.area_1_cm2_probabilities),
        "area_2_cm2_probabilities": dict(scenario.area_2_cm2_probabilities),
        "speed_1_m_s_probabilities": dict(scenario.speed_1_m_s_probabilities),
        "speed_2_m_s_probabilities": dict(scenario.speed_2_m_s_probabilities),
        "target_answer_probabilities": dict(scenario.target_answer_probabilities),
    }
