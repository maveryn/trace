"""Trace metadata helpers for electrostatic-field outputs."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record

from .state import (
    DirectionAxes,
    OptionLetterAxes,
    PotentialAxes,
    RenderedElectrostaticScene,
    SceneAxes,
    SceneSpec,
)


def build_font_trace(*, font_family: str, scope: str) -> dict[str, Any]:
    """Return font metadata for visible electrostatic diagram text."""

    font_record = get_font_family_record(str(font_family))
    return {
        "font_family": str(font_family),
        "font_asset_version": font_asset_version(),
        "font_asset": font_record.to_trace(),
        "scope": str(scope),
        "selection_policy": {
            "pool": "global_approved_font_pool",
            "include_tags": [],
            "exclude_tags": [],
            "exclusion_reason": "",
        },
    }


def build_render_spec(rendered: RenderedElectrostaticScene, *, scope: str) -> dict[str, Any]:
    """Return renderer metadata without public task identity."""

    return {
        "scene_variant": str(rendered.render_map.get("scene_variant", "")),
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "font": build_font_trace(font_family=str(rendered.font_family), scope=str(scope)),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "background_style": dict(rendered.background_meta),
        "layout_placement": dict(rendered.layout_placement_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def projected_point_map(annotation_value: Mapping[str, list[float]]) -> dict[str, Any]:
    """Project a point-map annotation into the trace payload shape."""

    points = {str(key): list(value) for key, value in annotation_value.items()}
    return {
        "type": "point_map",
        "point_map": dict(points),
        "pixel_point_map": dict(points),
    }


def point_map_value(rendered: RenderedElectrostaticScene) -> dict[str, list[float]]:
    """Return role-keyed annotation points from the final rendered scene."""

    return {str(key): list(point) for key, point in rendered.annotation_point_map.items()}


def annotation_symbolic_payload(rendered: RenderedElectrostaticScene) -> dict[str, Any]:
    """Return role-to-entity trace metadata for projected point annotations."""

    return {
        "type": "object_key_map",
        "ids": list(rendered.annotation_entity_ids),
        "keys": dict(rendered.annotation_key_by_entity_id),
    }


def visual_trace_sections(
    rendered: RenderedElectrostaticScene,
    *,
    annotation_value: Mapping[str, list[float]],
    render_scope: str,
) -> dict[str, Any]:
    """Return renderer, annotation projection, and visual-style trace sections."""

    return {
        "render_spec": build_render_spec(rendered, scope=str(render_scope)),
        "render_map": dict(rendered.render_map),
        "witness_symbolic": annotation_symbolic_payload(rendered),
        "projected_annotation": projected_point_map(annotation_value),
        "background": dict(rendered.background_meta),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def scene_axis_params(scene_axes: SceneAxes) -> dict[str, Any]:
    """Return sampled scene-level axes for task-owned trace payloads."""

    return {
        "scene_variant": str(scene_axes.scene_variant),
        "accent_color_name": str(scene_axes.accent_color_name),
        "scene_variant_probabilities": dict(scene_axes.scene_variant_probabilities),
        "accent_color_name_probabilities": dict(scene_axes.accent_color_name_probabilities),
    }


def direction_scenario_payload(scene_spec: SceneSpec, rendered: RenderedElectrostaticScene) -> dict[str, Any]:
    """Return symbolic field-direction payload for task-owned trace metadata."""

    scenario = scene_spec.direction_scenario
    if scenario is None:
        return {}
    return {
        "charges": [
            {
                "charge_id": str(charge.charge_id),
                "display_label": str(rendered.annotation_key_by_entity_id.get(str(charge.charge_id), "")),
                "charge_value": int(charge.charge_value),
                "x": int(charge.x),
                "y": int(charge.y),
            }
            for charge in scenario.charges
        ],
        "point": {"label": "P", "x": int(scenario.point_x), "y": int(scenario.point_y)},
        "direction_mode": str(scenario.direction_mode),
        "test_charge_sign": scenario.test_charge_sign,
        "field_direction": str(scenario.field_direction),
        "requested_direction": str(scenario.requested_direction),
        "option_directions": dict(scenario.option_directions),
    }


def zero_field_scenario_payload(scene_spec: SceneSpec, rendered: RenderedElectrostaticScene) -> dict[str, Any]:
    """Return symbolic zero-field payload for task-owned trace metadata."""

    scenario = scene_spec.zero_field_scenario
    if scenario is None:
        return {}
    return {
        "charges": [
            {
                "charge_id": str(charge.charge_id),
                "display_label": str(rendered.annotation_key_by_entity_id.get(str(charge.charge_id), "")),
                "charge_value": int(charge.charge_value),
                "x": int(charge.x),
                "y": int(charge.y),
            }
            for charge in scenario.charges
        ],
        "candidate_points": [
            {
                "option_letter": str(point.letter),
                "x": int(point.x),
                "y": int(point.y),
                "is_correct": bool(point.is_correct),
            }
            for point in scenario.candidate_points
        ],
        "charge_axis": str(scenario.symmetry_axis),
        "symmetry_axis": str(scenario.symmetry_axis),
        "correct_option_letter": str(scenario.correct_option_letter),
    }


def potential_scenario_payload(scene_spec: SceneSpec, rendered: RenderedElectrostaticScene) -> dict[str, Any]:
    """Return symbolic potential payload for task-owned trace metadata."""

    scenario = scene_spec.potential_scenario
    if scenario is None:
        return {}
    return {
        "charges": [
            {
                "charge_id": str(charge.charge_id),
                "display_label": str(rendered.annotation_key_by_entity_id.get(str(charge.charge_id), "")),
                "charge_value": int(charge.charge_value),
                "distance_units": int(charge.distance_units),
                "potential_contribution": int(charge.contribution),
                "x": int(charge.x),
                "y": int(charge.y),
            }
            for charge in scenario.charges
        ],
        "point": {"label": "P", "x": int(scenario.point_x), "y": int(scenario.point_y)},
        "resolved_value": int(scenario.potential_value),
        "formula": "V=sum(q/r), k=1",
    }


def direction_axis_params(direction_axes: DirectionAxes) -> dict[str, Any]:
    """Return sampled field-direction axes for task-owned trace metadata."""

    return {
        "direction_mode": str(direction_axes.direction_mode),
        "target_direction": str(direction_axes.target_direction),
        "correct_option_letter": str(direction_axes.correct_option_letter),
        "direction_mode_probabilities": dict(direction_axes.direction_mode_probabilities),
        "target_direction_probabilities": dict(direction_axes.target_direction_probabilities),
        "correct_option_letter_probabilities": dict(direction_axes.correct_option_letter_probabilities),
    }


def option_letter_axis_params(option_axes: OptionLetterAxes) -> dict[str, Any]:
    """Return sampled option-letter axis fields for task-owned trace metadata."""

    return {
        "correct_option_letter": str(option_axes.correct_option_letter),
        "correct_option_letter_probabilities": dict(option_axes.correct_option_letter_probabilities),
    }


def potential_axis_params(potential_axes: PotentialAxes) -> dict[str, Any]:
    """Return sampled potential axes for task-owned trace metadata."""

    return {
        "target_answer": int(potential_axes.target_answer),
        "target_answer_probabilities": dict(potential_axes.target_answer_probabilities),
        "potential_answer_support": list(potential_axes.answer_support),
        "potential_contribution_support": list(potential_axes.contribution_support),
    }
