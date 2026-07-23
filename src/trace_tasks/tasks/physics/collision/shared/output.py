"""Trace assembly helpers for the physics collision scene."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .prompts import (
    PROMPT_BUNDLE_ID,
    SCENE_PROMPT_KEY,
    aftermath_prompt_slots,
    build_collision_prompt_artifacts,
    sticky_direction_prompt_slots,
)
from .sampling import component_support, mass_support, speed_answer_tenths_support, speed_support
from .state import ANNOTATION_ENTITY_KEY_BY_ID


def build_font_trace(*, font_family: str, prompt_scope: str) -> dict[str, Any]:
    """Return reusable font trace metadata for one rendered collision diagram."""

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
    scene_variant: str,
    accent_color_name: str,
    prompt_scope: str,
) -> dict[str, Any]:
    """Assemble renderer metadata common to all collision objectives."""

    return {
        "scene_variant": str(scene_variant),
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "accent_color_name": str(accent_color_name),
        "font": build_font_trace(
            font_family=str(rendered.font_family),
            prompt_scope=str(prompt_scope),
        ),
        "technical_diagram_style": dict(rendered.diagram_style_meta),
        "background_style": dict(rendered.background_meta),
        "layout_placement": dict(rendered.layout_placement_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def build_object_key_witness(
    *,
    ids: Sequence[str],
    keys: Mapping[str, str],
) -> dict[str, Any]:
    """Represent semantic object ids and their annotation role keys."""

    return {
        "type": "object_key_map",
        "ids": [str(item) for item in ids],
        "keys": {str(key): str(value) for key, value in keys.items()},
    }


def build_bbox_projection(annotation_value: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Project keyed bounding boxes into the canonical trace shape."""

    bbox_map = {str(key): list(value) for key, value in annotation_value.items()}
    return {
        "type": "bbox_map",
        "bbox_map": bbox_map,
        "pixel_bbox_map": dict(bbox_map),
    }


def build_point_projection(annotation_value: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Project keyed points into the canonical trace shape."""

    point_map = {str(key): list(value) for key, value in annotation_value.items()}
    return {
        "type": "point_map",
        "point_map": point_map,
        "pixel_point_map": dict(point_map),
    }


def build_segment_projection(annotation_value: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Project one segment into the canonical trace shape."""

    segment = [list(point) for point in annotation_value]
    return {
        "type": "segment",
        "segment": segment,
        "pixel_segment": [list(point) for point in segment],
    }


def build_segment_set_projection(annotation_value: Sequence[Sequence[Sequence[float]]]) -> dict[str, Any]:
    """Project unordered segments into the canonical trace shape."""

    segments = [[list(point) for point in segment] for segment in annotation_value]
    return {
        "type": "segment_set",
        "segment_set": segments,
        "pixel_segment_set": [[list(point) for point in segment] for segment in segments],
    }


def build_annotation_projection(annotation_gt: Any) -> dict[str, Any]:
    """Project the supported sticky-collision annotation geometry."""

    annotation_type = str(annotation_gt.type)
    if annotation_type == "point_map":
        return build_point_projection(annotation_gt.value)
    if annotation_type == "segment":
        return build_segment_projection(annotation_gt.value)
    if annotation_type == "segment_set":
        return build_segment_set_projection(annotation_gt.value)
    raise ValueError(f"unsupported sticky-collision annotation type: {annotation_type}")


def build_sticky_scenario_payload(
    *,
    scenario: Any,
    option_angles_degrees: Mapping[str, float],
    correct_option_letter: str,
    direction_label: str,
    component_axis: str | None,
) -> dict[str, Any]:
    """Serialize one perpendicular sticky-collision setup for verifier tracing."""

    axis = None if component_axis is None else str(component_axis)
    return {
        "horizontal_mass": int(scenario.horizontal_mass),
        "vertical_mass": int(scenario.vertical_mass),
        "horizontal_speed": int(scenario.horizontal_speed),
        "vertical_speed": int(scenario.vertical_speed),
        "horizontal_direction": "right" if int(scenario.horizontal_sign) > 0 else "left",
        "vertical_direction": "up" if int(scenario.vertical_sign) > 0 else "down",
        "horizontal_momentum": int(
            scenario.horizontal_mass * scenario.horizontal_speed * scenario.horizontal_sign
        ),
        "vertical_momentum": int(
            scenario.vertical_mass * scenario.vertical_speed * scenario.vertical_sign
        ),
        "total_mass": int(scenario.total_mass),
        "final_vx": int(scenario.final_vx),
        "final_vy": int(scenario.final_vy),
        "direction_angle_degrees": float(option_angles_degrees[str(correct_option_letter)]),
        "direction_label": str(direction_label),
        "final_speed": float(math.hypot(float(scenario.final_vx), float(scenario.final_vy))),
        "final_speed_rounded": float(
            round(math.hypot(float(scenario.final_vx), float(scenario.final_vy)) + 1e-9, 1)
        ),
        "correct_option_letter": str(correct_option_letter),
        "option_letters": list(option_angles_degrees.keys()),
        "option_angles_degrees": dict(option_angles_degrees),
        "component_axis": axis,
        "component_axis_label": None
        if axis is None
        else ("horizontal" if axis == "x" else "vertical"),
    }


def build_sticky_support_payload(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    component_axis: str | None = None,
    scenario: Any | None = None,
) -> dict[str, Any]:
    """Return support metadata shared by sticky-collision objectives."""

    payload: dict[str, Any] = {
        "component_answer_support": list(component_support(params, defaults)),
        "speed_answer_tenths_support": list(speed_answer_tenths_support(params, defaults)),
        "speed_answer_support": [
            f"{float(value) / 10.0:.1f}"
            for value in speed_answer_tenths_support(params, defaults)
        ],
        "mass_support": list(mass_support(params, defaults)),
        "speed_support": list(speed_support(params, defaults)),
        "annotation_key_by_entity_id": dict(ANNOTATION_ENTITY_KEY_BY_ID),
    }
    if component_axis is not None and scenario is not None:
        axis = str(component_axis)
        payload.update(
            {
                "component_axis": axis,
                "component_axis_label": "horizontal" if axis == "x" else "vertical",
                "final_vx": int(scenario.final_vx),
                "final_vy": int(scenario.final_vy),
            }
        )
    return payload


def build_aftermath_scenario_payload(*, spec: Any) -> dict[str, Any]:
    """Serialize the visible aftermath direction and option mapping."""

    return {
        "final_motion_direction": str(spec.final_motion_direction),
        "correct_option_letter": str(spec.correct_option_letter),
        "correct_incoming_direction": str(spec.option_directions[str(spec.correct_option_letter)]),
        "option_directions": dict(spec.option_directions),
        "option_angles_degrees": dict(spec.option_angles_degrees),
        "candidate_rule": "incoming_path_vector_matches_observed_target_after_motion_vector",
    }


def build_aftermath_query_params(
    *,
    operation_kind: str,
    spec: Any,
    axes: Any,
    branch_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Return prompt-query parameters for a visible aftermath option task."""

    return {
        "operation_kind": str(operation_kind),
        "scene_variant": str(spec.scene_variant),
        "final_motion_direction": str(spec.final_motion_direction),
        "accent_color_name": str(axes.accent_color_name),
        "correct_option_letter": str(spec.correct_option_letter),
        "query_id_probabilities": dict(branch_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "final_motion_direction_probabilities": dict(axes.final_motion_direction_probabilities),
        "correct_option_letter_probabilities": dict(axes.correct_option_letter_probabilities),
        "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
    }


def build_aftermath_trace_payload(
    *,
    operation_kind: str,
    public_branch: str,
    spec: Any,
    axes: Any,
    rendered: Any,
    query_spec: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    scenario_payload: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble aftermath trace sections from task-bound scenario metadata."""

    annotation_ids = ("impact_point", "target_after_motion")
    return build_trace_payload(
        scene_kind=f"physics_collision_aftermath_{str(spec.scene_variant)}",
        rendered=rendered,
        query_spec=query_spec,
        render_spec=render_spec,
        scene_relations={
            "scene_variant": str(spec.scene_variant),
            "operation_kind": str(operation_kind),
            "public_branch": str(public_branch),
            "accent_color_name": str(axes.accent_color_name),
            "scenario": dict(scenario_payload),
            "annotation_entity_ids": list(annotation_ids),
        },
        execution_trace={
            "operation_kind": str(operation_kind),
            "public_branch": str(public_branch),
            "scene_variant": str(spec.scene_variant),
            "final_motion_direction": str(spec.final_motion_direction),
            "accent_color_name": str(axes.accent_color_name),
            "correct_option_letter": str(spec.correct_option_letter),
            "answer_type": "option_letter",
            "option_letters": list(spec.option_directions.keys()),
            "scenario": dict(scenario_payload),
            "annotation_entity_ids": list(annotation_ids),
        },
        witness_symbolic=build_object_key_witness(
            ids=annotation_ids,
            keys={"impact_point": "impact_point", "target_after_motion": "target_after_motion"},
        ),
        projected_annotation=projected_annotation,
    )


def build_sticky_query_params(
    *,
    operation_kind: str,
    axes: Any,
    answer_value: int | str,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return prompt-query parameters for a perpendicular sticky-collision task."""

    params = {
        "operation_kind": str(operation_kind),
        "scene_variant": str(axes.scene_variant),
        "accent_color_name": str(axes.accent_color_name),
        "correct_option_letter": str(axes.correct_option_letter),
        "target_answer": answer_value,
        "query_id_probabilities": dict(branch_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
        "correct_option_letter_probabilities": dict(axes.correct_option_letter_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_sticky_trace_payload(
    *,
    operation_kind: str,
    public_branch: str,
    axes: Any,
    rendered: Any,
    query_spec: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    scenario_payload: Mapping[str, Any],
    answer_value: int | str,
    answer_type: str,
    support_payload: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    annotation_entity_ids: Sequence[str] | None = None,
    witness_symbolic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble common sticky-collision trace sections around task-owned values."""

    annotation_ids = (
        [str(entity_id) for entity_id in annotation_entity_ids]
        if annotation_entity_ids is not None
        else list(rendered.annotation_entity_ids)
    )
    common_trace = {
        "operation_kind": str(operation_kind),
        "public_branch": str(public_branch),
        "scene_variant": str(axes.scene_variant),
        "accent_color_name": str(axes.accent_color_name),
        "target_answer": answer_value,
        "answer_type": str(answer_type),
        **dict(support_payload),
        "scenario": dict(scenario_payload),
        "annotation_entity_ids": list(annotation_ids),
        "annotation_key_by_entity_id": dict(support_payload.get("annotation_key_by_entity_id", {})),
    }
    common_trace["annotation_key_by_entity_id"] = dict(
        support_payload.get("annotation_key_by_entity_id", {})
    )
    scene_relations = {
        "scene_variant": str(axes.scene_variant),
        "operation_kind": str(operation_kind),
        "public_branch": str(public_branch),
        "accent_color_name": str(axes.accent_color_name),
        "target_answer": answer_value,
        "answer_type": str(answer_type),
        "scenario": dict(scenario_payload),
        "annotation_entity_ids": list(annotation_ids),
        "annotation_key_by_entity_id": dict(common_trace["annotation_key_by_entity_id"]),
    }
    return build_trace_payload(
        scene_kind=f"physics_sticky_collision_{str(axes.scene_variant)}",
        rendered=rendered,
        query_spec=query_spec,
        render_spec=render_spec,
        scene_relations=scene_relations,
        execution_trace=common_trace,
        witness_symbolic=dict(witness_symbolic)
        if witness_symbolic is not None
        else build_object_key_witness(
            ids=annotation_ids,
            keys=common_trace["annotation_key_by_entity_id"],
        ),
        projected_annotation=projected_annotation,
    )


def build_trace_payload(
    *,
    scene_kind: str,
    rendered: Any,
    query_spec: Mapping[str, Any],
    render_spec: Mapping[str, Any],
    scene_relations: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine task-owned trace fields with common rendered-scene sections."""

    return {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": dict(scene_relations),
        },
        "query_spec": dict(query_spec),
        "render_spec": dict(render_spec),
        "render_map": dict(rendered.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def build_query_spec(
    *,
    prompt_artifacts: Any,
    branch: str,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach rendered prompt metadata to task-owned query parameters."""

    return build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(branch),
        params=dict(params),
    )


def build_task_output_kwargs(
    *,
    prompt_artifacts: Any,
    rendered: Any,
    answer_gt: Any,
    annotation_gt: Any,
    trace_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return common keyword arguments consumed by public task constructors."""

    return {
        "prompt": str(prompt_artifacts.prompt),
        "prompt_variants": dict(prompt_artifacts.prompt_variants),
        "answer_gt": answer_gt,
        "annotation_gt": annotation_gt,
        "image": rendered.image,
        "image_id": "img0",
        "trace_payload": dict(trace_payload),
        "task_versions": default_task_versions(),
    }


def build_aftermath_output_kwargs(
    domain: str,
    prompt_defaults: Mapping[str, Any],
    task_key: str,
    prompt_query_key: str,
    instance_seed: int,
    branch: str,
    branch_probabilities: Mapping[str, float],
    operation_kind: str,
    spec: Any,
    axes: Any,
    rendered: Any,
    answer_gt: Any,
    annotation_gt: Any,
    scene_id: str,
) -> dict[str, Any]:
    """Build output kwargs after public code binds the aftermath answer."""

    resolved_prompt = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "task_key"),
        context="collision aftermath prompt defaults",
    )
    prompt_artifacts = build_collision_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(resolved_prompt.get("bundle_id", PROMPT_BUNDLE_ID)),
        task_key=str(resolved_prompt.get("task_key", task_key)),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=aftermath_prompt_slots(str(spec.scene_variant)),
        instance_seed=int(instance_seed),
    )
    scenario_payload = build_aftermath_scenario_payload(spec=spec)
    query_spec = build_query_spec(
        prompt_artifacts=prompt_artifacts,
        branch=str(branch),
        params=build_aftermath_query_params(
            operation_kind=str(operation_kind),
            spec=spec,
            axes=axes,
            branch_probabilities=branch_probabilities,
        ),
    )
    render_spec = build_render_spec(
        rendered=rendered,
        scene_variant=str(spec.scene_variant),
        accent_color_name=str(axes.accent_color_name),
        prompt_scope=SCENE_PROMPT_KEY,
    )
    trace_payload = build_aftermath_trace_payload(
        operation_kind=str(operation_kind),
        public_branch=str(branch),
        spec=spec,
        axes=axes,
        rendered=rendered,
        query_spec=query_spec,
        render_spec=render_spec,
        scenario_payload=scenario_payload,
        projected_annotation=build_bbox_projection(annotation_gt.value),
    )
    return build_task_output_kwargs(
        prompt_artifacts=prompt_artifacts,
        rendered=rendered,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        trace_payload=trace_payload,
    )


def build_sticky_output_kwargs(
    domain: str,
    prompt_defaults: Mapping[str, Any],
    task_key: str,
    prompt_query_key: str,
    instance_seed: int,
    branch: str,
    branch_probabilities: Mapping[str, float],
    operation_kind: str,
    axes: Any,
    rendered: Any,
    scene_spec: Any,
    answer_gt: Any,
    annotation_gt: Any,
    answer_type: str,
    support_payload: Mapping[str, Any],
    scene_id: str,
    prompt_slots: Mapping[str, Any] | None = None,
    component_axis: str | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
    annotation_entity_ids: Sequence[str] | None = None,
    witness_symbolic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build output kwargs after public code binds a sticky-collision answer."""

    resolved_prompt = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "task_key"),
        context="collision sticky prompt defaults",
    )
    slots = (
        dict(prompt_slots)
        if prompt_slots is not None
        else sticky_direction_prompt_slots(str(scene_spec.scene_variant))
    )
    prompt_artifacts = build_collision_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(resolved_prompt.get("bundle_id", PROMPT_BUNDLE_ID)),
        task_key=str(resolved_prompt.get("task_key", task_key)),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=slots,
        instance_seed=int(instance_seed),
    )
    query_spec = build_query_spec(
        prompt_artifacts=prompt_artifacts,
        branch=str(branch),
        params=build_sticky_query_params(
            operation_kind=str(operation_kind),
            axes=axes,
            answer_value=answer_gt.value,
            branch_probabilities=branch_probabilities,
            extra_params=extra_query_params,
        ),
    )
    scenario_payload = build_sticky_scenario_payload(
        scenario=scene_spec.scenario,
        option_angles_degrees=rendered.render_map["option_angles_degrees"],
        correct_option_letter=str(scene_spec.correct_option_letter),
        direction_label=str(scene_spec.direction_label),
        component_axis=component_axis,
    )
    render_spec = build_render_spec(
        rendered=rendered,
        scene_variant=str(axes.scene_variant),
        accent_color_name=str(axes.accent_color_name),
        prompt_scope=SCENE_PROMPT_KEY,
    )
    trace_payload = build_sticky_trace_payload(
        operation_kind=str(operation_kind),
        public_branch=str(branch),
        axes=axes,
        rendered=rendered,
        query_spec=query_spec,
        render_spec=render_spec,
        scenario_payload=scenario_payload,
        answer_value=answer_gt.value,
        answer_type=str(answer_type),
        support_payload=support_payload,
        projected_annotation=build_annotation_projection(annotation_gt),
        annotation_entity_ids=annotation_entity_ids,
        witness_symbolic=witness_symbolic,
    )
    return build_task_output_kwargs(
        prompt_artifacts=prompt_artifacts,
        rendered=rendered,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        trace_payload=trace_payload,
    )


__all__ = [
    "build_aftermath_query_params",
    "build_aftermath_output_kwargs",
    "build_aftermath_scenario_payload",
    "build_aftermath_trace_payload",
    "build_annotation_projection",
    "build_bbox_projection",
    "build_object_key_witness",
    "build_point_projection",
    "build_query_spec",
    "build_render_spec",
    "build_segment_projection",
    "build_segment_set_projection",
    "build_sticky_query_params",
    "build_sticky_output_kwargs",
    "build_sticky_scenario_payload",
    "build_sticky_support_payload",
    "build_sticky_trace_payload",
    "build_task_output_kwargs",
    "build_trace_payload",
]
