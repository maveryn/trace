"""Trace payload helpers for Sixteen Soldiers tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import (
    JUMP_SPECS,
    all_point_ids,
    board_to_dict,
    capture_lines,
    legal_destinations,
    piece_to_entity_id,
    player_name,
    point_coord,
    visible_board_trace,
)
from .state import (
    BLUE,
    EMPTY,
    RED,
    SCENE_ID,
    RenderedSixteenSoldiersScene,
    SixteenSoldiersSample,
    SixteenSoldiersTargetAxis,
    SixteenSoldiersVisualAxes,
)


def build_sixteen_soldiers_common_trace_params(
    *,
    axes: SixteenSoldiersVisualAxes,
    target_axis: SixteenSoldiersTargetAxis,
    branch_probabilities: Mapping[str, float],
    prompt_query_key: str,
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return prompt-query params common to Sixteen Soldiers outputs."""

    params: dict[str, Any] = {
        "prompt_query_key": str(prompt_query_key),
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "target_answer": int(target_axis.target_answer),
        "target_answer_support": [int(value) for value in target_axis.target_answer_support],
        "target_answer_probabilities": dict(target_axis.target_answer_probabilities),
        "marked_piece_color": player_name(int(axes.marked_piece_color)),
        "marked_piece_color_probabilities": dict(axes.marked_piece_color_probabilities),
        "piece_count_per_side": int(axes.piece_count_per_side),
        "piece_count_per_side_probabilities": dict(axes.piece_count_per_side_probabilities),
        "branch_probabilities": dict(branch_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def _point_state_name(value: int) -> str:
    """Return trace-facing point occupancy state."""

    if int(value) == EMPTY:
        return "empty"
    if int(value) == RED:
        return "red"
    if int(value) == BLUE:
        return "blue"
    raise ValueError(f"unsupported Sixteen Soldiers point value: {value!r}")


def build_sixteen_soldiers_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    annotation_kind: str,
    sample: SixteenSoldiersSample,
    axes: SixteenSoldiersVisualAxes,
    rendered_scene: RenderedSixteenSoldiersScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    prompt_query_key: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble objective-neutral trace payload sections after task binding."""

    values = board_to_dict(sample.board)
    marked_capture_lines = capture_lines(sample.board, sample.marked_point_id)
    execution_payload = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "prompt_query_key": str(prompt_query_key),
        "point_values": [
            {
                "point_id": str(point_id),
                "coord": [int(value) for value in point_coord(point_id)],
                "state": _point_state_name(int(values[point_id])),
            }
            for point_id in all_point_ids()
        ],
        "visible_board": visible_board_trace(sample.board),
        "construction_mode": str(sample.construction_mode),
        "target_answer": int(sample.answer),
        "target_color": player_name(int(sample.target_color)),
        "marked_point_id": str(sample.marked_point_id),
        "marked_coord": [int(value) for value in point_coord(sample.marked_point_id)],
        "marked_piece_id": piece_to_entity_id(sample.marked_point_id),
        "piece_count_per_side": int(axes.piece_count_per_side),
        "annotation_kind": str(annotation_kind),
        "annotation_point_ids": [str(point_id) for point_id in sample.annotation_point_ids],
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
        "legal_destinations_for_marked_piece": [
            str(point_id) for point_id in legal_destinations(sample.board, sample.marked_point_id)
        ],
        "capture_lines_for_marked_piece": [dict(item) for item in marked_capture_lines],
        "all_jump_specs": [
            {
                "origin_id": str(spec.origin_id),
                "middle_id": str(spec.middle_id),
                "landing_id": str(spec.landing_id),
            }
            for spec in JUMP_SPECS
        ],
    }
    if execution_extra:
        execution_payload.update(dict(execution_extra))

    return {
        "scene_ir": {
            "scene_kind": "games_sixteen_soldiers_board",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "prompt_query_key": str(prompt_query_key),
                "target_answer": int(sample.answer),
                "target_color": player_name(int(sample.target_color)),
                "marked_point_id": str(sample.marked_point_id),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style") or {}),
            "effective_board_height_px": float(rendered_scene.render_map["board_bbox_px"][3])
            - float(rendered_scene.render_map["board_bbox_px"][1]),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": execution_payload,
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_type),
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "prompt_defaults": {
            "bundle_id": str(prompt_defaults.get("bundle_id", "")),
            "scene_key": str(prompt_defaults.get("scene_key", "")),
            "task_key": str(prompt_defaults.get("task_key", "")),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = [
    "build_sixteen_soldiers_common_trace_params",
    "build_sixteen_soldiers_trace_payload",
]
