"""Objective-neutral trace payload helpers for Sokoban tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import json_safe
from .state import SCENE_ID, SokobanAxes, SokobanRenderParams, RenderedSokobanScene


def sokoban_trace_params(
    *,
    axes: SokobanAxes,
    dataset: Mapping[str, Any],
    prompt_query_key: str,
    answer_value: str,
    option_count_support: list[int],
    option_count_probabilities: Mapping[str, float],
    public_query_probabilities: Mapping[str, float],
    trace_extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return common query params plus task-owned metadata."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "option_count": int(dataset.get("option_count", len(dataset.get("option_specs", [])))),
        "option_count_support": [int(value) for value in option_count_support],
        "option_count_probabilities": dict(option_count_probabilities),
        "public_query_probabilities": dict(public_query_probabilities),
        "answer_value": str(answer_value),
        "move_count": int(len(dataset.get("move_sequence", []))),
    }
    if "answer_option_label" in dataset:
        params["answer_option_label"] = str(answer_value)
    if trace_extra_params:
        params.update(dict(trace_extra_params))
    return params


def build_sokoban_trace_payload(
    *,
    axes: SokobanAxes,
    dataset: Mapping[str, Any],
    rendered_scene: RenderedSokobanScene,
    render_params: SokobanRenderParams,
    prompt_query_key: str,
    answer_value: str,
    annotation_artifacts: AnnotationArtifacts,
    annotation_source: str,
    option_count_support: list[int],
    option_count_probabilities: Mapping[str, float],
    public_query_probabilities: Mapping[str, float],
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    trace_extra_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble Sokoban trace sections after task-owned binding."""

    execution_trace = {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "prompt_query_key": str(prompt_query_key),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "rows": int(dataset["rows"]),
        "cols": int(dataset["cols"]),
        "walls": json_safe(dataset.get("walls", [])),
        "player_start": json_safe(dataset.get("player_start")),
        "boxes_start": json_safe(dataset.get("boxes_start", {})),
        "targets": json_safe(dataset.get("targets", {})),
        "move_sequence": [str(item) for item in dataset.get("move_sequence", [])],
        "move_sequence_text": str(dataset.get("move_sequence_text", "")),
        "path_start": json_safe(dataset.get("path_start")),
        "path_goal": json_safe(dataset.get("path_goal")),
        "shortest_path_cells": json_safe(dataset.get("shortest_path_cells", [])),
        "query_entity_type": str(dataset.get("query_entity_type", "")),
        "query_entity_label": str(dataset.get("query_entity_label", "")),
        "answer_cell": json_safe(dataset.get("answer_cell")),
        "relation_support": json_safe(dataset.get("relation_support", {})),
        "option_specs": json_safe(dataset.get("option_specs", [])),
        "option_count": int(dataset.get("option_count", len(dataset.get("option_specs", [])))),
        "answer_value": json_safe(answer_value),
        "view_family": SCENE_ID,
        "solver_trace": json_safe(dataset.get("solver_trace", {})),
    }
    if "answer_option_label" in dataset:
        execution_trace["answer_option_label"] = str(answer_value)
    for key in (
        "matching_targets",
        "box_colors",
        "target_colors",
        "status_mode",
        "goal_status_count",
        "box_count",
        "boxes_on_matching_goals",
        "boxes_off_matching_goals",
        "counted_box_labels",
    ):
        if key in dataset:
            execution_trace[key] = json_safe(dataset[key])
    if execution_extra:
        execution_trace.update(dict(execution_extra))

    return {
        "scene_ir": {
            "scene_kind": "game_sokoban_grid",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "prompt_query_key": str(prompt_query_key),
                "answer_option_label": str(answer_value),
                "view_family": SCENE_ID,
            },
        },
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "scene_variant": str(axes.scene_variant),
            "panel_scene_style": dict(scene_style_meta),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "board_bbox_px": list(rendered_scene.board_bbox_px),
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        "render_map": {
            "image_id": "img0",
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "board_bbox_px": list(rendered_scene.board_bbox_px),
            "cell_bboxes_px": {str(key): list(value) for key, value in rendered_scene.cell_bbox_map.items()},
            "option_panel_bboxes_px": {
                str(key): list(value) for key, value in rendered_scene.option_panel_bbox_map.items()
            },
            "annotation_source": str(annotation_source),
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        "execution_trace": execution_trace,
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_type),
            "value": annotation_artifacts.value,
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "params_for_prompt": sokoban_trace_params(
            axes=axes,
            dataset=dataset,
            prompt_query_key=str(prompt_query_key),
            answer_value=str(answer_value),
            option_count_support=option_count_support,
            option_count_probabilities=option_count_probabilities,
            public_query_probabilities=public_query_probabilities,
            trace_extra_params=trace_extra_params,
        ),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = ["build_sokoban_trace_payload", "sokoban_trace_params"]
