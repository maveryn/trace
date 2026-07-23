"""Objective-neutral trace serialization for dots-and-boxes games tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .rendering import RenderedDotsAndBoxesTaskContext
from .rules import box_drawn_side_counts, immediate_capture_edge_ids
from .state import DotsAndBoxesBoardShapeAxis, DotsAndBoxesBoardState, DotsAndBoxesIntegerAxis, DotsAndBoxesSceneAxes


def edge_specs_for_trace(board_state: DotsAndBoxesBoardState) -> list[dict[str, Any]]:
    """Return serialized edge state for trace inspection."""

    return [
        {
            "edge_id": str(edge.edge_id),
            "orientation": str(edge.orientation),
            "dot_start": [int(value) for value in edge.dot_start],
            "dot_end": [int(value) for value in edge.dot_end],
            "is_drawn": bool(edge.is_drawn),
            "is_highlighted": bool(edge.is_highlighted),
        }
        for edge in board_state.edges
    ]


def box_specs_for_trace(board_state: DotsAndBoxesBoardState) -> list[dict[str, Any]]:
    """Return serialized box state for trace inspection."""

    return [
        {
            "box_id": str(box.box_id),
            "row_index": int(box.row_index),
            "column_index": int(box.column_index),
            "edge_ids": [str(edge_id) for edge_id in box.edge_ids],
            "owner": str(getattr(box, "owner", "") or ""),
        }
        for box in board_state.boxes
    ]


def build_dots_and_boxes_common_trace_params(
    *,
    scene_axes: DotsAndBoxesSceneAxes,
    board_shape: DotsAndBoxesBoardShapeAxis,
    extra_params: Mapping[str, Any] | None = None,
    candidate_edge_count_axis: DotsAndBoxesIntegerAxis | None = None,
) -> dict[str, Any]:
    """Return shared prompt params plus task-owned params."""

    params: dict[str, Any] = {
        "scene_variant": str(scene_axes.scene_variant),
        "style_variant": str(scene_axes.style_variant),
        "scene_variant_probabilities": dict(scene_axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(scene_axes.style_variant_probabilities),
        "box_rows": int(board_shape.box_rows),
        "box_cols": int(board_shape.box_cols),
        "board_shape_probabilities": dict(board_shape.probabilities),
    }
    if candidate_edge_count_axis is not None:
        params.update(
            {
                "candidate_edge_count": int(candidate_edge_count_axis.value),
                "candidate_edge_count_support": [int(value) for value in candidate_edge_count_axis.support],
                "candidate_edge_count_probabilities": dict(candidate_edge_count_axis.probabilities),
            }
        )
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_dots_and_boxes_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    scene_axes: DotsAndBoxesSceneAxes,
    board_shape: DotsAndBoxesBoardShapeAxis,
    board_state: DotsAndBoxesBoardState,
    rendered_context: RenderedDotsAndBoxesTaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    answer_value: Any,
    candidate_edge_count_axis: DotsAndBoxesIntegerAxis | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble trace sections after task-specific answer/annotation binding."""

    rendered_scene = rendered_context.rendered_scene
    annotation_ids = tuple(str(item) for item in annotation_entity_ids)
    box_edge_map = {str(box.box_id): tuple(str(edge_id) for edge_id in box.edge_ids) for box in board_state.boxes}
    side_counts = box_drawn_side_counts(
        drawn_edge_ids=tuple(str(edge_id) for edge_id in board_state.drawn_edge_ids),
        box_edges=box_edge_map,
    )
    immediate_edges = immediate_capture_edge_ids(
        drawn_edge_ids=tuple(str(edge_id) for edge_id in board_state.drawn_edge_ids),
        box_edges=box_edge_map,
    )
    candidate_edge_count = None if candidate_edge_count_axis is None else int(candidate_edge_count_axis.value)
    if candidate_edge_count is None and board_state.candidate_edge_ids:
        candidate_edge_count = int(len(board_state.candidate_edge_ids))
    option_label_by_box_id = {str(box_id): str(label) for box_id, label in board_state.option_label_by_box_id}
    option_box_id_by_label = {str(label): str(box_id) for box_id, label in board_state.option_label_by_box_id}
    execution_trace = {
        "scene_variant": str(scene_axes.scene_variant),
        "style_variant": str(scene_axes.style_variant),
        "target_answer": answer_value,
        "answer": answer_value,
        "box_rows": int(board_state.box_rows),
        "box_cols": int(board_state.box_cols),
        "candidate_edge_count": candidate_edge_count,
        "highlighted_edge_id": str(board_state.highlighted_edge_id),
        "highlighted_edge_ids": [str(edge_id) for edge_id in board_state.highlighted_edge_ids],
        "drawn_edge_ids": [str(edge_id) for edge_id in board_state.drawn_edge_ids],
        "captured_box_ids": [str(box_id) for box_id in board_state.captured_box_ids],
        "counted_box_ids": [str(box_id) for box_id in board_state.counted_box_ids],
        "counted_edge_ids": [str(edge_id) for edge_id in board_state.counted_edge_ids],
        "candidate_edge_ids": [str(edge_id) for edge_id in board_state.candidate_edge_ids],
        "answer_box_id": str(board_state.answer_box_id),
        "answer_label": str(board_state.answer_label),
        "option_box_id_by_label": dict(sorted(option_box_id_by_label.items())),
        "option_label_by_box_id": dict(sorted(option_label_by_box_id.items())),
        "box_owner_by_id": dict(rendered_scene.render_map.get("box_owner_by_id", {})),
        "immediate_capture_edge_ids": [str(edge_id) for edge_id in immediate_edges],
        "box_drawn_side_counts": {str(box_id): int(count) for box_id, count in sorted(side_counts.items())},
        "path_box_ids": [str(box_id) for box_id in board_state.path_box_ids],
        "move_edge_sequence": [str(edge_id) for edge_id in board_state.move_edge_sequence],
        "branching_edge_ids": [str(edge_id) for edge_id in board_state.branching_edge_ids],
        "path_turn_count": int(board_state.path_turn_count),
        "edge_specs": edge_specs_for_trace(board_state),
        "box_specs": box_specs_for_trace(board_state),
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_ids],
        **dict(execution_extra or {}),
    }
    return {
        "scene_ir": {
            "scene_kind": f"games_dots_and_boxes_{str(scene_axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(scene_axes.scene_variant),
                "style_variant": str(scene_axes.style_variant),
                "target_answer": answer_value,
                "answer": answer_value,
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_ids],
                "box_rows": int(board_shape.box_rows),
                "box_cols": int(board_shape.box_cols),
                "candidate_edge_count": candidate_edge_count,
                "highlighted_edge_id": str(board_state.highlighted_edge_id),
                "highlighted_edge_ids": [str(edge_id) for edge_id in board_state.highlighted_edge_ids],
                "answer_box_id": str(board_state.answer_box_id),
                "answer_label": str(board_state.answer_label),
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(scene_axes.scene_variant),
            "style_variant": str(scene_axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "box_rows": int(board_state.box_rows),
            "box_cols": int(board_state.box_cols),
            "candidate_edge_count": candidate_edge_count,
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "box_specs_for_trace",
    "build_dots_and_boxes_common_trace_params",
    "build_dots_and_boxes_trace_payload",
    "edge_specs_for_trace",
]
