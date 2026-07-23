"""Objective-neutral trace assembly helpers for Bingo games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedBingoTaskContext
from .sampling import ResolvedBingoSceneAxes
from .state import BingoCardState


def build_bingo_common_trace_params(
    *,
    axes: ResolvedBingoSceneAxes,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared Bingo query params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "mark_shape": str(axes.mark_shape),
        "cell_fill_pattern": str(axes.cell_fill_pattern),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "mark_shape_probabilities": dict(axes.mark_shape_probabilities),
        "cell_fill_pattern_probabilities": dict(axes.cell_fill_pattern_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_bingo_trace_payload(
    *,
    axes: ResolvedBingoSceneAxes,
    card_state: BingoCardState,
    rendered_context: RenderedBingoTaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    answer_value: int | str,
    annotation_cell_ids: Sequence[str],
    annotation_cell_id_pairs: Sequence[Sequence[str]] = (),
    annotation_bboxes: Sequence[Sequence[float]],
    annotation_points: Sequence[Sequence[float]] = (),
    annotation_point_pairs: Sequence[Sequence[Sequence[float]]] = (),
    projected_annotation: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble objective-neutral trace sections after task-specific binding."""

    rendered_scene = rendered_context.rendered_scene
    completed_line_sums = [
        {
            "axis": str(axis_name),
            "line_index": int(line_index),
            "sum": int(value),
        }
        for axis_name, line_index, value in card_state.completed_line_sums
    ]
    return {
        "scene_ir": {
            "scene_kind": f"games_bingo_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "mark_shape": str(rendered_context.render_params.mark_shape),
                "cell_fill_pattern": str(rendered_context.render_params.cell_fill_pattern),
                "target_answer": answer_value,
                "line_sum_target_axis": card_state.line_sum_target_axis,
                "line_sum_target_line_index": card_state.line_sum_target_line_index,
                "line_sum_target_cell_ids": list(card_state.line_sum_target_cell_ids),
                "line_sum_target_value": card_state.line_sum_target_value,
                "near_complete_row_indices": list(card_state.near_complete_row_indices),
                "near_complete_column_indices": list(card_state.near_complete_column_indices),
                "near_complete_gap_cell_ids": list(card_state.near_complete_gap_cell_ids),
                "called_numbers": [int(value) for value in card_state.called_numbers],
                "called_number_cell_ids": [str(value) for value in card_state.called_number_cell_ids],
                "completed_line_sums": completed_line_sums,
                "annotation_entity_ids": [str(cell_id) for cell_id in annotation_cell_ids],
                "annotation_entity_id_pairs": [
                    [str(pair[0]), str(pair[1])]
                    for pair in annotation_cell_id_pairs
                ],
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "mark_shape": str(rendered_context.render_params.mark_shape),
            "cell_fill_pattern": str(rendered_context.render_params.cell_fill_pattern),
            "called_panel_width_px": int(rendered_context.render_params.called_panel_width_px),
            "called_panel_gap_px": int(rendered_context.render_params.called_panel_gap_px),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
            "effective_cell_size_px": rendered_scene.render_map.get("effective_cell_size_px"),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "mark_shape": str(rendered_context.render_params.mark_shape),
            "cell_fill_pattern": str(rendered_context.render_params.cell_fill_pattern),
            "target_answer": answer_value,
            "numbers_grid": [[int(value) for value in row] for row in card_state.numbers_grid],
            "mark_grid": [[bool(value) for value in row] for row in card_state.mark_grid],
            "completed_row_indices": [int(value) for value in card_state.completed_row_indices],
            "completed_column_indices": [int(value) for value in card_state.completed_column_indices],
            "line_sum_target_axis": card_state.line_sum_target_axis,
            "line_sum_target_line_index": card_state.line_sum_target_line_index,
            "line_sum_target_cell_ids": list(card_state.line_sum_target_cell_ids),
            "line_sum_target_value": card_state.line_sum_target_value,
            "near_complete_row_indices": [int(value) for value in card_state.near_complete_row_indices],
            "near_complete_column_indices": [int(value) for value in card_state.near_complete_column_indices],
            "near_complete_gap_cell_ids": [str(value) for value in card_state.near_complete_gap_cell_ids],
            "called_numbers": [int(value) for value in card_state.called_numbers],
            "called_number_cell_ids": [str(value) for value in card_state.called_number_cell_ids],
            "completed_line_sums": completed_line_sums,
            "cell_specs": [
                {
                    "cell_id": str(spec.cell_id),
                    "row_index": int(spec.row_index),
                    "column_index": int(spec.column_index),
                    "column_label": str(spec.column_label),
                    "number": int(spec.number),
                    "is_marked": bool(spec.is_marked),
                }
                for spec in rendered_scene.cell_specs
            ],
            "annotation_entity_ids": [str(cell_id) for cell_id in annotation_cell_ids],
            "annotation_entity_id_pairs": [
                [str(pair[0]), str(pair[1])]
                for pair in annotation_cell_id_pairs
            ],
            "annotation_points": [list(point) for point in annotation_points],
            "annotation_point_pairs": [
                [list(point) for point in pair]
                for pair in annotation_point_pairs
            ],
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(cell_id) for cell_id in annotation_cell_ids],
        },
        "projected_annotation": dict(projected_annotation or {"bbox_set": [list(bbox) for bbox in annotation_bboxes]}),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = ["build_bingo_common_trace_params", "build_bingo_trace_payload"]
