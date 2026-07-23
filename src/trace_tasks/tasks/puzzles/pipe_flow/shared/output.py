"""Trace payload assembly for pipe-flow repair puzzles."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter

from .state import (
    PipeFlowDataset,
    PipeFlowMisrotatedDataset,
    RenderParams,
    RenderedPipeFlowScene,
    SCENE_ID,
)


def option_trace(dataset: PipeFlowDataset) -> list[dict[str, Any]]:
    """Serialize option pieces and in-place solvability metadata."""

    return [
        {
            "option_id": str(option.option_id),
            "label": str(option.label),
            "is_correct": bool(option.is_correct),
            "rotation_allowed": False,
            "gap_size_variant": str(dataset.gap_size_variant),
            "gap_size": int(dataset.gap_size),
            "connects_in_place": bool(option.connects_in_place),
            "display_rotation_turns": int(option.display_rotation_turns),
            "connects_after_rotation_turns": [
                int(value) for value in option.connects_after_rotation_turns
            ],
            "local_openings": [
                {"row": int(row), "col": int(col), "openings": list(openings)}
                for row, col, openings in option.local_openings
            ],
        }
        for option in dataset.options
    ]


def tile_trace(dataset: PipeFlowDataset | PipeFlowMisrotatedDataset) -> list[dict[str, Any]]:
    """Serialize visible grid tiles and their current openings."""

    return [
        {
            "tile_id": str(tile.tile_id),
            "label": str(tile.label),
            "row_index": int(tile.row),
            "col_index": int(tile.col),
            "current_openings": list(tile.current_openings),
            "required_openings": list(tile.required_openings),
            "is_path": bool(tile.is_path),
            "is_branch": bool(tile.is_branch),
        }
        for tile in dataset.tiles
    ]


def misrotated_candidate_trace(dataset: PipeFlowMisrotatedDataset) -> list[dict[str, Any]]:
    """Serialize labeled candidate tiles and their repair rotations."""

    return [
        {
            "candidate_id": str(candidate.candidate_id),
            "label": str(candidate.label),
            "tile_id": str(candidate.tile_id),
            "row_index": int(candidate.row),
            "col_index": int(candidate.col),
            "required_openings": list(candidate.required_openings),
            "current_openings": list(candidate.current_openings),
            "repair_rotation_turns": [
                int(value) for value in candidate.repair_rotation_turns
            ],
            "is_correct": bool(candidate.is_correct),
            "connects_after_rotation": bool(candidate.connects_after_rotation),
        }
        for candidate in dataset.candidates
    ]


def projected_annotation_value(projected_annotation: Mapping[str, Any]) -> Any:
    """Return the public annotation value from a projected annotation payload."""

    for key in (
        "bbox",
        "bbox_set",
        "bbox_sequence",
        "bbox_map",
        "bbox_set_map",
        "point",
        "point_set",
        "point_sequence",
        "point_map",
        "point_set_map",
        "segment",
        "segment_set",
    ):
        if key in projected_annotation:
            return projected_annotation.get(key)
    return projected_annotation.get("value")


def build_trace_payload(
    *,
    dataset: PipeFlowDataset,
    rendered_scene: RenderedPipeFlowScene,
    render_params: RenderParams,
    prompt_meta: Mapping[str, Any],
    task_fields: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    question_format: str,
) -> dict[str, Any]:
    """Build the metadata trace that backs answer and annotation verification."""

    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.scene_variant),
                "gap_size_variant": str(dataset.gap_size_variant),
                "gap_size": int(dataset.gap_size),
                "answer_label": str(dataset.answer_label),
                "correct_option_panel_id": str(dataset.correct_option_panel_id),
                "missing_region_id": str(dataset.missing_region_id),
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "cell_gap_px": int(render_params.cell_gap_px),
            "cell_size_min_px": int(render_params.cell_size_min_px),
            "cell_size_max_px": int(render_params.cell_size_max_px),
            "pipe_width_px": int(render_params.pipe_width_px),
            "tile_label_font_size_px": int(render_params.tile_label_font_size_px),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        "render_map": with_puzzle_unit_size_jitter(
            {
                "image_id": "img0",
                "scene_bbox_px": [
                    round(float(value), 3) for value in rendered_scene.scene_bbox_px
                ],
                "tile_bboxes_px": {
                    str(key): [round(float(v), 3) for v in value]
                    for key, value in rendered_scene.tile_bbox_map.items()
                },
                "item_bboxes_px": {
                    str(key): [round(float(v), 3) for v in value]
                    for key, value in rendered_scene.item_bbox_map.items()
                },
                "annotation_source": "item_bboxes_px",
            },
            render_params.unit_size_jitter,
        ),
        "execution_trace": {
            **dict(task_fields),
            "question_format": str(question_format),
            "tiles": tile_trace(dataset),
            "option_specs": option_trace(dataset),
            "path_cells": [[int(r), int(c)] for r, c in dataset.path_cells],
            "branch_cells": [[int(r), int(c)] for r, c in dataset.branch_cells],
            "branch_terminal_cells": [
                [int(r), int(c)] for r, c in dataset.branch_terminal_cells
            ],
            "start_cell": [int(dataset.start_cell[0]), int(dataset.start_cell[1])],
            "destination_cell": [
                int(dataset.destination_cell[0]),
                int(dataset.destination_cell[1]),
            ],
            "missing_origin": [int(dataset.missing_origin[0]), int(dataset.missing_origin[1])],
            "missing_cells": [[int(r), int(c)] for r, c in dataset.missing_cells],
            "gap_size_variant": str(dataset.gap_size_variant),
            "gap_size": int(dataset.gap_size),
            "missing_region_id": str(dataset.missing_region_id),
            "correct_option_panel_id": str(dataset.correct_option_panel_id),
            "supporting_item_ids": [
                str(dataset.correct_option_panel_id),
                str(dataset.missing_region_id),
            ],
            "answer_value": str(dataset.answer_label),
        },
        "witness_symbolic": {
            "type": str(projected_annotation.get("type", "")),
            "value": projected_annotation_value(projected_annotation),
        },
        "projected_annotation": dict(projected_annotation),
    }


def build_misrotated_trace_payload(
    *,
    dataset: PipeFlowMisrotatedDataset,
    rendered_scene: RenderedPipeFlowScene,
    render_params: RenderParams,
    prompt_meta: Mapping[str, Any],
    task_fields: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    scene_style_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
    question_format: str,
) -> dict[str, Any]:
    """Build metadata trace for the misrotated-tile task."""

    correct_candidate = next(candidate for candidate in dataset.candidates if candidate.is_correct)
    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.scene_variant),
                "answer_label": str(dataset.answer_label),
                "misrotated_tile_id": str(dataset.misrotated_tile_id),
                "correct_candidate_id": str(correct_candidate.candidate_id),
            },
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "cell_gap_px": int(render_params.cell_gap_px),
            "cell_size_min_px": int(render_params.cell_size_min_px),
            "cell_size_max_px": int(render_params.cell_size_max_px),
            "pipe_width_px": int(render_params.pipe_width_px),
            "tile_label_font_size_px": int(render_params.tile_label_font_size_px),
            "background_style": dict(background_meta),
            "scene_style": dict(scene_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
            "unit_size_jitter": dict(render_params.unit_size_jitter),
        },
        "render_map": with_puzzle_unit_size_jitter(
            {
                "image_id": "img0",
                "scene_bbox_px": [
                    round(float(value), 3) for value in rendered_scene.scene_bbox_px
                ],
                "tile_bboxes_px": {
                    str(key): [round(float(v), 3) for v in value]
                    for key, value in rendered_scene.tile_bbox_map.items()
                },
                "item_bboxes_px": {
                    str(key): [round(float(v), 3) for v in value]
                    for key, value in rendered_scene.item_bbox_map.items()
                },
                "annotation_source": "item_bboxes_px",
            },
            render_params.unit_size_jitter,
        ),
        "execution_trace": {
            **dict(task_fields),
            "question_format": str(question_format),
            "tiles": tile_trace(dataset),
            "candidate_specs": misrotated_candidate_trace(dataset),
            "path_cells": [[int(r), int(c)] for r, c in dataset.path_cells],
            "branch_cells": [[int(r), int(c)] for r, c in dataset.branch_cells],
            "branch_terminal_cells": [
                [int(r), int(c)] for r, c in dataset.branch_terminal_cells
            ],
            "start_cell": [int(dataset.start_cell[0]), int(dataset.start_cell[1])],
            "destination_cell": [
                int(dataset.destination_cell[0]),
                int(dataset.destination_cell[1]),
            ],
            "misrotated_tile_id": str(dataset.misrotated_tile_id),
            "correct_candidate_id": str(correct_candidate.candidate_id),
            "supporting_item_ids": [str(correct_candidate.candidate_id)],
            "answer_value": str(dataset.answer_label),
        },
        "witness_symbolic": {
            "type": str(projected_annotation.get("type", "")),
            "value": projected_annotation_value(projected_annotation),
        },
        "projected_annotation": dict(projected_annotation),
    }
