"""Trace payload helpers for Hex scene tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rendering import RenderedHexScene
from .rules import EMPTY, Coord, HexSample, all_coords, coord_to_cell_id
from .state import HexIntegerAxis, HexSceneAxes, HexStringAxis


def hex_candidate_trace(sample: HexSample) -> list[dict[str, Any]]:
    """Serialize candidate cells for trace/debug review."""

    return [
        {
            "label": str(spec.label),
            "coord": [int(spec.coord[0]), int(spec.coord[1])],
            "cell_id": coord_to_cell_id(spec.coord),
            "is_answer": bool(spec.is_answer),
        }
        for spec in sample.candidate_specs
    ]


def hex_coord_list(coords: Sequence[Coord]) -> list[list[int]]:
    """Serialize Hex coordinates as row/column integer pairs."""

    return [[int(row), int(col)] for row, col in coords]


def build_hex_common_trace_params(
    *,
    scene_axes: HexSceneAxes,
    target_axis: HexIntegerAxis | HexStringAxis | None,
    candidate_count_axis: HexIntegerAxis | None,
    branch_probabilities: Mapping[str, float],
    sample: HexSample,
    neighbor_state: str,
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build prompt-query metadata shared by Hex task outputs."""

    params: dict[str, Any] = {
        "scene_variant": str(scene_axes.scene_variant),
        "style_variant": str(scene_axes.style_variant),
        "player_color": str(scene_axes.player_color),
        "board_size": int(sample.board_size),
        "candidate_count": int(len(sample.candidate_specs)),
        "scene_variant_probabilities": dict(scene_axes.scene_variant_probabilities),
        "query_id_probabilities": {str(key): float(value) for key, value in dict(branch_probabilities).items()},
        "style_variant_probabilities": dict(scene_axes.style_variant_probabilities),
        "player_color_probabilities": dict(scene_axes.player_color_probabilities),
        "board_size_probabilities": dict(scene_axes.board_size_probabilities),
        "neighbor_target_state": str(neighbor_state),
        "occupied_count": sum(
            1
            for coord in all_coords(sample.board_size)
            if sample.board[coord[0]][coord[1]] != EMPTY
        ),
    }
    if isinstance(target_axis, HexIntegerAxis):
        params.update(
            {
                "target_answer": int(target_axis.value),
                "target_answer_support": [int(value) for value in target_axis.support],
                "target_answer_probabilities": dict(target_axis.probabilities),
            }
        )
    elif isinstance(target_axis, HexStringAxis):
        params.update(
            {
                "target_label": str(target_axis.value),
                "target_label_support": [str(value) for value in target_axis.support],
                "target_label_probabilities": dict(target_axis.probabilities),
            }
        )
    if candidate_count_axis is not None:
        params["candidate_count_support"] = [int(value) for value in candidate_count_axis.support]
        params["candidate_count_probabilities"] = dict(candidate_count_axis.probabilities)
    params.update(dict(extra_params or {}))
    return params


def build_hex_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    scene_axes: HexSceneAxes,
    sample: HexSample,
    annotation_coords: Sequence[Coord],
    rendered_scene: RenderedHexScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the verifier payload from one rendered Hex execution trace."""

    annotation_ids = tuple(str(value) for value in annotation_entity_ids)
    neighbor_state = str(sample.neighbor_target_state or "")
    execution_trace = {
        "scene_variant": str(scene_axes.scene_variant),
        "style_variant": str(scene_axes.style_variant),
        "player_color": str(scene_axes.player_color),
        "player_value": int(sample.player_value),
        "board_size": int(sample.board_size),
        "board_rows": [[int(value) for value in row] for row in sample.board],
        "answer": sample.answer,
        "candidate_specs": hex_candidate_trace(sample),
        "reference_coord": None
        if sample.reference_coord is None
        else [int(sample.reference_coord[0]), int(sample.reference_coord[1])],
        "reference_cell_id": None if sample.reference_coord is None else coord_to_cell_id(sample.reference_coord),
        "neighbor_target_state": neighbor_state,
        "neighbor_match_coords": hex_coord_list(sample.neighbor_match_coords),
        "winning_move_coord": None
        if sample.winning_move_coord is None
        else [int(sample.winning_move_coord[0]), int(sample.winning_move_coord[1])],
        "completed_winning_path_coords": hex_coord_list(sample.annotation_coords),
        "min_gap_path": hex_coord_list(sample.min_gap_path),
        "min_gap_empty_coords": hex_coord_list(sample.min_gap_empty_coords),
        "annotation_coords": hex_coord_list(annotation_coords),
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_ids],
        "construction_mode": str(sample.construction_mode),
        **dict(execution_extra or {}),
    }
    return {
        "scene_ir": {
            "scene_kind": f"games_hex_board_{str(scene_axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(scene_axes.scene_variant),
                "style_variant": str(scene_axes.style_variant),
                "player_color": str(scene_axes.player_color),
                "board_size": int(sample.board_size),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(scene_axes.scene_variant),
            "style_variant": str(scene_axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style") or {}),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_hex_common_trace_params",
    "build_hex_trace_payload",
    "hex_candidate_trace",
    "hex_coord_list",
]
