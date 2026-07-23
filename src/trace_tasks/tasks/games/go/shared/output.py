"""Trace payload helpers for Go scene tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rendering import RenderedGoScene
from .rules import Board, Coord, GoStoneSpec
from .state import GoIntegerAxis, GoPlayerColorAxis, GoSceneAxes


def go_stone_specs_for_trace(stone_specs: Sequence[GoStoneSpec]) -> list[dict[str, Any]]:
    """Serialize visible Go stones in trace-friendly row-major form."""

    return [
        {
            "stone_id": str(spec.stone_id),
            "point_id": str(spec.point_id),
            "row": int(spec.row),
            "col": int(spec.col),
            "color": str(spec.color),
            "is_marked_group": bool(spec.is_marked_group),
        }
        for spec in stone_specs
    ]


def go_coord_list(coords: Sequence[Coord]) -> list[list[int]]:
    """Serialize Go coordinates as row/column integer pairs."""

    return [[int(row), int(col)] for row, col in coords]


def build_go_common_trace_params(
    *,
    scene_axes: GoSceneAxes,
    player_color_axis: GoPlayerColorAxis | None,
    player_color: str,
    board_size_axis: GoIntegerAxis,
    target_axis: GoIntegerAxis,
    query_id_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build prompt-query metadata shared by Go task outputs."""

    params: dict[str, Any] = {
        "scene_variant": str(scene_axes.scene_variant),
        "player_color": str(player_color),
        "style_variant": str(scene_axes.style_variant),
        "board_size": int(board_size_axis.value),
        "board_size_support": [int(value) for value in board_size_axis.support],
        "board_size_probabilities": dict(board_size_axis.probabilities),
        "scene_variant_probabilities": dict(scene_axes.scene_variant_probabilities),
        "query_id_probabilities": {str(key): float(value) for key, value in query_id_probabilities.items()},
        "style_variant_probabilities": dict(scene_axes.style_variant_probabilities),
        "target_answer": int(target_axis.value),
        "target_answer_support": [int(value) for value in target_axis.support],
        "target_answer_probabilities": dict(target_axis.probabilities),
    }
    if player_color_axis is not None:
        params["player_color_probabilities"] = dict(player_color_axis.probabilities)
    params.update(dict(extra_params or {}))
    return params


def build_go_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    scene_axes: GoSceneAxes,
    player_color: str,
    board_size_axis: GoIntegerAxis,
    target_axis: GoIntegerAxis,
    board: Board,
    stone_specs: Sequence[GoStoneSpec],
    marked_group_coords: Sequence[Coord],
    liberty_coords: Sequence[Coord],
    rendered_scene: RenderedGoScene,
    prompt_defaults: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the verifier payload from one rendered Go execution trace."""

    annotation_ids = tuple(str(value) for value in annotation_entity_ids)
    execution_trace = {
        "scene_variant": str(scene_axes.scene_variant),
        "player_color": str(player_color),
        "style_variant": str(scene_axes.style_variant),
        "board_size": int(board_size_axis.value),
        "target_answer": int(target_axis.value),
        "target_answer_support": [int(value) for value in target_axis.support],
        "stone_specs": go_stone_specs_for_trace(stone_specs),
        "marked_group_coords": go_coord_list(tuple(marked_group_coords)),
        "liberty_coords": go_coord_list(tuple(liberty_coords)),
        "annotation_entity_ids": [str(value) for value in annotation_ids],
        **dict(execution_extra or {}),
    }
    return {
        "scene_ir": {
            "scene_kind": "games_go_single_board",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(scene_axes.scene_variant),
                "player_color": str(player_color),
                "style_variant": str(scene_axes.style_variant),
                "target_answer": int(target_axis.value),
                "annotation_entity_ids": [str(value) for value in annotation_ids],
                "board_size": int(board_size_axis.value),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(scene_axes.scene_variant),
            "style_variant": str(scene_axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "board_size": int(board_size_axis.value),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style") or {}),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(value) for value in annotation_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_go_common_trace_params",
    "build_go_trace_payload",
    "go_coord_list",
    "go_stone_specs_for_trace",
]
