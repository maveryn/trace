"""Identity-free trace serialization helpers for Checkers games scenes."""

from __future__ import annotations

from typing import Any, Mapping

from .rules import BOARD_SIZE, coord_to_cell_id, player_name
from .rendering import RenderedCheckersTaskContext
from .state import ResolvedCheckersSceneAxes, SampledCheckersScene


def legal_move_specs(sample: SampledCheckersScene) -> list[dict[str, Any]]:
    """Return serialized one-step move specs for trace payloads."""

    return [
        {
            "origin": [int(move.origin[0]), int(move.origin[1])],
            "landing": [int(move.landing[0]), int(move.landing[1])],
            "captured": None if move.captured is None else [int(move.captured[0]), int(move.captured[1])],
            "landing_cell_id": str(coord_to_cell_id(move.landing)),
        }
        for move in sorted(
            sample.evaluation.legal_moves,
            key=lambda move: (move.origin[0], move.origin[1], move.landing[0], move.landing[1]),
        )
    ]


def checkers_common_trace_params(
    axes: ResolvedCheckersSceneAxes,
    sample: SampledCheckersScene,
    *,
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return shared Checkers trace params plus task-owned params."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "board_size": int(BOARD_SIZE),
        "current_player": player_name(int(sample.current_player)),
        "occupied_count": int(sample.occupied_count),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_checkers_common_trace_payload(
    *,
    annotation_artifacts: Any,
    axes: ResolvedCheckersSceneAxes,
    sample: SampledCheckersScene,
    rendered_context: RenderedCheckersTaskContext,
    prompt_artifacts: Any,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble objective-neutral trace sections after task-specific binding."""

    del prompt_artifacts, prompt_defaults
    rendered_scene = rendered_context.rendered_scene
    board_rows = [[int(cell) for cell in row] for row in sample.board]
    witness_symbolic = {
        "type": "object_set",
        "ids": [str(entity_id) for entity_id in sample.evaluation.annotation_entity_ids],
    }
    return {
        "scene_ir": {
            "scene_kind": f"games_checkers_board_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "board_size": int(BOARD_SIZE),
                "current_player": player_name(int(sample.current_player)),
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.evaluation.annotation_entity_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
            "effective_cell_size_px": rendered_scene.render_map.get("effective_cell_size_px"),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "board_size": int(BOARD_SIZE),
            "current_player": player_name(int(sample.current_player)),
            "board_rows": board_rows,
            "construction_mode": str(sample.construction_mode),
            "occupied_count": int(sample.occupied_count),
            "legal_move_specs": legal_move_specs(sample),
            "marked_coord": None
            if sample.evaluation.marked_coord is None
            else [int(sample.evaluation.marked_coord[0]), int(sample.evaluation.marked_coord[1])],
            "marked_piece_kind": "king" if sample.evaluation.marked_coord is not None else None,
            "max_capture_chain_specs": [
                {
                    "origin": [int(chain.origin[0]), int(chain.origin[1])],
                    "landings": [[int(row), int(col)] for row, col in chain.landings],
                    "captured": [[int(row), int(col)] for row, col in chain.captured],
                }
                for chain in sample.evaluation.max_capture_chains
            ],
            "annotation_kind": str(sample.evaluation.annotation_kind),
            "annotation_coords": [[int(coord[0]), int(coord[1])] for coord in sample.evaluation.annotation_coords],
            "annotation_entity_ids": [str(entity_id) for entity_id in sample.evaluation.annotation_entity_ids],
            **dict(execution_extra or {}),
        },
        "witness_symbolic": witness_symbolic,
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = [
    "build_checkers_common_trace_payload",
    "checkers_common_trace_params",
    "legal_move_specs",
]
