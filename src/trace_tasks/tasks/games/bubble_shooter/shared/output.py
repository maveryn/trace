"""Objective-neutral output assembly for Bubble-shooter games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rules import occupied_coords
from .state import (
    BubbleShooterState,
    bubble_entity_id,
    landing_option_entity_id,
    option_entity_id,
)
from .sampling import ResolvedBubbleShooterSceneAxes
from .rendering import RenderedBubbleShooterTaskContext


def bubble_board_trace(state: BubbleShooterState) -> list[Dict[str, Any]]:
    """Serialize visible Bubble-shooter board bubbles."""

    return [
        {
            "coord": [int(coord[0]), int(coord[1])],
            "bubble_id": bubble_entity_id(coord),
            "color_key": str(state.board[coord[0]][coord[1]]),
        }
        for coord in occupied_coords(state.board)
    ]


def bubble_option_trace(state: BubbleShooterState) -> list[Dict[str, Any]]:
    """Serialize visible Bubble-shooter color options."""

    return [
        {
            "label": str(option.label),
            "color_key": str(option.color_key),
            "is_answer": bool(option.is_answer),
            "entity_id": option_entity_id(str(option.label)),
        }
        for option in state.option_specs
    ]


def bubble_landing_option_trace(state: BubbleShooterState) -> list[Dict[str, Any]]:
    """Serialize visible Bubble-shooter landing target options."""

    return [
        {
            "label": str(option.label),
            "coord": [int(option.landing_coord[0]), int(option.landing_coord[1])],
            "is_answer": bool(option.is_answer),
            "entity_id": landing_option_entity_id(str(option.label)),
        }
        for option in state.landing_option_specs
    ]


def common_bubble_shooter_trace_params(
    axes: ResolvedBubbleShooterSceneAxes,
    state: BubbleShooterState,
    *,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared Bubble-shooter trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "row_count": int(state.row_count),
        "col_count": int(state.col_count),
        "bubble_count": len(occupied_coords(state.board)),
        "option_count": len(state.option_specs),
        "landing_option_count": len(state.landing_option_specs),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_bubble_shooter_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    axes: ResolvedBubbleShooterSceneAxes,
    state: BubbleShooterState,
    rendered_context: RenderedBubbleShooterTaskContext,
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble Bubble-shooter trace sections after task-specific answer binding."""

    rendered_scene = rendered_context.rendered_scene
    annotation_ids = [str(entity_id) for entity_id in annotation_entity_ids]
    return {
        "scene_ir": {
            "scene_kind": f"games_bubble_shooter_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "row_count": int(state.row_count),
                "col_count": int(state.col_count),
                "annotation_entity_ids": annotation_ids,
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "row_count": int(state.row_count),
            "col_count": int(state.col_count),
            "board_bubbles": bubble_board_trace(state),
            "landing_coord": [int(state.landing_coord[0]), int(state.landing_coord[1])],
            "shooter_color_key": state.shooter_color_key,
            "option_specs": bubble_option_trace(state),
            "landing_option_specs": bubble_landing_option_trace(state),
            "outcome": {
                "color_key": str(state.outcome.color_key),
                "connected_same_color_coords": [
                    [int(row), int(col)]
                    for row, col in state.outcome.connected_same_color_coords
                ],
                "popped_coords": [
                    [int(row), int(col)] for row, col in state.outcome.popped_coords
                ],
                "dropped_coords": [
                    [int(row), int(col)] for row, col in state.outcome.dropped_coords
                ],
            },
            "annotation_entity_ids": annotation_ids,
            "construction_mode": str(state.construction_mode),
            **dict(execution_extra),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": annotation_ids,
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = [
    "bubble_board_trace",
    "bubble_landing_option_trace",
    "bubble_option_trace",
    "build_bubble_shooter_trace_payload",
    "common_bubble_shooter_trace_params",
]
