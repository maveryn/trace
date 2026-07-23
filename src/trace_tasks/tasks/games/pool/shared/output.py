"""Objective-neutral trace assembly for pool-table tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedPoolTaskContext
from .rules import object_balls
from .sampling import PoolVisualAxes
from .state import PoolSceneState


def build_pool_ball_trace(state: PoolSceneState) -> list[Dict[str, Any]]:
    """Build trace records for all visible pool balls."""

    return [
        {
            "ball_id": str(ball.ball_id),
            "number": int(ball.number),
            "group": str(ball.group),
            "center": [float(ball.center[0]), float(ball.center[1])],
            "is_cue": bool(ball.is_cue),
            "is_marked": bool(ball.is_marked),
        }
        for ball in state.balls
    ]


def common_pool_trace_params(
    axes: PoolVisualAxes,
    state: PoolSceneState,
    *,
    prompt_query_key: str,
    query_id_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared pool query trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "objective_key": str(prompt_query_key),
        "object_ball_count": int(axes.object_ball_count),
        "visible_object_ball_count": len(object_balls(state.balls)),
        "query_id_probabilities": dict(query_id_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "object_ball_count_probabilities": dict(axes.object_ball_count_probabilities),
        "current_player_group": state.current_player_group,
        "marked_ball_id": state.marked_ball_id,
        "marked_pocket_id": state.marked_pocket_id,
        "line_clearance": float(axes.line_clearance),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_pool_common_trace_payload(
    *,
    axes: PoolVisualAxes,
    state: PoolSceneState,
    rendered_context: RenderedPoolTaskContext,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    query_spec: Mapping[str, Any],
    witness_type: str,
    relations_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble objective-neutral pool trace sections after task binding."""

    rendered_scene = rendered_context.rendered_scene
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"games_pool_table_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "object_ball_count": len(object_balls(state.balls)),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
                **dict(relations_extra or {}),
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
            "object_ball_count": len(object_balls(state.balls)),
            "balls": build_pool_ball_trace(state),
            "pockets": [
                {
                    "pocket_id": str(pocket.pocket_id),
                    "display_name": str(pocket.display_name),
                    "center": [float(pocket.center[0]), float(pocket.center[1])],
                }
                for pocket in state.pockets
            ],
            "cue_ball_id": str(state.cue_ball_id),
            "marked_ball_id": state.marked_ball_id,
            "marked_pocket_id": state.marked_pocket_id,
            "current_player_group": state.current_player_group,
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(state.construction_mode),
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": str(witness_type),
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }
    return trace_payload


__all__ = [
    "build_pool_ball_trace",
    "build_pool_common_trace_payload",
    "common_pool_trace_params",
]
