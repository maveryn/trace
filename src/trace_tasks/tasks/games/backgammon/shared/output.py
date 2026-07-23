"""Objective-neutral trace-payload assembly for Backgammon games scene tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .rendering import RenderedBackgammonTaskContext
from .sampling import ResolvedBackgammonAxes
from .state import POINT_IDS, SCENE_ID, BackgammonSample, stack_at


def point_trace(sample: BackgammonSample) -> list[Dict[str, Any]]:
    """Serialize Backgammon point stacks for trace payloads."""

    return [
        {
            "point_id": int(point),
            "owner": stack_at(sample.points, int(point)).owner,
            "count": int(stack_at(sample.points, int(point)).count),
            "entity_id": f"point_{int(point)}",
        }
        for point in POINT_IDS
    ]


def common_backgammon_trace_params(
    *,
    axes: ResolvedBackgammonAxes,
    sample: BackgammonSample,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared Backgammon trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "active_player": str(sample.active_player),
        "destination_status": str(sample.destination_status),
        "checker_color": str(sample.checker_color),
        "stack_state": str(sample.stack_state),
        "dice": [int(value) for value in sample.dice],
        "target_answer": int(sample.answer),
        "use_dice_for_moves": bool(sample.use_dice_for_moves),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "active_player_probabilities": dict(axes.active_player_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_backgammon_trace_payload(
    *,
    annotation_gt: Any,
    annotation_entity_ids: tuple[str, ...],
    axes: ResolvedBackgammonAxes,
    sample: BackgammonSample,
    rendered_context: RenderedBackgammonTaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
    construction_mode: str,
) -> Dict[str, Any]:
    """Assemble common Backgammon trace payload after answer/annotation binding."""

    rendered_scene = rendered_context.rendered_scene
    target_points = tuple(int(point) for point in (sample.target_points or sample.target_destinations))
    return {
        "scene_ir": {
            "scene_kind": f"games_backgammon_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "active_player": str(sample.active_player),
                "destination_status": str(sample.destination_status),
                "checker_color": str(sample.checker_color),
                "stack_state": str(sample.stack_state),
                "dice": [int(value) for value in sample.dice],
                "target_destinations": [int(point) for point in sample.target_destinations],
                "target_points": [int(point) for point in target_points],
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
                "use_dice_for_moves": bool(sample.use_dice_for_moves),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "active_player": str(sample.active_player),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "text_style": dict(rendered_context.text_style_meta),
            "effective_point_width_px": rendered_scene.render_map.get("effective_point_width_px"),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "active_player": str(sample.active_player),
            "destination_status": str(sample.destination_status),
            "checker_color": str(sample.checker_color),
            "stack_state": str(sample.stack_state),
            "points": point_trace(sample),
            "dice": [int(value) for value in sample.dice],
            "outcome": {
                "legal_destinations": [int(point) for point in sample.outcome.legal_destinations],
                "hit_destinations": [int(point) for point in sample.outcome.hit_destinations],
                "blocked_destinations": [int(point) for point in sample.outcome.blocked_destinations],
            },
            "target_destinations": [int(point) for point in sample.target_destinations],
            "target_points": [int(point) for point in target_points],
            "pip_count_contributions": {
                str(point): int(value)
                for point, value in sorted(sample.pip_count_contributions.items())
            },
            "use_dice_for_moves": bool(sample.use_dice_for_moves),
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(construction_mode),
            **dict(execution_extra),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": {
            "bbox_set": [list(bbox) for bbox in annotation_gt.value],
        },
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = ["build_backgammon_trace_payload", "common_backgammon_trace_params", "point_trace"]
