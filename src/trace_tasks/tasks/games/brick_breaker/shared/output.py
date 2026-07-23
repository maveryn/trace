"""Objective-neutral trace assembly for Brick-breaker games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .state import BrickBreakerSample
from .sampling import ResolvedBrickBreakerSceneAxes
from .rendering import RenderedBrickBreakerTaskContext


def _brick_trace(sample: BrickBreakerSample) -> list[Dict[str, Any]]:
    return [
        {
            "brick_id": str(brick.brick_id),
            "label": str(brick.label),
            "row": int(brick.row),
            "col": int(brick.col),
        }
        for brick in sample.bricks
    ]


def build_brick_breaker_common_trace_params(
    *,
    axes: ResolvedBrickBreakerSceneAxes,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared Brick-breaker query params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_brick_breaker_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    axes: ResolvedBrickBreakerSceneAxes,
    sample: BrickBreakerSample,
    rendered_context: RenderedBrickBreakerTaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    answer_value: int | str,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble Brick-breaker trace sections after task-specific answer binding."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": f"games_brick_breaker_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "brick_rows": int(sample.brick_rows),
                "brick_cols": int(sample.brick_cols),
                "lane_count": int(sample.lane_count),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
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
            "brick_rows": int(sample.brick_rows),
            "brick_cols": int(sample.brick_cols),
            "lane_count": int(sample.lane_count),
            "bricks": _brick_trace(sample),
            "target_brick_id": sample.target_brick_id,
            "target_brick_label": sample.target_brick_label,
            "target_row_remaining_brick_ids": list(sample.target_row_remaining_brick_ids),
            "target_row_remaining_count": sample.target_row_remaining_count,
            "target_lane_index": sample.target_lane_index,
            "target_lane_label": sample.target_lane_label,
            "ball_start_lane_index": sample.ball_start_lane_index,
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(sample.construction_mode),
            "answer": answer_value,
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = ["build_brick_breaker_common_trace_params", "build_brick_breaker_trace_payload"]
