"""Objective-neutral trace assembly for platformer tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedPlatformerTaskContext
from .sampling import PlatformerVisualAxes
from .state import PlatformerSample


def build_platformer_object_trace(sample: PlatformerSample) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Build trace records for platforms, hazards, and collectibles."""

    platform_trace = [
        {
            "platform_id": str(platform.platform_id),
            "label": str(platform.label),
            "x_norm": float(platform.x_norm),
            "y_norm": float(platform.y_norm),
            "width_norm": float(platform.width_norm),
            "height_norm": float(platform.height_norm),
        }
        for platform in sample.platforms
    ]
    hazard_trace = [
        {
            "hazard_id": str(hazard.hazard_id),
            "label": str(hazard.label),
            "kind": str(hazard.kind),
            "x_norm": float(hazard.x_norm),
            "y_norm": float(hazard.y_norm),
            "width_norm": float(hazard.width_norm),
            "height_norm": float(hazard.height_norm),
        }
        for hazard in sample.hazards
    ]
    collectible_trace = [
        {
            "collectible_id": str(coin.collectible_id),
            "x_norm": float(coin.x_norm),
            "y_norm": float(coin.y_norm),
            "radius_norm": float(coin.radius_norm),
            "on_path": bool(coin.on_path),
            "kind": str(coin.kind),
            "score_value": int(coin.score_value) if coin.score_value is not None else 1,
            "display_text": str(int(coin.score_value)) if coin.score_value is not None else "",
        }
        for coin in sample.collectibles
    ]
    return platform_trace, hazard_trace, collectible_trace


def common_platformer_trace_params(
    axes: PlatformerVisualAxes,
    sample: PlatformerSample,
    *,
    prompt_query_key: str,
    query_id_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared platformer query trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "objective_key": str(prompt_query_key),
        "platform_count": int(axes.platform_count),
        "hazard_count": int(axes.hazard_count),
        "distractor_collectible_count": int(axes.distractor_collectible_count),
        "actual_platform_count": len(sample.platforms),
        "actual_hazard_count": len(sample.hazards),
        "actual_collectible_total": len(sample.collectibles),
        "query_id_probabilities": dict(query_id_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "platform_count_probabilities": dict(axes.platform_count_probabilities),
        "hazard_count_probabilities": dict(axes.hazard_count_probabilities),
        "distractor_collectible_count_probabilities": dict(axes.distractor_collectible_count_probabilities),
        "target_platform_id": sample.target_platform_id,
        "target_collectible_ids": list(sample.target_collectible_ids),
        "visible_path_fraction": float(sample.visible_path_fraction),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_platformer_common_trace_payload(
    *,
    axes: PlatformerVisualAxes,
    sample: PlatformerSample,
    rendered_context: RenderedPlatformerTaskContext,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    query_spec: Mapping[str, Any],
    witness_type: str,
    relations_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble objective-neutral platformer trace sections after task binding."""

    rendered_scene = rendered_context.rendered_scene
    platform_trace, hazard_trace, collectible_trace = build_platformer_object_trace(sample)
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"games_platformer_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "platform_total": len(sample.platforms),
                "hazard_total": len(sample.hazards),
                "collectible_total": len(sample.collectibles),
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
            "objective_key": str(sample.mode),
            "player_xy_norm": [float(sample.player_x_norm), float(sample.player_y_norm)],
            "path_points_norm": [[float(x), float(y)] for x, y in sample.path_points_norm],
            "visible_path_fraction": float(sample.visible_path_fraction),
            "platforms": platform_trace,
            "hazards": hazard_trace,
            "collectibles": collectible_trace,
            "target_platform_id": sample.target_platform_id,
            "target_platform_label": sample.target_platform_label,
            "target_collectible_ids": list(sample.target_collectible_ids),
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(sample.construction_mode),
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
    "build_platformer_common_trace_payload",
    "build_platformer_object_trace",
    "common_platformer_trace_params",
]
