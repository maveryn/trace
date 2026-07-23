"""Objective-neutral trace assembly for racing-track tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedRacingTrackTaskContext
from .sampling import RacingTrackVisualAxes
from .state import RacingTrackSceneState, visible_car_trace


def common_racing_track_trace_params(
    axes: RacingTrackVisualAxes,
    state: RacingTrackSceneState,
    *,
    prompt_query_key: str,
    query_id_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared racing-track query trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "objective_key": str(prompt_query_key),
        "car_count": len(state.cars),
        "query_id_probabilities": dict(query_id_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_racing_track_common_trace_payload(
    *,
    axes: RacingTrackVisualAxes,
    state: RacingTrackSceneState,
    rendered_context: RenderedRacingTrackTaskContext,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    query_spec: Mapping[str, Any],
    witness_type: str,
    relations_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble objective-neutral racing-track trace sections after task binding."""

    rendered_scene = rendered_context.rendered_scene
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"games_racing_track_{str(state.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "car_count": len(state.cars),
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
            "track_width_px": int(state.track_width_px),
            "track_height_px": int(state.track_height_px),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_context.panel_style_meta),
            "racing_track_style": dict(rendered_scene.render_map.get("racing_track_style", {})),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "track_width_px": int(state.track_width_px),
            "track_height_px": int(state.track_height_px),
            "centerline_points_px_local": [
                [round(float(point[0]), 3), round(float(point[1]), 3)]
                for point in state.centerline_points_px
            ],
            "finish_point_px_local": [
                round(float(state.finish_point_px[0]), 3),
                round(float(state.finish_point_px[1]), 3),
            ],
            "finish_tangent_px": [
                round(float(state.finish_tangent_px[0]), 6),
                round(float(state.finish_tangent_px[1]), 6),
            ],
            "cars": list(visible_car_trace(state.cars)),
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
    "build_racing_track_common_trace_payload",
    "common_racing_track_trace_params",
]
