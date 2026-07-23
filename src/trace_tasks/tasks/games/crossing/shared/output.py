"""Identity-free trace serialization helpers for Crossing scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rendering import RenderedCrossingTaskContext
from .state import CrossingRouteOption, CrossingSample, CrossingSceneAxes, CrossingVehicle


def vehicle_trace(vehicles: tuple[CrossingVehicle, ...]) -> list[dict[str, Any]]:
    """Serialize visible moving objects."""

    return [
        {
            "vehicle_id": str(vehicle.vehicle_id),
            "row": int(vehicle.row),
            "start_col": int(vehicle.start_col),
            "direction": int(vehicle.direction),
            "color_index": int(vehicle.color_index),
            "option_label": None if vehicle.option_label is None else str(vehicle.option_label),
        }
        for vehicle in vehicles
    ]


def route_trace(routes: tuple[CrossingRouteOption, ...]) -> list[dict[str, Any]]:
    """Serialize visible route options."""

    return [
        {
            "route_id": str(route.route_id),
            "label": str(route.label),
            "path_cols": [int(col) for col in route.path_cols],
            "color_index": int(route.color_index),
        }
        for route in routes
    ]


def common_trace_params(
    *,
    axes: CrossingSceneAxes,
    sample: CrossingSample,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return shared Crossing query params plus task-owned fields."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "lane_count": int(sample.lane_count),
        "row_count": int(sample.row_count),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "lane_count_probabilities": dict(axes.lane_count_probabilities),
        "row_count_probabilities": dict(axes.row_count_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def common_trace_sections(
    *,
    axes: CrossingSceneAxes,
    sample: CrossingSample,
    rendered_context: RenderedCrossingTaskContext,
    annotation_artifacts: AnnotationArtifacts,
    query_spec: dict[str, Any],
    execution_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return trace sections shared by Crossing objective-owned tasks."""

    rendered_scene = rendered_context.rendered_scene
    annotation_entity_ids = [str(entity_id) for entity_id in sample.annotation_entity_ids]
    return {
        "scene_ir": {
            "scene_kind": f"games_crossing_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "lane_count": int(sample.lane_count),
                "row_count": int(sample.row_count),
                "row_directions": [int(value) for value in sample.row_directions],
                "annotation_entity_ids": list(annotation_entity_ids),
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
            "lane_count": int(sample.lane_count),
            "row_count": int(sample.row_count),
            "row_directions": [int(value) for value in sample.row_directions],
            "vehicles": vehicle_trace(sample.vehicles),
            "start_labels": [str(label) for label in sample.start_labels],
            "route_options": route_trace(sample.route_options),
            "marked_route_label": sample.marked_route_label,
            "target_start_label": sample.target_start_label,
            "target_route_label": sample.target_route_label,
            "target_object_label": sample.target_object_label,
            "first_collision_tick": sample.first_collision_tick,
            "intersecting_vehicle_ids": [str(value) for value in sample.intersecting_vehicle_ids],
            "annotation_entity_ids": list(annotation_entity_ids),
            "construction_mode": str(sample.construction_mode),
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": list(annotation_entity_ids),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = ["common_trace_params", "common_trace_sections", "route_trace", "vehicle_trace"]
