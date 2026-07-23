"""Objective-neutral trace fragments for Mini-golf games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rendering import RenderedMinigolfScene
from .sampling import MinigolfAxes
from .state import MinigolfSample


def build_minigolf_common_trace_params(
    *,
    axes: MinigolfAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared scene params plus task-owned replay params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "obstacle_count": int(axes.obstacle_count),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "obstacle_count_probabilities": dict(axes.obstacle_count_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_minigolf_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: MinigolfAxes,
    sample: MinigolfSample,
    rendered: RenderedMinigolfScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    panel_style_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: Any,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble trace sections after a public task binds answer and annotation."""

    obstacle_trace = [
        {
            "obstacle_id": str(obstacle.obstacle_id),
            "label": str(obstacle.label),
            "kind": str(obstacle.kind),
            "x_norm": float(obstacle.x_norm),
            "y_norm": float(obstacle.y_norm),
            "radius_norm": float(obstacle.radius_norm),
        }
        for obstacle in sample.obstacles
    ]
    path_trace = [
        {
            "path_id": str(path.path_id),
            "label": str(path.label),
            "angle_rad": float(path.angle_rad),
        }
        for path in sample.shot_options
    ]
    hidden_paths_trace = {
        str(path_id): [[float(x), float(y)] for x, y in points]
        for path_id, points in sample.hidden_paths_norm.items()
    }
    relation_map = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_mode": str(sample.mode),
        "obstacle_count": len(sample.obstacles),
        "path_option_count": len(sample.shot_options),
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
    }
    execution = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_mode": str(sample.mode),
        "ball_xy_norm": [float(sample.ball_x_norm), float(sample.ball_y_norm)],
        "hole_xy_norm": [float(sample.hole_x_norm), float(sample.hole_y_norm)],
        "obstacles": obstacle_trace,
        "shot_options": path_trace,
        "target_obstacle_id": sample.target_obstacle_id,
        "target_obstacle_label": sample.target_obstacle_label,
        "target_path_id": sample.target_path_id,
        "target_path_label": sample.target_path_label,
        "cue_visible_fraction": float(sample.cue_visible_fraction),
        "hidden_paths_norm": hidden_paths_trace,
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
        "construction_mode": str(sample.construction_mode),
        "answer": answer_value,
    }
    execution.update(dict(execution_extra or {}))
    return {
        "scene_ir": {
            "scene_kind": f"games_minigolf_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": relation_map,
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(panel_style_meta),
            "text_style": dict(rendered.render_map.get("text_style", {})),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": execution,
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "panel_scene_style": dict(panel_style_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_minigolf_common_trace_params",
    "build_minigolf_trace_payload",
]
