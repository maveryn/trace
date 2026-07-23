"""Trace payload helpers for radial hunt board tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import all_possible_edges, point_id
from .state import (
    SCENE_ID,
    RadialHuntBoardSample,
    RadialHuntBoardVisualAxes,
    RenderedRadialHuntBoardScene,
)


def annotation_point_ids(sample: RadialHuntBoardSample) -> Tuple[str, ...]:
    """Return stable point ids for every symbolic annotation coordinate."""

    return tuple(point_id(coord) for coord in sample.annotation_coords)


def build_radial_hunt_common_trace_params(
    *,
    axes: RadialHuntBoardVisualAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return prompt-query params common to radial hunt board outputs."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "branch_probabilities": dict(branch_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_radial_hunt_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    sample: RadialHuntBoardSample,
    axes: RadialHuntBoardVisualAxes,
    rendered_scene: RenderedRadialHuntBoardScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    annotation_trace_key: str,
    relations_extra: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble objective-neutral trace payload sections after task binding."""

    execution_payload = {
        "scene_variant": str(sample.scene_variant),
        "style_variant": str(sample.style_variant),
        "construction_mode": str(sample.construction_mode),
        "target_answer": int(sample.answer),
        "marked_coord": [int(sample.marked_coord[0]), int(sample.marked_coord[1])],
        "occupied_coords": [[int(coord[0]), int(coord[1])] for coord in sample.occupied_coords],
        "edge_coords": [
            [[int(item[0][0]), int(item[0][1])], [int(item[1][0]), int(item[1][1])]]
            for item in all_possible_edges()
        ],
        "annotation_coords": [[int(coord[0]), int(coord[1])] for coord in sample.annotation_coords],
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
        str(annotation_trace_key): [[int(coord[0]), int(coord[1])] for coord in sample.annotation_coords],
    }
    if execution_extra:
        execution_payload.update(dict(execution_extra))

    relations = {
        "scene_id": SCENE_ID,
        "scene_variant": str(sample.scene_variant),
        "style_variant": str(sample.style_variant),
        "target_answer": int(sample.answer),
        "marked_piece_id": "piece_marked",
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
    }
    if relations_extra:
        relations.update(dict(relations_extra))

    return {
        "scene_ir": {
            "scene_kind": "games_radial_hunt_board",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": relations,
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.style_meta.get("panel_scene_style", {})),
            "radial_hunt_board_style": dict(rendered_scene.style_meta.get("radial_hunt_board_style", {})),
            "effective_outer_radius_px": float(rendered_scene.render_map["effective_outer_radius_px"]),
            "effective_piece_radius_px": int(rendered_scene.render_map["effective_piece_radius_px"]),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": execution_payload,
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_type),
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "prompt_defaults": {
            "bundle_id": str(prompt_defaults.get("bundle_id", "")),
            "scene_key": str(prompt_defaults.get("scene_key", "")),
            "task_key": str(prompt_defaults.get("task_key", "")),
        },
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }


__all__ = [
    "annotation_point_ids",
    "build_radial_hunt_common_trace_params",
    "build_radial_hunt_trace_payload",
]
