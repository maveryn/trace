"""Objective-neutral trace assembly for Pac-Man tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedPacmanTaskContext
from .sampling import PacmanVisualAxes
from .state import PacmanSceneState, visible_ghost_trace, visible_item_trace, visible_pellet_trace


def common_pacman_trace_params(
    axes: PacmanVisualAxes,
    scene: PacmanSceneState,
    *,
    query_id_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared Pac-Man query trace params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "row_count": int(scene.row_count),
        "col_count": int(scene.col_count),
        "route_length": len(scene.route_coords),
        "pellet_count": len(scene.pellets),
        "item_count": len(scene.items),
        "ghost_count": len(scene.ghosts),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "query_id_probabilities": dict(query_id_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "row_count_probabilities": dict(axes.row_count_probabilities),
        "col_count_probabilities": dict(axes.col_count_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_pacman_common_trace_payload(
    *,
    axes: PacmanVisualAxes,
    scene: PacmanSceneState,
    rendered_context: RenderedPacmanTaskContext,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble objective-neutral Pac-Man trace sections after task binding."""

    rendered_scene = rendered_context.rendered_scene
    trace_payload = {
        "scene_ir": {
            "scene_kind": f"games_pacman_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "row_count": int(scene.row_count),
                "col_count": int(scene.col_count),
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
            "row_count": int(scene.row_count),
            "col_count": int(scene.col_count),
            "open_cells": [[int(row), int(col)] for row, col in scene.open_cells],
            "wall_cells": [[int(row), int(col)] for row, col in scene.wall_cells],
            "pacman_coord": [int(scene.pacman_coord[0]), int(scene.pacman_coord[1])],
            "route_coords": [[int(row), int(col)] for row, col in scene.route_coords],
            "pellets": list(visible_pellet_trace(scene.pellets)),
            "items": list(visible_item_trace(scene.items)),
            "ghosts": list(visible_ghost_trace(scene.ghosts)),
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "construction_mode": str(scene.construction_mode),
            **dict(execution_extra),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }
    return trace_payload


__all__ = [
    "build_pacman_common_trace_payload",
    "common_pacman_trace_params",
]
