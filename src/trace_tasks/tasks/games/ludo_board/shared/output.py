"""Trace payload helpers for Ludo board scene tasks."""

from __future__ import annotations

from typing import Any, Mapping

from .annotations import LudoAnnotationBundle
from .rendering import RenderedLudoScene
from .state import Coord, LudoSceneAxes, PLAYER_COLORS, SCENE_ID


def ludo_token_coord_trace(token_coords: Mapping[str, Coord]) -> dict[str, list[int]]:
    """Return color-keyed board coordinates for visible Ludo tokens."""

    return {
        str(color): [int(coord[0]), int(coord[1])]
        for color, coord in dict(token_coords).items()
    }


def base_ludo_execution_trace(
    *,
    branch_field_name: str,
    selected_query_id: str,
    axes: LudoSceneAxes,
    construction_mode: str,
    token_coords: Mapping[str, Coord],
    query_color: str,
    target_color: str | None,
) -> dict[str, Any]:
    """Build the common symbolic execution header for Ludo task traces."""

    return {
        str(branch_field_name): str(selected_query_id),
        "style_variant": str(axes.style_variant),
        "construction_mode": str(construction_mode),
        "token_coords_by_color": ludo_token_coord_trace(token_coords),
        "query_color": str(query_color),
        "target_color": None if target_color is None else str(target_color),
    }


def build_ludo_common_trace_params(
    *,
    axes: LudoSceneAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build shared prompt-query params for Ludo task trace payloads."""

    params: dict[str, Any] = {
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "query_color": str(axes.query_color),
        "query_color_support": list(PLAYER_COLORS),
        "query_color_probabilities": dict(axes.query_color_probabilities),
        "target_color": str(axes.target_color),
        "target_color_support": list(PLAYER_COLORS),
        "target_color_probabilities": dict(axes.target_color_probabilities),
        "query_id_probabilities": {str(key): float(value) for key, value in dict(branch_probabilities).items()},
    }
    params.update(dict(extra_params or {}))
    return params


def build_ludo_trace_payload(
    *,
    annotation_bundle: LudoAnnotationBundle,
    axes: LudoSceneAxes,
    rendered_scene: RenderedLudoScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    scene_kind: str,
    selected_branch: str,
    branch_field_name: str,
    execution_trace: Mapping[str, Any],
    relations_extra: Mapping[str, Any] | None = None,
    render_spec_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build verifier trace payload from task-owned Ludo bindings."""

    branch_key = str(branch_field_name)
    relations = {
        "scene_id": SCENE_ID,
        branch_key: str(selected_branch),
        "query_color": str(axes.query_color),
        "target_color": str(axes.target_color),
        "style_variant": str(axes.style_variant),
        "annotation_entity_ids": dict(annotation_bundle.entity_ids),
    }
    relations.update(dict(relations_extra or {}))
    render_spec = {
        "style_variant": str(axes.style_variant),
        "canvas_width": int(image_size[0]),
        "canvas_height": int(image_size[1]),
        "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
        "panel_scene_style": dict(rendered_scene.style_meta.get("panel_scene_style", {})),
        "ludo_board_style": dict(rendered_scene.style_meta.get("ludo_board_style", {})),
        "label_font": dict(rendered_scene.style_meta.get("label_font", {})),
        "effective_cell_size_px": int(rendered_scene.render_map["effective_cell_size_px"]),
        "flow_arrow_count": int(len(rendered_scene.render_map.get("flow_arrow_markers_px", []))),
    }
    render_spec.update(dict(render_spec_extra or {}))
    return {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": relations,
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": render_spec,
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(annotation_bundle.witness_symbolic),
        "projected_annotation": dict(annotation_bundle.projected_annotation),
        "background": dict(rendered_scene.background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "base_ludo_execution_trace",
    "build_ludo_common_trace_params",
    "build_ludo_trace_payload",
    "ludo_token_coord_trace",
]
