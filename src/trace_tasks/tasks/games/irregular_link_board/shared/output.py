"""Trace payload helpers for irregular-link-board scene tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import edge, point_id, serialize_coords
from .state import IrregularLinkBoardAxes, IrregularLinkBoardSample, RenderedIrregularLinkBoardScene


def build_irregular_link_common_trace_params(
    *,
    axes: IrregularLinkBoardAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build prompt-query metadata shared by the scene's public tasks."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "board_size": int(axes.board_size),
        "board_size_probabilities": dict(axes.board_size_probabilities),
        "target_answer": int(axes.target_answer),
        "target_answer_support": [int(value) for value in axes.target_answer_support],
        "target_answer_probabilities": dict(axes.target_answer_probabilities),
        "query_id_probabilities": {str(key): float(value) for key, value in dict(branch_probabilities).items()},
    }
    params.update(dict(extra_params or {}))
    return params


def build_irregular_link_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    sample: IrregularLinkBoardSample,
    axes: IrregularLinkBoardAxes,
    rendered_scene: RenderedIrregularLinkBoardScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    selected_branch: str,
    branch_field_name: str,
    annotation_trace_key: str,
) -> dict[str, Any]:
    """Build verifier payload from one rendered movement-board instance."""

    annotation_ids = tuple(str(value) for value in annotation_entity_ids)
    annotation_coords = serialize_coords(tuple(sample.annotation_coords))
    branch_key = str(branch_field_name)
    execution_trace = {
        "scene_variant": str(sample.scene_variant),
        branch_key: str(selected_branch),
        "style_variant": str(sample.style_variant),
        "board_size": int(sample.board_size),
        "construction_mode": str(sample.construction_mode),
        "target_answer": int(sample.answer),
        "target_answer_support": [int(value) for value in axes.target_answer_support],
        "marked_coord": [int(sample.marked_coord[0]), int(sample.marked_coord[1])],
        "occupied_coords": serialize_coords(tuple(sample.occupied_coords)),
        "edge_coords": [
            [[int(link[0][0]), int(link[0][1])], [int(link[1][0]), int(link[1][1])]]
            for link in sample.edges
        ],
        "annotation_coords": annotation_coords,
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_ids],
        str(annotation_trace_key): annotation_coords,
    }
    return {
        "scene_ir": {
            "scene_kind": "games_irregular_link_board",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": "irregular_link_board",
                "scene_variant": str(sample.scene_variant),
                branch_key: str(selected_branch),
                "style_variant": str(sample.style_variant),
                "board_size": int(sample.board_size),
                "target_answer": int(sample.answer),
                "marked_piece_id": "piece_marked",
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(sample.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.style_meta.get("panel_scene_style", {})),
            "irregular_link_board_style": dict(rendered_scene.style_meta.get("irregular_link_board_style", {})),
            "effective_board_step_px": float(rendered_scene.render_map["effective_board_step_px"]),
            "effective_piece_radius_px": int(rendered_scene.render_map["effective_piece_radius_px"]),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": {
            "type": "point_set",
            "ids": [str(entity_id) for entity_id in annotation_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


def annotation_point_ids(sample: IrregularLinkBoardSample) -> tuple[str, ...]:
    """Return point ids corresponding to the sample's selected annotation coords."""

    return tuple(point_id(coord) for coord in sample.annotation_coords)


def edge_coords_for_trace(sample: IrregularLinkBoardSample) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return normalized edge coords for tests and trace comparisons."""

    return tuple(edge(link[0], link[1]) for link in sample.edges)


__all__ = [
    "annotation_point_ids",
    "build_irregular_link_common_trace_params",
    "build_irregular_link_trace_payload",
    "edge_coords_for_trace",
]
