"""Objective-neutral trace fragments for Nine Men's Morris tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rendering import RenderedNineMensMorrisScene
from .sampling import NineMensMorrisVisualAxes
from .state import NineMensMorrisBoardState


def build_morris_common_trace_params(
    *,
    axes: NineMensMorrisVisualAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return shared scene params plus task-owned replay params."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_morris_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: NineMensMorrisVisualAxes,
    board_state: NineMensMorrisBoardState,
    rendered: RenderedNineMensMorrisScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    panel_style_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: int,
    construction_mode: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble trace sections after a public task binds answer and annotation."""

    piece_specs = [
        {
            "piece_id": str(spec.piece_id),
            "node_index": int(spec.node_index),
            "node_label": str(spec.node_label),
            "color": str(spec.color),
        }
        for spec in board_state.piece_specs
    ]
    relations = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "piece_count": len(board_state.piece_specs),
        "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
    }
    execution = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "target_answer": int(board_state.target_answer),
        "piece_specs": piece_specs,
        "white_piece_ids_in_mill": [str(value) for value in board_state.white_piece_ids_in_mill],
        "black_piece_ids_in_mill": [str(value) for value in board_state.black_piece_ids_in_mill],
        "all_piece_ids_in_mill": [str(value) for value in board_state.all_piece_ids_in_mill],
        "white_mill_ids": [str(value) for value in board_state.white_mill_ids],
        "black_mill_ids": [str(value) for value in board_state.black_mill_ids],
        "white_mill_completion_node_labels": [str(value) for value in board_state.white_mill_completion_node_labels],
        "black_mill_completion_node_labels": [str(value) for value in board_state.black_mill_completion_node_labels],
        "overlapping_piece_ids": [str(value) for value in board_state.overlapping_piece_ids],
        "annotation_entity_ids": [str(value) for value in annotation_entity_ids],
        "construction_mode": str(construction_mode),
        "answer": int(answer_value),
    }
    execution.update(dict(execution_extra or {}))
    return {
        "scene_ir": {
            "scene_kind": f"games_nine_mens_morris_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": relations,
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
    "build_morris_common_trace_params",
    "build_morris_trace_payload",
]
