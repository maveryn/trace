"""Trace payload helpers for Mancala pit-board tasks."""

from __future__ import annotations

from typing import Any, Mapping

from .annotations import MancalaAnnotationBundle
from .rules import pit_label
from .state import LABELS, SCENE_ID, MancalaSample, MancalaSceneAxes, RenderedMancalaScene


def mancala_counts_by_label(counts: tuple[int, ...]) -> dict[str, int]:
    """Return pit-count values keyed by visible pit label."""

    return {pit_label(index): int(counts[index]) for index in range(len(LABELS))}


def build_mancala_common_trace_params(
    *,
    axes: MancalaSceneAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build prompt-query params shared by Mancala pit-board tasks."""

    params: dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant": str(axes.style_variant),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "query_id_probabilities": {str(key): float(value) for key, value in dict(branch_probabilities).items()},
    }
    params.update(dict(extra_params or {}))
    return params


def build_mancala_execution_trace(
    *,
    branch_field_name: str,
    selected_branch: str,
    sample: MancalaSample,
    answer: Any,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the common symbolic execution trace for one sowing move."""

    target_index = sample.target_index
    trace = {
        str(branch_field_name): str(selected_branch),
        "construction_mode": str(sample.construction_mode),
        "initial_counts_by_label": mancala_counts_by_label(tuple(sample.initial_counts)),
        "final_counts_by_label": mancala_counts_by_label(tuple(sample.final_counts)),
        "source_index": int(sample.source_index),
        "source_label": str(pit_label(sample.source_index)),
        "source_seed_count": int(sample.initial_counts[sample.source_index]),
        "sowing_path_indices": [int(value) for value in sample.sowing_path_indices],
        "sowing_path_labels": [str(pit_label(value)) for value in sample.sowing_path_indices],
        "landing_index": int(sample.landing_index),
        "landing_label": str(pit_label(sample.landing_index)),
        "target_index": None if target_index is None else int(target_index),
        "target_label": None if target_index is None else str(pit_label(int(target_index))),
        "answer": answer,
    }
    trace.update(dict(extra_fields or {}))
    return trace


def build_mancala_trace_payload(
    *,
    annotation_bundle: MancalaAnnotationBundle,
    axes: MancalaSceneAxes,
    rendered_scene: RenderedMancalaScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    selected_branch: str,
    branch_field_name: str,
    execution_trace: Mapping[str, Any],
    relations_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build verifier trace payload from task-owned Mancala bindings."""

    branch_key = str(branch_field_name)
    relations = {
        "scene_id": SCENE_ID,
        branch_key: str(selected_branch),
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "annotation_entity_ids": dict(annotation_bundle.entity_ids),
    }
    relations.update(dict(relations_extra or {}))
    return {
        "scene_ir": {
            "scene_kind": "games_mancala_pit_board",
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
            "mancala_pit_board_style": dict(rendered_scene.style_meta.get("mancala_pit_board_style", {})),
            "marker_font": dict(rendered_scene.style_meta.get("marker_font", {})),
            "effective_pit_width_px": int(rendered_scene.render_map["effective_pit_width_px"]),
            "effective_pit_height_px": int(rendered_scene.render_map["effective_pit_height_px"]),
            "effective_seed_diameter_px": int(rendered_scene.render_map["effective_seed_diameter_px"]),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(annotation_bundle.witness_symbolic),
        "projected_annotation": dict(annotation_bundle.projected_annotation),
        "background": dict(rendered_scene.background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_mancala_common_trace_params",
    "build_mancala_execution_trace",
    "build_mancala_trace_payload",
    "mancala_counts_by_label",
]
