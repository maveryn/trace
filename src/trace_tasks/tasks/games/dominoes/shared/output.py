"""Objective-neutral trace assembly for dominoes games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedDominoTaskContext
from .state import DominoSceneAxes, SampledDominoScene


def build_domino_common_trace_params(
    *,
    axes: DominoSceneAxes,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared domino prompt params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_domino_trace_payload(
    *,
    annotation_artifacts: Any,
    annotation_entity_ids: Sequence[str],
    axes: DominoSceneAxes,
    sample: SampledDominoScene,
    rendered_context: RenderedDominoTaskContext,
    prompt_defaults: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    answer_value: int | str,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble dominoes trace sections after task-specific answer binding."""

    rendered_scene = rendered_context.rendered_scene
    layout_kind = str(rendered_scene.render_map.get("layout_kind") or ("chain_tableau" if sample.chain_tiles else "tableau"))
    return {
        "scene_ir": {
            "scene_kind": f"games_dominoes_{layout_kind}_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "layout_kind": layout_kind,
                "reference_tile_id": None if sample.reference_tile_id is None else str(sample.reference_tile_id),
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
            "layout_kind": layout_kind,
            "answer": answer_value,
            "reference_tile_id": None if sample.reference_tile_id is None else str(sample.reference_tile_id),
            "open_end_value": None if sample.open_end_value is None else int(sample.open_end_value),
            "reference_sum": None if sample.reference_sum is None else int(sample.reference_sum),
            "target_total": None if sample.target_total is None else int(sample.target_total),
            "first_step_tile_id": None if sample.first_step_tile_id is None else str(sample.first_step_tile_id),
            "second_step_tile_id": None if sample.second_step_tile_id is None else str(sample.second_step_tile_id),
            "bridge_value": None if sample.bridge_value is None else int(sample.bridge_value),
            "chain_tile_specs": [dict(spec) for spec in sample.chain_tile_specs],
            "candidate_tile_specs": [dict(spec) for spec in sample.candidate_tile_specs],
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = ["build_domino_common_trace_params", "build_domino_trace_payload"]
