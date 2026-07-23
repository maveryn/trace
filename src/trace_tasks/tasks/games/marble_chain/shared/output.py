"""Objective-neutral trace fragments for marble-chain tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .rules import closure_creates_same_color_match, closure_pair_color, closure_pair_indices
from .state import MarbleSample, MarbleSceneAxes, RenderedMarbleScene


def build_marble_common_trace_params(
    *,
    axes: MarbleSceneAxes,
    branch_probabilities: Mapping[str, float],
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return shared scene params plus task-owned params."""

    params: Dict[str, Any] = {
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
        "query_id_probabilities": dict(branch_probabilities),
    }
    if extra_params:
        params.update(dict(extra_params))
    return params


def build_shot_option_trace(sample: MarbleSample) -> list[dict[str, Any]]:
    """Serialize visible shot-option mechanics for replay/debug traces."""

    options: list[dict[str, Any]] = []
    for option in sample.option_specs:
        pair_indices = closure_pair_indices(sample.chain_colors, option.outcome)
        options.append(
            {
                "label": str(option.label),
                "entity_id": str(option.entity_id),
                "slot_index": int(option.slot_index),
                "pop_count": int(option.outcome.pop_count),
                "remaining_count": int(option.outcome.remaining_count),
                "popped_indices": [int(index) for index in option.outcome.popped_indices],
                "closure_pair_indices": [int(index) for index in pair_indices],
                "closure_pair_color": closure_pair_color(sample.chain_colors, option.outcome),
                "creates_closure_match": closure_creates_same_color_match(sample.chain_colors, option.outcome),
                "is_answer": bool(option.is_answer),
            }
        )
    return options


def build_marked_outcome_trace(sample: MarbleSample) -> dict[str, Any] | None:
    """Serialize the marked-shot mechanics for value tasks."""

    if sample.marked_outcome is None:
        return None
    return {
        "slot_index": int(sample.marked_outcome.slot_index),
        "pop_count": int(sample.marked_outcome.pop_count),
        "remaining_count": int(sample.marked_outcome.remaining_count),
        "popped_indices": [int(index) for index in sample.marked_outcome.popped_indices],
        "affected_indices": [int(index) for index in sample.marked_outcome.affected_indices],
    }


def build_marble_trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: MarbleSceneAxes,
    sample: MarbleSample,
    rendered: RenderedMarbleScene,
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
    image_size: tuple[int, int],
    answer_value: int | str,
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble marble-chain trace sections after task-specific answer binding."""

    return {
        "scene_ir": {
            "scene_kind": f"games_marble_chain_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(axes.scene_variant),
                "style_variant": str(axes.style_variant),
                "chain_length": int(len(sample.chain_colors)),
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered.style_meta),
            "marble_chain_style": dict(rendered.render_map.get("marble_chain_style", {})),
            "text_style": dict(rendered.render_map.get("text_style", {})),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "chain_colors": list(sample.chain_colors),
            "shooter_color": str(sample.shooter_color),
            "shot_options": build_shot_option_trace(sample),
            "marked_outcome": build_marked_outcome_trace(sample),
            "target_pop_count": None if sample.target_pop_count is None else int(sample.target_pop_count),
            "answer": answer_value,
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            **dict(execution_extra or {}),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults["bundle_id"])},
    }


__all__ = [
    "build_marked_outcome_trace",
    "build_marble_common_trace_params",
    "build_marble_trace_payload",
    "build_shot_option_trace",
]
