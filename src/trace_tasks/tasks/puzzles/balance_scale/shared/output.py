"""Trace payload fragments for balance-scale public task files."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .rendering import RenderedBalanceContext, balance_render_map
from .state import BalanceSceneAxes, SCENE_ID


def json_ready(value: Any) -> Any:
    """Convert nested balance-scale values to JSON-friendly containers."""

    if isinstance(value, Mapping):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, tuple):
        return [json_ready(inner) for inner in value]
    if isinstance(value, list):
        return [json_ready(inner) for inner in value]
    return value


def balance_trace_params(
    *,
    axes: BalanceSceneAxes,
    prompt_query_key: str,
    answer_type: str,
    extra_params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return scene/task params for prompt-backed query-spec metadata."""

    params: Dict[str, Any] = {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "target_cue_mode": str(axes.target_cue_mode),
        "target_cue_mode_probabilities": dict(axes.target_cue_mode_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "answer_type": str(answer_type),
    }
    if extra_params:
        params.update(json_ready(dict(extra_params)))
    return params


def build_balance_trace_payload(
    *,
    annotation_artifacts: Any,
    axes: BalanceSceneAxes,
    dataset: Mapping[str, Any],
    rendered_context: RenderedBalanceContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    execution_extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Assemble common trace payload sections after task-owned binding."""

    rendered_scene = rendered_context.rendered_scene
    execution = {
        "scene_id": SCENE_ID,
        "scene_variant": str(axes.scene_variant),
        "target_cue_mode": str(axes.target_cue_mode),
        "target_label": str(dataset["target_label"]),
        "answer_value": json_ready(dataset["answer_value"]),
        "object_labels": list(dataset["object_labels"]),
        "object_specs": json_ready(dataset["object_specs"]),
        "object_weights": json_ready(dataset["object_weights"]),
        "panels": json_ready([dict(panel) for panel in dataset["panels"]]),
        "equations": json_ready(
            [dict(equation) for equation in dataset.get("equations", [])]
        ),
        "supporting_role_item_ids": json_ready(dataset["supporting_role_item_ids"]),
    }
    if "comparisons" in dataset:
        execution["comparisons"] = json_ready(
            [dict(comparison) for comparison in dataset["comparisons"]]
        )
    for optional_key in (
        "answer_range",
        "target_answer_support",
        "answer_labels",
        "helper_label",
        "source_label",
        "repeated_label",
        "candidate_labels",
        "order_mode",
        "correct_order",
        "object_weight_support",
        "order_options",
        "query_left_terms",
        "query_right_terms",
        "query_relation_outcomes",
        "consistent_assignment_count",
        "target_relation",
        "relation_options",
        "construction_mode",
    ):
        if optional_key in dataset:
            execution[optional_key] = json_ready(dataset[optional_key])
    if execution_extra:
        execution.update(json_ready(dict(execution_extra)))
    render_map = balance_render_map(rendered_context)
    annotation_type = str(annotation_artifacts.annotation_type)
    if annotation_type == "bbox":
        render_map["annotation_bbox_px"] = list(annotation_artifacts.value)
        render_map["annotation_source"] = "annotation_bbox_px"
    elif annotation_type == "bbox_set":
        render_map["annotation_bboxes_px"] = [
            list(value) for value in annotation_artifacts.value
        ]
        render_map["annotation_source"] = "annotation_bboxes_px"
    else:
        render_map["keyed_item_bboxes_px"] = {
            str(key): list(value) for key, value in annotation_artifacts.value.items()
        }
        render_map["annotation_source"] = "keyed_item_bboxes_px"
    return {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "target_label": str(dataset["target_label"]),
                "answer_value": json_ready(dataset["answer_value"]),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": dict(rendered_context.render_meta),
        "render_map": render_map,
        "execution_trace": execution,
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_type),
            "value": json_ready(annotation_artifacts.value),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "prompt_spec": {
            "defaults": dict(prompt_defaults),
            "active": dict(prompt_artifacts.prompt_variant),
        },
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }


__all__ = [
    "balance_trace_params",
    "build_balance_trace_payload",
    "json_ready",
]
