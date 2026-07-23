"""Task-local utility helpers for single-transform option objectives."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.types import TypedValue
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.labeling import LABEL_POOL_A_L
from ....shared.output_metadata import default_task_versions
from ....shared.prompt_variants import build_prompt_query_spec
from ...shared.annotation import bbox_map_annotation

from .annotations import bbox_map_roles, matching_scene_cell
from .prompts import render_single_transform_prompt_artifacts
from .rendering import sample_and_render_single_transform_scene
from .sampling import resolve_fixed_option_count
from .styles import single_transform_style_trace
from .styles import resolve_single_transform_render_params


_OPERATION_TO_TARGET_TRANSFORM: dict[str, str] = {
    "rotate_90_clockwise": "rot270",
    "rotate_90_counterclockwise": "rot90",
    "rotate_180": "rot180",
    "flip_horizontal": "flip_h",
    "flip_vertical": "flip_v",
}
_OPERATION_TO_CUE: dict[str, str] = {
    "rotate_90_clockwise": "Rotate 90 CW",
    "rotate_90_counterclockwise": "Rotate 90 CCW",
    "rotate_180": "Rotate 180",
    "flip_horizontal": "Flip horizontal",
    "flip_vertical": "Flip vertical",
}


def option_labels(option_count: int) -> Tuple[str, ...]:
    """Return stable option labels for a transform-option scene."""

    return tuple(str(value) for value in LABEL_POOL_A_L[: int(option_count)])


def resolve_answer_index(
    rng,
    *,
    params: Mapping[str, Any],
    labels: Sequence[str],
    default_index: int | None = None,
) -> Tuple[int, Dict[str, float]]:
    """Resolve the unique correct option position."""

    label_set = {str(value) for value in labels}
    if params.get("answer_index") is not None:
        index = int(params["answer_index"])
        if not 0 <= index < len(labels):
            raise ValueError("answer_index out of range")
        return index, {str(label): (1.0 if i == index else 0.0) for i, label in enumerate(labels)}
    if params.get("answer_label") is not None:
        label = str(params["answer_label"]).strip().upper()
        if label not in label_set:
            raise ValueError("answer_label is not supported by the sampled option count")
        index = int(list(str(value) for value in labels).index(label))
        return index, {str(candidate): (1.0 if str(candidate) == label else 0.0) for candidate in labels}
    index = int(default_index) if default_index is not None else int(rng.randrange(len(labels)))
    index %= int(len(labels))
    probability = 1.0 / float(len(labels))
    return index, {str(label): float(probability) for label in labels}


def resolve_pool_manifest(params: Mapping[str, Any], generation_defaults: Mapping[str, Any], fallback_defaults: Any) -> str:
    """Resolve the curated icon pool manifest."""

    return str(
        params.get(
            "pool_manifest",
            group_default(generation_defaults, "pool_manifest", fallback_defaults.pool_manifest),
        )
    )


def resolve_transform_check_size_px(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    fallback_defaults: Any,
) -> int:
    """Resolve the transform-signature validation size."""

    return int(
        params.get(
            "transform_check_size_px",
            group_default(generation_defaults, "transform_check_size_px", fallback_defaults.transform_check_size_px),
        )
    )


def transform_for_operation(operation_key: str) -> str:
    """Return the rendered transform id for a semantic operation key."""

    key = str(operation_key)
    if key not in _OPERATION_TO_TARGET_TRANSFORM:
        raise ValueError(f"unsupported transform operation: {key}")
    return str(_OPERATION_TO_TARGET_TRANSFORM[key])


def cue_for_operation(operation_key: str) -> str:
    """Return the visible operation cue for a semantic operation key."""

    key = str(operation_key)
    if key not in _OPERATION_TO_CUE:
        raise ValueError(f"unsupported transform operation: {key}")
    return str(_OPERATION_TO_CUE[key])


def prepare_transform_scene_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
) -> dict[str, Any]:
    """Resolve scene-level sampling inputs shared by transform-option tasks."""

    scene_rng = spawn_rng(int(instance_seed), "scene")
    option_count = resolve_fixed_option_count(
        params,
        generation_defaults=generation_defaults,
        fallback_defaults=fallback_defaults,
    )
    labels = option_labels(int(option_count))
    balance_cursor = params.get("_sample_cursor", int(instance_seed))
    answer_index, answer_label_probabilities = resolve_answer_index(
        scene_rng,
        params=params,
        labels=labels,
        default_index=int(balance_cursor) % max(1, len(labels)),
    )
    render_params = resolve_single_transform_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    return {
        "scene_rng": scene_rng,
        "option_count": int(option_count),
        "labels": tuple(labels),
        "answer_index": int(answer_index),
        "answer_label_probabilities": dict(answer_label_probabilities),
        "render_params": render_params,
        "pool_manifest": resolve_pool_manifest(params, generation_defaults, fallback_defaults),
        "transform_check_size_px": resolve_transform_check_size_px(params, generation_defaults, fallback_defaults),
    }


def render_transform_scene_with_retries(
    rng,
    *,
    instance_seed: int,
    max_attempts: int,
    option_count: int,
    answer_index: int,
    target_transform_id: str,
    operation_cue: str,
    pool_manifest: str,
    transform_check_size_px: int,
    render_params: Mapping[str, Any],
    reference_transform_id: str,
    option_transform_ids: Sequence[str] | None = None,
):
    """Render a neutral transform-option scene, retrying only renderer failures."""

    last_error: Exception | None = None
    for _ in range(max(1, int(max_attempts))):
        try:
            return sample_and_render_single_transform_scene(
                rng,
                instance_seed=int(instance_seed),
                option_count=int(option_count),
                answer_index=int(answer_index),
                target_transform_id=str(target_transform_id),
                operation_cue=str(operation_cue),
                pool_manifest=str(pool_manifest),
                transform_check_size_px=int(transform_check_size_px),
                render_params=render_params,
                reference_transform_id=str(reference_transform_id),
                option_transform_ids=None if option_transform_ids is None else tuple(str(value) for value in option_transform_ids),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError("failed to render single-transform option scene") from last_error


def selected_option_annotation(scene_payload):
    """Return selected option cell, annotation artifacts, and typed annotation."""

    selected_cell = matching_scene_cell(scene_payload.scene_cells)
    annotation_artifacts = bbox_map_annotation(
        bbox_map_roles(
            reference_cell=scene_payload.reference_cell,
            selected_cell=selected_cell,
        )
    )
    annotation_gt = TypedValue(
        type=str(annotation_artifacts["annotation_type"]),
        value=dict(annotation_artifacts["annotation_value"]),
    )
    return selected_cell, annotation_artifacts, annotation_gt


def prepare_selected_option_prompt_binding(
    *,
    instance_seed: int,
    prompt_defaults: Mapping[str, Any],
    operation_key: str,
    public_id: str,
    scene_id: str,
    branch_key: str,
    query_probabilities: Mapping[str, float],
    scene_context: Mapping[str, Any],
    scene_payload,
) -> dict[str, Any]:
    """Bind prompt, option-letter answer, and selected-option annotation."""

    prompt_bundle_defaults, prompt_artifacts = render_single_transform_prompt_artifacts(
        instance_seed=int(instance_seed),
        prompt_defaults=prompt_defaults,
        operation_key=str(operation_key),
    )
    selected_cell, annotation_artifacts, annotation_gt = selected_option_annotation(scene_payload)
    answer_gt = TypedValue(type="option_letter", value=str(scene_payload.answer_label))
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(branch_key),
        params=base_query_params(
            public_id=str(public_id),
            scene_id=str(scene_id),
            query_probabilities=query_probabilities,
            answer_label_probabilities=scene_context["answer_label_probabilities"],
            scene_payload=scene_payload,
            pool_manifest=str(scene_context["pool_manifest"]),
            transform_check_size_px=int(scene_context["transform_check_size_px"]),
        ),
    )
    query_spec["template_id"] = str(prompt_bundle_defaults["bundle_id"])
    return {
        "prompt_artifacts": prompt_artifacts,
        "selected_cell": selected_cell,
        "annotation_artifacts": annotation_artifacts,
        "annotation_gt": annotation_gt,
        "answer_gt": answer_gt,
        "query_spec": dict(query_spec),
    }


def scene_entity_records(scene_payload) -> list[dict[str, Any]]:
    """Return trace entity records for the reference and option cells."""

    return [dict(scene_payload.reference_cell), *[dict(item) for item in scene_payload.scene_cells]]


def pixel_panel_frames(scene_payload) -> dict[str, Any]:
    """Return the common pixel/panel frame payload."""

    return {
        "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
        "panels": dict(scene_payload.panel_geometry),
    }


def transform_render_style(*, render_params: Mapping[str, Any], scene_payload) -> dict[str, Any]:
    """Return common render-style metadata for transform-option scenes."""

    return single_transform_style_trace(
        render_params=render_params,
        sampled_palette_rgb=scene_payload.sampled_palette_rgb,
    )


def selected_option_render_map(scene_payload, selected_cell: Mapping[str, Any]) -> dict[str, Any]:
    """Return the common render-map anchors for transform-option scenes."""

    return {
        "image_id": "img0",
        "anchors": {
            "reference_icon": dict(scene_payload.reference_cell),
            "answer_label": str(scene_payload.answer_label),
            "selected_option": dict(selected_cell),
            "scene_cells": [dict(item) for item in scene_payload.scene_cells],
        },
    }


def common_selected_option_execution_payload(
    *,
    public_id: str,
    scene_id: str,
    branch_key: str,
    query_probabilities: Mapping[str, float],
    scene_variant: str,
    question_format: str,
    scene_payload,
    answer_label_probabilities: Mapping[str, float],
    option_transform_map_key: str,
) -> dict[str, Any]:
    """Return common execution fields for selected-option transform scenes."""

    return {
        "task" + "_id": str(public_id),
        "scene_id": str(scene_id),
        "query" + "_id": str(branch_key),
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant": str(scene_variant),
        "question_format": str(question_format),
        "object_count": int(scene_payload.object_count),
        "cell_labels": list(scene_payload.cell_labels),
        "answer": str(scene_payload.answer_label),
        "answer_label": str(scene_payload.answer_label),
        "answer_label_probabilities": dict(answer_label_probabilities),
        "icon_id": str(scene_payload.icon_id),
        "operation_cue": str(scene_payload.operation_cue),
        "target_transform_id": str(scene_payload.target_transform_id),
        str(option_transform_map_key): {
            str(cell["label"]): str(cell["transform_id"])
            for cell in scene_payload.scene_cells
        },
        "annotation_roles": ["reference_icon", "selected_option"],
    }


def base_query_params(
    *,
    public_id: str,
    scene_id: str,
    query_probabilities: Mapping[str, float],
    answer_label_probabilities: Mapping[str, float],
    scene_payload,
    pool_manifest: str,
    transform_check_size_px: int,
) -> dict[str, Any]:
    """Return common query-spec params for transform-option tasks."""

    return {
        "task" + "_id": str(public_id),
        "scene_id": str(scene_id),
        "query_id_probabilities": dict(query_probabilities),
        "answer_label_probabilities": dict(answer_label_probabilities),
        "object_count": int(scene_payload.object_count),
        "pool_manifest": str(pool_manifest),
        "transform_check_size_px": int(transform_check_size_px),
        "target_transform_id": str(scene_payload.target_transform_id),
        "operation_cue": str(scene_payload.operation_cue),
        "answer_label": str(scene_payload.answer_label),
    }


def render_spec_payload(
    *,
    public_id: str,
    scene_id: str,
    branch_key: str,
    render_params: Mapping[str, Any],
    scene_payload,
) -> dict[str, Any]:
    """Return common render-spec metadata for transform-option tasks."""

    return {
        "task" + "_id": str(public_id),
        "scene_id": str(scene_id),
        "query" + "_id": str(branch_key),
        "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
        "coord_space": "pixel",
        "panel_geometry": dict(scene_payload.panel_geometry),
        "style": transform_render_style(render_params=render_params, scene_payload=scene_payload),
    }


def build_transform_trace_payload(
    *,
    public_id: str,
    scene_id: str,
    branch_key: str,
    scene_kind: str,
    scene_payload,
    selected_cell: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    render_params: Mapping[str, Any],
    relation_payload: Mapping[str, Any],
    execution_payload: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble neutral trace scaffolding after a task binds its semantics."""

    return {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "task" + "_id": str(public_id),
            "scene_id": str(scene_id),
            "query" + "_id": str(branch_key),
            "entities": scene_entity_records(scene_payload),
            "relations": dict(relation_payload),
            "frames": pixel_panel_frames(scene_payload),
        },
        "query_spec": dict(query_spec),
        "render_spec": render_spec_payload(
            public_id=str(public_id),
            scene_id=str(scene_id),
            branch_key=str(branch_key),
            render_params=render_params,
            scene_payload=scene_payload,
        ),
        "render_map": selected_option_render_map(scene_payload, selected_cell),
        "execution_trace": dict(execution_payload),
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(projected_annotation),
    }


def task_output_fields(
    *,
    prompt_artifacts,
    answer_gt: TypedValue,
    annotation_gt: TypedValue,
    image,
    trace_payload: Mapping[str, Any],
    scene_id: str,
    branch_key: str,
) -> dict[str, Any]:
    """Return common keyword fields for a public task's TaskOutput call."""

    return {
        "prompt": str(prompt_artifacts.prompt),
        "answer_gt": answer_gt,
        "annotation_gt": annotation_gt,
        "image": image,
        "image_id": "img0",
        "trace_payload": dict(trace_payload),
        "task_versions": default_task_versions(),
        "scene_id": str(scene_id),
        "query" + "_id": str(branch_key),
        "prompt_variants": dict(prompt_artifacts.prompt_variants),
    }


__all__ = [
    "option_labels",
    "base_query_params",
    "build_transform_trace_payload",
    "common_selected_option_execution_payload",
    "cue_for_operation",
    "prepare_selected_option_prompt_binding",
    "render_transform_scene_with_retries",
    "render_spec_payload",
    "prepare_transform_scene_context",
    "resolve_answer_index",
    "resolve_pool_manifest",
    "resolve_transform_check_size_px",
    "pixel_panel_frames",
    "scene_entity_records",
    "selected_option_annotation",
    "selected_option_render_map",
    "task_output_fields",
    "transform_for_operation",
    "transform_render_style",
]
