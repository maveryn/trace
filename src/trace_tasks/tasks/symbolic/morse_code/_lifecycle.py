"""Neutral output plumbing for Morse-code scene tasks."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import role_bbox_map
from .shared.output import compose_morse_trace_payload, draw_morse_scene_artifacts, morse_entity_trace_records
from .shared.prompts import render_morse_prompt
from .shared.sampling import build_with_retries, resolve_morse_scene_variant


def build_morse_task_output(
    *,
    prompt_runtime: Any,
    render_artifacts: Any,
    scene_id: str,
    scene_variant: str,
    internal_query_key: str,
    scene_variant_probabilities: Mapping[str, float],
    target_answer_support: Sequence[Any],
    answer_gt: TypedValue,
    annotation_artifacts: Any,
    annotation_item_ids: Sequence[str],
    morse_metadata: Mapping[str, Any],
) -> TaskOutput:
    """Build the common TaskOutput shell from task-owned Morse bindings."""

    projection = render_artifacts.projection
    query_params = {
        "query_id": str(internal_query_key),
        "internal_query_id": str(internal_query_key),
        "internal_query_id_probabilities": {str(internal_query_key): 1.0},
        "scene_id": str(scene_id),
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": {str(key): float(value) for key, value in scene_variant_probabilities.items()},
        "target_answer_support": [value for value in target_answer_support],
        "question_format": str(internal_query_key),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_runtime.artifacts,
        query_id=str(internal_query_key),
        params=query_params,
    )
    trace_payload = compose_morse_trace_payload(
        scene_id=str(scene_id),
        scene_variant=str(scene_variant),
        projection=projection,
        prompt_query_spec=prompt_query_spec,
        prompt_bundle_id=str(prompt_runtime.metadata["bundle_id"]),
        relations={
            "query_id": str(internal_query_key),
            "internal_query_id": str(internal_query_key),
            "scene_id": str(scene_id),
            "scene_variant": str(scene_variant),
            "answer_value": answer_gt.value,
        },
        execution_trace={
            **dict(query_params),
            "answer_value": answer_gt.value,
            "answer_type": str(answer_gt.type),
            "annotation_item_ids": [str(item) for item in annotation_item_ids],
            "morse_metadata": dict(morse_metadata),
            "morse_entities": morse_entity_trace_records(projection.entity_records),
        },
        annotation_artifacts=annotation_artifacts,
        answer_gt=answer_gt,
    )
    return TaskOutput(
        prompt=str(prompt_runtime.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=render_artifacts.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=dict(projection.task_versions),
        scene_id=str(scene_id),
        query_id=str(internal_query_key),
        prompt_variants=dict(prompt_runtime.prompt_variants),
    )


def run_morse_choice_instance(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    scene_id: str,
    internal_query_key: str,
    scene_variant_namespace: str,
    failure_message: str,
    dataset_factory: Callable[[int, str, Mapping[str, float]], Any],
    render_scene: Callable[..., Any],
    render_kwargs_factory: Callable[[Any], Mapping[str, Any]],
    task_prompt_key: str,
    object_description_key_factory: Callable[[Any], str],
    annotation_hint_key: str,
    answer_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    annotation_roles_factory: Callable[[Any], Mapping[str, str]],
    answer_value_factory: Callable[[Any], Any],
    answer_type: str,
    answer_support_factory: Callable[[Any], Sequence[Any]],
    annotation_item_ids_factory: Callable[[Any], Sequence[str]],
    morse_metadata_factory: Callable[[Any], Mapping[str, Any]],
) -> TaskOutput:
    """Run the common render/prompt/output lifecycle for Morse option tasks."""

    scene_variant, scene_variant_probabilities = resolve_morse_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(scene_variant_namespace),
    )
    dataset = build_with_retries(
        lambda retry_seed: dataset_factory(int(retry_seed), str(scene_variant), scene_variant_probabilities),
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        failure_message=str(failure_message),
    )
    render_artifacts = draw_morse_scene_artifacts(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        namespace=str(scene_variant_namespace).replace(".scene_variant", ".background"),
        render_scene=render_scene,
        render_kwargs=render_kwargs_factory(dataset),
        annotation_source="item_bboxes_px",
    )
    prompt_runtime = render_morse_prompt(
        prompt_defaults,
        domain="symbolic",
        scene_id=str(scene_id),
        scene_variant=str(scene_variant),
        task_key=str(task_prompt_key),
        object_description_key=str(object_description_key_factory(dataset)),
        annotation_hint_key=str(annotation_hint_key),
        answer_hint_key=str(answer_hint_key),
        json_example_key=str(json_example_key),
        json_example_answer_only_key=str(json_example_answer_only_key),
        instance_seed=int(instance_seed),
        context=f"prompt defaults for {scene_variant_namespace}",
    )
    annotation_artifacts = role_bbox_map(render_artifacts.projection.item_bboxes, annotation_roles_factory(dataset))
    answer_gt = TypedValue(type=str(answer_type), value=answer_value_factory(dataset))
    return build_morse_task_output(
        prompt_runtime=prompt_runtime,
        render_artifacts=render_artifacts,
        scene_id=str(scene_id),
        scene_variant=str(scene_variant),
        internal_query_key=str(internal_query_key),
        scene_variant_probabilities=scene_variant_probabilities,
        target_answer_support=answer_support_factory(dataset),
        annotation_artifacts=annotation_artifacts,
        answer_gt=answer_gt,
        annotation_item_ids=annotation_item_ids_factory(dataset),
        morse_metadata=morse_metadata_factory(dataset),
    )


__all__ = ["build_morse_task_output", "run_morse_choice_instance"]
