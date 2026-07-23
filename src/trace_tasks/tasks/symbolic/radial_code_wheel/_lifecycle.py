"""Neutral lifecycle helpers for radial code-wheel choice tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import role_point_map
from .shared.output import compose_radial_trace_payload, draw_radial_scene_artifacts, radial_entity_trace_records
from .shared.prompts import render_radial_prompt
from .shared.rendering import render_radial_choice_scene
from .shared.sampling import build_with_retries, resolve_radial_scene_variant


@dataclass(frozen=True)
class RadialChoiceTaskBinding:
    """Task-owned identity and prompt/annotation role binding for one objective."""

    seed_namespace: str
    internal_query_key: str
    task_prompt_key: str
    object_description_prefix: str
    annotation_role_names: tuple[str, str, str]
    annotation_hint_key: str
    answer_hint_key: str
    json_example_key: str
    json_example_answer_only_key: str
    failure_message: str


def build_radial_task_output(
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
    radial_metadata: Mapping[str, Any],
) -> TaskOutput:
    """Build the common TaskOutput shell from task-owned radial bindings."""

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
    trace_payload = compose_radial_trace_payload(
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
            "radial_code_wheel_metadata": dict(radial_metadata),
            "radial_code_wheel_entities": radial_entity_trace_records(projection.entity_records),
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


def run_radial_choice_instance(
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
    radial_metadata_factory: Callable[[Any], Mapping[str, Any]],
) -> TaskOutput:
    """Run the common render/prompt/output lifecycle for radial option tasks."""

    scene_variant, scene_variant_probabilities = resolve_radial_scene_variant(
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
    render_artifacts = draw_radial_scene_artifacts(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        namespace=str(scene_variant_namespace).replace(".scene_variant", ".background"),
        render_scene=render_scene,
        render_kwargs=render_kwargs_factory(dataset),
        annotation_source="item_points_px",
    )
    prompt_runtime = render_radial_prompt(
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
    annotation_artifacts = role_point_map(render_artifacts.projection.item_points, annotation_roles_factory(dataset))
    answer_gt = TypedValue(type=str(answer_type), value=answer_value_factory(dataset))
    return build_radial_task_output(
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
        radial_metadata=radial_metadata_factory(dataset),
    )


def run_bound_radial_choice_instance(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    dataset_factory: Callable[[int, str, Mapping[str, float]], Any],
    binding: RadialChoiceTaskBinding,
) -> TaskOutput:
    """Run the standard radial choice lifecycle from a task-owned binding."""

    role_names = tuple(str(role) for role in binding.annotation_role_names)
    return run_radial_choice_instance(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        scene_id="radial_code_wheel",
        internal_query_key=str(binding.internal_query_key),
        scene_variant_namespace=f"{binding.seed_namespace}.scene_variant",
        failure_message=str(binding.failure_message),
        dataset_factory=dataset_factory,
        render_scene=render_radial_choice_scene,
        render_kwargs_factory=lambda d: {
            "reference": d.reference,
            "options": d.options,
            "terminal_specs": d.terminal_specs,
            "target_code": d.target_code,
        },
        task_prompt_key=str(binding.task_prompt_key),
        object_description_key_factory=lambda d: f"{binding.object_description_prefix}_{d.scene_variant}",
        annotation_hint_key=str(binding.annotation_hint_key),
        answer_hint_key=str(binding.answer_hint_key),
        json_example_key=str(binding.json_example_key),
        json_example_answer_only_key=str(binding.json_example_answer_only_key),
        annotation_roles_factory=lambda d: {
            role_names[0]: d.annotation_item_ids[0],
            role_names[1]: d.annotation_item_ids[1],
            role_names[2]: d.annotation_item_ids[2],
        },
        answer_value_factory=lambda d: d.answer_value,
        answer_type="string",
        answer_support_factory=lambda d: d.target_answer_support,
        annotation_item_ids_factory=lambda d: d.annotation_item_ids,
        radial_metadata_factory=lambda d: d.metadata,
    )


__all__ = ["RadialChoiceTaskBinding", "build_radial_task_output", "run_bound_radial_choice_instance", "run_radial_choice_instance"]
