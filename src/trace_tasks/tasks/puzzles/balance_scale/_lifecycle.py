"""Scene-private lifecycle for balance-scale public task files."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.visual_defaults import (
    default_noise_fallback,
    load_scene_noise_defaults,
)

from .shared.sampling import resolve_integer_answer, resolve_scene_axes
from .shared.annotations import (
    bbox_annotation_artifacts,
    bbox_map_annotation_artifacts,
    bbox_set_annotation_artifacts,
)
from .shared.output import balance_trace_params, build_balance_trace_payload
from .shared.prompts import build_balance_prompt_artifacts
from .shared.rendering import render_balance_scene_context
from .shared.state import SCENE_ID

DatasetBuilder = Callable[[int], Dict[str, Any]]
NumericDatasetHook = Callable[..., Dict[str, Any]]


def run_balance_scene_lifecycle(
    *,
    task_name: str,
    domain: str,
    selected_query: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    axes: Any,
    instance_seed: int,
    max_attempts: int,
    dataset_builder: DatasetBuilder,
    task_prompt_key: str,
    answer_type: str,
    extra_query_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> TaskOutput:
    """Run neutral render/prompt/annotation plumbing for one task-owned dataset."""

    visual_defaults = load_scene_noise_defaults(
        domain="puzzles",
        scene_id=SCENE_ID,
        fallback=default_noise_fallback(apply_prob=0.15),
        merge_with_fallback=True,
    )
    dataset: Dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            dataset = dataset_builder(int(instance_seed) + int(attempt_index))
            break
        except RuntimeError as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError(
            f"failed to generate balance-scale task {task_name}"
        ) from last_error

    rendered_context = render_balance_scene_context(
        dataset=dataset,
        axes=axes,
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        visual_defaults=visual_defaults,
        namespace=str(task_name),
    )
    item_bbox_map = rendered_context.rendered_scene.item_bbox_map
    if "annotation_item_id" in dataset:
        annotation_artifacts = bbox_annotation_artifacts(
            item_bbox_map,
            str(dataset["annotation_item_id"]),
        )
    elif "annotation_item_ids" in dataset:
        annotation_artifacts = bbox_set_annotation_artifacts(
            item_bbox_map,
            [str(item_id) for item_id in dataset["annotation_item_ids"]],
        )
    else:
        annotation_artifacts = bbox_map_annotation_artifacts(
            item_bbox_map,
            dataset["supporting_role_item_ids"],
        )
    prompt_query_key = str(dataset["prompt_query_key"])
    prompt_defaults, prompt_artifacts = build_balance_prompt_artifacts(
        domain=str(domain),
        task_prompt_key=str(task_prompt_key),
        prompt_query_key=prompt_query_key,
        instance_seed=int(instance_seed),
    )
    query_params = {
        "branch_probabilities": dict(branch_probabilities),
        **dict(extra_query_params or {}),
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=balance_trace_params(
            axes=axes,
            prompt_query_key=prompt_query_key,
            answer_type=str(answer_type),
            extra_params=query_params,
        ),
    )
    trace_payload = build_balance_trace_payload(
        annotation_artifacts=annotation_artifacts,
        axes=axes,
        dataset=dataset,
        rendered_context=rendered_context,
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
        query_spec=query_spec,
        execution_extra={
            "query_id": str(selected_query),
            "prompt_query_key": prompt_query_key,
            **dict(execution_extra or {}),
        },
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(answer_type), value=dataset["answer_value"]),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=rendered_context.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
    )


def run_fixed_numeric_balance_lifecycle(
    *,
    task_name: str,
    domain: str,
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_query_key: str,
    fallback_min: int,
    fallback_max: int,
    dataset_hook: NumericDatasetHook,
) -> TaskOutput:
    """Run fixed-query numeric task plumbing around a task-owned dataset hook."""

    selected_query, branch_probs, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(DEFAULT_QUERY_ID,),
        default_query_id=DEFAULT_QUERY_ID,
        task_id=str(task_name),
        namespace=f"{task_name}.branch",
    )
    answer_value, answer_support = resolve_integer_answer(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(task_name),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    axes = resolve_scene_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(task_name),
        include_target_cue=False,
    )
    return run_balance_scene_lifecycle(
        task_name=str(task_name),
        domain=str(domain),
        selected_query=str(selected_query),
        branch_probabilities=branch_probs,
        task_params=task_params,
        render_defaults=render_defaults,
        axes=axes,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        dataset_builder=lambda seed: dataset_hook(
            task_params=task_params,
            gen_defaults=gen_defaults,
            instance_seed=int(seed),
            answer_value=int(answer_value),
            answer_support=answer_support,
            axes=axes,
            task_name=str(task_name),
            prompt_query_key=str(prompt_query_key),
        ),
        task_prompt_key="balance_scale_query",
        answer_type="integer",
        extra_query_params={
            "answer_range": [int(min(answer_support)), int(max(answer_support))],
            "target_answer_support": [int(value) for value in answer_support],
        },
        execution_extra={"answer": int(answer_value)},
    )


__all__ = ["run_balance_scene_lifecycle", "run_fixed_numeric_balance_lifecycle"]
