"""Neutral lifecycle helpers for error-interval public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.error_interval.shared.annotations import annotation_payload
from trace_tasks.tasks.charts.error_interval.shared.defaults import SCENE_ID, SCENE_NAMESPACE, support_probability_map
from trace_tasks.tasks.charts.error_interval.shared.output import build_trace_scaffold, render_dataset
from trace_tasks.tasks.charts.error_interval.shared.prompts import build_prompt_artifacts, dynamic_slots
from trace_tasks.tasks.charts.error_interval.shared.sampling import (
    choose_labels,
    construct_reference_intervals,
    construct_width_rank_intervals,
    palette,
    resolve_scene_variant,
    sample_category_count,
    sample_int_range,
    sample_title,
)
from trace_tasks.tasks.charts.error_interval.shared.state import _Dataset, _Query
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class ErrorIntervalTaskPlan:
    """Task-owned semantic plan for one error-interval objective sample."""

    dataset: _Dataset
    params: Mapping[str, Any]
    answer_gt: TypedValue
    prompt_artifacts: PromptTraceArtifacts
    relation_params: Mapping[str, Any]


@dataclass(frozen=True)
class MaterializedErrorIntervalTask:
    """Rendered payload assembled from a public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]

    def task_output_kwargs(self) -> dict[str, Any]:
        """Return neutral kwargs for the public file's final TaskOutput call."""

        return {
            "prompt": self.prompt,
            "answer_gt": self.answer_gt,
            "annotation_gt": self.annotation_gt,
            "image": self.image,
            "image_id": "img0",
            "trace_payload": self.trace_payload,
            "scene_id": SCENE_ID,
            "query_id": self.query_id,
            "prompt_variants": dict(self.prompt_variants),
        }


def error_interval_attempt_seed(instance_seed: int, task_id: str, attempt: int) -> int:
    """Return the deterministic retry seed for one error-interval public task."""

    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(task_id), int(attempt)))


def _base_scene_parts(params: Mapping[str, Any], *, instance_seed: int) -> tuple[int, dict[str, float], list[str], list[tuple[int, int, int]], str, str, dict[str, float]]:
    """Sample render-variant-independent category labels, colors, title, and scene variant."""

    category_count, category_count_probabilities = sample_category_count(params, instance_seed=int(instance_seed))
    labels = choose_labels(count=int(category_count), instance_seed=int(instance_seed))
    colors = palette(params, count=int(category_count), instance_seed=int(instance_seed))
    title = sample_title(params, instance_seed=int(instance_seed))
    scene_variant, scene_variant_probabilities = resolve_scene_variant(params, instance_seed=int(instance_seed))
    return int(category_count), dict(category_count_probabilities), labels, colors, str(title), str(scene_variant), dict(scene_variant_probabilities)


def build_reference_count_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    selected_query_id: str,
    prompt_key: str,
    predicate: str,
) -> ErrorIntervalTaskPlan:
    """Bind a count of intervals satisfying one reference-value predicate."""

    category_count, category_count_probabilities, labels, colors, title, scene_variant, scene_variant_probabilities = _base_scene_parts(
        params,
        instance_seed=int(instance_seed),
    )
    answer_count, answer_count_probabilities = sample_int_range(
        params,
        min_key="reference_answer_count_min",
        max_key="reference_answer_count_max",
        fallback_min=1,
        fallback_max=5,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.reference.answer_count.{predicate}",
    )
    items, reference_value, annotation_item_ids, construction_meta = construct_reference_intervals(
        predicate=str(predicate),
        category_count=int(category_count),
        answer_count=int(answer_count),
        labels=labels,
        colors=colors,
        params=params,
        instance_seed=int(instance_seed),
    )
    relation_params = {
        "query_id": str(selected_query_id),
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "category_count": int(category_count),
        "category_count_probabilities": dict(category_count_probabilities),
        "reference_predicate": str(predicate),
        "reference_value": int(reference_value),
        "answer_value": int(answer_count),
        "answer_count_probabilities": dict(answer_count_probabilities),
        **dict(construction_meta),
    }
    dataset = _Dataset(
        items=tuple(items),
        prompt_key=str(prompt_key),
        query_probabilities={},
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        reference_value=int(reference_value),
        title=str(title),
        query=_Query(
            prompt_key=str(prompt_key),
            answer=int(answer_count),
            answer_type="integer",
            annotation_type="segment_set",
            annotation_item_ids=tuple(annotation_item_ids),
            params=dict(relation_params),
        ),
    )
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_key),
        dynamic_slot_values=dynamic_slots(dataset),
        instance_seed=int(instance_seed),
    )
    return ErrorIntervalTaskPlan(
        dataset=dataset,
        params=dict(params),
        answer_gt=TypedValue(type="integer", value=int(answer_count)),
        prompt_artifacts=prompt_artifacts,
        relation_params=relation_params,
    )


def build_width_rank_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    selected_query_id: str,
    prompt_key: str,
    rank_key: str,
    relation_phrase: str,
) -> ErrorIntervalTaskPlan:
    """Bind the category label whose interval width has the requested rank."""

    category_count, category_count_probabilities, labels, colors, title, scene_variant, scene_variant_probabilities = _base_scene_parts(
        params,
        instance_seed=int(instance_seed),
    )
    items, annotation_item_ids, answer_label, construction_meta = construct_width_rank_intervals(
        rank_key=str(rank_key),
        category_count=int(category_count),
        labels=labels,
        colors=colors,
        instance_seed=int(instance_seed),
    )
    relation_params = {
        "query_id": str(selected_query_id),
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "category_count": int(category_count),
        "category_count_probabilities": dict(category_count_probabilities),
        "width_rank_key": str(rank_key),
        "relation_phrase": str(relation_phrase),
        "answer_value": str(answer_label),
        **dict(construction_meta),
    }
    dataset = _Dataset(
        items=tuple(items),
        prompt_key=str(prompt_key),
        query_probabilities={},
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        reference_value=None,
        title=str(title),
        query=_Query(
            prompt_key=str(prompt_key),
            answer=str(answer_label),
            answer_type="string",
            annotation_type="segment",
            annotation_item_ids=tuple(annotation_item_ids),
            params=dict(relation_params),
        ),
    )
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_key),
        dynamic_slot_values=dynamic_slots(dataset),
        instance_seed=int(instance_seed),
    )
    return ErrorIntervalTaskPlan(
        dataset=dataset,
        params=dict(params),
        answer_gt=TypedValue(type="string", value=str(answer_label)),
        prompt_artifacts=prompt_artifacts,
        relation_params=relation_params,
    )


def materialize_error_interval_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: ErrorIntervalTaskPlan,
) -> MaterializedErrorIntervalTask:
    """Render a task-owned plan and project its bound interval annotations."""

    rendered, render_meta, sidecar_meta = render_dataset(plan.dataset, params=plan.params, instance_seed=int(instance_seed))
    annotation_type, annotation, projected_annotation, annotation_refs = annotation_payload(
        dataset=plan.dataset,
        rendered=rendered,
    )
    trace_payload = build_trace_scaffold(
        dataset=plan.dataset,
        rendered=rendered,
        render_meta=render_meta,
        sidecar_meta=sidecar_meta,
        projected_annotation=projected_annotation,
        annotation_refs=annotation_refs,
        answer_value=plan.answer_gt.value,
    )
    relation_params = {
        **dict(plan.relation_params),
        "query_id": str(selected_query_id),
        "query_id_probabilities": {str(key): float(value) for key, value in query_probabilities.items()},
    }
    trace_payload["scene_ir"]["relations"].update(relation_params)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params=relation_params,
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    trace_payload["execution_trace"]["query_params"] = relation_params
    return MaterializedErrorIntervalTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=TypedValue(type=str(annotation_type), value=annotation),
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def run_error_interval_plan(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_plan: Any,
) -> TaskOutput:
    """Run neutral query selection, retry, and materialization for a task-owned plan."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = error_interval_attempt_seed(int(instance_seed), str(task_id), int(attempt_index))
        try:
            plan = build_plan(
                int(attempt_seed),
                params=dict(task_params),
                selected_query_id=str(selected_query_id),
            )
            materialized = materialize_error_interval_plan(
                instance_seed=int(attempt_seed),
                selected_query_id=str(selected_query_id),
                query_probabilities=dict(query_probabilities),
                plan=plan,
            )
            return TaskOutput(
                **materialized.task_output_kwargs(),
                task_versions=default_task_versions(),
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task_id}: {last_error}") from last_error


__all__ = [
    "ErrorIntervalTaskPlan",
    "MaterializedErrorIntervalTask",
    "build_reference_count_plan",
    "build_width_rank_plan",
    "error_interval_attempt_seed",
    "materialize_error_interval_plan",
    "run_error_interval_plan",
]
