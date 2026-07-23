"""Private neutral materialization helpers for histogram chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.histogram.shared.annotations import (
    bbox_artifacts_for_label,
    bbox_map_for_labels,
    bbox_set_artifacts_for_labels,
)
from trace_tasks.tasks.charts.histogram.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.histogram.shared.output import build_trace_scaffold, histogram_relations
from trace_tasks.tasks.charts.histogram.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.histogram.shared.rendering import render_histogram_dataset
from trace_tasks.tasks.charts.histogram.shared.state import HistogramTaskPlan, MaterializedHistogramTask
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


def histogram_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for histogram-scene attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), f"{SCENE_ID}.retry", int(attempt)))
    )


def _annotation_for_plan(plan: HistogramTaskPlan, rendered_labels: dict[str, list[float]]) -> tuple[Any, dict[str, Any]]:
    if str(plan.annotation_type) == "bbox_set":
        return bbox_set_artifacts_for_labels(rendered_labels, plan.annotation_labels)
    if str(plan.annotation_type) == "bbox":
        labels = [str(label) for label in plan.annotation_labels]
        if len(labels) != 1:
            raise RuntimeError("scalar histogram bbox annotation requires exactly one label")
        return bbox_artifacts_for_label(rendered_labels, labels[0])
    raise ValueError(f"unsupported histogram annotation_type: {plan.annotation_type}")


def make_histogram_task_plan(
    *,
    bins: Any,
    params: Mapping[str, Any],
    mark_style: Mapping[str, Any],
    answer_value: int,
    annotation_labels: Any,
    answer_type: str,
    annotation_type: str,
    question_format: str,
    prompt_query_key: str,
    dataset_variant: str,
    trace_extras: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    objective_contract: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> HistogramTaskPlan:
    """Build one task-owned plan from public task semantic arguments."""

    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return HistogramTaskPlan(
        bins=bins,
        params=dict(params),
        mark_style=dict(mark_style),
        answer_gt=TypedValue(type=str(answer_type), value=int(answer_value)),
        answer_value=int(answer_value),
        question_format=str(question_format),
        annotation_type=str(annotation_type),
        annotation_labels=tuple(str(label) for label in annotation_labels),
        relations=histogram_relations(
            prompt_key=str(prompt_query_key),
            dataset_variant=str(dataset_variant),
            trace_extras=dict(trace_extras),
            annotation_labels=annotation_labels,
            query_probabilities=dict(query_probabilities),
            extra={"objective_contract": str(objective_contract)},
        ),
        prompt_artifacts=prompt_artifacts,
    )


def materialize_histogram_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    plan: HistogramTaskPlan,
) -> MaterializedHistogramTask:
    """Render a histogram plan and project task-bound annotations."""

    artifacts = render_histogram_dataset(
        bins=plan.bins,
        params=plan.params,
        mark_style=plan.mark_style,
        instance_seed=int(instance_seed),
    )
    label_to_bbox = bbox_map_for_labels(artifacts.rendered_scene, plan.annotation_labels)
    annotation, witness_symbolic = _annotation_for_plan(plan, label_to_bbox)
    trace_payload = build_trace_scaffold(
        artifacts=artifacts,
        relations=plan.relations,
        answer_value=plan.answer_value,
        question_format=str(plan.question_format),
        annotation_labels=plan.annotation_labels,
        witness_symbolic=witness_symbolic,
        projected_annotation=annotation.projected_annotation,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params={"query_id": str(selected_query_id), **dict(plan.relations)},
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedHistogramTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation.annotation_gt,
        image=artifacts.rendered_scene.image,
        trace_payload=trace_payload,
        branch=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


PlanBuilder = Callable[
    [Mapping[str, Any], int, str, Mapping[str, float]],
    HistogramTaskPlan,
]


def run_histogram_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: PlanBuilder,
) -> TaskOutput:
    """Apply shared retry/materialization policy to a task-owned plan builder."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = histogram_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(
                dict(task_params),
                int(attempt_seed),
                str(selected_query_id),
                dict(query_probabilities),
            )
            materialized = materialize_histogram_plan(
                instance_seed=int(attempt_seed),
                selected_query_id=str(selected_query_id),
                plan=plan,
            )
            return TaskOutput(
                prompt=materialized.prompt,
                answer_gt=materialized.answer_gt,
                annotation_gt=materialized.annotation_gt,
                image=materialized.image,
                image_id="img0",
                trace_payload=materialized.trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=materialized.branch,
                prompt_variants=materialized.prompt_variants,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}") from last_error


__all__ = [
    "histogram_attempt_seed",
    "make_histogram_task_plan",
    "materialize_histogram_plan",
    "run_histogram_lifecycle",
]
