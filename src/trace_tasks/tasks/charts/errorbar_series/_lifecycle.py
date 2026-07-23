"""Neutral lifecycle helpers for error-bar series public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.errorbar_series.shared.annotations import annotation_payload
from trace_tasks.tasks.charts.errorbar_series.shared.output import build_trace_scaffold, render_dataset
from trace_tasks.tasks.charts.errorbar_series.shared.state import SCENE_ID, ErrorbarDataset
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class ErrorbarSeriesTaskPlan:
    """Task-owned semantic plan for one objective sample."""

    dataset: ErrorbarDataset
    params: Mapping[str, Any]
    answer_gt: TypedValue
    prompt_artifacts: PromptTraceArtifacts
    relation_params: Mapping[str, Any]


@dataclass(frozen=True)
class MaterializedErrorbarSeriesTask:
    """Rendered payload assembled from a public task-owned semantic plan."""

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


def errorbar_attempt_seed(instance_seed: int, namespace: str, attempt: int) -> int:
    """Return the deterministic retry seed for one public task."""

    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(namespace), int(attempt)))


def materialize_errorbar_series_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: ErrorbarSeriesTaskPlan,
) -> MaterializedErrorbarSeriesTask:
    """Render a task-owned plan and project its bound mark annotations."""

    rendered, render_meta, sidecar_meta = render_dataset(plan.dataset, params=plan.params, instance_seed=int(instance_seed))
    annotation_type, annotation, projected_annotation, annotation_refs = annotation_payload(dataset=plan.dataset, rendered=rendered)
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
    trace_payload["execution_trace"]["query_id_probabilities"] = dict(query_probabilities)
    trace_payload["execution_trace"]["query_params"] = relation_params
    return MaterializedErrorbarSeriesTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=TypedValue(type=str(annotation_type), value=annotation),
        image=rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def run_errorbar_series_plan(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_plan: Any,
) -> TaskOutput:
    """Run shared query selection, retry, and materialization for one task plan."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = errorbar_attempt_seed(int(instance_seed), str(task_id), int(attempt_index))
        try:
            plan = build_plan(
                int(attempt_seed),
                params=dict(task_params),
                selected_query_id=str(selected_query_id),
            )
            materialized = materialize_errorbar_series_plan(
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
    "ErrorbarSeriesTaskPlan",
    "MaterializedErrorbarSeriesTask",
    "errorbar_attempt_seed",
    "materialize_errorbar_series_plan",
    "run_errorbar_series_plan",
]
