"""Private materialization helpers for scatter-point chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.scatter_points.shared.annotations import (
    bbox_annotation_for_ids,
    point_set_annotation_for_ids,
)
from trace_tasks.tasks.charts.scatter_points.shared.output import (
    answer_value,
    base_execution_record,
    render_map,
    render_spec,
)
from trace_tasks.tasks.charts.scatter_points.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.scatter_points.shared.rendering import render_scatter_points_dataset
from trace_tasks.tasks.charts.scatter_points.shared.state import Dataset, SCENE_ID
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


@dataclass(frozen=True)
class ScatterPointsTaskPlan:
    dataset: Dataset
    params: dict[str, Any]
    prompt_query_key: str
    dynamic_slots: dict[str, Any]
    question_format: str
    program_code: str
    query_params: dict[str, Any]
    reasoning_load: float
    annotation_kind: str = "point_set"


PlanBuilder = Callable[
    [Mapping[str, Any], int, str, Mapping[str, float]],
    ScatterPointsTaskPlan,
]


def scatter_points_attempt_seed(instance_seed: int, task_id: str, attempt: int) -> int:
    """Return the deterministic retry seed for one public objective."""

    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(task_id), int(attempt)))


def materialize_scatter_points_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: ScatterPointsTaskPlan,
) -> TaskOutput:
    """Render a task-owned semantic plan and bind answer/annotation together."""

    dataset = plan.dataset
    rendered = render_scatter_points_dataset(
        dataset=dataset,
        params=dict(plan.params),
        instance_seed=int(instance_seed),
    )
    if str(plan.annotation_kind) == "bbox":
        annotation, witness_symbolic = bbox_annotation_for_ids(dataset=dataset, rendered=rendered)
    elif str(plan.annotation_kind) == "point_set":
        annotation, witness_symbolic = point_set_annotation_for_ids(dataset=dataset, rendered=rendered)
    else:
        raise ValueError(f"unsupported scatter-points annotation kind: {plan.annotation_kind}")
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(plan.prompt_query_key),
        dynamic_slot_values=dict(plan.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    spec = render_spec(rendered)
    spec["scene_variant"] = str(dataset.scene_variant)
    query_params = {
        "query_id": str(selected_query_id),
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant": str(dataset.scene_variant),
        "program_code": str(plan.program_code),
        **dict(plan.query_params),
        **dict(dataset.query.trace),
    }
    execution = {
        **base_execution_record(dataset, rendered),
        "query_id": str(selected_query_id),
        "query_id_probabilities": dict(query_probabilities),
        "question_format": str(plan.question_format),
        "annotation_type": str(annotation.annotation_type),
        "reasoning_load": float(plan.reasoning_load),
        "program_code": str(plan.program_code),
    }
    resolved_answer = answer_value(dataset)
    trace_payload = {
        "scene_ir": {
            "scene_kind": "chart_scatter_points",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "query_id": str(selected_query_id),
                "scene_variant": str(dataset.scene_variant),
                "answer": resolved_answer,
                "annotation_type": str(annotation.annotation_type),
                "annotation_point_ids": [str(point_id) for point_id in dataset.query.annotation_point_ids],
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params=query_params,
        ),
        "render_spec": dict(spec),
        "render_map": render_map(rendered),
        "execution_trace": dict(execution),
        "witness_symbolic": {
            **dict(witness_symbolic),
            "answer": resolved_answer,
            "annotation_type": str(annotation.annotation_type),
            **dict(dataset.query.trace),
        },
        "projected_annotation": dict(annotation.projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(dataset.query.answer_type), value=resolved_answer),
        annotation_gt=annotation.annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


def run_scatter_points_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: PlanBuilder,
) -> TaskOutput:
    """Apply common retry policy to one task-owned scatter-points plan builder."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = scatter_points_attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            attempt_params = {**dict(task_params), "_attempt_index": int(attempt)}
            plan = build_plan(
                dict(attempt_params),
                int(attempt_seed),
                str(selected_query_id),
                dict(query_probabilities),
            )
            return materialize_scatter_points_plan(
                instance_seed=int(attempt_seed),
                selected_query_id=str(selected_query_id),
                query_probabilities=dict(query_probabilities),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}") from last_error


__all__ = [
    "ScatterPointsTaskPlan",
    "materialize_scatter_points_plan",
    "run_scatter_points_lifecycle",
    "scatter_points_attempt_seed",
]
