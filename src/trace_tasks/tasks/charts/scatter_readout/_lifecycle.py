"""Private materialization helpers for scatter-readout chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.scatter_readout.shared.annotations import annotation_for_binding
from trace_tasks.tasks.charts.scatter_readout.shared.output import (
    answer_value,
    base_execution_record,
    render_map,
    render_spec,
)
from trace_tasks.tasks.charts.scatter_readout.shared.prompts import build_prompt_artifacts, dynamic_slots
from trace_tasks.tasks.charts.scatter_readout.shared.rendering import render_scatter_readout_dataset
from trace_tasks.tasks.charts.scatter_readout.shared.sampling import build_base_dataset, select_series_point_pair
from trace_tasks.tasks.charts.scatter_readout.shared.state import QueryBinding, SCENE_ID, SceneDataset
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


@dataclass(frozen=True)
class ScatterReadoutTaskPlan:
    dataset: SceneDataset
    binding: QueryBinding
    params: dict[str, Any]
    prompt_query_key: str
    dynamic_slots: dict[str, Any]
    question_format: str
    program_code: str
    annotation_kind: str
    query_params: dict[str, Any]
    reasoning_load: float


PlanBuilder = Callable[
    [Mapping[str, Any], int, str, Mapping[str, float]],
    ScatterReadoutTaskPlan,
]
NumericPairAnswer = Callable[[Any, Any], int]


def scatter_readout_attempt_seed(instance_seed: int, task_id: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(task_id), int(attempt)))


def materialize_scatter_readout_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: ScatterReadoutTaskPlan,
) -> TaskOutput:
    """Render one task-owned semantic plan and bind prompt, answer, and annotation together."""

    dataset = plan.dataset
    binding = plan.binding
    rendered = render_scatter_readout_dataset(
        dataset=dataset,
        params=dict(plan.params),
        instance_seed=int(instance_seed),
    )
    annotation, witness_symbolic = annotation_for_binding(
        binding=binding,
        rendered=rendered,
        annotation_kind=str(plan.annotation_kind),
    )
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
        **dict(binding.trace),
    }
    execution = {
        **base_execution_record(
            dataset=dataset,
            binding=binding,
            rendered=rendered,
            annotation_type=str(annotation.annotation_type),
            annotation_value=annotation.value,
        ),
        "query_id": str(selected_query_id),
        "query_id_probabilities": dict(query_probabilities),
        "question_format": str(plan.question_format),
        "annotation_type": str(annotation.annotation_type),
        "reasoning_load": float(plan.reasoning_load),
        "program_code": str(plan.program_code),
    }
    resolved_answer = answer_value(binding)
    trace_payload = {
        "scene_ir": {
            "scene_kind": "chart_scatter_readout",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "query_id": str(selected_query_id),
                "scene_variant": str(dataset.scene_variant),
                "answer": resolved_answer,
                "annotation_type": str(annotation.annotation_type),
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
            **dict(binding.trace),
        },
        "projected_annotation": dict(annotation.projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(binding.answer_type), value=resolved_answer),
        annotation_gt=annotation.annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


def two_point_readout_binding(
    *,
    answer: int,
    operation: str,
    target_series_label: str,
    target_point_id: str,
    target_x_label: str,
    target_y_value: int,
    comparison_series_label: str,
    comparison_point_id: str,
    comparison_y_value: int,
) -> QueryBinding:
    """Build the common role-bound trace for two visible same-x readout points."""

    return QueryBinding(
        answer=int(answer),
        answer_type="integer",
        target_series_label=str(target_series_label),
        target_point_id=str(target_point_id),
        annotation_point_ids=(str(target_point_id), str(comparison_point_id)),
        annotation_x_label=str(target_x_label),
        trace={
            "target_series_label": str(target_series_label),
            "target_point_id": str(target_point_id),
            "target_x_label": str(target_x_label),
            "target_y_value": int(target_y_value),
            "comparison_series_label": str(comparison_series_label),
            "comparison_point_id": str(comparison_point_id),
            "comparison_y_value": int(comparison_y_value),
            "annotation_point_ids": [str(target_point_id), str(comparison_point_id)],
            "annotation_x_label": str(target_x_label),
            "answer": int(answer),
            "answer_type": "integer",
            "operation": str(operation),
        },
    )


def single_point_readout_binding(
    *,
    answer: int | str,
    answer_type: str,
    target_series_label: str,
    target_point_id: str,
    target_x_label: str,
    target_y_value: int | str,
    operation: str = "",
    extra_trace: Mapping[str, Any] | None = None,
) -> QueryBinding:
    """Build the common trace for objectives grounded by one visible point."""

    trace: dict[str, Any] = {
        "target_series_label": str(target_series_label),
        "target_point_id": str(target_point_id),
        "target_x_label": str(target_x_label),
        "target_y_value": target_y_value,
        "annotation_point_ids": [str(target_point_id)] if str(target_point_id) else [],
        "annotation_x_label": str(target_x_label),
        "answer": int(answer) if str(answer_type) == "integer" else str(answer),
        "answer_type": str(answer_type),
    }
    if operation:
        trace["operation"] = str(operation)
    if extra_trace:
        trace.update(dict(extra_trace))
    return QueryBinding(
        answer=int(answer) if str(answer_type) == "integer" else str(answer),
        answer_type=str(answer_type),
        target_series_label=str(target_series_label),
        target_point_id=str(target_point_id),
        annotation_point_ids=(str(target_point_id),) if str(target_point_id) else (),
        annotation_x_label=str(target_x_label),
        trace=trace,
    )


def single_point_readout_plan(
    *,
    dataset: SceneDataset,
    binding: QueryBinding,
    params: Mapping[str, Any],
    prompt_query_key: str,
    question_format: str,
    program_code: str,
    query_params: Mapping[str, Any],
    reasoning_load: float,
    include_unanswerable_instruction: bool = False,
) -> ScatterReadoutTaskPlan:
    """Build the common output plan for objectives grounded by one point."""

    return ScatterReadoutTaskPlan(
        dataset=dataset,
        binding=binding,
        params=dict(params),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dynamic_slots(
            binding=binding,
            include_unanswerable_instruction=bool(include_unanswerable_instruction),
        ),
        question_format=str(question_format),
        program_code=str(program_code),
        annotation_kind="target_point",
        query_params=dict(query_params),
        reasoning_load=float(reasoning_load),
    )


def numeric_pair_readout_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    query_probabilities: Mapping[str, float],
    namespace: str,
    prompt_query_key: str,
    question_format: str,
    program_code: str,
    annotation_kind: str,
    operation: str,
    reasoning_load: float,
    answer_fn: NumericPairAnswer,
) -> ScatterReadoutTaskPlan:
    """Build a task-owned plan for objectives comparing two same-x visible points."""

    dataset = build_base_dataset(params=params, instance_seed=int(instance_seed))
    target_series, target_point, comparison_series, comparison_point = select_series_point_pair(
        dataset=dataset,
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    binding = two_point_readout_binding(
        answer=int(answer_fn(target_point, comparison_point)),
        operation=str(operation),
        target_series_label=str(target_series.label),
        target_point_id=str(target_point.point_id),
        target_x_label=str(target_point.x_label),
        target_y_value=int(target_point.y_value),
        comparison_series_label=str(comparison_series.label),
        comparison_point_id=str(comparison_point.point_id),
        comparison_y_value=int(comparison_point.y_value),
    )
    return ScatterReadoutTaskPlan(
        dataset=dataset,
        binding=binding,
        params=dict(params),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dynamic_slots(binding=binding, include_unanswerable_instruction=False),
        question_format=str(question_format),
        program_code=str(program_code),
        annotation_kind=str(annotation_kind),
        query_params={
            "operation": str(operation),
            "query_id_probabilities": dict(query_probabilities),
        },
        reasoning_load=float(reasoning_load),
    )


def run_scatter_readout_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: PlanBuilder,
) -> TaskOutput:
    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = scatter_readout_attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            attempt_params = {**dict(task_params), "_attempt_index": int(attempt)}
            plan = build_plan(
                dict(attempt_params),
                int(attempt_seed),
                str(selected_query_id),
                dict(query_probabilities),
            )
            return materialize_scatter_readout_plan(
                instance_seed=int(attempt_seed),
                selected_query_id=str(selected_query_id),
                query_probabilities=dict(query_probabilities),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}") from last_error


__all__ = [
    "ScatterReadoutTaskPlan",
    "materialize_scatter_readout_plan",
    "numeric_pair_readout_plan",
    "run_scatter_readout_lifecycle",
    "scatter_readout_attempt_seed",
    "single_point_readout_binding",
    "single_point_readout_plan",
    "two_point_readout_binding",
]
