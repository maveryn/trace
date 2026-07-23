"""Private materialization helpers for scientific axis-frame chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.scientific_axis_frame.shared.annotations import segment_for_tick_roles
from trace_tasks.tasks.charts.scientific_axis_frame.shared.output import (
    base_execution_record,
    render_map,
    render_spec,
)
from trace_tasks.tasks.charts.scientific_axis_frame.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.scientific_axis_frame.shared.rendering import render_axis_frame_dataset
from trace_tasks.tasks.charts.scientific_axis_frame.shared.state import AxisFrameDataset, SCENE_ID
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


@dataclass(frozen=True)
class AxisFrameTaskPlan:
    dataset: AxisFrameDataset
    params: dict[str, Any]
    prompt_key: str
    dynamic_slots: dict[str, Any]
    question_format: str
    program_code: str
    query_params: dict[str, Any]
    reasoning_load: float
    highlight_tick_keys: tuple[str, ...]


PlanBuilder = Callable[
    [Mapping[str, Any], int, str, Mapping[str, float]],
    AxisFrameTaskPlan,
]


def axis_frame_attempt_seed(instance_seed: int, public_task_id: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(public_task_id), int(attempt)))


def materialize_axis_frame_plan(
    *,
    instance_seed: int,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    plan: AxisFrameTaskPlan,
) -> TaskOutput:
    """Bind one task-owned semantic plan to prompt, image, answer, and annotation.

    Invariant: the rendered tick labels, annotation segment, answer, and trace
    all come from the same sampled dataset and render pass.
    """

    dataset = plan.dataset
    binding = dataset.binding
    rendered = render_axis_frame_dataset(
        dataset=dataset,
        params=dict(plan.params),
        instance_seed=int(instance_seed),
        highlight_tick_keys=tuple(plan.highlight_tick_keys),
    )
    annotation, witness_symbolic = segment_for_tick_roles(
        rendered=rendered,
        role_tick_keys=dict(binding.annotation_roles),
    )
    prompt_artifacts = build_prompt_artifacts(
        prompt_key=str(plan.prompt_key),
        dynamic_slot_values=dict(plan.dynamic_slots),
        instance_seed=int(instance_seed),
    )
    query_params = {
        "query_id": str(selected_query_id),
        "scene_id": SCENE_ID,
        "query_id_probabilities": dict(query_probabilities),
        "program_code": str(plan.program_code),
        **dict(plan.query_params),
        **dict(binding.trace),
    }
    execution = {
        **base_execution_record(
            dataset=dataset,
            annotation_segment=[list(point) for point in annotation.value],
        ),
        "query_id": str(selected_query_id),
        "query_id_probabilities": dict(query_probabilities),
        "question_format": str(plan.question_format),
        "query_params": dict(binding.trace),
        "annotation_type": str(annotation.annotation_type),
        "reasoning_load": float(plan.reasoning_load),
        "program_code": str(plan.program_code),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": "chart_scientific_axis_frame",
            "entities": [dict(entity) for entity in rendered.rendered_scene.entities],
            "relations": {
                "query_id": str(selected_query_id),
                "scene_id": SCENE_ID,
                "annotation_tick_keys": [str(value) for value in binding.annotation_roles.values()],
                "answer": int(binding.answer),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params=query_params,
        ),
        "render_spec": render_spec(rendered),
        "render_map": render_map(rendered),
        "execution_trace": dict(execution),
        "witness_symbolic": {
            **dict(witness_symbolic),
            "answer": int(binding.answer),
            **dict(binding.trace),
        },
        "projected_annotation": dict(annotation.projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(binding.answer_type), value=int(binding.answer)),
        annotation_gt=annotation.annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


def run_axis_frame_lifecycle(
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
        attempt_seed = axis_frame_attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            attempt_params = {**dict(task_params), "_attempt_index": int(attempt)}
            plan = build_plan(
                dict(attempt_params),
                int(attempt_seed),
                str(selected_query_id),
                dict(query_probabilities),
            )
            return materialize_axis_frame_plan(
                instance_seed=int(attempt_seed),
                selected_query_id=str(selected_query_id),
                query_probabilities=dict(query_probabilities),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}") from last_error


__all__ = [
    "AxisFrameTaskPlan",
    "materialize_axis_frame_plan",
    "run_axis_frame_lifecycle",
]
