"""Private materialization lifecycle for waterfall chart public tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.waterfall.shared.annotations import bbox_map_artifacts
from trace_tasks.tasks.charts.waterfall.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.waterfall.shared.output import build_trace_payload
from trace_tasks.tasks.charts.waterfall.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.waterfall.shared.rendering import render_waterfall_dataset
from trace_tasks.tasks.charts.waterfall.shared.sampling import choose_step_index, sample_waterfall_dataset
from trace_tasks.tasks.charts.waterfall.shared.state import RenderedWaterfall, WaterfallDataset
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


BindWaterfallAnnotation = Callable[[RenderedWaterfall], AnnotationArtifacts]


@dataclass(frozen=True)
class WaterfallTaskPlan:
    """Public-task-owned plan for one materialized waterfall instance."""

    dataset: WaterfallDataset
    answer_gt: TypedValue
    annotation_builder: BindWaterfallAnnotation
    prompt_query_key: str
    dynamic_slots: Mapping[str, Any]
    query_params: Mapping[str, Any]
    relations: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    question_format: str
    threshold_value: int | None = None


BuildWaterfallPlan = Callable[
    [Mapping[str, Any], int, str, Mapping[str, float]],
    WaterfallTaskPlan,
]
CounterfactualValue = Callable[[int, int], int]


def build_counterfactual_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    prompt_query_key: str,
    operation_name: str,
    counterfactual_phrase: str,
    answer_from_final_and_delta: CounterfactualValue,
    query_probabilities: Mapping[str, float],
    step_count_min: int | None = None,
    step_count_max: int | None = None,
    target_delta_abs_min: int | None = None,
    target_delta_abs_max: int | None = None,
    answer_min: int | None = None,
    answer_max: int | None = None,
) -> WaterfallTaskPlan:
    """Build a counterfactual plan from semantic operation arguments.

    Public task files choose the operation and answer transform. This helper
    only applies that task-owned transform to a sampled waterfall sequence and
    binds the shared visual witnesses used by both counterfactual objectives.
    """

    dataset = sample_waterfall_dataset(
        params,
        instance_seed=int(instance_seed),
        step_count_min=step_count_min,
        step_count_max=step_count_max,
    )
    feasible_target_indices: list[int] = []
    for step_index, step in enumerate(dataset.steps):
        delta_abs = abs(int(step.delta))
        if target_delta_abs_min is not None and delta_abs < int(target_delta_abs_min):
            continue
        if target_delta_abs_max is not None and delta_abs > int(target_delta_abs_max):
            continue
        candidate_answer = int(answer_from_final_and_delta(int(dataset.final_value), int(step.delta)))
        if answer_min is not None and candidate_answer < int(answer_min):
            continue
        if answer_max is not None and candidate_answer > int(answer_max):
            continue
        feasible_target_indices.append(int(step_index))
    if feasible_target_indices:
        rng = spawn_rng(int(instance_seed), f"{operation_name}.target_step.constrained")
        target_index = int(uniform_choice(rng, tuple(feasible_target_indices)))
    else:
        target_index = choose_step_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{operation_name}.target_step",
            step_count=len(dataset.steps),
        )
    target_step = dataset.steps[int(target_index)]
    answer_value = int(answer_from_final_and_delta(int(dataset.final_value), int(target_step.delta)))
    if target_delta_abs_min is not None and abs(int(target_step.delta)) < int(target_delta_abs_min):
        raise ValueError("no waterfall target step satisfies minimum delta magnitude")
    if target_delta_abs_max is not None and abs(int(target_step.delta)) > int(target_delta_abs_max):
        raise ValueError("no waterfall target step satisfies maximum delta magnitude")
    if answer_min is not None and int(answer_value) < int(answer_min):
        raise ValueError("counterfactual waterfall answer is below requested range")
    if answer_max is not None and int(answer_value) > int(answer_max):
        raise ValueError("counterfactual waterfall answer is above requested range")

    def _bind_annotation(rendered: RenderedWaterfall) -> AnnotationArtifacts:
        return bbox_map_artifacts(
            {
                "final_total_bar": rendered.bar_bboxes_px["final"],
                "target_contribution_bar": rendered.bar_bboxes_px[str(target_step.step_id)],
            }
        )

    return WaterfallTaskPlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_builder=_bind_annotation,
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={
            "target_step_label": str(target_step.label),
            "counterfactual_phrase": str(counterfactual_phrase),
            "target_delta_text": f"{int(target_step.delta):+d}",
            "reversed_delta_text": f"{-int(target_step.delta):+d}",
        },
        query_params={
            "target_step_id": str(target_step.step_id),
            "target_step_label": str(target_step.label),
            "target_step_index": int(target_index),
            "target_delta": int(target_step.delta),
            "counterfactual_operation": str(operation_name),
            "step_count": int(len(dataset.steps)),
            "start_value": int(dataset.start_value),
            "final_value": int(dataset.final_value),
            "annotation_roles": {
                "final_total_bar": "final",
                "target_contribution_bar": str(target_step.step_id),
            },
        },
        relations={
            "target_step_id": str(target_step.step_id),
            "target_step_label": str(target_step.label),
            "target_step_index": int(target_index),
            "target_delta": int(target_step.delta),
            "counterfactual_operation": str(operation_name),
            "query_id_probabilities": dict(query_probabilities),
        },
        witness_symbolic={
            "type": "waterfall_counterfactual_witness",
            "operation": str(operation_name),
            "final_bar_id": "final",
            "target_step_id": str(target_step.step_id),
            "answer": int(answer_value),
        },
        question_format=f"waterfall_{operation_name}_final_total",
    )


def _attempt_seed(instance_seed: int, attempt_index: int, *, task: Any) -> int:
    return (
        int(instance_seed)
        if int(attempt_index) == 0
        else int(hash64(int(instance_seed), str(task.task_id), int(attempt_index)))
    )


def run_waterfall_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: BuildWaterfallPlan,
) -> TaskOutput:
    """Select a task branch, materialize the scene, and assemble TaskOutput."""

    selected_branch, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = _attempt_seed(int(instance_seed), int(attempt_index), task=task)
        try:
            plan = build_plan(
                dict(task_params),
                int(attempt_seed),
                str(selected_branch),
                dict(query_probabilities),
            )
            artifacts = render_waterfall_dataset(
                dataset=plan.dataset,
                params=task_params,
                instance_seed=int(attempt_seed),
                threshold_value=plan.threshold_value,
            )
            annotation = plan.annotation_builder(artifacts.rendered_scene)
            prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(plan.prompt_query_key),
                dynamic_slots=dict(plan.dynamic_slots),
                instance_seed=int(attempt_seed),
            )
            answer_value = plan.answer_gt.value
            trace_payload = build_trace_payload(
                artifacts=artifacts,
                dataset=plan.dataset,
                answer_value=answer_value,
                answer_type=str(plan.answer_gt.type),
                question_format=str(plan.question_format),
                relations=dict(plan.relations),
                witness_symbolic=dict(plan.witness_symbolic),
                projected_annotation=annotation.projected_annotation,
            )
            query_params = {
                "prompt_query_key": str(plan.prompt_query_key),
                "query_id_probabilities": dict(query_probabilities),
                **dict(plan.query_params),
            }
            trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_branch)
            trace_payload["scene_ir"]["relations"]["prompt_query_key"] = str(plan.prompt_query_key)
            trace_payload["query_spec"] = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=query_params,
            )
            trace_payload["execution_trace"]["query_id"] = str(selected_branch)
            trace_payload["execution_trace"]["prompt_query_key"] = str(plan.prompt_query_key)
            trace_payload["execution_trace"].update(dict(plan.query_params))
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation.annotation_gt,
                image=artifacts.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"failed to generate {task.task_id} after {max_attempts} attempts: {last_error}") from last_error


__all__ = [
    "BindWaterfallAnnotation",
    "BuildWaterfallPlan",
    "CounterfactualValue",
    "WaterfallTaskPlan",
    "build_counterfactual_plan",
    "run_waterfall_lifecycle",
]
