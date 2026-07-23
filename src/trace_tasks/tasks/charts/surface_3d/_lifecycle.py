"""Private lifecycle plumbing for synthetic 3D chart public tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.surface_3d.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.surface_3d.shared.output import build_trace_payload
from trace_tasks.tasks.charts.surface_3d.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.surface_3d.shared.rendering import render_surface_3d_dataset
from trace_tasks.tasks.charts.surface_3d.shared.state import RenderedSurface3D, Surface3DDataset
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


BindSurface3DAnnotation = Callable[[RenderedSurface3D], AnnotationArtifacts]


@dataclass(frozen=True)
class Surface3DTaskPlan:
    """Public-task-owned plan for one 3D chart instance."""

    dataset: Surface3DDataset
    answer_gt: TypedValue
    annotation_builder: BindSurface3DAnnotation
    prompt_query_key: str
    dynamic_slots: Mapping[str, Any]
    branch_params: Mapping[str, Any]
    relations: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    question_format: str


BuildSurface3DPlan = Callable[
    [Mapping[str, Any], int, str, Mapping[str, float]],
    Surface3DTaskPlan,
]


def _attempt_seed(instance_seed: int, attempt_index: int, *, task: Any) -> int:
    return (
        int(instance_seed)
        if int(attempt_index) == 0
        else int(hash64(int(instance_seed), str(task.task_id), int(attempt_index)))
    )


def run_surface_3d_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: BuildSurface3DPlan,
) -> TaskOutput:
    """Select a branch, render the 3D chart, and assemble TaskOutput."""

    selected_branch, probabilities, task_params = select_task_query_id(
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
                dict(probabilities),
            )
            artifacts = render_surface_3d_dataset(
                dataset=plan.dataset,
                params=task_params,
                instance_seed=int(attempt_seed),
                font_namespace=f"{task.task_id}.chart_font",
            )
            annotation = plan.annotation_builder(artifacts.rendered_scene)
            prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(plan.prompt_query_key),
                dynamic_slots=dict(plan.dynamic_slots),
                instance_seed=int(attempt_seed),
            )
            query_params = {
                "prompt_query_key": str(plan.prompt_query_key),
                "query_id_probabilities": dict(probabilities),
                **dict(plan.branch_params),
            }
            trace_payload = build_trace_payload(
                artifacts=artifacts,
                dataset=plan.dataset,
                answer_value=plan.answer_gt.value,
                answer_type=str(plan.answer_gt.type),
                question_format=str(plan.question_format),
                relations=dict(plan.relations),
                witness_symbolic=dict(plan.witness_symbolic),
                projected_annotation=annotation.projected_annotation,
            )
            trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_branch)
            trace_payload["scene_ir"]["relations"]["prompt_query_key"] = str(plan.prompt_query_key)
            trace_payload["query_spec"] = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_branch),
                params=query_params,
            )
            trace_payload["execution_trace"]["query_id"] = str(selected_branch)
            trace_payload["execution_trace"]["prompt_query_key"] = str(plan.prompt_query_key)
            trace_payload["execution_trace"].update(dict(plan.branch_params))
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=plan.answer_gt,
                annotation_gt=annotation.annotation_gt,
                image=artifacts.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_branch),
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}") from last_error


__all__ = [
    "BindSurface3DAnnotation",
    "BuildSurface3DPlan",
    "Surface3DTaskPlan",
    "run_surface_3d_lifecycle",
]
