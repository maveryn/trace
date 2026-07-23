"""Neutral lifecycle plumbing for violin chart public tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.violin.shared.annotations import bbox_artifacts_for_label
from trace_tasks.tasks.charts.violin.shared.defaults import SCENE_ID, SCENE_VARIANT
from trace_tasks.tasks.charts.violin.shared.output import build_trace_payload
from trace_tasks.tasks.charts.violin.shared.prompts import build_prompt_artifacts
from trace_tasks.tasks.charts.violin.shared.rendering import render_violin_dataset, resolve_mark_style
from trace_tasks.tasks.charts.violin.shared.sampling import build_violin_dataset, normalize_support_trace
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions


@dataclass(frozen=True)
class ViolinTaskPlan:
    """Public-task-owned semantic plan for one violin instance."""

    violins: Sequence[Any]
    support_by_label: Mapping[str, Mapping[str, Any]]
    trace_extras: Mapping[str, Any]
    answer_label: str
    annotation_values: Sequence[int]
    prompt_query_key: str
    extra_relations: Mapping[str, Any] | None = None


BuildViolinPlan = Callable[
    [Mapping[str, Any], int, str, Mapping[str, float], Mapping[str, Any]],
    ViolinTaskPlan,
]


def sample_violin_support(
    *,
    dataset_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    mark_style: Mapping[str, Any],
) -> tuple[Sequence[Any], Mapping[str, Mapping[str, Any]], Mapping[str, Any]]:
    """Sample the scene dataset and normalized support table for one objective."""

    violins, _, _, trace_extras = build_violin_dataset(
        dataset_variant=str(dataset_variant),
        params=params,
        instance_seed=int(instance_seed),
        mark_style=mark_style,
    )
    return violins, normalize_support_trace(trace_extras), trace_extras


def _relations_and_prompt_params(
    *,
    selected_branch: str,
    probabilities: Mapping[str, float],
    plan: ViolinTaskPlan,
) -> tuple[dict[str, Any], dict[str, Any]]:
    relations = {
        "query_id": str(selected_branch),
        "scene_variant": SCENE_VARIANT,
        "answer_label": str(plan.answer_label),
        "annotation_label": str(plan.answer_label),
        "annotation_values": [int(value) for value in plan.annotation_values],
        "generation_profile": str(plan.trace_extras.get("generation_profile", "")),
        "question_format": "label_open",
        **dict(plan.extra_relations or {}),
    }
    prompt_params = {
        **dict(relations),
        "query_id_probabilities": dict(probabilities),
        "category_count": int(plan.trace_extras["category_count"]),
        "category_count_range": list(plan.trace_extras["category_count_range"]),
        "value_range": list(plan.trace_extras["value_range"]),
    }
    return relations, prompt_params


def run_violin_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    build_plan: BuildViolinPlan,
) -> TaskOutput:
    """Run retry/render/output plumbing around a public task's semantic plan."""

    selected_branch, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = (
            int(instance_seed)
            if attempt_index == 0
            else int(hash64(int(instance_seed), str(task.task_id), attempt_index))
        )
        try:
            mark_style = resolve_mark_style(task_params, instance_seed=int(attempt_seed))
            plan = build_plan(
                task_params,
                int(attempt_seed),
                str(selected_branch),
                probabilities,
                mark_style,
            )
            artifacts = render_violin_dataset(
                violins=plan.violins,
                params=task_params,
                instance_seed=int(attempt_seed),
                mark_style=mark_style,
            )
            annotation = bbox_artifacts_for_label(artifacts.rendered_scene, str(plan.answer_label))
            prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(plan.prompt_query_key),
                instance_seed=int(attempt_seed),
            )
            relations, prompt_params = _relations_and_prompt_params(
                selected_branch=str(selected_branch),
                probabilities=probabilities,
                plan=plan,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                answer_gt=TypedValue(type="string", value=str(plan.answer_label)),
                annotation_gt=annotation.annotation_gt,
                image=artifacts.image,
                image_id="img0",
                trace_payload=build_trace_payload(
                    prompt_artifacts=prompt_artifacts,
                    artifacts=artifacts,
                    support_by_label=plan.support_by_label,
                    trace_extras=plan.trace_extras,
                    answer_label=str(plan.answer_label),
                    annotation_values=list(plan.annotation_values),
                    relations=relations,
                    prompt_params=prompt_params,
                    witness_symbolic={"type": "object", "label": str(plan.answer_label)},
                    projected_annotation=annotation.projected_annotation,
                ),
                task_versions=default_task_versions(),
                query_id=str(selected_branch),
                scene_id=SCENE_ID,
                prompt_variants=dict(prompt_artifacts.prompt_variants),
            )
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {task.task_id}: {last_error}")


def generate_violin_task(task: Any, instance_seed: int, params: Mapping[str, Any], max_attempts: int) -> TaskOutput:
    """Public-class generate adapter for task-owned violin plans."""

    return run_violin_lifecycle(
        task=task,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        default_query_id=str(task.default_query_id),
        build_plan=task.build_plan,
    )


__all__ = [
    "BuildViolinPlan",
    "ViolinTaskPlan",
    "generate_violin_task",
    "run_violin_lifecycle",
    "sample_violin_support",
]
