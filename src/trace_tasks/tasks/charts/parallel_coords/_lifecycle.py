"""Private lifecycle helpers for parallel-coordinates public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ....core.seed import hash64
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec
from .shared.annotations import (
    crossing_point_set,
    point_set_annotation,
    profile_segment,
    profile_segment_set,
    segment_annotation,
    segment_set_annotation,
)
from .shared.defaults import SCENE_ID
from .shared.output import build_trace_payload
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import render_dataset
from .shared.state import ParallelDataset, ParallelRenderResult


@dataclass(frozen=True)
class ParallelCoordsTaskPlan:
    """Task-owned semantic plan after sampling, rendering, and annotation binding."""

    dataset: ParallelDataset
    rendered: ParallelRenderResult
    prompt_artifacts: PromptTraceArtifacts
    answer_gt: TypedValue
    annotation_gt: TypedValue
    witness_symbolic: dict[str, Any]
    projected_annotation: dict[str, Any]
    trace_params: dict[str, Any]
    annotation_profile_ids: tuple[str, ...]


def parallel_attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    """Return the deterministic retry seed for a public task attempt."""

    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def crossing_point_set_plan(
    *,
    dataset: ParallelDataset,
    params: dict[str, Any],
    instance_seed: int,
    prompt_branch_key: str,
) -> ParallelCoordsTaskPlan:
    """Render and annotate crossing-count tasks with one point per counted crossing."""

    rendered = render_dataset(dataset=dataset, params=params, instance_seed=int(instance_seed))
    points = crossing_point_set(
        dataset,
        rendered.rendered_scene,
        crossing_pairs=dataset.query.crossing_pairs,
    )
    annotation_gt, witness_symbolic, projected_annotation = point_set_annotation(points)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_branch_key),
        dynamic_slot_values=dynamic_slots(dataset),
        instance_seed=int(instance_seed),
    )
    return ParallelCoordsTaskPlan(
        dataset=dataset,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="integer", value=int(dataset.query.answer)),
        annotation_gt=annotation_gt,
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        trace_params={"prompt_branch_key": str(prompt_branch_key), **dict(dataset.query.params)},
        annotation_profile_ids=tuple(str(value) for value in dataset.query.annotation_profile_ids),
    )


def profile_segment_set_plan(
    *,
    dataset: ParallelDataset,
    params: dict[str, Any],
    instance_seed: int,
    prompt_branch_key: str,
    extra_trace_params: dict[str, Any],
) -> ParallelCoordsTaskPlan:
    """Render and annotate count tasks with one segment per counted profile line."""

    rendered = render_dataset(dataset=dataset, params=params, instance_seed=int(instance_seed))
    segments = profile_segment_set(
        dataset,
        rendered.rendered_scene,
        profile_ids=dataset.query.annotation_profile_ids,
    )
    annotation_gt, witness_symbolic, projected_annotation = segment_set_annotation(segments)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_branch_key),
        dynamic_slot_values=dynamic_slots(dataset),
        instance_seed=int(instance_seed),
    )
    return ParallelCoordsTaskPlan(
        dataset=dataset,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="integer", value=int(dataset.query.answer)),
        annotation_gt=annotation_gt,
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        trace_params={**dict(extra_trace_params), **dict(dataset.query.params)},
        annotation_profile_ids=tuple(str(value) for value in dataset.query.annotation_profile_ids),
    )


def profile_segment_plan(
    *,
    dataset: ParallelDataset,
    params: dict[str, Any],
    instance_seed: int,
    prompt_branch_key: str,
    extra_trace_params: dict[str, Any],
) -> ParallelCoordsTaskPlan:
    """Render and annotate one selected profile segment."""

    rendered = render_dataset(dataset=dataset, params=params, instance_seed=int(instance_seed))
    target_profile_id = str(dataset.query.annotation_profile_ids[0])
    segment = profile_segment(dataset, rendered.rendered_scene, profile_id=target_profile_id)
    annotation_gt, witness_symbolic, projected_annotation = segment_annotation(segment)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(prompt_branch_key),
        dynamic_slot_values=dynamic_slots(dataset),
        instance_seed=int(instance_seed),
    )
    return ParallelCoordsTaskPlan(
        dataset=dataset,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        answer_gt=TypedValue(type="string", value=str(dataset.query.answer)),
        annotation_gt=annotation_gt,
        witness_symbolic=witness_symbolic,
        projected_annotation=projected_annotation,
        trace_params={**dict(extra_trace_params), **dict(dataset.query.params)},
        annotation_profile_ids=tuple(str(value) for value in dataset.query.annotation_profile_ids),
    )


def materialize_parallel_plan(
    *,
    selected: str,
    probabilities: dict[str, float],
    plan: ParallelCoordsTaskPlan,
) -> TaskOutput:
    """Assemble prompt, trace payload, answer, annotation, and image into output."""

    trace_params = {
        "query_id_probabilities": dict(probabilities),
        **dict(plan.trace_params),
    }
    prompt_spec = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected),
        params=trace_params,
    )
    trace_payload = build_trace_payload(
        dataset=plan.dataset,
        rendered=plan.rendered,
        prompt_spec=prompt_spec,
        relation_payload={"query_id": str(selected), **trace_params},
        execution_payload={"query_id": str(selected), **trace_params},
        witness_symbolic=plan.witness_symbolic,
        projected_annotation=plan.projected_annotation,
        annotation_type=str(plan.annotation_gt.type),
        annotation_profile_ids=plan.annotation_profile_ids,
    )
    return TaskOutput(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=plan.annotation_gt,
        image=plan.rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def run_parallel_coords_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    """Run one public task through shared retry and materialization plumbing."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params={**dict(getattr(task, "task_param_defaults", {})), **dict(params)},
        supported_query_ids=tuple(str(value) for value in getattr(task, "supported_query_ids")),
        default_query_id=str(getattr(task, "default_query_id")),
        task_id=str(getattr(task, "task_id")),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = parallel_attempt_seed(int(instance_seed), str(getattr(task, "task_id")), int(attempt))
        try:
            plan = task._build_plan(dict(task_params), int(attempt_seed), str(selected))
            return materialize_parallel_plan(selected=str(selected), probabilities=probabilities, plan=plan)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {getattr(task, 'task_id')}: {last_error}")


__all__ = [
    "ParallelCoordsTaskPlan",
    "crossing_point_set_plan",
    "profile_segment_plan",
    "profile_segment_set_plan",
    "run_parallel_coords_task",
]
