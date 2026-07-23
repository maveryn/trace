"""Private neutral materialization helpers for 3D bar-grid chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.bar_3d.shared.annotations import annotation_artifacts_for_selection
from trace_tasks.tasks.charts.bar_3d.shared.output import bar_grid_relation_fields, build_trace_scaffold
from trace_tasks.tasks.charts.bar_3d.shared.rendering import render_bar_grid_scene
from trace_tasks.tasks.charts.bar_3d.shared.state import SCENE_ID, _Dataset
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class Bar3DTaskPlan:
    """Task-owned semantic plan consumed by neutral 3D bar rendering."""

    dataset: _Dataset
    prompt_artifacts: PromptTraceArtifacts
    relation_fields: Mapping[str, Any]
    question_format: str


@dataclass(frozen=True)
class MaterializedBar3DTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


def bar_3d_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for 3D bar-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), "charts.bar_3d.retry", int(attempt)))
    )


def materialize_bar_3d_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    plan: Bar3DTaskPlan,
) -> MaterializedBar3DTask:
    """Render one task-owned plan and build neutral trace payload sections.

    Public task files own objective selection, answer binding, annotation-bar
    binding, prompt slots, retry handling, and final output construction. This
    helper only projects the already-selected semantic plan into rendered image
    geometry and trace sections.
    """

    artifacts = render_bar_grid_scene(
        dataset=plan.dataset,
        params=params,
        instance_seed=int(instance_seed),
    )
    annotation, witness_symbolic = annotation_artifacts_for_selection(
        rendered=artifacts.rendered,
        selection=plan.dataset.selection,
    )
    trace_payload = build_trace_scaffold(
        dataset=plan.dataset,
        artifacts=artifacts,
        relations=plan.relation_fields,
        projected_annotation=annotation.projected_annotation,
        witness_symbolic=witness_symbolic,
        question_format=str(plan.question_format),
    )
    query_params = {"query_id": str(selected_query_id), **dict(plan.relation_fields)}
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params=query_params,
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedBar3DTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(plan.dataset.selection.answer)),
        annotation_gt=annotation.annotation_gt,
        image=artifacts.rendered.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


def materialize_bar_3d_plan_with_retries(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    build_plan: Callable[[int, Mapping[str, Any], str], Bar3DTaskPlan],
) -> MaterializedBar3DTask:
    """Retry task-owned plan construction, then return the rendered payload."""

    attempts = max(1, int(max_attempts))
    last_error: Exception | None = None
    for attempt in range(attempts):
        attempt_seed = bar_3d_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), params, str(selected_query_id))
            return materialize_bar_3d_plan(
                instance_seed=int(attempt_seed),
                params=params,
                selected_query_id=str(selected_query_id),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to materialize bar_3d plan: {last_error}")


def bar_3d_task_output_fields(materialized: MaterializedBar3DTask) -> dict[str, Any]:
    """Return neutral final-output fields from an already materialized payload."""

    return {
        "prompt": materialized.prompt,
        "answer_gt": materialized.answer_gt,
        "annotation_gt": materialized.annotation_gt,
        "image": materialized.image,
        "image_id": "img0",
        "trace_payload": materialized.trace_payload,
        "task_versions": default_task_versions(),
        "scene_id": SCENE_ID,
        "query_id": materialized.query_id,
        "prompt_variants": materialized.prompt_variants,
    }


def run_bar_3d_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    build_plan: Callable[[int, Mapping[str, Any], str], Bar3DTaskPlan],
    build_output: Callable[[MaterializedBar3DTask], Any],
) -> Any:
    """Materialize a public-file-owned plan and call its output factory."""

    materialized = materialize_bar_3d_plan_with_retries(
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        selected_query_id=str(selected_query_id),
        build_plan=build_plan,
    )
    return build_output(materialized)


def build_bar_3d_plan(
    *,
    dataset: _Dataset,
    ranges: Mapping[str, Any],
    selection_trace: Mapping[str, Any],
    prompt_artifacts: PromptTraceArtifacts,
    question_format: str = "numeric_open",
) -> Bar3DTaskPlan:
    """Package a task-owned dataset and prompt into a neutral render plan."""

    return Bar3DTaskPlan(
        dataset=dataset,
        prompt_artifacts=prompt_artifacts,
        relation_fields=bar_grid_relation_fields(
            dataset=dataset,
            ranges=ranges,
            selection_trace=selection_trace,
        ),
        question_format=str(question_format),
    )


__all__ = [
    "Bar3DTaskPlan",
    "MaterializedBar3DTask",
    "bar_3d_attempt_seed",
    "bar_3d_task_output_fields",
    "build_bar_3d_plan",
    "materialize_bar_3d_plan",
    "materialize_bar_3d_plan_with_retries",
    "run_bar_3d_public_task",
]
