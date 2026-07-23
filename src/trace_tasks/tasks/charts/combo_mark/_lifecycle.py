"""Private neutral materialization helpers for combo-mark chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.combo_mark.shared.annotations import combo_annotation_artifacts
from trace_tasks.tasks.charts.combo_mark.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.combo_mark.shared.output import build_trace_scaffold
from trace_tasks.tasks.charts.combo_mark.shared.rendering import render_dataset
from trace_tasks.tasks.charts.combo_mark.shared.state import ComboDataset
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec


@dataclass(frozen=True)
class ComboTaskPlan:
    """Task-owned semantic plan consumed by neutral combo-scene rendering."""

    dataset: ComboDataset
    dataset_trace: Mapping[str, Any]
    answer_gt: TypedValue
    answer_value: Any
    answer_type: str
    question_format: str
    annotation_indices: tuple[int, ...]
    annotation_mode: str
    annotation_mark_role: str
    relations: Mapping[str, Any]
    prompt_artifacts: PromptTraceArtifacts


@dataclass(frozen=True)
class MaterializedComboTask:
    """Rendered payload assembled from one public task's semantic plan."""

    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    query_id: str
    prompt_variants: dict[str, Any]


PlanBuilder = Callable[[int, Mapping[str, Any], str], ComboTaskPlan]


def make_combo_plan(
    *,
    dataset: ComboDataset,
    dataset_trace: Mapping[str, Any],
    answer_type: str,
    answer_value: Any,
    question_format: str,
    annotation_indices: tuple[int, ...],
    annotation_mode: str,
    annotation_mark_role: str = "",
    relations: Mapping[str, Any],
    prompt_artifacts: PromptTraceArtifacts,
) -> ComboTaskPlan:
    """Wrap task-bound semantics in the shared combo plan container."""

    return ComboTaskPlan(
        dataset=dataset,
        dataset_trace=dataset_trace,
        answer_gt=TypedValue(type=str(answer_type), value=answer_value),
        answer_value=answer_value,
        answer_type=str(answer_type),
        question_format=str(question_format),
        annotation_indices=tuple(int(idx) for idx in annotation_indices),
        annotation_mode=str(annotation_mode),
        annotation_mark_role=str(annotation_mark_role),
        relations=dict(relations),
        prompt_artifacts=prompt_artifacts,
    )


def make_combo_label_plan(
    *,
    dataset: ComboDataset,
    dataset_trace: Mapping[str, Any],
    answer_label: str,
    question_format: str,
    annotation_index: int,
    relations: Mapping[str, Any],
    prompt_artifacts: PromptTraceArtifacts,
) -> ComboTaskPlan:
    """Package a task-selected category-label answer and its paired marks."""

    return make_combo_plan(
        dataset=dataset,
        dataset_trace=dataset_trace,
        answer_type="string",
        answer_value=str(answer_label),
        question_format=str(question_format),
        annotation_indices=(int(annotation_index),),
        annotation_mode="paired_mark_map",
        relations=relations,
        prompt_artifacts=prompt_artifacts,
    )


def combo_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for combo-scene generation attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), "charts.combo_mark.retry", int(attempt)))
    )


def run_combo_public_task(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    selected_query_id: str,
    failure_label: str,
    build_plan: PlanBuilder,
    build_output: Callable[[MaterializedComboTask], Any] | None = None,
) -> Any:
    """Materialize a public-file-owned plan and optionally call its output factory."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = combo_attempt_seed(int(instance_seed), int(attempt))
        try:
            plan = build_plan(int(attempt_seed), dict(params), str(selected_query_id))
            materialized = materialize_combo_plan(
                instance_seed=int(attempt_seed),
                params=params,
                selected_query_id=str(selected_query_id),
                plan=plan,
            )
            return materialized if build_output is None else build_output(materialized)
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate {failure_label}: {last_error}")


def combo_task_output_fields(materialized: MaterializedComboTask) -> dict[str, Any]:
    """Return the common TaskOutput kwargs for public combo task files."""

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


def build_combo_task_output(materialized: MaterializedComboTask) -> TaskOutput:
    """Return the common TaskOutput for a materialized combo task."""

    return TaskOutput(**combo_task_output_fields(materialized))


def materialize_combo_plan(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    selected_query_id: str,
    plan: ComboTaskPlan,
) -> MaterializedComboTask:
    """Render one task-owned plan and build trace payload sections.

    Public task files own objective sampling, answer calculation, annotation
    index binding, prompt slots, retry handling, and final TaskOutput
    construction. This helper only projects those task-owned semantics into
    rendered combo-chart geometry and shared trace sections.
    """

    artifacts = render_dataset(dataset=plan.dataset, params=params, instance_seed=int(instance_seed))
    annotation, annotation_labels = combo_annotation_artifacts(
        artifacts.scene,
        indices=plan.annotation_indices,
        mode=str(plan.annotation_mode),
        mark_role=str(plan.annotation_mark_role),
    )
    relations = {
        **dict(plan.dataset_trace),
        **dict(plan.relations),
        "question_format": str(plan.question_format),
        "answer": plan.answer_value,
        "answer_type": str(plan.answer_type),
        "labels": list(plan.dataset.labels),
        "primary_name": str(plan.dataset.primary_name),
        "line_name": str(plan.dataset.line_name),
        "primary_values": [int(value) for value in plan.dataset.primary_values],
        "line_values": [int(value) for value in plan.dataset.line_values],
        "annotation_labels": annotation_labels,
    }
    trace_payload = build_trace_scaffold(
        artifacts=artifacts,
        annotation=annotation,
        relations=relations,
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(selected_query_id)
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=plan.prompt_artifacts,
        query_id=str(selected_query_id),
        params={"query_id": str(selected_query_id), **relations},
    )
    trace_payload["execution_trace"]["query_id"] = str(selected_query_id)
    return MaterializedComboTask(
        prompt=str(plan.prompt_artifacts.prompt),
        answer_gt=plan.answer_gt,
        annotation_gt=annotation.annotation_gt,
        image=artifacts.scene.image,
        trace_payload=trace_payload,
        query_id=str(selected_query_id),
        prompt_variants=dict(plan.prompt_artifacts.prompt_variants),
    )


__all__ = [
    "ComboTaskPlan",
    "MaterializedComboTask",
    "build_combo_task_output",
    "combo_task_output_fields",
    "combo_attempt_seed",
    "make_combo_label_plan",
    "make_combo_plan",
    "materialize_combo_plan",
    "run_combo_public_task",
]
