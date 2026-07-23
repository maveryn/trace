"""Private neutral lifecycle helpers for scatter-cluster public tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import ScatterClusterAnnotationBundle
from .shared.output import base_execution_record, render_map, render_spec
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import render_scatter_cluster_dataset
from .shared.state import SCENE_ID, ScatterClusterDataset, ScatterClusterInputs, ScatterClusterRenderResult


AnnotationBuilder = Callable[[ScatterClusterDataset, ScatterClusterRenderResult], ScatterClusterAnnotationBundle]


@dataclass(frozen=True)
class ScatterClusterTaskPlan:
    dataset: ScatterClusterDataset
    inputs: ScatterClusterInputs
    prompt_key: str
    question_format: str
    witness_type: str
    annotation_builder: AnnotationBuilder
    trace_params: dict[str, Any]


def attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def build_scatter_cluster_plan(
    *,
    dataset: ScatterClusterDataset,
    inputs: ScatterClusterInputs,
    prompt_key: str,
    question_format: str,
    witness_type: str,
    annotation_builder: AnnotationBuilder,
    trace_params: Mapping[str, Any] | None = None,
) -> ScatterClusterTaskPlan:
    return ScatterClusterTaskPlan(
        dataset=dataset,
        inputs=inputs,
        prompt_key=str(prompt_key),
        question_format=str(question_format),
        witness_type=str(witness_type),
        annotation_builder=annotation_builder,
        trace_params=dict(trace_params or {}),
    )


def _trace_payload(
    *,
    plan: ScatterClusterTaskPlan,
    rendered: ScatterClusterRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    annotation: ScatterClusterAnnotationBundle,
) -> dict[str, Any]:
    """Assemble verifier records after task-owned answer and annotation binding."""

    dataset = plan.dataset
    question = dataset.question
    rendered_scene = rendered.rendered_scene
    option_labels = [str(marker.option_label) for marker in dataset.option_markers]
    params = {
        "query_id": str(question.branch_id),
        "query_id_probabilities": dict(question.branch_probabilities),
        "scene_variant": str(dataset.scene_variant),
        "scene_variant_probabilities": {str(dataset.scene_variant): 1.0},
        "annotation_refs": list(annotation.annotation_refs),
        **dict(question.params),
        **dict(plan.trace_params),
    }
    execution = {
        **base_execution_record(dataset=dataset, inputs=plan.inputs),
        "query_id": str(question.branch_id),
        "query_id_probabilities": dict(question.branch_probabilities),
        "question_format": str(plan.question_format),
        "annotation_refs": list(annotation.annotation_refs),
        "annotation_point_ids": list(annotation.annotation_point_ids),
        "annotation_cluster_labels": list(annotation.annotation_cluster_labels),
    }
    spec = render_spec(rendered)
    spec["scene_variant"] = str(dataset.scene_variant)
    return {
        "scene_ir": {
            "scene_kind": "chart_scatter_cluster",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(question.branch_id),
                "scene_variant": str(dataset.scene_variant),
                "answer": str(question.answer),
                "annotation_type": str(annotation.annotation_type),
                "annotation_refs": list(annotation.annotation_refs),
                "annotation_point_ids": list(annotation.annotation_point_ids),
                "annotation_cluster_labels": list(annotation.annotation_cluster_labels),
                "option_labels": list(option_labels),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(question.branch_id),
            params=params,
        ),
        "render_spec": dict(spec),
        "render_map": render_map(rendered),
        "execution_trace": dict(execution),
        "witness_symbolic": {
            "type": str(plan.witness_type),
            "answer": str(question.answer),
            "annotation_type": str(annotation.annotation_type),
            "annotation_refs": list(annotation.annotation_refs),
            **dict(question.params),
        },
        "projected_annotation": dict(annotation.projected_annotation),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def materialize_scatter_cluster_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    plan: ScatterClusterTaskPlan,
) -> TaskOutput:
    rendered = render_scatter_cluster_dataset(
        dataset=plan.dataset,
        params=dict(params),
        instance_seed=int(instance_seed),
    )
    annotation = plan.annotation_builder(plan.dataset, rendered)
    prompt_artifacts = build_prompt_artifacts(
        prompt_query_key=str(plan.prompt_key),
        dynamic_slot_values=dynamic_slots(dataset=plan.dataset),
        instance_seed=int(instance_seed),
    )
    trace_payload = _trace_payload(
        plan=plan,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation=annotation,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(plan.dataset.question.answer_type), value=str(plan.dataset.question.answer)),
        annotation_gt=annotation.annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.dataset.question.branch_id),
    )


def run_scatter_cluster_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        current_seed = attempt_seed(int(instance_seed), str(task.task_id), int(attempt))
        try:
            plan = task._build_plan(dict(task_params), int(current_seed), str(selected), dict(probabilities))
            return materialize_scatter_cluster_plan(
                params=dict(task_params),
                instance_seed=int(current_seed),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate scatter-cluster task: {last_error}") from last_error


__all__ = [
    "ScatterClusterTaskPlan",
    "build_scatter_cluster_plan",
    "materialize_scatter_cluster_plan",
    "run_scatter_cluster_task",
]
