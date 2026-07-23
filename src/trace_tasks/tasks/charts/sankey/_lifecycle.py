"""Private neutral lifecycle helpers for standard Sankey public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts, build_prompt_query_spec

from .shared.annotations import annotation_payload
from .shared.output import render_map, render_spec, scene_records
from .shared.prompts import build_prompt_artifacts, dynamic_slots
from .shared.rendering import render_sankey_dataset
from .shared.state import SCENE_ID, SankeyDataset, SankeyRenderResult


@dataclass(frozen=True)
class SankeyTaskPlan:
    dataset: SankeyDataset
    prompt_key: str
    question_format: str
    witness_type: str
    trace_params: dict[str, Any]


def attempt_seed(instance_seed: int, unique_key: str, attempt: int) -> int:
    return int(instance_seed) if int(attempt) == 0 else int(hash64(int(instance_seed), str(unique_key), int(attempt)))


def build_sankey_plan(
    *,
    dataset: SankeyDataset,
    prompt_key: str,
    question_format: str,
    witness_type: str,
    trace_params: Mapping[str, Any] | None = None,
) -> SankeyTaskPlan:
    return SankeyTaskPlan(
        dataset=dataset,
        prompt_key=str(prompt_key),
        question_format=str(question_format),
        witness_type=str(witness_type),
        trace_params=dict(trace_params or {}),
    )


def _trace_payload(
    *,
    plan: SankeyTaskPlan,
    rendered: SankeyRenderResult,
    prompt_artifacts: PromptTraceArtifacts,
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble verifier trace records after task-owned objective binding and rendering."""

    dataset = plan.dataset
    frame = dataset.frame
    question = dataset.question
    rendered_scene = rendered.rendered_scene
    records = scene_records(dataset)
    annotation_refs = [str(segment_id) for segment_id in annotation["segment_refs"]]
    question_params = {
        "query_id": str(question.branch_id),
        "query_id_probabilities": dict(question.branch_probabilities),
        "scene_variant": str(frame.scene_variant),
        "scene_variant_probabilities": dict(frame.scene_probabilities),
        "annotation_segment_ids": list(annotation_refs),
        **dict(question.params),
        **dict(plan.trace_params),
    }
    path_lookup = {str(path["path_id"]): dict(path) for path in records["paths"]}
    path_details = [path_lookup[str(path_id)] for path_id in question.params.get("query_path_ids", [])]
    return {
        "scene_ir": {
            "scene_kind": "chart_sankey",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(question.branch_id),
                "scene_variant": str(frame.scene_variant),
                "answer_value": int(question.answer),
                "annotation_type": str(annotation["type"]),
                "annotation_segment_ids": list(annotation_refs),
                "query_path_ids": list(question.params.get("query_path_ids", [])),
            },
        },
        "query_spec": build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(question.branch_id),
            params=dict(question_params),
        ),
        "render_spec": render_spec(dataset, rendered),
        "render_map": render_map(rendered),
        "execution_trace": {
            "query_id": str(question.branch_id),
            "scene_variant": str(frame.scene_variant),
            "query_id_probabilities": dict(question.branch_probabilities),
            "scene_variant_probabilities": dict(frame.scene_probabilities),
            "question_format": str(plan.question_format),
            "answer_value": int(question.answer),
            "answer_type": str(question.answer_type),
            "annotation_type": str(annotation["type"]),
            "annotation_segment_ids": list(annotation_refs),
            "query_path_details": list(path_details),
            **dict(records),
            **dict(question.params),
            **dict(plan.trace_params),
        },
        "witness_symbolic": {
            "type": str(plan.witness_type),
            "answer_value": int(question.answer),
            "annotation_type": str(annotation["type"]),
            "annotation_segment_ids": list(annotation_refs),
            **dict(question.params),
        },
        "projected_annotation": dict(annotation["projected_annotation"]),
        "background": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
    }


def materialize_sankey_plan(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    plan: SankeyTaskPlan,
) -> TaskOutput:
    rendered = render_sankey_dataset(dataset=plan.dataset, params=dict(params), instance_seed=int(instance_seed))
    annotation = annotation_payload(
        rendered=rendered,
        segment_refs=plan.dataset.question.annotation_segment_ids,
        annotation_type=str(plan.dataset.question.annotation_type),
    )
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
        answer_gt=TypedValue(type=str(plan.dataset.question.answer_type), value=int(plan.dataset.question.answer)),
        annotation_gt=TypedValue(type=str(annotation["type"]), value=annotation["value"]),
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(plan.dataset.question.branch_id),
    )


def run_sankey_task(task: Any, instance_seed: int, params: dict[str, Any], max_attempts: int) -> TaskOutput:
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
            plan = task._build_plan(
                dict(task_params),
                int(current_seed),
                str(selected),
                dict(probabilities),
            )
            return materialize_sankey_plan(
                params=dict(task_params),
                instance_seed=int(current_seed),
                plan=plan,
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate Sankey task: {last_error}") from last_error


__all__ = ["SankeyTaskPlan", "build_sankey_plan", "materialize_sankey_plan", "run_sankey_task"]
