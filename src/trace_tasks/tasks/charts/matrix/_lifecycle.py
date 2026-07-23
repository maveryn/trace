"""Private neutral lifecycle for matrix chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.matrix.shared.annotations import MatrixAnnotationBundle
from trace_tasks.tasks.charts.matrix.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.matrix.shared.output import base_prompt_params, build_trace_scaffold
from trace_tasks.tasks.charts.matrix.shared.prompts import build_prompt_artifacts, dynamic_slots
from trace_tasks.tasks.charts.matrix.shared.rendering import MatrixRenderResult, render_matrix_scene
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


@dataclass(frozen=True)
class MatrixBoundObjective:
    """Task-owned answer, annotation, and verifier relations after rendering."""

    answer_gt: TypedValue
    annotation: MatrixAnnotationBundle
    relations: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]


def matrix_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for matrix scene attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), f"{SCENE_ID}.retry", int(attempt)))
    )


def branch_argument_probabilities(
    branch_probabilities: Mapping[str, float],
    branch_args_by_id: Mapping[str, tuple[str, ...]],
    *,
    argument_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    """Aggregate selected-branch probabilities by semantic argument value."""

    totals = {str(name): defaultdict(float) for name in argument_names}
    for branch, probability in branch_probabilities.items():
        values = branch_args_by_id[str(branch)]
        for index, name in enumerate(argument_names):
            totals[str(name)][str(values[int(index)])] += float(probability)
    return {str(name): dict(value) for name, value in totals.items()}


def run_matrix_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    prompt_query_key: str,
    question_format: str,
    supports_unanswerable: bool,
    build_dataset: Callable[[int, Mapping[str, Any], str, Mapping[str, float]], Mapping[str, Any]],
    bind_objective: Callable[[Mapping[str, Any], MatrixRenderResult, str], MatrixBoundObjective],
) -> TaskOutput:
    """Materialize one task-owned matrix objective with neutral scene plumbing."""

    selected_query_id, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = matrix_attempt_seed(int(instance_seed), int(attempt))
        try:
            dataset = build_dataset(
                int(attempt_seed),
                dict(task_params),
                str(selected_query_id),
                dict(branch_probabilities),
            )
            scene_variant = str(dataset["_scene_variant"])
            palette_variant = str(dataset["_palette_variant"])
            header_layout = str(dataset["_header_layout"])
            grid_style = str(dataset["_grid_style"])
            rendered = render_matrix_scene(
                dataset=dataset,
                scene_variant=scene_variant,
                palette_variant=palette_variant,
                header_layout=header_layout,
                grid_style=grid_style,
                params=dict(task_params),
                instance_seed=int(attempt_seed),
            )
            prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots(
                    dataset,
                    scene_variant=scene_variant,
                    supports_unanswerable=bool(supports_unanswerable),
                ),
                instance_seed=int(attempt_seed),
            )
            bound = bind_objective(dataset, rendered, str(selected_query_id))
            annotation = bound.annotation
            qparams = base_prompt_params(
                dataset=dataset,
                scene_variant=scene_variant,
                scene_variant_probabilities=dict(dataset["_scene_variant_probabilities"]),
                palette_variant=palette_variant,
                palette_variant_probabilities=dict(dataset["_palette_variant_probabilities"]),
                header_layout=header_layout,
                header_layout_probabilities=dict(dataset["_header_layout_probabilities"]),
                grid_style=grid_style,
                grid_style_probabilities=dict(dataset["_grid_style_probabilities"]),
            )
            qparams["query_id"] = str(selected_query_id)
            qparams["query_id_probabilities"] = dict(branch_probabilities)
            prompt_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query_id),
                params=qparams,
            )
            trace_payload = build_trace_scaffold(
                dataset=dataset,
                rendered=rendered,
                scene_variant=scene_variant,
                palette_variant=palette_variant,
                header_layout=header_layout,
                grid_style=grid_style,
                prompt_spec=prompt_spec,
                question_format=str(question_format),
                answer_value=bound.answer_gt.value,
                answer_type=str(bound.answer_gt.type),
                annotation_type=str(annotation.annotation_type),
                annotation_cell_ids=annotation.annotation_cell_ids,
                support_header_keys=[str(key) for key in dataset["annotation_header_keys"]],
                projected_annotation=annotation.projected_annotation,
                relations=bound.relations,
                witness_symbolic=bound.witness_symbolic,
                annotation_refs=annotation.annotation_refs,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=bound.answer_gt,
                annotation_gt=annotation.annotation_gt,
                image=rendered.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_query_id),
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate matrix task: {last_error}") from last_error


def run_configured_matrix_lifecycle(task: Any, instance_seed: int, params: Mapping[str, Any], max_attempts: int) -> TaskOutput:
    """Run a public matrix task using task-owned hooks and class metadata."""

    return run_matrix_lifecycle(
        task=task,
        instance_seed=int(instance_seed),
        params=dict(params),
        max_attempts=int(max_attempts),
        default_query_id=str(task.default_query_id),
        prompt_query_key=str(task.prompt_query_key),
        question_format=str(task.question_format),
        supports_unanswerable=bool(getattr(task, "supports_unanswerable", False)),
        build_dataset=task._construct_dataset,
        bind_objective=task._bind_objective,
    )


__all__ = [
    "MatrixBoundObjective",
    "branch_argument_probabilities",
    "matrix_attempt_seed",
    "run_configured_matrix_lifecycle",
    "run_matrix_lifecycle",
]
