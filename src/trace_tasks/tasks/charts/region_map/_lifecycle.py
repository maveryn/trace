"""Private neutral lifecycle for region-map chart tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.region_map.shared.annotations import RegionMapAnnotationBundle
from trace_tasks.tasks.charts.region_map.shared.defaults import SCENE_ID
from trace_tasks.tasks.charts.region_map.shared.annotations import MarkerMapAnnotationBundle
from trace_tasks.tasks.charts.region_map.shared.output import base_scene_params, build_trace_scaffold
from trace_tasks.tasks.charts.region_map.shared.output import base_marker_query_params, build_marker_trace_scaffold
from trace_tasks.tasks.charts.region_map.shared.prompts import build_prompt_artifacts, dynamic_slots
from trace_tasks.tasks.charts.region_map.shared.rendering import (
    MarkerMapRenderResult,
    RegionMapRenderResult,
    render_region_map,
    render_region_marker_layer,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


@dataclass(frozen=True)
class RegionMapBoundObjective:
    """Task-owned answer, annotation, and verifier relations after rendering."""

    answer_gt: TypedValue
    annotation: RegionMapAnnotationBundle
    relations: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]


@dataclass(frozen=True)
class MarkerMapBoundObjective:
    """Task-owned answer, annotation, and verifier relations for marker-layer map tasks."""

    answer_gt: TypedValue
    annotation: MarkerMapAnnotationBundle
    relations: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]


def region_map_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for region-map scene attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), f"{SCENE_ID}.retry", int(attempt)))
    )


def semantic_axis_probabilities(
    query_probabilities: Mapping[str, float],
    value_by_query_id: Mapping[str, str],
) -> dict[str, float]:
    """Map query-id probabilities onto a semantic axis used by a task sampler."""

    return {
        str(value_by_query_id[str(query_id)]): float(probability)
        for query_id, probability in query_probabilities.items()
    }


def marker_layer_attempt_seed(instance_seed: int, attempt: int) -> int:
    """Return the neutral retry seed for region-map marker-layer attempts."""

    return (
        int(instance_seed)
        if int(attempt) == 0
        else int(hash64(int(instance_seed), f"{SCENE_ID}.retry", int(attempt)))
    )


def run_region_map_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    prompt_query_key: str,
    answer_type: str,
    question_format: str,
    categorical: bool,
    show_region_value_labels: bool,
    build_dataset: Callable[[int, Mapping[str, Any], str, Mapping[str, float]], Mapping[str, Any]],
    bind_objective: Callable[[Mapping[str, Any], RegionMapRenderResult, str], RegionMapBoundObjective],
) -> TaskOutput:
    """Materialize one task-owned region-map objective with neutral retry/render plumbing."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = region_map_attempt_seed(int(instance_seed), int(attempt))
        try:
            dataset = dict(
                build_dataset(
                    int(attempt_seed),
                    dict(task_params),
                    str(selected_query_id),
                    dict(query_probabilities),
                )
            )
            dataset["categorical"] = bool(categorical)
            dataset["show_region_value_labels"] = bool(show_region_value_labels)
            rendered = render_region_map(
                dataset=dataset,
                params=dict(task_params),
                instance_seed=int(attempt_seed),
                categorical=bool(categorical),
                show_region_value_labels=bool(show_region_value_labels),
            )
            prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots(dataset),
                instance_seed=int(attempt_seed),
            )
            bound = bind_objective(dataset, rendered, str(selected_query_id))
            annotation_bundle = bound.annotation
            qparams = base_scene_params(
                dataset=dataset,
                scene_variant=str(dataset["scene_variant"]),
                scene_variant_probabilities=dict(dataset["_scene_variant_probabilities"]),
            )
            qparams["query_id"] = str(selected_query_id)
            qparams["query_id_probabilities"] = dict(query_probabilities)
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query_id),
                params=qparams,
            )
            trace_payload = build_trace_scaffold(
                dataset=dataset,
                rendered=rendered,
                scene_variant=str(dataset["scene_variant"]),
                scene_variant_probabilities=dict(dataset["_scene_variant_probabilities"]),
                query_spec=query_spec,
                question_format=str(question_format),
                answer_value=bound.answer_gt.value,
                answer_type=str(answer_type),
                annotation_type=str(annotation_bundle.annotation_type),
                annotation_region_ids=annotation_bundle.annotation_region_ids,
                projected_annotation=annotation_bundle.projected_annotation,
                relations=bound.relations,
                witness_symbolic=bound.witness_symbolic,
                annotation_refs=annotation_bundle.annotation_refs,
                show_region_value_labels=bool(show_region_value_labels),
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=bound.answer_gt,
                annotation_gt=annotation_bundle.annotation_gt,
                image=rendered.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_query_id),
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate region-map task: {last_error}") from last_error


def run_marker_layer_lifecycle(
    *,
    task: Any,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    default_query_id: str,
    prompt_query_key: str,
    answer_type: str,
    question_format: str,
    build_dataset: Callable[[int, Mapping[str, Any], str, Mapping[str, float]], Mapping[str, Any]],
    bind_objective: Callable[[Mapping[str, Any], MarkerMapRenderResult, str], MarkerMapBoundObjective],
) -> TaskOutput:
    """Materialize one task-owned region-map marker objective with neutral plumbing."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=task.supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task.task_id),
    )
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        attempt_seed = marker_layer_attempt_seed(int(instance_seed), int(attempt))
        try:
            dataset = build_dataset(
                int(attempt_seed),
                dict(task_params),
                str(selected_query_id),
                dict(query_probabilities),
            )
            rendered = render_region_marker_layer(
                dataset=dataset,
                params=dict(task_params),
                instance_seed=int(attempt_seed),
            )
            prompt_artifacts = build_prompt_artifacts(
                prompt_query_key=str(prompt_query_key),
                dynamic_slot_values=dynamic_slots(dataset),
                instance_seed=int(attempt_seed),
            )
            bound = bind_objective(dataset, rendered, str(selected_query_id))
            annotation_bundle = bound.annotation
            qparams = base_marker_query_params(
                dataset=dataset,
                scene_variant=str(dataset["scene_variant"]),
                scene_variant_probabilities=dict(dataset["_scene_variant_probabilities"]),
            )
            qparams["query_id"] = str(selected_query_id)
            qparams["query_id_probabilities"] = dict(query_probabilities)
            query_spec = build_prompt_query_spec(
                prompt_artifacts=prompt_artifacts,
                query_id=str(selected_query_id),
                params=qparams,
            )
            trace_payload = build_marker_trace_scaffold(
                dataset=dataset,
                rendered=rendered,
                scene_variant=str(dataset["scene_variant"]),
                scene_variant_probabilities=dict(dataset["_scene_variant_probabilities"]),
                query_spec=query_spec,
                question_format=str(question_format),
                answer_value=bound.answer_gt.value,
                answer_type=str(answer_type),
                annotation_type=str(annotation_bundle.annotation_type),
                annotation_region_ids=annotation_bundle.annotation_region_ids,
                projected_annotation=annotation_bundle.projected_annotation,
                relations=bound.relations,
                witness_symbolic=bound.witness_symbolic,
                annotation_refs=annotation_bundle.annotation_refs,
            )
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=bound.answer_gt,
                annotation_gt=annotation_bundle.annotation_gt,
                image=rendered.image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(selected_query_id),
            )
        except ValueError as exc:
            last_error = exc
    raise RuntimeError(f"failed to generate region-map marker task: {last_error}") from last_error


__all__ = [
    "MarkerMapBoundObjective",
    "RegionMapBoundObjective",
    "marker_layer_attempt_seed",
    "region_map_attempt_seed",
    "run_marker_layer_lifecycle",
    "run_region_map_lifecycle",
    "semantic_axis_probabilities",
]
