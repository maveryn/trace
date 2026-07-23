"""Scene-private output assembly for coordinate-plane public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.geometry.shared.option_count import resolve_geometry_option_count

from .shared.algebra import (
    DEFAULT_LABEL_POOL as ALGEBRA_LABEL_POOL,
    _resolve_label_pool as resolve_algebra_label_pool,
    _select_winner_label as select_algebra_winner_label,
    _split_defaults_for_task as split_algebra_defaults,
    build_algebra_artifacts,
)
from .shared.construction import (
    DEFAULT_LABEL_POOL as LOCUS_LABEL_POOL,
    _resolve_label_pool as resolve_locus_label_pool,
    _select_winner_label as select_locus_winner_label,
    _split_defaults_for_task as split_locus_defaults,
    build_locus_panel_artifacts,
    build_locus_point_artifacts,
)
from .shared.rendering import build_relation_artifacts, relation_query


SCENE_ID = "coordinate_plane"


@dataclass(frozen=True)
class CoordinateRelationObjective:
    """Task-owned relation semantics after public query selection."""

    scene_variant: str
    semantic_operation: str
    prompt_query_key: str


@dataclass(frozen=True)
class CoordinateAlgebraObjective:
    """Task-owned coordinate algebra semantics after public query selection."""

    semantic_operation: str
    prompt_query_key: str
    scene_key: str


@dataclass(frozen=True)
class CoordinateLocusObjective:
    """Task-owned coordinate locus semantics after public query selection."""

    semantic_operation: str
    prompt_query_key: str


def _select_public_query(
    task: Any,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[str, Mapping[str, float], dict[str, Any]]:
    """Resolve the public query id through the global fixed-query helper."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(tuple(task.supported_query_ids)[0]),
        task_id=str(task.task_id),
    )
    return str(selected_query), dict(query_probabilities), dict(task_params)


def _attach_public_query_trace(
    trace_payload: Mapping[str, Any],
    *,
    selected_query: str,
    query_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Attach public query metadata after identity-free scene artifacts return."""

    payload = dict(trace_payload)
    probabilities = {str(key): float(value) for key, value in query_probabilities.items()}
    query_spec = dict(payload.get("query_spec", {}))
    query_params = dict(query_spec.get("params", {}))
    query_spec["query_id"] = str(selected_query)
    query_params["query_id"] = str(selected_query)
    query_params["query_id_probabilities"] = dict(probabilities)
    query_spec["params"] = query_params
    payload["query_spec"] = query_spec

    execution_trace = dict(payload.get("execution_trace", {}))
    execution_trace["query_id"] = str(selected_query)
    execution_trace["query_id_probabilities"] = dict(probabilities)
    payload["execution_trace"] = execution_trace

    scene_ir = dict(payload.get("scene_ir", {}))
    relations = dict(scene_ir.get("relations", {}))
    relations["query_id"] = str(selected_query)
    scene_ir["relations"] = relations
    payload["scene_ir"] = scene_ir
    return payload


def _attach_output_operation_trace(
    trace_payload: Mapping[str, Any],
    *,
    output_operation_key: str,
    output_operation_probabilities: Mapping[str, float],
    semantic_operation_key: str,
) -> dict[str, Any]:
    """Record public output operation separately from scene semantic operation."""

    payload = dict(trace_payload)
    probabilities = {str(key): float(value) for key, value in output_operation_probabilities.items()}

    query_spec = dict(payload.get("query_spec", {}))
    query_params = dict(query_spec.get("params", {}))
    query_spec["operation_key"] = str(output_operation_key)
    query_params["operation_key"] = str(output_operation_key)
    query_params["operation_key_probabilities"] = dict(probabilities)
    query_params["semantic_operation"] = str(semantic_operation_key)
    query_spec["params"] = query_params
    payload["query_spec"] = query_spec

    execution_trace = dict(payload.get("execution_trace", {}))
    execution_trace["operation_key"] = str(output_operation_key)
    execution_trace["operation_key_probabilities"] = dict(probabilities)
    execution_trace["semantic_operation"] = str(semantic_operation_key)
    payload["execution_trace"] = execution_trace

    scene_ir = dict(payload.get("scene_ir", {}))
    relations = dict(scene_ir.get("relations", {}))
    relations["operation_key"] = str(output_operation_key)
    relations["semantic_operation"] = str(semantic_operation_key)
    scene_ir["relations"] = relations
    payload["scene_ir"] = scene_ir
    return payload


def run_coordinate_relation_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Generate one relation-count instance from public task-owned semantics."""

    selected_query, query_probabilities, task_params = _select_public_query(
        task,
        instance_seed=int(instance_seed),
        params=params,
    )
    objective = task.prepare_objective(  # type: ignore[attr-defined]
        selected_query,
        query_probabilities,
        task_params,
    )
    semantic_probabilities = {str(objective.semantic_operation): 1.0}
    query = relation_query(
        instance_seed=int(instance_seed),
        params=task_params,
        operation_key=str(objective.semantic_operation),
        operation_key_probabilities=semantic_probabilities,
        scene_variant=str(objective.scene_variant),
        scene_variant_probabilities={str(objective.scene_variant): 1.0},
    )
    artifacts = build_relation_artifacts(
        instance_seed=int(instance_seed),
        params=task_params,
        query=query,
        output_operation_key=str(selected_query),
        output_query_probabilities=query_probabilities,
        prompt_query_key=str(objective.prompt_query_key),
        max_attempts=int(max_attempts),
    )
    trace_payload = _attach_public_query_trace(
        artifacts.trace_payload,
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
    )
    return TaskOutput(
        prompt=str(artifacts.prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=artifacts.rendered_scene.answer_value),
        annotation_gt=TypedValue(
            type=str(artifacts.rendered_scene.annotation_type),
            value=artifacts.rendered_scene.annotation_value,
        ),
        image=artifacts.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=artifacts.task_versions,
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
    )


def _resolve_label_choice(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    label_key: str,
    option_count_field: str,
    label_pool: Sequence[str],
    instance_seed: int,
    namespace: str,
    resolver,
    selector,
) -> tuple[tuple[str, ...], str, Mapping[str, float]]:
    """Resolve option labels and the answer label for one candidate task."""

    labels = resolver(params, generation_defaults, str(label_key), tuple(label_pool))
    option_count, _ = resolve_geometry_option_count(
        params=params,
        gen_defaults=generation_defaults,
        field_name=str(option_count_field),
        supported_counts=(4, 6),
        task_id=str(namespace),
        instance_seed=int(instance_seed),
    )
    if int(option_count) > len(labels):
        raise ValueError(f"{option_count_field} cannot exceed label pool length")
    labels = tuple(str(label) for label in labels[: int(option_count)])
    winner_label, winner_probabilities = selector(
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        label_pool=labels,
    )
    return tuple(str(label) for label in labels), str(winner_label), dict(winner_probabilities)


def run_coordinate_algebra_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Generate one coordinate algebra option-selection instance."""

    del max_attempts
    selected_query, query_probabilities, task_params = _select_public_query(
        task,
        instance_seed=int(instance_seed),
        params=params,
    )
    objective = task.prepare_objective(  # type: ignore[attr-defined]
        selected_query,
        query_probabilities,
        task_params,
    )
    generation_defaults, _, _ = split_algebra_defaults(str(task.task_id))
    label_pool, winner_label, winner_probabilities = _resolve_label_choice(
        params=task_params,
        generation_defaults=generation_defaults,
        label_key="algebra_candidate_labels",
        option_count_field="algebra_candidate_count",
        label_pool=ALGEBRA_LABEL_POOL,
        instance_seed=int(instance_seed),
        namespace=str(task.task_id),
        resolver=resolve_algebra_label_pool,
        selector=select_algebra_winner_label,
    )
    artifacts = build_algebra_artifacts(
        namespace=str(task.task_id),
        config_key=str(task.task_id),
        semantic_operation_key=str(objective.semantic_operation),
        semantic_query_probabilities={str(objective.semantic_operation): 1.0},
        prompt_query_key=str(objective.prompt_query_key),
        winner_label=str(winner_label),
        winner_label_probabilities=winner_probabilities,
        label_pool=label_pool,
        scene_key=str(objective.scene_key),
        instance_seed=int(instance_seed),
        params=task_params,
    )
    trace_payload = _attach_output_operation_trace(
        artifacts.trace_payload,
        output_operation_key=str(selected_query),
        output_operation_probabilities=query_probabilities,
        semantic_operation_key=str(objective.semantic_operation),
    )
    trace_payload = _attach_public_query_trace(
        trace_payload,
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
    )
    trace_payload["witness_symbolic"] = {
        **dict(trace_payload.get("witness_symbolic", {})),
        "task_id": str(task.task_id),
    }
    return TaskOutput(
        prompt=str(artifacts.prompt_artifacts.prompt),
        answer_gt=TypedValue(type="option_letter", value=str(artifacts.query.winner_label)),
        annotation_gt=TypedValue(type="point", value=artifacts.annotation_value),
        image=artifacts.rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
    )


def run_coordinate_locus_point_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Generate one shaded-locus candidate-point option instance."""

    del max_attempts
    selected_query, query_probabilities, task_params = _select_public_query(
        task,
        instance_seed=int(instance_seed),
        params=params,
    )
    objective = task.prepare_objective(  # type: ignore[attr-defined]
        selected_query,
        query_probabilities,
        task_params,
    )
    generation_defaults, _, _ = split_locus_defaults(str(task.task_id))
    label_pool, winner_label, winner_probabilities = _resolve_label_choice(
        params=task_params,
        generation_defaults=generation_defaults,
        label_key="locus_candidate_labels",
        option_count_field="locus_candidate_count",
        label_pool=LOCUS_LABEL_POOL,
        instance_seed=int(instance_seed),
        namespace=str(task.task_id),
        resolver=resolve_locus_label_pool,
        selector=select_locus_winner_label,
    )
    artifacts = build_locus_point_artifacts(
        namespace=str(task.task_id),
        config_key=str(task.task_id),
        semantic_operation_key=str(objective.semantic_operation),
        semantic_query_probabilities={str(objective.semantic_operation): 1.0},
        prompt_query_key=str(objective.prompt_query_key),
        winner_label=str(winner_label),
        winner_label_probabilities=winner_probabilities,
        label_pool=label_pool,
        instance_seed=int(instance_seed),
        params=task_params,
    )
    trace_payload = _attach_output_operation_trace(
        artifacts.trace_payload,
        output_operation_key=str(selected_query),
        output_operation_probabilities=query_probabilities,
        semantic_operation_key=str(objective.semantic_operation),
    )
    trace_payload = _attach_public_query_trace(
        trace_payload,
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
    )
    trace_payload["witness_symbolic"] = {
        **dict(trace_payload.get("witness_symbolic", {})),
        "task_id": str(task.task_id),
    }
    return TaskOutput(
        prompt=str(artifacts.prompt_artifacts.prompt),
        answer_gt=TypedValue(type="option_letter", value=str(artifacts.query.winner_label)),
        annotation_gt=TypedValue(type="point", value=artifacts.annotation_value),
        image=artifacts.rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
    )


def run_coordinate_locus_panel_entry(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Generate one coordinate-locus panel-matching option instance."""

    del max_attempts
    selected_query, query_probabilities, task_params = _select_public_query(
        task,
        instance_seed=int(instance_seed),
        params=params,
    )
    objective = task.prepare_objective(  # type: ignore[attr-defined]
        selected_query,
        query_probabilities,
        task_params,
    )
    generation_defaults, _, _ = split_locus_defaults(str(task.task_id))
    label_pool, winner_label, winner_probabilities = _resolve_label_choice(
        params=task_params,
        generation_defaults=generation_defaults,
        label_key="locus_panel_labels",
        option_count_field="locus_panel_count",
        label_pool=LOCUS_LABEL_POOL,
        instance_seed=int(instance_seed),
        namespace=str(task.task_id),
        resolver=resolve_locus_label_pool,
        selector=select_locus_winner_label,
    )
    artifacts = build_locus_panel_artifacts(
        namespace=str(task.task_id),
        config_key=str(task.task_id),
        semantic_operation_key=str(objective.semantic_operation),
        semantic_query_probabilities={str(objective.semantic_operation): 1.0},
        prompt_query_key=str(objective.prompt_query_key),
        winner_label=str(winner_label),
        winner_label_probabilities=winner_probabilities,
        label_pool=label_pool,
        instance_seed=int(instance_seed),
        params=task_params,
    )
    trace_payload = _attach_output_operation_trace(
        artifacts.trace_payload,
        output_operation_key=str(selected_query),
        output_operation_probabilities=query_probabilities,
        semantic_operation_key=str(objective.semantic_operation),
    )
    trace_payload = _attach_public_query_trace(
        trace_payload,
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
    )
    trace_payload["witness_symbolic"] = {
        **dict(trace_payload.get("witness_symbolic", {})),
        "task_id": str(task.task_id),
    }
    return TaskOutput(
        prompt=str(artifacts.prompt_artifacts.prompt),
        answer_gt=TypedValue(type="option_letter", value=str(artifacts.query.winner_label)),
        annotation_gt=TypedValue(type="bbox", value=artifacts.annotation_value),
        image=artifacts.rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
    )


__all__ = [
    "CoordinateAlgebraObjective",
    "CoordinateLocusObjective",
    "CoordinateRelationObjective",
    "run_coordinate_algebra_entry",
    "run_coordinate_locus_panel_entry",
    "run_coordinate_locus_point_entry",
    "run_coordinate_relation_entry",
]
