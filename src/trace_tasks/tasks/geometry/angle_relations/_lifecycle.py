"""Neutral render/prompt plumbing for angle-relations public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeVar

from trace_tasks.tasks.geometry.angle_relations.shared.annotations import angle_vertex_annotation_artifacts
from trace_tasks.tasks.geometry.angle_relations.shared.output import angle_relation_trace_payload
from trace_tasks.tasks.geometry.angle_relations.shared.prompts import build_angle_relation_prompt_artifacts
from trace_tasks.tasks.geometry.angle_relations.shared.rendering import RenderedAngleRelationContext, render_angle_relation_case
from trace_tasks.tasks.geometry.angle_relations.shared.sampling import select_indexed_case
from trace_tasks.tasks.geometry.angle_relations.shared.state import SCENE_ID, AngleRelationCase
from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec


_CaseT = TypeVar("_CaseT")


@dataclass(frozen=True)
class AngleRelationRuntime:
    """Rendered diagram, prompt artifacts, and annotation for one selected case."""

    rendered_context: RenderedAngleRelationContext
    prompt_artifacts: Any
    annotation_artifacts: PixelAnnotationArtifacts
    case_index: int


def select_angle_relation_case(
    *,
    cases: Sequence[_CaseT],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[_CaseT, int]:
    """Resolve an index for an already task-selected case support."""

    return select_indexed_case(
        cases=cases,
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def render_angle_relation_runtime(
    *,
    case: AngleRelationCase,
    case_index: int,
    prompt_query_key: str,
    prompt_task_key: str | None = None,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    max_attempts: int,
) -> AngleRelationRuntime:
    """Render and annotate one already selected angle-relations case."""

    rendered_context = render_angle_relation_case(
        case=case,
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        max_attempts=int(max_attempts),
    )
    annotation_artifacts = angle_vertex_annotation_artifacts(rendered_context.rendered_scene)
    _prompt_defaults, prompt_artifacts = build_angle_relation_prompt_artifacts(
        prompt_query_key=str(prompt_query_key),
        prompt_task_key=prompt_task_key,
        instance_seed=int(instance_seed),
    )
    return AngleRelationRuntime(
        rendered_context=rendered_context,
        prompt_artifacts=prompt_artifacts,
        annotation_artifacts=annotation_artifacts,
        case_index=int(case_index),
    )


def build_integer_angle_relation_trace(
    *,
    runtime: AngleRelationRuntime,
    branch_name: str,
    branch_probabilities: Mapping[str, float],
    answer_value: int,
    query_params: Mapping[str, Any] | None = None,
    scene_relation_fields: Mapping[str, Any] | None = None,
    execution_fields_extra: Mapping[str, Any] | None = None,
    witness_fields_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge common render trace sections with public task-bound integer fields."""

    rendered_scene = runtime.rendered_context.rendered_scene
    annotation_roles = [str(role) for role in rendered_scene.annotation_roles]
    query_spec = build_prompt_query_spec(
        prompt_artifacts=runtime.prompt_artifacts,
        query_id=str(branch_name),
        params={
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "query_id_probabilities": dict(branch_probabilities),
            "case_index": int(runtime.case_index),
            "case_answer": int(answer_value),
            **dict(query_params or {}),
        },
    )
    query_spec["scene_id"] = SCENE_ID
    return angle_relation_trace_payload(
        rendered_context=runtime.rendered_context,
        annotation_artifacts=runtime.annotation_artifacts,
        query_spec=query_spec,
        scene_relations={
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "answer_value": int(answer_value),
            "annotation_roles": list(annotation_roles),
            **dict(scene_relation_fields or {}),
        },
        execution_fields={
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "query_id_probabilities": dict(branch_probabilities),
            "answer_type": "integer",
            "answer_value": int(answer_value),
            "annotation_roles": list(annotation_roles),
            **dict(execution_fields_extra or {}),
        },
        witness_fields={
            "type": "analytical_measurement_geometry_value",
            "scene_id": SCENE_ID,
            "query_id": str(branch_name),
            "answer_value": int(answer_value),
            "annotation_roles": list(annotation_roles),
            **dict(witness_fields_extra or {}),
        },
    )


__all__ = [
    "AngleRelationRuntime",
    "build_integer_angle_relation_trace",
    "render_angle_relation_runtime",
    "select_angle_relation_case",
]
