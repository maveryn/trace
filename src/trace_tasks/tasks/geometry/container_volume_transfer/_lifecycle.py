"""Neutral lifecycle plumbing for container volume-transfer tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.measurements import json_answer_value
from .shared.output import container_volume_trace_payload, prepare_container_volume_artifacts
from .shared.relations import ContainerVolumeTaskBinding, container_measurement_fields
from .shared.state import ResolvedProblem
from .shared.defaults import SCENE_ID


@dataclass(frozen=True)
class ContainerVolumeTaskParts:
    """Rendered output fields prepared after a public task binds semantics."""

    prompt: str
    prompt_variants: dict[str, str]
    image: Image.Image
    annotation_value: dict[str, list[float]]
    trace_payload: dict[str, Any]
    task_versions: dict[str, str]
    scene_id: str


def prepare_container_volume_task_parts(
    *,
    selected_query: str,
    prompt_query_key: str,
    query_probabilities: Mapping[str, float],
    problem: ResolvedProblem,
    binding: ContainerVolumeTaskBinding,
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    random_namespace: str,
) -> ContainerVolumeTaskParts:
    """Prepare shared render, prompt, and trace plumbing without routing."""

    artifacts = prepare_container_volume_artifacts(
        prompt_query_key=str(prompt_query_key),
        problem=problem,
        binding=binding,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        random_namespace=str(random_namespace),
    )
    measurement_fields = container_measurement_fields(problem)
    query_params = {
        "scene_id": SCENE_ID,
        "query_id": str(selected_query),
        "internal_query_id": str(prompt_query_key),
        "query_id_probabilities": dict(query_probabilities),
        "case_probabilities": dict(problem.case_probabilities),
        "answer_support_probabilities": dict(problem.answer_support_probabilities),
        **measurement_fields,
    }
    query_spec = build_prompt_query_spec(
        prompt_artifacts=artifacts.prompt_artifacts,
        query_id=str(selected_query),
        params=query_params,
    )
    query_spec["scene_id"] = SCENE_ID
    annotation_roles = list(binding.annotation_keys)
    trace_payload = container_volume_trace_payload(
        artifacts=artifacts,
        query_spec=query_spec,
        relations={
            "type": str(problem.formula_family),
            "source_shape": str(problem.source_shape),
            "target_shape": str(problem.target_shape),
            "source_volume": int(problem.source_volume),
            "target_volume": int(problem.target_volume),
            "pour_count": int(problem.pour_count),
            "annotation_roles": annotation_roles,
        },
        execution_trace={
            "query_id": str(selected_query),
            "internal_query_id": str(prompt_query_key),
            "formula_family": str(problem.formula_family),
            "formula": str(problem.formula),
            "answer": json_answer_value(problem.answer),
            "annotation_roles": annotation_roles,
            **query_params,
        },
        witness_symbolic=dict(query_params),
    )
    return ContainerVolumeTaskParts(
        prompt=artifacts.prompt,
        prompt_variants=dict(artifacts.prompt_variants),
        image=artifacts.image,
        annotation_value=dict(artifacts.annotation_value),
        trace_payload=trace_payload,
        task_versions=artifacts.task_versions,
        scene_id=artifacts.scene_id,
    )

def build_container_volume_result(
    selected_query: str,
    prompt_query_key: str,
    query_probabilities: Mapping[str, float],
    problem: ResolvedProblem,
    binding: ContainerVolumeTaskBinding,
    answer_value: int | float,
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    random_namespace: str,
) -> TaskOutput:
    """Create the final output from semantics selected by a public task."""

    parts = prepare_container_volume_task_parts(
        selected_query=str(selected_query),
        prompt_query_key=str(prompt_query_key),
        query_probabilities=query_probabilities,
        problem=problem,
        binding=binding,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        random_namespace=str(random_namespace),
    )
    return TaskOutput(
        parts.prompt,
        TypedValue(type=binding.answer_type, value=answer_value),
        TypedValue(type="bbox_map", value=dict(parts.annotation_value)),
        parts.image,
        "img0",
        parts.trace_payload,
        parts.task_versions,
        parts.scene_id,
        str(selected_query),
        dict(parts.prompt_variants),
    )


__all__ = [
    "ContainerVolumeTaskParts",
    "build_container_volume_result",
    "prepare_container_volume_task_parts",
]
