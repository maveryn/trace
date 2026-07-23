"""Neutral lifecycle runner for concentric-chord public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

from trace_tasks.tasks.geometry.shared.annotation_values import PixelAnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import SCENE_ID, load_concentric_chord_task_defaults
from .shared.output import (
    ConcentricChordArtifacts,
    concentric_trace_base,
    prepare_concentric_chord_artifacts,
)
from .shared.state import ConcentricChordDiagramSpec


@dataclass(frozen=True)
class ConcentricChordObjectivePlan:
    """Task-selected formula case and trace metadata."""

    spec: ConcentricChordDiagramSpec
    case_index: int
    answer_probabilities: Mapping[str, float]
    prompt_query_key: str
    random_namespace: str


@dataclass(frozen=True)
class ConcentricChordPreparedParts:
    """Prepared fields for a public task to assemble its TaskOutput."""

    prompt: str
    prompt_variants: dict[str, str]
    image: Image.Image
    annotation_artifacts: PixelAnnotationArtifacts
    trace_payload: dict[str, Any]
    task_versions: dict[str, str]
    scene_id: str
    selected_query: str
    answer_value: int


def _build_trace_payload(
    *,
    artifacts: ConcentricChordArtifacts,
    selected_query: str,
    query_probabilities: Mapping[str, float],
    plan: ConcentricChordObjectivePlan,
) -> dict[str, Any]:
    """Build common trace metadata after the public task binds the objective."""

    base = concentric_trace_base(
        rendered=artifacts.rendered,
        annotation_artifacts=artifacts.annotation_artifacts,
        spec=plan.spec,
        case_index=int(plan.case_index),
        answer_probabilities=plan.answer_probabilities,
        render_meta=artifacts.render_meta,
        noise_meta=artifacts.noise_meta,
        image_size=(int(artifacts.image.size[0]), int(artifacts.image.size[1])),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=artifacts.prompt_artifacts,
        query_id=str(selected_query),
        params={
            **dict(base["query_params"]),
            "query_id": str(selected_query),
            "internal_query_id": str(plan.prompt_query_key),
            "query_id_probabilities": dict(query_probabilities),
        },
    )
    query_spec["scene_id"] = SCENE_ID
    return {
        "scene_ir": {
            **dict(base["scene_ir_base"]),
            "relations": {
                **dict(base["relation_base"]),
                "query_id": str(selected_query),
                "internal_query_id": str(plan.prompt_query_key),
            },
        },
        "query_spec": query_spec,
        "render_spec": dict(base["render_spec"]),
        "render_map": dict(base["render_map"]),
        "execution_trace": {
            **dict(base["execution_base"]),
            "query_id": str(selected_query),
            "internal_query_id": str(plan.prompt_query_key),
            "query_id_probabilities": dict(query_probabilities),
        },
        "witness_symbolic": {
            **dict(base["witness_base"]),
            "query_id": str(selected_query),
            "internal_query_id": str(plan.prompt_query_key),
        },
        "projected_annotation": dict(base["projected_annotation"]),
    }


def prepare_concentric_chord_parts(
    task: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> ConcentricChordPreparedParts:
    """Run shared rendering after a public task prepares its objective plan."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=task.supported_query_ids,
        default_query_id=task.default_query_id,
        task_id=task.task_id,
    )
    plan = task.prepare_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query=str(selected_query),
        query_probabilities=query_probabilities,
    )
    render_defaults, prompt_defaults = load_concentric_chord_task_defaults(task.task_id)
    artifacts = prepare_concentric_chord_artifacts(
        spec=plan.spec,
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(plan.prompt_query_key),
        max_attempts=int(max_attempts),
        random_namespace=str(plan.random_namespace),
    )
    return ConcentricChordPreparedParts(
        prompt=str(artifacts.prompt_artifacts.prompt),
        prompt_variants=dict(artifacts.prompt_artifacts.prompt_variants),
        image=artifacts.image,
        annotation_artifacts=artifacts.annotation_artifacts,
        trace_payload=_build_trace_payload(
            artifacts=artifacts,
            selected_query=str(selected_query),
            query_probabilities=query_probabilities,
            plan=plan,
        ),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        selected_query=str(selected_query),
        answer_value=int(plan.spec.answer),
    )


__all__ = [
    "ConcentricChordObjectivePlan",
    "ConcentricChordPreparedParts",
    "prepare_concentric_chord_parts",
]
