"""Scene-private lifecycle orchestration for Pac-Man public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import build_pacman_common_trace_payload, common_pacman_trace_params
from .shared.prompts import build_pacman_prompt_artifacts
from .shared.rendering import RenderedPacmanTaskContext, render_pacman_task_context
from .shared.sampling import PacmanVisualAxes, resolve_pacman_visual_axes
from .shared.state import PacmanSceneState, validate_pacman_scene_state
from .shared.defaults import SCENE_ID


AnnotationBuilder = Callable[[RenderedPacmanTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, PacmanVisualAxes], "AttemptPacmanResult"]
ObjectivePreparer = Callable[[int, Mapping[str, Any], Mapping[str, float], str], "ObjectivePacmanPlan"]


@dataclass(frozen=True)
class AttemptPacmanResult:
    """Task-owned result of one constructed Pac-Man attempt."""

    scene: PacmanSceneState
    answer_gt: TypedValue
    build_annotation: AnnotationBuilder
    annotation_entity_ids: tuple[str, ...]
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectivePacmanPlan:
    """Prepared task-owned Pac-Man objective hooks for one generated instance."""

    attempt_namespace: str
    prompt_query_key: str
    answer_hint: str
    annotation_hint: str
    json_example: str
    json_example_answer_only: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder


def run_pacman_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run common Pac-Man query, render, prompt, annotation, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_pacman_visual_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=f"{SCENE_ID}.visual",
        params=task_params,
    )
    objective = prepare_objective(int(instance_seed), task_params, query_probabilities, query_id)

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            attempt = objective.construct_attempt(rng, axes)
            validate_pacman_scene_state(attempt.scene)
        except ValueError:
            continue

        rendered_context = render_pacman_task_context(
            axes=axes,
            scene=attempt.scene,
            params=task_params,
            render_defaults=render_defaults,
            namespace=f"{SCENE_ID}.render",
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_pacman_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            scene_variant=str(axes.scene_variant),
            instance_seed=int(instance_seed),
            answer_hint=str(objective.answer_hint),
            annotation_hint=str(objective.annotation_hint),
            json_example=str(objective.json_example),
            json_example_answer_only=str(objective.json_example_answer_only),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=common_pacman_trace_params(
                axes,
                attempt.scene,
                query_id_probabilities=query_probabilities,
                extra_params=dict(objective.query_params),
            ),
        )
        trace_payload = build_pacman_common_trace_payload(
            axes=axes,
            scene=attempt.scene,
            rendered_context=rendered_context,
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            query_spec=query_spec,
            execution_extra={
                "answer": attempt.answer_gt.value,
                **dict(attempt.execution_extra),
            },
        )
        trace_payload["scene_ir"]["relations"]["query_id"] = str(query_id)
        trace_payload["render_spec"]["query_id"] = str(query_id)
        trace_payload["execution_trace"]["query_id"] = str(query_id)
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid Pac-Man maze after {max_attempts} attempts")


__all__ = [
    "AttemptPacmanResult",
    "ObjectivePacmanPlan",
    "run_pacman_lifecycle",
]
