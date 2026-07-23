"""Scene-private lifecycle orchestration for platformer public tasks."""

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

from .shared.defaults import SCENE_ID
from .shared.output import build_platformer_common_trace_payload, common_platformer_trace_params
from .shared.prompts import build_platformer_prompt_artifacts
from .shared.rendering import RenderedPlatformerTaskContext, render_platformer_task_context
from .shared.sampling import PlatformerVisualAxes, resolve_platformer_visual_axes
from .shared.state import PlatformerSample, validate_platformer_sample


AnnotationBuilder = Callable[[RenderedPlatformerTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, PlatformerVisualAxes], "AttemptPlatformerResult"]
ObjectivePreparer = Callable[[int, Mapping[str, Any], Mapping[str, float], str, PlatformerVisualAxes], "ObjectivePlatformerPlan"]


@dataclass(frozen=True)
class AttemptPlatformerResult:
    """Task-owned result of one constructed platformer attempt."""

    sample: PlatformerSample
    answer_gt: TypedValue
    build_annotation: AnnotationBuilder
    annotation_entity_ids: tuple[str, ...]
    witness_type: str
    relations_extra: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectivePlatformerPlan:
    """Prepared task-owned platformer objective hooks for one generated instance."""

    attempt_namespace: str
    prompt_query_key: str
    answer_hint: str
    annotation_hint: str
    json_example: str
    json_example_answer_only: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder


def run_platformer_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run common platformer query, render, prompt, annotation, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_platformer_visual_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=f"{SCENE_ID}.visual",
        params=task_params,
    )
    objective = prepare_objective(int(instance_seed), task_params, query_probabilities, query_id, axes)

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            attempt = objective.construct_attempt(rng, axes)
            validate_platformer_sample(attempt.sample)
        except ValueError:
            continue

        rendered_context = render_platformer_task_context(
            sample=attempt.sample,
            params=task_params,
            render_defaults=render_defaults,
            namespace=f"{SCENE_ID}.render",
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_platformer_prompt_artifacts(
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
            params=common_platformer_trace_params(
                axes,
                attempt.sample,
                prompt_query_key=str(objective.prompt_query_key),
                query_id_probabilities=query_probabilities,
                extra_params=dict(objective.query_params),
            ),
        )
        trace_payload = build_platformer_common_trace_payload(
            axes=axes,
            sample=attempt.sample,
            rendered_context=rendered_context,
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            query_spec=query_spec,
            witness_type=str(attempt.witness_type),
            relations_extra=dict(attempt.relations_extra),
            execution_extra={
                "answer": attempt.answer_gt.value,
                **dict(attempt.execution_extra),
            },
        )
        trace_payload["scene_ir"]["relations"]["query_id"] = str(query_id)
        trace_payload["render_spec"]["query_id"] = str(query_id)
        trace_payload["execution_trace"]["query_id"] = str(query_id)
        trace_payload["query_spec"]["template_id"] = str(prompt_defaults["bundle_id"])
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

    raise RuntimeError(f"{task_id} failed to generate a valid platformer scene after {max_attempts} attempts")


__all__ = [
    "AttemptPlatformerResult",
    "ObjectivePlatformerPlan",
    "run_platformer_lifecycle",
]
