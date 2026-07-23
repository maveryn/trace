"""Scene-private lifecycle plumbing for dominoes public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import domino_bbox_set_annotation
from .shared.defaults import SCENE_ID
from .shared.output import build_domino_common_trace_params, build_domino_trace_payload
from .shared.prompts import (
    build_domino_prompt_artifacts,
    domino_integer_json_examples,
    domino_object_description,
    domino_output_slots,
)
from .shared.rendering import RenderedDominoTaskContext, render_domino_task_scene
from .shared.sampling import (
    CountedCandidateRecipe,
    resolve_domino_candidate_count_axis,
    resolve_domino_render_params,
    resolve_domino_scene_axes,
    resolve_domino_target_axis,
    sample_counted_candidate_scene_from_recipe,
)
from .shared.state import DominoIntegerAxis, DominoSceneAxes, SampledDominoScene


AnnotationBuilder = Callable[[RenderedDominoTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, DominoSceneAxes], "DominoAttemptResult"]
CountSampleBuilder = Callable[[Any, int, int], SampledDominoScene]
RecipeBuilder = Callable[[Any], CountedCandidateRecipe]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], DominoSceneAxes],
    "DominoObjectivePlan",
]


@dataclass(frozen=True)
class DominoCountAxes:
    """Task-owned target answer and visible candidate-count axes."""

    target_axis: DominoIntegerAxis
    candidate_axis: DominoIntegerAxis
    query_params: Mapping[str, Any]


@dataclass(frozen=True)
class DominoAttemptResult:
    """Task-owned result of one constructed dominoes sample."""

    sample: SampledDominoScene
    answer_gt: TypedValue
    annotation_entity_ids: tuple[str, ...]
    build_annotation: AnnotationBuilder
    query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)
    prompt_dynamic_slots: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DominoObjectivePlan:
    """Prepared task-owned objective hooks for one dominoes instance."""

    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    prompt_dynamic_slots: Mapping[str, Any]
    construct_attempt: AttemptBuilder


def resolve_domino_count_axes(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    axes: DominoSceneAxes,
    target_support_key: str,
    target_fallback_support: tuple[int, ...],
    target_namespace: str,
    minimum_candidate_count: int | Callable[[int], int],
    candidate_namespace: str,
) -> DominoCountAxes:
    """Resolve common answer/candidate axes after the task owns feasibility."""

    target_axis = resolve_domino_target_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(target_support_key),
        fallback_support=target_fallback_support,
        namespace=str(target_namespace),
    )
    if callable(minimum_candidate_count):
        resolved_minimum_candidate_count = int(minimum_candidate_count(int(target_axis.value)))
    else:
        resolved_minimum_candidate_count = int(minimum_candidate_count)
    candidate_axis = resolve_domino_candidate_count_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        scene_variant=str(axes.scene_variant),
        minimum_candidate_count=int(resolved_minimum_candidate_count),
        namespace=str(candidate_namespace),
    )
    return DominoCountAxes(
        target_axis=target_axis,
        candidate_axis=candidate_axis,
        query_params={
            "target_answer": int(target_axis.value),
            "target_answer_index": int(target_axis.value),
            "target_answer_support": [int(value) for value in target_axis.support],
            "target_answer_probabilities": dict(target_axis.probabilities),
            "candidate_count": int(candidate_axis.value),
            "candidate_count_support": [int(value) for value in candidate_axis.support],
            "candidate_count_probabilities": dict(candidate_axis.probabilities),
        },
    )


def domino_bbox_set_attempt(
    *,
    sample: SampledDominoScene,
    answer_gt: TypedValue,
    annotation_entity_ids: tuple[str, ...] | None = None,
    query_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
    prompt_dynamic_slots: Mapping[str, Any] | None = None,
) -> DominoAttemptResult:
    """Package an answer whose annotation is selected domino tile bboxes."""

    resolved_entity_ids = tuple(str(entity_id) for entity_id in (annotation_entity_ids or sample.annotation_tile_ids))
    return DominoAttemptResult(
        sample=sample,
        answer_gt=answer_gt,
        annotation_entity_ids=resolved_entity_ids,
        build_annotation=lambda rendered_context: domino_bbox_set_annotation(
            rendered_context,
            resolved_entity_ids,
        ),
        query_params=dict(query_params or {}),
        execution_extra=dict(execution_extra or {}),
        prompt_dynamic_slots=dict(prompt_dynamic_slots or {}),
    )


def prepare_domino_count_objective(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    axes: DominoSceneAxes,
    prompt_query_key: str,
    attempt_namespace: str,
    target_support_key: str,
    target_fallback_support: tuple[int, ...],
    target_namespace: str,
    minimum_candidate_count: int | Callable[[int], int],
    candidate_namespace: str,
    sample_scene: CountSampleBuilder,
    example_answer: int,
) -> DominoObjectivePlan:
    """Prepare a neutral integer-count objective after public code owns sampling."""

    count_axes = resolve_domino_count_axes(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        axes=axes,
        target_support_key=str(target_support_key),
        target_fallback_support=target_fallback_support,
        target_namespace=str(target_namespace),
        minimum_candidate_count=minimum_candidate_count,
        candidate_namespace=str(candidate_namespace),
    )
    json_example, json_example_answer_only = domino_integer_json_examples(answer_value=int(example_answer))
    prompt_dynamic_slots = domino_output_slots(
        prompt_query_key=str(prompt_query_key),
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
    )

    def construct_attempt(rng, _axes: DominoSceneAxes):
        sample = sample_scene(
            rng,
            candidate_count=int(count_axes.candidate_axis.value),
            target_answer=int(count_axes.target_axis.value),
        )
        return domino_bbox_set_attempt(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(sample.answer_value)),
            execution_extra={"target_answer": int(sample.answer_value)},
        )

    return DominoObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(prompt_query_key),
        query_params=dict(count_axes.query_params),
        prompt_dynamic_slots=prompt_dynamic_slots,
        construct_attempt=construct_attempt,
    )


def prepare_domino_recipe_count_objective(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    axes: DominoSceneAxes,
    prompt_query_key: str,
    attempt_namespace: str,
    target_support_key: str,
    target_fallback_support: tuple[int, ...],
    target_namespace: str,
    minimum_candidate_count: int | Callable[[int], int],
    candidate_namespace: str,
    build_recipe: RecipeBuilder,
    recipe_attempts: int,
    example_answer: int,
) -> DominoObjectivePlan:
    """Prepare a counted objective whose task supplies one counted-candidate recipe."""

    def sample_scene(rng, *, candidate_count: int, target_answer: int) -> SampledDominoScene:
        return sample_counted_candidate_scene_from_recipe(
            rng,
            attempts=int(recipe_attempts),
            candidate_count=int(candidate_count),
            target_answer=int(target_answer),
            build_recipe=build_recipe,
        )

    return prepare_domino_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        axes=axes,
        prompt_query_key=str(prompt_query_key),
        attempt_namespace=str(attempt_namespace),
        target_support_key=str(target_support_key),
        target_fallback_support=target_fallback_support,
        target_namespace=str(target_namespace),
        minimum_candidate_count=minimum_candidate_count,
        candidate_namespace=str(candidate_namespace),
        sample_scene=sample_scene,
        example_answer=int(example_answer),
    )


def run_domino_lifecycle(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prepare_objective: ObjectivePreparer,
    domain: str = "games",
) -> TaskOutput:
    """Run neutral domino query, render, prompt, trace, retry, and output plumbing."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_domino_scene_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
    )
    render_params = resolve_domino_render_params(
        task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query),
        dict(query_probabilities),
        axes,
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        sample = attempt.sample
        rendered_context = render_domino_task_scene(
            chain_tiles=sample.chain_tiles,
            candidate_tiles=sample.candidate_tiles,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            render_params=render_params,
            render_defaults=render_defaults,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_domino_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            dynamic_slots={
                **dict(objective.prompt_dynamic_slots),
                **dict(attempt.prompt_dynamic_slots),
                "object_description": domino_object_description(
                    has_chain=bool(sample.chain_tiles),
                    has_reference=sample.reference_tile_id is not None,
                    has_candidates=bool(sample.candidate_tiles),
                    scene_variant=str(axes.scene_variant),
                ),
            },
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params=build_domino_common_trace_params(
                axes=axes,
                extra_params={
                    **dict(objective.query_params),
                    **dict(attempt.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_domino_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            axes=axes,
            sample=sample,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            query_spec=query_spec,
            answer_value=attempt.answer_gt.value,
            execution_extra={
                **dict(attempt.execution_extra),
                "answer": attempt.answer_gt.value,
            },
        )
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
            query_id=str(selected_query),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid dominoes scene after {max_attempts} attempts")


__all__ = [
    "DominoAttemptResult",
    "DominoCountAxes",
    "DominoObjectivePlan",
    "domino_bbox_set_attempt",
    "prepare_domino_count_objective",
    "prepare_domino_recipe_count_objective",
    "resolve_domino_count_axes",
    "run_domino_lifecycle",
]
