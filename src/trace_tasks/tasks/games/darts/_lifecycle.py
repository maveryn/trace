"""Scene-private lifecycle plumbing for simplified darts public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import dart_bbox_set_annotation, dart_center_point_annotation, dart_center_point_set_annotation
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.output import build_darts_common_trace_params, build_darts_trace_payload
from .shared.prompts import build_darts_prompt_artifacts
from .shared.rendering import (
    DartboardRenderParams,
    RenderedDartsTaskContext,
    render_darts_task_scene,
)
from .shared.sampling import (
    resolve_darts_count_target_axis,
    resolve_darts_integer_axis,
    resolve_darts_render_params,
    resolve_darts_scene_axes,
    sample_darts_for_count,
)
from .shared.state import DartsSampledScene, DartsSceneAxes, DartsScoreSlot


AnnotationBuilder = Callable[[RenderedDartsTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, DartsSceneAxes], "DartsAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], DartboardRenderParams],
    "DartsObjectivePlan",
]


@dataclass(frozen=True)
class DartsAttemptResult:
    """Task-owned result of one constructed darts sample."""

    sample: DartsSampledScene
    answer_gt: TypedValue
    annotation_entity_ids: tuple[str, ...]
    build_annotation: AnnotationBuilder
    query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DartsObjectivePlan:
    """Prepared task-owned objective hooks for one generated darts instance."""

    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    prompt_dynamic_slots: Mapping[str, Any]
    construct_attempt: AttemptBuilder


def dart_point_set_attempt(
    *,
    sample: DartsSampledScene,
    answer_gt: TypedValue,
    annotation_entity_ids: Sequence[str] | None = None,
    query_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> DartsAttemptResult:
    """Package an answer whose annotation is the selected dart center set."""

    resolved_entity_ids = tuple(str(entity_id) for entity_id in (annotation_entity_ids or sample.annotation_dart_ids))
    return DartsAttemptResult(
        sample=sample,
        answer_gt=answer_gt,
        annotation_entity_ids=resolved_entity_ids,
        build_annotation=lambda rendered_context: dart_center_point_set_annotation(
            rendered_context,
            resolved_entity_ids,
        ),
        query_params=dict(query_params or {}),
        execution_extra=dict(execution_extra or {}),
    )


def dart_bbox_set_attempt(
    *,
    sample: DartsSampledScene,
    answer_gt: TypedValue,
    annotation_entity_ids: Sequence[str] | None = None,
    query_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> DartsAttemptResult:
    """Package an answer whose annotation is the selected dart bbox set."""

    resolved_entity_ids = tuple(str(entity_id) for entity_id in (annotation_entity_ids or sample.annotation_dart_ids))
    return DartsAttemptResult(
        sample=sample,
        answer_gt=answer_gt,
        annotation_entity_ids=resolved_entity_ids,
        build_annotation=lambda rendered_context: dart_bbox_set_annotation(
            rendered_context,
            resolved_entity_ids,
        ),
        query_params=dict(query_params or {}),
        execution_extra=dict(execution_extra or {}),
    )


def dart_point_attempt(
    *,
    sample: DartsSampledScene,
    answer_gt: TypedValue,
    annotation_entity_id: str | None = None,
    query_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> DartsAttemptResult:
    """Package an answer whose annotation is one selected dart center point."""

    resolved_entity_ids = tuple(
        str(entity_id)
        for entity_id in (
            sample.annotation_dart_ids
            if annotation_entity_id is None
            else (annotation_entity_id,)
        )
    )
    if len(resolved_entity_ids) != 1:
        raise ValueError("dart scalar point annotation requires exactly one dart id")
    entity_id = str(resolved_entity_ids[0])
    return DartsAttemptResult(
        sample=sample,
        answer_gt=answer_gt,
        annotation_entity_ids=(entity_id,),
        build_annotation=lambda rendered_context: dart_center_point_annotation(
            rendered_context,
            entity_id,
        ),
        query_params=dict(query_params or {}),
        execution_extra=dict(execution_extra or {}),
    )


def prepare_darts_exact_count_objective(
    *,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: DartboardRenderParams,
    instance_seed: int,
    target_namespace: str,
    attempt_namespace: str,
    prompt_query_key: str,
    prompt_dynamic_slots: Mapping[str, Any],
    qualifying_slots: tuple[DartsScoreSlot, ...],
    nonqualifying_slots: tuple[DartsScoreSlot, ...],
    extra_query_params: Mapping[str, Any],
    extra_execution_params: Mapping[str, Any],
) -> DartsObjectivePlan:
    """Prepare a neutral exact-count objective after public code defines membership."""

    dart_count_axis = resolve_darts_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key="count_query_dart_count_support",
        explicit_key="dart_count",
        fallback_support=DEFAULTS.count_query_dart_count_support,
        namespace=f"{target_namespace}.dart_count",
        balanced_flag_key="balanced_dart_count_sampling",
    )
    target_axis = resolve_darts_count_target_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        dart_count=int(dart_count_axis.value),
        namespace=f"{target_namespace}.target_answer",
    )
    query_params = {
        "dart_count": int(dart_count_axis.value),
        "dart_count_support": [int(value) for value in dart_count_axis.support],
        "dart_count_probabilities": dict(dart_count_axis.probabilities),
        "target_answer": int(target_axis.value),
        "target_answer_support": [int(value) for value in target_axis.support],
        "target_answer_probabilities": dict(target_axis.probabilities),
        **dict(extra_query_params),
    }

    def construct_attempt(rng, _axes: DartsSceneAxes):
        sample = sample_darts_for_count(
            rng,
            dart_count=int(dart_count_axis.value),
            target_answer=int(target_axis.value),
            render_params=render_params,
            qualifying_slots=qualifying_slots,
            nonqualifying_slots=nonqualifying_slots,
        )
        return dart_bbox_set_attempt(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(target_axis.value)),
            execution_extra={
                "target_answer": int(target_axis.value),
                **dict(extra_execution_params),
            },
        )

    return DartsObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(prompt_query_key),
        query_params=query_params,
        prompt_dynamic_slots=dict(prompt_dynamic_slots),
        construct_attempt=construct_attempt,
    )


def run_darts_lifecycle(
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
    """Run neutral darts query, render, prompt, trace, retry, and output plumbing."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_darts_scene_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
    )
    render_params = resolve_darts_render_params(
        task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query),
        dict(query_probabilities),
        render_params,
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        sample = attempt.sample
        rendered_context = render_darts_task_scene(
            darts=sample.darts,
            style_variant=str(axes.style_variant),
            render_params=render_params,
            render_defaults=render_defaults,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_darts_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            dynamic_slots=dict(objective.prompt_dynamic_slots),
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params=build_darts_common_trace_params(
                axes=axes,
                extra_params={
                    **dict(objective.query_params),
                    **dict(attempt.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_darts_trace_payload(
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

    raise RuntimeError(f"{task_id} failed to generate a valid darts scene after {max_attempts} attempts")


__all__ = [
    "DartsAttemptResult",
    "DartsObjectivePlan",
    "dart_bbox_set_attempt",
    "dart_point_attempt",
    "dart_point_set_attempt",
    "prepare_darts_exact_count_objective",
    "run_darts_lifecycle",
]
