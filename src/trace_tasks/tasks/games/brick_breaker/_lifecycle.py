"""Scene-private lifecycle plumbing for Brick-breaker public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    point_set_annotation_artifacts,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import SCENE_ID
from .shared.output import build_brick_breaker_common_trace_params, build_brick_breaker_trace_payload
from .shared.prompts import build_brick_breaker_prompt_artifacts
from .shared.rendering import RenderedBrickBreakerTaskContext, render_brick_breaker_task_scene
from .shared.sampling import (
    ResolvedBrickBreakerIntegerAxis,
    ResolvedBrickBreakerPlayfieldAxes,
    ResolvedBrickBreakerSceneAxes,
    resolve_brick_breaker_integer_axis,
    resolve_brick_breaker_playfield_axes,
    resolve_brick_breaker_render_params,
    resolve_brick_breaker_scene_axes,
)
from .shared.state import BrickBreakerSample


AnnotationBuilder = Callable[[RenderedBrickBreakerTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, ResolvedBrickBreakerSceneAxes], "BrickBreakerAttemptResult"]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "BrickBreakerObjectivePlan"]


@dataclass(frozen=True)
class BrickBreakerAttemptResult:
    """Task-owned result of one constructed Brick-breaker sample."""

    sample: BrickBreakerSample
    answer_gt: TypedValue
    annotation_entity_ids: tuple[str, ...]
    build_annotation: AnnotationBuilder
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrickBreakerObjectivePlan:
    """Prepared task-owned objective hooks for one generated Brick-breaker instance."""

    attempt_namespace: str
    prompt_query_key: str
    render_mode: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder


@dataclass(frozen=True)
class BrickBreakerIntegerAxisSpec:
    """Task-owned integer-axis declaration resolved by neutral scene plumbing."""

    name: str
    fallback_support: tuple[int, ...]
    balanced_flag_key: str
    support_key: str | None = None
    explicit_key: str | None = None


def brick_breaker_integer_axis_spec(
    name: str,
    fallback_support: tuple[int, ...],
    *,
    balanced_flag_key: str,
    support_key: str | None = None,
    explicit_key: str | None = None,
) -> BrickBreakerIntegerAxisSpec:
    """Declare one task-local Brick-breaker integer axis without embedding identity."""

    return BrickBreakerIntegerAxisSpec(
        name=str(name),
        fallback_support=tuple(int(value) for value in fallback_support),
        balanced_flag_key=str(balanced_flag_key),
        support_key=None if support_key is None else str(support_key),
        explicit_key=None if explicit_key is None else str(explicit_key),
    )


def resolve_brick_breaker_playfield_axis_specs(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    brick_row_count_support: tuple[int, ...],
    brick_col_count_support: tuple[int, ...],
    catch_lane_count_support: tuple[int, ...],
) -> tuple[ResolvedBrickBreakerPlayfieldAxes, dict[str, Any]]:
    """Resolve task-declared playfield axes and return standard trace params."""

    playfield_axis_seed = hash64(int(instance_seed), f"{task_id}.playfield_axes")
    playfield_axes = resolve_brick_breaker_playfield_axes(
        int(playfield_axis_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        brick_row_count_support=tuple(int(value) for value in brick_row_count_support),
        brick_col_count_support=tuple(int(value) for value in brick_col_count_support),
        catch_lane_count_support=tuple(int(value) for value in catch_lane_count_support),
    )
    return playfield_axes, {
        "brick_row_count_support": [int(value) for value in playfield_axes.brick_rows.support],
        "brick_row_count_probabilities": dict(playfield_axes.brick_rows.probabilities),
        "brick_col_count_support": [int(value) for value in playfield_axes.brick_cols.support],
        "brick_col_count_probabilities": dict(playfield_axes.brick_cols.probabilities),
        "catch_lane_count_support": [int(value) for value in playfield_axes.lane_count.support],
        "catch_lane_count_probabilities": dict(playfield_axes.lane_count.probabilities),
    }


def resolve_brick_breaker_integer_axis_spec(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    spec: BrickBreakerIntegerAxisSpec,
) -> tuple[ResolvedBrickBreakerIntegerAxis, dict[str, Any]]:
    """Resolve one task-declared integer axis and return standard trace params."""

    axis = resolve_brick_breaker_integer_axis(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(spec.support_key or f"{spec.name}_support"),
        explicit_key=str(spec.explicit_key or spec.name),
        fallback_support=tuple(int(value) for value in spec.fallback_support),
        namespace=f"{task_id}.{spec.name}",
        balanced_flag_key=str(spec.balanced_flag_key),
    )
    return axis, {
        str(spec.name): int(axis.value),
        f"{spec.name}_support": [int(value) for value in axis.support],
        f"{spec.name}_probabilities": dict(axis.probabilities),
    }


def point_set_attempt(
    *,
    sample: BrickBreakerSample,
    answer_gt: TypedValue,
    annotation_entity_ids: tuple[str, ...] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> BrickBreakerAttemptResult:
    """Package an answer whose annotation is the center point set for scene entities."""

    resolved_entity_ids = tuple(str(entity_id) for entity_id in (annotation_entity_ids or sample.annotation_entity_ids))

    def build_annotation(rendered_context: RenderedBrickBreakerTaskContext) -> AnnotationArtifacts:
        entity_bboxes = rendered_context.rendered_scene.render_map["entity_bboxes_px"]
        return point_set_annotation_artifacts(
            [
                [
                    round(float(entity_bboxes[str(entity_id)][0] + entity_bboxes[str(entity_id)][2]) / 2.0, 3),
                    round(float(entity_bboxes[str(entity_id)][1] + entity_bboxes[str(entity_id)][3]) / 2.0, 3),
                ]
                for entity_id in resolved_entity_ids
            ]
        )

    return BrickBreakerAttemptResult(
        sample=sample,
        answer_gt=answer_gt,
        annotation_entity_ids=resolved_entity_ids,
        build_annotation=build_annotation,
        execution_extra=dict(execution_extra or {}),
    )


def bbox_set_attempt(
    *,
    sample: BrickBreakerSample,
    answer_gt: TypedValue,
    annotation_entity_ids: tuple[str, ...] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> BrickBreakerAttemptResult:
    """Package an answer whose annotation is the bbox set for scene entities."""

    resolved_entity_ids = tuple(str(entity_id) for entity_id in (annotation_entity_ids or sample.annotation_entity_ids))

    def build_annotation(rendered_context: RenderedBrickBreakerTaskContext) -> AnnotationArtifacts:
        entity_bboxes = rendered_context.rendered_scene.render_map["entity_bboxes_px"]
        return bbox_set_annotation_artifacts(
            [entity_bboxes[str(entity_id)] for entity_id in resolved_entity_ids]
        )

    return BrickBreakerAttemptResult(
        sample=sample,
        answer_gt=answer_gt,
        annotation_entity_ids=resolved_entity_ids,
        build_annotation=build_annotation,
        execution_extra=dict(execution_extra or {}),
    )


def point_attempt(
    *,
    sample: BrickBreakerSample,
    answer_gt: TypedValue,
    annotation_entity_ids: tuple[str, ...] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
) -> BrickBreakerAttemptResult:
    """Package a label answer whose annotation is one scene-entity center point."""

    resolved_entity_ids = tuple(str(entity_id) for entity_id in (annotation_entity_ids or sample.annotation_entity_ids))
    if len(resolved_entity_ids) != 1:
        raise ValueError("Brick-breaker scalar point annotation requires exactly one entity")

    def build_annotation(rendered_context: RenderedBrickBreakerTaskContext) -> AnnotationArtifacts:
        entity_bboxes = rendered_context.rendered_scene.render_map["entity_bboxes_px"]
        bbox = entity_bboxes[str(resolved_entity_ids[0])]
        return point_annotation_artifacts(
            [
                round(float(bbox[0] + bbox[2]) / 2.0, 3),
                round(float(bbox[1] + bbox[3]) / 2.0, 3),
            ]
        )

    return BrickBreakerAttemptResult(
        sample=sample,
        answer_gt=answer_gt,
        annotation_entity_ids=resolved_entity_ids,
        build_annotation=build_annotation,
        execution_extra=dict(execution_extra or {}),
    )


def run_brick_breaker_lifecycle(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    domain: str = "games",
) -> TaskOutput:
    """Run neutral Brick-breaker query, render, prompt, trace, retry, and output plumbing."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_brick_breaker_scene_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query),
        dict(query_probabilities),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        render_params = resolve_brick_breaker_render_params(task_params, instance_seed=int(instance_seed))
        sample = attempt.sample
        rendered_context = render_brick_breaker_task_scene(
            brick_rows=int(sample.brick_rows),
            brick_cols=int(sample.brick_cols),
            lane_count=int(sample.lane_count),
            bricks=sample.bricks,
            render_mode=str(objective.render_mode),
            target_brick_id=sample.target_brick_id,
            target_lane_index=sample.target_lane_index,
            ball_start_lane_index=sample.ball_start_lane_index,
            style_variant=str(axes.style_variant),
            render_params=render_params,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_brick_breaker_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            instance_seed=int(instance_seed),
        )
        sample_query_params = {
            "brick_rows": int(sample.brick_rows),
            "brick_cols": int(sample.brick_cols),
            "brick_count": int(len(sample.bricks)),
            "lane_count": int(sample.lane_count),
            "target_brick_id": sample.target_brick_id,
            "target_brick_label": sample.target_brick_label,
            "target_row_remaining_brick_ids": list(sample.target_row_remaining_brick_ids),
            "target_row_remaining_count": sample.target_row_remaining_count,
            "target_lane_index": sample.target_lane_index,
            "target_lane_label": sample.target_lane_label,
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params=build_brick_breaker_common_trace_params(
                axes=axes,
                extra_params={
                    **sample_query_params,
                    **dict(objective.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_brick_breaker_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            axes=axes,
            sample=sample,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
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

    raise RuntimeError(f"{task_id} failed to generate a valid Brick-breaker scene after {max_attempts} attempts")


__all__ = [
    "BrickBreakerAttemptResult",
    "BrickBreakerIntegerAxisSpec",
    "BrickBreakerObjectivePlan",
    "bbox_set_attempt",
    "brick_breaker_integer_axis_spec",
    "point_attempt",
    "point_set_attempt",
    "resolve_brick_breaker_integer_axis_spec",
    "resolve_brick_breaker_playfield_axis_specs",
    "run_brick_breaker_lifecycle",
]
