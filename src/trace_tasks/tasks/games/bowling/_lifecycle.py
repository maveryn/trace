"""Scene-private lifecycle plumbing for Bowling public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_annotation_artifacts,
    segment_annotation_artifacts,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import SCENE_ID
from .shared.output import build_bowling_common_trace_params, build_bowling_trace_payload
from .shared.prompts import build_bowling_prompt_artifacts
from .shared.rendering import RenderedBowlingTaskContext, render_bowling_task_scene
from .shared.sampling import (
    ResolvedBowlingIntegerAxis,
    ResolvedBowlingSceneAxes,
    resolve_bowling_integer_axis,
    resolve_bowling_render_params,
    resolve_bowling_scene_axes,
)
from .shared.state import BowlingSample


AnnotationBuilder = Callable[[RenderedBowlingTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, ResolvedBowlingSceneAxes], "BowlingAttemptResult"]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "BowlingObjectivePlan"]


@dataclass(frozen=True)
class BowlingAttemptResult:
    """Task-owned result of one constructed Bowling sample."""

    sample: BowlingSample
    answer_gt: TypedValue
    annotation_entity_ids: tuple[str, ...]
    build_annotation: AnnotationBuilder
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BowlingObjectivePlan:
    """Prepared task-owned objective hooks for one generated Bowling instance."""

    attempt_namespace: str
    prompt_query_key: str
    render_mode: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder


@dataclass(frozen=True)
class BowlingIntegerAxisSpec:
    """Task-owned integer-axis declaration resolved by neutral Bowling plumbing."""

    name: str
    fallback_support: tuple[int, ...]
    balanced_flag_key: str
    support_key: str | None = None
    explicit_key: str | None = None
    trace_aliases: tuple[str, ...] = ()


def bowling_integer_axis_spec(
    name: str,
    fallback_support: tuple[int, ...],
    *,
    balanced_flag_key: str,
    support_key: str | None = None,
    explicit_key: str | None = None,
    trace_aliases: tuple[str, ...] = (),
) -> BowlingIntegerAxisSpec:
    """Declare one task-local Bowling integer axis without embedding task identity."""

    return BowlingIntegerAxisSpec(
        name=str(name),
        fallback_support=tuple(int(value) for value in fallback_support),
        balanced_flag_key=str(balanced_flag_key),
        support_key=None if support_key is None else str(support_key),
        explicit_key=None if explicit_key is None else str(explicit_key),
        trace_aliases=tuple(str(value) for value in trace_aliases),
    )


def resolve_bowling_integer_axis_specs(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    axis_specs: tuple[BowlingIntegerAxisSpec, ...],
) -> tuple[dict[str, ResolvedBowlingIntegerAxis], dict[str, Any]]:
    """Resolve task-declared Bowling axes and return standard trace params."""

    axes: dict[str, ResolvedBowlingIntegerAxis] = {}
    trace_params: dict[str, Any] = {}
    for spec in axis_specs:
        axis = resolve_bowling_integer_axis(
            int(instance_seed),
            params=task_params,
            gen_defaults=gen_defaults,
            support_key=str(spec.support_key or f"{spec.name}_support"),
            explicit_key=str(spec.explicit_key or spec.name),
            fallback_support=tuple(int(value) for value in spec.fallback_support),
            namespace=f"{task_id}.{spec.name}",
            balanced_flag_key=str(spec.balanced_flag_key),
        )
        axes[str(spec.name)] = axis
        trace_params[str(spec.name)] = int(axis.value)
        trace_params[f"{spec.name}_support"] = [int(value) for value in axis.support]
        trace_params[f"{spec.name}_probabilities"] = dict(axis.probabilities)
        for alias in spec.trace_aliases:
            trace_params[str(alias)] = int(axis.value)
    return axes, trace_params


def pin_point_label_attempt(
    *,
    sample: BowlingSample,
    answer_value: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> BowlingAttemptResult:
    """Package a pin-label answer whose annotation is the target pin center."""

    annotation_entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    if len(annotation_entity_ids) != 1:
        raise ValueError("Bowling first-pin annotation requires exactly one target pin")

    def build_annotation(rendered_context: RenderedBowlingTaskContext) -> AnnotationArtifacts:
        bbox = rendered_context.rendered_scene.render_map["entity_bboxes_px"][str(annotation_entity_ids[0])]
        point = [
            0.5 * (float(bbox[0]) + float(bbox[2])),
            0.5 * (float(bbox[1]) + float(bbox[3])),
        ]
        return point_annotation_artifacts(point)

    return BowlingAttemptResult(
        sample=sample,
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_entity_ids=annotation_entity_ids,
        build_annotation=build_annotation,
        execution_extra=dict(execution_extra or {}),
    )


def path_segment_label_attempt(
    *,
    sample: BowlingSample,
    answer_value: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> BowlingAttemptResult:
    """Package a path-label answer whose annotation is the selected cue endpoints."""

    if sample.target_path_id is None:
        raise ValueError("Bowling path sample is missing target path id")
    return BowlingAttemptResult(
        sample=sample,
        answer_gt=TypedValue(type="string", value=str(answer_value)),
        annotation_entity_ids=tuple(str(entity_id) for entity_id in sample.annotation_entity_ids),
        build_annotation=lambda rendered_context: segment_annotation_artifacts(
            rendered_context.rendered_scene.render_map["path_point_pairs_px"][str(sample.target_path_id)]
        ),
        execution_extra=dict(execution_extra or {}),
    )


def pin_bbox_set_count_attempt(
    *,
    sample: BowlingSample,
    answer_value: int,
    execution_extra: Mapping[str, Any] | None = None,
) -> BowlingAttemptResult:
    """Package a pin-count answer whose annotation is the counted pin bboxes."""

    annotation_entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    if int(answer_value) != len(annotation_entity_ids):
        raise ValueError("Bowling bbox-set count answer must match target pin count")

    def build_annotation(rendered_context: RenderedBowlingTaskContext) -> AnnotationArtifacts:
        bboxes_by_id = rendered_context.rendered_scene.render_map["entity_bboxes_px"]
        return bbox_set_annotation_artifacts(
            [bboxes_by_id[str(entity_id)] for entity_id in annotation_entity_ids]
        )

    return BowlingAttemptResult(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_entity_ids=annotation_entity_ids,
        build_annotation=build_annotation,
        execution_extra=dict(execution_extra or {}),
    )


def run_bowling_lifecycle(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    domain: str = "games",
) -> TaskOutput:
    """Run neutral Bowling query, render, prompt, trace, retry, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_bowling_scene_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(query_id),
        dict(query_probabilities),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        render_params = resolve_bowling_render_params(task_params, instance_seed=int(instance_seed))
        sample = attempt.sample
        rendered_context = render_bowling_task_scene(
            pins=sample.pins,
            path_options=sample.path_options,
            render_mode=str(objective.render_mode),
            ball_x_norm=float(sample.ball_x_norm),
            target_pin_id=sample.target_pin_id,
            target_path_id=sample.target_path_id,
            path_visible_fraction=sample.path_visible_fraction,
            style_variant=str(axes.style_variant),
            render_params=render_params,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_bowling_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=build_bowling_common_trace_params(
                axes=axes,
                extra_params={
                    **dict(objective.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_bowling_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            axes=axes,
            sample=sample,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            query_spec=query_spec,
            answer_value=str(attempt.answer_gt.value),
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
            query_id=str(query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid Bowling scene after {max_attempts} attempts")


__all__ = [
    "BowlingAttemptResult",
    "BowlingIntegerAxisSpec",
    "BowlingObjectivePlan",
    "bowling_integer_axis_spec",
    "path_segment_label_attempt",
    "pin_bbox_set_count_attempt",
    "pin_point_label_attempt",
    "resolve_bowling_integer_axis_specs",
    "run_bowling_lifecycle",
]
