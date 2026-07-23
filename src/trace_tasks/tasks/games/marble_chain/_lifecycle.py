"""Private lifecycle plumbing for marble-chain public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import marble_bbox_set_annotation, marble_point_annotation, marble_point_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .shared.output import build_marble_common_trace_params, build_marble_trace_payload
from .shared.prompts import (
    MarblePromptContext,
    MarblePromptSlots,
    build_marble_prompt_artifacts,
    format_marble_json_examples,
)
from .shared.rendering import render_marble_scene
from .shared.rules import insertion_point_annotation_ids
from .shared.sampling import (
    DisplayValidator,
    SlotGroupBuilder,
    StateSlotGroupBuilder,
    answer_option_label,
    resolve_chain_length_axis,
    resolve_color_count_axis,
    resolve_marble_scene_axes,
    resolve_option_count_axis,
    sample_direction_option_scene_from_state_groups,
)
from .shared.state import MarbleSample, MarbleSceneAxes, RenderedMarbleScene


AnnotationBuilder = Callable[[RenderedMarbleScene], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, MarbleSceneAxes], "MarbleAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], MarbleSceneAxes, Mapping[str, Any]],
    "MarbleObjectivePlan",
]


@dataclass(frozen=True)
class MarbleAttemptResult:
    """Task-owned sample plus answer, prompt, trace, and annotation hooks."""

    answer_gt: TypedValue
    sample: MarbleSample
    prompt_slots: MarblePromptSlots
    build_annotation: AnnotationBuilder
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarbleObjectivePlan:
    """Prepared task-owned objective for one generated instance."""

    attempt_namespace: str
    construct_attempt: AttemptBuilder


@dataclass(frozen=True)
class MarbleLifecycleResult:
    """Rendered prompt/image/annotation payload returned to public task files."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_artifacts: AnnotationArtifacts
    image: Image.Image
    trace_payload: Mapping[str, Any]


class MarbleSingleQueryTaskBase:
    """Private base for marble-chain public tasks exposing the single sentinel."""

    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = ("single",)
    _namespace: str
    _prepare_objective: ObjectivePreparer


@lru_cache(maxsize=None)
def marble_task_defaults(public_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache generation/rendering/prompt defaults for one public task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(public_id),
    )


def select_marble_single_query(
    *,
    public_id: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    namespace: str,
) -> tuple[str, Mapping[str, float], Mapping[str, Any]]:
    """Select and validate the public single-query sentinel for a task."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=("single",),
        default_query_id="single",
        task_id=str(public_id),
        namespace=f"{namespace}.query",
    )
    return str(selected), dict(probabilities), dict(task_params)


def marble_point_set_attempt(
    *,
    answer_gt: TypedValue,
    sample: MarbleSample,
    prompt_slots: MarblePromptSlots,
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> MarbleAttemptResult:
    """Package a marble-chain result whose annotation is selected entity centers."""

    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    return MarbleAttemptResult(
        answer_gt=answer_gt,
        sample=sample,
        prompt_slots=prompt_slots,
        build_annotation=lambda rendered: marble_point_set_annotation(
            rendered=rendered,
            entity_ids=entity_ids,
        ),
        execution_extra=dict(execution_extra or {}),
        extra_query_params=dict(extra_query_params or {}),
    )


def marble_bbox_set_attempt(
    *,
    answer_gt: TypedValue,
    sample: MarbleSample,
    prompt_slots: MarblePromptSlots,
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> MarbleAttemptResult:
    """Package a marble-chain result whose annotation is selected marble boxes."""

    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    return MarbleAttemptResult(
        answer_gt=answer_gt,
        sample=sample,
        prompt_slots=prompt_slots,
        build_annotation=lambda rendered: marble_bbox_set_annotation(
            rendered=rendered,
            entity_ids=entity_ids,
        ),
        execution_extra=dict(execution_extra or {}),
        extra_query_params=dict(extra_query_params or {}),
    )


def marble_point_attempt(
    *,
    answer_gt: TypedValue,
    sample: MarbleSample,
    prompt_slots: MarblePromptSlots,
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> MarbleAttemptResult:
    """Package a marble-chain result whose annotation is one selected entity center."""

    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    if len(entity_ids) != 1:
        raise ValueError("marble scalar point annotation requires exactly one entity id")
    entity_id = str(entity_ids[0])
    return MarbleAttemptResult(
        answer_gt=answer_gt,
        sample=sample,
        prompt_slots=prompt_slots,
        build_annotation=lambda rendered: marble_point_annotation(
            rendered=rendered,
            entity_id=entity_id,
        ),
        execution_extra=dict(execution_extra or {}),
        extra_query_params=dict(extra_query_params or {}),
    )


def _prepare_marble_direction_option_plan_from_state_groups(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    prompt_slots: MarblePromptSlots,
    slot_group_builder: StateSlotGroupBuilder,
    display_validator: DisplayValidator,
    target_pop_count: int | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> MarbleObjectivePlan:
    """Prepare common labeled-arrow plumbing around state-aware slot semantics."""

    chain_axis = resolve_chain_length_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=str(namespace),
    )
    color_axis = resolve_color_count_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=str(namespace),
    )
    option_axis = resolve_option_count_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=str(namespace),
    )
    resolved_answer_label = answer_option_label(
        int(instance_seed),
        params=task_params,
        option_count=int(option_axis.value),
    )
    trace_params = {
        "chain_length": int(chain_axis.value),
        "chain_length_support": [int(value) for value in chain_axis.support],
        "chain_length_probabilities": dict(chain_axis.probabilities),
        "color_count": int(color_axis.value),
        "color_count_support": [int(value) for value in color_axis.support],
        "color_count_probabilities": dict(color_axis.probabilities),
        "option_count": int(option_axis.value),
        "option_count_support": [int(value) for value in option_axis.support],
        "option_count_probabilities": dict(option_axis.probabilities),
        "answer_option_label": str(resolved_answer_label),
        **dict(extra_params or {}),
    }

    def construct_attempt(rng: Any, axes: MarbleSceneAxes):
        """Bind one labeled shot option, its answer, and insertion-point annotation."""

        chain_colors, shooter_color, option_specs = sample_direction_option_scene_from_state_groups(
            rng,
            chain_length=int(chain_axis.value),
            color_count=int(color_axis.value),
            option_count=int(option_axis.value),
            answer_label=str(resolved_answer_label),
            slot_group_builder=slot_group_builder,
            display_validator=display_validator,
        )
        answer_spec = next(option for option in option_specs if bool(option.is_answer))
        sample = MarbleSample(
            scene_variant=str(axes.scene_variant),
            chain_colors=tuple(chain_colors),
            shooter_color=str(shooter_color),
            answer=str(resolved_answer_label),
            answer_type="option_letter",
            option_specs=tuple(option_specs),
            marked_slot_index=None,
            marked_outcome=None,
            target_pop_count=None if target_pop_count is None else int(target_pop_count),
            annotation_entity_ids=insertion_point_annotation_ids(str(answer_spec.entity_id)),
            metadata=dict(trace_params),
        )
        return marble_point_attempt(
            answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
            sample=sample,
            prompt_slots=prompt_slots,
        )

    return MarbleObjectivePlan(
        attempt_namespace=str(namespace),
        construct_attempt=construct_attempt,
    )


def prepare_marble_direction_option_plan(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    prompt_slots: MarblePromptSlots,
    slot_group_builder: SlotGroupBuilder,
    display_validator: DisplayValidator,
    target_pop_count: int | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> MarbleObjectivePlan:
    """Prepare common labeled-arrow plumbing around task-owned slot semantics."""

    return _prepare_marble_direction_option_plan_from_state_groups(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        prompt_slots=prompt_slots,
        slot_group_builder=lambda _chain_colors, outcomes: slot_group_builder(outcomes),
        display_validator=display_validator,
        target_pop_count=target_pop_count,
        extra_params=extra_params,
    )


def prepare_marble_state_direction_option_plan(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    prompt_slots: MarblePromptSlots,
    slot_group_builder: StateSlotGroupBuilder,
    display_validator: DisplayValidator,
    target_pop_count: int | None = None,
    extra_params: Mapping[str, Any] | None = None,
) -> MarbleObjectivePlan:
    """Prepare labeled-arrow plumbing for objectives that inspect chain colors."""

    return _prepare_marble_direction_option_plan_from_state_groups(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
        prompt_slots=prompt_slots,
        slot_group_builder=slot_group_builder,
        display_validator=display_validator,
        target_pop_count=target_pop_count,
        extra_params=extra_params,
    )


def render_marble_lifecycle(
    *,
    domain: str,
    selected_query_id: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: MarbleSceneAxes,
    attempt: MarbleAttemptResult,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> MarbleLifecycleResult:
    """Run neutral render, prompt, noise, annotation, and trace assembly."""

    rendered_scene = render_marble_scene(
        sample=attempt.sample,
        instance_seed=int(instance_seed),
        params=task_params,
        style_variant=str(axes.style_variant),
        render_defaults=render_defaults,
        namespace=str(namespace),
    )
    annotation_artifacts = attempt.build_annotation(rendered_scene)
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    json_example, json_example_answer_only = format_marble_json_examples(
        annotation=attempt.prompt_slots.example_annotation,
        answer=attempt.prompt_slots.example_answer,
    )
    prompt_defaults_used, prompt_artifacts = build_marble_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        context=MarblePromptContext(
            prompt_query_key=str(attempt.prompt_slots.prompt_query_key),
            answer_hint_key=str(attempt.prompt_slots.answer_hint_key),
            annotation_hint_key=str(attempt.prompt_slots.annotation_hint_key),
            target_pop_count=attempt.sample.target_pop_count,
            json_example=str(json_example),
            json_example_answer_only=str(json_example_answer_only),
        ),
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=build_marble_common_trace_params(
            axes=axes,
            branch_probabilities=branch_probabilities,
            extra_params={
                "prompt_query_key": str(attempt.prompt_slots.prompt_query_key),
                **dict(attempt.sample.metadata),
                **dict(attempt.extra_query_params),
            },
        ),
    )
    trace_payload = build_marble_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=attempt.sample.annotation_entity_ids,
        axes=axes,
        sample=attempt.sample,
        rendered=rendered_scene,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=query_spec,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        answer_value=attempt.answer_gt.value,
        execution_extra=attempt.execution_extra,
    )
    return MarbleLifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_artifacts=annotation_artifacts,
        image=image,
        trace_payload=dict(trace_payload),
    )


def run_marble_lifecycle(
    *,
    public_id: str,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    namespace: str,
) -> TaskOutput:
    """Run shared marble-chain plumbing around task-owned objective hooks."""

    selected_query_id, branch_probabilities, task_params = select_marble_single_query(
        public_id=str(public_id),
        instance_seed=int(instance_seed),
        params=dict(params),
        namespace=str(namespace),
    )
    axes = resolve_marble_scene_axes(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=str(namespace),
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query_id),
        dict(branch_probabilities),
        axes,
        gen_defaults,
    )
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue
        lifecycle = render_marble_lifecycle(
            domain=str(domain),
            selected_query_id=str(selected_query_id),
            branch_probabilities=branch_probabilities,
            task_params=task_params,
            axes=axes,
            attempt=attempt,
            prompt_defaults=prompt_defaults,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        return TaskOutput(
            prompt=str(lifecycle.prompt),
            prompt_variants=dict(lifecycle.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=lifecycle.annotation_artifacts.annotation_gt,
            image=lifecycle.image,
            image_id="img0",
            trace_payload=dict(lifecycle.trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
        )
    raise RuntimeError(f"{public_id} failed to generate after {max_attempts} attempts")


def run_marble_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private marble-chain lifecycle for a registered public task."""

    gen_defaults, render_defaults, prompt_defaults = marble_task_defaults(str(task_obj.task_id))
    return run_marble_lifecycle(
        public_id=str(task_obj.task_id),
        domain=str(task_obj.domain),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        max_attempts=int(max_attempts),
        prepare_objective=task_obj._prepare_objective,
        namespace=str(task_obj._namespace),
    )


__all__ = [
    "MarbleAttemptResult",
    "MarbleLifecycleResult",
    "MarbleObjectivePlan",
    "MarbleSingleQueryTaskBase",
    "marble_bbox_set_attempt",
    "marble_point_attempt",
    "marble_point_set_attempt",
    "prepare_marble_direction_option_plan",
    "prepare_marble_state_direction_option_plan",
    "render_marble_lifecycle",
    "run_marble_lifecycle",
    "run_marble_registered_task",
    "select_marble_single_query",
]
