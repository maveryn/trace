"""Private lifecycle plumbing for match-3 public tasks."""

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

from .shared.annotations import match3_bbox_set_annotation, match3_point_annotation, match3_point_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, SCENE_ID
from .shared.output import build_match3_common_trace_params, build_match3_trace_payload
from .shared.prompts import (
    Match3PromptContext,
    Match3PromptSlots,
    build_match3_prompt_artifacts,
    format_match3_json_examples,
)
from .shared.rendering import render_match3_scene
from .shared.sampling import (
    all_outcomes_for_board,
    answer_option_index,
    labeled_swap_options,
    make_base_board,
    outcome_histograms,
    resolve_match3_scene_axes,
    resolve_option_count_axis,
)
from .shared.state import Match3Sample, Match3SceneAxes, MoveOutcome, RenderedMatch3Scene


AnnotationBuilder = Callable[[RenderedMatch3Scene], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, Match3SceneAxes], "Match3AttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], Match3SceneAxes, Mapping[str, Any]],
    "Match3ObjectivePlan",
]
OutcomeSelector = Callable[[Sequence[MoveOutcome], Any], tuple[MoveOutcome, Sequence[MoveOutcome], Mapping[str, Any]]]


@dataclass(frozen=True)
class Match3AttemptResult:
    """Task-owned sample plus answer, prompt, trace, and annotation hooks."""

    answer_gt: TypedValue
    sample: Match3Sample
    prompt_slots: Match3PromptSlots
    build_annotation: AnnotationBuilder
    target_color_label: str = ""
    row_index: int | None = None
    col_index: int | None = None
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Match3ObjectivePlan:
    """Prepared task-owned objective for one generated instance."""

    attempt_namespace: str
    construct_attempt: AttemptBuilder


@dataclass(frozen=True)
class Match3LifecycleResult:
    """Rendered prompt/image/annotation payload returned to public task files."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_artifacts: AnnotationArtifacts
    image: Image.Image
    trace_payload: Mapping[str, Any]


class Match3SingleQueryTaskBase:
    """Private base for match-3 public tasks exposing the single sentinel."""

    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = ("single",)
    _namespace: str
    _prepare_objective: ObjectivePreparer


@lru_cache(maxsize=None)
def match3_task_defaults(public_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache generation/rendering/prompt defaults for one public task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(public_id),
    )


def select_match3_task_branch(
    *,
    public_id: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    supported_branches: tuple[str, ...],
    default_branch: str,
    namespace: str,
) -> tuple[str, Mapping[str, float], Mapping[str, Any]]:
    """Select and validate a public task's internal semantic branch."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=tuple(str(value) for value in supported_branches),
        default_query_id=str(default_branch),
        task_id=str(public_id),
        namespace=f"{namespace}.query",
    )
    return str(selected), dict(probabilities), dict(task_params)


def match3_point_set_attempt(
    *,
    answer_gt: TypedValue,
    sample: Match3Sample,
    prompt_slots: Match3PromptSlots,
    target_color_label: str = "",
    row_index: int | None = None,
    col_index: int | None = None,
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> Match3AttemptResult:
    """Package a match-3 result whose annotation is selected entity centers."""

    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    return Match3AttemptResult(
        answer_gt=answer_gt,
        sample=sample,
        prompt_slots=prompt_slots,
        build_annotation=lambda rendered: match3_point_set_annotation(
            rendered=rendered,
            entity_ids=entity_ids,
        ),
        target_color_label=str(target_color_label),
        row_index=row_index,
        col_index=col_index,
        execution_extra=dict(execution_extra or {}),
        extra_query_params=dict(extra_query_params or {}),
    )


def match3_bbox_set_attempt(
    *,
    answer_gt: TypedValue,
    sample: Match3Sample,
    prompt_slots: Match3PromptSlots,
    target_color_label: str = "",
    row_index: int | None = None,
    col_index: int | None = None,
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> Match3AttemptResult:
    """Package a match-3 result whose annotation is selected entity boxes."""

    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    return Match3AttemptResult(
        answer_gt=answer_gt,
        sample=sample,
        prompt_slots=prompt_slots,
        build_annotation=lambda rendered: match3_bbox_set_annotation(
            rendered=rendered,
            entity_ids=entity_ids,
        ),
        target_color_label=str(target_color_label),
        row_index=row_index,
        col_index=col_index,
        execution_extra=dict(execution_extra or {}),
        extra_query_params=dict(extra_query_params or {}),
    )


def match3_point_attempt(
    *,
    answer_gt: TypedValue,
    sample: Match3Sample,
    prompt_slots: Match3PromptSlots,
    execution_extra: Mapping[str, Any] | None = None,
    extra_query_params: Mapping[str, Any] | None = None,
) -> Match3AttemptResult:
    """Package a match-3 result whose annotation is exactly one entity center."""

    entity_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
    if len(entity_ids) != 1:
        raise ValueError("scalar point annotation requires exactly one entity id")
    return Match3AttemptResult(
        answer_gt=answer_gt,
        sample=sample,
        prompt_slots=prompt_slots,
        build_annotation=lambda rendered: match3_point_annotation(
            rendered=rendered,
            entity_id=str(entity_ids[0]),
        ),
        execution_extra=dict(execution_extra or {}),
        extra_query_params=dict(extra_query_params or {}),
    )


def prepare_match3_swap_option_plan(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    prompt_slots: Match3PromptSlots,
    outcome_selector: OutcomeSelector,
) -> Match3ObjectivePlan:
    """Prepare common labeled-swap plumbing around task-owned outcome semantics."""

    option_axis = resolve_option_count_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=str(namespace),
    )
    answer_index = answer_option_index(
        int(instance_seed),
        params=task_params,
        option_count=int(option_axis.value),
        namespace=str(namespace),
    )

    def construct_attempt(rng: Any, axes: Match3SceneAxes):
        """Construct one visible option set after the task selects valid outcomes."""

        board_spec = make_base_board(
            rng,
            gen_defaults=gen_defaults,
            namespace=str(namespace),
            instance_seed=int(instance_seed),
            params=task_params,
            scene_variant=str(axes.scene_variant),
        )
        outcomes = tuple(all_outcomes_for_board(board_spec.board))
        if len(outcomes) < int(option_axis.value):
            raise ValueError("not enough legal swaps")
        answer_outcome, distractors, selector_metadata = outcome_selector(tuple(outcomes), rng)
        if len(tuple(distractors)) < int(option_axis.value) - 1:
            raise ValueError("not enough swap distractors")
        option_specs = labeled_swap_options(
            option_count=int(option_axis.value),
            answer_index=int(answer_index),
            answer_outcome=answer_outcome,
            distractor_outcomes=tuple(distractors),
        )
        answer_spec = option_specs[int(answer_index)]
        sample = Match3Sample(
            scene_variant=str(axes.scene_variant),
            board=board_spec.board,
            answer=str(answer_spec.label),
            answer_type="option_letter",
            option_specs=tuple(option_specs),
            annotation_entity_ids=(str(answer_spec.entity_id),),
            metadata={
                **dict(board_spec.metadata),
                "option_count": int(option_axis.value),
                "option_count_probabilities": dict(option_axis.probabilities),
                "answer_option_index": int(answer_index),
                "answer_label": str(answer_spec.label),
                "move_count": int(len(outcomes)),
                **outcome_histograms(outcomes),
                **dict(selector_metadata),
            },
        )
        return match3_point_attempt(
            answer_gt=TypedValue(type="option_letter", value=str(sample.answer)),
            sample=sample,
            prompt_slots=prompt_slots,
        )

    return Match3ObjectivePlan(
        attempt_namespace=str(namespace),
        construct_attempt=construct_attempt,
    )


def render_match3_lifecycle(
    *,
    domain: str,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: Match3SceneAxes,
    attempt: Match3AttemptResult,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Match3LifecycleResult:
    """Run neutral render, prompt, noise, annotation, and trace assembly."""

    rendered_scene = render_match3_scene(
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
    json_example, json_example_answer_only = format_match3_json_examples(
        annotation=attempt.prompt_slots.example_annotation,
        answer=attempt.prompt_slots.example_answer,
    )
    prompt_defaults_used, prompt_artifacts = build_match3_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        context=Match3PromptContext(
            prompt_query_key=str(attempt.prompt_slots.prompt_query_key),
            object_description_key=str(attempt.prompt_slots.object_description_key),
            answer_hint_key=str(attempt.prompt_slots.answer_hint_key),
            annotation_hint_key=str(attempt.prompt_slots.annotation_hint_key),
            target_color_label=str(attempt.target_color_label),
            row_index=attempt.row_index,
            col_index=attempt.col_index,
            json_example=str(json_example),
            json_example_answer_only=str(json_example_answer_only),
        ),
        instance_seed=int(instance_seed),
    )
    prompt_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_branch),
        params=build_match3_common_trace_params(
            axes=axes,
            branch_probabilities=branch_probabilities,
            extra_params={
                "prompt_query_key": str(attempt.prompt_slots.prompt_query_key),
                **dict(attempt.sample.metadata),
                **dict(attempt.extra_query_params),
            },
        ),
    )
    trace_payload = build_match3_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=attempt.sample.annotation_entity_ids,
        axes=axes,
        sample=attempt.sample,
        rendered=rendered_scene,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=prompt_spec,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        answer_value=attempt.answer_gt.value,
        execution_extra=attempt.execution_extra,
    )
    return Match3LifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_artifacts=annotation_artifacts,
        image=image,
        trace_payload=dict(trace_payload),
    )


def run_match3_lifecycle(
    *,
    public_id: str,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    supported_branches: tuple[str, ...],
    default_branch: str,
    prepare_objective: ObjectivePreparer,
    namespace: str,
) -> TaskOutput:
    """Run shared match-3 plumbing around task-owned objective hooks."""

    selected_branch, branch_probabilities, task_params = select_match3_task_branch(
        public_id=str(public_id),
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_branches=tuple(str(value) for value in supported_branches),
        default_branch=str(default_branch),
        namespace=str(namespace),
    )
    axes = resolve_match3_scene_axes(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        namespace=str(namespace),
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_branch),
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
        lifecycle = render_match3_lifecycle(
            domain=str(domain),
            selected_branch=str(selected_branch),
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
            query_id=str(selected_branch),
        )
    raise RuntimeError(f"{public_id} failed to generate after {max_attempts} attempts")


def run_match3_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private match-3 lifecycle for a registered public task."""

    gen_defaults, render_defaults, prompt_defaults = match3_task_defaults(str(task_obj.task_id))
    return run_match3_lifecycle(
        public_id=str(task_obj.task_id),
        domain=str(task_obj.domain),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        max_attempts=int(max_attempts),
        supported_branches=tuple(str(value) for value in task_obj.supported_query_ids),
        default_branch=str(task_obj._default_branch),
        prepare_objective=task_obj._prepare_objective,
        namespace=str(task_obj._namespace),
    )


__all__ = [
    "Match3AttemptResult",
    "Match3LifecycleResult",
    "Match3ObjectivePlan",
    "Match3SingleQueryTaskBase",
    "match3_bbox_set_attempt",
    "match3_point_attempt",
    "match3_point_set_attempt",
    "prepare_match3_swap_option_plan",
    "render_match3_lifecycle",
    "run_match3_lifecycle",
    "run_match3_registered_task",
    "select_match3_task_branch",
]
