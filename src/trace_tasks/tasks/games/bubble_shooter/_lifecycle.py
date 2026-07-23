"""Scene-private lifecycle plumbing for Bubble-shooter public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
)
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import (
    resolve_integer_choice,
    resolve_integer_support,
)

from .shared.defaults import SCENE_ID
from .shared.labels import resolve_bubble_shooter_label_choice
from .shared.output import (
    build_bubble_shooter_trace_payload,
    common_bubble_shooter_trace_params,
)
from .shared.prompts import build_bubble_shooter_prompt_artifacts
from .shared.rendering import (
    RenderedBubbleShooterTaskContext,
    render_bubble_shooter_task_scene,
)
from .shared.sampling import (
    ResolvedBubbleShooterBoardAxes,
    ResolvedBubbleShooterSceneAxes,
    bubble_entity_ids_for_coords,
    resolve_bubble_shooter_board_axes,
    resolve_bubble_shooter_render_params,
    resolve_bubble_shooter_scene_axes,
)
from .shared.state import BubbleShooterState, Coord, landing_option_entity_id

_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
    )
)

AnnotationBuilder = Callable[[RenderedBubbleShooterTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[
    [Any, ResolvedBubbleShooterSceneAxes], "BubbleShooterAttemptResult"
]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float]], "BubbleShooterObjectivePlan"
]
IntegerOutcomeSampler = Callable[
    [Any, ResolvedBubbleShooterSceneAxes, ResolvedBubbleShooterBoardAxes, int],
    BubbleShooterState,
]
OutcomeCoords = Callable[[BubbleShooterState], Sequence[Coord]]
LandingTargetSampler = Callable[
    [
        Any,
        ResolvedBubbleShooterSceneAxes,
        ResolvedBubbleShooterBoardAxes,
        str,
        Sequence[str],
        int,
    ],
    BubbleShooterState,
]


@dataclass(frozen=True)
class BubbleShooterAttemptResult:
    """Task-owned result of one constructed Bubble-shooter attempt."""

    state: BubbleShooterState
    answer_gt: TypedValue
    annotation_entity_ids: tuple[str, ...]
    build_annotation: AnnotationBuilder
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BubbleShooterObjectivePlan:
    """Prepared task-owned objective hooks for one generated instance."""

    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder


def resolve_bubble_shooter_board_axis_specs(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    row_count_support: tuple[int, ...],
    col_count_support: tuple[int, ...],
) -> tuple[ResolvedBubbleShooterBoardAxes, dict[str, Any]]:
    """Resolve task-declared board axes and return standard trace params."""

    board_axis_seed = hash64(int(instance_seed), f"{task_id}.board_axes")
    board_axes = resolve_bubble_shooter_board_axes(
        int(board_axis_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        row_count_support=tuple(int(value) for value in row_count_support),
        col_count_support=tuple(int(value) for value in col_count_support),
    )
    return board_axes, {
        "row_count_support": [int(value) for value in board_axes.rows.support],
        "row_count_probabilities": dict(board_axes.rows.probabilities),
        "col_count_support": [int(value) for value in board_axes.cols.support],
        "col_count_probabilities": dict(board_axes.cols.probabilities),
    }


def bbox_set_attempt(
    *,
    state: BubbleShooterState,
    answer_gt: TypedValue,
    annotation_entity_ids: tuple[str, ...],
    execution_extra: Mapping[str, Any] | None = None,
) -> BubbleShooterAttemptResult:
    """Package an answer whose annotation is the bbox set for scene entities."""

    resolved_entity_ids = tuple(str(entity_id) for entity_id in annotation_entity_ids)
    return BubbleShooterAttemptResult(
        state=state,
        answer_gt=answer_gt,
        annotation_entity_ids=resolved_entity_ids,
        build_annotation=lambda rendered_context: bbox_set_annotation_artifacts(
            [
                rendered_context.rendered_scene.render_map["entity_bboxes_px"][
                    str(entity_id)
                ]
                for entity_id in resolved_entity_ids
            ]
        ),
        execution_extra=dict(execution_extra or {}),
    )


def bbox_attempt(
    *,
    state: BubbleShooterState,
    answer_gt: TypedValue,
    annotation_entity_id: str,
    execution_extra: Mapping[str, Any] | None = None,
) -> BubbleShooterAttemptResult:
    """Package an answer whose annotation is one scene-entity bbox."""

    resolved_entity_id = str(annotation_entity_id)
    return BubbleShooterAttemptResult(
        state=state,
        answer_gt=answer_gt,
        annotation_entity_ids=(resolved_entity_id,),
        build_annotation=lambda rendered_context: bbox_annotation_artifacts(
            rendered_context.rendered_scene.render_map["entity_bboxes_px"][
                resolved_entity_id
            ]
        ),
        execution_extra=dict(execution_extra or {}),
    )


def prepare_integer_outcome_objective(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    row_count_support: tuple[int, ...],
    col_count_support: tuple[int, ...],
    support_key: str,
    fallback_support: tuple[int, ...],
    prompt_query_key: str,
    target_namespace: str,
    attempt_namespace: str,
    sample_state: IntegerOutcomeSampler,
    outcome_coords: OutcomeCoords,
) -> BubbleShooterObjectivePlan:
    """Prepare a count objective over one sampled Bubble-shooter outcome coordinate set."""

    board_axes, board_query_params = resolve_bubble_shooter_board_axis_specs(
        instance_seed=int(instance_seed),
        task_params=params,
        gen_defaults=gen_defaults,
        task_id=str(task_id),
        row_count_support=tuple(int(value) for value in row_count_support),
        col_count_support=tuple(int(value) for value in col_count_support),
    )
    target_value, target_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(target_namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )

    def construct_attempt(
        rng, scene_axes: ResolvedBubbleShooterSceneAxes
    ) -> BubbleShooterAttemptResult:
        state = sample_state(rng, scene_axes, board_axes, int(target_value))
        if state.shooter_color_key != state.outcome.color_key:
            raise ValueError("Bubble-shooter count state has mismatched shooter color")
        coords = tuple(outcome_coords(state))
        return bbox_set_attempt(
            state=state,
            answer_gt=TypedValue(type="integer", value=int(len(coords))),
            annotation_entity_ids=bubble_entity_ids_for_coords(coords),
        )

    return BubbleShooterObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(prompt_query_key),
        query_params={
            "target_answer": int(target_value),
            "target_answer_support": [int(value) for value in target_support],
            "target_answer_probabilities": dict(target_probabilities),
            **dict(board_query_params),
        },
        construct_attempt=construct_attempt,
    )


def prepare_landing_target_label_objective(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    task_id: str,
    row_count_support: tuple[int, ...],
    col_count_support: tuple[int, ...],
    prompt_query_key: str,
    target_label_support_key: str,
    fallback_target_label_support: tuple[str, ...],
    displayed_option_labels: tuple[str, ...],
    positive_pop_count_support_key: str,
    fallback_positive_pop_count_support: tuple[int, ...],
    target_namespace: str,
    pop_count_namespace: str,
    attempt_namespace: str,
    sample_state: LandingTargetSampler,
) -> BubbleShooterObjectivePlan:
    """Prepare a label objective over displayed Bubble-shooter landing targets."""

    board_axes, board_query_params = resolve_bubble_shooter_board_axis_specs(
        instance_seed=int(instance_seed),
        task_params=params,
        gen_defaults=gen_defaults,
        task_id=str(task_id),
        row_count_support=tuple(int(value) for value in row_count_support),
        col_count_support=tuple(int(value) for value in col_count_support),
    )
    target_label, target_probabilities, target_support = (
        resolve_bubble_shooter_label_choice(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=gen_defaults,
            support_key=str(target_label_support_key),
            explicit_key="target_label",
            fallback_support=tuple(
                str(value) for value in fallback_target_label_support
            ),
            namespace=str(target_namespace),
            allowed_labels=tuple(str(value) for value in displayed_option_labels),
        )
    )
    positive_pop_count, positive_pop_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(positive_pop_count_support_key),
        explicit_key="positive_pop_count",
        fallback_support=tuple(
            int(value) for value in fallback_positive_pop_count_support
        ),
        namespace=str(pop_count_namespace),
        balanced_flag_key="balanced_positive_pop_count_sampling",
        namespace_support_permutation=True,
    )
    positive_pop_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(positive_pop_count_support_key),
        fallback=tuple(int(value) for value in fallback_positive_pop_count_support),
    )

    def construct_attempt(
        rng, scene_axes: ResolvedBubbleShooterSceneAxes
    ) -> BubbleShooterAttemptResult:
        state = sample_state(
            rng,
            scene_axes,
            board_axes,
            str(target_label),
            tuple(str(value) for value in displayed_option_labels),
            int(positive_pop_count),
        )
        answer_options = [
            option for option in state.landing_option_specs if option.is_answer
        ]
        if len(answer_options) != 1 or str(answer_options[0].label) != str(
            target_label
        ):
            raise ValueError("Bubble-shooter landing-target state has ambiguous answer")
        return bbox_attempt(
            state=state,
            answer_gt=TypedValue(type="string", value=str(target_label)),
            annotation_entity_id=landing_option_entity_id(str(target_label)),
            execution_extra={
                "target_label": str(target_label),
                "displayed_landing_option_labels": [
                    str(label) for label in displayed_option_labels
                ],
                "positive_pop_count": int(positive_pop_count),
            },
        )

    return BubbleShooterObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(prompt_query_key),
        query_params={
            "target_label": str(target_label),
            "target_label_support": [str(value) for value in target_support],
            "target_label_probabilities": dict(target_probabilities),
            "landing_option_labels": [str(value) for value in displayed_option_labels],
            "positive_pop_count": int(positive_pop_count),
            "positive_pop_count_support": [
                int(value) for value in positive_pop_support
            ],
            "positive_pop_count_probabilities": dict(positive_pop_probabilities),
            **dict(board_query_params),
        },
        construct_attempt=construct_attempt,
    )


def run_bubble_shooter_lifecycle(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    domain: str = "games",
) -> TaskOutput:
    """Run common Bubble-shooter query, render, prompt, trace, and output plumbing."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    scene_axes = resolve_bubble_shooter_scene_axes(
        int(instance_seed), params=task_params
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query),
        dict(query_probabilities),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            attempt = objective.construct_attempt(rng, scene_axes)
        except ValueError:
            continue

        render_params = resolve_bubble_shooter_render_params(
            task_params, instance_seed=int(instance_seed)
        )
        state = attempt.state
        rendered_context = render_bubble_shooter_task_scene(
            board=state.board,
            landing_coord=state.landing_coord,
            shooter_color_key=state.shooter_color_key,
            option_specs=state.option_specs,
            landing_option_specs=state.landing_option_specs,
            scene_variant=str(scene_axes.scene_variant),
            style_variant=str(scene_axes.style_variant),
            render_params=render_params,
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        _prompt_defaults, prompt_artifacts = build_bubble_shooter_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query),
            params=common_bubble_shooter_trace_params(
                scene_axes,
                state,
                extra_params={
                    **dict(objective.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_bubble_shooter_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=attempt.annotation_entity_ids,
            axes=scene_axes,
            state=state,
            rendered_context=rendered_context,
            prompt_artifacts=prompt_artifacts,
            query_spec=query_spec,
            execution_extra={
                "answer": attempt.answer_gt.value,
                **dict(attempt.execution_extra),
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

    raise RuntimeError(
        f"{task_id} failed to generate a valid Bubble-shooter scene after {max_attempts} attempts"
    )


__all__ = [
    "BubbleShooterAttemptResult",
    "BubbleShooterObjectivePlan",
    "bbox_attempt",
    "bbox_set_attempt",
    "prepare_integer_outcome_objective",
    "prepare_landing_target_label_objective",
    "resolve_bubble_shooter_board_axis_specs",
    "run_bubble_shooter_lifecycle",
]
