"""Scene-private lifecycle plumbing for cards public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from trace_tasks.tasks.games.shared.style_card_table import SUPPORTED_CARD_STYLE_VARIANTS
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .shared.annotations import card_bboxes_for_ids
from .shared.output import (
    build_cards_hand_count_trace_payload,
    build_cards_rule_trace_payload,
    cards_hand_count_trace_params,
    cards_rule_trace_params,
)
from .shared.prompts import build_cards_prompt_artifacts
from .shared.rendering import (
    RenderedCardsTaskContext,
    apply_cards_render_overrides,
    render_cards_task_scene,
    resolve_cards_render_params,
)
from .shared.state import SCENE_ID, RuleSample, SampledHand
from .shared.sampling import feasible_card_count_support


AnnotationBuilder = Callable[[RenderedCardsTaskContext, SampledHand], tuple[TypedValue, Mapping[str, Any]]]
HandCountAttemptBuilder = Callable[[Any], SampledHand]
HandCountPreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "CardsHandCountObjectivePlan"]
RuleAttemptBuilder = Callable[[Any], RuleSample]
RulePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "CardsRuleObjectivePlan"]


@dataclass(frozen=True)
class CardsIntegerAxis:
    """Resolved integer axis and the probabilities used for trace metadata."""

    value: int
    support: tuple[int, ...]
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class CardsHandCountAxes:
    """Resolved answer target and feasible visible-card-count axes."""

    target: CardsIntegerAxis
    card_count: CardsIntegerAxis


@dataclass(frozen=True)
class CardsHandCountObjectivePlan:
    """Task-owned hand-count construction hooks for one generated instance."""

    attempt_namespace: str
    prompt_query_key: str
    target_answer: int
    target_answer_support: tuple[int, ...]
    target_answer_probabilities: Mapping[str, float]
    card_count: int
    card_count_support: tuple[int, ...]
    card_count_probabilities: Mapping[str, float]
    card_ordering: str
    center_label_mode: str
    construct_attempt: HandCountAttemptBuilder
    build_annotation: AnnotationBuilder
    show_continuation_cue: bool = False


@dataclass(frozen=True)
class CardsRuleObjectivePlan:
    """Task-owned labelled-rule construction hooks for one generated instance."""

    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    construct_attempt: RuleAttemptBuilder
    scalar_annotation: bool = False


def resolve_cards_style_variant(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, Dict[str, float]]:
    """Resolve the shared card-table visual style axis for one public task."""

    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=SUPPORTED_CARD_STYLE_VARIANTS,
    )


def resolve_cards_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str = "balanced_option_count_sampling",
) -> CardsIntegerAxis:
    """Resolve a task-owned integer sampling axis using the repo standard policy."""

    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    axis_params = dict(params)
    axis_params[f"{explicit_key}_support"] = [int(value) for value in support]
    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=axis_params,
        gen_defaults=gen_defaults,
        support_key=f"{explicit_key}_support",
        explicit_key=str(explicit_key),
        fallback_support=support,
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return CardsIntegerAxis(
        value=int(value),
        support=tuple(int(item) for item in support),
        probabilities=dict(probabilities),
    )


def resolve_cards_hand_count_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    hand_kind: str,
    target_support_key: str,
    target_fallback_support: Sequence[int],
    card_count_support_key: str,
    card_count_fallback_support: Sequence[int],
    target_namespace: str,
    card_count_namespace: str,
) -> CardsHandCountAxes:
    """Resolve a task target and then restrict visible-card count to feasible values."""

    target_axis = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(target_support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in target_fallback_support),
        namespace=str(target_namespace),
        balanced_flag_key="balanced_target_answer_sampling",
    )
    raw_card_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(card_count_support_key),
        fallback=tuple(int(value) for value in card_count_fallback_support),
    )
    feasible_support = feasible_card_count_support(
        hand_kind=str(hand_kind),
        target_answer=int(target_axis.value),
        raw_support=raw_card_support,
    )
    if not feasible_support:
        raise ValueError(f"no feasible card_count values remain for {hand_kind} at target {target_axis.value}")
    card_axis = resolve_cards_integer_axis(
        instance_seed=int(instance_seed),
        params={**dict(params), "card_count_support": [int(value) for value in feasible_support]},
        gen_defaults=gen_defaults,
        support_key="card_count_support",
        explicit_key="card_count",
        fallback_support=feasible_support,
        namespace=str(card_count_namespace),
        balanced_flag_key="balanced_card_count_sampling",
    )
    return CardsHandCountAxes(target=target_axis, card_count=card_axis)


def resolve_cards_boolean_flag(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: bool,
) -> bool:
    """Resolve a task-owned boolean flag from params, config defaults, or fallback."""

    return bool(dict(params).get(str(key), group_default(gen_defaults, str(key), bool(fallback))))


def bbox_set_for_cards(
    rendered_context: RenderedCardsTaskContext,
    sampled_hand: SampledHand,
) -> tuple[TypedValue, Mapping[str, Any]]:
    """Build an unordered card bbox annotation from a task-owned card-id set."""

    annotation_bboxes = card_bboxes_for_ids(
        rendered_context.rendered_scene.render_map,
        sampled_hand.annotation_card_ids,
    )
    annotation_gt = TypedValue(type="bbox_set", value=[list(bbox) for bbox in annotation_bboxes])
    return annotation_gt, {"bbox_set": [list(bbox) for bbox in annotation_bboxes]}


def build_cards_hand_count_objective_plan(
    *,
    attempt_namespace: str,
    prompt_query_key: str,
    axes: CardsHandCountAxes,
    card_ordering: str,
    construct_attempt: HandCountAttemptBuilder,
    build_annotation: AnnotationBuilder,
    center_label_mode: str | None = None,
    center_label_key: str | None = None,
    params: Mapping[str, Any] | None = None,
    gen_defaults: Mapping[str, Any] | None = None,
    show_continuation_cue: bool = False,
) -> CardsHandCountObjectivePlan:
    """Package task-owned hand-count hooks into the common lifecycle contract."""

    resolved_center_label_mode = center_label_mode
    if resolved_center_label_mode is None:
        if center_label_key is None or gen_defaults is None:
            raise ValueError("center_label_mode or center_label_key/gen_defaults must be provided")
        source_params = dict(params or {})
        resolved_center_label_mode = str(
            source_params.get("center_label_mode", group_default(gen_defaults, str(center_label_key), "suit_symbol"))
        )
    return CardsHandCountObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(prompt_query_key),
        target_answer=int(axes.target.value),
        target_answer_support=axes.target.support,
        target_answer_probabilities=dict(axes.target.probabilities),
        card_count=int(axes.card_count.value),
        card_count_support=axes.card_count.support,
        card_count_probabilities=dict(axes.card_count.probabilities),
        card_ordering=str(card_ordering),
        center_label_mode=str(resolved_center_label_mode),
        construct_attempt=construct_attempt,
        build_annotation=build_annotation,
        show_continuation_cue=bool(show_continuation_cue),
    )


def build_cards_rule_objective_plan(
    *,
    attempt_namespace: str,
    prompt_query_key: str,
    construct_attempt: RuleAttemptBuilder,
    query_params: Mapping[str, Any] | None = None,
    scalar_annotation: bool = False,
) -> CardsRuleObjectivePlan:
    """Package task-owned rule hooks into the common lifecycle contract."""

    return CardsRuleObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(prompt_query_key),
        query_params=dict(query_params or {}),
        construct_attempt=construct_attempt,
        scalar_annotation=bool(scalar_annotation),
    )


def run_cards_hand_count_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: HandCountPreparer,
):
    """Run query, style, render, prompt, trace, and retry plumbing for card counts."""

    runtime_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    style_variant, style_probabilities = resolve_cards_style_variant(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(runtime_query_id),
        dict(query_probabilities),
    )
    last_error: ValueError | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sampled_hand = objective.construct_attempt(rng)
        except ValueError as exc:
            last_error = exc
            continue

        render_params = resolve_cards_render_params(task_params, instance_seed=int(instance_seed))
        render_params = apply_cards_render_overrides(
            render_params,
            {},
            center_label_mode=str(objective.center_label_mode),
        )
        rendered_context = render_cards_task_scene(
            cards=sampled_hand.cards,
            scene_variant="multi_row",
            style_variant=str(style_variant),
            params=task_params,
            instance_seed=int(instance_seed),
            render_params=render_params,
            show_continuation_cue=bool(objective.show_continuation_cue),
        )
        annotation_gt, projected_annotation = objective.build_annotation(rendered_context, sampled_hand)
        prompt_defaults, prompt_artifacts = build_cards_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(objective.target_answer))
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(runtime_query_id),
            params=cards_hand_count_trace_params(
                sample=sampled_hand,
                rendered_context=rendered_context,
                hand_kind=str(objective.prompt_query_key),
                scene_variant="multi_row",
                style_variant=str(style_variant),
                target_answer=int(objective.target_answer),
                target_answer_support=objective.target_answer_support,
                target_answer_probabilities=objective.target_answer_probabilities,
                card_count=int(objective.card_count),
                card_count_support=objective.card_count_support,
                card_count_probabilities=objective.card_count_probabilities,
                style_variant_probabilities=style_probabilities,
                card_ordering=str(objective.card_ordering),
                query_id_probabilities=query_probabilities,
            ),
        )
        trace_payload = build_cards_hand_count_trace_payload(
            annotation_gt=annotation_gt,
            projected_annotation=projected_annotation,
            sample=sampled_hand,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            query_spec=query_spec,
            hand_kind=str(objective.prompt_query_key),
            scene_variant="multi_row",
            style_variant=str(style_variant),
            target_answer=int(objective.target_answer),
            target_answer_support=objective.target_answer_support,
            target_answer_probabilities=objective.target_answer_probabilities,
            card_count=int(objective.card_count),
            card_count_support=objective.card_count_support,
            card_count_probabilities=objective.card_count_probabilities,
            style_variant_probabilities=style_probabilities,
            card_ordering=str(objective.card_ordering),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(runtime_query_id),
        )
    raise RuntimeError(f"{task_id} failed to generate a valid cards hand scene after {max_attempts} attempts") from last_error


def run_cards_rule_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: RulePreparer,
):
    """Run query, render, prompt, annotation, trace, and retry plumbing for card-label rules."""

    runtime_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    style_variant, style_probabilities = resolve_cards_style_variant(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(runtime_query_id),
        dict(query_probabilities),
    )
    last_error: ValueError | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng)
        except ValueError as exc:
            last_error = exc
            continue
        sample = replace(sample, pattern_kind=str(objective.prompt_query_key))

        render_params = resolve_cards_render_params(task_params, instance_seed=int(instance_seed))
        render_params = apply_cards_render_overrides(
            render_params,
            sample.render_overrides,
            center_label_mode=str(sample.center_label_mode),
            max_cards_per_row=int(sample.cards_per_row),
        )
        rendered_context = render_cards_task_scene(
            cards=sample.cards,
            scene_variant=str(sample.scene_variant),
            style_variant=str(style_variant),
            params=task_params,
            instance_seed=int(instance_seed),
            render_params=render_params,
            show_continuation_cue=False,
            row_card_counts=sample.row_card_counts if sample.row_card_counts else None,
        )
        annotation_bboxes = card_bboxes_for_ids(
            rendered_context.rendered_scene.render_map,
            sample.annotation_card_ids,
        )
        if objective.scalar_annotation:
            if len(annotation_bboxes) != 1:
                raise RuntimeError(f"{objective.prompt_query_key} scalar annotation must contain exactly one card bbox")
            annotation_gt = TypedValue(type="bbox", value=list(annotation_bboxes[0]))
        else:
            annotation_gt = TypedValue(type="bbox_set", value=[list(bbox) for bbox in annotation_bboxes])
        prompt_defaults, prompt_artifacts = build_cards_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            dynamic_slots={str(key): str(value) for key, value in sample.prompt_slots.items()},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="string", value=str(sample.answer))
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(runtime_query_id),
            params=cards_rule_trace_params(
                sample=sample,
                rendered_context=rendered_context,
                style_variant=str(style_variant),
                style_variant_probabilities=style_probabilities,
                extra_trace_params={
                    "query_id_probabilities": dict(query_probabilities),
                    **dict(objective.query_params),
                },
            ),
        )
        trace_payload = build_cards_rule_trace_payload(
            annotation_gt=annotation_gt,
            sample=sample,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            query_spec=query_spec,
            style_variant=str(style_variant),
            style_variant_probabilities=style_probabilities,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(runtime_query_id),
        )
    raise RuntimeError(f"{task_id} failed to generate a valid cards rule scene after {max_attempts} attempts") from last_error


__all__ = [
    "CardsHandCountObjectivePlan",
    "CardsHandCountAxes",
    "CardsIntegerAxis",
    "CardsRuleObjectivePlan",
    "bbox_set_for_cards",
    "build_cards_hand_count_objective_plan",
    "build_cards_rule_objective_plan",
    "resolve_cards_integer_axis",
    "resolve_cards_boolean_flag",
    "resolve_cards_hand_count_axes",
    "resolve_cards_style_variant",
    "run_cards_hand_count_lifecycle",
    "run_cards_rule_lifecycle",
]
