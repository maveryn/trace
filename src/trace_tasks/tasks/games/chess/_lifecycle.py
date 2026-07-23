"""Scene-private lifecycle plumbing for Chess public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.piece_board_rules import coord_to_cell_id
from trace_tasks.tasks.games.shared.piece_board_rules import color_name
from trace_tasks.tasks.games.shared.piece_board_rules import piece_name
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import (
    bbox_set_for_entities,
    move_point_map,
    projected_bbox_payload,
    projected_point_map_payload,
)
from .shared.prompts import build_chess_prompt_artifacts
from .shared.rendering import RenderedChessTaskContext
from .shared.rendering import render_chess_task_scene
from .shared.sampling import (
    resolve_chess_scene_axes,
    resolve_integer_axis,
    resolve_string_axis,
    resolve_target_answer,
    sample_marked_piece_destination_scene,
    sample_piece_count_scene,
)
from .shared.state import (
    PIECE_COUNT_COLOR_SUPPORT,
    PIECE_COUNT_KIND_SUPPORT,
    SCENE_ID,
    ChessCheckmateSample,
    ChessSceneSample,
    ResolvedChessSceneAxes,
    piece_kind_plural,
)


ChessSample = ChessSceneSample | ChessCheckmateSample
AttemptBuilder = Callable[[Any, ResolvedChessSceneAxes], ChessSample]
RenderBuilder = Callable[[ChessSample, Mapping[str, Any], int], RenderedChessTaskContext]
AnswerBuilder = Callable[[ChessSample], TypedValue]
AnnotationBuilder = Callable[[ChessSample, RenderedChessTaskContext], "ChessAnnotationArtifacts"]
TraceBuilder = Callable[[ChessSample, RenderedChessTaskContext], dict[str, Any]]
ExecutionExtraBuilder = Callable[[ChessSample], Mapping[str, Any]]
PromptDynamicSlotBuilder = Callable[[ChessSample], Mapping[str, Any]]
QueryParamBuilder = Callable[[ChessSample], Mapping[str, Any]]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "ChessObjectivePlan"]
CountSampleConstructor = Callable[[Any, ResolvedChessSceneAxes, int], ChessSceneSample]
BadgeBuilder = Callable[[ChessSceneSample], str]


@dataclass(frozen=True)
class ChessAnnotationArtifacts:
    """Bound annotation output plus trace projection for one Chess sample."""

    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]


@dataclass(frozen=True)
class ChessObjectivePlan:
    """Task-owned hooks prepared for one selected Chess objective/query."""

    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder
    render_sample: RenderBuilder
    build_answer: AnswerBuilder
    build_annotation: AnnotationBuilder
    build_trace_payload: TraceBuilder
    prompt_dynamic_slots: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)
    build_query_params: QueryParamBuilder | None = None
    build_prompt_dynamic_slots: PromptDynamicSlotBuilder | None = None
    build_execution_extra: ExecutionExtraBuilder | None = None


def chess_target_trace_params(
    *,
    target_answer: int,
    answer_support: tuple[int, ...],
    answer_probabilities: Mapping[int, float] | Mapping[str, float],
) -> dict[str, Any]:
    """Return standard target-answer support/probability trace fields."""

    return {
        "target_answer": int(target_answer),
        "target_answer_support": [int(value) for value in answer_support],
        "target_answer_probabilities": dict(answer_probabilities),
    }


def resolve_chess_task_target(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: tuple[int, ...],
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[int, float]]:
    """Resolve a task-local Chess target answer axis."""

    target_answer, answer_support, answer_probabilities = resolve_target_answer(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        gen_defaults=gen_defaults,
    )
    return int(target_answer), tuple(int(value) for value in answer_support), dict(answer_probabilities)


def prepare_chess_piece_count_objective(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    task_id: str,
    query_id: str,
    gen_defaults: Mapping[str, Any],
    attempt_namespace: str,
    include_color: bool,
) -> ChessObjectivePlan:
    """Prepare a visible-piece count objective, optionally constrained by color."""

    target_answer, answer_support, answer_probs = resolve_chess_task_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        support_key="piece_type_count_support",
        fallback_support=(0, 1, 2, 3, 4, 5, 6),
        namespace=f"{task_id}.target_answer",
    )
    target_kind, target_kind_probs = resolve_string_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        explicit_key="target_piece_kind",
        weights_key="target_piece_kind_weights",
        balance_flag_key="balanced_target_piece_kind_sampling",
        supported_values=PIECE_COUNT_KIND_SUPPORT,
        namespace=f"{task_id}.target_piece_kind",
        gen_defaults=gen_defaults,
    )
    distractor_count, distractor_support, distractor_probs = resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="piece_count_distractor_count_support",
        explicit_key="piece_count_distractor_count",
        fallback_support=(1, 2, 3, 4, 5, 6, 7, 8),
        namespace=f"{task_id}.distractor_count",
        balance_flag_key="balanced_piece_count_distractor_count_sampling",
        gen_defaults=gen_defaults,
    )
    target_color = None
    target_color_probs: Mapping[str, float] = {}
    if bool(include_color):
        target_color, target_color_probs = resolve_string_axis(
            instance_seed=int(instance_seed),
            params=task_params,
            explicit_key="target_piece_color",
            weights_key="target_piece_color_weights",
            balance_flag_key="balanced_target_piece_color_sampling",
            supported_values=PIECE_COUNT_COLOR_SUPPORT,
            namespace=f"{task_id}.target_piece_color",
            gen_defaults=gen_defaults,
        )

    def construct_attempt(rng, axes):
        return sample_piece_count_scene(
            rng=rng,
            axes=axes,
            target_answer=int(target_answer),
            target_kind=str(target_kind),
            target_color=None if target_color is None else str(target_color),
            distractor_count=int(distractor_count),
        )

    def render_sample(sample, params, seed):
        label = piece_kind_plural(str(target_kind))
        if target_color is not None:
            label = f"{color_name(str(target_color))} {label}"
        return render_chess_task_scene(
            board=sample.board,
            scene_variant=sample.scene_variant,
            style_variant=sample.style_variant,
            badge_text=f"Count {label}",
            marked_coord=None,
            params=params,
            instance_seed=int(seed),
        )

    def prompt_slots(_sample) -> dict[str, str]:
        slots = {
            "target_piece_kind": str(target_kind),
            "target_piece_kind_plural": piece_kind_plural(str(target_kind)),
        }
        if target_color is not None:
            slots["target_color_name"] = color_name(str(target_color))
        return slots

    query_params = {
        **chess_target_trace_params(
            target_answer=int(target_answer),
            answer_support=answer_support,
            answer_probabilities=answer_probs,
        ),
        "target_piece_kind": str(target_kind),
        "target_piece_kind_probabilities": dict(target_kind_probs),
        "piece_count_distractor_count": int(distractor_count),
        "piece_count_distractor_count_support": [int(value) for value in distractor_support],
        "piece_count_distractor_count_probabilities": dict(distractor_probs),
    }
    if target_color is not None:
        query_params["target_piece_color"] = str(target_color)
        query_params["target_piece_color_probabilities"] = dict(target_color_probs)

    from .shared.output import common_trace_sections

    return ChessObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(query_id),
        query_params=query_params,
        construct_attempt=construct_attempt,
        render_sample=render_sample,
        build_answer=lambda _sample: TypedValue(type="integer", value=int(target_answer)),
        build_annotation=lambda sample, rendered: bbox_set_annotation_for_sample(sample, rendered, witness_type="piece_set"),
        build_trace_payload=lambda sample, rendered: common_trace_sections(sample=sample, rendered_context=rendered),
        build_prompt_dynamic_slots=prompt_slots,
        execution_extra={"target_answer": int(target_answer)},
    )


def prepare_chess_bbox_count_objective(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    task_id: str,
    query_id: str,
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: tuple[int, ...],
    attempt_namespace: str,
    construct_sample: CountSampleConstructor,
    badge_text: str | None = None,
    badge_builder: BadgeBuilder | None = None,
    witness_type: str,
    query_params: Mapping[str, Any] | None = None,
    execution_extra: Mapping[str, Any] | None = None,
    build_query_params: QueryParamBuilder | None = None,
    build_execution_extra: ExecutionExtraBuilder | None = None,
    build_prompt_dynamic_slots: PromptDynamicSlotBuilder | None = None,
) -> ChessObjectivePlan:
    """Prepare a bbox-set integer count objective from task-owned semantics."""

    target_answer, answer_support, answer_probs = resolve_chess_task_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{task_id}.target_answer.{query_id}",
    )

    def construct_attempt(rng, axes):
        return construct_sample(rng, axes, int(target_answer))

    def render_sample(sample, params, seed):
        resolved_badge = str(badge_builder(sample)) if badge_builder is not None else str(badge_text or "")
        return render_chess_task_scene(
            board=sample.board,
            scene_variant=sample.scene_variant,
            style_variant=sample.style_variant,
            badge_text=resolved_badge,
            marked_coord=sample.marked_coord,
            target_coord=sample.target_coord,
            params=params,
            instance_seed=int(seed),
        )

    from .shared.output import common_trace_sections

    return ChessObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(query_id),
        query_params={
            **chess_target_trace_params(
                target_answer=int(target_answer),
                answer_support=answer_support,
                answer_probabilities=answer_probs,
            ),
            **dict(query_params or {}),
        },
        construct_attempt=construct_attempt,
        render_sample=render_sample,
        build_answer=lambda _sample: TypedValue(type="integer", value=int(target_answer)),
        build_annotation=lambda sample, rendered: bbox_set_annotation_for_sample(sample, rendered, witness_type=str(witness_type)),
        build_trace_payload=lambda sample, rendered: common_trace_sections(sample=sample, rendered_context=rendered),
        execution_extra={"target_answer": int(target_answer), **dict(execution_extra or {})},
        build_query_params=build_query_params,
        build_execution_extra=build_execution_extra,
        build_prompt_dynamic_slots=build_prompt_dynamic_slots,
    )


def prepare_marked_piece_destination_family_objective(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    task_id: str,
    query_id: str,
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: tuple[int, ...],
    marked_piece_kind_support: tuple[str, ...],
    destination_mode: str,
    query_destination_mode: str,
    attempt_namespace: str,
) -> ChessObjectivePlan:
    """Prepare one marked-piece destination-family objective from task-owned constants."""

    marked_piece_kind, marked_piece_kind_probs = resolve_string_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        explicit_key="marked_piece_kind",
        weights_key="marked_piece_kind_weights",
        balance_flag_key="balanced_marked_piece_kind_sampling",
        supported_values=tuple(str(value) for value in marked_piece_kind_support),
        namespace=f"{task_id}.marked_piece_kind",
        gen_defaults=gen_defaults,
    )

    def construct_sample(rng, axes, target_answer):
        return sample_marked_piece_destination_scene(
            rng=rng,
            axes=axes,
            destination_mode=str(destination_mode),
            target_answer=int(target_answer),
            marked_piece_kind=str(marked_piece_kind),
        )

    return prepare_chess_bbox_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=str(task_id),
        query_id=str(query_id),
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        attempt_namespace=str(attempt_namespace),
        construct_sample=construct_sample,
        badge_builder=lambda sample: "Marked piece"
        if sample.marked_piece is None
        else f"Marked {piece_name(sample.marked_piece)}",
        witness_type="cell_set",
        query_params={
            "destination_mode": str(query_destination_mode),
            "marked_piece_kind": str(marked_piece_kind),
            "marked_piece_kind_probabilities": dict(marked_piece_kind_probs),
        },
        execution_extra={"marked_piece_kind": str(marked_piece_kind)},
    )


def bbox_set_annotation_for_sample(
    sample: ChessSceneSample,
    rendered_context: RenderedChessTaskContext,
    *,
    witness_type: str,
) -> ChessAnnotationArtifacts:
    """Project one Chess sample's annotation entity ids to bbox-set output."""

    annotation_bboxes = bbox_set_for_entities(
        rendered_context.rendered_scene,
        entity_ids=sample.annotation_entity_ids,
        annotation_kind=sample.annotation_kind,
    )
    return ChessAnnotationArtifacts(
        annotation_gt=TypedValue(type="bbox_set", value=annotation_bboxes),
        projected_annotation=projected_bbox_payload(annotation_bboxes),
        witness_symbolic={
            "type": str(witness_type),
            "ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        },
    )


def keyed_checkmate_annotation_for_sample(
    sample: ChessCheckmateSample,
    rendered_context: RenderedChessTaskContext,
) -> ChessAnnotationArtifacts:
    """Project source, destination, and opposing king cells to point-map entries."""

    annotation_map = move_point_map(
        rendered_context.rendered_scene,
        source=sample.correct_option.source,
        destination=sample.correct_option.destination,
        king=sample.defender_king_coord,
    )
    return ChessAnnotationArtifacts(
        annotation_gt=TypedValue(type="point_map", value=annotation_map),
        projected_annotation=projected_point_map_payload(annotation_map),
        witness_symbolic={
            "type": "cell_map",
            "ids": {
                "from": coord_to_cell_id(sample.correct_option.source),
                "to": coord_to_cell_id(sample.correct_option.destination),
                "king": coord_to_cell_id(sample.defender_king_coord),
            },
        },
    )


def run_chess_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run neutral Chess query, retry, render, prompt, trace, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_chess_scene_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(query_id),
        dict(query_probabilities),
    )

    last_error: ValueError | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, axes)
        except ValueError as exc:
            last_error = exc
            continue

        rendered_context = objective.render_sample(sample, task_params, int(instance_seed))
        annotation_artifacts = objective.build_annotation(sample, rendered_context)
        sample_dynamic_slots = (
            dict(objective.build_prompt_dynamic_slots(sample))
            if objective.build_prompt_dynamic_slots is not None
            else {}
        )
        _prompt_defaults, prompt_artifacts = build_chess_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            dynamic_slots={**dict(objective.prompt_dynamic_slots), **sample_dynamic_slots},
            instance_seed=int(instance_seed),
        )
        answer_gt = objective.build_answer(sample)
        sample_query_params = (
            dict(objective.build_query_params(sample))
            if objective.build_query_params is not None
            else {}
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                **dict(objective.query_params),
                **sample_query_params,
                "query_id_probabilities": dict(query_probabilities),
            },
        )
        trace_payload = objective.build_trace_payload(sample, rendered_context)
        trace_payload["query_spec"] = query_spec
        task_execution_extra = (
            dict(objective.build_execution_extra(sample))
            if objective.build_execution_extra is not None
            else {}
        )
        trace_payload["execution_trace"].update(
            {
                **dict(objective.execution_extra),
                **task_execution_extra,
                "query_id": str(query_id),
                "answer": answer_gt.value,
            }
        )
        trace_payload["projected_annotation"] = dict(annotation_artifacts.projected_annotation)
        trace_payload["witness_symbolic"] = dict(annotation_artifacts.witness_symbolic)
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid Chess scene after {max_attempts} attempts") from last_error


def run_chess_public_entry(task: Any, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int) -> TaskOutput:
    """Run the scene lifecycle using public-task-owned objective metadata."""

    return run_chess_lifecycle(
        task_id=str(task.task_id),
        domain=str(task.domain),
        supported_query_ids=tuple(str(value) for value in task.supported_query_ids),
        default_query_id=str(task.default_query_id),
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        prepare_objective=task.prepare_objective,
    )


__all__ = [
    "ChessAnnotationArtifacts",
    "ChessObjectivePlan",
    "bbox_set_annotation_for_sample",
    "chess_target_trace_params",
    "keyed_checkmate_annotation_for_sample",
    "prepare_chess_bbox_count_objective",
    "prepare_marked_piece_destination_family_objective",
    "prepare_chess_piece_count_objective",
    "resolve_chess_task_target",
    "run_chess_public_entry",
    "run_chess_lifecycle",
]
