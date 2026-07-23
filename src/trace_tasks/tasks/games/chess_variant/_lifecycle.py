"""Scene-private lifecycle plumbing for Chess Variant public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.piece_board_rules import piece_name
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import annotation_from_evaluation
from .shared.defaults import SCENE_ID
from .shared.output import common_trace_sections
from .shared.prompts import build_chess_variant_prompt_artifacts, prompt_defaults
from .shared.rendering import (
    draw_marked_outline,
    render_chess_variant_scene,
    resolve_chess_variant_render_params,
    resolve_scene_background,
    rule_badge_text,
    text_style_metadata,
)
from .shared.sampling import (
    max_possible_marked_destination_answer,
    resolve_chess_variant_scene_axes,
    resolve_task_target_answer,
    sample_marked_destination_scene,
)
from .shared.state import ChessVariantSample, ChessVariantSceneAxes


AttemptBuilder = Callable[[Any], ChessVariantSample]
CountAttemptBuilder = Callable[[Any, int], ChessVariantSample]
ExecutionExtraBuilder = Callable[[ChessVariantSample], Mapping[str, Any]]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], ChessVariantSceneAxes, str],
    "ChessVariantObjectivePlan",
]

POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


@dataclass(frozen=True)
class ChessVariantObjectivePlan:
    """Task-owned semantic hooks for one selected Chess Variant query."""

    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder
    outline_rgb: tuple[int, int, int]
    landing_rule_text: str = ""
    example_answer: int = 3
    execution_extra: Mapping[str, Any] = field(default_factory=dict)
    build_execution_extra: ExecutionExtraBuilder | None = None


@dataclass(frozen=True)
class ChessVariantTargetAnswerPlan:
    """Resolved integer answer plus standard trace query parameters."""

    answer: int
    query_params: Mapping[str, Any]


def resolve_chess_variant_count_target(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    fallback_support: tuple[int, ...],
    possible_max: int,
    namespace: str,
) -> ChessVariantTargetAnswerPlan:
    """Resolve a task-owned count target and package its trace fields."""

    target_answer, answer_support, answer_probs = resolve_task_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        support_key=str(support_key),
        fallback_support=tuple(int(v) for v in fallback_support),
        possible_max=int(possible_max),
        namespace=str(namespace),
    )
    return ChessVariantTargetAnswerPlan(
        answer=int(target_answer),
        query_params={
            "target_answer": int(target_answer),
            "target_answer_support": [int(v) for v in answer_support],
            "target_answer_probabilities": dict(answer_probs),
        },
    )


def prepare_chess_variant_count_objective(
    *,
    task_id: str,
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    support_key: str,
    fallback_support: tuple[int, ...],
    possible_max: int,
    attempt_namespace: str,
    semantic_query_params: Mapping[str, Any],
    construct_sample: CountAttemptBuilder,
    outline_rgb: tuple[int, int, int],
    landing_rule_text: str = "",
    example_answer: int = 3,
    execution_extra: Mapping[str, Any] | None = None,
    build_execution_extra: ExecutionExtraBuilder | None = None,
) -> ChessVariantObjectivePlan:
    """Build a count-objective plan from task-owned semantic arguments."""

    target_plan = resolve_chess_variant_count_target(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key=str(support_key),
        fallback_support=tuple(int(v) for v in fallback_support),
        possible_max=int(possible_max),
        namespace=f"{task_id}.target_answer.{query_id}",
    )

    def construct_attempt(rng):
        return construct_sample(rng, int(target_plan.answer))

    return ChessVariantObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(query_id),
        query_params={**dict(semantic_query_params), **dict(target_plan.query_params)},
        construct_attempt=construct_attempt,
        outline_rgb=tuple(int(value) for value in outline_rgb),
        landing_rule_text=str(landing_rule_text),
        example_answer=int(example_answer),
        execution_extra=dict(execution_extra or {}),
        build_execution_extra=build_execution_extra,
    )


def prepare_marked_piece_count_objective(
    *,
    task_id: str,
    instance_seed: int,
    task_params: Mapping[str, Any],
    axes: ChessVariantSceneAxes,
    query_id: str,
    support_key: str,
    fallback_support: tuple[int, ...],
    destination_mode: str,
    attempt_namespace: str,
    landing_rule_text: str,
    example_answer: int,
) -> ChessVariantObjectivePlan:
    """Prepare a marked-piece count using task-owned destination semantics."""

    possible_max = max_possible_marked_destination_answer(
        destination_mode=str(destination_mode),
        rule_family=str(axes.rule_family),
        range_k=int(axes.range_k),
    )

    def execution_extra(sample: ChessVariantSample) -> Mapping[str, Any]:
        marked_piece = sample.evaluation.marked_piece
        return {
            "marked_piece_name": "" if marked_piece is None else piece_name(marked_piece),
        }

    return prepare_chess_variant_count_objective(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        task_params=task_params,
        query_id=str(query_id),
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        possible_max=int(possible_max),
        attempt_namespace=str(attempt_namespace),
        semantic_query_params={"destination_mode": str(destination_mode)},
        construct_sample=lambda rng, target_answer: sample_marked_destination_scene(
            rng=rng,
            axes=axes,
            destination_mode=str(destination_mode),
            target_answer=int(target_answer),
        ),
        outline_rgb=(220, 38, 38),
        landing_rule_text=str(landing_rule_text),
        example_answer=int(example_answer),
        execution_extra={"destination_mode": str(destination_mode)},
        build_execution_extra=execution_extra,
    )


def _axis_query_params(axes: ChessVariantSceneAxes) -> dict[str, Any]:
    """Return shared trace parameters for scene, style, rule, and range axes."""

    return {
        "rule_family": str(axes.rule_family),
        "range_k": int(axes.range_k),
        "scene_variant": str(axes.scene_variant),
        "style_variant": str(axes.style_variant),
        "rule_family_probabilities": dict(axes.rule_family_probabilities),
        "range_k_probabilities": dict(axes.range_k_probabilities),
        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
        "style_variant_probabilities": dict(axes.style_variant_probabilities),
    }


def run_chess_variant_lifecycle(
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
    """Run neutral Chess Variant query, retry, render, prompt, trace, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_chess_variant_scene_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        axes,
        str(query_id),
    )

    last_error: ValueError | None = None
    sample: ChessVariantSample | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{attempt_index}")
        try:
            sample = objective.construct_attempt(rng)
        except ValueError as exc:
            last_error = exc
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate a valid scene after {max_attempts} attempts") from last_error

    render_params = resolve_chess_variant_render_params(task_params, instance_seed=int(instance_seed))
    prompt_defaults_map = prompt_defaults()
    badge_text = rule_badge_text(str(axes.rule_family), int(axes.range_k), prompt_defaults_map)
    background, background_meta, panel_style, panel_style_meta = resolve_scene_background(
        params=task_params,
        render_params=render_params,
        instance_seed=int(instance_seed),
    )
    rendered_image, render_map, scene_entities = render_chess_variant_scene(
        board=sample.board,
        axes=axes,
        background=background,
        params=render_params,
        badge_text=badge_text,
        panel_style=panel_style,
    )
    draw_marked_outline(
        rendered_image,
        render_map,
        sample.evaluation.marked_coord,
        render_params,
        outline_rgb=tuple(int(value) for value in objective.outline_rgb),
    )
    annotation_type, annotation_value, projected_annotation = annotation_from_evaluation(
        evaluation=sample.evaluation,
        render_map=render_map,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    _prompt_defaults, prompt_artifacts = build_chess_variant_prompt_artifacts(
        domain=str(domain),
        prompt_query_key=str(objective.prompt_query_key),
        scene_variant=str(axes.scene_variant),
        rule_family=str(axes.rule_family),
        range_k=int(axes.range_k),
        landing_rule_text=str(objective.landing_rule_text),
        point_annotation=False,
        example_answer=int(objective.example_answer),
        instance_seed=int(instance_seed),
    )
    answer_gt = TypedValue(type="integer", value=int(sample.evaluation.answer))
    annotation_gt = TypedValue(type=str(annotation_type), value=[list(item) for item in annotation_value])
    trace_payload = common_trace_sections(
        sample=sample,
        image_size=(int(image.size[0]), int(image.size[1])),
        render_map=dict(render_map),
        scene_entities=tuple(scene_entities),
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=text_style_metadata(str(render_params.font_family)),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        rule_family=str(axes.rule_family),
        range_k=int(axes.range_k),
    )
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params={
            "query_id": str(query_id),
            **_axis_query_params(axes),
            **dict(objective.query_params),
            "query_id_probabilities": dict(query_probabilities),
        },
    )
    sample_execution_extra = (
        dict(objective.build_execution_extra(sample))
        if objective.build_execution_extra is not None
        else {}
    )
    trace_payload["execution_trace"].update(
        {
            "query_id": str(query_id),
            "answer": int(answer_gt.value),
            "target_answer_support": list(objective.query_params.get("target_answer_support", [])),
            **dict(objective.execution_extra),
            **sample_execution_extra,
        }
    )
    trace_payload["scene_ir"]["relations"]["query_id"] = str(query_id)
    trace_payload["projected_annotation"] = dict(projected_annotation)
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
    )


def run_chess_variant_public_entry(task: Any, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int) -> TaskOutput:
    """Run the scene lifecycle using public-task-owned objective metadata."""

    return run_chess_variant_lifecycle(
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
    "ChessVariantObjectivePlan",
    "ChessVariantTargetAnswerPlan",
    "prepare_chess_variant_count_objective",
    "prepare_marked_piece_count_objective",
    "resolve_chess_variant_count_target",
    "run_chess_variant_lifecycle",
    "run_chess_variant_public_entry",
]
