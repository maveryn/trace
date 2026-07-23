"""Scene-private lifecycle orchestration for 2048 public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, segment_set_annotation_artifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import source_merge_cell_id_pairs, source_merge_point_pairs
from .shared.rules import simulate_2048_move
from .shared.output import build_2048_common_trace_payload, common_2048_trace_params
from .shared.prompts import build_2048_prompt_artifacts
from .shared.rendering import Rendered2048TaskContext, render_2048_sample
from .shared.sampling import Resolved2048Axes, board_for_merge_values, resolve_2048_axes, resolve_2048_integer_target
from .shared.state import Board, SCENE_ID, SUPPORTED_2048_DIRECTIONS, Sample2048, validate_2048_scene_sample


AnnotationBuilder = Callable[[Rendered2048TaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, Resolved2048Axes], "Attempt2048Result"]
ObjectivePreparer = Callable[[int, Mapping[str, Any], Mapping[str, float], str], "Objective2048Plan"]
MergeValuesForTarget = Callable[[int, Any], Sequence[int]]
MergeResultPredicate = Callable[[Any, int], bool]


@dataclass(frozen=True)
class Attempt2048Result:
    """Task-owned result of one constructed 2048 attempt."""

    sample: Sample2048
    answer_gt: TypedValue
    build_annotation: AnnotationBuilder
    annotation_entity_ids: tuple[str, ...] = tuple()
    annotation_cell_id_pairs: list[list[str]] = field(default_factory=list)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Objective2048Plan:
    """Prepared task-owned objective hooks for one generated instance."""

    attempt_namespace: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder


def all_direction_results(board: Board) -> Dict[str, Any]:
    """Return the standard one-move result map for one 2048 board."""

    return {
        str(direction): simulate_2048_move(board, str(direction))
        for direction in SUPPORTED_2048_DIRECTIONS
    }


def prepare_merge_source_integer_objective(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    target_namespace: str,
    attempt_namespace: str,
    construction_mode: str,
    merge_values_for_target: MergeValuesForTarget,
    result_matches_target: MergeResultPredicate,
) -> Objective2048Plan:
    """Prepare an integer objective whose annotation is the set of merge-source pairs."""

    target_axis = resolve_2048_integer_target(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(target_namespace),
    )
    target = int(target_axis.target_answer)

    def construct_attempt(rng, axes) -> Attempt2048Result:
        board = board_for_merge_values(
            rng=rng,
            direction=str(axes.move_direction),
            merge_values=tuple(int(value) for value in merge_values_for_target(target, rng)),
            force_slide_when_no_merge=(target == 0),
        )
        result = simulate_2048_move(board, str(axes.move_direction))
        if not result.moved or not bool(result_matches_target(result, target)):
            raise ValueError("constructed move does not match the requested merge-source integer target")

        annotation_pairs = source_merge_cell_id_pairs(result)
        sample = Sample2048(
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            board=board,
            move_direction=str(axes.move_direction),
            move_result=result,
            all_move_results=all_direction_results(board),
            construction_mode=str(construction_mode),
        )
        return Attempt2048Result(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=target),
            annotation_entity_ids=tuple(coord for pair in annotation_pairs for coord in pair),
            annotation_cell_id_pairs=annotation_pairs,
            build_annotation=lambda rendered: segment_set_annotation_artifacts(
                source_merge_point_pairs(result, rendered.rendered_scene)
            ),
        )

    return Objective2048Plan(
        attempt_namespace=str(attempt_namespace),
        query_params={
            "target_answer": int(target_axis.target_answer),
            "target_answer_support": [int(value) for value in target_axis.target_answer_support],
            "target_answer_probabilities": dict(target_axis.target_answer_probabilities),
        },
        construct_attempt=construct_attempt,
    )


def run_2048_lifecycle(
    *,
    task_id: str,
    domain: str,
    prompt_query_key: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    render_param_overrides: Mapping[str, Any] | None = None,
) -> TaskOutput:
    """Run common 2048 query, render, prompt, annotation, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    effective_params = {**dict(render_param_overrides or {}), **dict(task_params)}
    axes = resolve_2048_axes(int(instance_seed), params=effective_params, gen_defaults=gen_defaults)
    objective = prepare_objective(int(instance_seed), effective_params, query_probabilities, query_id)

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        validate_2048_scene_sample(attempt.sample)
        rendered_context = render_2048_sample(
            axes=axes,
            sample=attempt.sample,
            params=effective_params,
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_2048_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(prompt_query_key),
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=common_2048_trace_params(
                axes,
                attempt.sample,
                extra_params={
                    **dict(objective.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_2048_common_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            annotation_cell_id_pairs=[list(pair) for pair in attempt.annotation_cell_id_pairs],
            axes=axes,
            sample=attempt.sample,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
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
            query_id=query_id,
        )

    raise RuntimeError(f"{task_id} failed to generate a valid 2048 board after {max_attempts} attempts")


__all__ = [
    "Attempt2048Result",
    "Objective2048Plan",
    "all_direction_results",
    "prepare_merge_source_integer_objective",
    "run_2048_lifecycle",
]
