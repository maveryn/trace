"""Scene-private lifecycle orchestration for dots-and-boxes public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import dots_and_boxes_annotation_artifacts
from .shared.defaults import SCENE_ID
from .shared.output import (
    build_dots_and_boxes_common_trace_params,
    build_dots_and_boxes_trace_payload,
)
from .shared.prompts import build_dots_and_boxes_prompt_artifacts
from .shared.rendering import render_dots_and_boxes_task_context
from .shared.sampling import (
    resolve_dots_and_boxes_board_shape_axis,
    resolve_dots_and_boxes_render_params,
    resolve_dots_and_boxes_scene_axes,
)
from .shared.state import (
    DotsAndBoxesBoardShapeAxis,
    DotsAndBoxesBoardState,
    DotsAndBoxesIntegerAxis,
)
from .shared.rules import build_dots_and_boxes_count_board_state

AttemptBuilder = Callable[[Any], "DotsAndBoxesAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], Mapping[str, float], str, DotsAndBoxesBoardShapeAxis],
    "DotsAndBoxesObjectivePlan",
]


@dataclass(frozen=True)
class DotsAndBoxesAttemptResult:
    """Task-owned board result plus annotation witnesses for one attempt."""

    board_state: DotsAndBoxesBoardState
    annotation_kind: str
    annotation_entity_ids: Sequence[str]
    execution_extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class DotsAndBoxesObjectivePlan:
    """Task-owned objective hooks prepared by one public task file."""

    prompt_query_key: str
    annotation_example_shape: str
    answer_gt: TypedValue
    query_params: Mapping[str, Any]
    attempt_namespace: str
    construct_attempt: AttemptBuilder
    candidate_edge_count_axis: DotsAndBoxesIntegerAxis | None = None
    prompt_dynamic_slots: Mapping[str, Any] | None = None


def make_count_objective_plan(
    *,
    prompt_query_key: str,
    annotation_example_shape: str,
    annotation_kind: str,
    annotation_entity_attr: str,
    target_axis: DotsAndBoxesIntegerAxis,
    board_shape: DotsAndBoxesBoardShapeAxis,
    count_mode: str,
    attempt_namespace: str,
    candidate_edge_count_axis: DotsAndBoxesIntegerAxis | None = None,
    owner: str = "",
    query_params_extra: Mapping[str, Any] | None = None,
    prompt_dynamic_slots: Mapping[str, Any] | None = None,
) -> DotsAndBoxesObjectivePlan:
    """Build a lifecycle objective from task-owned semantic count arguments."""

    def construct_attempt(rng: Any) -> DotsAndBoxesAttemptResult:
        board_state = build_dots_and_boxes_count_board_state(
            rng=rng,
            count_mode=str(count_mode),
            owner=str(owner),
            target_answer=int(target_axis.value),
            box_rows=int(board_shape.box_rows),
            box_cols=int(board_shape.box_cols),
            candidate_edge_count=(
                0
                if candidate_edge_count_axis is None
                else int(candidate_edge_count_axis.value)
            ),
        )
        return DotsAndBoxesAttemptResult(
            board_state=board_state,
            annotation_kind=str(annotation_kind),
            annotation_entity_ids=tuple(
                str(entity_id)
                for entity_id in getattr(board_state, str(annotation_entity_attr))
            ),
        )

    return DotsAndBoxesObjectivePlan(
        prompt_query_key=str(prompt_query_key),
        annotation_example_shape=str(annotation_example_shape),
        answer_gt=TypedValue(type="integer", value=int(target_axis.value)),
        query_params={
            "target_answer": int(target_axis.value),
            "target_answer_support": [int(value) for value in target_axis.support],
            "target_answer_probabilities": dict(target_axis.probabilities),
            **dict(query_params_extra or {}),
        },
        attempt_namespace=str(attempt_namespace),
        construct_attempt=construct_attempt,
        candidate_edge_count_axis=candidate_edge_count_axis,
        prompt_dynamic_slots=dict(prompt_dynamic_slots or {}),
    )


def run_dots_and_boxes_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run common dots-and-boxes scene plumbing around task-owned hooks."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(query_id) for query_id in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    scene_axes = resolve_dots_and_boxes_scene_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
    )
    board_shape = resolve_dots_and_boxes_board_shape_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        query_probabilities,
        str(selected_query_id),
        board_shape,
    )
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            attempt = objective.construct_attempt(rng)
        except (RuntimeError, ValueError):
            continue

        render_params = resolve_dots_and_boxes_render_params(
            task_params,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
        )
        rendered_context = render_dots_and_boxes_task_context(
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=render_defaults,
            board_state=attempt.board_state,
            scene_variant=str(scene_axes.scene_variant),
            style_variant=str(scene_axes.style_variant),
            render_params=render_params,
        )
        annotation_artifacts = dots_and_boxes_annotation_artifacts(
            annotation_kind=str(attempt.annotation_kind),
            annotation_entity_ids=tuple(
                str(entity_id) for entity_id in attempt.annotation_entity_ids
            ),
            render_map=rendered_context.rendered_scene.render_map,
        )
        prompt_defaults, prompt_artifacts = build_dots_and_boxes_prompt_artifacts(
            domain=str(domain),
            scene_variant=str(scene_axes.scene_variant),
            prompt_query_key=str(objective.prompt_query_key),
            annotation_example_shape=str(objective.annotation_example_shape),
            answer_example=objective.answer_gt.value,
            dynamic_slots=dict(objective.prompt_dynamic_slots or {}),
            instance_seed=int(instance_seed),
        )
        common_query_params = build_dots_and_boxes_common_trace_params(
            scene_axes=scene_axes,
            board_shape=board_shape,
            candidate_edge_count_axis=objective.candidate_edge_count_axis,
            extra_params={
                "query_id_probabilities": {
                    str(key): float(value) for key, value in query_probabilities.items()
                },
                **dict(objective.query_params),
            },
        )
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params=common_query_params,
        )
        trace_payload = build_dots_and_boxes_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(
                str(entity_id) for entity_id in attempt.annotation_entity_ids
            ),
            scene_axes=scene_axes,
            board_shape=board_shape,
            board_state=attempt.board_state,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_query_spec=prompt_query_spec,
            answer_value=objective.answer_gt.value,
            candidate_edge_count_axis=objective.candidate_edge_count_axis,
            execution_extra={
                "query_id": str(selected_query_id),
                "answer": objective.answer_gt.value,
                **dict(objective.query_params),
                **dict(attempt.execution_extra or {}),
            },
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=objective.answer_gt,
            annotation_gt=TypedValue(
                type=str(annotation_artifacts.annotation_type),
                value=annotation_artifacts.value,
            ),
            image=rendered_context.image,
            image_id="img0",
            trace_payload=dict(trace_payload),
            task_versions=default_task_versions(),
            query_id=str(selected_query_id),
            scene_id=SCENE_ID,
        )

    raise RuntimeError(
        f"{task_id} failed to generate a valid dots-and-boxes scene after {max_attempts} attempts"
    )


__all__ = [
    "DotsAndBoxesAttemptResult",
    "DotsAndBoxesObjectivePlan",
    "make_count_objective_plan",
    "run_dots_and_boxes_lifecycle",
]
