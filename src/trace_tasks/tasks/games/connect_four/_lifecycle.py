"""Scene-private lifecycle plumbing for Connect Four public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import bbox_set_annotation_artifacts, point_annotation_artifacts, point_set_annotation_artifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import cell_bboxes_for_coords, cell_points_for_coords
from .shared.defaults import SCENE_ID
from .shared.output import common_trace_params, common_trace_sections
from .shared.prompts import (
    build_connect_four_prompt_artifacts,
    connect_four_object_description,
    connect_four_output_slots,
    connect_four_rule_slots,
    json_examples_for_label_answer,
    json_examples_for_integer_answer,
)
from .shared.rendering import render_connect_four_sample
from .shared.rules import Coord, opponent, player_name
from .shared.sampling import resolve_connect_four_scene_axes, resolve_target_answer, sample_count_scene
from .shared.state import ConnectFourColumnProfileSample, ConnectFourCountSample, ConnectFourLabelSample, ConnectFourSceneAxes


ConnectFourSample = ConnectFourCountSample | ConnectFourLabelSample | ConnectFourColumnProfileSample
AttemptBuilder = Callable[[Any], ConnectFourSample]
SampleAnswerBuilder = Callable[[ConnectFourSample], TypedValue]
SampleCoordsBuilder = Callable[[ConnectFourSample], Sequence[Coord]]
SampleMappingBuilder = Callable[[ConnectFourSample], Mapping[str, Any]]
SampleMarkedCoordBuilder = Callable[[ConnectFourSample], Coord | None]
SampleColumnLabelsBuilder = Callable[[ConnectFourSample], Sequence[str] | None]
SceneObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float]],
    "ConnectFourObjectivePlan",
]


@dataclass(frozen=True)
class ConnectFourObjectivePlan:
    """Task-owned objective hooks consumed by neutral scene plumbing."""

    axes: ConnectFourSceneAxes
    attempt_namespace: str
    construct_attempt: AttemptBuilder
    prompt_query_key: str
    prompt_dynamic_slots: SampleMappingBuilder
    answer_gt: SampleAnswerBuilder
    annotation_coords: SampleCoordsBuilder
    annotation_type: str
    render_marked_square: SampleMarkedCoordBuilder
    render_column_labels: SampleColumnLabelsBuilder
    query_spec_params: SampleMappingBuilder
    execution_updates: SampleMappingBuilder


@dataclass(frozen=True)
class ConnectFourCountObjectiveSpec:
    """Task-owned semantic parameters for one count-style Connect Four objective."""

    prompt_query_key: str
    count_mode: str
    support_key: str
    fallback_support: tuple[int, ...]
    safe_board_defaults: bool
    include_safety_rule: bool


def prepare_count_objective_from_spec(
    *,
    task_id: str,
    spec: ConnectFourCountObjectiveSpec,
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    gen_defaults: Mapping[str, Any],
) -> ConnectFourObjectivePlan:
    """Build a count objective from task-owned semantics without task dispatch."""

    target_answer, target_support, target_probabilities = resolve_target_answer(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(spec.support_key),
        fallback_support=tuple(int(value) for value in spec.fallback_support),
        namespace=f"{task_id}.target_answer",
    )
    axes = resolve_connect_four_scene_axes(
        int(instance_seed),
        params=task_params,
        safe_board_defaults=bool(spec.safe_board_defaults),
        target_answer=int(target_answer) if bool(spec.safe_board_defaults) else None,
        gen_defaults=gen_defaults,
        namespace_suffix=str(selected_query_id),
    )

    def construct_attempt(rng):
        return sample_count_scene(
            rng=rng,
            axes=axes,
            params=task_params,
            count_mode=str(spec.count_mode),
            target_answer=int(target_answer),
            gen_defaults=gen_defaults,
        )

    def prompt_slots(sample) -> dict[str, Any]:
        json_example, json_example_answer_only = json_examples_for_integer_answer()
        return {
            "object_description": connect_four_object_description(str(sample.scene_variant)),
            **connect_four_rule_slots(
                current_player=int(sample.current_player),
                include_safety_rule=bool(spec.include_safety_rule),
            ),
            **connect_four_output_slots(
                prompt_query_key=str(spec.prompt_query_key),
                json_example=json_example,
                json_example_answer_only=json_example_answer_only,
            ),
        }

    def query_spec_params(_sample) -> dict[str, Any]:
        return {
            "target_answer": int(target_answer),
            "target_answer_support": [int(value) for value in target_support],
            "target_answer_probabilities": dict(target_probabilities),
            "count_mode": str(spec.count_mode),
        }

    return ConnectFourObjectivePlan(
        axes=axes,
        attempt_namespace=str(task_id),
        construct_attempt=construct_attempt,
        prompt_query_key=str(spec.prompt_query_key),
        prompt_dynamic_slots=prompt_slots,
        answer_gt=lambda sample: TypedValue(type="integer", value=int(sample.evaluation.answer)),
        annotation_coords=lambda sample: sample.evaluation.annotation_coords,
        annotation_type="bbox_set",
        render_marked_square=lambda _sample: None,
        render_column_labels=lambda _sample: None,
        query_spec_params=query_spec_params,
        execution_updates=lambda _sample: {"target_answer": int(target_answer)},
    )


def prepare_column_label_objective_from_semantics(
    *,
    task_id: str,
    prompt_query_key: str,
    gen_defaults: Mapping[str, Any],
    sample_scene: Callable[..., ConnectFourLabelSample],
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    include_opponent_player: bool,
    line_trace_key: str,
    coord_trace_key: str,
) -> ConnectFourObjectivePlan:
    """Build a column-label objective from task-owned label semantics."""

    axes = resolve_connect_four_scene_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace_suffix=str(selected_query_id),
    )

    def construct_attempt(rng):
        return sample_scene(
            rng=rng,
            axes=axes,
            params=task_params,
            instance_seed=int(instance_seed),
            gen_defaults=gen_defaults,
        )

    def prompt_slots(sample) -> dict[str, Any]:
        json_example, json_example_answer_only = json_examples_for_label_answer(
            scalar_annotation=True
        )
        slots = {
            "object_description": connect_four_object_description(
                str(sample.scene_variant)
            ),
            **connect_four_rule_slots(current_player=int(sample.current_player)),
            **connect_four_output_slots(
                prompt_query_key=str(prompt_query_key),
                json_example=json_example,
                json_example_answer_only=json_example_answer_only,
            ),
        }
        if bool(include_opponent_player):
            slots["opponent_player_name"] = player_name(
                opponent(int(sample.current_player))
            )
        return slots

    def query_spec_params(sample) -> dict[str, Any]:
        params = {
            "answer_label": str(sample.answer_label),
            "answer_column": int(sample.answer_column),
            "answer_support": [str(label) for label in sample.column_labels],
            "threat_kind": str(sample.threat_kind),
            "threat_kind_probabilities": dict(sample.threat_kind_probabilities),
        }
        if bool(include_opponent_player):
            params["opponent_player"] = player_name(
                opponent(int(sample.current_player))
            ).lower()
        return params

    def execution_updates(sample) -> dict[str, Any]:
        updates = {
            "answer_label": str(sample.answer_label),
            "answer_column": int(sample.answer_column),
            "answer_support": [str(label) for label in sample.column_labels],
            "column_labels": [str(label) for label in sample.column_labels],
            str(coord_trace_key): [
                [int(row), int(col)] for row, col in sample.evaluation.annotation_coords
            ],
            str(line_trace_key): [
                [int(row), int(col)] for row, col in sample.winning_line_coords
            ],
            "threat_kind": str(sample.threat_kind),
        }
        if bool(include_opponent_player):
            updates["opponent_player"] = player_name(
                opponent(int(sample.current_player))
            ).lower()
        return updates

    return ConnectFourObjectivePlan(
        axes=axes,
        attempt_namespace=str(task_id),
        construct_attempt=construct_attempt,
        prompt_query_key=str(prompt_query_key),
        prompt_dynamic_slots=prompt_slots,
        answer_gt=lambda sample: TypedValue(type="string", value=str(sample.answer_label)),
        annotation_coords=lambda sample: sample.evaluation.annotation_coords,
        annotation_type="point",
        render_marked_square=lambda _sample: None,
        render_column_labels=lambda sample: sample.column_labels,
        query_spec_params=query_spec_params,
        execution_updates=execution_updates,
    )


def sample_with_retries(
    *,
    task_id: str,
    instance_seed: int,
    max_attempts: int,
    attempt_namespace: str,
    construct_attempt: AttemptBuilder,
) -> ConnectFourSample:
    """Run task-owned Connect Four construction attempts with stable retry seeds."""

    last_error: ValueError | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            return construct_attempt(rng)
        except ValueError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{task_id} failed to generate a valid scene after {max_attempts} attempts") from last_error


def run_connect_four_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: SceneObjectivePreparer,
) -> TaskOutput:
    """Run neutral Connect Four query, retry, render, prompt, and output plumbing."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query_id),
        query_probabilities,
    )
    sample = sample_with_retries(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        attempt_namespace=str(objective.attempt_namespace),
        construct_attempt=objective.construct_attempt,
    )
    rendered_context = render_connect_four_sample(
        sample=sample,
        params=task_params,
        instance_seed=int(instance_seed),
        marked_square=objective.render_marked_square(sample),
        column_labels=objective.render_column_labels(sample),
    )
    annotation_points = cell_points_for_coords(
        rendered_context.rendered_scene,
        objective.annotation_coords(sample),
    )
    annotation_bboxes = cell_bboxes_for_coords(
        rendered_context.rendered_scene,
        objective.annotation_coords(sample),
    )
    if str(objective.annotation_type) == "point":
        if len(annotation_points) != 1:
            raise ValueError("scalar Connect Four point annotation requires exactly one cell")
        annotation_artifacts = point_annotation_artifacts(annotation_points[0])
    elif str(objective.annotation_type) == "point_set":
        annotation_artifacts = point_set_annotation_artifacts(annotation_points)
    elif str(objective.annotation_type) == "bbox_set":
        annotation_artifacts = bbox_set_annotation_artifacts(annotation_bboxes)
    else:
        raise ValueError(f"unsupported Connect Four annotation type: {objective.annotation_type}")
    _prompt_defaults, prompt_artifacts = build_connect_four_prompt_artifacts(
        domain=str(domain),
        prompt_query_key=str(objective.prompt_query_key),
        dynamic_slots=objective.prompt_dynamic_slots(sample),
        instance_seed=int(instance_seed),
    )
    answer_gt = objective.answer_gt(sample)
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=common_trace_params(
            axes=objective.axes,
            sample=sample,
            extra_params={
                "query_id_probabilities": dict(query_probabilities),
                **dict(objective.query_spec_params(sample)),
            },
        ),
    )
    trace_payload = common_trace_sections(
        axes=objective.axes,
        sample=sample,
        rendered_context=rendered_context,
        annotation_artifacts=annotation_artifacts,
        query_spec=query_spec,
        execution_extra={
            "query_id": str(selected_query_id),
            "answer": answer_gt.value,
            **dict(objective.execution_updates(sample)),
        },
    )
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
        query_id=str(selected_query_id),
    )


__all__ = [
    "ConnectFourCountObjectiveSpec",
    "ConnectFourObjectivePlan",
    "prepare_count_objective_from_spec",
    "run_connect_four_lifecycle",
    "sample_with_retries",
]
