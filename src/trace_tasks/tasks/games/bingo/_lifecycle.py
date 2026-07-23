"""Scene-private lifecycle plumbing for Bingo public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import (
    bbox_set_annotation_artifacts,
    segment_annotation_artifacts,
    segment_set_annotation_artifacts,
    point_set_annotation_artifacts,
)
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import cell_bboxes_for_ids, cell_point_pairs_for_id_pairs, cell_points_for_ids
from .shared.defaults import SCENE_ID
from .shared.output import build_bingo_common_trace_params, build_bingo_trace_payload
from .shared.prompts import build_bingo_prompt_artifacts
from .shared.rendering import render_bingo_task_scene
from .shared.sampling import (
    ResolvedBingoSceneAxes,
    ResolvedBingoTarget,
    resolve_bingo_float_param,
    resolve_bingo_integer_target,
    resolve_bingo_scene_axes,
)
from .shared.state import BingoCardState


AttemptBuilder = Callable[[Any, ResolvedBingoSceneAxes], "BingoAttemptResult"]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "BingoObjectivePlan"]


@dataclass(frozen=True)
class BingoAttemptResult:
    """Task-owned result of one constructed Bingo card."""

    card_state: BingoCardState
    answer_value: int | str
    annotation_cell_ids: tuple[str, ...]
    annotation_type: str = "bbox_set"
    annotation_cell_id_pairs: tuple[tuple[str, str], ...] = ()
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BingoObjectivePlan:
    """Prepared task-owned objective hooks for one generated Bingo instance."""

    attempt_namespace: str
    prompt_query_key: str
    query_params: Mapping[str, Any]
    prompt_dynamic_slots: Mapping[str, Any] = field(default_factory=dict)
    show_called_panel: bool = False
    construct_attempt: AttemptBuilder | None = None


def bingo_target_trace_params(target: ResolvedBingoTarget) -> dict[str, Any]:
    """Return standard target-answer support/probability trace fields."""

    return {
        "target_answer_support": [int(value) for value in target.target_answer_support],
        "target_answer_probabilities": dict(target.target_answer_probabilities),
    }


def bingo_named_count_trace_params(name: str, target: ResolvedBingoTarget) -> dict[str, Any]:
    """Return named count support/probability trace fields."""

    key = str(name)
    return {
        key: int(target.target_answer),
        f"{key}_support": [int(value) for value in target.target_answer_support],
        f"{key}_probabilities": dict(target.target_answer_probabilities),
    }


def run_bingo_lifecycle(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    domain: str = "games",
) -> TaskOutput:
    """Run neutral Bingo retry, rendering, prompt, trace, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_bingo_scene_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(query_id),
        dict(query_probabilities),
    )
    if objective.construct_attempt is None:
        raise ValueError("Bingo objective plan must provide a construct_attempt callback")

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        rendered_context = render_bingo_task_scene(
            cells=attempt.card_state.cells,
            called_numbers=attempt.card_state.called_numbers if bool(objective.show_called_panel) else (),
            params=task_params,
            instance_seed=int(instance_seed),
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            mark_shape=str(axes.mark_shape),
            cell_fill_pattern=str(axes.cell_fill_pattern),
            show_called_panel=bool(objective.show_called_panel),
        )
        annotation_bboxes = cell_bboxes_for_ids(rendered_context.rendered_scene, attempt.annotation_cell_ids)
        annotation_points = cell_points_for_ids(rendered_context.rendered_scene, attempt.annotation_cell_ids)
        annotation_point_pairs = cell_point_pairs_for_id_pairs(
            rendered_context.rendered_scene,
            attempt.annotation_cell_id_pairs,
        )
        if str(attempt.annotation_type) == "segment":
            if len(annotation_point_pairs) != 1:
                raise ValueError("scalar segment annotation requires exactly one cell pair")
            annotation_artifacts = segment_annotation_artifacts(annotation_point_pairs[0])
        elif str(attempt.annotation_type) == "segment_set":
            annotation_artifacts = segment_set_annotation_artifacts(annotation_point_pairs)
        elif str(attempt.annotation_type) == "point_set":
            annotation_artifacts = point_set_annotation_artifacts(annotation_points)
        elif str(attempt.annotation_type) == "bbox_set":
            annotation_artifacts = bbox_set_annotation_artifacts(annotation_bboxes)
        else:
            raise ValueError(f"unsupported Bingo annotation type: {attempt.annotation_type}")
        answer_type = "integer" if isinstance(attempt.answer_value, int) else "string"
        answer_gt = TypedValue(type=answer_type, value=attempt.answer_value)
        annotation_gt = annotation_artifacts.annotation_gt
        prompt_defaults, prompt_artifacts = build_bingo_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            instance_seed=int(instance_seed),
            dynamic_slots=dict(objective.prompt_dynamic_slots),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=build_bingo_common_trace_params(
                axes=axes,
                extra_params={
                    **dict(objective.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        trace_payload = build_bingo_trace_payload(
            axes=axes,
            card_state=attempt.card_state,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            query_spec=query_spec,
            answer_value=attempt.answer_value,
            annotation_cell_ids=attempt.annotation_cell_ids,
            annotation_cell_id_pairs=attempt.annotation_cell_id_pairs,
            annotation_bboxes=annotation_bboxes,
            annotation_points=annotation_points,
            annotation_point_pairs=annotation_point_pairs,
            projected_annotation=annotation_artifacts.projected_annotation,
            execution_extra={
                **dict(attempt.execution_extra),
                "answer": attempt.answer_value,
            },
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
            query_id=str(query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid Bingo scene after {max_attempts} attempts")


def resolve_bingo_task_integer_target(
    *,
    instance_seed: int,
    task_id: str,
    task_params: Mapping[str, Any],
    support_key: str,
    fallback_support: tuple[int, ...],
    namespace: str,
    explicit_key: str = "target_answer",
    balanced_flag_key: str = "balanced_target_answer_sampling",
) -> ResolvedBingoTarget:
    """Resolve a task-local integer target using that public task's config layer."""

    gen_defaults, _render_defaults, _prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(task_id),
    )
    return resolve_bingo_integer_target(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        balanced_flag_key=str(balanced_flag_key),
    )


def resolve_bingo_task_float_param(
    *,
    task_id: str,
    task_params: Mapping[str, Any],
    key: str,
    fallback: float,
) -> float:
    """Resolve a task-local float parameter using that public task's config layer."""

    gen_defaults, _render_defaults, _prompt_defaults = load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(task_id),
    )
    return resolve_bingo_float_param(
        task_params,
        gen_defaults=gen_defaults,
        key=str(key),
        fallback=float(fallback),
    )


__all__ = [
    "BingoAttemptResult",
    "BingoObjectivePlan",
    "bingo_named_count_trace_params",
    "bingo_target_trace_params",
    "resolve_bingo_task_float_param",
    "resolve_bingo_task_integer_target",
    "run_bingo_lifecycle",
]
