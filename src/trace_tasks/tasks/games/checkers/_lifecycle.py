"""Scene-private lifecycle plumbing for Checkers public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import checkers_annotation_artifacts
from .shared.output import build_checkers_common_trace_payload, checkers_common_trace_params
from .shared.prompts import build_checkers_prompt_artifacts
from .shared.rendering import render_checkers_task_scene
from .shared.rules import player_name
from .shared.sampling import (
    ResolvedCheckersSceneAxes,
    resolve_checkers_scene_axes,
    resolve_checkers_target_answer,
    resolve_task_occupied_range,
)
from .shared.state import SCENE_ID, SampledCheckersScene, TargetAnswerAxis


AttemptBuilder = Callable[[Any, ResolvedCheckersSceneAxes], SampledCheckersScene]
ExecutionExtraBuilder = Callable[[SampledCheckersScene], Mapping[str, Any]]
PromptDynamicSlotBuilder = Callable[[SampledCheckersScene], Mapping[str, Any]]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "CheckersObjectivePlan"]
MoveConditionSampler = Callable[..., SampledCheckersScene]


@dataclass(frozen=True)
class CheckersObjectivePlan:
    """Prepared task-owned objective hooks for one generated Checkers instance."""

    attempt_namespace: str
    prompt_query_key: str
    target: TargetAnswerAxis
    query_params: Mapping[str, Any]
    prompt_dynamic_slots: Mapping[str, Any] = field(default_factory=dict)
    construct_attempt: AttemptBuilder | None = None
    execution_extra: Mapping[str, Any] = field(default_factory=dict)
    build_execution_extra: ExecutionExtraBuilder | None = None
    build_prompt_dynamic_slots: PromptDynamicSlotBuilder | None = None


def checkers_target_trace_params(target: TargetAnswerAxis) -> dict[str, Any]:
    """Return standard target-answer support/probability trace fields."""

    return {
        "target_answer": int(target.target_answer),
        "target_answer_support": [int(value) for value in target.target_answer_support],
        "target_answer_probabilities": dict(target.target_answer_probabilities),
    }


def resolve_checkers_task_target(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: tuple[int, ...],
    namespace: str,
) -> TargetAnswerAxis:
    """Resolve a task-local Checkers target answer."""

    return resolve_checkers_target_answer(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        gen_defaults=gen_defaults,
    )


def run_checkers_lifecycle(
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
    """Run neutral Checkers query, retry, render, prompt, trace, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_checkers_scene_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(query_id),
        dict(query_probabilities),
    )
    if objective.construct_attempt is None:
        raise ValueError("Checkers objective plan must provide a construct_attempt callback")

    last_error: ValueError | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, axes)
        except ValueError as exc:
            last_error = exc
            continue
        if int(sample.evaluation.answer) != int(objective.target.target_answer):
            last_error = ValueError("Checkers construction did not match the requested target answer")
            continue

        rendered_context = render_checkers_task_scene(
            axes=axes,
            sample=sample,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation_artifacts = checkers_annotation_artifacts(
            rendered_scene=rendered_context.rendered_scene,
            entity_ids=sample.evaluation.annotation_entity_ids,
            annotation_kind=sample.evaluation.annotation_kind,
        )
        sample_dynamic_slots = (
            dict(objective.build_prompt_dynamic_slots(sample))
            if objective.build_prompt_dynamic_slots is not None
            else {}
        )
        prompt_defaults, prompt_artifacts = build_checkers_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            dynamic_slots={**dict(objective.prompt_dynamic_slots), **sample_dynamic_slots},
            instance_seed=int(instance_seed),
        )
        answer_gt = TypedValue(type="integer", value=int(sample.evaluation.answer))
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=checkers_common_trace_params(
                axes,
                sample,
                extra_params={
                    **dict(objective.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                },
            ),
        )
        task_execution_extra = (
            dict(objective.build_execution_extra(sample))
            if objective.build_execution_extra is not None
            else {}
        )
        trace_payload = build_checkers_common_trace_payload(
            annotation_artifacts=annotation_artifacts,
            axes=axes,
            sample=sample,
            rendered_context=rendered_context,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            prompt_query_spec=query_spec,
            execution_extra={
                **dict(objective.execution_extra),
                **task_execution_extra,
                "query_id": str(query_id),
                "prompt_query_key": str(objective.prompt_query_key),
                "target_answer": int(objective.target.target_answer),
                "answer": int(answer_gt.value),
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
            query_id=str(query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid Checkers scene after {max_attempts} attempts") from last_error


def current_player_prompt_slots(sample_player: int) -> dict[str, str]:
    """Return common dynamic prompt slots for current-player Checkers tasks."""

    from .shared.sampling import movement_rule_text

    return {
        "current_player_name": player_name(int(sample_player)),
        "movement_rule_text": movement_rule_text(int(sample_player)),
    }


def prepare_checkers_move_condition_objective(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    task_id: str,
    query_id: str,
    support_key: str,
    fallback_support: tuple[int, ...],
    capture_only: bool,
    count_trace_keys: tuple[str, str],
    gen_defaults: Mapping[str, Any],
    attempt_namespace: str,
    sample_scene: MoveConditionSampler,
) -> CheckersObjectivePlan:
    """Prepare a current-player move-condition count without query routing."""

    from .shared.sampling import scene_object_description

    target = resolve_checkers_task_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=f"{task_id}.target_answer.{str(query_id)}",
    )
    occupied_range = resolve_task_occupied_range(
        params=task_params,
        gen_defaults=gen_defaults,
    )

    def construct_attempt(rng, axes):
        return sample_scene(
            rng=rng,
            axes=axes,
            params=task_params,
            target_answer=int(target.target_answer),
            capture_only=bool(capture_only),
            occupied_range=occupied_range,
        )

    def prompt_slots(sample: SampledCheckersScene) -> dict[str, str]:
        return {
            "object_description": scene_object_description(str(sample.scene_variant)),
            **current_player_prompt_slots(int(sample.current_player)),
        }

    def execution_fields(sample: SampledCheckersScene) -> dict[str, Any]:
        legal_count_key, capture_count_key = count_trace_keys
        return {
            "capture_only": bool(capture_only),
            str(legal_count_key): int(len(sample.evaluation.legal_moves)),
            str(capture_count_key): int(len(sample.evaluation.capture_moves)),
        }

    return CheckersObjectivePlan(
        attempt_namespace=str(attempt_namespace),
        prompt_query_key=str(query_id),
        target=target,
        query_params={
            **checkers_target_trace_params(target),
            "capture_only": bool(capture_only),
        },
        construct_attempt=construct_attempt,
        build_prompt_dynamic_slots=prompt_slots,
        build_execution_extra=execution_fields,
    )


__all__ = [
    "CheckersObjectivePlan",
    "checkers_target_trace_params",
    "current_player_prompt_slots",
    "prepare_checkers_move_condition_objective",
    "resolve_checkers_task_target",
    "run_checkers_lifecycle",
]
