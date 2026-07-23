"""Scene-private lifecycle plumbing for Backgammon public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
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

from .shared.annotations import (
    annotation_bboxes_for_entity_ids,
    annotation_entity_ids_for_points,
)
from .shared.output import (
    build_backgammon_trace_payload,
    common_backgammon_trace_params,
)
from .shared.prompts import build_backgammon_prompt_artifacts
from .shared.rendering import render_backgammon_sample
from .shared.sampling import ResolvedBackgammonAxes, resolve_backgammon_axes
from .shared.state import SCENE_ID, BackgammonSample

AttemptBuilder = Callable[[Any, ResolvedBackgammonAxes], "BackgammonAttemptResult"]
PromptDynamicSlotBuilder = Callable[[BackgammonSample], Mapping[str, Any]]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str],
    "BackgammonObjectivePlan",
]


@dataclass(frozen=True)
class BackgammonAttemptResult:
    """Task-owned result of one constructed Backgammon sample."""

    sample: BackgammonSample
    answer_gt: TypedValue
    target_points: tuple[int, ...]
    construction_mode: str
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackgammonObjectivePlan:
    """Prepared task-owned objective hooks for one generated Backgammon instance."""

    attempt_namespace: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder
    prompt_query_key: str | None = None
    prompt_dynamic_slots: Mapping[str, Any] = field(default_factory=dict)
    prompt_dynamic_slot_builder: PromptDynamicSlotBuilder | None = None


@dataclass(frozen=True)
class BackgammonCountTarget:
    """Resolved integer-answer axis for a Backgammon count objective."""

    target_answer: int
    target_answer_support: tuple[int, ...]
    target_answer_probabilities: Mapping[str, float]


def resolve_backgammon_count_target(
    *,
    instance_seed: int,
    task_params: Mapping[str, Any],
    task_id: str,
    support_key: str,
    fallback_support: tuple[int, ...],
    namespace: str,
) -> BackgammonCountTarget:
    """Resolve a task-owned Backgammon integer target from config or sampling."""

    gen_defaults, _render_defaults, _prompt_defaults = (
        load_scene_generation_rendering_prompt_defaults(
            "games",
            SCENE_ID,
            task_id=str(task_id),
        )
    )
    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    target_answer_support = resolve_integer_support(
        task_params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return BackgammonCountTarget(
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in target_answer_support),
        target_answer_probabilities=dict(target_answer_probabilities),
    )


def integer_count_attempt_result(
    *,
    sample: BackgammonSample,
    target_points: tuple[int, ...],
    construction_mode: str,
) -> BackgammonAttemptResult:
    """Wrap a task-constructed count sample with answer and annotation targets."""

    return BackgammonAttemptResult(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        target_points=tuple(int(point) for point in target_points),
        construction_mode=str(construction_mode),
        execution_extra={"answer": int(sample.answer)},
    )


def run_backgammon_lifecycle(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    domain: str = "games",
) -> TaskOutput:
    """Run query selection, render/prompt assembly, retry plumbing, and output assembly."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    axes = resolve_backgammon_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(query_id),
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        rendered_context = render_backgammon_sample(
            sample=attempt.sample,
            params=task_params,
            instance_seed=int(instance_seed),
        )
        annotation_entity_ids = annotation_entity_ids_for_points(
            tuple(int(point) for point in attempt.target_points)
        )
        annotation_bboxes = annotation_bboxes_for_entity_ids(
            rendered_context.rendered_scene,
            annotation_entity_ids,
        )
        annotation_gt = TypedValue(
            type="bbox_set", value=[list(bbox) for bbox in annotation_bboxes]
        )
        prompt_query_key = str(objective.prompt_query_key or query_id)
        dynamic_slots = dict(objective.prompt_dynamic_slots)
        if objective.prompt_dynamic_slot_builder is not None:
            dynamic_slots.update(dict(objective.prompt_dynamic_slot_builder(attempt.sample)))
        prompt_defaults, prompt_artifacts = build_backgammon_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=prompt_query_key,
            dynamic_slots=dynamic_slots,
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=common_backgammon_trace_params(
                axes=axes,
                sample=attempt.sample,
                extra_params={
                    **dict(objective.query_params),
                    "query_id_probabilities": dict(query_probabilities),
                    "prompt_query_key": prompt_query_key,
                },
            ),
        )
        trace_payload = build_backgammon_trace_payload(
            annotation_gt=annotation_gt,
            annotation_entity_ids=annotation_entity_ids,
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
            construction_mode=str(attempt.construction_mode),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=annotation_gt,
            image=rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )

    raise RuntimeError(
        f"{task_id} failed to generate a valid Backgammon scene after {max_attempts} attempts"
    )


__all__ = [
    "BackgammonAttemptResult",
    "BackgammonCountTarget",
    "BackgammonObjectivePlan",
    "integer_count_attempt_result",
    "resolve_backgammon_count_target",
    "run_backgammon_lifecycle",
]
