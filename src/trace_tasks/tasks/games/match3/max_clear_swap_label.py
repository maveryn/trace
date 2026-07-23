"""Choose the labeled match-3 swap that clears the most gems."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import Match3ObjectivePlan, Match3SingleQueryTaskBase, prepare_match3_swap_option_plan, run_match3_registered_task
from .shared.defaults import SCENE_ID
from .shared.prompts import make_match3_prompt_slots
from .shared.sampling import select_random_outcomes
from .shared.state import Match3SceneAxes, MoveOutcome


TASK_ID = "task_games__match3__max_clear_swap_label"
PROMPT_SLOTS = make_match3_prompt_slots(
    prompt_query_key="max_clear_swap_label",
    object_description_key="object_description_match3_grid",
    answer_hint_key="answer_hint_max_clear_swap_label",
    annotation_hint_key="annotation_hint_max_clear_swap_label",
    example_annotation=[456, 284],
    example_answer="C",
)


def _select_max_clear_outcomes(outcomes: tuple[MoveOutcome, ...], rng: Any):
    """Pick one maximum-clear answer and lower-clear distractors."""

    positive = [outcome for outcome in outcomes if int(outcome.clear_count) > 0]
    rng.shuffle(positive)
    for candidate in sorted(positive, key=lambda item: int(item.clear_count), reverse=True):
        lower = [
            outcome
            for outcome in outcomes
            if outcome.move.key != candidate.move.key and int(outcome.clear_count) < int(candidate.clear_count)
        ]
        if len(lower) >= 3:
            return candidate, select_random_outcomes(lower, rng, count=len(lower)), {}
    raise ValueError("no unique max-clear option set")


def _prepare_max_clear_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_branch: str,
    _branch_probabilities: Mapping[str, float],
    _axes: Match3SceneAxes,
    gen_defaults: Mapping[str, Any],
) -> Match3ObjectivePlan:
    """Resolve option axes and bind maximum-clear swap semantics."""

    namespace = f"{SCENE_ID}.max_clear_swap"
    fixed_task_params = {**dict(task_params), "option_count": 4}
    return prepare_match3_swap_option_plan(
        instance_seed=int(instance_seed),
        task_params=fixed_task_params,
        gen_defaults=gen_defaults,
        namespace=namespace,
        prompt_slots=PROMPT_SLOTS,
        outcome_selector=_select_max_clear_outcomes,
    )


@register_task
class GamesMatch3MaxClearSwapLabelTask(Match3SingleQueryTaskBase):
    """Choose the labeled match-3 swap that clears the most gems."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'state_update')
    _namespace = f"{SCENE_ID}.max_clear_swap"
    _default_branch = "single"
    _prepare_objective = staticmethod(_prepare_max_clear_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_match3_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMatch3MaxClearSwapLabelTask", "TASK_ID"]
