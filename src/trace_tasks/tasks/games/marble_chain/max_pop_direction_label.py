"""Choose the marble-chain shot direction that pops the most marbles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import MarbleObjectivePlan, MarbleSingleQueryTaskBase, prepare_marble_direction_option_plan, run_marble_registered_task
from .shared.defaults import SCENE_ID
from .shared.prompts import make_marble_prompt_slots
from .shared.state import MarbleOutcome, MarbleSceneAxes, ShotOption


TASK_ID = "task_games__marble_chain__max_pop_direction_label"
PROMPT_SLOTS = make_marble_prompt_slots(
    prompt_query_key="max_pop_direction_label",
    answer_hint_key="answer_hint_max_pop_direction_label",
    annotation_hint_key="annotation_hint_max_pop_direction_label",
    example_annotation=[465, 286],
    example_answer="C",
)


def _max_pop_slot_groups(outcomes: Mapping[int, MarbleOutcome]) -> tuple[list[int], list[int]]:
    """Return answer and distractor slots for the maximum-pop objective."""

    max_pop = max(int(outcome.pop_count) for outcome in outcomes.values())
    if int(max_pop) <= 0:
        return [], []
    return (
        [slot for slot, outcome in outcomes.items() if int(outcome.pop_count) == int(max_pop)],
        [slot for slot, outcome in outcomes.items() if int(outcome.pop_count) < int(max_pop)],
    )


def _has_unique_displayed_max_pop(options: Sequence[ShotOption]) -> bool:
    """Check that exactly one displayed arrow has the largest pop count."""

    displayed_max = max(int(option.outcome.pop_count) for option in options)
    return sum(1 for option in options if int(option.outcome.pop_count) == int(displayed_max)) == 1


def _prepare_max_pop_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: MarbleSceneAxes,
    gen_defaults: Mapping[str, Any],
) -> MarbleObjectivePlan:
    """Resolve axes and bind maximum-pop option semantics."""

    namespace = f"{SCENE_ID}.max_pop_direction"
    return prepare_marble_direction_option_plan(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        namespace=namespace,
        prompt_slots=PROMPT_SLOTS,
        slot_group_builder=_max_pop_slot_groups,
        display_validator=_has_unique_displayed_max_pop,
    )


@register_task
class GamesMarbleChainMaxPopDirectionLabelTask(MarbleSingleQueryTaskBase):
    """Choose the marble-chain shot direction that pops the most marbles."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology', 'state_update')
    _namespace = f"{SCENE_ID}.max_pop_direction"
    _prepare_objective = staticmethod(_prepare_max_pop_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_marble_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMarbleChainMaxPopDirectionLabelTask", "TASK_ID"]
