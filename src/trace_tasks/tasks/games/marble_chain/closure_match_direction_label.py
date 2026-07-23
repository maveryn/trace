"""Choose the marble-chain shot direction that creates a same-color closure."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    MarbleObjectivePlan,
    MarbleSingleQueryTaskBase,
    prepare_marble_state_direction_option_plan,
    run_marble_registered_task,
)
from .shared.defaults import SCENE_ID
from .shared.prompts import make_marble_prompt_slots
from .shared.rules import closure_creates_same_color_match
from .shared.state import MarbleOutcome, MarbleSceneAxes


TASK_ID = "task_games__marble_chain__closure_match_direction_label"
PROMPT_SLOTS = make_marble_prompt_slots(
    prompt_query_key="closure_match_direction_label",
    answer_hint_key="answer_hint_closure_match_direction_label",
    annotation_hint_key="annotation_hint_closure_match_direction_label",
    example_annotation=[465, 286],
    example_answer="C",
)


def _closure_match_slot_groups(
    chain_colors: Sequence[str],
    outcomes: Mapping[int, MarbleOutcome],
) -> tuple[list[int], list[int]]:
    """Return answer and distractor slots for same-color closure matching."""

    answer_slots = [
        int(slot)
        for slot, outcome in outcomes.items()
        if closure_creates_same_color_match(chain_colors, outcome)
    ]
    distractor_slots = [
        int(slot)
        for slot, outcome in outcomes.items()
        if not closure_creates_same_color_match(chain_colors, outcome)
    ]
    return answer_slots, distractor_slots


def _prepare_closure_match_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: MarbleSceneAxes,
    gen_defaults: Mapping[str, Any],
) -> MarbleObjectivePlan:
    """Resolve axes and bind closure-match option semantics."""

    namespace = f"{SCENE_ID}.closure_match_direction"
    return prepare_marble_state_direction_option_plan(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        namespace=namespace,
        prompt_slots=PROMPT_SLOTS,
        slot_group_builder=_closure_match_slot_groups,
        display_validator=lambda options: sum(1 for option in options if bool(option.is_answer)) == 1,
        extra_params={"closure_match_rule": "same_color_boundary_after_immediate_pop"},
    )


@register_task
class GamesMarbleChainClosureMatchDirectionLabelTask(MarbleSingleQueryTaskBase):
    """Choose the marble-chain shot direction that creates a same-color closure."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'topology', 'state_update', 'matching')
    _namespace = f"{SCENE_ID}.closure_match_direction"
    _prepare_objective = staticmethod(_prepare_closure_match_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_marble_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMarbleChainClosureMatchDirectionLabelTask", "TASK_ID"]
