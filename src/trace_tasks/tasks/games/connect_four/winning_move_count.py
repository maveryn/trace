"""Count Connect Four legal drops that win immediately."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.connect_four._lifecycle import (
    ConnectFourCountObjectiveSpec,
    ConnectFourObjectivePlan,
    prepare_count_objective_from_spec,
    run_connect_four_lifecycle,
)
from trace_tasks.tasks.games.connect_four.shared.defaults import SCENE_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


TASK_ID = "task_games__connect_four__winning_move_count"
QUERY_ID = "winning_move_count"
PROMPT_QUERY_KEY = QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
WINNING_MOVE_COUNT_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4)
COUNT_OBJECTIVE_SPEC = ConnectFourCountObjectiveSpec(
    prompt_query_key=PROMPT_QUERY_KEY,
    count_mode="winning",
    support_key="winning_move_count_support",
    fallback_support=WINNING_MOVE_COUNT_SUPPORT,
    safe_board_defaults=False,
    include_safety_rule=False,
)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_winning_count_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ConnectFourObjectivePlan:
    """Bind immediate-win count semantics and exact-answer construction."""

    return prepare_count_objective_from_spec(
        task_id=TASK_ID,
        spec=COUNT_OBJECTIVE_SPEC,
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query_id=str(selected_query_id),
        gen_defaults=_GEN_DEFAULTS,
    )


@register_task
class GamesConnectFourWinningMoveCountTask:
    """Count legal Connect Four drop columns that win immediately."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate an exact immediate-winning legal-drop count."""

        return run_connect_four_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_winning_count_objective,
        )


__all__ = ["GamesConnectFourWinningMoveCountTask"]
