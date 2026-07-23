"""Choose the labeled Connect Four column that wins immediately."""

from __future__ import annotations

from typing import Any, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.connect_four._lifecycle import (
    prepare_column_label_objective_from_semantics,
    run_connect_four_lifecycle,
)
from trace_tasks.tasks.games.connect_four.shared.defaults import SCENE_ID
from trace_tasks.tasks.games.connect_four.shared.sampling import sample_winning_column_label_scene
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


TASK_ID = "task_games__connect_four__winning_move_column_label"
QUERY_ID = "winning_move_column_label"
PROMPT_QUERY_KEY = QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_winning_column_label_objective(
    instance_seed: int,
    task_params,
    selected_query_id: str,
    _query_probabilities,
):
    """Bind single winning-column label semantics and construction."""

    return prepare_column_label_objective_from_semantics(
        task_id=TASK_ID,
        prompt_query_key=PROMPT_QUERY_KEY,
        gen_defaults=_GEN_DEFAULTS,
        sample_scene=sample_winning_column_label_scene,
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query_id=str(selected_query_id),
        include_opponent_player=False,
        line_trace_key="winning_line_coords",
        coord_trace_key="winning_move_coords",
    )


@register_task
class GamesConnectFourWinningMoveColumnLabelTask:
    """Return the visible column label for the immediate winning drop."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one board with a unique immediate winning column."""

        task_params = dict(params)
        output = run_connect_four_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_winning_column_label_objective,
        )
        return output


__all__ = ["GamesConnectFourWinningMoveColumnLabelTask", "TASK_ID"]
