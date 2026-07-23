"""Choose the labeled Connect Four column that blocks an immediate opponent win."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.connect_four._lifecycle import (
    prepare_column_label_objective_from_semantics,
    run_connect_four_lifecycle,
)
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from .shared.defaults import SCENE_ID
from .shared.sampling import sample_blocking_column_label_scene


TASK_ID = "task_games__connect_four__blocking_move_column_label"
QUERY_ID = "blocking_move_column_label"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def prepare_blocking_column_label_objective(instance_seed, task_params, selected_query_id, _query_probabilities):
    """Bind blocking-column label semantics and construction for this task."""

    return prepare_column_label_objective_from_semantics(
        task_id=TASK_ID,
        prompt_query_key=QUERY_ID,
        gen_defaults=_GEN_DEFAULTS,
        sample_scene=sample_blocking_column_label_scene,
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query_id=str(selected_query_id),
        include_opponent_player=True,
        line_trace_key="opponent_winning_line_coords",
        coord_trace_key="blocking_move_coords",
    )


@register_task
class GamesConnectFourBlockingMoveColumnLabelTask:
    """Return the visible column label for the immediate blocking drop."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one board with a unique column that blocks the opponent's immediate win."""

        task_params = dict(params)
        output = run_connect_four_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            prepare_objective=prepare_blocking_column_label_objective,
        )
        return output


__all__ = ["GamesConnectFourBlockingMoveColumnLabelTask", "TASK_ID"]
