"""Find the largest connected component size for one named color."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import run_single_query_cell_board_task
from .shared.rules import build_largest_component_case
from .shared.state import SCENE_ID

TASK_ID = "task_puzzles__cell_board__largest_component_size"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "largest_component_size"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = load_scene_generation_rendering_prompt_defaults(
    "puzzles", SCENE_ID, task_id=TASK_ID
)


def _build_largest_component_case(*, instance_seed: int, params):
    """Bind the largest-component objective to the color-board builder."""

    return build_largest_component_case(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        prompt_query_key=PROMPT_QUERY_KEY,
    )


@register_task
class PuzzlesCellBoardLargestComponentSizeTask:
    """Return the size of the largest target-color connected component."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
    domain = "puzzles"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_single_query_cell_board_task(
            task_id=TASK_ID,
            domain=self.domain,
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            prompt_query_key=PROMPT_QUERY_KEY,
            construct_case=lambda seed: _build_largest_component_case(
                instance_seed=int(seed),
                params=params,
            ),
            namespace="puzzles.cell_board.largest_component.query",
        )


__all__ = ["PuzzlesCellBoardLargestComponentSizeTask"]
