"""Count visible rows or columns in cell-style board grids."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import (
    CounterfactualBoardObjectivePlan,
    run_counterfactual_board_lifecycle,
)
from .shared.sampling import build_counterfactual_board_case
from .shared.state import (
    CELL_BOARD_KIND,
    CHESS_CHECKERS_STYLE,
    COLUMN_AXIS,
    ROW_AXIS,
    SCENE_ID,
    SUDOKU_STYLE,
)

TASK_ID = "task_games__counterfactual_board__board_dimension_count"
ROW_COUNT_QUERY = "row_count"
COLUMN_COUNT_QUERY = "column_count"
SUPPORTED_QUERY_IDS = (ROW_COUNT_QUERY, COLUMN_COUNT_QUERY)
SUPPORTED_STYLES = (CHESS_CHECKERS_STYLE, SUDOKU_STYLE)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)

_AXIS_BY_QUERY = {ROW_COUNT_QUERY: ROW_AXIS, COLUMN_COUNT_QUERY: COLUMN_AXIS}


def _build_dimension_case(
    *,
    instance_seed: int,
    params,
    query_id: str,
):
    """Bind the row/column objective before shared rendering/projection."""

    counted_axis = _AXIS_BY_QUERY[str(query_id)]
    return build_counterfactual_board_case(
        instance_seed=int(instance_seed),
        params=params,
        supported_styles=SUPPORTED_STYLES,
        board_kind=CELL_BOARD_KIND,
        counted_axis=str(counted_axis),
        prompt_query_key=str(query_id),
        style_namespace="games.counterfactual_board.dimension.style",
        dimension_namespace="games.counterfactual_board.dimension.dimensions",
    )


@register_task
class GamesCounterfactualBoardDimensionCountTask:
    """Count visible rows or columns on the sampled board grid."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        selected_query, branch_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=ROW_COUNT_QUERY,
            task_id=TASK_ID,
            namespace="games.counterfactual_board.dimension.query",
        )
        return run_counterfactual_board_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            selected_query_id=str(selected_query),
            query_probabilities=branch_probs,
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            objective=CounterfactualBoardObjectivePlan(
                construct_case=lambda seed: _build_dimension_case(
                    instance_seed=int(seed),
                    params=task_params,
                    query_id=str(selected_query),
                ),
            ),
        )


__all__ = [
    "COLUMN_COUNT_QUERY",
    "GamesCounterfactualBoardDimensionCountTask",
    "ROW_COUNT_QUERY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
