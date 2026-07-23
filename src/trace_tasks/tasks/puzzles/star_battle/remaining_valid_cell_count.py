"""Count legal remaining Star Battle cells in one marked scope."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import bind_remaining_count_output, run_star_battle_public_task
from .shared.sampling import build_remaining_count_dataset
from .shared.state import DOMAIN, SCENE_ID, StarBattleDataset


TASK_ID = "task_puzzles__star_battle__remaining_valid_cell_count"
SUPPORTED_QUERY_IDS = (
    "remaining_valid_cells_in_marked_row_count",
    "remaining_valid_cells_in_marked_column_count",
)
PROMPT_TASK_KEY = "star_battle_remaining_count_query"
SCOPE_KIND_BY_QUERY = {
    "remaining_valid_cells_in_marked_row_count": "marked_row",
    "remaining_valid_cells_in_marked_column_count": "marked_column",
}
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _build_dataset_for_query(
    *,
    query_id: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> StarBattleDataset:
    """Translate the selected marked scope into a count-target sampling program."""

    return build_remaining_count_dataset(
        scope_kind=str(SCOPE_KIND_BY_QUERY[str(query_id)]),
        params=params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )


def _bind_output(
    *,
    dataset: StarBattleDataset,
    context: Any,
    query_id: str,
    query_probabilities: Mapping[str, float],
) -> Any:
    """Bind bbox-set annotation for all counted legal cells."""

    return bind_remaining_count_output(
        dataset=dataset,
        context=context,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        scope_kind=str(SCOPE_KIND_BY_QUERY[str(query_id)]),
    )


@register_task
class PuzzlesStarBattleRemainingValidCellCountTask:
    """Count legal remaining Star Battle cells in a marked row or column."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one marked-scope Star Battle count task."""

        return run_star_battle_public_task(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            prompt_task_key=PROMPT_TASK_KEY,
            params=params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            dataset_builder=_build_dataset_for_query,
            output_binder=_bind_output,
        )


__all__ = [
    "PROMPT_TASK_KEY",
    "PuzzlesStarBattleRemainingValidCellCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
