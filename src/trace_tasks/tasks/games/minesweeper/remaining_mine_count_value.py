"""Count remaining mines needed around one opened Minesweeper clue."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import MinesweeperObjectivePlan, minesweeper_integer_point_attempt, run_minesweeper_registered_task
from .shared.defaults import DEFAULT_BRANCH_ID
from .shared.sampling import resolve_minesweeper_axes, sample_remaining_adjacent_mine_scene


TASK_ID = "task_games__minesweeper__remaining_mine_count_value"
QUERY_ID = DEFAULT_BRANCH_ID
PROMPT_QUERY_KEY = "remaining_mine_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
BOARD_SIZE_SUPPORT_KEY = "board_size_support"
BOARD_SIZE_FALLBACK_SUPPORT = (4, 5, 6, 7, 8)
TARGET_SUPPORT_KEY = "remaining_mine_count_support"
TARGET_FALLBACK_SUPPORT = (0, 1, 2, 3, 4, 5)


def _prepare_remaining_mine_objective(
    instance_seed,
    task_params,
    _selected_branch,
    branch_probabilities,
    gen_defaults,
) -> MinesweeperObjectivePlan:
    """Resolve marked-clue axes and bind remaining adjacent mine construction."""

    axes = resolve_minesweeper_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace="games.minesweeper.remaining_adjacent_mines",
        params=task_params,
        branch_probabilities=branch_probabilities,
        board_size_support_key=BOARD_SIZE_SUPPORT_KEY,
        board_size_fallback_support=BOARD_SIZE_FALLBACK_SUPPORT,
        target_support_key=TARGET_SUPPORT_KEY,
        target_fallback_support=TARGET_FALLBACK_SUPPORT,
    )

    def construct_attempt(rng, resolved_axes):
        sample = sample_remaining_adjacent_mine_scene(
            rng=rng,
            axes=resolved_axes,
            target_count=int(resolved_axes.target_answer or 0),
        )
        marked_coord = sample.annotation_coords[0]
        return minesweeper_integer_point_attempt(
            sample=sample,
            prompt_key=PROMPT_QUERY_KEY,
            object_description_key=f"object_description_{str(resolved_axes.scene_variant)}",
            answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
            annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
            example_annotation=[245, 255],
            example_answer=2,
            coord=marked_coord,
            highlighted_clue_coords=tuple(sample.forcing_clue_coords),
            extra_query_params={"prompt_query_key": PROMPT_QUERY_KEY},
        )

    return MinesweeperObjectivePlan(
        axes=axes,
        attempt_namespace="games.minesweeper.remaining_adjacent_mines",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinesweeperRemainingMineCountValueTask:
    """Count how many more mines a marked opened clue still needs."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = QUERY_ID
    _namespace = "games.minesweeper.remaining_adjacent_mines"
    _prepare_objective = staticmethod(_prepare_remaining_mine_objective)

    def generate(self, instance_seed, *, params=None, max_attempts=100):
        """Generate a marked-clue remaining mine count task instance."""

        return run_minesweeper_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesMinesweeperRemainingMineCountValueTask"]
