"""Count hidden Minesweeper cells forced to be mines or safe."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    MinesweeperObjectivePlan,
    minesweeper_integer_bbox_set_attempt,
    run_minesweeper_registered_task,
)
from .shared.sampling import resolve_minesweeper_axes, sample_forced_cell_scene


TASK_ID = "task_games__minesweeper__forced_cell_count"
FORCED_MINE_QUERY_ID = "forced_mine_count"
FORCED_SAFE_QUERY_ID = "forced_safe_count"
SUPPORTED_QUERY_IDS = (FORCED_MINE_QUERY_ID, FORCED_SAFE_QUERY_ID)
QUERY_SUPPORTS = {
    FORCED_MINE_QUERY_ID: ("forced_mine_count_support", (1, 2, 3, 4, 5), "mine"),
    FORCED_SAFE_QUERY_ID: ("forced_safe_count_support", (1, 2, 3, 4, 5), "safe"),
}
BOARD_SIZE_SUPPORT_KEY = "forced_cell_board_size_support"
BOARD_SIZE_FALLBACK_SUPPORT = (4, 5)


def _prepare_forced_cell_objective(
    instance_seed,
    task_params,
    selected_branch,
    branch_probabilities,
    gen_defaults,
) -> MinesweeperObjectivePlan:
    """Resolve forced-cell axes and bind mine/safe count construction."""

    support_key, fallback_support, force_kind = QUERY_SUPPORTS[str(selected_branch)]
    axes = resolve_minesweeper_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace=f"games.minesweeper.forced_cell.{str(selected_branch)}",
        params=task_params,
        branch_probabilities=branch_probabilities,
        board_size_support_key=BOARD_SIZE_SUPPORT_KEY,
        board_size_fallback_support=BOARD_SIZE_FALLBACK_SUPPORT,
        target_support_key=str(support_key),
        target_fallback_support=tuple(int(value) for value in fallback_support),
    )

    def construct_attempt(rng, resolved_axes):
        sample = sample_forced_cell_scene(
            rng=rng,
            axes=resolved_axes,
            force_kind=str(force_kind),
            target_count=int(resolved_axes.target_answer or 1),
        )
        return minesweeper_integer_bbox_set_attempt(
            sample=sample,
            prompt_key=str(selected_branch),
            object_description_key=f"object_description_{str(resolved_axes.scene_variant)}",
            answer_hint_key=f"answer_hint_{str(selected_branch)}",
            annotation_hint_key=f"annotation_hint_{str(selected_branch)}",
            example_annotation=[[147, 227, 203, 283], [217, 227, 273, 283], [287, 227, 343, 283]],
            example_answer=3,
            coords=sample.annotation_coords,
            highlighted_clue_coords=tuple(sample.forcing_clue_coords),
            extra_query_params={"prompt_query_key": str(selected_branch), "force_kind": str(force_kind)},
            execution_extra={"force_kind": str(force_kind)},
        )

    return MinesweeperObjectivePlan(
        axes=axes,
        attempt_namespace=f"games.minesweeper.forced_cell.{str(selected_branch)}",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinesweeperForcedCellCountTask:
    """Count hidden cells forced by visible Minesweeper clues."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = FORCED_MINE_QUERY_ID
    _namespace = "games.minesweeper.forced_cell"
    _prepare_objective = staticmethod(_prepare_forced_cell_objective)

    def generate(self, instance_seed, *, params=None, max_attempts=100):
        """Generate a forced mine-or-safe cell count task instance."""

        return run_minesweeper_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesMinesweeperForcedCellCountTask"]
