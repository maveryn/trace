"""Identify the labeled hidden Minesweeper cell forced to be a mine."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    MinesweeperObjectivePlan,
    minesweeper_option_letter_point_attempt,
    run_minesweeper_registered_task,
)
from .shared.defaults import DEFAULT_BRANCH_ID
from .shared.sampling import resolve_minesweeper_axes, sample_forced_mine_option_scene


TASK_ID = "task_games__minesweeper__forced_mine_cell_label"
QUERY_ID = DEFAULT_BRANCH_ID
PROMPT_QUERY_KEY = "forced_mine_cell_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
OPTION_LABELS = ("A", "B", "C", "D")
BOARD_SIZE_SUPPORT_KEY = "forced_mine_cell_label_board_size_support"
BOARD_SIZE_FALLBACK_SUPPORT = (4, 5)
TARGET_SUPPORT_KEY = "forced_mine_cell_label_support"
TARGET_FALLBACK_SUPPORT = (0, 1, 2, 3)


def _prepare_forced_mine_cell_label_objective(
    instance_seed,
    task_params,
    _selected_branch,
    branch_probabilities,
    gen_defaults,
) -> MinesweeperObjectivePlan:
    """Resolve option-label axes and bind forced-mine label construction."""

    axes = resolve_minesweeper_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        namespace="games.minesweeper.forced_mine_cell_label",
        params=task_params,
        branch_probabilities=branch_probabilities,
        board_size_support_key=BOARD_SIZE_SUPPORT_KEY,
        board_size_fallback_support=BOARD_SIZE_FALLBACK_SUPPORT,
        target_support_key=TARGET_SUPPORT_KEY,
        target_fallback_support=TARGET_FALLBACK_SUPPORT,
    )

    def construct_attempt(rng, resolved_axes):
        target_index = 0 if resolved_axes.target_answer is None else int(resolved_axes.target_answer)
        sample = sample_forced_mine_option_scene(
            rng=rng,
            axes=resolved_axes,
            target_label_index=int(target_index),
            option_labels=OPTION_LABELS,
        )
        correct_coord = sample.annotation_coords[0]
        return minesweeper_option_letter_point_attempt(
            sample=sample,
            prompt_key=PROMPT_QUERY_KEY,
            object_description_key=f"object_description_{str(resolved_axes.scene_variant)}",
            answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
            annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
            example_annotation=[245, 255],
            example_answer="B",
            coord=correct_coord,
            extra_query_params={"prompt_query_key": PROMPT_QUERY_KEY},
            execution_extra={
                "option_labels": [str(label) for label in OPTION_LABELS],
                "correct_option_label": str(sample.answer),
                "correct_option_coord": [int(correct_coord[0]), int(correct_coord[1])],
            },
        )

    return MinesweeperObjectivePlan(
        axes=axes,
        attempt_namespace="games.minesweeper.forced_mine_cell_label",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesMinesweeperForcedMineCellLabelTask:
    """Choose the labeled hidden cell that is guaranteed to be a mine."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = QUERY_ID
    _namespace = "games.minesweeper.forced_mine_cell_label"
    _prepare_objective = staticmethod(_prepare_forced_mine_cell_label_objective)

    def generate(self, instance_seed, *, params=None, max_attempts=100):
        """Generate a forced-mine hidden-cell option task instance."""

        return run_minesweeper_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesMinesweeperForcedMineCellLabelTask"]
