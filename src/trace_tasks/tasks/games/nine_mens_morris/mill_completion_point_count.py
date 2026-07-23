"""Count empty points that complete a Nine Men's Morris mill."""

from __future__ import annotations

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.style import SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    NineMensMorrisObjectivePlan,
    morris_node_count_attempt,
    resolve_morris_count_target,
    run_morris_registered_task,
)
from .shared.sampling import (
    MILL_COMPLETION_COUNT_SUPPORT,
    resolve_nine_mens_morris_visual_axes,
    sample_mill_completion_board,
)


TASK_ID = "task_games__nine_mens_morris__mill_completion_point_count"
SUPPORTED_QUERY_IDS = (
    "white_mill_completion_point_count",
    "black_mill_completion_point_count",
)
QUERY_SPECS = {
    "white_mill_completion_point_count": ("white", "white_mill_completion_point_count_support"),
    "black_mill_completion_point_count": ("black", "black_mill_completion_point_count_support"),
}
TARGET_FALLBACK_SUPPORT = MILL_COMPLETION_COUNT_SUPPORT


def _completion_labels_for_color(board_state, *, color: str) -> tuple[str, ...]:
    """Return the completion-node labels for the requested piece color."""

    if str(color) == "white":
        return tuple(str(label) for label in board_state.white_mill_completion_node_labels)
    return tuple(str(label) for label in board_state.black_mill_completion_node_labels)


def _prepare_mill_completion_objective(
    instance_seed,
    task_params,
    selected_branch,
    branch_probabilities,
    gen_defaults,
) -> NineMensMorrisObjectivePlan:
    """Resolve color and exact-count axes for mill-completion construction."""

    del branch_probabilities
    color, support_key = QUERY_SPECS[str(selected_branch)]
    axes = resolve_nine_mens_morris_visual_axes(
        int(instance_seed),
        gen_defaults=gen_defaults,
        params=task_params,
        namespace=f"games.nine_mens_morris.mill_completion.{str(color)}",
        supported_style_variants=SUPPORTED_NINE_MENS_MORRIS_STYLE_VARIANTS,
    )
    target = resolve_morris_count_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        fallback_support=TARGET_FALLBACK_SUPPORT,
        namespace=f"games.nine_mens_morris.mill_completion.{str(color)}.target_answer",
    )

    def construct_attempt(rng, _resolved_axes):
        board_state = sample_mill_completion_board(
            rng=rng,
            color=str(color),
            target_answer=int(target.target_answer),
        )
        node_labels = _completion_labels_for_color(board_state, color=str(color))
        return morris_node_count_attempt(
            board_state=board_state,
            prompt_key=str(selected_branch),
            annotation_entity_ids=node_labels,
            color=str(color),
            target=target,
            extra_query_params={
                "target_answer": int(target.target_answer),
            },
        )

    return NineMensMorrisObjectivePlan(
        axes=axes,
        attempt_namespace=f"games.nine_mens_morris.mill_completion.{str(color)}",
        construct_attempt=construct_attempt,
    )


@register_task
class GamesNineMensMorrisMillCompletionPointCountTask:
    """Count empty points where one piece would complete a mill."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _default_branch = SUPPORTED_QUERY_IDS[0]
    _namespace = "games.nine_mens_morris.mill_completion"
    _prepare_objective = staticmethod(_prepare_mill_completion_objective)

    def generate(self, instance_seed: int, *, params: dict | None = None, max_attempts: int = 100) -> TaskOutput:
        """Generate a mill-completion count instance."""

        return run_morris_registered_task(
            self,
            int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
        )


__all__ = ["GamesNineMensMorrisMillCompletionPointCountTask"]
