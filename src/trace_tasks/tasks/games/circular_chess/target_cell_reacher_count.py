"""Count pieces that can reach a marked target cell on a circular chess board."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.circular_chess._lifecycle import (
    CircularChessObjectivePlan,
    run_circular_chess_lifecycle,
)
from trace_tasks.tasks.games.circular_chess.shared.defaults import SCENE_ID
from trace_tasks.tasks.games.circular_chess.shared.sampling import (
    resolve_task_target_answer,
    sample_target_reacher_scene,
)
from trace_tasks.tasks.games.shared.piece_board_rules import BLACK, WHITE
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


TASK_ID = "task_games__circular_chess__target_cell_reacher_count"
WHITE_REACHER_QUERY_ID = "white_piece_reaches_target_count"
BLACK_REACHER_QUERY_ID = "black_piece_reaches_target_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (WHITE_REACHER_QUERY_ID, BLACK_REACHER_QUERY_ID)
REACHER_COUNT_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _target_color_for_query(selected: str) -> str:
    if str(selected) == BLACK_REACHER_QUERY_ID:
        return BLACK
    return WHITE


def _prepare_target_reacher_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _query_probabilities: Mapping[str, float],
    scene_axes,
) -> CircularChessObjectivePlan:
    """Resolve target-side semantics and bind exact reacher-count construction."""

    target_color = _target_color_for_query(str(selected_query_id))
    target_answer, answer_support, answer_probabilities = resolve_task_target_answer(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key=f"{str(target_color)}_piece_reaches_target_count_support",
        fallback_support=REACHER_COUNT_SUPPORT,
        possible_max=6,
        namespace=f"{TASK_ID}.target_answer.{selected_query_id}",
    )

    def construct_attempt(rng):
        return sample_target_reacher_scene(
            rng=rng,
            scene_axes=scene_axes,
            target_color=str(target_color),
            target_answer=int(target_answer),
        )

    def query_spec_params(sample) -> dict[str, Any]:
        return {
            "target_color": str(target_color),
            "scene_variant": str(scene_axes.scene_variant),
            "style_variant": str(scene_axes.style_variant),
            "target_answer": int(sample.evaluation.answer),
            "target_answer_support": [int(value) for value in answer_support],
            "target_answer_probabilities": dict(answer_probabilities),
            "scene_variant_probabilities": dict(scene_axes.scene_variant_probabilities),
            "style_variant_probabilities": dict(scene_axes.style_variant_probabilities),
        }

    return CircularChessObjectivePlan(
        attempt_namespace=TASK_ID,
        construct_attempt=construct_attempt,
        prompt_target_color=str(target_color),
        prompt_marked_piece_present=False,
        prompt_example_answer=3,
        render_marked_coord=lambda _sample: None,
        render_target_coord=lambda sample: sample.evaluation.target_coord,
        query_spec_params=query_spec_params,
        execution_updates=lambda _sample: {"target_answer_support": [int(value) for value in answer_support]},
        relation_updates=lambda _sample: {},
    )


@register_task
class GamesCircularChessTargetCellReacherCountTask:
    """Count same-side pieces that can legally move to one marked target cell."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate an exact target-cell reacher count for one selected side."""

        return run_circular_chess_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=WHITE_REACHER_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_target_reacher_objective,
            post_image_noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
        )


__all__ = [
    "BLACK_REACHER_QUERY_ID",
    "GamesCircularChessTargetCellReacherCountTask",
    "WHITE_REACHER_QUERY_ID",
]
