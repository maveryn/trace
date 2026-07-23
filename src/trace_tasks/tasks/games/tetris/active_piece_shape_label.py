"""Active falling-piece shape label task for Tetris."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import TetrisObjectivePlan, run_tetris_lifecycle
from .shared.prompts import format_json_examples
from .shared.rendering import RENDER_MODE_ACTIVE_SHAPE
from .shared.sampling import build_active_piece_shape_sample
from .shared.state import PIECE_ORDER


TASK_ID = "task_games__tetris__active_piece_shape_label"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
PROMPT_QUERY_KEY = "active_piece_shape_label"
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(annotation=[246, 92, 352, 166], answer="B")


def _target_piece(instance_seed: int, task_params: Mapping[str, Any]) -> tuple[str, dict[str, float]]:
    """Resolve the task-owned target tetromino label."""

    labels = tuple(str(piece) for piece in PIECE_ORDER)
    raw = task_params.get("target_piece")
    if raw is not None:
        piece = str(raw).strip().upper()
        if piece not in labels:
            raise ValueError(f"unsupported Tetris target_piece: {raw}")
        return str(piece), {str(label): 1.0 / float(len(labels)) for label in labels}
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.target_piece")
    return str(uniform_choice(rng, labels)), {str(label): 1.0 / float(len(labels)) for label in labels}


def _prepare_active_piece_shape_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query: str,
    query_probabilities: Mapping[str, float],
    axes,
) -> TetrisObjectivePlan:
    """Bind the target shape answer for the active falling piece."""

    if str(selected_query) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported Tetris active-piece query: {selected_query}")
    target_piece, piece_probabilities = _target_piece(int(instance_seed), task_params)

    def construct_attempt(rng, resolved_axes):
        return build_active_piece_shape_sample(
            rng,
            scene_variant=str(resolved_axes.scene_variant),
            board_rows=int(resolved_axes.board_rows),
            board_cols=int(resolved_axes.board_cols),
            target_piece=str(target_piece),
        )

    return TetrisObjectivePlan(
        attempt_namespace=f"games.tetris.active_piece_shape.{str(target_piece)}",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint_key="answer_hint_active_piece_shape_label",
        annotation_hint_key="annotation_hint_active_piece_shape_label",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        render_mode=RENDER_MODE_ACTIVE_SHAPE,
        query_params={
            "target_piece": str(target_piece),
            "target_piece_probabilities": dict(piece_probabilities),
            "shape_option_count": 4,
            "shape_option_policy": "four_text_options_in_image",
            "active_piece_query_probabilities": dict(query_probabilities),
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesTetrisActivePieceShapeLabelTask:
    """Identify the tetromino shape of the shown falling piece."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_tetris_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_active_piece_shape_objective,
        )


__all__ = ["GamesTetrisActivePieceShapeLabelTask"]
