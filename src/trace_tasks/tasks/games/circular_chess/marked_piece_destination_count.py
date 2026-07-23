"""Count destinations or captures for one marked circular-chess piece."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.circular_chess._lifecycle import (
    CircularChessObjectivePlan,
    run_circular_chess_lifecycle,
)
from trace_tasks.tasks.games.circular_chess.shared.defaults import SCENE_ID
from trace_tasks.tasks.games.circular_chess.shared.rules import max_possible_marked_destination_answer
from trace_tasks.tasks.games.circular_chess.shared.sampling import (
    resolve_marked_piece_axes,
    resolve_task_target_answer,
    sample_marked_destination_scene,
)
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


TASK_ID = "task_games__circular_chess__marked_piece_destination_count"
MARKED_MOVE_QUERY_ID = "marked_piece_move_count"
MARKED_CAPTURE_QUERY_ID = "marked_piece_capture_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (MARKED_MOVE_QUERY_ID, MARKED_CAPTURE_QUERY_ID)
MOVE_COUNT_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8)
CAPTURE_COUNT_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class MarkedDestinationQuerySpec:
    """Task-local contract for one marked-piece destination query."""

    query_id: str
    destination_mode: str
    support_key: str
    fallback_support: Tuple[int, ...]
    example_answer: int


MARKED_DESTINATION_QUERY_SPECS: Mapping[str, MarkedDestinationQuerySpec] = {
    MARKED_MOVE_QUERY_ID: MarkedDestinationQuerySpec(
        query_id=MARKED_MOVE_QUERY_ID,
        destination_mode="move",
        support_key="marked_piece_move_count_support",
        fallback_support=MOVE_COUNT_SUPPORT,
        example_answer=3,
    ),
    MARKED_CAPTURE_QUERY_ID: MarkedDestinationQuerySpec(
        query_id=MARKED_CAPTURE_QUERY_ID,
        destination_mode="capture",
        support_key="marked_piece_capture_count_support",
        fallback_support=CAPTURE_COUNT_SUPPORT,
        example_answer=2,
    ),
}


def _marked_query_spec(selected: str) -> MarkedDestinationQuerySpec:
    return MARKED_DESTINATION_QUERY_SPECS.get(str(selected), MARKED_DESTINATION_QUERY_SPECS[MARKED_MOVE_QUERY_ID])


def _prepare_marked_piece_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _query_probabilities: Mapping[str, float],
    scene_axes,
) -> CircularChessObjectivePlan:
    """Resolve the marked-piece query and bind exact-count construction."""

    query_spec = _marked_query_spec(str(selected_query_id))
    marked_axes = resolve_marked_piece_axes(int(instance_seed), params=task_params)
    possible_max = max_possible_marked_destination_answer(
        destination_mode=str(query_spec.destination_mode),
        piece_kind=str(marked_axes.piece_kind),
    )
    target_answer, answer_support, answer_probabilities = resolve_task_target_answer(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key=str(query_spec.support_key),
        fallback_support=query_spec.fallback_support,
        possible_max=int(possible_max),
        namespace=f"{TASK_ID}.target_answer.{selected_query_id}",
    )

    def construct_attempt(rng):
        return sample_marked_destination_scene(
            rng=rng,
            scene_axes=scene_axes,
            marked_axes=marked_axes,
            destination_mode=str(query_spec.destination_mode),
            target_answer=int(target_answer),
        )

    def common_params(sample) -> dict[str, Any]:
        return {
            "destination_mode": str(query_spec.destination_mode),
            "marked_piece_kind": str(marked_axes.piece_kind),
            "marked_piece_color": str(marked_axes.piece_color),
            "target_answer": int(sample.evaluation.answer),
            "target_answer_support": [int(value) for value in answer_support],
            "target_answer_probabilities": dict(answer_probabilities),
        }

    def query_spec_params(sample) -> dict[str, Any]:
        return {
            **common_params(sample),
            "scene_variant": str(scene_axes.scene_variant),
            "style_variant": str(scene_axes.style_variant),
            "scene_variant_probabilities": dict(scene_axes.scene_variant_probabilities),
            "style_variant_probabilities": dict(scene_axes.style_variant_probabilities),
            "piece_kind_probabilities": dict(marked_axes.piece_kind_probabilities),
            "marked_piece_color_probabilities": dict(marked_axes.piece_color_probabilities),
        }

    return CircularChessObjectivePlan(
        attempt_namespace=TASK_ID,
        construct_attempt=construct_attempt,
        prompt_target_color="",
        prompt_marked_piece_present=True,
        prompt_example_answer=int(query_spec.example_answer),
        render_marked_coord=lambda sample: sample.evaluation.marked_coord,
        render_target_coord=lambda _sample: None,
        query_spec_params=query_spec_params,
        execution_updates=common_params,
        relation_updates=lambda _sample: {
            "destination_mode": str(query_spec.destination_mode),
            "marked_piece_kind": str(marked_axes.piece_kind),
        },
    )


@register_task
class GamesCircularChessMarkedPieceDestinationCountTask:
    """Count legal destinations or capture destinations for a marked piece."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate an exact marked-piece count while this task owns query semantics."""

        return run_circular_chess_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=MARKED_MOVE_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_marked_piece_objective,
            post_image_noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
        )


__all__ = [
    "GamesCircularChessMarkedPieceDestinationCountTask",
    "MARKED_CAPTURE_QUERY_ID",
    "MARKED_MOVE_QUERY_ID",
]
