"""Select the labeled chess move that gives immediate checkmate."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.games.shared.piece_board_rules import color_name
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import ChessObjectivePlan, keyed_checkmate_annotation_for_sample, run_chess_public_entry
from .shared.output import common_checkmate_trace_sections
from .shared.rendering import render_chess_task_scene
from .shared.sampling import resolve_checkmate_answer_label, resolve_integer_axis, sample_checkmate_scene
from .shared.state import CHESS_OPTION_LABELS, SCENE_ID


TASK_ID = "task_games__chess__checkmate_move_label"
QUERY_ID = "checkmate_move_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
OPTION_COUNT_SUPPORT = (4, 6)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_checkmate_move_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    _query_probabilities: Mapping[str, float],
) -> ChessObjectivePlan:
    """Bind visible move-option selection for an immediate checkmate."""

    option_count, option_count_support, option_count_probs = resolve_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="checkmate_option_count_support",
        explicit_key="option_count",
        fallback_support=OPTION_COUNT_SUPPORT,
        namespace=f"{TASK_ID}.option_count",
        balance_flag_key="balanced_checkmate_option_count_sampling",
        gen_defaults=_GEN_DEFAULTS,
    )
    option_label_support = tuple(CHESS_OPTION_LABELS[: int(option_count)])

    def construct_attempt(rng, axes):
        answer_label, answer_label_probs = resolve_checkmate_answer_label(
            rng,
            params=task_params,
            labels=option_label_support,
        )
        sample = sample_checkmate_scene(
            instance_seed=int(instance_seed),
            rng=rng,
            params=task_params,
            axes=axes,
            option_count=int(option_count),
            option_label_support=option_label_support,
            answer_label=str(answer_label),
        )
        return replace(sample, extra={**dict(sample.extra), "answer_label_probabilities": dict(answer_label_probs)})

    def render_sample(sample, params, seed):
        return render_chess_task_scene(
            board=sample.board,
            scene_variant=sample.scene_variant,
            style_variant=sample.style_variant,
            badge_text=f"{color_name(sample.player_color)} to move",
            marked_coord=None,
            params=params,
            instance_seed=int(seed),
            show_coordinates=True,
            move_options=tuple({"label": option.label, "text": option.text} for option in sample.options),
        )

    def prompt_slots(sample) -> dict[str, str]:
        return {
            "player_color_name": color_name(sample.player_color),
            "defender_color_name": color_name(sample.defender_color),
        }

    def query_params(sample) -> dict[str, Any]:
        return {
            "answer_support": [str(label) for label in option_label_support],
            "answer_label_probabilities": dict(sample.extra.get("answer_label_probabilities", {})),
            "answer_option_label": str(sample.correct_option.label),
            "player_color": str(sample.player_color),
        }

    return ChessObjectivePlan(
        attempt_namespace="games.chess.checkmate_move_label",
        prompt_query_key=str(query_id),
        query_params={
            "option_count": int(option_count),
            "option_count_support": [int(value) for value in option_count_support],
            "option_count_probabilities": dict(option_count_probs),
        },
        construct_attempt=construct_attempt,
        render_sample=render_sample,
        build_answer=lambda sample: TypedValue(type="option_letter", value=str(sample.correct_option.label)),
        build_annotation=keyed_checkmate_annotation_for_sample,
        build_trace_payload=lambda sample, rendered: common_checkmate_trace_sections(sample=sample, rendered_context=rendered),
        build_prompt_dynamic_slots=prompt_slots,
        build_query_params=query_params,
        execution_extra={"annotation_kind": "point_map"},
    )


@register_task
class GamesChessCheckmateMoveLabelTask:
    """Choose the visible move option that checkmates immediately."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = QUERY_ID
    prepare_objective = staticmethod(_prepare_checkmate_move_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate a checkmate move-option label scene."""

        return run_chess_public_entry(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesChessCheckmateMoveLabelTask"]
