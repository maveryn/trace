"""Choose the Ludo destination option reached by a shown roll sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import LudoSingleQueryTaskBase, build_ludo_bound_attempt, make_ludo_value_option_preparer, run_ludo_registered_task
from .shared.rules import roll_sequence_for_total, route_for_color
from .shared.sampling import make_ludo_value_option_axis_config, make_destination_options, sample_other_token_coords
from .shared.state import DEFAULTS, PLAYER_COLORS, SCENE_NAMESPACE, Coord, LudoDestinationOption, LudoSceneAxes


TASK_ID = "task_games__ludo_board__move_result_option_label"
_AXIS_CONFIG = make_ludo_value_option_axis_config(
    value=("move_roll_total_support", "move_roll_total", DEFAULTS.move_roll_total_support, f"{SCENE_NAMESPACE}.move_result.total", "balanced_move_roll_total_sampling"),
    options=("move_result_option_count_support", DEFAULTS.move_result_option_count_support, f"{SCENE_NAMESPACE}.move_result", "balanced_move_result_option_count_sampling"),
)
_MOVE_RESULT_PROMPT_KEYS = (
    "move_result_option_label",
    "move_sequence_rule_text",
    "answer_hint_move_result_option_label",
    "annotation_hint_move_result_option_label",
)
_MOVE_RESULT_EXAMPLE_ANNOTATION = {
    "moving_token": [115, 215],
    "destination_cell": [342, 282],
}


@dataclass(frozen=True)
class _MoveResultSample:
    """Task-owned symbolic sample for applying a shown Ludo roll sequence."""

    token_coords: Mapping[str, Coord]
    query_color: str
    move_roll_total: int
    roll_sequence: tuple[int, ...]
    destination_options: tuple[LudoDestinationOption, ...]
    answer: str
    final_coord: Coord
    construction_mode: str


def _construct_move_result_sample(
    *,
    rng: Any,
    axes: LudoSceneAxes,
    move_roll_total: int,
    answer_label: str,
    option_labels: tuple[str, ...],
) -> _MoveResultSample:
    """Place one mover and board-letter options so the shown roll reaches one label."""

    total = int(move_roll_total)
    roll_sequence = roll_sequence_for_total(total)
    route = route_for_color(str(axes.query_color))
    valid_indices = [index for index in range(0, len(route) - total)]
    rng.shuffle(valid_indices)
    for current_index in valid_indices[:200]:
        start_coord = tuple(route[int(current_index)])
        final_coord = tuple(route[int(current_index) + total])
        token_coords = {str(axes.query_color): start_coord}
        reserved = {start_coord, final_coord}
        other_colors = [color for color in PLAYER_COLORS if color != str(axes.query_color)]
        token_coords.update(sample_other_token_coords(rng=rng, occupied=reserved, colors=other_colors))
        options = make_destination_options(
            rng=rng,
            route=route,
            current_index=int(current_index),
            final_coord=final_coord,
            occupied={tuple(coord) for coord in token_coords.values()},
            answer_label=str(answer_label),
            option_labels=option_labels,
        )
        return _MoveResultSample(
            token_coords=dict(token_coords),
            query_color=str(axes.query_color),
            move_roll_total=total,
            roll_sequence=tuple(int(value) for value in roll_sequence),
            destination_options=tuple(options),
            answer=str(answer_label),
            final_coord=tuple(final_coord),
            construction_mode="target_conditioned_move_result_option",
        )
    raise ValueError("failed to construct Ludo move-result sample")


def _build_move_result_attempt(rng: Any, axes: LudoSceneAxes, selected_query_id: str, movement_axes: Any):
    """Bind one roll sequence to its visible destination option and annotation."""

    roll_total_axis = movement_axes.value_axis
    destination_choice_axis = movement_axes.option_axes
    move_sample = _construct_move_result_sample(
        rng=rng,
        axes=axes,
        move_roll_total=int(roll_total_axis.value),
        answer_label=str(destination_choice_axis.answer_label),
        option_labels=tuple(destination_choice_axis.option_labels),
    )
    return build_ludo_bound_attempt(
        answer_type="option_letter",
        answer_value=str(move_sample.answer),
        selected_query_id=str(selected_query_id),
        axes=axes,
        construction_mode=str(move_sample.construction_mode),
        token_coords=move_sample.token_coords,
        query_color=str(move_sample.query_color),
        target_color=None,
        roll_sequence=move_sample.roll_sequence,
        destination_options=move_sample.destination_options,
        role_sources={
            "moving_token": ("token_centers_px", f"token_{move_sample.query_color}"),
            "destination_cell": ("destination_option_centers_px", str(move_sample.answer)),
        },
        role_entity_ids={
            "moving_token": f"token_{move_sample.query_color}",
            "destination_cell": f"destination_option_{move_sample.answer}",
        },
        extra_execution_trace={
            "move_roll_total": int(move_sample.move_roll_total),
            "roll_sequence": [int(value) for value in move_sample.roll_sequence],
            "destination_options": [{"label": str(option.label), "coord": [int(option.coord[0]), int(option.coord[1])]} for option in move_sample.destination_options],
            "final_coord": [int(move_sample.final_coord[0]), int(move_sample.final_coord[1])],
            "answer": str(move_sample.answer),
        },
        extra_query_params={
            "move_roll_total": int(roll_total_axis.value),
            "move_roll_total_support": [int(value) for value in roll_total_axis.support],
            "move_roll_total_probabilities": dict(roll_total_axis.probabilities),
            **destination_choice_axis.trace_params(),
        },
        relations_extra={"target_color": None},
    )


_PREPARE_OBJECTIVE = make_ludo_value_option_preparer(
    prompt_keys=_MOVE_RESULT_PROMPT_KEYS,
    example_annotation=_MOVE_RESULT_EXAMPLE_ANNOTATION,
    example_answer="D",
    axis_config=_AXIS_CONFIG,
    attempt_namespace=f"{SCENE_NAMESPACE}.move_result",
    build_attempt=_build_move_result_attempt,
)


@register_task
class GamesLudoBoardMoveResultOptionLabelTask(LudoSingleQueryTaskBase):
    """Choose which visible board letter a shown roll sequence reaches."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update')
    _namespace = f"{SCENE_NAMESPACE}.move_result"
    _prepare_objective = staticmethod(_PREPARE_OBJECTIVE)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_ludo_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesLudoBoardMoveResultOptionLabelTask"]
