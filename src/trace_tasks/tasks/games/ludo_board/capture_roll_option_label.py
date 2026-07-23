"""Choose the roll option that lets a Ludo token capture a target token."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import LudoSingleQueryTaskBase, build_ludo_bound_attempt, make_ludo_value_option_preparer, run_ludo_registered_task
from .shared.rules import path_index
from .shared.sampling import make_ludo_value_option_axis_config, make_roll_options, sample_other_token_coords
from .shared.state import DEFAULTS, MAIN_PATH, PLAYER_COLORS, SCENE_NAMESPACE, START_COORDS, Coord, LudoRollOption, LudoSceneAxes


TASK_ID = "task_games__ludo_board__capture_roll_option_label"
_AXIS_CONFIG = make_ludo_value_option_axis_config(
    value=("capture_distance_support", "capture_distance", DEFAULTS.capture_distance_support, f"{SCENE_NAMESPACE}.capture_roll.distance", "balanced_capture_distance_sampling"),
    options=("capture_option_count_support", DEFAULTS.capture_option_count_support, f"{SCENE_NAMESPACE}.capture_roll", "balanced_capture_option_count_sampling"),
)


@dataclass(frozen=True)
class _CaptureRollSample:
    """Task-owned symbolic sample for choosing the capture roll option."""

    token_coords: Mapping[str, Coord]
    query_color: str
    target_color: str
    capture_distance: int
    options: tuple[LudoRollOption, ...]
    answer: str
    construction_mode: str


def _construct_capture_roll_sample(
    *,
    rng: Any,
    axes: LudoSceneAxes,
    capture_distance: int,
    answer_label: str,
    option_labels: tuple[str, ...],
) -> _CaptureRollSample:
    """Place mover and target so exactly one displayed roll option captures the target."""

    distance = int(capture_distance)
    if distance < 1 or distance > 11:
        raise ValueError("capture distance must be in 1..11")
    start_indices = {path_index(coord) for coord in START_COORDS.values()}
    for _attempt in range(200):
        mover_index = int(rng.randrange(len(MAIN_PATH)))
        target_index = (mover_index + distance) % len(MAIN_PATH)
        if target_index in start_indices:
            continue
        mover_coord = tuple(MAIN_PATH[mover_index])
        target_coord = tuple(MAIN_PATH[target_index])
        if mover_coord == target_coord:
            continue
        token_coords = {
            str(axes.query_color): mover_coord,
            str(axes.target_color): target_coord,
        }
        occupied = {mover_coord, target_coord}
        other_colors = [color for color in PLAYER_COLORS if color not in {str(axes.query_color), str(axes.target_color)}]
        token_coords.update(sample_other_token_coords(rng=rng, occupied=occupied, colors=other_colors))
        options = make_roll_options(
            rng=rng,
            correct_distance=distance,
            answer_label=str(answer_label),
            option_labels=option_labels,
        )
        return _CaptureRollSample(
            token_coords=dict(token_coords),
            query_color=str(axes.query_color),
            target_color=str(axes.target_color),
            capture_distance=distance,
            options=tuple(options),
            answer=str(answer_label),
            construction_mode="target_conditioned_capture_roll_option",
        )
    raise ValueError("failed to construct Ludo capture sample")


def _build_capture_roll_attempt(rng: Any, axes: LudoSceneAxes, selected_query_id: str, axis_bundle: Any):
    """Bind the one roll option that lands the mover on the target token."""

    capture_distance_axis = axis_bundle.value_axis
    roll_choice_axis = axis_bundle.option_axes
    capture_sample = _construct_capture_roll_sample(
        rng=rng,
        axes=axes,
        capture_distance=int(capture_distance_axis.value),
        answer_label=str(roll_choice_axis.answer_label),
        option_labels=tuple(roll_choice_axis.option_labels),
    )
    return build_ludo_bound_attempt(
        answer_type="option_letter",
        answer_value=str(capture_sample.answer),
        selected_query_id=str(selected_query_id),
        axes=axes,
        construction_mode=str(capture_sample.construction_mode),
        token_coords=capture_sample.token_coords,
        query_color=str(capture_sample.query_color),
        target_color=str(capture_sample.target_color),
        roll_options=capture_sample.options,
        role_sources={
            "mover_token": ("token_centers_px", f"token_{capture_sample.query_color}"),
            "target_token": ("token_centers_px", f"token_{capture_sample.target_color}"),
        },
        role_entity_ids={
            "mover_token": f"token_{capture_sample.query_color}",
            "target_token": f"token_{capture_sample.target_color}",
        },
        extra_execution_trace={
            "capture_distance": int(capture_sample.capture_distance),
            "options": [{"label": str(option.label), "distance": int(option.distance), "text": str(option.text)} for option in capture_sample.options],
            "answer": str(capture_sample.answer),
        },
        extra_query_params={
            "capture_distance": int(capture_distance_axis.value),
            "capture_distance_support": [int(value) for value in capture_distance_axis.support],
            "capture_distance_probabilities": dict(capture_distance_axis.probabilities),
            **roll_choice_axis.trace_params(),
        },
        relations_extra={"target_color": str(capture_sample.target_color)},
    )


_PREPARE_OBJECTIVE = make_ludo_value_option_preparer(
    prompt_keys=("capture_roll_option_label", "capture_option_rule_text", "answer_hint_capture_roll_option_label", "annotation_hint_capture_roll_option_label"),
    example_annotation={"mover_token": [115, 215], "target_token": [275, 215]},
    example_answer="C",
    axis_config=_AXIS_CONFIG,
    attempt_namespace=f"{SCENE_NAMESPACE}.capture_roll",
    build_attempt=_build_capture_roll_attempt,
)


@register_task
class GamesLudoBoardCaptureRollOptionLabelTask(LudoSingleQueryTaskBase):
    """Choose which visible roll option lands the mover on the target token."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update')
    _namespace = f"{SCENE_NAMESPACE}.capture_roll"
    _prepare_objective = staticmethod(_PREPARE_OBJECTIVE)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_ludo_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesLudoBoardCaptureRollOptionLabelTask"]
