"""Select the Snake result option after a visible move sequence."""

from __future__ import annotations

from typing import Any, Mapping

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SnakeLifecycleTask, SnakeObjective, run_snake_task, snake_prompt_json_examples
from .shared.option_rendering import draw_path_result_options
from .shared.rules import coord_to_cell_id, validate_snake_state
from .shared.sampling import (
    find_sequence_for_path_result,
    planned_annotation_ids,
    point_option_coords,
    random_snake_state,
    sample_obstacles,
    select_planned_outcome_target,
    with_obstacles,
)
from .shared.state import Coord, SnakeSample, SnakeSceneAxes


TASK_ID = "task_games__snake__path_outcome_option_label"
PROMPT_KEY = "path_result_option_label"
OPTION_LABELS = ("A", "B", "C", "D")


def _build_path_result_options(
    *,
    rng: Any,
    state: Any,
    target_result: str,
    final_head: Coord,
) -> tuple[str, tuple[Mapping[str, object], ...]]:
    """Build four image-visible result options with one game-over card."""

    answer_label = str(rng.choice(OPTION_LABELS))
    game_over_label = answer_label if str(target_result) == "game_over" else str(rng.choice([label for label in OPTION_LABELS if label != answer_label]))
    options_by_label: dict[str, dict[str, object]] = {
        str(game_over_label): {
            "label": str(game_over_label),
            "kind": "game_over",
            "text": "GAME OVER",
            "is_answer": str(target_result) == "game_over",
        }
    }

    point_labels = [label for label in OPTION_LABELS if label != game_over_label]
    point_coords: dict[str, Coord] = {}
    excluded: set[Coord] = set()
    if str(target_result) == "point":
        point_coords[answer_label] = (int(final_head[0]), int(final_head[1]))
        excluded.add((int(final_head[0]), int(final_head[1])))
    distractor_labels = [label for label in point_labels if label not in point_coords]
    for label, coord in zip(
        distractor_labels,
        point_option_coords(rng=rng, state=state, count=len(distractor_labels), exclude=excluded),
    ):
        point_coords[str(label)] = (int(coord[0]), int(coord[1]))

    for label in point_labels:
        coord = point_coords[str(label)]
        options_by_label[str(label)] = {
            "label": str(label),
            "kind": "point",
            "coord": [int(coord[0]), int(coord[1])],
            "cell_id": coord_to_cell_id(coord),
            "is_answer": str(target_result) == "point" and str(label) == answer_label,
        }
    return answer_label, tuple(options_by_label[str(label)] for label in OPTION_LABELS)


def _decorate_path_options(
    image: Image.Image,
    render_map: Mapping[str, Any],
    sample: SnakeSample,
    font_family: str,
) -> tuple[Image.Image, dict[str, list[float]]]:
    """Draw the task-owned option markers after the base board is rendered."""

    return draw_path_result_options(
        image=image,
        render_map=render_map,
        sample=sample,
        font_family=str(font_family),
    )


def _prepare_path_outcome_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    axes: SnakeSceneAxes,
) -> SnakeObjective:
    """Bind one option label after simulating a listed move sequence."""

    target_outcome, target_probabilities = select_planned_outcome_target(
        task_params,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    rng = spawn_rng(int(attempt_seed), f"{TASK_ID}.sample")
    for _attempt in range(900):
        state = random_snake_state(rng=rng, board_size=int(axes.board_size), body_length=int(axes.body_length))
        try:
            obstacles = sample_obstacles(rng=rng, state=state, count=int(axes.obstacle_count))
        except ValueError:
            continue
        state = with_obstacles(state, obstacles)
        validate_snake_state(state)
        result = find_sequence_for_path_result(
            rng=rng,
            state=state,
            length=int(axes.planned_move_count),
            target_result=str(target_outcome),
        )
        if result is None:
            continue
        final_state, sequence, simulation = result
        try:
            answer_label, options = _build_path_result_options(
                rng=rng,
                state=final_state,
                target_result=str(target_outcome),
                final_head=simulation.final_head,
            )
        except ValueError:
            continue
        sample = SnakeSample(
            answer=str(answer_label),
            state=final_state,
            annotation_cell_ids=planned_annotation_ids(simulation, fallback_state=final_state),
            construction_mode=f"path_result_option_{str(target_outcome)}",
            planned_moves=tuple(sequence),
            result_options=tuple(options),
            target_outcome=str(target_outcome),
            observed_event_step=int(simulation.event_step),
        )
        json_example, json_example_answer_only = snake_prompt_json_examples(
            answer="B",
            annotation_box_count=3,
        )
        return SnakeObjective(
            sample=sample,
            answer_gt=TypedValue(type="option_letter", value=str(answer_label)),
            prompt_key=PROMPT_KEY,
            prompt_json_example=str(json_example),
            prompt_json_example_answer_only=str(json_example_answer_only),
            answer_support=[str(value) for value in OPTION_LABELS],
            annotation_cell_ids=tuple(sample.annotation_cell_ids),
            trace_extra_params={
                "target_planned_outcome": str(target_outcome),
                "target_planned_outcome_probabilities": dict(target_probabilities),
                "option_labels": [str(value) for value in OPTION_LABELS],
            },
            execution_extra={"target_planned_outcome": str(target_outcome)},
            decorate_image=_decorate_path_options,
        )
    raise ValueError("failed to construct Snake path-result option sample")


@register_task
class GamesSnakePathOutcomeTask(SnakeLifecycleTask):
    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update', 'matching')

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_snake_task(self, instance_seed, params, max_attempts, _prepare_path_outcome_objective)
