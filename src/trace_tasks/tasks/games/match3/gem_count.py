"""Count named-color gems in a match-3 grid scope."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.color_format import format_named_color_with_hex, rgb_to_hex
from trace_tasks.tasks.shared.config_defaults import group_default

from ._lifecycle import Match3ObjectivePlan, match3_bbox_set_attempt, run_match3_registered_task
from .shared.defaults import DEFAULTS, GEM_RGB, SCENE_ID
from .shared.prompts import make_match3_prompt_slots
from .shared.rules import cell_entity_id, gem_count_matches
from .shared.sampling import make_base_board, resolve_match3_integer_axis
from .shared.state import Match3Sample, Match3SceneAxes


TASK_ID = "task_games__match3__gem_count"
GRID_COLOR_GEM_COUNT = "grid_color_gem_count"
ROW_COLOR_GEM_COUNT = "row_color_gem_count"
COLUMN_COLOR_GEM_COUNT = "column_color_gem_count"
SUPPORTED_QUERY_IDS = (GRID_COLOR_GEM_COUNT, ROW_COLOR_GEM_COUNT, COLUMN_COLOR_GEM_COUNT)


def _scope_for_branch(selected_branch: str) -> str:
    """Map the public semantic branch onto a counting scope."""

    if str(selected_branch) == ROW_COLOR_GEM_COUNT:
        return "row"
    if str(selected_branch) == COLUMN_COLOR_GEM_COUNT:
        return "column"
    return "grid"


def _prompt_slots_for_branch(selected_branch: str):
    """Return task-owned prompt slots for the selected color-count branch."""

    return make_match3_prompt_slots(
        prompt_query_key=str(selected_branch),
        object_description_key="object_description_match3_count_grid",
        answer_hint_key=f"answer_hint_{str(selected_branch)}",
        annotation_hint_key=f"annotation_hint_{str(selected_branch)}",
        example_annotation=[[224, 310, 280, 366], [290, 310, 346, 366], [356, 310, 412, 366], [422, 310, 478, 366]],
        example_answer=4,
    )


def _prepare_gem_count_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_branch: str,
    _branch_probabilities: Mapping[str, float],
    _axes: Match3SceneAxes,
    gen_defaults: Mapping[str, Any],
) -> Match3ObjectivePlan:
    """Resolve scope and target-count semantics for named-color gem counting."""

    scope = _scope_for_branch(str(selected_branch))
    namespace = f"{SCENE_ID}.gem_count.{scope}"
    target_axis = resolve_match3_integer_axis(
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="gem_count_answer_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.gem_count_answer_support,
        namespace=f"{namespace}.answer_count",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    prompt_slots = _prompt_slots_for_branch(str(selected_branch))

    def _construct_attempt(rng: Any, axes: Match3SceneAxes):
        """Construct a board where the selected scope has the target color count."""

        board_spec = make_base_board(
            rng,
            gen_defaults=gen_defaults,
            namespace=namespace,
            instance_seed=int(instance_seed),
            params=task_params,
            scene_variant=str(axes.scene_variant),
        )
        answer_support = tuple(
            int(value)
            for value in task_params.get(
                "gem_count_answer_support",
                group_default(gen_defaults, "gem_count_answer_support", DEFAULTS.gem_count_answer_support),
            )
        )
        color_name = str(board_spec.gem_keys[int(rng.randrange(len(board_spec.gem_keys)))])
        alternate_colors = [str(key) for key in board_spec.gem_keys if str(key) != str(color_name)]
        if not alternate_colors:
            raise ValueError("gem-count task requires at least two gem colors")

        row_index: int | None = None
        col_index: int | None = None
        if scope == "grid":
            scoped_coords = [(int(row), int(col)) for row in range(int(board_spec.rows)) for col in range(int(board_spec.cols))]
        elif scope == "row":
            row_index = int(rng.randrange(int(board_spec.rows)))
            scoped_coords = [(int(row_index), int(col)) for col in range(int(board_spec.cols))]
        else:
            col_index = int(rng.randrange(int(board_spec.cols)))
            scoped_coords = [(int(row), int(col_index)) for row in range(int(board_spec.rows))]

        target_answer = int(target_axis.value)
        if int(target_answer) > len(scoped_coords):
            scoped_support = [int(value) for value in answer_support if 1 <= int(value) <= len(scoped_coords)]
            if not scoped_support:
                raise ValueError("gem-count support has no feasible value for selected scope")
            target_answer = int(uniform_choice(rng, tuple(scoped_support)))

        chosen_coords = list(scoped_coords)
        rng.shuffle(chosen_coords)
        target_coords = {tuple(coord) for coord in chosen_coords[: int(target_answer)]}
        mutable_board = [list(row) for row in board_spec.board]
        for row, col in scoped_coords:
            if (int(row), int(col)) in target_coords:
                mutable_board[int(row)][int(col)] = str(color_name)
            elif str(mutable_board[int(row)][int(col)]) == str(color_name):
                mutable_board[int(row)][int(col)] = str(alternate_colors[int(rng.randrange(len(alternate_colors)))])
        board = tuple(tuple(str(value) for value in row) for row in mutable_board)
        matches = gem_count_matches(
            board,
            scope=scope,
            color_name=str(color_name),
            row_index=row_index,
            col_index=col_index,
        )
        if len(matches) != int(target_answer):
            raise ValueError("constructed gem-count board did not match target answer")

        color_rgb = tuple(int(value) for value in GEM_RGB[str(color_name)])
        color_label = format_named_color_with_hex(str(color_name), color_rgb)
        sample = Match3Sample(
            scene_variant=str(axes.scene_variant),
            board=board,
            answer=int(len(matches)),
            answer_type="integer",
            option_specs=(),
            annotation_entity_ids=tuple(cell_entity_id(coord) for coord in matches),
            metadata={
                **dict(board_spec.metadata),
                "target_answer": int(target_answer),
                "balanced_target_matched": True,
                "target_answer_probabilities": dict(target_axis.probabilities),
                "target_color_name": str(color_name),
                "target_color_rgb": [int(value) for value in color_rgb],
                "target_color_hex": rgb_to_hex(color_rgb),
                "target_color_label": str(color_label),
                "scope": str(scope),
                "row_index": None if row_index is None else int(row_index + 1),
                "col_index": None if col_index is None else int(col_index + 1),
                "answer_support": [int(value) for value in answer_support],
            },
        )
        return match3_bbox_set_attempt(
            answer_gt=TypedValue(type="integer", value=int(sample.answer)),
            sample=sample,
            prompt_slots=prompt_slots,
            target_color_label=str(color_label),
            row_index=None if row_index is None else int(row_index + 1),
            col_index=None if col_index is None else int(col_index + 1),
            execution_extra={
                "target_color_name": str(color_name),
                "scope": str(scope),
            },
        )

    return Match3ObjectivePlan(
        attempt_namespace=namespace,
        construct_attempt=_construct_attempt,
    )


@register_task
class GamesMatch3GemCountTask:
    """Count canonical named-color gems in the grid or one numbered row/column."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    _namespace = f"{SCENE_ID}.gem_count"
    _default_branch = GRID_COLOR_GEM_COUNT
    _prepare_objective = staticmethod(_prepare_gem_count_objective)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_match3_registered_task(self, instance_seed, params=params, max_attempts=max_attempts)


__all__ = ["GamesMatch3GemCountTask", "TASK_ID"]
