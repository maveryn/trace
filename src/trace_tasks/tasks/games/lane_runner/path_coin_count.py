"""Count coins collected by a displayed lane-runner path."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import LaneRunnerAttemptResult, LaneRunnerObjectivePlan, run_lane_runner_lifecycle
from .shared.annotations import lane_runner_coin_point_annotation
from .shared.rules import (
    coin_entity_id,
    path_coin_collection,
    validate_lane_runner_path_coin_sample,
    visible_coin_trace,
)
from .shared.sampling import resolve_lane_runner_integer_axis
from .shared.state import (
    SCENE_ID,
    SCENE_NAMESPACE,
    LaneRunnerCoin,
    LaneRunnerPathCoinSample,
    LaneRunnerSceneAxes,
)


TASK_ID = "task_games__lane_runner__path_coin_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "path_coin_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
TARGET_ANSWER_SUPPORT: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _construct_path_coin_sample(
    *,
    rng: Any,
    axes: LaneRunnerSceneAxes,
    target_answer: int,
) -> LaneRunnerPathCoinSample:
    """Construct one path where the collected coin count is fixed."""

    rows = int(axes.row_count)
    lanes = int(axes.lane_count)
    if lanes != 2:
        raise ValueError("lane-runner path-coin task requires two lanes")
    if int(target_answer) > rows:
        raise ValueError("target_answer must be no greater than row_count")

    path: list[int] = []
    previous = int(axes.start_lane)
    for _row in range(rows):
        lane = 1 - int(previous) if rng.random() < 0.42 else int(previous)
        path.append(int(lane))
        previous = int(lane)

    collected_rows = sorted(rng.sample(range(rows), int(target_answer)))
    coin_cells: set[Tuple[int, int]] = {(int(row), int(path[row])) for row in collected_rows}

    parallel_row = int(rng.choice(collected_rows))
    coin_cells.add((parallel_row, 1 - int(path[parallel_row])))

    off_path_cells = [(row, 1 - int(path[row])) for row in range(rows)]
    max_extra = max(1, min(rows - int(target_answer), 3))
    extra_count = int(rng.randint(1, max_extra))
    rng.shuffle(off_path_cells)
    for row, lane in off_path_cells:
        if len(coin_cells) >= min(rows * lanes, int(target_answer) + extra_count):
            break
        coin_cells.add((int(row), int(lane)))
    if len(coin_cells) <= int(target_answer):
        raise ValueError("path-coin task failed to add off-path coin distractors")

    coins = tuple(
        LaneRunnerCoin(coin_id=coin_entity_id(row, lane), row=int(row), lane=int(lane))
        for row, lane in sorted(coin_cells)
    )
    answer, annotation_ids = path_coin_collection(
        coins=coins,
        shown_path_lanes=tuple(path),
        row_count=rows,
        lane_count=lanes,
        start_lane=int(axes.start_lane),
    )
    if int(answer) != int(target_answer):
        raise ValueError("path-coin sampled answer does not match target")
    sample = LaneRunnerPathCoinSample(
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        row_count=rows,
        lane_count=lanes,
        start_lane=int(axes.start_lane),
        coins=coins,
        shown_path_lanes=tuple(int(value) for value in path),
        answer=int(answer),
        annotation_entity_ids=tuple(str(value) for value in annotation_ids),
        construction_mode="sample_shown_path_with_parallel_coin_distractors",
    )
    validate_lane_runner_path_coin_sample(sample)
    return sample


def _prepare_path_coin_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    axes: LaneRunnerSceneAxes,
) -> LaneRunnerObjectivePlan:
    """Resolve the target coin count and bind path-coin construction."""

    target_axis = resolve_lane_runner_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="target_answer_support",
        explicit_key="target_answer",
        fallback_support=TARGET_ANSWER_SUPPORT,
        namespace=f"{SCENE_NAMESPACE}.path_coin.target_answer.row_count_{int(axes.row_count)}",
        balanced_flag_key="balanced_target_answer_sampling",
        upper_bound=int(axes.row_count),
    )

    def construct_attempt(rng: Any, resolved_axes: LaneRunnerSceneAxes) -> LaneRunnerAttemptResult:
        sample = _construct_path_coin_sample(
            rng=rng,
            axes=resolved_axes,
            target_answer=int(target_axis.value),
        )
        return LaneRunnerAttemptResult(
            answer_gt=TypedValue(type="integer", value=int(sample.answer)),
            render_inputs={
                "coins": sample.coins,
                "shown_path_lanes": sample.shown_path_lanes,
                "start_lane": int(sample.start_lane),
            },
            build_annotation=lambda rendered: lane_runner_coin_point_annotation(
                rendered,
                sample.annotation_entity_ids,
            ),
            annotation_entity_ids=tuple(str(entity_id) for entity_id in sample.annotation_entity_ids),
            scene_kind="games_lane_runner_two_lane_track",
            execution_trace={
                "coins": list(visible_coin_trace(sample.coins)),
                "shown_path_lanes": [int(value) for value in sample.shown_path_lanes],
                "answer": int(sample.answer),
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
                "construction_mode": str(sample.construction_mode),
                "target_answer": int(target_axis.value),
            },
            relations_extra={
                "shown_path_lanes": [int(value) for value in sample.shown_path_lanes],
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            },
        )

    return LaneRunnerObjectivePlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        object_description_suffix="path_coin",
        rule_slot_name="lane_runner_path_rule_text",
        attempt_namespace=f"{SCENE_NAMESPACE}.path_coin",
        construct_attempt=construct_attempt,
        extra_query_params={
            "target_answer": int(target_axis.value),
            "target_answer_support": [int(value) for value in target_axis.support],
            "target_answer_probabilities": dict(target_axis.probabilities),
        },
    )


@register_task
class GamesLaneRunnerPathCoinCountTask:
    """Count coins collected by a displayed two-lane runner path."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'spatial_relations', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_lane_runner_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_path_coin_objective,
            namespace=f"{SCENE_NAMESPACE}.path_coin",
        )


__all__ = ["GamesLaneRunnerPathCoinCountTask"]
