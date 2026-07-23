"""Select the lane-runner path card that avoids every hazard."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import LaneRunnerAttemptResult, LaneRunnerObjectivePlan, run_lane_runner_lifecycle
from .shared.annotations import lane_runner_path_card_bbox_annotation
from .shared.rules import (
    cell_entity_id,
    hazard_entity_id,
    path_hits_hazard,
    path_option_entity_id,
    validate_lane_runner_safe_path_sample,
    visible_hazard_trace,
    visible_path_option_trace,
)
from .shared.sampling import resolve_lane_runner_integer_axis
from .shared.state import (
    OPTION_LABELS,
    SCENE_ID,
    SCENE_NAMESPACE,
    LaneRunnerHazard,
    LaneRunnerPathOption,
    LaneRunnerSafePathSample,
    LaneRunnerSceneAxes,
)


TASK_ID = "task_games__lane_runner__safe_path_label"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "safe_path_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
OPTION_COUNT_SUPPORT: Tuple[int, ...] = (4, 6)
ANSWER_OPTION_INDEX_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _candidate_distractor_paths(
    *,
    safe_lanes: Tuple[int, ...],
    hazard_rows: Sequence[int],
) -> Tuple[Tuple[int, ...], ...]:
    """Return paths that differ from the safe route and enter hazards."""

    rows = len(safe_lanes)
    hazard_set = {int(row) for row in hazard_rows}
    candidates: list[Tuple[int, ...]] = []
    for diff_count in range(2, rows + 1):
        for diff_rows in combinations(range(rows), diff_count):
            if not any(int(row) in hazard_set for row in diff_rows):
                continue
            path = list(safe_lanes)
            for row in diff_rows:
                path[int(row)] = 1 - int(path[int(row)])
            candidate = tuple(int(value) for value in path)
            if candidate != safe_lanes and candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def _construct_safe_path_sample(
    *,
    rng: Any,
    axes: LaneRunnerSceneAxes,
    option_count: int,
    answer_option_index: int,
) -> LaneRunnerSafePathSample:
    """Construct option cards with exactly one hazard-free path."""

    labels = OPTION_LABELS[: int(option_count)]
    answer_label = labels[int(answer_option_index)]
    safe_lanes = tuple(int(rng.randrange(int(axes.lane_count))) for _ in range(int(axes.row_count)))
    hazard_count = int(rng.randint(2, min(4, int(axes.row_count))))
    hazard_rows = tuple(sorted(rng.sample(range(int(axes.row_count)), hazard_count)))
    hazards = tuple(
        LaneRunnerHazard(
            hazard_id=hazard_entity_id(row, 1 - int(safe_lanes[int(row)])),
            row=int(row),
            lane=1 - int(safe_lanes[int(row)]),
        )
        for row in hazard_rows
    )

    distractor_pool = list(_candidate_distractor_paths(safe_lanes=safe_lanes, hazard_rows=hazard_rows))
    rng.shuffle(distractor_pool)
    needed_distractors = int(option_count) - 1
    if len(distractor_pool) < needed_distractors:
        raise ValueError("not enough unique safe-path distractors")

    options: list[LaneRunnerPathOption] = []
    distractor_index = 0
    for label in labels:
        if str(label) == str(answer_label):
            lanes = safe_lanes
        else:
            lanes = distractor_pool[distractor_index]
            distractor_index += 1
        options.append(LaneRunnerPathOption(label=str(label), lanes_by_row=tuple(int(value) for value in lanes)))

    for option in options:
        if str(option.label) != str(answer_label) and not path_hits_hazard(lanes_by_row=option.lanes_by_row, hazards=hazards):
            raise ValueError("lane-runner distractor path did not hit a hazard")

    sample = LaneRunnerSafePathSample(
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        row_count=int(axes.row_count),
        lane_count=int(axes.lane_count),
        start_lane=int(axes.start_lane),
        hazards=tuple(hazards),
        path_options=tuple(options),
        answer_label=str(answer_label),
        safe_path_cell_ids=tuple(cell_entity_id(row, lane) for row, lane in enumerate(safe_lanes)),
        construction_mode="safe_path_with_off_route_hazards_and_invalid_distractors",
    )
    validate_lane_runner_safe_path_sample(sample)
    return sample


def _prepare_safe_path_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_query_id: str,
    _branch_probabilities: Mapping[str, float],
    _axes: LaneRunnerSceneAxes,
) -> LaneRunnerObjectivePlan:
    """Resolve option axes and bind safe-path card construction."""

    option_axis = resolve_lane_runner_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="option_count_support",
        explicit_key="option_count",
        fallback_support=OPTION_COUNT_SUPPORT,
        namespace=f"{SCENE_NAMESPACE}.safe_path.option_count",
        balanced_flag_key="balanced_option_count_sampling",
    )
    answer_index_axis = resolve_lane_runner_integer_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="answer_option_index_support",
        explicit_key="answer_option_index",
        fallback_support=ANSWER_OPTION_INDEX_SUPPORT,
        namespace=f"{SCENE_NAMESPACE}.safe_path.answer_option_index.option_count_{int(option_axis.value)}",
        balanced_flag_key="balanced_answer_option_index_sampling",
        upper_bound=int(option_axis.value) - 1,
    )

    def construct_attempt(rng: Any, resolved_axes: LaneRunnerSceneAxes) -> LaneRunnerAttemptResult:
        sample = _construct_safe_path_sample(
            rng=rng,
            axes=resolved_axes,
            option_count=int(option_axis.value),
            answer_option_index=int(answer_index_axis.value),
        )
        answer_entity_id = path_option_entity_id(str(sample.answer_label))
        return LaneRunnerAttemptResult(
            answer_gt=TypedValue(type="option_letter", value=str(sample.answer_label)),
            render_inputs={
                "coins": (),
                "hazards": sample.hazards,
                "path_options": sample.path_options,
                "start_lane": int(sample.start_lane),
                "show_board": False,
            },
            build_annotation=lambda rendered: lane_runner_path_card_bbox_annotation(
                rendered,
                sample.answer_label,
            ),
            annotation_entity_ids=(str(answer_entity_id),),
            scene_kind="games_lane_runner_safe_path_option_cards",
            execution_trace={
                "hazards": list(visible_hazard_trace(sample.hazards)),
                "path_options": list(visible_path_option_trace(sample.path_options)),
                "answer": str(sample.answer_label),
                "answer_entity_id": str(answer_entity_id),
                "annotation_entity_ids": [str(answer_entity_id)],
                "safe_path_cell_ids": [str(entity_id) for entity_id in sample.safe_path_cell_ids],
                "construction_mode": str(sample.construction_mode),
            },
            relations_extra={
                "answer_label": str(sample.answer_label),
                "annotation_entity_ids": [str(answer_entity_id)],
                "safe_path_cell_ids": [str(entity_id) for entity_id in sample.safe_path_cell_ids],
            },
        )

    return LaneRunnerObjectivePlan(
        prompt_query_key=PROMPT_QUERY_KEY,
        object_description_suffix="safe_path",
        rule_slot_name="safe_path_rule_text",
        attempt_namespace=f"{SCENE_NAMESPACE}.safe_path",
        construct_attempt=construct_attempt,
        option_count=int(option_axis.value),
        extra_query_params={
            "option_count": int(option_axis.value),
            "answer_option_index": int(answer_index_axis.value),
            "option_count_support": [int(value) for value in option_axis.support],
            "answer_option_index_support": [int(value) for value in answer_index_axis.support],
            "option_count_probabilities": dict(option_axis.probabilities),
            "answer_option_index_probabilities": dict(answer_index_axis.probabilities),
        },
    )


@register_task
class GamesLaneRunnerSafePathLabelTask:
    """Select the only displayed lane-runner route that avoids hazards."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
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
            prepare_objective=_prepare_safe_path_objective,
            namespace=f"{SCENE_NAMESPACE}.safe_path",
        )


__all__ = ["GamesLaneRunnerSafePathLabelTask"]
