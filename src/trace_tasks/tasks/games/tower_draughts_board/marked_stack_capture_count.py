"""Count immediate captures for a marked tower draughts stack."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import TowerDraughtsObjectivePlan, run_tower_draughts_lifecycle
from .shared.defaults import DEFAULTS
from .shared.prompts import format_json_examples
from .shared.sampling import max_capture_count_for_board, sample_marked_stack_capture_count_scene


TASK_ID = "task_games__tower_draughts_board__marked_stack_capture_count"
PROMPT_QUERY_KEY = "marked_stack_capture_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(
    annotation=[[130, 190, 184, 244], [242, 300, 296, 354]],
    answer=2,
)


def _prepare_capture_count_objective(
    _instance_seed: int,
    _params: Mapping[str, Any],
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
) -> TowerDraughtsObjectivePlan:
    """Bind the immediate capture count objective."""

    if str(selected_branch) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported tower draughts capture branch: {selected_branch}")

    def construct_attempt(rng, axes):
        return sample_marked_stack_capture_count_scene(rng=rng, axes=axes)

    return TowerDraughtsObjectivePlan(
        attempt_namespace="games.tower_draughts_board.marked_stack_capture_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint_key=f"answer_hint_{PROMPT_QUERY_KEY}",
        annotation_hint_key=f"annotation_hint_{PROMPT_QUERY_KEY}",
        annotation_kind="marked_or_stack_bbox_set",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        target_answer_support_key="marked_capture_count_support",
        target_answer_fallback=DEFAULTS.marked_capture_count_support,
        max_count_for_board=max_capture_count_for_board,
        construct_attempt=construct_attempt,
        force_crowned_for_large_target=True,
        trace_params={"capture_count_branch_probabilities": dict(branch_probabilities)},
    )


@register_task
class GamesTowerDraughtsBoardMarkedStackCaptureCountTask:
    """Count opponent-controlled stacks the X-marked stack can capture now."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_tower_draughts_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_capture_count_objective,
        )


__all__ = ["GamesTowerDraughtsBoardMarkedStackCaptureCountTask", "TASK_ID"]
