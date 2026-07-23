from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import ObjectiveRhythmPlan, run_rhythm_lifecycle
from .shared.sampling import resolve_selected_lane, sample_earliest_hit_lane_scene
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_games__rhythm__earliest_hit_lane_label"
PROMPT_QUERY_KEY = "earliest_hit_lane_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_earliest_hit_objective(
    _instance_seed: int,
    params: Mapping[str, Any],
    _query_probabilities: Mapping[str, float],
    query_id: str,
) -> ObjectiveRhythmPlan:
    if str(query_id) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported Rhythm earliest-hit query: {query_id}")
    return ObjectiveRhythmPlan(
        attempt_namespace=f"{SCENE_NAMESPACE}.earliest_hit",
        prompt_query_key=PROMPT_QUERY_KEY,
        annotation_kind="note_bbox",
        prompt_rule_keys=("rhythm_motion_rule_text",),
        query_params={},
        construct_attempt=lambda rng, axes: sample_earliest_hit_lane_scene(
            rng=rng,
            axes=axes,
            target_lane=resolve_selected_lane(rng, lane_count=int(axes.lane_count), params=params, key="target_lane_index"),
        ),
    )


@register_task
class GamesRhythmEarliestHitLaneLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'comparison', 'ranking')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)

    def generate(self, instance_seed: int, *, params: dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_rhythm_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_earliest_hit_objective,
        )


__all__ = ["GamesRhythmEarliestHitLaneLabelTask", "TASK_ID"]
