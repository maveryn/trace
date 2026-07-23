from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import ObjectiveRhythmPlan, run_rhythm_lifecycle
from .shared.defaults import DEFAULTS
from .shared.sampling import resolve_rhythm_count_target_axis, resolve_selected_lane, sample_lane_note_score_scene
from .shared.state import SCENE_ID, SCENE_NAMESPACE, SUPPORTED_COLOR_KEYS


TASK_ID = "task_games__rhythm__lane_note_score_value"
PROMPT_QUERY_KEY = "lane_note_score_value"
_SCORE_VALUES = (1, 2, 3)
_SCORE_COLOR_KEYS = SUPPORTED_COLOR_KEYS[:3]
_MAX_TARGET_LANE_NOTES = 4

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _score_values_for_instance(instance_seed: int) -> dict[str, int]:
    """Return a deterministic color-score palette for one public instance."""

    values = [int(value) for value in _SCORE_VALUES]
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.lane_note_score.score_palette")
    rng.shuffle(values)
    return {str(color): int(value) for color, value in zip(_SCORE_COLOR_KEYS, values)}


def _prepare_lane_note_score_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    _query_probabilities: Mapping[str, float],
    query_id: str,
) -> ObjectiveRhythmPlan:
    """Bind the lane score objective, including its visible color-score palette."""

    if str(query_id) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported Rhythm lane-note-score query: {query_id}")
    target_axis = resolve_rhythm_count_target_axis(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{SCENE_NAMESPACE}.lane_note_score.target_score",
        support_key="score_total_support",
        explicit_key="target_score",
        fallback_support=DEFAULTS.score_total_support,
        balanced_flag_key="balanced_score_total_sampling",
    )
    score_values_by_color = _score_values_for_instance(int(instance_seed))
    return ObjectiveRhythmPlan(
        attempt_namespace=f"{SCENE_NAMESPACE}.lane_note_score",
        prompt_query_key=PROMPT_QUERY_KEY,
        annotation_kind="note_bbox_set",
        prompt_rule_keys=("score_palette_rule_text",),
        json_example_annotation=((260, 498, 341, 545), (260, 432, 341, 479)),
        json_example_answer=5,
        query_params={
            "target_score": int(target_axis.target_count),
            "target_score_support": [int(value) for value in target_axis.target_count_support],
            "target_score_probabilities": dict(target_axis.target_count_probabilities),
            "score_values_by_color": dict(score_values_by_color),
        },
        construct_attempt=lambda rng, axes: sample_lane_note_score_scene(
            rng=rng,
            axes=axes,
            selected_lane=resolve_selected_lane(rng, lane_count=int(axes.lane_count), params=params),
            target_score=int(target_axis.target_count),
            score_values_by_color=score_values_by_color,
            max_target_notes=_MAX_TARGET_LANE_NOTES,
        ),
    )


@register_task
class GamesRhythmLaneNoteScoreValueTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'aggregation', 'formula_evaluation')
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
            prepare_objective=_prepare_lane_note_score_objective,
        )


__all__ = ["GamesRhythmLaneNoteScoreValueTask", "TASK_ID"]
