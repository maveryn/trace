from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import ObjectiveRhythmPlan, run_rhythm_lifecycle
from .shared.defaults import DEFAULTS
from .shared.sampling import resolve_rhythm_count_target_axis, resolve_selected_lane, sample_lane_note_count_scene
from .shared.state import SCENE_ID, SCENE_NAMESPACE


TASK_ID = "task_games__rhythm__lane_note_count"
PROMPT_QUERY_KEY = "lane_note_count"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_lane_note_count_objective(instance_seed, params, _query_probabilities, query_id) -> ObjectiveRhythmPlan:
    if str(query_id) != DEFAULT_QUERY_ID:
        raise ValueError(f"unsupported Rhythm lane-note-count query: {query_id}")
    target_axis = resolve_rhythm_count_target_axis(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=f"{SCENE_NAMESPACE}.lane_note_count.target_count",
        support_key="note_count_support",
        explicit_key="target_note_count",
        fallback_support=DEFAULTS.note_count_support,
        balanced_flag_key="balanced_note_count_sampling",
    )
    return ObjectiveRhythmPlan(
        attempt_namespace=f"{SCENE_NAMESPACE}.lane_note_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        annotation_kind="note_bbox_set",
        query_params={
            "target_note_count": int(target_axis.target_count),
            "target_note_count_support": [int(value) for value in target_axis.target_count_support],
            "target_note_count_probabilities": dict(target_axis.target_count_probabilities),
        },
        construct_attempt=lambda rng, axes: sample_lane_note_count_scene(
            rng=rng,
            axes=axes,
            selected_lane=resolve_selected_lane(rng, lane_count=int(axes.lane_count), params=params),
            target_count=int(target_axis.target_count),
        ),
    )


@register_task
class GamesRhythmLaneNoteCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)

    def generate(self, instance_seed: int, *, params: dict | None = None, max_attempts: int = 100):
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
            prepare_objective=_prepare_lane_note_count_objective,
        )


__all__ = ["GamesRhythmLaneNoteCountTask", "TASK_ID"]
