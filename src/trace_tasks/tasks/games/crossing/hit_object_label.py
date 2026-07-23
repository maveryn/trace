"""Identify the labeled moving object that hits the marked crossing route."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.crossing._lifecycle import (
    CrossingLabelObjectiveSpec,
    CrossingObjectivePlan,
    prepare_label_objective_from_spec,
    run_crossing_lifecycle,
)
from trace_tasks.tasks.games.crossing.shared.defaults import SCENE_ID
from trace_tasks.tasks.games.crossing.shared.sampling import sample_labeled_route_collision_scene
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults


TASK_ID = "task_games__crossing__hit_object_label"
QUERY_ID = "hit_object_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
TARGET_LABEL_INDEX_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _construct_unique_hit_scene(
    rng: Any,
    axes,
    target_label: str,
    _task_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
):
    """Construct the one-collision scene required by this objective."""

    return sample_labeled_route_collision_scene(
        rng=rng,
        axes=axes,
        target_label=str(target_label),
        max_extra_per_row=int(group_default(gen_defaults, "hit_object_max_extra_per_row", 1)),
    )


def _prepare_hit_object_label_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _query_probabilities: Mapping[str, float],
) -> CrossingObjectivePlan:
    """Bind unique route-hit semantics to the generic label-output lifecycle."""

    return prepare_label_objective_from_spec(
        task_id=TASK_ID,
        spec=CrossingLabelObjectiveSpec(
            prompt_query_key=QUERY_ID,
            count_mode="hit_object_label",
            label_support_key="hit_object_label_index_support",
            fallback_label_index_support=TARGET_LABEL_INDEX_SUPPORT,
            construct_attempt=_construct_unique_hit_scene,
        ),
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query_id=str(selected_query_id),
        gen_defaults=_GEN_DEFAULTS,
    )


@register_task
class GamesCrossingHitObjectLabelTask:
    """Identify which labeled moving object collides with the marked route."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a straight marked-route collision label question."""

        return run_crossing_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_hit_object_label_objective,
        )


__all__ = ["GamesCrossingHitObjectLabelTask"]
