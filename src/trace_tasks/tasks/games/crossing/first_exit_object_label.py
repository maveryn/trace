"""Identify the labeled moving object that leaves the crossing board first."""

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
from trace_tasks.tasks.games.crossing.shared.rules import vehicle_exit_tick
from trace_tasks.tasks.games.crossing.shared.sampling import sample_labeled_first_exit_scene
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


TASK_ID = "task_games__crossing__first_exit_object_label"
QUERY_ID = "first_exit_object_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
TARGET_LABEL_INDEX_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _construct_first_exit_scene(
    rng: Any,
    axes,
    target_label: str,
    _task_params: Mapping[str, Any],
    _gen_defaults: Mapping[str, Any],
):
    """Construct the no-route scene with the target label as earliest exit."""

    return sample_labeled_first_exit_scene(
        rng=rng,
        axes=axes,
        target_label=str(target_label),
    )


def _prepare_first_exit_object_label_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _query_probabilities: Mapping[str, float],
) -> CrossingObjectivePlan:
    """Bind earliest board-exit semantics to the generic label-output lifecycle."""

    plan = prepare_label_objective_from_spec(
        task_id=TASK_ID,
        spec=CrossingLabelObjectiveSpec(
            prompt_query_key=QUERY_ID,
            count_mode="first_exit_object_label",
            label_support_key="first_exit_label_index_support",
            fallback_label_index_support=TARGET_LABEL_INDEX_SUPPORT,
            construct_attempt=_construct_first_exit_scene,
            include_route_in_description=False,
            use_exit_motion_rule_text=True,
        ),
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query_id=str(selected_query_id),
        gen_defaults=_GEN_DEFAULTS,
    )

    return CrossingObjectivePlan(
        axes=plan.axes,
        attempt_namespace=plan.attempt_namespace,
        construct_attempt=plan.construct_attempt,
        prompt_query_key=plan.prompt_query_key,
        prompt_dynamic_slots=plan.prompt_dynamic_slots,
        answer_gt=plan.answer_gt,
        annotation_entity_ids=plan.annotation_entity_ids,
        annotation_type=plan.annotation_type,
        query_spec_params=plan.query_spec_params,
        execution_updates=lambda sample: {
            **dict(plan.execution_updates(sample)),
            "first_exit_tick": min(
                vehicle_exit_tick(vehicle, lane_count=int(sample.lane_count))
                for vehicle in sample.vehicles
                if vehicle.option_label is not None
            ),
        },
    )


@register_task
class GamesCrossingFirstExitObjectLabelTask:
    """Identify which labeled moving object leaves the board first."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a straight marked-route first-collision label question."""

        return run_crossing_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_first_exit_object_label_objective,
        )


__all__ = ["GamesCrossingFirstExitObjectLabelTask"]
