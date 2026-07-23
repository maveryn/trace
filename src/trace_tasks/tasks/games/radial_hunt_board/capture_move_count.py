from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import ObjectiveRadialHuntBoardPlan, run_radial_hunt_board_lifecycle
from .shared.sampling import (
    resolve_radial_hunt_board_target_axis,
    sample_capture_scene,
)
from .shared.state import (
    SCENE_ID,
    SCENE_NAMESPACE,
)


TASK_ID = "task_games__radial_hunt_board__capture_move_count"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "capture_move_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_capture_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> ObjectiveRadialHuntBoardPlan:
    """Prepare a board constructor with a fixed capture-move answer."""

    target_axis = resolve_radial_hunt_board_target_axis(
        int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace=f"{SCENE_NAMESPACE}.capture.target_answer",
    )
    target = int(target_axis.target_answer)
    if target < 0 or target > 6:
        raise ValueError("radial hunt capture target answer must be in 0..6")

    return ObjectiveRadialHuntBoardPlan(
        attempt_namespace=f"{SCENE_NAMESPACE}.capture",
        prompt_query_key=PROMPT_QUERY_KEY,
        rule_slot_name="capture_rule_text",
        annotation_trace_key="capture_destinations",
        common_params={
            "target_answer": int(target),
            "target_answer_support": [int(value) for value in target_axis.target_answer_support],
            "target_answer_probabilities": dict(target_axis.target_answer_probabilities),
        },
        construct_attempt=lambda rng, axes: sample_capture_scene(
            rng=rng,
            axes=axes,
            target_axis=target_axis,
            gen_defaults=gen_defaults,
        ),
    )


@register_task
class GamesRadialHuntBoardCaptureMoveCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any] | None = None, max_attempts: int = 100) -> TaskOutput:
        return run_radial_hunt_board_lifecycle(
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
            prepare_objective=_prepare_capture_objective,
            namespace=f"{SCENE_NAMESPACE}.capture",
        )
