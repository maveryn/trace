"""Public task wrapper for same-wall reference selection."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.three_d.room._lifecycle import (
    RoomObjectivePlan,
    prepare_same_wall_reference_objective_from_semantics,
    run_room_lifecycle,
)
from trace_tasks.tasks.three_d.room.shared.state import SCENE_ID
from trace_tasks.tasks.three_d.shared.object_scene import ObjectSceneRenderParams


TASK_ID = "task_three_d__room__wall_object_same_wall_reference_label"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _prepare_same_wall_reference_objective(
    objective_seed: int,
    objective_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _prompt_defaults: Mapping[str, Any],
    render_params: ObjectSceneRenderParams,
) -> RoomObjectivePlan:
    """Bind same-wall reference selection semantics for this public task."""

    return prepare_same_wall_reference_objective_from_semantics(
        task_identifier=TASK_ID,
        single_branch=QUERY_ID,
        objective_seed=int(objective_seed),
        objective_params=objective_params,
        gen_defaults=gen_defaults,
        render_params=render_params,
    )


@register_task
class ThreeDRoomWallObjectSameWallReferenceLabelTask:
    """Choose the option-panel wall object on the same wall as a reference."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        task_params = dict(params)
        output = run_room_lifecycle(
            int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            task_identifier=TASK_ID,
            prepare_objective=_prepare_same_wall_reference_objective,
        )
        return output


__all__ = ["SCENE_ID", "SUPPORTED_QUERY_IDS", "TASK_ID", "ThreeDRoomWallObjectSameWallReferenceLabelTask"]
