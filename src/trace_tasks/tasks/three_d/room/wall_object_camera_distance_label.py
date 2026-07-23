"""Public task wrapper for closest-camera wall-object selection."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.three_d.room._lifecycle import (
    RoomObjectivePlan,
    build_room_option_objective_plan,
    require_single_query,
    resolve_room_option_context,
    run_room_lifecycle,
)
from trace_tasks.tasks.three_d.room.shared.metrics import (
    CAMERA_DISTANCE_MIN_MARGIN,
    CAMERA_DISTANCE_MIN_ROOM_DEPTH_MARGIN,
    LETTERED_WALL_OBJECT_MIN_VISIBLE_PX,
    build_room_wall_camera_distance_dataset,
)
from trace_tasks.tasks.three_d.room.shared.state import SCENE_ID
from trace_tasks.tasks.three_d.shared.object_scene import ObjectSceneRenderParams


TASK_ID = "task_three_d__room__wall_object_camera_distance_label"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _prepare_camera_distance_objective(
    objective_seed: int,
    objective_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _prompt_defaults: Mapping[str, Any],
    render_params: ObjectSceneRenderParams,
) -> RoomObjectivePlan:
    """Bind closest-camera option semantics for this public task."""

    require_single_query(
        objective_params,
        expected=QUERY_ID,
        task_identifier=TASK_ID,
    )
    context = resolve_room_option_context(
        objective_params,
        gen_defaults,
        instance_seed=int(objective_seed),
        namespace=TASK_ID,
        candidate_min=6,
        candidate_max=6,
        candidate_lower=4,
        candidate_upper=6,
    )
    dataset = build_room_wall_camera_distance_dataset(
        scene_variant=str(context.scene_variant),
        candidate_count=int(context.counts.candidate_count),
        context_wall_count=int(context.counts.context_wall_count),
        floor_context_count=int(context.counts.floor_context_count),
        render_params=render_params,
        namespace=TASK_ID,
        instance_seed=int(objective_seed),
    )
    return build_room_option_objective_plan(
        dataset=dict(dataset),
        public_query_id=QUERY_ID,
        query_probabilities={QUERY_ID: 1.0},
        prompt_query_key="closest_to_camera",
        prompt_dynamic_slots={},
        context=context,
    )


@register_task
class ThreeDRoomWallObjectCameraDistanceLabelTask:
    """Choose the option-panel wall-mounted object closest to the camera."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
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
            prepare_objective=_prepare_camera_distance_objective,
        )
        return output


__all__ = [
    "CAMERA_DISTANCE_MIN_MARGIN",
    "CAMERA_DISTANCE_MIN_ROOM_DEPTH_MARGIN",
    "LETTERED_WALL_OBJECT_MIN_VISIBLE_PX",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDRoomWallObjectCameraDistanceLabelTask",
]
