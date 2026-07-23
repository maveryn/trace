"""Public task wrapper for wall-plane side-relation selection."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.three_d.room._lifecycle import (
    RoomObjectivePlan,
    build_room_option_objective_plan,
    resolve_room_choice,
    resolve_room_option_context,
    run_room_lifecycle,
)
from trace_tasks.tasks.three_d.room.shared.spatial_primitives import REFERENCE_OBJECT_TYPE
from trace_tasks.tasks.three_d.room.shared.spatial_primitives import build_room_wall_side_relation_dataset
from trace_tasks.tasks.three_d.room.shared.state import SCENE_ID
from trace_tasks.tasks.three_d.shared.object_scene import ObjectSceneRenderParams
from trace_tasks.tasks.three_d.shared.task_support import resolve_axis_variant_for_namespace


TASK_ID = "task_three_d__room__wall_object_side_relation_label"
SUPPORTED_QUERY_IDS = ("left_of_reference_on_wall", "right_of_reference_on_wall")
SIDE_RELATION_BY_QUERY_ID = {
    "left_of_reference_on_wall": "left",
    "right_of_reference_on_wall": "right",
}


def _prepare_side_relation_objective(
    objective_seed: int,
    objective_params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _prompt_defaults: Mapping[str, Any],
    render_params: ObjectSceneRenderParams,
) -> RoomObjectivePlan:
    """Bind wall-plane side relation semantics for this public task."""

    public_branch, branch_probabilities = resolve_axis_variant_for_namespace(
        objective_params,
        namespace=f"{TASK_ID}.query_id",
        gen_defaults=gen_defaults,
        instance_seed=int(objective_seed),
        supported_variants=SUPPORTED_QUERY_IDS,
        explicit_key="query_id",
        weights_key="query_id_weights",
        balance_flag_key="balanced_query_id_sampling",
    )
    side_relation = str(SIDE_RELATION_BY_QUERY_ID[str(public_branch)])
    context = resolve_room_option_context(
        objective_params,
        gen_defaults,
        instance_seed=int(objective_seed),
        namespace=TASK_ID,
        candidate_min=6,
        candidate_max=6,
        candidate_lower=6,
        candidate_upper=6,
    )
    reference_wall_support = (
        ("back", "right") if side_relation == "right" else ("back", "left", "right")
    )
    reference_wall, reference_wall_probabilities = resolve_room_choice(
        objective_params,
        instance_seed=int(objective_seed),
        key="reference_wall",
        support=reference_wall_support,
        namespace=TASK_ID,
    )
    dataset = build_room_wall_side_relation_dataset(
        side_relation=str(side_relation),
        scene_variant=str(context.scene_variant),
        candidate_count=int(context.counts.candidate_count),
        context_wall_count=int(context.counts.context_wall_count),
        floor_context_count=int(context.counts.floor_context_count),
        reference_wall=str(reference_wall),
        render_params=render_params,
        namespace=TASK_ID,
        instance_seed=int(objective_seed),
    )
    reference_name = str(dataset["reference_object"]["prompt_name"])
    return build_room_option_objective_plan(
        dataset=dict(dataset),
        public_query_id=str(public_branch),
        query_probabilities=dict(branch_probabilities),
        prompt_query_key=str(public_branch),
        prompt_dynamic_slots={"reference_name": reference_name},
        context=context,
        extra_query_params={
            "reference_wall": str(reference_wall),
            "reference_wall_probabilities": dict(reference_wall_probabilities),
            "reference_object_type": REFERENCE_OBJECT_TYPE,
            "reference_name": reference_name,
            "side_relation": str(side_relation),
        },
    )


@register_task
class ThreeDRoomWallObjectSideRelationLabelTask:
    """Choose the option-panel wall object on the requested side of a TV."""

    task_id = TASK_ID
    reasoning_operations = ('logical_composition', 'spatial_relations')
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
            prepare_objective=_prepare_side_relation_objective,
        )
        return output


__all__ = [
    "REFERENCE_OBJECT_TYPE",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDRoomWallObjectSideRelationLabelTask",
]
