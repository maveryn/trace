from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    TURNING_COUNT_PROMPT_KEYS,
    integer_count_prompt_slots,
    prepare_function_graph_count_plan,
    run_function_graph_count_entry,
)
from .shared.defaults import DOMAIN
from .shared.prompts import function_object_description, prompt_asset_slot
from .shared.sampling import (
    sample_turning_scene,
    turning_count_support_by_family,
)

TASK_ID = "task_geometry__function_graph__extremum_count_turning_point_count"
SUPPORTED_QUERY_IDS = ("single",)
PROMPT_TEMPLATE_KEY = "turning_point_count"


def _prompt_slots(defaults: Mapping[str, Any], *, family: str):
    return integer_count_prompt_slots(
        defaults,
        object_description=function_object_description(defaults=defaults, family=str(family), has_guide_line=False),
        annotation_hint=prompt_asset_slot(defaults, "annotation_hint_turning_point_count"),
        json_example=prompt_asset_slot(defaults, "json_example_turning_point_count"),
        json_example_answer_only=prompt_asset_slot(defaults, "json_example_turning_point_count_answer_only"),
    )


def _prepare_turning_point_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    query_probabilities: Mapping[str, float],
):
    return prepare_function_graph_count_plan(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        branch_name=str(query_id),
        branch_probabilities=query_probabilities,
        support_by_family=turning_count_support_by_family(),
        prompt_template_key=PROMPT_TEMPLATE_KEY,
        prompt_default_keys=TURNING_COUNT_PROMPT_KEYS,
        build_prompt_slots=lambda defaults, family, _target: _prompt_slots(defaults, family=str(family)),
        sample_scene=lambda rng, family, target: sample_turning_scene(rng, family=str(family), target_count=int(target)),
        build_scene_relations=lambda family, target: {"target_count": int(target), "scene_variant": str(family)},
    )


@register_task
class GeometryGraphingTurningPointCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_function_graph_count_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id="single",
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_turning_point_objective,
        )
