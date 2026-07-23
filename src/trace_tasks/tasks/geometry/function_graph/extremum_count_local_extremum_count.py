from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    LOCAL_EXTREMUM_PROMPT_KEYS,
    local_extremum_count_prompt_slots,
    prepare_function_graph_count_plan,
    run_function_graph_count_entry,
)
from .shared.defaults import DOMAIN
from .shared.sampling import (
    local_extremum_support_by_family,
    sample_local_extremum_scene,
)

TASK_ID = "task_geometry__function_graph__extremum_count_local_extremum_count"
SUPPORTED_QUERY_IDS = ("minimum", "maximum")
PROMPT_TEMPLATE_KEY = "local_extremum_count"


def _prepare_local_extremum_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    query_id: str,
    query_probabilities: Mapping[str, float],
):
    suffix = "minimum" if str(query_id) == "minimum" else "maximum"
    extremum_sign = -1 if suffix == "minimum" else 1
    return prepare_function_graph_count_plan(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        branch_name=str(query_id),
        branch_probabilities=query_probabilities,
        support_by_family=local_extremum_support_by_family(),
        prompt_template_key=PROMPT_TEMPLATE_KEY,
        prompt_default_keys=LOCAL_EXTREMUM_PROMPT_KEYS,
        build_prompt_slots=lambda defaults, family, _target: local_extremum_count_prompt_slots(
            defaults,
            family=str(family),
            extremum_kind=suffix,
        ),
        sample_scene=lambda rng, family, target: sample_local_extremum_scene(
            rng,
            family=str(family),
            target_count=int(target),
            extremum_sign=int(extremum_sign),
        ),
        build_scene_relations=lambda family, target: {
            "target_count": int(target),
            "scene_variant": str(family),
            "extremum_kind": suffix,
        },
        extra_query_params={"extremum_sign": int(extremum_sign)},
    )


@register_task
class GeometryGraphingLocalExtremumCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_function_graph_count_entry(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id="minimum",
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_local_extremum_objective,
        )
