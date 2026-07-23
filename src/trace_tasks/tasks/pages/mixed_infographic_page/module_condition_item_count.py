"""Mixed-infographic module condition count task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    DOMAIN,
    condition_count_payload,
    run_bound_task,
    task_generation_defaults,
    task_render_defaults,
)
from .shared.metrics import _select_condition_target


TASK_ID = "task_pages__mixed_infographic_page__module_condition_item_count"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "module_condition_item_count"
GEN_DEFAULTS = task_generation_defaults(
    module_count_support=(5,),
    item_count_support=(6, 7, 8),
    field_count_support=(2, 3),
)
RENDER_DEFAULTS = task_render_defaults()


def _bind_target(ctx: Any, task_params: Mapping[str, Any], instance_seed: int) -> Any:
    """Bind all item value cells matching one numeric condition."""

    module, field, target, *probabilities = _select_condition_target(
        sampling_namespace=TASK_ID,
        gen_defaults=ctx.gen_defaults,
        spec=ctx.spec,
        params=task_params,
        instance_seed=int(instance_seed),
    )
    probability_keys = (
        "target_module_index_probabilities",
        "target_field_index_probabilities",
        "condition_operator_probabilities",
        "threshold_rank_index_probabilities",
        "condition_answer_count_probabilities",
    )
    return condition_count_payload(
        ctx=ctx,
        target_module=module,
        target_field=field,
        target=target,
        probability_payload=dict(zip(probability_keys, probabilities)),
        prompt_key=PROMPT_QUERY_KEY,
    )


@register_task
class PagesMixedInfographicModuleConditionItemCountTask:
    """Count items in one module whose field value satisfies a condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Generate one module condition count instance."""

        return run_bound_task(
            instance_seed=int(instance_seed),
            params=params,
            public_task=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            prompt_key=PROMPT_QUERY_KEY,
            gen_defaults=GEN_DEFAULTS,
            render_defaults=RENDER_DEFAULTS,
            bind_target=_bind_target,
            max_attempts=int(max_attempts),
        )
