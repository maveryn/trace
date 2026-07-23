"""Mixed-infographic two-field condition item task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    DOMAIN,
    run_bound_task,
    task_generation_defaults,
    task_render_defaults,
    two_field_condition_payload,
)
from .shared.metrics import _select_two_field_condition_target


TASK_ID = "task_pages__mixed_infographic_page__module_two_field_condition_item_label"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "module_two_field_condition_item_label"
GEN_DEFAULTS = task_generation_defaults(
    module_count_support=(7, 8),
    item_count_support=(4, 5),
    field_count_support=(3,),
)
RENDER_DEFAULTS = task_render_defaults()


def _bind_target(ctx: Any, task_params: Mapping[str, Any], instance_seed: int) -> Any:
    """Bind the unique item satisfying numeric and categorical conditions."""

    module, numeric_field, category_field, target, *probabilities = (
        _select_two_field_condition_target(
            sampling_namespace=TASK_ID,
            gen_defaults=ctx.gen_defaults,
            spec=ctx.spec,
            params=task_params,
            instance_seed=int(instance_seed),
        )
    )
    probability_keys = (
        "target_module_index_probabilities",
        "target_numeric_field_index_probabilities",
        "target_category_field_index_probabilities",
        "condition_operator_probabilities",
        "category_value_probabilities",
        "threshold_rank_index_probabilities",
    )
    return two_field_condition_payload(
        ctx=ctx,
        target_module=module,
        numeric_field=numeric_field,
        category_field=category_field,
        target=target,
        probability_payload=dict(zip(probability_keys, probabilities)),
        prompt_key=PROMPT_QUERY_KEY,
    )


@register_task
class PagesMixedInfographicModuleTwoFieldConditionItemLabelTask:
    """Find the item satisfying one numeric and one categorical condition."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'comparison')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Generate one two-field condition lookup instance."""

        return run_bound_task(
            instance_seed=int(instance_seed),
            params=params,
            public_task=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            prompt_key=PROMPT_QUERY_KEY,
            gen_defaults=GEN_DEFAULTS,
            render_defaults=RENDER_DEFAULTS,
            bind_target=_bind_target,
            context_options={"allow_categorical_value_reuse": True},
            max_attempts=int(max_attempts),
        )
