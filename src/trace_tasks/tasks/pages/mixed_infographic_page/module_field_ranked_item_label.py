"""Mixed-infographic module field ranked item task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    DOMAIN,
    module_rank_payload,
    run_bound_task,
    task_generation_defaults,
    task_render_defaults,
)
from .shared.metrics import _select_ranked_target


TASK_ID = "task_pages__mixed_infographic_page__module_field_ranked_item_label"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "module_field_ranked_item_label"
GEN_DEFAULTS = task_generation_defaults(
    module_count_support=(7, 8, 9),
    item_count_support=(3, 4),
    field_count_support=(2, 3),
    extra={"rank_position_support": [1, 2, 3]},
)
RENDER_DEFAULTS = task_render_defaults()


def _bind_target(ctx: Any, task_params: Mapping[str, Any], instance_seed: int) -> Any:
    """Bind the requested ranked item in one module field."""

    module, field, target, module_probs, field_probs, direction_probs, rank_probs = (
        _select_ranked_target(
            sampling_namespace=TASK_ID,
            gen_defaults=ctx.gen_defaults,
            spec=ctx.spec,
            params=task_params,
            instance_seed=int(instance_seed),
        )
    )
    return module_rank_payload(
        ctx=ctx,
        target_module=module,
        target_field=field,
        target=target,
        module_probs=module_probs,
        field_probs=field_probs,
        direction_probs=direction_probs,
        prompt_key=PROMPT_QUERY_KEY,
        item_key="ranked_item",
        value_key="ranked_value",
        rank_probs=rank_probs,
    )


@register_task
class PagesMixedInfographicModuleFieldRankedItemLabelTask:
    """Find the item at a requested numeric rank in one module field."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Generate one module field ranked item instance."""

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
