"""Mixed-infographic module field total task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    DOMAIN,
    module_total_payload,
    run_bound_task,
    task_generation_defaults,
    task_render_defaults,
)
from .shared.metrics import _select_total_target


TASK_ID = "task_pages__mixed_infographic_page__module_field_total_value"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "module_field_total_value"
GEN_DEFAULTS = task_generation_defaults(
    module_count_support=(8, 9),
    item_count_support=(3, 4),
    field_count_support=(3,),
)
RENDER_DEFAULTS = task_render_defaults()


def _bind_target(ctx: Any, task_params: Mapping[str, Any], instance_seed: int) -> Any:
    """Bind the value cells summed for one module field."""

    module, field, target, module_probs, field_probs = _select_total_target(
        sampling_namespace=TASK_ID,
        spec=ctx.spec,
        params=task_params,
        instance_seed=int(instance_seed),
    )
    return module_total_payload(
        ctx=ctx,
        target_module=module,
        target_field=field,
        target=target,
        module_probs=module_probs,
        field_probs=field_probs,
        prompt_key=PROMPT_QUERY_KEY,
    )


@register_task
class PagesMixedInfographicModuleFieldTotalValueTask:
    """Sum one additive numeric field across all items in one module."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Generate one module field total instance."""

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
