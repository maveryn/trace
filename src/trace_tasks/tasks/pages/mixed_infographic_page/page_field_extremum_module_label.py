"""Mixed-infographic page field extremum module task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    DOMAIN,
    page_field_extremum_payload,
    run_bound_task,
    task_generation_defaults,
    task_render_defaults,
)
from .shared.metrics import _select_page_field_extremum_target
from .shared.state import NUMERIC_FIELD_LABELS


TASK_ID = "task_pages__mixed_infographic_page__page_field_extremum_module_label"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "page_field_extremum_module_label"
GEN_DEFAULTS = task_generation_defaults(
    module_count_support=(7, 8, 9),
    item_count_support=(2, 3, 4),
    field_count_support=(2, 3),
)
RENDER_DEFAULTS = task_render_defaults()


def _context_options(task_params: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep the queried numeric field available across modules."""

    return {
        "ensure_shared_numeric_field": True,
        "shared_numeric_field_label": (
            str(task_params["target_field_label"])
            if task_params.get("target_field_label") is not None
            else None
        ),
        "shared_numeric_field_choices": NUMERIC_FIELD_LABELS,
    }


def _bind_target(ctx: Any, task_params: Mapping[str, Any], instance_seed: int) -> Any:
    """Bind the page-wide highest/lowest module for one field."""

    module, field, target, field_probs, direction_probs = _select_page_field_extremum_target(
        sampling_namespace=TASK_ID,
        gen_defaults=ctx.gen_defaults,
        spec=ctx.spec,
        params=task_params,
        instance_seed=int(instance_seed),
    )
    return page_field_extremum_payload(
        ctx=ctx,
        target_module=module,
        target_field=field,
        target=target,
        field_probs=field_probs,
        direction_probs=direction_probs,
        prompt_key=PROMPT_QUERY_KEY,
    )


@register_task
class PagesMixedInfographicPageFieldExtremumModuleLabelTask:
    """Find the module containing the page-wide extremum for one field."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Generate one page-wide field extremum instance."""

        return run_bound_task(
            instance_seed=int(instance_seed),
            params=params,
            public_task=TASK_ID,
            supported_branches=SUPPORTED_QUERY_IDS,
            prompt_key=PROMPT_QUERY_KEY,
            gen_defaults=GEN_DEFAULTS,
            render_defaults=RENDER_DEFAULTS,
            bind_target=_bind_target,
            context_options=_context_options,
            max_attempts=int(max_attempts),
        )
