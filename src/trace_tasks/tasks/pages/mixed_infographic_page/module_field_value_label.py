"""Mixed-infographic module field value lookup task."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index

from ._lifecycle import (
    DOMAIN,
    module_ranked_field_value_payload,
    run_bound_task,
    task_generation_defaults,
    task_render_defaults,
)
from .shared.metrics import _select_ranked_target


TASK_ID = "task_pages__mixed_infographic_page__module_field_value_label"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "module_field_value_label"
GEN_DEFAULTS = task_generation_defaults(
    module_count_support=(8, 9),
    item_count_support=(4,),
    field_count_support=(3,),
    extra={"rank_position_support": [1]},
)
RENDER_DEFAULTS = task_render_defaults()


def _select_answer_field(
    *,
    module: Any,
    selector_field: Any,
    target: Mapping[str, Any],
    task_params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Any, Dict[str, float]]:
    """Choose a non-selector field for the already uniquely ranked item."""

    candidates = []
    for field_index, field in enumerate(module.fields):
        if str(field.field_id) == str(selector_field.field_id):
            continue
        candidates.append((int(field_index), field))
    if not candidates:
        raise ValueError("ranked field value lookup requires a non-selector answer field")

    explicit = task_params.get("answer_field_index")
    if explicit is not None:
        selected_index = int(explicit)
        selected = next((pair for pair in candidates if int(pair[0]) == selected_index), None)
        if selected is None:
            raise ValueError("answer_field_index does not identify a valid answer field")
    else:
        selected = candidates[
            int(
                resolve_selection_index(
                    params=task_params,
                    instance_seed=int(instance_seed),
                    namespace=(
                        f"{TASK_ID}.answer_field.{target['module_id']}."
                        f"{selector_field.field_id}.{target['item_id']}"
                    ),
                )
            )
            % int(len(candidates))
        ]
    probability = 1.0 / float(len(candidates))
    probs = {str(index): float(probability) for index, _field in candidates}
    if explicit is not None:
        probs = {str(int(selected[0])): 1.0}
    return selected[1], probs


def _bind_target(ctx: Any, task_params: Mapping[str, Any], instance_seed: int) -> Any:
    """Bind a ranked item and the requested answer field value."""

    module, selector_field, target, module_probs, selector_field_probs, direction_probs, rank_probs = (
        _select_ranked_target(
            sampling_namespace=TASK_ID,
            gen_defaults=GEN_DEFAULTS,
            spec=ctx.spec,
            params=task_params,
            instance_seed=int(instance_seed),
            module_predicate=lambda candidate_module: len(candidate_module.fields) > 1,
        )
    )
    answer_field, answer_field_probs = _select_answer_field(
        module=module,
        selector_field=selector_field,
        target=target,
        task_params=task_params,
        instance_seed=int(instance_seed),
    )
    item = next(item for item in module.items if str(item.item_id) == str(target["item_id"]))
    bound_target = dict(target)
    bound_target["answer_value"] = str(item.values_by_field_id[str(answer_field.field_id)])
    return module_ranked_field_value_payload(
        ctx=ctx,
        target_module=module,
        selector_field=selector_field,
        answer_field=answer_field,
        target=bound_target,
        module_probs=module_probs,
        selector_field_probs=selector_field_probs,
        answer_field_probs=answer_field_probs,
        direction_probs=direction_probs,
        rank_probs=rank_probs,
        prompt_key=PROMPT_QUERY_KEY,
    )


@register_task
class PagesMixedInfographicModuleFieldValueLabelTask:
    """Read a visible value from one module on a dense mixed infographic page."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int):
        """Generate one module-field value lookup instance."""

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
