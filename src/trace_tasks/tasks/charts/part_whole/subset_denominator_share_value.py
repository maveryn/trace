"""Public task for `task_charts__part_whole__subset_denominator_share_value`."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_part_whole_plan, run_part_whole_task, sample_part_whole_base
from .shared.defaults import DOMAIN, SAMPLING_NAMESPACE
from .shared.sampling import base_extras, sample_subset_denominator
from .shared.state import PartWholeDataset


SINGLE = "single"
SUPPORTED_QUERY_IDS = (SINGLE,)


def _build_plan(params, instance_seed: int, selected: str, _probabilities):
    """Bind one named denominator subset and compute target share within it."""

    base = sample_part_whole_base(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.subset.category_count",
        min_key="part_whole_category_count_min",
        max_key="part_whole_category_count_max",
        fallback_min=4,
        fallback_max=7,
    )
    categories, subset_categories, subset_extras = sample_subset_denominator(
        category_count=len(base.categories),
        share_min=int(base.value_min),
        share_max=int(base.value_max),
        params=params,
        count_params=base.count_params,
        instance_seed=int(instance_seed),
    )
    answer_value = int(subset_extras["subset_denominator_percent"])
    extras = {
        **base_extras(
            categories,
            category_count_range=base.category_count_range,
            value_min=int(base.value_min),
            value_max=int(base.value_max),
        ),
        **dict(subset_extras),
    }
    dataset = PartWholeDataset(
        categories=tuple(categories),
        answer_value=int(answer_value),
        annotation_labels=tuple(str(category.label) for category in subset_categories),
        trace_extras=dict(extras),
    )
    return build_part_whole_plan(
        dataset=dataset,
        base=base,
        selected=str(selected),
        instance_seed=int(instance_seed),
    )


@register_task
class ChartsCompositionSubsetDenominatorShareValueTask:
    """Return a category's percentage within a visible category subset."""

    task_id = "task_charts__part_whole__subset_denominator_share_value"
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "subset_denominator_share_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Return a generated part-whole instance for the single subset query."""

        return run_part_whole_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsCompositionSubsetDenominatorShareValueTask"]
