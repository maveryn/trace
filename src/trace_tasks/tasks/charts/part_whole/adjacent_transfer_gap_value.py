from __future__ import annotations

from trace_tasks.tasks.registry import register_task

from ._lifecycle import finish_part_whole_plan, run_part_whole_task, sample_part_whole_base
from .shared.defaults import DOMAIN, SAMPLING_NAMESPACE
from .shared.sampling import sample_adjacent_transfer


SUPPORTED_QUERY_IDS = ("clockwise_adjacent_transfer", "counterclockwise_adjacent_transfer")


def _build_plan(params, instance_seed: int, selected: str, _probabilities):
    # Bind adjacent source/target segments, then compute the post-transfer gap.
    direction = "counterclockwise" if str(selected).startswith("counter") else "clockwise"
    base = sample_part_whole_base(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.adjacent_transfer.category_count",
        min_key="counterfactual_category_count_min",
        max_key="counterfactual_category_count_max",
        fallback_min=4,
        fallback_max=6,
    )
    transfer = sample_adjacent_transfer(
        base.categories,
        direction=direction,
        params=params,
        count_params=base.count_params,
        instance_seed=int(instance_seed),
    )
    source_new = transfer.source.value - transfer.delta
    target_new = transfer.target.value + transfer.delta
    answer_value = abs(target_new - source_new)
    extras = {
        **dict(base.base_extras),
        **dict(transfer.extras),
        "source_category": transfer.source.label,
        "target_category": transfer.target.label,
        "source_original_value": transfer.source.value,
        "target_original_value": transfer.target.value,
        "source_new_value": source_new,
        "target_new_value": target_new,
        "transfer_delta": transfer.delta,
    }
    return finish_part_whole_plan(
        base=base,
        selected=selected,
        instance_seed=int(instance_seed),
        answer_value=answer_value,
        annotation_labels=(transfer.source.label, transfer.target.label),
        trace_extras=extras,
    )


@register_task
class ChartsCompositionChartAdjacentTransferGapValueTask:
    task_id = "task_charts__part_whole__adjacent_transfer_gap_value"
    reasoning_operations = ('spatial_relations', 'state_update', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "adjacent_transfer_gap_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_part_whole_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsCompositionChartAdjacentTransferGapValueTask"]
