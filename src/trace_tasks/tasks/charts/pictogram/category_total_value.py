"""Read one category total from repeated unit marks."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from ._lifecycle import make_pictogram_query, plan_from_mark_counts, run_pictogram_task
from .shared.defaults import DOMAIN, support_probability_map
from .shared.sampling import sample_balanced_int, sample_base


SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _build_plan(params: dict, instance_seed: int, selected: str, probabilities: dict[str, float]):
    # Select one visible category row, force its mark count, then bind that row box as the single witness.
    base = sample_base(params, instance_seed=int(instance_seed))
    mark_min, mark_max = tuple(int(value) for value in base.mark_count_range)
    category_count = len(base.mark_counts)
    target_index = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace="charts.pictogram.category_total.target_index",
        low=0,
        high=category_count - 1,
    )
    target_mark = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace="charts.pictogram.category_total.answer_marks",
        low=mark_min,
        high=mark_max,
    )
    mark_counts = list(base.mark_counts)
    mark_counts[target_index] = int(target_mark)
    query = make_pictogram_query(
        selected=str(selected),
        answer=int(target_mark) * int(base.unit_scale),
        annotation_type="bbox",
        annotation_category_ids=(f"cat_{target_index}",),
        params={
            "target_category_id": f"cat_{target_index}",
            "target_category_index": int(target_index),
            "target_mark_count": int(target_mark),
            "answer_mark_count": int(target_mark),
            "answer_mark_count_probabilities": support_probability_map(range(mark_min, mark_max + 1)),
        },
    )
    return plan_from_mark_counts(
        base=base,
        mark_counts=mark_counts,
        query=query,
        selected=str(selected),
        probabilities=probabilities,
        params=params,
        instance_seed=int(instance_seed),
        prompt_task_key="category_total_value",
    )


@register_task
class ChartsPictogramCategoryTotalValueTask:
    task_id = "task_charts__pictogram__category_total_value"
    reasoning_operations = ('filtering', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "category_total_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_pictogram_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsPictogramCategoryTotalValueTask"]
