"""Select the pictogram category with an extremal scaled total."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.composition.values import select_unique_extremum
from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_pictogram_plan, dataset_from_base, make_pictogram_query, run_pictogram_task
from .shared.defaults import DOMAIN, support_probability_map
from .shared.sampling import resolve_categories, sample_balanced_int, sample_base


LARGEST_TOTAL_QUERY_ID = "largest_total_category_label"
SMALLEST_TOTAL_QUERY_ID = "smallest_total_category_label"
CATEGORY_TOTAL_EXTREMUM_QUERY_IDS = (LARGEST_TOTAL_QUERY_ID, SMALLEST_TOTAL_QUERY_ID)


def _build_plan(params: dict, instance_seed: int, selected: str, probabilities: dict[str, float]):
    # Use distinct mark counts so the requested extremum has one unambiguous category-row witness.
    base = sample_base(params, instance_seed=int(instance_seed))
    mark_min, mark_max = tuple(int(value) for value in base.mark_count_range)
    category_count = len(base.mark_counts)
    mark_support = list(range(int(mark_min), int(mark_max) + 1))
    if int(category_count) > len(mark_support):
        raise ValueError("category-total extremum task requires enough distinct mark counts")

    rng = spawn_rng(int(instance_seed), "charts.pictogram.category_total_extremum.mark_counts")
    mark_counts = [int(value) for value in rng.sample(mark_support, k=int(category_count))]
    target_index = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"charts.pictogram.category_total_extremum.answer_index.{selected}",
        low=0,
        high=int(category_count) - 1,
    )
    if str(selected) == LARGEST_TOTAL_QUERY_ID:
        target_mark_count = max(mark_counts)
        direction = "largest"
    elif str(selected) == SMALLEST_TOTAL_QUERY_ID:
        target_mark_count = min(mark_counts)
        direction = "smallest"
    else:
        raise ValueError(f"unsupported pictogram total-extremum query id: {selected}")
    source_index = mark_counts.index(int(target_mark_count))
    mark_counts[int(source_index)], mark_counts[int(target_index)] = (
        mark_counts[int(target_index)],
        mark_counts[int(source_index)],
    )

    categories = resolve_categories(
        mark_counts=mark_counts,
        unit_scale=int(base.unit_scale),
        params=params,
        instance_seed=int(instance_seed),
    )
    selected_category = select_unique_extremum(
        tuple((category, int(category.total)) for category in categories),
        select_largest=str(selected) == LARGEST_TOTAL_QUERY_ID,
        min_margin=1,
        error_label="pictogram category-total extremum",
        item_label="categories",
    ).item
    if str(selected_category.category_id) != str(categories[int(target_index)].category_id):
        raise RuntimeError("pictogram category-total extremum construction drifted")
    answer_category = selected_category
    query = make_pictogram_query(
        selected=str(selected),
        answer=str(answer_category.label),
        answer_type="string",
        annotation_type="bbox",
        annotation_category_ids=(str(answer_category.category_id),),
        params={
            "answer_category_id": str(answer_category.category_id),
            "answer_category_index": int(target_index),
            "answer_category_label": str(answer_category.label),
            "answer_mark_count": int(target_mark_count),
            "answer_total": int(target_mark_count) * int(base.unit_scale),
            "extremum_direction": str(direction),
            "answer_mark_count_probabilities": support_probability_map(range(mark_min, mark_max + 1)),
        },
    )
    dataset = dataset_from_base(
        base=base,
        query=query,
        categories=tuple(categories),
        selected=str(selected),
        probabilities=probabilities,
    )
    return build_pictogram_plan(
        dataset=dataset,
        prompt_task_key="category_total_extremum_label",
        prompt_query_key=str(selected),
    )


@register_task
class ChartsPictogramCategoryTotalExtremumLabelTask:
    task_id = "task_charts__pictogram__category_total_extremum_label"
    reasoning_operations = ('filtering', 'ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "category_total_extremum_label"
    supported_query_ids = CATEGORY_TOTAL_EXTREMUM_QUERY_IDS
    default_query_id = LARGEST_TOTAL_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_pictogram_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsPictogramCategoryTotalExtremumLabelTask", "CATEGORY_TOTAL_EXTREMUM_QUERY_IDS"]
