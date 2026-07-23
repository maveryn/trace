"""Select the pictogram category total nearest to a target value."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.composition.values import select_unique_nearest
from trace_tasks.tasks.registry import register_task

from ._lifecycle import build_pictogram_plan, dataset_from_base, make_pictogram_query, run_pictogram_task
from .shared.defaults import DOMAIN
from .shared.sampling import resolve_categories, sample_balanced_int, sample_base


SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _nearest_category_candidates(totals: list[int], *, target_index: int) -> list[int]:
    """Return integer target values for which target_index is the unique nearest total."""

    lower = min(int(value) for value in totals)
    upper = max(int(value) for value in totals)
    candidates: list[int] = []
    for target_value in range(int(lower), int(upper) + 1):
        distances = [abs(int(total) - int(target_value)) for total in totals]
        best = min(distances)
        if distances.count(best) == 1 and distances[int(target_index)] == best:
            candidates.append(int(target_value))
    return candidates


def _build_plan(params: dict, instance_seed: int, selected: str, probabilities: dict[str, float]):
    # Force distinct totals, then choose a target value whose nearest category is unique.
    base = sample_base(params, instance_seed=int(instance_seed))
    mark_min, mark_max = tuple(int(value) for value in base.mark_count_range)
    category_count = len(base.mark_counts)
    mark_support = list(range(int(mark_min), int(mark_max) + 1))
    if int(category_count) > len(mark_support):
        raise ValueError("target-value nearest task requires enough distinct mark counts")

    rng = spawn_rng(int(instance_seed), "charts.pictogram.target_value_nearest.mark_counts")
    mark_counts = [int(value) for value in rng.sample(mark_support, k=int(category_count))]
    categories = resolve_categories(
        mark_counts=mark_counts,
        unit_scale=int(base.unit_scale),
        params=params,
        instance_seed=int(instance_seed),
    )
    totals = [int(category.total) for category in categories]
    target_index = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace="charts.pictogram.target_value_nearest.answer_index",
        low=0,
        high=int(category_count) - 1,
    )
    candidates = _nearest_category_candidates(totals, target_index=int(target_index))
    if not candidates:
        raise ValueError("could not construct unique nearest-category target value")
    candidate_index = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace="charts.pictogram.target_value_nearest.target_value",
        low=0,
        high=len(candidates) - 1,
    )
    target_value = int(candidates[int(candidate_index)])
    selected_category = select_unique_nearest(
        tuple(categories),
        value_fn=lambda category: int(category.total),
        target_value=int(target_value),
        min_margin=1,
        error_label="pictogram target-value nearest category",
        item_label="categories",
    ).item
    if str(selected_category.category_id) != str(categories[int(target_index)].category_id):
        raise RuntimeError("pictogram nearest-category construction drifted")
    answer_category = selected_category
    nearest_distance = abs(int(answer_category.total) - int(target_value))
    query = make_pictogram_query(
        selected=str(selected),
        answer=str(answer_category.label),
        answer_type="string",
        annotation_type="bbox",
        annotation_category_ids=(str(answer_category.category_id),),
        params={
            "target_value": int(target_value),
            "answer_category_id": str(answer_category.category_id),
            "answer_category_index": int(target_index),
            "answer_category_label": str(answer_category.label),
            "answer_category_total": int(answer_category.total),
            "nearest_distance": int(nearest_distance),
            "target_value_candidate_count": int(len(candidates)),
            "target_value_range": [int(min(totals)), int(max(totals))],
        },
    )
    dataset = dataset_from_base(
        base=base,
        categories=tuple(categories),
        query=query,
        selected=str(selected),
        probabilities=probabilities,
    )
    return build_pictogram_plan(
        dataset=dataset,
        prompt_task_key="target_value_nearest_category_label",
    )


@register_task
class ChartsPictogramTargetValueNearestCategoryLabelTask:
    task_id = "task_charts__pictogram__target_value_nearest_category_label"
    reasoning_operations = ('filtering', 'ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "target_value_nearest_category_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_pictogram_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsPictogramTargetValueNearestCategoryLabelTask"]
