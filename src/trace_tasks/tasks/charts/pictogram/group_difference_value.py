"""Compute the difference between two repeated-mark category totals."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds

from ._lifecycle import make_pictogram_query, plan_from_mark_counts, run_pictogram_task
from .shared.defaults import DOMAIN, GEN_DEFAULTS, SCENE_NAMESPACE, support_probability_map
from .shared.sampling import sample_balanced_int, sample_base


SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


def _build_plan(params: dict, instance_seed: int, selected: str, probabilities: dict[str, float]):
    # Select two visible category rows, force a mark-count gap, and bind both row boxes by label.
    base = sample_base(params, instance_seed=int(instance_seed))
    mark_min, mark_max = tuple(int(value) for value in base.mark_count_range)
    diff_min, diff_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="difference_mark_min",
        max_key="difference_mark_max",
        fallback_min=2,
        fallback_max=10,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    diff_max = min(int(diff_max), max(1, int(mark_max) - int(mark_min)))
    diff_min = min(int(diff_min), int(diff_max))
    diff_marks = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace="charts.pictogram.group_difference.diff_marks",
        low=int(diff_min),
        high=int(diff_max),
    )
    rng = spawn_rng(int(instance_seed), "charts.pictogram.group_difference.pair")
    pair = sorted(rng.sample(range(len(base.mark_counts)), k=2))
    low_value = rng.randint(int(mark_min), int(mark_max) - int(diff_marks))
    high_value = int(low_value) + int(diff_marks)
    mark_counts = list(base.mark_counts)
    if rng.random() < 0.5:
        mark_counts[pair[0]] = int(low_value)
        mark_counts[pair[1]] = int(high_value)
    else:
        mark_counts[pair[0]] = int(high_value)
        mark_counts[pair[1]] = int(low_value)
    query = make_pictogram_query(
        selected=str(selected),
        answer=int(diff_marks) * int(base.unit_scale),
        annotation_type="bbox_map",
        annotation_category_ids=(f"cat_{pair[0]}", f"cat_{pair[1]}"),
        params={
            "category_id_a": f"cat_{pair[0]}",
            "category_id_b": f"cat_{pair[1]}",
            "category_index_a": int(pair[0]),
            "category_index_b": int(pair[1]),
            "difference_mark_count": int(diff_marks),
            "difference_mark_count_probabilities": support_probability_map(range(int(diff_min), int(diff_max) + 1)),
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
        prompt_task_key="group_difference_value",
    )


@register_task
class ChartsPictogramGroupDifferenceValueTask:
    task_id = "task_charts__pictogram__group_difference_value"
    reasoning_operations = ('filtering', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "group_difference_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_pictogram_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsPictogramGroupDifferenceValueTask"]
