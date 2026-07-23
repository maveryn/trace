"""Count repeated-mark categories satisfying a threshold."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds

from ._lifecycle import make_pictogram_query, plan_from_mark_counts, run_pictogram_task
from .shared.defaults import DOMAIN, GEN_DEFAULTS, SCENE_NAMESPACE, support_probability_map
from .shared.sampling import sample_balanced_int, sample_base


GREATER_THAN_QUERY_ID = "greater_than_threshold"
LESS_THAN_QUERY_ID = "less_than_threshold"
SUPPORTED_QUERY_IDS = (GREATER_THAN_QUERY_ID, LESS_THAN_QUERY_ID)


def _build_plan(params: dict, instance_seed: int, selected: str, probabilities: dict[str, float]):
    # Force a known number of category rows on one side of the selected threshold.
    base = sample_base(params, instance_seed=int(instance_seed))
    mark_min, mark_max = tuple(int(value) for value in base.mark_count_range)
    answer_min, answer_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="threshold_answer_min",
        max_key="threshold_answer_max",
        fallback_min=1,
        fallback_max=5,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    answer_max = min(int(answer_max), len(base.mark_counts) - 1)
    answer_min = min(int(answer_min), int(answer_max))
    target_count = sample_balanced_int(
        params=params,
        instance_seed=int(instance_seed),
        namespace="charts.pictogram.threshold.target_count",
        low=int(answer_min),
        high=int(answer_max),
    )
    direction = "less_than" if str(selected) == LESS_THAN_QUERY_ID else "greater_than"
    rng = spawn_rng(int(instance_seed), "charts.pictogram.threshold")
    threshold_mark = rng.randint(int(mark_min) + 1, int(mark_max) - 1)
    target_indices = set(rng.sample(range(len(base.mark_counts)), k=int(target_count)))
    mark_counts = list(base.mark_counts)
    for index in range(len(mark_counts)):
        if str(direction) == "greater_than":
            mark_counts[index] = (
                rng.randint(int(threshold_mark) + 1, int(mark_max))
                if index in target_indices
                else rng.randint(int(mark_min), int(threshold_mark))
            )
        else:
            mark_counts[index] = (
                rng.randint(int(mark_min), int(threshold_mark) - 1)
                if index in target_indices
                else rng.randint(int(threshold_mark), int(mark_max))
            )
    threshold_value = int(threshold_mark) * int(base.unit_scale)
    query = make_pictogram_query(
        selected=str(selected),
        answer=int(target_count),
        annotation_type="bbox_set",
        annotation_category_ids=tuple(f"cat_{index}" for index in sorted(target_indices)),
        params={
            "threshold_direction": str(direction),
            "threshold_mark_count": int(threshold_mark),
            "threshold_value": int(threshold_value),
            "target_count": int(target_count),
            "target_count_probabilities": support_probability_map(range(int(answer_min), int(answer_max) + 1)),
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
        prompt_task_key="threshold_count",
        prompt_query_key=str(selected),
    )


@register_task
class ChartsPictogramThresholdCountTask:
    task_id = "task_charts__pictogram__threshold_count"
    reasoning_operations = ('filtering', 'counting', 'comparison', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "threshold_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = GREATER_THAN_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_pictogram_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = ["ChartsPictogramThresholdCountTask"]
