"""Count population-pyramid age groups satisfying a threshold predicate."""

from __future__ import annotations

from typing import Any

from ....core.seed import spawn_rng
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import build_population_pyramid_plan, run_population_pyramid_task
from .shared.defaults import GEN_DEFAULTS, choose_from_values, support_probability_map
from .shared.sampling import build_dataset_from_rows, sample_pair_for_total, sample_scene_base, threshold_metric
from .shared.state import DOMAIN, PopulationPyramidRow


LEFT_AT_LEAST_QUERY_ID = "left_side_at_least_threshold_count"
LEFT_AT_MOST_QUERY_ID = "left_side_at_most_threshold_count"
RIGHT_AT_LEAST_QUERY_ID = "right_side_at_least_threshold_count"
RIGHT_AT_MOST_QUERY_ID = "right_side_at_most_threshold_count"
COMBINED_AT_LEAST_QUERY_ID = "combined_total_at_least_threshold_count"
COMBINED_AT_MOST_QUERY_ID = "combined_total_at_most_threshold_count"
THRESHOLD_QUERY_IDS = (
    LEFT_AT_LEAST_QUERY_ID,
    LEFT_AT_MOST_QUERY_ID,
    RIGHT_AT_LEAST_QUERY_ID,
    RIGHT_AT_MOST_QUERY_ID,
    COMBINED_AT_LEAST_QUERY_ID,
    COMBINED_AT_MOST_QUERY_ID,
)

_QUERY_SPECS = {
    LEFT_AT_LEAST_QUERY_ID: ("left", "at_least"),
    LEFT_AT_MOST_QUERY_ID: ("left", "at_most"),
    RIGHT_AT_LEAST_QUERY_ID: ("right", "at_least"),
    RIGHT_AT_MOST_QUERY_ID: ("right", "at_most"),
    COMBINED_AT_LEAST_QUERY_ID: ("combined_total", "at_least"),
    COMBINED_AT_MOST_QUERY_ID: ("combined_total", "at_most"),
}


def _threshold_support(params: dict[str, Any], side: str) -> tuple[int, ...]:
    if str(side) == "combined_total":
        return tuple(
            range(
                int(params.get("combined_threshold_min", group_default(GEN_DEFAULTS, "combined_threshold_min", 70))),
                int(params.get("combined_threshold_max", group_default(GEN_DEFAULTS, "combined_threshold_max", 150))) + 1,
                int(params.get("combined_threshold_step", group_default(GEN_DEFAULTS, "combined_threshold_step", 10))),
            )
        )
    return tuple(
        range(
            int(params.get("side_threshold_min", group_default(GEN_DEFAULTS, "side_threshold_min", 28))),
            int(params.get("side_threshold_max", group_default(GEN_DEFAULTS, "side_threshold_max", 78))) + 1,
            int(params.get("side_threshold_step", group_default(GEN_DEFAULTS, "side_threshold_step", 5))),
        )
    )


def _metric_phrase(side: str, *, left_label: str, right_label: str) -> str:
    if str(side) == "left":
        return f'the "{left_label}" value'
    if str(side) == "right":
        return f'the "{right_label}" value'
    return f'the sum of "{left_label}" and "{right_label}"'


def _sample_rows(
    *,
    params: dict[str, Any],
    labels: tuple[str, ...],
    side: str,
    relation: str,
    instance_seed: int,
) -> tuple[tuple[PopulationPyramidRow, ...], tuple[str, ...], dict[str, Any]]:
    """Construct rows so the selected threshold predicate has a known count."""

    rng = spawn_rng(int(instance_seed), f"charts.population_pyramid.threshold.{side}.{relation}")
    row_count = len(labels)
    value_min = int(params.get("value_min", group_default(GEN_DEFAULTS, "value_min", 8)))
    value_max = int(params.get("value_max", group_default(GEN_DEFAULTS, "value_max", 96)))
    count_min = int(params.get("threshold_count_min", group_default(GEN_DEFAULTS, "threshold_count_min", 2)))
    count_max = min(
        int(params.get("threshold_count_max", group_default(GEN_DEFAULTS, "threshold_count_max", 9))),
        int(row_count) - 1,
    )
    count_support = tuple(range(max(1, int(count_min)), max(int(count_min), int(count_max)) + 1))
    target_count = int(
        choose_from_values(
            params,
            values=count_support,
            instance_seed=int(instance_seed),
            namespace=f"charts.population_pyramid.threshold.answer_count.{side}.{relation}",
        )
    )
    target_indices = tuple(sorted(rng.sample(list(range(int(row_count))), k=int(target_count))))
    target_index_set = set(int(index) for index in target_indices)
    threshold_support = _threshold_support(params, str(side))
    threshold = int(
        choose_from_values(
            params,
            values=threshold_support,
            instance_seed=int(instance_seed),
            namespace=f"charts.population_pyramid.threshold.value.{side}.{relation}",
        )
    )

    rows: list[PopulationPyramidRow] = []
    for index, label in enumerate(labels):
        is_target = int(index) in target_index_set
        if str(side) == "combined_total":
            if str(relation) == "at_least":
                total_min, total_max = (threshold, min(2 * value_max, threshold + 42)) if is_target else (2 * value_min, threshold - 1)
            else:
                total_min, total_max = (2 * value_min, threshold) if is_target else (threshold + 1, min(2 * value_max, threshold + 42))
            left, right = sample_pair_for_total(
                rng,
                total_min=int(total_min),
                total_max=int(total_max),
                value_min=int(value_min),
                value_max=int(value_max),
            )
        else:
            if str(relation) == "at_least":
                metric_low, metric_high = (threshold, value_max) if is_target else (value_min, threshold - 1)
            else:
                metric_low, metric_high = (value_min, threshold) if is_target else (threshold + 1, value_max)
            if int(metric_low) > int(metric_high):
                raise ValueError("empty side metric support")
            metric_value = int(rng.randint(int(metric_low), int(metric_high)))
            other_value = int(rng.randint(int(value_min), int(value_max)))
            left, right = (int(metric_value), int(other_value)) if str(side) == "left" else (int(other_value), int(metric_value))
        rows.append(PopulationPyramidRow(row_id=f"row_{index}", label=str(label), left_value=int(left), right_value=int(right)))

    annotation_row_ids = tuple(f"row_{index}" for index in target_indices)
    metric_values = [threshold_metric(row, str(side)) for row in rows]
    if str(relation) == "at_least":
        observed = tuple(row.row_id for row, value in zip(rows, metric_values, strict=True) if int(value) >= int(threshold))
        relation_phrase = "at least"
    else:
        observed = tuple(row.row_id for row, value in zip(rows, metric_values, strict=True) if int(value) <= int(threshold))
        relation_phrase = "at most"
    if tuple(observed) != tuple(annotation_row_ids):
        raise ValueError("constructed threshold support does not match target rows")
    return tuple(rows), tuple(annotation_row_ids), {
        "threshold_side": str(side),
        "threshold_relation": str(relation),
        "threshold_relation_phrase": str(relation_phrase),
        "threshold_value": int(threshold),
        "threshold_value_probabilities": support_probability_map(threshold_support),
        "target_count": int(target_count),
        "target_count_probabilities": support_probability_map(count_support),
        "target_row_labels": [str(labels[index]) for index in target_indices],
    }


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    side, relation = _QUERY_SPECS[str(selected)]
    base = sample_scene_base(params, instance_seed=int(instance_seed))
    rows, annotation_row_ids, query_params = _sample_rows(
        params=params,
        labels=tuple(base.age_labels),
        side=str(side),
        relation=str(relation),
        instance_seed=int(instance_seed),
    )
    query_params["metric_phrase"] = _metric_phrase(
        str(side),
        left_label=str(base.left_series_label),
        right_label=str(base.right_series_label),
    )
    dataset = build_dataset_from_rows(
        base=base,
        rows=tuple(rows),
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        answer=int(len(annotation_row_ids)),
        answer_type="integer",
        annotation_type="bbox_set",
        annotation_row_ids=tuple(annotation_row_ids),
        params=dict(query_params),
    )
    return build_population_pyramid_plan(dataset=dataset, prompt_query_key=str(selected))


@register_task
class ChartsPopulationPyramidAgeGroupThresholdCountTask:
    task_id = "task_charts__population_pyramid__age_group_threshold_count"
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "age_group_threshold_count"
    supported_query_ids = THRESHOLD_QUERY_IDS
    default_query_id = LEFT_AT_LEAST_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_population_pyramid_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsPopulationPyramidAgeGroupThresholdCountTask",
    "THRESHOLD_QUERY_IDS",
]
