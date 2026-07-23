"""Count population-pyramid rows where one side is larger than the other."""

from __future__ import annotations

from typing import Any

from ....core.seed import spawn_rng
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import build_population_pyramid_plan, run_population_pyramid_task
from .shared.defaults import GEN_DEFAULTS, choose_from_values, support_probability_map
from .shared.sampling import build_dataset_from_rows, sample_pair_for_gap, sample_scene_base
from .shared.state import DOMAIN, PopulationPyramidRow


LEFT_GREATER_QUERY_ID = "left_side_greater_count"
RIGHT_GREATER_QUERY_ID = "right_side_greater_count"
DOMINANT_SIDE_COUNT_QUERY_IDS = (LEFT_GREATER_QUERY_ID, RIGHT_GREATER_QUERY_ID)

_SIDE_BY_QUERY = {
    LEFT_GREATER_QUERY_ID: "left",
    RIGHT_GREATER_QUERY_ID: "right",
}


def _sample_rows(
    *,
    selected: str,
    dominant_side: str,
    labels: tuple[str, ...],
    params: dict[str, Any],
    instance_seed: int,
) -> tuple[tuple[PopulationPyramidRow, ...], tuple[str, ...], dict[str, Any]]:
    """Construct rows with a known count of left-dominant or right-dominant pairs."""

    rng = spawn_rng(int(instance_seed), f"charts.population_pyramid.dominant_side.{selected}")
    row_count = len(labels)
    value_min = int(params.get("value_min", group_default(GEN_DEFAULTS, "value_min", 8)))
    value_max = int(params.get("value_max", group_default(GEN_DEFAULTS, "value_max", 96)))
    gap_min = int(params.get("gap_min", group_default(GEN_DEFAULTS, "gap_min", 4)))
    gap_max = int(params.get("gap_max", group_default(GEN_DEFAULTS, "gap_max", 58)))
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
            namespace=f"charts.population_pyramid.dominant_side.answer_count.{selected}",
        )
    )
    target_indices = tuple(sorted(rng.sample(list(range(int(row_count))), k=int(target_count))))
    target_index_set = set(int(index) for index in target_indices)

    rows: list[PopulationPyramidRow] = []
    for index, label in enumerate(labels):
        is_target = int(index) in target_index_set
        if str(dominant_side) == "left":
            direction = 1 if is_target else -1
        elif str(dominant_side) == "right":
            direction = -1 if is_target else 1
        else:
            raise ValueError(f"unsupported dominant side: {dominant_side}")
        left_value, right_value = sample_pair_for_gap(
            rng,
            gap=int(rng.randint(int(gap_min), int(gap_max))),
            value_min=int(value_min),
            value_max=int(value_max),
            direction=int(direction),
        )
        rows.append(
            PopulationPyramidRow(
                row_id=f"row_{index}",
                label=str(label),
                left_value=int(left_value),
                right_value=int(right_value),
            )
        )

    if str(dominant_side) == "left":
        observed = tuple(row.row_id for row in rows if int(row.left_value) > int(row.right_value))
    else:
        observed = tuple(row.row_id for row in rows if int(row.right_value) > int(row.left_value))
    annotation_row_ids = tuple(f"row_{index}" for index in target_indices)
    if tuple(observed) != tuple(annotation_row_ids):
        raise ValueError("constructed dominant-side count does not match target rows")
    return tuple(rows), tuple(annotation_row_ids), {
        "dominant_side": str(dominant_side),
        "target_count": int(target_count),
        "target_count_probabilities": support_probability_map(count_support),
        "target_row_labels": [str(labels[index]) for index in target_indices],
        "annotation_bar_scope": "row",
    }


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    dominant_side = str(_SIDE_BY_QUERY[str(selected)])
    base = sample_scene_base(params, instance_seed=int(instance_seed))
    rows, annotation_row_ids, query_params = _sample_rows(
        selected=str(selected),
        dominant_side=str(dominant_side),
        labels=tuple(base.age_labels),
        params=params,
        instance_seed=int(instance_seed),
    )
    query_params["side_series_label"] = (
        str(base.left_series_label) if str(dominant_side) == "left" else str(base.right_series_label)
    )
    query_params["other_series_label"] = (
        str(base.right_series_label) if str(dominant_side) == "left" else str(base.left_series_label)
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
class ChartsPopulationPyramidDominantSideCountTask:
    task_id = "task_charts__population_pyramid__dominant_side_count"
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "dominant_side_count"
    supported_query_ids = DOMINANT_SIDE_COUNT_QUERY_IDS
    default_query_id = LEFT_GREATER_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_population_pyramid_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsPopulationPyramidDominantSideCountTask",
    "DOMINANT_SIDE_COUNT_QUERY_IDS",
]
