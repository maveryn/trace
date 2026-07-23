"""Select the age group with an extremal value on one population-pyramid side."""

from __future__ import annotations

from typing import Any

from ....core.seed import spawn_rng
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import build_population_pyramid_plan, run_population_pyramid_task
from .shared.defaults import GEN_DEFAULTS, choose_from_values
from .shared.sampling import build_dataset_from_rows, sample_scene_base
from .shared.state import DOMAIN, PopulationPyramidRow


LEFT_LARGEST_QUERY_ID = "left_side_largest_value_label"
LEFT_SMALLEST_QUERY_ID = "left_side_smallest_value_label"
RIGHT_LARGEST_QUERY_ID = "right_side_largest_value_label"
RIGHT_SMALLEST_QUERY_ID = "right_side_smallest_value_label"
SIDE_VALUE_EXTREMUM_QUERY_IDS = (
    LEFT_LARGEST_QUERY_ID,
    LEFT_SMALLEST_QUERY_ID,
    RIGHT_LARGEST_QUERY_ID,
    RIGHT_SMALLEST_QUERY_ID,
)

_QUERY_SPECS = {
    LEFT_LARGEST_QUERY_ID: ("left", "largest"),
    LEFT_SMALLEST_QUERY_ID: ("left", "smallest"),
    RIGHT_LARGEST_QUERY_ID: ("right", "largest"),
    RIGHT_SMALLEST_QUERY_ID: ("right", "smallest"),
}


def _side_value(row: PopulationPyramidRow, side: str) -> int:
    if str(side) == "left":
        return int(row.left_value)
    if str(side) == "right":
        return int(row.right_value)
    raise ValueError(f"unsupported side-value side: {side}")


def _sample_rows(
    *,
    selected: str,
    side: str,
    direction: str,
    labels: tuple[str, ...],
    params: dict[str, Any],
    instance_seed: int,
) -> tuple[tuple[PopulationPyramidRow, ...], tuple[str, ...], dict[str, Any]]:
    """Construct rows with one unique side-specific value extremum."""

    rng = spawn_rng(int(instance_seed), f"charts.population_pyramid.side_value.{selected}")
    value_min = int(params.get("value_min", group_default(GEN_DEFAULTS, "value_min", 8)))
    value_max = int(params.get("value_max", group_default(GEN_DEFAULTS, "value_max", 96)))
    row_count = len(labels)
    answer_index = int(
        choose_from_values(
            params,
            values=tuple(range(row_count)),
            instance_seed=int(instance_seed),
            namespace=f"charts.population_pyramid.side_value.answer_index.{selected}",
        )
    )
    if str(direction) == "largest":
        target_value = int(rng.randint(max(value_min + 20, 78), int(value_max)))
        distractor_support = list(range(int(value_min), max(int(value_min), int(target_value) - 7)))
        extremum_phrase = "largest"
    elif str(direction) == "smallest":
        target_value = int(rng.randint(int(value_min), min(int(value_max - 20), 24)))
        distractor_support = list(range(min(int(value_max), int(target_value) + 7), int(value_max) + 1))
        extremum_phrase = "smallest"
    else:
        raise ValueError(f"unsupported side-value direction: {direction}")
    if len(distractor_support) < int(row_count) - 1:
        raise ValueError("not enough side-value support")
    distractor_values = list(rng.sample(distractor_support, int(row_count) - 1))

    rows: list[PopulationPyramidRow] = []
    for index, label in enumerate(labels):
        side_value = int(target_value if int(index) == int(answer_index) else distractor_values.pop())
        other_value = int(rng.randint(int(value_min), int(value_max)))
        left_value, right_value = (
            (side_value, other_value)
            if str(side) == "left"
            else (other_value, side_value)
        )
        rows.append(
            PopulationPyramidRow(
                row_id=f"row_{index}",
                label=str(label),
                left_value=int(left_value),
                right_value=int(right_value),
            )
        )

    values_by_row = {str(row.row_id): _side_value(row, str(side)) for row in rows}
    if str(direction) == "largest":
        observed_value = max(values_by_row.values())
    else:
        observed_value = min(values_by_row.values())
    winners = [row_id for row_id, value in values_by_row.items() if int(value) == int(observed_value)]
    if winners != [f"row_{answer_index}"]:
        raise ValueError("constructed side-value extremum is not unique")
    return tuple(rows), (f"row_{answer_index}",), {
        "side": str(side),
        "value_direction": str(direction),
        "extremum_phrase": str(extremum_phrase),
        "target_value": int(target_value),
        "target_row_label": str(labels[int(answer_index)]),
        "target_row_index": int(answer_index),
        "annotation_bar_scope": str(side),
    }


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    side, direction = _QUERY_SPECS[str(selected)]
    base = sample_scene_base(params, instance_seed=int(instance_seed))
    rows, annotation_row_ids, query_params = _sample_rows(
        selected=str(selected),
        side=str(side),
        direction=str(direction),
        labels=tuple(base.age_labels),
        params=params,
        instance_seed=int(instance_seed),
    )
    answer = str(next(row.label for row in rows if str(row.row_id) == str(annotation_row_ids[0])))
    query_params["side_series_label"] = (
        str(base.left_series_label) if str(side) == "left" else str(base.right_series_label)
    )
    dataset = build_dataset_from_rows(
        base=base,
        rows=tuple(rows),
        branch_id=str(selected),
        branch_probabilities=dict(probabilities),
        answer=str(answer),
        answer_type="string",
        annotation_type="bbox",
        annotation_row_ids=tuple(annotation_row_ids),
        params=dict(query_params),
    )
    return build_population_pyramid_plan(dataset=dataset, prompt_query_key=str(selected))


@register_task
class ChartsPopulationPyramidSideValueExtremumLabelTask:
    task_id = "task_charts__population_pyramid__side_value_extremum_label"
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "side_value_extremum_label"
    supported_query_ids = SIDE_VALUE_EXTREMUM_QUERY_IDS
    default_query_id = LEFT_LARGEST_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_population_pyramid_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsPopulationPyramidSideValueExtremumLabelTask",
    "SIDE_VALUE_EXTREMUM_QUERY_IDS",
]
