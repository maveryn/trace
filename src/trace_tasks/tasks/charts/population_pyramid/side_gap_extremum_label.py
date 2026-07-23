"""Select the age group with an extremal left-right population-pyramid gap."""

from __future__ import annotations

from typing import Any

from ....core.seed import spawn_rng
from ...registry import register_task
from ...shared.config_defaults import group_default
from ._lifecycle import build_population_pyramid_plan, run_population_pyramid_task
from .shared.defaults import GEN_DEFAULTS, choose_from_values
from .shared.sampling import build_dataset_from_rows, sample_pair_for_gap, sample_scene_base
from .shared.state import DOMAIN, PopulationPyramidRow


LARGEST_GAP_QUERY_ID = "largest_side_gap_label"
SMALLEST_NONZERO_GAP_QUERY_ID = "smallest_nonzero_side_gap_label"
GAP_QUERY_IDS = (LARGEST_GAP_QUERY_ID, SMALLEST_NONZERO_GAP_QUERY_ID)


def _sample_rows(
    *,
    selected: str,
    labels: tuple[str, ...],
    params: dict[str, Any],
    instance_seed: int,
) -> tuple[tuple[PopulationPyramidRow, ...], tuple[str, ...], dict[str, Any]]:
    """Construct one unique extremal absolute side-gap row for the selected rank."""

    rng = spawn_rng(int(instance_seed), f"charts.population_pyramid.gap.{selected}")
    value_min = int(params.get("value_min", group_default(GEN_DEFAULTS, "value_min", 8)))
    value_max = int(params.get("value_max", group_default(GEN_DEFAULTS, "value_max", 96)))
    gap_min = int(params.get("gap_min", group_default(GEN_DEFAULTS, "gap_min", 4)))
    gap_max = int(params.get("gap_max", group_default(GEN_DEFAULTS, "gap_max", 58)))
    row_count = len(labels)
    answer_index = int(
        choose_from_values(
            params,
            values=tuple(range(row_count)),
            instance_seed=int(instance_seed),
            namespace=f"charts.population_pyramid.gap.answer_index.{selected}",
        )
    )
    if str(selected) == LARGEST_GAP_QUERY_ID:
        target_gap = int(rng.randint(max(32, gap_min + 8), int(gap_max)))
        other_support = [gap for gap in range(int(gap_min), max(int(gap_min), int(target_gap) - 7))]
        rank_phrase = "largest"
    else:
        target_gap = int(rng.randint(int(gap_min), min(int(gap_min) + 4, int(gap_max))))
        other_support = [gap for gap in range(int(target_gap) + 7, int(gap_max) + 1)]
        rank_phrase = "smallest nonzero"
    if len(other_support) < row_count - 1:
        raise ValueError("not enough gap support")
    other_gaps = list(rng.sample(other_support, int(row_count) - 1))
    rows: list[PopulationPyramidRow] = []
    for index, label in enumerate(labels):
        gap = int(target_gap if int(index) == int(answer_index) else other_gaps.pop())
        direction = 1 if bool(rng.randint(0, 1)) else -1
        left, right = sample_pair_for_gap(
            rng,
            gap=int(gap),
            value_min=int(value_min),
            value_max=int(value_max),
            direction=int(direction),
        )
        rows.append(PopulationPyramidRow(row_id=f"row_{index}", label=str(label), left_value=int(left), right_value=int(right)))
    annotation_row_id = f"row_{answer_index}"
    return tuple(rows), (annotation_row_id,), {
        "rank_phrase": str(rank_phrase),
        "target_gap": int(target_gap),
        "target_row_label": str(labels[int(answer_index)]),
        "target_row_index": int(answer_index),
    }


def _build_plan(params: dict[str, Any], instance_seed: int, selected: str, probabilities: dict[str, float]):
    base = sample_scene_base(params, instance_seed=int(instance_seed))
    rows, annotation_row_ids, query_params = _sample_rows(
        selected=str(selected),
        labels=tuple(base.age_labels),
        params=params,
        instance_seed=int(instance_seed),
    )
    answer = str(next(row.label for row in rows if str(row.row_id) == str(annotation_row_ids[0])))
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
class ChartsPopulationPyramidSideGapExtremumLabelTask:
    task_id = "task_charts__population_pyramid__side_gap_extremum_label"
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "side_gap_extremum_label"
    supported_query_ids = GAP_QUERY_IDS
    default_query_id = LARGEST_GAP_QUERY_ID
    default_dataset_enabled = True
    _build_plan = staticmethod(_build_plan)

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int):
        return run_population_pyramid_task(self, int(instance_seed), dict(params), int(max_attempts))


__all__ = [
    "ChartsPopulationPyramidSideGapExtremumLabelTask",
    "GAP_QUERY_IDS",
]
