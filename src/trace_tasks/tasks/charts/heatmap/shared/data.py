"""Scene sample construction for heatmap chart tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.balanced_sampling import balanced_int_from_support as _balanced_int
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.shared.unanswerable import (
    UNANSWERABLE_ANSWER,
    absence_proof,
    choose_missing_label,
    should_use_unanswerable_branch,
)
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds

from .defaults import (
    GEN_DEFAULTS,
    SCENE_NAMESPACE,
    _MISSING_CONDITION_PHRASES,
    _TITLE_OPTIONS,
    _condition_matches,
    _condition_phrase,
    _extremum_phrase,
)
from .sampling import (
    _colorbar_interval_bounds,
    _colorbar_threshold_values,
    _colorbar_ticks,
    _labels_for_scene,
    _resolve_row_column_count,
)


Cell = Dict[str, Any]
CandidateBuilder = Callable[..., Dict[str, Any] | None]


def _longest_run(mask: Sequence[bool]) -> Tuple[int, int, int]:
    """Return length and inclusive bounds for the longest true run."""

    best_start = -1
    best_end = -1
    best_len = 0
    start = -1
    current = 0
    for index, active in enumerate(mask):
        if bool(active):
            if current == 0:
                start = int(index)
            current += 1
            if int(current) > int(best_len):
                best_len = int(current)
                best_start = int(start)
                best_end = int(index)
        else:
            current = 0
            start = -1
    return int(best_len), int(best_start), int(best_end)


def _make_cells(
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: Sequence[Sequence[int]],
    bin_count: int,
) -> List[Cell]:
    cells: List[Cell] = []
    for row_index, row_label in enumerate(row_labels):
        for column_index, column_label in enumerate(column_labels):
            value = int(values[int(row_index)][int(column_index)])
            cells.append(
                {
                    "cell_id": f"cell_r{int(row_index)}_c{int(column_index)}",
                    "row_index": int(row_index),
                    "column_index": int(column_index),
                    "row_label": str(row_label),
                    "column_label": str(column_label),
                    "heat_level": int(value),
                    "numeric_value": int(value),
                    "heat_fraction": round(float(value) / float(max(1, int(bin_count) - 1)), 4),
                    "is_hot": bool(_condition_matches(value, condition_kind="hot", bin_count=int(bin_count))),
                    "is_cool": bool(_condition_matches(value, condition_kind="cool", bin_count=int(bin_count))),
                    "is_increase": bool(_condition_matches(value, condition_kind="increase", bin_count=int(bin_count))),
                    "is_decrease": bool(_condition_matches(value, condition_kind="decrease", bin_count=int(bin_count))),
                }
            )
    return cells


def _cells_by_position(cells: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, int], Cell]:
    return {
        (int(cell["row_index"]), int(cell["column_index"])): dict(cell)
        for cell in cells
    }


def _reading_order_cell_ids(cells: Sequence[Mapping[str, Any]]) -> List[str]:
    ordered = sorted(cells, key=lambda item: (int(item["row_index"]), int(item["column_index"])))
    return [str(item["cell_id"]) for item in ordered]


def _target_count_support(
    params: Mapping[str, Any],
    *,
    prefix: str,
    total_cells: int,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, ...]:
    low, high = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key=f"{prefix}_answer_min",
        max_key=f"{prefix}_answer_max",
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=f"{SCENE_NAMESPACE} {prefix} answer support",
    )
    upper = min(int(high), max(1, int(total_cells) - 1))
    lower = min(max(1, int(low)), int(upper))
    return tuple(range(int(lower), int(upper) + 1))


def _bounded_randint(rng: Any, low: int, high: int) -> int:
    return int(rng.randint(max(0, int(low)), min(100, int(high))))


def _base_dataset(
    *,
    scene_title: str,
    prompt_key: str,
    scene_variant: str,
    query_axis: str,
    condition_kind: str,
    extremum_direction: str,
    row_count: int,
    column_count: int,
    row_count_probabilities: Mapping[str, float],
    column_count_probabilities: Mapping[str, float],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    heat_bin_count: int,
    values: Sequence[Sequence[int]],
    cells: Sequence[Mapping[str, Any]],
    answer_value: int | str,
    answer_type: str,
    answer_row_index: int,
    answer_column_index: int,
    annotation_cell_ids: Sequence[str],
    question_params: Mapping[str, Any],
    colorbar_value_min: int | None = None,
    colorbar_value_max: int | None = None,
    colorbar_ticks: Sequence[int] = (),
    is_unanswerable: bool = False,
    absence_proof_payload: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalize one constructed heatmap sample into the scene trace data shape."""

    return {
        "scene_title": str(scene_title),
        "prompt_key": str(prompt_key),
        "scene_variant": str(scene_variant),
        "query_axis": str(query_axis),
        "condition_kind": str(condition_kind),
        "extremum_direction": str(extremum_direction),
        "row_count": int(row_count),
        "column_count": int(column_count),
        "row_count_probabilities": dict(row_count_probabilities),
        "column_count_probabilities": dict(column_count_probabilities),
        "row_labels": [str(label) for label in row_labels],
        "column_labels": [str(label) for label in column_labels],
        "heat_bin_count": int(heat_bin_count),
        "colorbar_value_min": colorbar_value_min,
        "colorbar_value_max": colorbar_value_max,
        "colorbar_ticks": [int(value) for value in colorbar_ticks],
        "values": [[int(value) for value in row] for row in values],
        "cells": [dict(cell) for cell in cells],
        "cells_by_id": {str(cell["cell_id"]): dict(cell) for cell in cells},
        "answer_value": int(answer_value) if str(answer_type) == "integer" else str(answer_value),
        "answer_type": str(answer_type),
        "answer_row_index": int(answer_row_index),
        "answer_column_index": int(answer_column_index),
        "annotation_cell_ids": [str(cell_id) for cell_id in annotation_cell_ids],
        "question_params": dict(question_params),
        "is_unanswerable": bool(is_unanswerable),
        "absence_proof": dict(absence_proof_payload or {}),
    }


def construct_colorbar_threshold_sample(
    *,
    prompt_key: str,
    relation: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Build a continuous-colorbar heatmap with a controlled threshold count."""

    if str(relation) not in {"above", "below"}:
        raise ValueError(f"unsupported colorbar relation: {relation}")
    row_count, column_count, row_probabilities, column_probabilities = _resolve_row_column_count(
        params,
        scene_variant="continuous_colorbar_heatmap",
        instance_seed=int(instance_seed),
    )
    total_cells = int(row_count) * int(column_count)
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.continuous_threshold")
    row_labels, column_labels = _labels_for_scene(
        scene_variant="continuous_colorbar_heatmap",
        row_count=int(row_count),
        column_count=int(column_count),
        rng=rng,
    )
    positions = [(row, column) for row in range(int(row_count)) for column in range(int(column_count))]
    rng.shuffle(positions)
    target_support = _target_count_support(
        params,
        prefix="colorbar_threshold",
        total_cells=int(total_cells),
        fallback_min=1,
        fallback_max=16,
    )
    target_count = _balanced_int(
        target_support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{relation}.target_count",
    )
    threshold = _balanced_int(
        _colorbar_threshold_values(params),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.{relation}.threshold",
    )
    gap = max(4, int(params.get("colorbar_value_margin", group_default(GEN_DEFAULTS, "colorbar_value_margin", 6))))
    selected = set(positions[: int(target_count)])
    values: List[List[int]] = [[0 for _ in range(int(column_count))] for _ in range(int(row_count))]
    for row, column in positions:
        if str(relation) == "above":
            value = (
                _bounded_randint(rng, int(threshold) + int(gap), 100)
                if (row, column) in selected
                else _bounded_randint(rng, 0, int(threshold) - int(gap))
            )
        else:
            value = (
                _bounded_randint(rng, 0, int(threshold) - int(gap))
                if (row, column) in selected
                else _bounded_randint(rng, int(threshold) + int(gap), 100)
            )
        values[int(row)][int(column)] = int(value)
    cells = _make_cells(row_labels=row_labels, column_labels=column_labels, values=values, bin_count=101)
    by_pos = _cells_by_position(cells)
    annotation_cells = [by_pos[(row, column)] for row, column in sorted(selected)]
    return _base_dataset(
        scene_title=str(_TITLE_OPTIONS[int(rng.randrange(len(_TITLE_OPTIONS)))]),
        prompt_key=str(prompt_key),
        scene_variant="continuous_colorbar_heatmap",
        query_axis="",
        condition_kind="",
        extremum_direction="",
        row_count=int(row_count),
        column_count=int(column_count),
        row_count_probabilities=row_probabilities,
        column_count_probabilities=column_probabilities,
        row_labels=row_labels,
        column_labels=column_labels,
        heat_bin_count=101,
        colorbar_value_min=0,
        colorbar_value_max=100,
        colorbar_ticks=_colorbar_ticks(params),
        values=values,
        cells=cells,
        answer_value=int(target_count),
        answer_type="integer",
        answer_row_index=-1,
        answer_column_index=-1,
        annotation_cell_ids=_reading_order_cell_ids(annotation_cells),
        question_params={
            "colorbar_predicate": str(relation),
            "threshold_value": int(threshold),
            "condition_phrase": f"{str(relation)} {int(threshold)}",
            "answer_support": [int(value) for value in target_support],
        },
    )


def construct_colorbar_interval_sample(
    *,
    prompt_key: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Build a continuous-colorbar heatmap with a controlled interval count."""

    row_count, column_count, row_probabilities, column_probabilities = _resolve_row_column_count(
        params,
        scene_variant="continuous_colorbar_heatmap",
        instance_seed=int(instance_seed),
    )
    total_cells = int(row_count) * int(column_count)
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.continuous_interval")
    row_labels, column_labels = _labels_for_scene(
        scene_variant="continuous_colorbar_heatmap",
        row_count=int(row_count),
        column_count=int(column_count),
        rng=rng,
    )
    positions = [(row, column) for row in range(int(row_count)) for column in range(int(column_count))]
    rng.shuffle(positions)
    target_support = _target_count_support(
        params,
        prefix="colorbar_interval",
        total_cells=int(total_cells),
        fallback_min=1,
        fallback_max=14,
    )
    target_count = _balanced_int(
        target_support,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.interval.target_count",
    )
    bounds = _colorbar_interval_bounds(params)
    lower, upper = uniform_choice(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.interval.bounds"),
        bounds,
        sort_keys=True,
    )
    gap = max(4, int(params.get("colorbar_value_margin", group_default(GEN_DEFAULTS, "colorbar_value_margin", 6))))
    selected = set(positions[: int(target_count)])
    values: List[List[int]] = [[0 for _ in range(int(column_count))] for _ in range(int(row_count))]
    for row, column in positions:
        if (row, column) in selected:
            value = _bounded_randint(rng, int(lower) + int(gap), int(upper) - int(gap))
        else:
            below_ok = int(lower) - int(gap) >= 0
            above_ok = int(upper) + int(gap) <= 100
            if bool(below_ok) and (not bool(above_ok) or (int(row) + int(column)) % 2 == 0):
                value = _bounded_randint(rng, 0, int(lower) - int(gap))
            elif bool(above_ok):
                value = _bounded_randint(rng, int(upper) + int(gap), 100)
            else:
                raise ValueError(f"invalid colorbar interval bounds: {lower}, {upper}")
        values[int(row)][int(column)] = int(value)
    cells = _make_cells(row_labels=row_labels, column_labels=column_labels, values=values, bin_count=101)
    by_pos = _cells_by_position(cells)
    annotation_cells = [by_pos[(row, column)] for row, column in sorted(selected)]
    return _base_dataset(
        scene_title=str(_TITLE_OPTIONS[int(rng.randrange(len(_TITLE_OPTIONS)))]),
        prompt_key=str(prompt_key),
        scene_variant="continuous_colorbar_heatmap",
        query_axis="",
        condition_kind="",
        extremum_direction="",
        row_count=int(row_count),
        column_count=int(column_count),
        row_count_probabilities=row_probabilities,
        column_count_probabilities=column_probabilities,
        row_labels=row_labels,
        column_labels=column_labels,
        heat_bin_count=101,
        colorbar_value_min=0,
        colorbar_value_max=100,
        colorbar_ticks=_colorbar_ticks(params),
        values=values,
        cells=cells,
        answer_value=int(target_count),
        answer_type="integer",
        answer_row_index=-1,
        answer_column_index=-1,
        annotation_cell_ids=_reading_order_cell_ids(annotation_cells),
        question_params={
            "lower_bound": int(lower),
            "upper_bound": int(upper),
            "condition_phrase": f"from {int(lower)} to {int(upper)}, inclusive",
            "answer_support": [int(value) for value in target_support],
        },
    )


def _axis_condition_candidate(
    *,
    scene_variant: str,
    query_axis: str,
    condition_kind: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: Sequence[Sequence[int]],
    cells: Sequence[Mapping[str, Any]],
    bin_count: int,
) -> Dict[str, Any] | None:
    """Find the unique row or column with the most cells matching one condition."""

    by_pos = _cells_by_position(cells)
    row_count = len(row_labels)
    column_count = len(column_labels)
    if str(query_axis) == "row":
        counts = [
            sum(
                1
                for column_index in range(int(column_count))
                if _condition_matches(
                    int(values[int(row_index)][int(column_index)]),
                    condition_kind=str(condition_kind),
                    bin_count=int(bin_count),
                )
            )
            for row_index in range(int(row_count))
        ]
        label_pool = list(row_labels)
    elif str(query_axis) == "column":
        counts = [
            sum(
                1
                for row_index in range(int(row_count))
                if _condition_matches(
                    int(values[int(row_index)][int(column_index)]),
                    condition_kind=str(condition_kind),
                    bin_count=int(bin_count),
                )
            )
            for column_index in range(int(column_count))
        ]
        label_pool = list(column_labels)
    else:
        raise ValueError(f"unsupported query_axis: {query_axis}")
    maximum = max(counts)
    winners = [index for index, count in enumerate(counts) if int(count) == int(maximum)]
    if len(winners) != 1 or int(maximum) < 1:
        return None
    axis_index = int(winners[0])
    if str(query_axis) == "row":
        row_index = int(axis_index)
        column_index = -1
        annotation_cells = [
            by_pos[(row_index, col)]
            for col in range(int(column_count))
            if _condition_matches(
                int(values[row_index][int(col)]),
                condition_kind=str(condition_kind),
                bin_count=int(bin_count),
            )
        ]
        condition_counts_key = "row_condition_counts"
    else:
        row_index = -1
        column_index = int(axis_index)
        annotation_cells = [
            by_pos[(row, column_index)]
            for row in range(int(row_count))
            if _condition_matches(
                int(values[int(row)][column_index]),
                condition_kind=str(condition_kind),
                bin_count=int(bin_count),
            )
        ]
        condition_counts_key = "column_condition_counts"
    return {
        "answer_value": str(label_pool[axis_index]),
        "answer_type": "string",
        "answer_row_index": int(row_index),
        "answer_column_index": int(column_index),
        "annotation_cell_ids": _reading_order_cell_ids(annotation_cells),
        "question_params": {
            "query_axis": str(query_axis),
            "answer_axis": str(query_axis),
            "condition_kind": str(condition_kind),
            "condition_phrase": _condition_phrase(str(condition_kind), scene_variant=str(scene_variant)),
            condition_counts_key: {str(label_pool[index]): int(counts[index]) for index in range(len(label_pool))},
        },
    }


def _axis_cell_candidate(
    *,
    scene_variant: str,
    query_axis: str,
    extremum_direction: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: Sequence[Sequence[int]],
    cells: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any] | None:
    """Choose a named row or column whose requested extremal cell has a unique label."""

    by_pos = _cells_by_position(cells)
    row_count = len(row_labels)
    column_count = len(column_labels)
    if str(query_axis) == "column":
        feasible_columns: List[int] = []
        for column_index in range(int(column_count)):
            column_values = [int(values[row_index][column_index]) for row_index in range(int(row_count))]
            target = max(column_values) if str(extremum_direction) == "hottest" else min(column_values)
            if sum(1 for value in column_values if int(value) == int(target)) == 1:
                feasible_columns.append(int(column_index))
        if not feasible_columns:
            return None
        column_index = _balanced_int(
            feasible_columns,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.selected_column",
        )
        column_values = [int(values[row_index][int(column_index)]) for row_index in range(int(row_count))]
        target = max(column_values) if str(extremum_direction) == "hottest" else min(column_values)
        row_index = int(column_values.index(int(target)))
        return {
            "answer_value": str(row_labels[row_index]),
            "answer_type": "string",
            "answer_row_index": int(row_index),
            "answer_column_index": int(column_index),
            "annotation_cell_ids": [str(by_pos[(int(row_index), int(column_index))]["cell_id"])],
            "question_params": {
                "query_axis": str(query_axis),
                "answer_axis": "row",
                "axis_label": str(column_labels[column_index]),
                "column_label": str(column_labels[column_index]),
                "selected_column_index": int(column_index),
                "extremum_direction": str(extremum_direction),
                "extremum_phrase": _extremum_phrase(str(extremum_direction), scene_variant=str(scene_variant)),
            },
        }

    if str(query_axis) == "row":
        feasible_rows: List[int] = []
        for row_index in range(int(row_count)):
            row_values = [int(values[row_index][column_index]) for column_index in range(int(column_count))]
            target = max(row_values) if str(extremum_direction) == "hottest" else min(row_values)
            if sum(1 for value in row_values if int(value) == int(target)) == 1:
                feasible_rows.append(int(row_index))
        if not feasible_rows:
            return None
        row_index = _balanced_int(
            feasible_rows,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.selected_row",
        )
        row_values = [int(values[int(row_index)][column_index]) for column_index in range(int(column_count))]
        target = max(row_values) if str(extremum_direction) == "hottest" else min(row_values)
        column_index = int(row_values.index(int(target)))
        return {
            "answer_value": str(column_labels[column_index]),
            "answer_type": "string",
            "answer_row_index": int(row_index),
            "answer_column_index": int(column_index),
            "annotation_cell_ids": [str(by_pos[(int(row_index), int(column_index))]["cell_id"])],
            "question_params": {
                "query_axis": str(query_axis),
                "answer_axis": "column",
                "axis_label": str(row_labels[row_index]),
                "row_label": str(row_labels[row_index]),
                "selected_row_index": int(row_index),
                "extremum_direction": str(extremum_direction),
                "extremum_phrase": _extremum_phrase(str(extremum_direction), scene_variant=str(scene_variant)),
            },
        }
    raise ValueError(f"unsupported query_axis: {query_axis}")


def _condition_run_candidate(
    *,
    scene_variant: str,
    condition_kind: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    values: Sequence[Sequence[int]],
    cells: Sequence[Mapping[str, Any]],
    bin_count: int,
) -> Dict[str, Any] | None:
    """Find the unique row with the longest left-to-right run of matching cells."""

    by_pos = _cells_by_position(cells)
    row_count = len(row_labels)
    column_count = len(column_labels)
    run_specs: List[Tuple[int, int, int]] = []
    for row_index in range(int(row_count)):
        mask = [
            _condition_matches(
                int(values[row_index][column_index]),
                condition_kind=str(condition_kind),
                bin_count=int(bin_count),
            )
            for column_index in range(int(column_count))
        ]
        run_specs.append(_longest_run(mask))
    longest = max(run_len for run_len, _, _ in run_specs)
    winners = [index for index, (run_len, _, _) in enumerate(run_specs) if int(run_len) == int(longest)]
    if len(winners) != 1 or int(longest) < 2:
        return None
    row_index = int(winners[0])
    _, start, end = run_specs[row_index]
    annotation_cells = [by_pos[(row_index, column_index)] for column_index in range(int(start), int(end) + 1)]
    return {
        "answer_value": str(row_labels[row_index]),
        "answer_type": "string",
        "answer_row_index": int(row_index),
        "answer_column_index": -1,
        "annotation_cell_ids": _reading_order_cell_ids(annotation_cells),
        "question_params": {
            "query_axis": "row",
            "answer_axis": "row",
            "condition_kind": str(condition_kind),
            "condition_phrase": _condition_phrase(str(condition_kind), scene_variant=str(scene_variant)),
            "longest_run_length": int(longest),
            "row_run_lengths": {str(row_labels[index]): int(run_specs[index][0]) for index in range(int(row_count))},
        },
    }


def _missing_axis_cell_candidate(
    *,
    query_axis: str,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    scene_variant: str,
    extremum_direction: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Create a controlled-unanswerable axis-cell request for a hidden axis label."""

    if str(query_axis) == "row":
        missing_label = choose_missing_label(
            visible_labels=row_labels,
            candidate_labels=resolve_chart_entity_labels(
                spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.missing_row_candidates"),
                count=max(16, len(row_labels) + 8),
                min_chars=2,
                max_chars=7,
                allow_spaces=False,
            ).labels,
            fallback_prefix="Row ",
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.missing_row",
        )
        return {
            "answer_value": UNANSWERABLE_ANSWER,
            "answer_type": "string",
            "answer_row_index": -1,
            "answer_column_index": -1,
            "annotation_cell_ids": [],
            "question_params": {
                "query_axis": "row",
                "answer_axis": "column",
                "axis_label": str(missing_label),
                "row_label": str(missing_label),
                "extremum_direction": str(extremum_direction),
                "extremum_phrase": _extremum_phrase(str(extremum_direction), scene_variant=str(scene_variant)),
            },
            "is_unanswerable": True,
            "absence_proof": absence_proof(
                requested_item=str(missing_label),
                visible_candidates=[str(label) for label in row_labels],
                checked_scope="heatmap row labels",
                absence_reason="requested row label is not visible in the heatmap",
            ),
        }
    missing_label = choose_missing_label(
        visible_labels=column_labels,
        candidate_labels=resolve_chart_entity_labels(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.missing_column_candidates"),
            count=max(16, len(column_labels) + 8),
            min_chars=2,
            max_chars=7,
            allow_spaces=False,
        ).labels,
        fallback_prefix="Column ",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.missing_column",
    )
    return {
        "answer_value": UNANSWERABLE_ANSWER,
        "answer_type": "string",
        "answer_row_index": -1,
        "answer_column_index": -1,
        "annotation_cell_ids": [],
        "question_params": {
            "query_axis": "column",
            "answer_axis": "row",
            "axis_label": str(missing_label),
            "column_label": str(missing_label),
            "extremum_direction": str(extremum_direction),
            "extremum_phrase": _extremum_phrase(str(extremum_direction), scene_variant=str(scene_variant)),
        },
        "is_unanswerable": True,
        "absence_proof": absence_proof(
            requested_item=str(missing_label),
            visible_candidates=[str(label) for label in column_labels],
            checked_scope="heatmap column labels",
            absence_reason="requested column label is not visible in the heatmap",
        ),
    }


def _missing_condition_candidate(
    *,
    query_axis: str,
    scene_variant: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Create a controlled-unanswerable request for a condition absent from the legend."""

    missing_condition = choose_missing_label(
        visible_labels=(
            "the highest intensity level",
            "the lowest intensity level",
            "the strongest increase color",
            "the strongest decrease color",
            "the highest activity level",
            "the lowest activity level",
        ),
        candidate_labels=_MISSING_CONDITION_PHRASES,
        fallback_prefix="missing condition ",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.missing_condition",
    )
    visible_conditions = (
        ["the highest activity level", "the lowest activity level"]
        if str(scene_variant) == "calendar_heatmap"
        else [
            "the highest intensity level",
            "the lowest intensity level",
            "the strongest increase color",
            "the strongest decrease color",
        ]
    )
    return {
        "answer_value": UNANSWERABLE_ANSWER,
        "answer_type": "string",
        "answer_row_index": -1,
        "answer_column_index": -1,
        "annotation_cell_ids": [],
        "question_params": {
            "query_axis": str(query_axis),
            "answer_axis": str(query_axis),
            "condition_phrase": str(missing_condition),
            "missing_condition_phrase": str(missing_condition),
        },
        "is_unanswerable": True,
        "absence_proof": absence_proof(
            requested_item=str(missing_condition),
            visible_candidates=visible_conditions,
            checked_scope="heatmap legend/color condition vocabulary",
            absence_reason="requested color condition is not represented by the heatmap legend",
        ),
    }


def _construct_discrete_sample(
    *,
    prompt_key: str,
    scene_variant: str,
    query_axis: str,
    condition_kind: str,
    extremum_direction: str,
    params: Mapping[str, Any],
    instance_seed: int,
    candidate_builder: CandidateBuilder,
    missing_builder: Callable[..., Dict[str, Any]] | None = None,
    allow_unanswerable: bool = False,
) -> Dict[str, Any]:
    """Retry random grids until the objective builder returns a unique answer sample."""

    bin_count = int(params.get("heat_bin_count", group_default(GEN_DEFAULTS, "heat_bin_count", 5)))
    if int(bin_count) < 5:
        raise ValueError(f"{SCENE_NAMESPACE} requires heat_bin_count >= 5")
    row_count, column_count, row_probabilities, column_probabilities = _resolve_row_column_count(
        params,
        scene_variant=str(scene_variant),
        instance_seed=int(instance_seed),
    )
    for attempt in range(512):
        rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.dataset.{attempt}")
        row_labels, column_labels = _labels_for_scene(
            scene_variant=str(scene_variant),
            row_count=int(row_count),
            column_count=int(column_count),
            rng=rng,
        )
        values = [
            [int(rng.randrange(int(bin_count))) for _ in range(int(column_count))]
            for _ in range(int(row_count))
        ]
        cells = _make_cells(row_labels=row_labels, column_labels=column_labels, values=values, bin_count=int(bin_count))
        if bool(allow_unanswerable) and missing_builder is not None and should_use_unanswerable_branch(
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.{prompt_key}",
            enabled=True,
        ):
            candidate = missing_builder(
                query_axis=str(query_axis),
                row_labels=row_labels,
                column_labels=column_labels,
                scene_variant=str(scene_variant),
                extremum_direction=str(extremum_direction),
                instance_seed=int(instance_seed),
            )
        else:
            candidate = candidate_builder(
                scene_variant=str(scene_variant),
                query_axis=str(query_axis),
                condition_kind=str(condition_kind),
                extremum_direction=str(extremum_direction),
                row_labels=row_labels,
                column_labels=column_labels,
                values=values,
                cells=cells,
                bin_count=int(bin_count),
                params=params,
                instance_seed=int(instance_seed),
            )
        if candidate is None:
            continue
        candidate_params = dict(candidate["question_params"])
        query_condition_kind = str(candidate_params.get("condition_kind", condition_kind))
        return _base_dataset(
            scene_title=str(_TITLE_OPTIONS[int(rng.randrange(len(_TITLE_OPTIONS)))]),
            prompt_key=str(prompt_key),
            scene_variant=str(scene_variant),
            query_axis=str(query_axis),
            condition_kind=str(query_condition_kind),
            extremum_direction=str(extremum_direction),
            row_count=int(row_count),
            column_count=int(column_count),
            row_count_probabilities=row_probabilities,
            column_count_probabilities=column_probabilities,
            row_labels=row_labels,
            column_labels=column_labels,
            heat_bin_count=int(bin_count),
            values=values,
            cells=cells,
            answer_value=candidate["answer_value"],
            answer_type=str(candidate["answer_type"]),
            answer_row_index=int(candidate["answer_row_index"]),
            answer_column_index=int(candidate["answer_column_index"]),
            annotation_cell_ids=[str(cell_id) for cell_id in candidate["annotation_cell_ids"]],
            question_params=candidate_params,
            is_unanswerable=bool(candidate.get("is_unanswerable", False)),
            absence_proof_payload=dict(candidate.get("absence_proof", {})),
        )
    raise ValueError(f"could not construct unique-answer heatmap for {SCENE_NAMESPACE}")


def construct_axis_condition_sample(
    *,
    prompt_key: str,
    scene_variant: str,
    query_axis: str,
    condition_kind: str,
    params: Mapping[str, Any],
    instance_seed: int,
    allow_unanswerable: bool,
) -> Dict[str, Any]:
    """Build a discrete heatmap where one row or column wins by condition count."""

    def build_candidate(**kwargs: Any) -> Dict[str, Any] | None:
        return _axis_condition_candidate(
            scene_variant=str(kwargs["scene_variant"]),
            query_axis=str(kwargs["query_axis"]),
            condition_kind=str(kwargs["condition_kind"]),
            row_labels=kwargs["row_labels"],
            column_labels=kwargs["column_labels"],
            values=kwargs["values"],
            cells=kwargs["cells"],
            bin_count=int(kwargs["bin_count"]),
        )

    def build_missing(**kwargs: Any) -> Dict[str, Any]:
        return _missing_condition_candidate(
            query_axis=str(kwargs["query_axis"]),
            scene_variant=str(kwargs["scene_variant"]),
            instance_seed=int(kwargs["instance_seed"]),
        )

    return _construct_discrete_sample(
        prompt_key=str(prompt_key),
        scene_variant=str(scene_variant),
        query_axis=str(query_axis),
        condition_kind=str(condition_kind),
        extremum_direction="hottest",
        params=params,
        instance_seed=int(instance_seed),
        candidate_builder=build_candidate,
        missing_builder=build_missing,
        allow_unanswerable=bool(allow_unanswerable),
    )


def construct_axis_cell_sample(
    *,
    prompt_key: str,
    scene_variant: str,
    query_axis: str,
    extremum_direction: str,
    params: Mapping[str, Any],
    instance_seed: int,
    allow_unanswerable: bool,
) -> Dict[str, Any]:
    """Build a discrete heatmap where a named axis contains a unique extremal cell."""

    def build_candidate(**kwargs: Any) -> Dict[str, Any] | None:
        return _axis_cell_candidate(
            scene_variant=str(kwargs["scene_variant"]),
            query_axis=str(kwargs["query_axis"]),
            extremum_direction=str(kwargs["extremum_direction"]),
            row_labels=kwargs["row_labels"],
            column_labels=kwargs["column_labels"],
            values=kwargs["values"],
            cells=kwargs["cells"],
            params=kwargs["params"],
            instance_seed=int(kwargs["instance_seed"]),
        )

    def build_missing(**kwargs: Any) -> Dict[str, Any]:
        return _missing_axis_cell_candidate(
            query_axis=str(kwargs["query_axis"]),
            row_labels=kwargs["row_labels"],
            column_labels=kwargs["column_labels"],
            scene_variant=str(kwargs["scene_variant"]),
            extremum_direction=str(kwargs["extremum_direction"]),
            instance_seed=int(kwargs["instance_seed"]),
        )

    return _construct_discrete_sample(
        prompt_key=str(prompt_key),
        scene_variant=str(scene_variant),
        query_axis=str(query_axis),
        condition_kind="hot",
        extremum_direction=str(extremum_direction),
        params=params,
        instance_seed=int(instance_seed),
        candidate_builder=build_candidate,
        missing_builder=build_missing,
        allow_unanswerable=bool(allow_unanswerable),
    )


def construct_condition_run_sample(
    *,
    prompt_key: str,
    scene_variant: str,
    condition_kind: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Build a discrete heatmap where one row has the longest matching run."""

    def build_candidate(**kwargs: Any) -> Dict[str, Any] | None:
        return _condition_run_candidate(
            scene_variant=str(kwargs["scene_variant"]),
            condition_kind=str(kwargs["condition_kind"]),
            row_labels=kwargs["row_labels"],
            column_labels=kwargs["column_labels"],
            values=kwargs["values"],
            cells=kwargs["cells"],
            bin_count=int(kwargs["bin_count"]),
        )

    return _construct_discrete_sample(
        prompt_key=str(prompt_key),
        scene_variant=str(scene_variant),
        query_axis="row",
        condition_kind=str(condition_kind),
        extremum_direction="hottest",
        params=params,
        instance_seed=int(instance_seed),
        candidate_builder=build_candidate,
        missing_builder=None,
        allow_unanswerable=False,
    )


__all__ = [
    "construct_axis_cell_sample",
    "construct_axis_condition_sample",
    "construct_colorbar_interval_sample",
    "construct_colorbar_threshold_sample",
    "construct_condition_run_sample",
]
