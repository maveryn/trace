"""Dataset and query construction for matrix chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.shared.unanswerable import (
    UNANSWERABLE_ANSWER,
    absence_proof,
    choose_missing_label,
    should_use_unanswerable_branch,
)
from .defaults import (
    SCENE_NAMESPACE,
    _SCENE_TITLES,
    _active_values,
    _balanced_int,
    _cells_from_values,
    _column_header_key,
    _column_size_support,
    _decoupled_sampling_params,
    _generate_values,
    _header_keys_for_cells,
    _labels_for_scene,
    _line_cell_ids,
    _matrix_size_support,
    _row_header_key,
)


def _choose_axis_extremum(
    *,
    values: List[List[int | None]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    query_axis: str,
    extremum_direction: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Choose one row/column whose ranked extremum answer is unique."""

    target_rank = 2
    line_count = len(values) if str(query_axis) == "row" else len(values[0])
    candidates: List[Tuple[int, List[str], int]] = []
    for axis_index in range(int(line_count)):
        cell_ids = _line_cell_ids(values, query_axis=str(query_axis), axis_index=int(axis_index))
        if len(cell_ids) < int(target_rank):
            continue
        line_values = [int(values[int(cell_id.split("_c")[0][1:])][int(cell_id.split("_c")[1])]) for cell_id in cell_ids]
        ranked_distinct_values = sorted(
            {int(value) for value in line_values},
            reverse=str(extremum_direction) == "highest",
        )
        if len(ranked_distinct_values) < int(target_rank):
            continue
        target = int(ranked_distinct_values[int(target_rank) - 1])
        if line_values.count(int(target)) == 1:
            candidates.append((int(axis_index), list(cell_ids), int(target)))
    if not candidates:
        raise ValueError("could not find unique axis extremum candidate")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.axis_extremum")
    axis_index, cell_ids, target_value = candidates[int(rng.randrange(len(candidates)))]
    winning_cell_id = [
        cell_id
        for cell_id in cell_ids
        if int(values[int(cell_id.split("_c")[0][1:])][int(cell_id.split("_c")[1])]) == int(target_value)
    ][0]
    row_index = int(winning_cell_id.split("_c")[0][1:])
    column_index = int(winning_cell_id.split("_c")[1])
    answer_value = str(column_labels[column_index] if str(query_axis) == "row" else row_labels[row_index])
    selected_header = _row_header_key(axis_index) if str(query_axis) == "row" else _column_header_key(axis_index)
    answer_header = _column_header_key(column_index) if str(query_axis) == "row" else _row_header_key(row_index)
    answer_axis = "column" if str(query_axis) == "row" else "row"
    axis_label = str(row_labels[axis_index] if str(query_axis) == "row" else column_labels[axis_index])
    return {
        "answer_value": answer_value,
        "answer_type": "string",
        "answer_row_index": row_index,
        "answer_column_index": column_index,
        "annotation_cell_ids": list(cell_ids),
        "annotation_header_keys": [selected_header, answer_header],
        "question_params": {
            "query_axis": str(query_axis),
            "axis_label": axis_label,
            "answer_axis": answer_axis,
            "extremum_rank": int(target_rank),
            "extremum_phrase": "second-highest printed value" if str(extremum_direction) == "highest" else "second-lowest printed value",
        },
        "extremum_rank": int(target_rank),
    }


def _choose_unanswerable_axis_extremum(
    *,
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    query_axis: str,
    extremum_direction: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Construct a missing row/column query with a recorded absence proof."""

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
        checked_scope = "matrix row labels"
        visible = [str(label) for label in row_labels]
        answer_axis = "column"
    else:
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
        checked_scope = "matrix column labels"
        visible = [str(label) for label in column_labels]
        answer_axis = "row"
    return {
        "answer_value": UNANSWERABLE_ANSWER,
        "answer_type": "string",
        "answer_row_index": -1,
        "answer_column_index": -1,
        "annotation_cell_ids": [],
        "annotation_header_keys": [],
        "question_params": {
            "query_axis": str(query_axis),
            "axis_label": str(missing_label),
            "answer_axis": str(answer_axis),
            "extremum_rank": 2,
            "extremum_phrase": "second-highest printed value" if str(extremum_direction) == "highest" else "second-lowest printed value",
        },
        "extremum_rank": 2,
        "is_unanswerable": True,
        "absence_proof": absence_proof(
            requested_item=str(missing_label),
            visible_candidates=visible,
            checked_scope=checked_scope,
            absence_reason=f"requested {str(query_axis)} label is not visible in the matrix",
        ),
    }


def _choose_off_diagonal_confusion(
    *,
    values: List[List[int | None]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    instance_seed: int,
) -> Dict[str, Any]:
    row_count = len(values)
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.off_diagonal")
    row_index = int(rng.randrange(row_count))
    winning_columns = [index for index in range(row_count) if index != row_index]
    column_index = int(winning_columns[int(rng.randrange(len(winning_columns)))])
    other_values = [int(values[row_index][c] or 0) for c in range(row_count) if c != row_index and c != column_index]
    values[row_index][column_index] = int(max(other_values + [0]) + 3)
    cell_ids = [f"r{row_index}_c{c}" for c in range(row_count) if c != row_index]
    return {
        "answer_value": str(column_labels[column_index]),
        "answer_type": "string",
        "answer_row_index": row_index,
        "answer_column_index": column_index,
        "annotation_cell_ids": list(cell_ids),
        "annotation_header_keys": [_row_header_key(row_index), _column_header_key(column_index)],
        "question_params": {
            "row_label": str(row_labels[row_index]),
            "answer_axis": "predicted column",
        },
    }


def _threshold_options(values: Sequence[int], *, comparison: str) -> List[Tuple[int, int]]:
    if not values:
        return []
    options: List[Tuple[int, int]] = []
    for threshold in sorted(set(int(value) for value in values)):
        if str(comparison) == "at_least":
            count = sum(1 for value in values if int(value) >= int(threshold))
        else:
            count = sum(1 for value in values if int(value) <= int(threshold))
        if 1 <= int(count) <= min(8, len(values)):
            options.append((int(threshold), int(count)))
    return options


def _choose_threshold_count(
    *,
    values: List[List[int | None]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    cells_by_id: Mapping[str, Dict[str, Any]],
    query_axis: str,
    comparison: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Choose one row/column threshold query with a bounded nontrivial count."""

    line_count = len(values) if str(query_axis) == "row" else len(values[0])
    candidates_by_count: Dict[int, List[Tuple[int, int, int, List[str]]]] = {}
    for axis_index in range(int(line_count)):
        cell_ids = _line_cell_ids(values, query_axis=str(query_axis), axis_index=int(axis_index))
        line_values = [int(cells_by_id[cell_id]["value"]) for cell_id in cell_ids]
        for threshold, count in _threshold_options(line_values, comparison=str(comparison)):
            candidates_by_count.setdefault(int(count), []).append(
                (int(axis_index), int(threshold), int(count), list(cell_ids))
            )
    if not candidates_by_count:
        raise ValueError("could not find threshold-count candidate")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.threshold_count")
    count_choices = sorted(candidates_by_count)
    selected_count = int(count_choices[int(rng.randrange(len(count_choices)))])
    candidates = candidates_by_count[int(selected_count)]
    axis_index, threshold, count, cell_ids = candidates[int(rng.randrange(len(candidates)))]
    if str(comparison) == "at_least":
        matching = [cell_id for cell_id in cell_ids if int(cells_by_id[cell_id]["value"]) >= int(threshold)]
        comparison_phrase = f"at least {int(threshold)}"
    else:
        matching = [cell_id for cell_id in cell_ids if int(cells_by_id[cell_id]["value"]) <= int(threshold)]
        comparison_phrase = f"at most {int(threshold)}"
    selected_header = _row_header_key(axis_index) if str(query_axis) == "row" else _column_header_key(axis_index)
    row_index = int(cells_by_id[matching[0]]["row_index"])
    column_index = int(cells_by_id[matching[0]]["column_index"])
    axis_label = str(row_labels[axis_index] if str(query_axis) == "row" else column_labels[axis_index])
    return {
        "answer_value": int(count),
        "answer_type": "integer",
        "answer_row_index": row_index,
        "answer_column_index": column_index,
        "annotation_cell_ids": list(matching),
        "annotation_header_keys": [selected_header] + _header_keys_for_cells(matching, cells_by_id),
        "question_params": {
            "query_axis": str(query_axis),
            "axis_label": axis_label,
            "comparison_phrase": str(comparison_phrase),
            "threshold_value": int(threshold),
        },
    }


def _base_matrix_dataset(
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Sample the matrix labels, dimensions, cell values, and cell records."""

    row_min, row_max = _matrix_size_support(params)
    col_min, col_max = _column_size_support(params)
    size_params = _decoupled_sampling_params(params, divisor=2, explicit_keys=("row_count_min", "row_count_max"))
    row_count = _balanced_int(
        tuple(range(int(row_min), int(row_max) + 1)),
        params=size_params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.row_count",
    )
    if str(scene_variant) in {"confusion_matrix_counts", "correlation_matrix_signed", "triangular_pairwise_matrix"}:
        column_count = int(row_count)
    else:
        col_params = _decoupled_sampling_params(params, divisor=3, explicit_keys=("column_count_min", "column_count_max"))
        column_count = _balanced_int(
            tuple(range(int(col_min), int(col_max) + 1)),
            params=col_params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.column_count",
        )
    row_labels, column_labels = _labels_for_scene(
        str(scene_variant),
        int(row_count),
        int(column_count),
        instance_seed=int(instance_seed),
    )
    values, scene_meta = _generate_values(
        scene_variant=str(scene_variant),
        row_count=int(row_count),
        column_count=int(column_count),
        instance_seed=int(instance_seed),
    )
    cells, cells_by_id = _cells_from_values(values, row_labels=row_labels, column_labels=column_labels)
    return {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "row_labels": list(row_labels),
        "column_labels": list(column_labels),
        "values": values,
        "cells": list(cells),
        "cells_by_id": {str(key): dict(value) for key, value in cells_by_id.items()},
        "scene_meta": dict(scene_meta),
    }


def _finalize_dataset(
    *,
    base: Mapping[str, Any],
    scene_variant: str,
    query_axis: str,
    extremum_direction: str,
    comparison: str,
    query: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    values = [list(row) for row in base["values"]]
    cells_by_id = {str(key): dict(value) for key, value in dict(base["cells_by_id"]).items()}
    row_count = int(base["row_count"])
    column_count = int(base["column_count"])
    cells = [dict(cells_by_id[f"r{r}_c{c}"]) for r in range(int(row_count)) for c in range(int(column_count))]
    active_values = _active_values(values)
    scene_title_options = _SCENE_TITLES[str(scene_variant)]
    title_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.title")
    scene_title = str(scene_title_options[int(title_rng.randrange(len(scene_title_options)))])
    return {
        "scene_title": scene_title,
        "row_count": int(row_count),
        "column_count": int(column_count),
        "row_labels": list(base["row_labels"]),
        "column_labels": list(base["column_labels"]),
        "values": values,
        "cells": list(cells),
        "cells_by_id": {str(key): dict(value) for key, value in cells_by_id.items()},
        "value_min": int(min(active_values)),
        "value_max": int(max(active_values)),
        "scene_meta": dict(base["scene_meta"]),
        "query_axis": str(query_axis),
        "extremum_direction": str(extremum_direction),
        "comparison": str(comparison),
        "is_unanswerable": bool(query.get("is_unanswerable", False)),
        "absence_proof": dict(query.get("absence_proof", {})),
        **dict(query),
    }


def construct_axis_ranked_dataset(
    *,
    scene_variant: str,
    query_axis: str,
    extremum_direction: str,
    supports_unanswerable: bool,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    """Build matrix data for a ranked-extreme cell selection over one axis."""

    base = _base_matrix_dataset(scene_variant=str(scene_variant), params=params, instance_seed=int(instance_seed))
    values = [list(row) for row in base["values"]]
    row_labels = list(base["row_labels"])
    column_labels = list(base["column_labels"])
    if should_use_unanswerable_branch(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.axis_ranked_extreme",
        enabled=bool(supports_unanswerable),
    ):
        query = _choose_unanswerable_axis_extremum(
            row_labels=row_labels,
            column_labels=column_labels,
            query_axis=str(query_axis),
            extremum_direction=str(extremum_direction),
            instance_seed=int(instance_seed),
        )
    else:
        query = _choose_axis_extremum(
            values=values,
            row_labels=row_labels,
            column_labels=column_labels,
            query_axis=str(query_axis),
            extremum_direction=str(extremum_direction),
            instance_seed=int(instance_seed),
        )
    return _finalize_dataset(
        base=base,
        scene_variant=str(scene_variant),
        query_axis=str(query_axis),
        extremum_direction=str(extremum_direction),
        comparison="",
        query=query,
        instance_seed=int(instance_seed),
    )


def construct_confusion_off_diagonal_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    scene_variant = "confusion_matrix_counts"
    base = _base_matrix_dataset(scene_variant=scene_variant, params=params, instance_seed=int(instance_seed))
    values = [list(row) for row in base["values"]]
    row_labels = list(base["row_labels"])
    column_labels = list(base["column_labels"])
    query = _choose_off_diagonal_confusion(
        values=values,
        row_labels=row_labels,
        column_labels=column_labels,
        instance_seed=int(instance_seed),
    )
    cells, cells_by_id = _cells_from_values(values, row_labels=row_labels, column_labels=column_labels)
    adjusted_base = {**dict(base), "values": values, "cells": cells, "cells_by_id": cells_by_id}
    return _finalize_dataset(
        base=adjusted_base,
        scene_variant=scene_variant,
        query_axis="row",
        extremum_direction="highest",
        comparison="",
        query=query,
        instance_seed=int(instance_seed),
    )


def construct_threshold_dataset(
    *,
    scene_variant: str,
    query_axis: str,
    comparison: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    base = _base_matrix_dataset(scene_variant=str(scene_variant), params=params, instance_seed=int(instance_seed))
    values = [list(row) for row in base["values"]]
    query = _choose_threshold_count(
        values=values,
        row_labels=list(base["row_labels"]),
        column_labels=list(base["column_labels"]),
        cells_by_id={str(key): dict(value) for key, value in dict(base["cells_by_id"]).items()},
        query_axis=str(query_axis),
        comparison=str(comparison),
        instance_seed=int(instance_seed),
    )
    return _finalize_dataset(
        base=base,
        scene_variant=str(scene_variant),
        query_axis=str(query_axis),
        extremum_direction="",
        comparison=str(comparison),
        query=query,
        instance_seed=int(instance_seed),
    )
