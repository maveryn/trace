"""Raven-matrix symbolic rule constructors."""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.puzzles.shared.symbol_rendering import PUZZLE_OBJECT_TYPES
from trace_tasks.tasks.shared.mcq import option_label_for_index


COLOR_POOL: tuple[dict[str, Any], ...] = (
    {"name": "blue", "rgb": [74, 127, 214]},
    {"name": "orange", "rgb": [214, 130, 74]},
    {"name": "green", "rgb": [64, 164, 108]},
    {"name": "red", "rgb": [196, 90, 100]},
    {"name": "purple", "rgb": [136, 100, 196]},
    {"name": "gold", "rgb": [205, 162, 62]},
)
ROTATION_OPS: tuple[str, ...] = (
    "identity",
    "rotate_cw_90",
    "rotate_ccw_90",
)
ROTATION_SEQUENCES: tuple[tuple[str, str, tuple[str, str, str]], ...] = (
    ("row_clockwise_first", "row", ("identity", "rotate_cw_90", "rotate_ccw_90")),
    ("row_counterclockwise_first", "row", ("identity", "rotate_ccw_90", "rotate_cw_90")),
    ("column_clockwise_first", "column", ("identity", "rotate_cw_90", "rotate_ccw_90")),
    ("column_counterclockwise_first", "column", ("identity", "rotate_ccw_90", "rotate_cw_90")),
)
SET_OPERATIONS: tuple[str, ...] = ("union", "intersection")
ANALOGICAL_TRANSFORMS: tuple[str, ...] = (
    "shape_cycle",
    "color_cycle",
    "size_cycle",
)
SIZE_CYCLE: tuple[float, ...] = (0.48, 0.66, 0.84)
POSITION_PROGRESSIONS: tuple[tuple[str, str, str], ...] = (
    ("row_left_to_right", "row", "left_to_right"),
    ("row_right_to_left", "row", "right_to_left"),
    ("column_top_to_bottom", "column", "top_to_bottom"),
    ("column_bottom_to_top", "column", "bottom_to_top"),
)
FEATURE_BINDING_MODES: tuple[tuple[str, str, str], ...] = (
    ("row_shape_column_color", "shape", "color"),
    ("row_color_column_shape", "color", "shape"),
    ("row_shape_column_size", "shape", "size"),
    ("row_size_column_shape", "size", "shape"),
)


def canonical_panel_spec(panel_spec: Mapping[str, Any]) -> str:
    """Return a stable signature for one Raven panel spec."""

    return json.dumps(panel_spec, sort_keys=True, separators=(",", ":"))


def matrix_rows_from_specs(
    panel_grid: Sequence[Sequence[Mapping[str, Any]]],
) -> list[list[dict[str, Any]]]:
    """Build traced matrix rows with the lower-right cell hidden."""

    rows: list[list[dict[str, Any]]] = []
    for row_index, row in enumerate(panel_grid):
        row_cells: list[dict[str, Any]] = []
        for col_index, panel_spec in enumerate(row):
            is_unknown = bool(row_index == 2 and col_index == 2)
            row_cells.append(
                {
                    "cell_id": f"cell_{row_index}_{col_index}",
                    "row_index": int(row_index),
                    "col_index": int(col_index),
                    "is_unknown": bool(is_unknown),
                    "panel_spec": None if is_unknown else dict(panel_spec),
                }
            )
        rows.append(row_cells)
    return rows


def build_option_specs(
    *,
    correct_panel_spec: Mapping[str, Any],
    distractor_panel_specs: Sequence[Mapping[str, Any]],
    correct_option_index: int,
    option_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build labeled image options with exactly one unique correct panel."""

    correct_signature = canonical_panel_spec(correct_panel_spec)
    unique_distractors: list[dict[str, Any]] = []
    seen = {correct_signature}
    for panel_spec in distractor_panel_specs:
        signature = canonical_panel_spec(panel_spec)
        if signature in seen:
            continue
        seen.add(signature)
        unique_distractors.append(dict(panel_spec))
    if len(unique_distractors) < int(option_count) - 1:
        raise ValueError("not enough unique Raven distractor panels")

    option_panel_specs = [
        dict(spec) for spec in unique_distractors[: int(option_count) - 1]
    ]
    option_panel_specs.insert(int(correct_option_index), dict(correct_panel_spec))
    option_specs: list[dict[str, Any]] = []
    option_labels: list[str] = []
    for option_index, panel_spec in enumerate(option_panel_specs):
        option_label = str(option_label_for_index(int(option_index)))
        option_labels.append(option_label)
        option_specs.append(
            {
                "option_panel_id": f"option_{option_label}",
                "option_index": int(option_index),
                "option_label": str(option_label),
                "panel_spec": dict(panel_spec),
                "panel_signature": canonical_panel_spec(panel_spec),
                "is_correct": bool(option_index == int(correct_option_index)),
            }
        )
    return option_specs, option_labels


def attribute_panel_spec(
    *,
    object_type: str,
    color: Mapping[str, Any],
    size_scale: float = 0.78,
) -> dict[str, Any]:
    """Build one shape/color/size attribute panel spec."""

    return {
        "panel_kind": "attribute",
        "object_type": str(object_type),
        "fill_name": str(color["name"]),
        "fill_rgb": [int(value) for value in color["rgb"]],
        "size_scale": round(float(size_scale), 3),
    }


def _feature_value_label(feature_name: str, value: Any) -> str:
    """Return a compact trace label for one feature value."""

    if str(feature_name) == "color":
        return str(value["name"])
    if str(feature_name) == "size":
        return f"{float(value):.2f}"
    return str(value)


def _feature_panel_spec(
    *,
    feature_values: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one attribute panel from feature-axis values."""

    return attribute_panel_spec(
        object_type=str(feature_values["shape"]),
        color=dict(feature_values["color"]),
        size_scale=float(feature_values["size"]),
    )


def count_panel_spec(
    *,
    count: int,
    object_type: str,
    color: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one repeated-object count panel spec."""

    return {
        "panel_kind": "count",
        "object_type": str(object_type),
        "fill_name": str(color["name"]),
        "fill_rgb": [int(value) for value in color["rgb"]],
        "count": int(count),
    }


def build_count_progression_dataset(
    *,
    rng,
    count_min: int,
    count_max: int,
    option_count: int,
    correct_option_index: int,
) -> dict[str, Any]:
    """Construct a direct row/column count-progression Raven dataset."""

    if int(count_max - count_min + 1) < int(option_count):
        raise ValueError("count support must contain at least option_count values")
    table: list[list[int]] | None = None
    progression_axis = "row"
    progression_delta = 1
    progression_starts: list[int] = []
    delta_candidates = (-3, -2, -1, 1, 2, 3)
    for _ in range(200):
        axis = str(rng.choice(("row", "column")))
        delta = int(rng.choice(delta_candidates))
        valid_starts = [
            int(start)
            for start in range(int(count_min), int(count_max) + 1)
            if int(count_min) <= int(start + delta) <= int(count_max)
            and int(count_min) <= int(start + (2 * delta)) <= int(count_max)
        ]
        if len(valid_starts) < 3:
            continue
        starts = [int(value) for value in rng.sample(valid_starts, 3)]
        if axis == "row":
            candidate_table = [
                [
                    int(start),
                    int(start + delta),
                    int(start + (2 * delta)),
                ]
                for start in starts
            ]
        else:
            candidate_table = [[0 for _ in range(3)] for _ in range(3)]
            for col_index, start in enumerate(starts):
                candidate_table[0][col_index] = int(start)
                candidate_table[1][col_index] = int(start + delta)
                candidate_table[2][col_index] = int(start + (2 * delta))
        flat_counts = [int(value) for row in candidate_table for value in row]
        if min(flat_counts) >= int(count_min) and max(flat_counts) <= int(count_max):
            table = candidate_table
            progression_axis = axis
            progression_delta = int(delta)
            progression_starts = starts
            break
    if table is None:
        raise RuntimeError("failed to construct count-progression Raven matrix")

    color = dict(rng.choice(COLOR_POOL))
    object_type = str(rng.choice(tuple(PUZZLE_OBJECT_TYPES)))
    panel_grid = [
        [
            count_panel_spec(
                count=int(table[row_index][col_index]),
                object_type=object_type,
                color=color,
            )
            for col_index in range(3)
        ]
        for row_index in range(3)
    ]
    answer_panel_spec = dict(panel_grid[2][2])
    answer_count = int(answer_panel_spec["count"])
    count_support = [
        int(value)
        for value in range(int(count_min), int(count_max) + 1)
        if int(value) != answer_count
    ]
    rng.shuffle(count_support)
    distractors = [
        count_panel_spec(count=int(count), object_type=object_type, color=color)
        for count in count_support
    ]
    option_specs, option_labels = build_option_specs(
        correct_panel_spec=answer_panel_spec,
        distractor_panel_specs=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    return {
        "matrix_rows": matrix_rows_from_specs(panel_grid),
        "matrix_panel_specs": [[dict(spec) for spec in row] for row in panel_grid],
        "answer_panel_spec": dict(answer_panel_spec),
        "answer_count": int(answer_count),
        "correct_option_index": int(correct_option_index),
        "correct_option_panel_id": str(option_specs[correct_option_index]["option_panel_id"]),
        "answer_option_label": str(option_label_for_index(correct_option_index)),
        "option_specs": option_specs,
        "option_labels": option_labels,
        "option_count": int(option_count),
        "solver_trace": {
            "rule_type": "count_progression_matrix",
            "count_rule": (
                "repeat the same count change across rows"
                if progression_axis == "row"
                else "repeat the same count change down columns"
            ),
            "progression_axis": str(progression_axis),
            "progression_delta": int(progression_delta),
            "progression_deltas": [int(progression_delta), int(progression_delta)],
            "progression_starts": [int(value) for value in progression_starts],
            "count_table": [[int(value) for value in row] for row in table],
            "target_row_index": 2,
            "target_col_index": 2,
            "answer_count": int(answer_count),
            "correct_option_index": int(correct_option_index),
            "correct_option_label": str(option_label_for_index(correct_option_index)),
        },
    }


def transform_coord(row: int, col: int, op: str) -> tuple[int, int]:
    """Apply one D4 transform to a 3x3 pattern coordinate."""

    r = int(row)
    c = int(col)
    if op == "identity":
        return r, c
    if op == "rotate_cw_90":
        return c, 2 - r
    if op == "rotate_ccw_90":
        return 2 - c, r
    raise ValueError(f"unsupported Raven transform op: {op}")


def apply_transform(
    coords: Sequence[Sequence[int]],
    op: str,
) -> tuple[tuple[int, int], ...]:
    """Apply one transform op to a coordinate set."""

    return tuple(
        sorted(
            transform_coord(int(row), int(col), str(op))
            for row, col in coords
        )
    )


def pattern_panel_spec(
    *,
    cells: Sequence[Sequence[int]],
    color: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one spatial pattern panel spec."""

    return {
        "panel_kind": "pattern",
        "grid_size": 3,
        "fill_name": str(color["name"]),
        "fill_rgb": [int(value) for value in color["rgb"]],
        "cells": [
            [int(row), int(col)]
            for row, col in sorted((int(r), int(c)) for r, c in cells)
        ],
    }


def all_grid_cells() -> list[tuple[int, int]]:
    """Return all coordinates in a 3x3 mini-grid."""

    return [(int(row), int(col)) for row in range(3) for col in range(3)]


def sample_cell_set(
    *,
    rng,
    min_count: int = 2,
    max_count: int = 5,
) -> tuple[tuple[int, int], ...]:
    """Sample one nonempty 3x3 cell subset."""

    count = int(rng.randint(int(min_count), int(max_count)))
    return tuple(sorted(rng.sample(all_grid_cells(), int(count))))


def random_pattern_distractor_specs(
    *,
    target_cells: Sequence[Sequence[int]],
    color: Mapping[str, Any],
    rng,
    min_count: int = 1,
    max_count: int = 5,
    target_count: int = 12,
) -> list[dict[str, Any]]:
    """Build unique random pattern distractor specs around a target."""

    target_spec = pattern_panel_spec(cells=target_cells, color=color)
    seen = {canonical_panel_spec(target_spec)}
    specs: list[dict[str, Any]] = []
    candidate_sets: list[tuple[tuple[int, int], ...]] = []
    for count in range(int(min_count), int(max_count) + 1):
        candidate_sets.extend(
            tuple(sorted(cells))
            for cells in combinations(all_grid_cells(), int(count))
        )
    rng.shuffle(candidate_sets)
    for cells in candidate_sets:
        spec = pattern_panel_spec(cells=cells, color=color)
        signature = canonical_panel_spec(spec)
        if signature in seen:
            continue
        seen.add(signature)
        specs.append(spec)
        if len(specs) >= int(target_count):
            break
    return specs


def sample_rotation_base_cells(*, rng) -> tuple[tuple[int, int], ...]:
    """Sample a 3x3 cell set with distinct 90-degree rotation states."""

    for _ in range(400):
        cells = sample_cell_set(rng=rng, min_count=3, max_count=5)
        signatures = {
            tuple(apply_transform(cells, str(op)))
            for op in ROTATION_OPS
        }
        if len(signatures) == len(ROTATION_OPS):
            return tuple(cells)
    raise RuntimeError("failed to sample non-symmetric Raven rotation base")


def spatial_distractor_specs(
    *,
    target_cells: Sequence[Sequence[int]],
    base_cells: Sequence[Sequence[int]],
    color: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build unique spatial-transform distractors."""

    target_signature = canonical_panel_spec(
        pattern_panel_spec(cells=target_cells, color=color)
    )
    specs: list[dict[str, Any]] = []
    seen = {target_signature}
    for op in ROTATION_OPS:
        spec = pattern_panel_spec(cells=apply_transform(base_cells, op), color=color)
        signature = canonical_panel_spec(spec)
        if signature not in seen:
            seen.add(signature)
            specs.append(spec)
    target_set = {(int(row), int(col)) for row, col in target_cells}
    for row_index in range(3):
        for col_index in range(3):
            toggled = set(target_set)
            coord = (int(row_index), int(col_index))
            if coord in toggled:
                if len(toggled) <= 3:
                    continue
                toggled.remove(coord)
            else:
                if len(toggled) >= 5:
                    continue
                toggled.add(coord)
            spec = pattern_panel_spec(cells=sorted(toggled), color=color)
            signature = canonical_panel_spec(spec)
            if signature not in seen:
                seen.add(signature)
                specs.append(spec)
    return specs


def build_spatial_transform_dataset(
    *,
    rng,
    option_count: int,
    correct_option_index: int,
) -> dict[str, Any]:
    """Construct a row/column Raven matrix from 90-degree rotations only."""

    color = dict(rng.choice(COLOR_POOL))
    panel_grid: list[list[dict[str, Any]]] | None = None
    base_patterns: list[tuple[tuple[int, int], ...]] = []
    rotation_mode = "row_clockwise_first"
    progression_axis = "row"
    rotation_sequence: tuple[str, str, str] = ROTATION_SEQUENCES[0][2]
    for _ in range(300):
        mode, axis, sequence = rng.choice(ROTATION_SEQUENCES)
        bases = [sample_rotation_base_cells(rng=rng) for _ in range(3)]
        candidate_grid: list[list[dict[str, Any]]]
        if str(axis) == "row":
            candidate_grid = [
                [
                    pattern_panel_spec(
                        cells=apply_transform(bases[row_index], str(op)),
                        color=color,
                    )
                    for op in sequence
                ]
                for row_index in range(3)
            ]
        else:
            candidate_grid = [
                [
                    pattern_panel_spec(
                        cells=apply_transform(bases[col_index], str(sequence[row_index])),
                        color=color,
                    )
                    for col_index in range(3)
                ]
                for row_index in range(3)
            ]
        target_signature = canonical_panel_spec(candidate_grid[2][2])
        visible_signatures = {
            canonical_panel_spec(candidate_grid[row_index][col_index])
            for row_index in range(3)
            for col_index in range(3)
            if not (row_index == 2 and col_index == 2)
        }
        if target_signature in visible_signatures:
            continue
        panel_grid = candidate_grid
        base_patterns = bases
        rotation_mode = str(mode)
        progression_axis = str(axis)
        rotation_sequence = tuple(str(op) for op in sequence)
        break
    if panel_grid is None:
        raise RuntimeError("failed to construct rotation-only Raven matrix")

    answer_panel_spec = dict(panel_grid[2][2])
    target_cells = tuple(tuple(coord) for coord in answer_panel_spec["cells"])
    target_base = base_patterns[2]
    distractors = spatial_distractor_specs(
        target_cells=target_cells,
        base_cells=target_base,
        color=color,
    )
    rng.shuffle(distractors)
    option_specs, option_labels = build_option_specs(
        correct_panel_spec=answer_panel_spec,
        distractor_panel_specs=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    return {
        "matrix_rows": matrix_rows_from_specs(panel_grid),
        "matrix_panel_specs": [[dict(spec) for spec in row] for row in panel_grid],
        "answer_panel_spec": dict(answer_panel_spec),
        "correct_option_index": int(correct_option_index),
        "correct_option_panel_id": str(option_specs[correct_option_index]["option_panel_id"]),
        "answer_option_label": str(option_label_for_index(correct_option_index)),
        "option_specs": option_specs,
        "option_labels": option_labels,
        "option_count": int(option_count),
        "solver_trace": {
            "rule_type": "spatial_transform_matrix",
            "rotation_mode": str(rotation_mode),
            "progression_axis": str(progression_axis),
            "rotation_sequence": [str(value) for value in rotation_sequence],
            "base_patterns": [
                [[int(row), int(col)] for row, col in base_cells]
                for base_cells in base_patterns
            ],
            "target_row_index": 2,
            "target_col_index": 2,
            "answer_cells": [[int(row), int(col)] for row, col in target_cells],
            "correct_option_index": int(correct_option_index),
            "correct_option_label": str(option_label_for_index(correct_option_index)),
        },
    }


def apply_set_operation(
    left_cells: Sequence[Sequence[int]],
    right_cells: Sequence[Sequence[int]],
    operation: str,
) -> tuple[tuple[int, int], ...]:
    """Apply one cell-set operation over two 3x3 mini-grid patterns."""

    left = {(int(row), int(col)) for row, col in left_cells}
    right = {(int(row), int(col)) for row, col in right_cells}
    if operation == "union":
        result = left | right
    elif operation == "intersection":
        result = left & right
    else:
        raise ValueError(f"unsupported Raven set operation: {operation}")
    return tuple(sorted(result))


def set_operation_matches(
    left_cells: Sequence[Sequence[int]],
    right_cells: Sequence[Sequence[int]],
    result_cells: Sequence[Sequence[int]],
) -> list[str]:
    """Return set operations that map two inputs to the target result."""

    target = tuple(sorted((int(row), int(col)) for row, col in result_cells))
    return [
        str(operation)
        for operation in SET_OPERATIONS
        if apply_set_operation(left_cells, right_cells, str(operation)) == target
    ]


def build_set_operation_dataset(
    *,
    rng,
    option_count: int,
    correct_option_index: int,
) -> dict[str, Any]:
    """Construct a row-wise set-operation Raven matrix dataset."""

    operation = str(rng.choice(SET_OPERATIONS))
    color = dict(rng.choice(COLOR_POOL))
    panel_grid: list[list[dict[str, Any]]] | None = None
    row_inputs: list[dict[str, Any]] = []
    for _ in range(400):
        candidate_grid: list[list[dict[str, Any]]] = []
        candidate_inputs: list[dict[str, Any]] = []
        valid = True
        for row_index in range(3):
            row_valid = False
            for _row_attempt in range(120):
                left_cells = sample_cell_set(rng=rng, min_count=2, max_count=5)
                right_cells = sample_cell_set(rng=rng, min_count=2, max_count=5)
                result_cells = apply_set_operation(left_cells, right_cells, operation)
                if not 1 <= len(result_cells) <= 5:
                    continue
                if tuple(result_cells) in {tuple(left_cells), tuple(right_cells)}:
                    continue
                if set_operation_matches(left_cells, right_cells, result_cells) != [operation]:
                    continue
                candidate_grid.append(
                    [
                        pattern_panel_spec(cells=left_cells, color=color),
                        pattern_panel_spec(cells=right_cells, color=color),
                        pattern_panel_spec(cells=result_cells, color=color),
                    ]
                )
                candidate_inputs.append(
                    {
                        "row_index": int(row_index),
                        "left_cells": [[int(row), int(col)] for row, col in left_cells],
                        "right_cells": [[int(row), int(col)] for row, col in right_cells],
                        "result_cells": [[int(row), int(col)] for row, col in result_cells],
                    }
                )
                row_valid = True
                break
            if not row_valid:
                valid = False
                break
        if valid and len(candidate_grid) == 3:
            panel_grid = candidate_grid
            row_inputs = candidate_inputs
            break
    if panel_grid is None:
        raise RuntimeError("failed to construct set-operation Raven matrix")

    answer_panel_spec = dict(panel_grid[2][2])
    target_cells = tuple(tuple(coord) for coord in answer_panel_spec["cells"])
    distractors: list[dict[str, Any]] = []
    row2_left = tuple(tuple(coord) for coord in panel_grid[2][0]["cells"])
    row2_right = tuple(tuple(coord) for coord in panel_grid[2][1]["cells"])
    for alt_operation in SET_OPERATIONS:
        alt_cells = apply_set_operation(row2_left, row2_right, str(alt_operation))
        if 1 <= len(alt_cells) <= 5:
            distractors.append(pattern_panel_spec(cells=alt_cells, color=color))
    distractors.extend(
        dict(panel_grid[row_index][col_index])
        for row_index in range(3)
        for col_index in range(3)
        if not (row_index == 2 and col_index == 2)
    )
    distractors.extend(
        random_pattern_distractor_specs(
            target_cells=target_cells,
            color=color,
            rng=rng,
            min_count=1,
            max_count=5,
            target_count=12,
        )
    )
    rng.shuffle(distractors)
    option_specs, option_labels = build_option_specs(
        correct_panel_spec=answer_panel_spec,
        distractor_panel_specs=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    return {
        "matrix_rows": matrix_rows_from_specs(panel_grid),
        "matrix_panel_specs": [[dict(spec) for spec in row] for row in panel_grid],
        "answer_panel_spec": dict(answer_panel_spec),
        "correct_option_index": int(correct_option_index),
        "correct_option_panel_id": str(option_specs[correct_option_index]["option_panel_id"]),
        "answer_option_label": str(option_label_for_index(correct_option_index)),
        "option_specs": option_specs,
        "option_labels": option_labels,
        "option_count": int(option_count),
        "solver_trace": {
            "rule_type": "set_operation_matrix",
            "operation": str(operation),
            "row_inputs": list(row_inputs),
            "target_row_index": 2,
            "target_col_index": 2,
            "answer_cells": [[int(row), int(col)] for row, col in target_cells],
            "correct_option_index": int(correct_option_index),
            "correct_option_label": str(option_label_for_index(correct_option_index)),
        },
    }


def color_by_name(name: str) -> dict[str, Any]:
    """Return one color record by palette name."""

    for color in COLOR_POOL:
        if str(color["name"]) == str(name):
            return dict(color)
    raise ValueError(f"unknown Raven color name: {name}")


def attribute_distractor_specs(
    *,
    target_panel_spec: Mapping[str, Any],
    rng,
) -> list[dict[str, Any]]:
    """Build unique attribute-panel distractors for analogical transforms."""

    target_signature = canonical_panel_spec(target_panel_spec)
    target_color = color_by_name(str(target_panel_spec["fill_name"]))
    target_object_type = str(target_panel_spec["object_type"])
    target_size = float(target_panel_spec.get("size_scale", 0.78))
    specs: list[dict[str, Any]] = []
    seen = {target_signature}

    candidates: list[dict[str, Any]] = []
    for shape in PUZZLE_OBJECT_TYPES:
        candidates.append(
            attribute_panel_spec(
                object_type=str(shape),
                color=target_color,
                size_scale=target_size,
            )
        )
    for color in COLOR_POOL:
        candidates.append(
            attribute_panel_spec(
                object_type=target_object_type,
                color=dict(color),
                size_scale=target_size,
            )
        )
    for size_scale in SIZE_CYCLE:
        candidates.append(
            attribute_panel_spec(
                object_type=target_object_type,
                color=target_color,
                size_scale=float(size_scale),
            )
        )
    for shape in PUZZLE_OBJECT_TYPES:
        for color in COLOR_POOL:
            candidates.append(
                attribute_panel_spec(
                    object_type=str(shape),
                    color=dict(color),
                    size_scale=target_size,
                )
            )
    rng.shuffle(candidates)
    for spec in candidates:
        signature = canonical_panel_spec(spec)
        if signature in seen:
            continue
        seen.add(signature)
        specs.append(spec)
        if len(specs) >= 16:
            break
    return specs


def build_analogical_transform_dataset(
    *,
    rng,
    option_count: int,
    correct_option_index: int,
) -> dict[str, Any]:
    """Construct a row-wise analogical-transform Raven matrix dataset."""

    transform_kind = str(rng.choice(ANALOGICAL_TRANSFORMS))
    shapes = [str(value) for value in PUZZLE_OBJECT_TYPES]
    rng.shuffle(shapes)
    shape_cycle = [str(value) for value in shapes[:3]]
    colors = [dict(value) for value in COLOR_POOL]
    rng.shuffle(colors)
    color_cycle = [dict(value) for value in colors[:3]]
    row_offsets = [0, 1, 2]
    rng.shuffle(row_offsets)

    panel_grid: list[list[dict[str, Any]]] = []
    for row_index in range(3):
        row_specs: list[dict[str, Any]] = []
        for col_index in range(3):
            offset = int((row_offsets[row_index] + col_index) % 3)
            if transform_kind == "shape_cycle":
                object_type = str(shape_cycle[offset])
                color = dict(color_cycle[row_index])
                size_scale = 0.76
            elif transform_kind == "color_cycle":
                object_type = str(shape_cycle[row_index])
                color = dict(color_cycle[offset])
                size_scale = 0.76
            else:
                object_type = str(shape_cycle[row_index])
                color = dict(color_cycle[row_index])
                size_scale = float(SIZE_CYCLE[offset])
            row_specs.append(
                attribute_panel_spec(
                    object_type=object_type,
                    color=color,
                    size_scale=float(size_scale),
                )
            )
        panel_grid.append(row_specs)

    answer_panel_spec = dict(panel_grid[2][2])
    distractors = attribute_distractor_specs(
        target_panel_spec=answer_panel_spec,
        rng=rng,
    )
    distractors.extend(
        dict(panel_grid[row_index][col_index])
        for row_index in range(3)
        for col_index in range(3)
        if not (row_index == 2 and col_index == 2)
    )
    rng.shuffle(distractors)
    option_specs, option_labels = build_option_specs(
        correct_panel_spec=answer_panel_spec,
        distractor_panel_specs=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    return {
        "matrix_rows": matrix_rows_from_specs(panel_grid),
        "matrix_panel_specs": [[dict(spec) for spec in row] for row in panel_grid],
        "answer_panel_spec": dict(answer_panel_spec),
        "correct_option_index": int(correct_option_index),
        "correct_option_panel_id": str(option_specs[correct_option_index]["option_panel_id"]),
        "answer_option_label": str(option_label_for_index(correct_option_index)),
        "option_specs": option_specs,
        "option_labels": option_labels,
        "option_count": int(option_count),
        "solver_trace": {
            "rule_type": "analogical_transform_matrix",
            "transform_kind": str(transform_kind),
            "shape_cycle": [str(value) for value in shape_cycle],
            "color_cycle": [str(color["name"]) for color in color_cycle],
            "size_cycle": [float(value) for value in SIZE_CYCLE],
            "row_offsets": [int(value) for value in row_offsets],
            "target_row_index": 2,
            "target_col_index": 2,
            "correct_option_index": int(correct_option_index),
            "correct_option_label": str(option_label_for_index(correct_option_index)),
        },
    }


def build_feature_binding_dataset(
    *,
    rng,
    option_count: int,
    correct_option_index: int,
) -> dict[str, Any]:
    """Construct a Raven matrix where row and column bind independent features."""

    binding_mode, row_feature, column_feature = [
        str(value) for value in rng.choice(FEATURE_BINDING_MODES)
    ]
    shapes = [str(value) for value in PUZZLE_OBJECT_TYPES]
    rng.shuffle(shapes)
    colors = [dict(value) for value in COLOR_POOL]
    rng.shuffle(colors)
    sizes = [float(value) for value in SIZE_CYCLE]
    rng.shuffle(sizes)

    axis_values: dict[str, list[Any]] = {
        "shape": list(shapes[:3]),
        "color": [dict(value) for value in colors[:3]],
        "size": [float(value) for value in sizes[:3]],
    }
    default_values: dict[str, Any] = {
        "shape": str(shapes[3] if len(shapes) > 3 else shapes[0]),
        "color": dict(colors[3] if len(colors) > 3 else colors[0]),
        "size": 0.76,
    }

    def values_for(row_index: int, col_index: int) -> dict[str, Any]:
        values = {
            "shape": default_values["shape"],
            "color": dict(default_values["color"]),
            "size": float(default_values["size"]),
        }
        values[row_feature] = axis_values[row_feature][int(row_index)]
        col_value = axis_values[column_feature][int(col_index)]
        values[column_feature] = dict(col_value) if column_feature == "color" else col_value
        return values

    panel_grid = [
        [
            _feature_panel_spec(feature_values=values_for(row_index, col_index))
            for col_index in range(3)
        ]
        for row_index in range(3)
    ]
    answer_panel_spec = dict(panel_grid[2][2])
    distractors = [
        dict(panel_grid[row_index][col_index])
        for row_index in range(3)
        for col_index in range(3)
        if not (row_index == 2 and col_index == 2)
    ]
    rng.shuffle(distractors)
    option_specs, option_labels = build_option_specs(
        correct_panel_spec=answer_panel_spec,
        distractor_panel_specs=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    feature_table = [
        [
            {
                str(feature): _feature_value_label(str(feature), value)
                for feature, value in values_for(row_index, col_index).items()
            }
            for col_index in range(3)
        ]
        for row_index in range(3)
    ]
    return {
        "matrix_rows": matrix_rows_from_specs(panel_grid),
        "matrix_panel_specs": [[dict(spec) for spec in row] for row in panel_grid],
        "answer_panel_spec": dict(answer_panel_spec),
        "correct_option_index": int(correct_option_index),
        "correct_option_panel_id": str(option_specs[correct_option_index]["option_panel_id"]),
        "answer_option_label": str(option_label_for_index(correct_option_index)),
        "option_specs": option_specs,
        "option_labels": option_labels,
        "option_count": int(option_count),
        "solver_trace": {
            "rule_type": "feature_binding_matrix",
            "binding_mode": str(binding_mode),
            "row_feature": str(row_feature),
            "column_feature": str(column_feature),
            "row_feature_values": [
                _feature_value_label(str(row_feature), value)
                for value in axis_values[row_feature]
            ],
            "column_feature_values": [
                _feature_value_label(str(column_feature), value)
                for value in axis_values[column_feature]
            ],
            "feature_table": feature_table,
            "target_row_index": 2,
            "target_col_index": 2,
            "correct_option_index": int(correct_option_index),
            "correct_option_label": str(option_label_for_index(correct_option_index)),
        },
    }


def build_position_progression_dataset(
    *,
    rng,
    option_count: int,
    correct_option_index: int,
) -> dict[str, Any]:
    """Construct the neutral non-wrapping marker-position progression dataset."""

    color = dict(rng.choice(COLOR_POOL))
    progression_mode, progression_axis, progression_direction = [
        str(value) for value in rng.choice(POSITION_PROGRESSIONS)
    ]
    progression_line_indices = [int(value) for value in rng.sample(range(3), 3)]

    def position_for(row_index: int, col_index: int) -> tuple[int, int]:
        if progression_axis == "row":
            mini_row = int(progression_line_indices[row_index])
            mini_col = int(col_index)
            if progression_direction == "right_to_left":
                mini_col = int(2 - mini_col)
            return mini_row, mini_col
        mini_row = int(row_index)
        mini_col = int(progression_line_indices[col_index])
        if progression_direction == "bottom_to_top":
            mini_row = int(2 - mini_row)
        return mini_row, mini_col

    panel_grid = [
        [
            pattern_panel_spec(cells=(position_for(row_index, col_index),), color=color)
            for col_index in range(3)
        ]
        for row_index in range(3)
    ]
    answer_panel_spec = dict(panel_grid[2][2])
    target_position = tuple(answer_panel_spec["cells"][0])
    distractors = [
        pattern_panel_spec(cells=((row_index, col_index),), color=color)
        for row_index in range(3)
        for col_index in range(3)
        if (int(row_index), int(col_index)) != (
            int(target_position[0]),
            int(target_position[1]),
        )
    ]
    rng.shuffle(distractors)
    option_specs, option_labels = build_option_specs(
        correct_panel_spec=answer_panel_spec,
        distractor_panel_specs=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    return {
        "matrix_rows": matrix_rows_from_specs(panel_grid),
        "matrix_panel_specs": [[dict(spec) for spec in row] for row in panel_grid],
        "answer_panel_spec": dict(answer_panel_spec),
        "correct_option_index": int(correct_option_index),
        "correct_option_panel_id": str(option_specs[correct_option_index]["option_panel_id"]),
        "answer_option_label": str(option_label_for_index(correct_option_index)),
        "option_specs": option_specs,
        "option_labels": option_labels,
        "option_count": int(option_count),
        "solver_trace": {
            "rule_type": "position_progression_matrix",
            "progression_axis": str(progression_axis),
            "progression_direction": str(progression_direction),
            "progression_mode": str(progression_mode),
            "progression_line_indices": [
                int(value) for value in progression_line_indices
            ],
            "position_table": [
                [
                    [
                        int(value)
                        for value in position_for(int(row_index), int(col_index))
                    ]
                    for col_index in range(3)
                ]
                for row_index in range(3)
            ],
            "target_row_index": 2,
            "target_col_index": 2,
            "answer_position": [int(target_position[0]), int(target_position[1])],
            "correct_option_index": int(correct_option_index),
            "correct_option_label": str(option_label_for_index(correct_option_index)),
        },
    }


__all__ = [
    "ANALOGICAL_TRANSFORMS",
    "COLOR_POOL",
    "SET_OPERATIONS",
    "SIZE_CYCLE",
    "ROTATION_OPS",
    "ROTATION_SEQUENCES",
    "FEATURE_BINDING_MODES",
    "POSITION_PROGRESSIONS",
    "build_analogical_transform_dataset",
    "build_count_progression_dataset",
    "build_feature_binding_dataset",
    "build_position_progression_dataset",
    "build_set_operation_dataset",
    "build_spatial_transform_dataset",
    "canonical_panel_spec",
]
