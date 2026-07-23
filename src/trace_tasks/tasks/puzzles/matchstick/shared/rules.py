"""Matchstick-number transition rules."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Tuple


DIGIT_SEGMENTS: Dict[int, frozenset[str]] = {
    0: frozenset(("a", "b", "c", "d", "e", "f")),
    1: frozenset(("b", "c")),
    2: frozenset(("a", "b", "g", "e", "d")),
    3: frozenset(("a", "b", "c", "d", "g")),
    4: frozenset(("f", "g", "b", "c")),
    5: frozenset(("a", "f", "g", "c", "d")),
    6: frozenset(("a", "f", "e", "d", "c", "g")),
    7: frozenset(("a", "b", "c")),
    8: frozenset(("a", "b", "c", "d", "e", "f", "g")),
    9: frozenset(("a", "b", "c", "d", "f", "g")),
}
SEGMENT_POINTS: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {
    "a": ((0.0, 0.0), (1.0, 0.0)),
    "b": ((1.0, 0.0), (1.0, 1.0)),
    "c": ((1.0, 1.0), (1.0, 2.0)),
    "d": ((0.0, 2.0), (1.0, 2.0)),
    "e": ((0.0, 1.0), (0.0, 2.0)),
    "f": ((0.0, 0.0), (0.0, 1.0)),
    "g": ((0.0, 1.0), (1.0, 1.0)),
}
SEGMENTS_TO_DIGIT: Dict[frozenset[str], int] = {
    frozenset(segments): int(digit) for digit, segments in DIGIT_SEGMENTS.items()
}


def number_digits(value: int) -> Tuple[int, int]:
    """Return the two seven-segment digits for one rendered number."""

    text = f"{int(value):02d}"
    return int(text[0]), int(text[1])


def number_text(value: int) -> str:
    """Format one rendered matchstick number as two digits."""

    return f"{int(value):02d}"


def number_segment_keys(value: int) -> frozenset[str]:
    """Return stable segment ids for all sticks in a two-digit number."""

    keys: List[str] = []
    for digit_index, digit in enumerate(number_digits(int(value))):
        for segment in sorted(DIGIT_SEGMENTS[int(digit)]):
            keys.append(f"digit{digit_index}:{segment}")
    return frozenset(keys)


def digit_segment_keys(digit: int, *, digit_index: int) -> frozenset[str]:
    """Return stable segment ids for one rendered equation digit."""

    return frozenset(
        f"digit{int(digit_index)}:{segment}"
        for segment in sorted(DIGIT_SEGMENTS[int(digit)])
    )


def digit_after_removing_segment(digit: int, segment: str) -> int | None:
    """Return the digit formed by removing one segment, if it is valid."""

    remaining = frozenset(DIGIT_SEGMENTS[int(digit)] - frozenset({str(segment)}))
    return SEGMENTS_TO_DIGIT.get(remaining)


def digit_source_candidates_for_removal(target_digit: int) -> tuple[tuple[int, str], ...]:
    """Return source digits that become target_digit after one stick removal."""

    target_segments = DIGIT_SEGMENTS[int(target_digit)]
    candidates: list[tuple[int, str]] = []
    for source_digit, source_segments in sorted(DIGIT_SEGMENTS.items()):
        added = source_segments - target_segments
        if int(source_digit) != int(target_digit) and len(added) == 1:
            candidates.append((int(source_digit), str(next(iter(added)))))
    return tuple(candidates)


def equation_text(digits: tuple[int, int, int], operator: str) -> str:
    """Format a single-digit matchstick equation."""

    left, right, result = (int(value) for value in digits)
    return f"{left} {str(operator)} {right} = {result}"


def equation_is_true(digits: tuple[int, int, int], operator: str) -> bool:
    """Return whether the visible single-digit equation is arithmetically true."""

    left, right, result = (int(value) for value in digits)
    if str(operator) == "+":
        return int(left + right) == int(result)
    if str(operator) == "-":
        return int(left - right) == int(result)
    raise ValueError(f"unsupported equation operator: {operator!r}")


def remove_equation_stick(
    digits: tuple[int, int, int],
    *,
    operator: str,
    stick_id: str,
) -> tuple[tuple[int, int, int] | None, bool]:
    """Remove one digit stick and report the resulting equation if valid."""

    prefix, segment = str(stick_id).split(":", 1)
    if not prefix.startswith("digit"):
        raise ValueError(f"unsupported equation stick id: {stick_id!r}")
    digit_index = int(prefix.removeprefix("digit"))
    updated = list(int(value) for value in digits)
    next_digit = digit_after_removing_segment(updated[int(digit_index)], str(segment))
    if next_digit is None:
        return None, False
    updated[int(digit_index)] = int(next_digit)
    result = tuple(int(value) for value in updated)
    return result, equation_is_true(result, str(operator))


def number_transition_allowed(
    source_number: int,
    target_number: int,
    *,
    stick_delta: int,
) -> bool:
    """Return whether target is reachable by adding or removing one stick."""

    source = number_segment_keys(int(source_number))
    target = number_segment_keys(int(target_number))
    removed = source - target
    added = target - source
    if int(stick_delta) == 1:
        return not removed and len(added) == 1
    if int(stick_delta) == -1:
        return len(removed) == 1 and not added
    raise ValueError(f"unsupported matchstick stick_delta: {stick_delta}")


def changed_digit_index(source_number: int, target_number: int) -> int:
    """Return which of the two digits changed, or -1 if neither changed."""

    source_digits = number_digits(int(source_number))
    target_digits = number_digits(int(target_number))
    changed = [
        index
        for index, (left, right) in enumerate(zip(source_digits, target_digits))
        if int(left) != int(right)
    ]
    return int(changed[0]) if changed else -1


def number_segments(value: int) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
    """Return drawable segment ids and normalized endpoints for one number."""

    segments: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
    gap = 0.42
    for digit_index, digit in enumerate(number_digits(int(value))):
        base_x = float(digit_index) * (1.0 + gap)
        for segment in sorted(DIGIT_SEGMENTS[int(digit)]):
            start, end = SEGMENT_POINTS[str(segment)]
            segments.append(
                (
                    f"digit{digit_index}:{segment}",
                    (base_x + start[0], start[1]),
                    (base_x + end[0], end[1]),
                )
            )
    return segments


def horizontal_edge_id(row: int, col: int) -> str:
    """Return the stable id for one horizontal lattice edge."""

    return f"h:{int(row)}:{int(col)}"


def vertical_edge_id(row: int, col: int) -> str:
    """Return the stable id for one vertical lattice edge."""

    return f"v:{int(row)}:{int(col)}"


def square_id(row: int, col: int) -> str:
    """Return the stable id for one unit lattice square."""

    return f"square:{int(row)}:{int(col)}"


def parse_square_id(cell_id: str) -> tuple[int, int]:
    """Parse one stable lattice square id."""

    prefix, row, col = str(cell_id).split(":", 2)
    if prefix != "square":
        raise ValueError(f"not a square id: {cell_id!r}")
    return int(row), int(col)


def lattice_edge_item_id(edge_id: str) -> str:
    """Return the render-map item id for one lattice edge."""

    return f"edge_{str(edge_id).replace(':', '_')}"


def lattice_square_item_id(cell_id: str) -> str:
    """Return the render-map item id for one lattice unit square."""

    row, col = parse_square_id(str(cell_id))
    return f"square_{int(row)}_{int(col)}"


def lattice_edges(rows: int, cols: int) -> tuple[str, ...]:
    """Return all possible unit-grid matchstick edge ids."""

    edges: list[str] = []
    for row in range(int(rows) + 1):
        for col in range(int(cols)):
            edges.append(horizontal_edge_id(row, col))
    for row in range(int(rows)):
        for col in range(int(cols) + 1):
            edges.append(vertical_edge_id(row, col))
    return tuple(edges)


def square_edges(row: int, col: int) -> frozenset[str]:
    """Return the four edge ids around one unit lattice square."""

    row_i = int(row)
    col_i = int(col)
    return frozenset(
        (
            horizontal_edge_id(row_i, col_i),
            horizontal_edge_id(row_i + 1, col_i),
            vertical_edge_id(row_i, col_i),
            vertical_edge_id(row_i, col_i + 1),
        )
    )


def completed_lattice_squares(
    present_edges: frozenset[str] | set[str] | tuple[str, ...] | list[str],
    *,
    rows: int,
    cols: int,
) -> tuple[str, ...]:
    """Return ids for all unit squares whose four matchstick edges are present."""

    present = frozenset(str(edge) for edge in present_edges)
    completed: list[str] = []
    for row in range(int(rows)):
        for col in range(int(cols)):
            if square_edges(row, col).issubset(present):
                completed.append(square_id(row, col))
    return tuple(completed)


def optimal_lattice_square_additions(
    present_edges: frozenset[str] | set[str] | tuple[str, ...] | list[str],
    *,
    rows: int,
    cols: int,
    add_count: int,
) -> dict[str, object]:
    """Enumerate add-count edge choices and summarize the best square outcome."""

    present = frozenset(str(edge) for edge in present_edges)
    all_edges = frozenset(lattice_edges(int(rows), int(cols)))
    missing_edges = tuple(sorted(all_edges - present))
    if int(add_count) < 0 or int(add_count) > len(missing_edges):
        raise ValueError("add_count must fit the number of missing lattice edges")

    best_count = -1
    best_added_sets: list[tuple[str, ...]] = []
    best_square_sets: set[tuple[str, ...]] = set()
    for added_edges in combinations(missing_edges, int(add_count)):
        final_edges = present | frozenset(str(edge) for edge in added_edges)
        completed = completed_lattice_squares(final_edges, rows=int(rows), cols=int(cols))
        square_count = len(completed)
        if square_count > best_count:
            best_count = int(square_count)
            best_added_sets = [tuple(str(edge) for edge in added_edges)]
            best_square_sets = {tuple(str(square) for square in completed)}
        elif square_count == best_count:
            best_added_sets.append(tuple(str(edge) for edge in added_edges))
            best_square_sets.add(tuple(str(square) for square in completed))

    return {
        "best_count": int(best_count),
        "best_added_sets": tuple(tuple(edge for edge in edges) for edges in best_added_sets),
        "best_square_sets": tuple(sorted(best_square_sets)),
    }


__all__ = [
    "DIGIT_SEGMENTS",
    "SEGMENTS_TO_DIGIT",
    "SEGMENT_POINTS",
    "changed_digit_index",
    "digit_after_removing_segment",
    "digit_segment_keys",
    "digit_source_candidates_for_removal",
    "equation_is_true",
    "equation_text",
    "remove_equation_stick",
    "completed_lattice_squares",
    "number_segment_keys",
    "number_segments",
    "number_text",
    "number_transition_allowed",
    "horizontal_edge_id",
    "lattice_edge_item_id",
    "lattice_edges",
    "lattice_square_item_id",
    "optimal_lattice_square_additions",
    "parse_square_id",
    "square_edges",
    "square_id",
    "vertical_edge_id",
]
