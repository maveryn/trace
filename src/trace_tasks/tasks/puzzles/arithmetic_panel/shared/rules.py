"""Pure arithmetic case constructors for the puzzle scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .sampling import answer_range_from_support, visible_value_bounds
from .state import ArithmeticCase


def _case(
    *,
    kind: str,
    layout_style: str,
    answer_value: int,
    answer_support: Sequence[int],
    data: Mapping[str, Any],
) -> ArithmeticCase:
    return ArithmeticCase(
        kind=str(kind),
        layout_style=str(layout_style),
        answer_value=int(answer_value),
        answer_support=tuple(int(value) for value in answer_support),
        answer_range=answer_range_from_support(answer_support),
        target_item_id="target",
        data=dict(data),
    )


def build_equal_sum_case(
    rng,
    *,
    answer_value: int,
    answer_support: Sequence[int],
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
) -> ArithmeticCase:
    """Build a polygon-side sum puzzle with one missing node value."""

    low, high = visible_value_bounds(params, gen_defaults)
    for _attempt in range(400):
        side_count = int(rng.randint(3, 5))
        target_side = int(rng.randrange(side_count))
        side_total_min = max(int(answer_value) + 2 * low, 3 * low)
        side_total_max = min(int(answer_value) + 2 * high, 3 * high)
        if side_total_min > side_total_max:
            continue
        side_total = int(rng.randint(side_total_min, side_total_max))
        target_corner_sum = int(side_total) - int(answer_value)
        target_left_min = max(low, target_corner_sum - high)
        target_left_max = min(high, target_corner_sum - low)
        if target_left_min > target_left_max:
            continue
        corners = [int(rng.randint(low, high)) for _ in range(side_count)]
        left_corner = int(rng.randint(target_left_min, target_left_max))
        right_corner = int(target_corner_sum) - int(left_corner)
        corners[target_side] = int(left_corner)
        corners[(target_side + 1) % side_count] = int(right_corner)
        mids: list[int | None] = []
        valid = True
        for index in range(side_count):
            if index == target_side:
                mids.append(None)
                continue
            value = (
                int(side_total)
                - int(corners[index])
                - int(corners[(index + 1) % side_count])
            )
            if not (low <= value <= high):
                valid = False
                break
            mids.append(int(value))
        if not valid:
            continue
        break
    else:
        raise ValueError("could not construct a valid equal-side-sum arithmetic case")
    return _case(
        kind="line_equal_sum",
        layout_style="polygon_side_sum",
        answer_value=int(answer_value),
        answer_support=answer_support,
        data={
            "side_count": side_count,
            "corner_values": corners,
            "middle_values": mids,
            "target_side": target_side,
            "side_total": side_total,
        },
    )


def build_vertical_arithmetic_case(
    rng, *, answer_value: int, answer_support: Sequence[int], operation: str
) -> ArithmeticCase:
    """Build a vertical addition or subtraction problem with one hidden digit."""

    width = int(rng.choice((2, 3)))
    place = int(rng.randrange(width))
    if str(operation) == "addition":
        a_digits = [
            int(rng.randint(1 if index == 0 else 0, 9)) for index in range(width)
        ]
        b_digits = [
            int(rng.randint(1 if index == 0 else 0, 9)) for index in range(width)
        ]
        a_digits[place] = int(answer_value)
        a = int("".join(str(value) for value in a_digits))
        b = int("".join(str(value) for value in b_digits))
        result = a + b
        rows = [
            ("addend", a_digits, place),
            ("addend", b_digits, None),
            ("result", [int(ch) for ch in str(result)], None),
        ]
    else:
        minuend = int(rng.randint(30, 900))
        subtrahend = int(rng.randint(10, max(10, minuend - 1)))
        result = minuend - subtrahend
        digits = [int(ch) for ch in str(minuend).zfill(width)]
        place = min(place, len(digits) - 1)
        digits[place] = int(answer_value)
        minuend = int("".join(str(value) for value in digits))
        result = max(0, minuend - subtrahend)
        rows = [
            ("top", digits, place),
            ("minus", [int(ch) for ch in str(subtrahend).zfill(len(digits))], None),
            ("result", [int(ch) for ch in str(result).zfill(len(digits))], None),
        ]
    return _case(
        kind=f"vertical_{operation}",
        layout_style="stacked_arithmetic",
        answer_value=int(answer_value),
        answer_support=answer_support,
        data={
            "operation": str(operation),
            "rows": rows,
            "number_width": width,
            "target_place": place,
        },
    )


def build_operation_table_case(
    rng, *, answer_value: int, answer_support: Sequence[int]
) -> ArithmeticCase:
    """Build a small operation table where the marked cell is computed from headers."""

    operator = str(rng.choice(("+", "x")))
    if operator == "x":
        left = max(1, min(12, int(answer_value)))
        top = 1
    else:
        left = int(rng.randint(1, max(1, int(answer_value))))
        top = int(answer_value) - left
    row_headers = [left, int(rng.randint(1, 12)), int(rng.randint(1, 12))]
    col_headers = [top, int(rng.randint(1, 12)), int(rng.randint(1, 12))]
    target_row = 0
    target_col = 0
    return _case(
        kind="operation_table",
        layout_style="operator_grid",
        answer_value=int(answer_value),
        answer_support=answer_support,
        data={
            "operator": operator,
            "row_headers": row_headers,
            "col_headers": col_headers,
            "target_row": target_row,
            "target_col": target_col,
        },
    )


def build_row_column_total_case(
    rng,
    *,
    answer_value: int,
    answer_support: Sequence[int],
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
) -> ArithmeticCase:
    """Build a row/column sum grid with one hidden interior value."""

    low, high = visible_value_bounds(params, gen_defaults)
    rows = int(rng.randint(3, 4))
    cols = int(rng.randint(3, 4))
    target_row = int(rng.randrange(rows))
    target_col = int(rng.randrange(cols))
    values = [[int(rng.randint(low, high)) for _ in range(cols)] for _ in range(rows)]
    values[target_row][target_col] = int(answer_value)
    return _case(
        kind="row_column_total",
        layout_style="sum_grid",
        answer_value=int(answer_value),
        answer_support=answer_support,
        data={
            "values": values,
            "target_row": target_row,
            "target_col": target_col,
            "row_totals": [sum(row) for row in values],
            "col_totals": [
                sum(values[row][col] for row in range(rows)) for col in range(cols)
            ],
        },
    )


def _number_wall_next_row(row: Sequence[int], operator: str) -> list[int]:
    """Apply one row transition for the selected wall rule."""

    if operator == "sum":
        return [int(row[index]) + int(row[index + 1]) for index in range(len(row) - 1)]
    if operator == "difference":
        return [
            abs(int(row[index]) - int(row[index + 1])) for index in range(len(row) - 1)
        ]
    return [int(row[index]) * int(row[index + 1]) for index in range(len(row) - 1)]


def _number_wall_levels(base: Sequence[int], operator: str) -> list[list[int]]:
    """Build every visible wall level from the bottom row."""

    levels = [[int(value) for value in base]]
    current = levels[0]
    while len(current) > 1:
        current = _number_wall_next_row(current, operator)
        levels.append(current)
    return levels


def _number_wall_candidate_count(
    *,
    levels: Sequence[Sequence[int]],
    target_level: int,
    target_index: int,
    answer_support: Sequence[int],
    operator: str,
) -> int:
    """Count support values that reproduce all visible non-target bricks."""

    if int(target_level) > 0:
        parent_value = _number_wall_next_row(
            levels[int(target_level) - 1],
            operator,
        )[int(target_index)]
        return sum(1 for candidate in answer_support if int(candidate) == parent_value)

    base_template = [int(value) for value in levels[0]]
    valid_count = 0
    for candidate in answer_support:
        base = list(base_template)
        base[int(target_index)] = int(candidate)
        candidate_levels = _number_wall_levels(base, operator)
        if len(candidate_levels) != len(levels):
            continue
        matched = True
        for level_index, level in enumerate(levels):
            for brick_index, value in enumerate(level):
                if level_index == 0 and brick_index == int(target_index):
                    continue
                if int(candidate_levels[level_index][brick_index]) != int(value):
                    matched = False
                    break
            if not matched:
                break
        if matched:
            valid_count += 1
    return valid_count


def _number_wall_candidate_positions(
    *,
    levels: Sequence[Sequence[int]],
    answer_value: int,
    answer_support: Sequence[int],
    operator: str,
) -> list[tuple[int, int]]:
    """Return every uniquely solvable brick position with the target value."""

    positions: list[tuple[int, int]] = []
    for level_index, level in enumerate(levels):
        for brick_index, value in enumerate(level):
            if int(value) != int(answer_value):
                continue
            if (
                _number_wall_candidate_count(
                    levels=levels,
                    target_level=level_index,
                    target_index=brick_index,
                    answer_support=answer_support,
                    operator=operator,
                )
                == 1
            ):
                positions.append((int(level_index), int(brick_index)))
    return positions


def _random_number_wall_base(rng, *, base_count: int, operator: str) -> list[int]:
    """Sample an ordinary base row before optional target placement."""

    if operator == "product":
        return [int(rng.choice((1, 2, 3))) for _ in range(int(base_count))]
    return [int(rng.randint(1, 9)) for _ in range(int(base_count))]


def _place_upper_number_wall_answer(
    rng,
    *,
    base: list[int],
    answer_value: int,
    operator: str,
) -> bool:
    """Edit one adjacent base pair so the brick above equals the answer."""

    if len(base) < 2:
        return False
    pair_index = int(rng.randrange(len(base) - 1))
    answer = int(answer_value)
    if operator == "sum":
        left_min = max(1, answer - 18)
        left_max = min(18, answer - 1)
        if left_min > left_max:
            return False
        left = int(rng.randint(left_min, left_max))
        right = int(answer - left)
    elif operator == "difference":
        if answer < 0 or answer > 17:
            return False
        low = int(rng.randint(1, 18 - answer))
        left, right = low + answer, low
        if float(rng.random()) < 0.5:
            left, right = right, left
    else:
        factors = [
            (factor, answer // factor)
            for factor in range(1, 19)
            if answer % factor == 0 and answer // factor <= 18
        ]
        if not factors:
            return False
        left, right = factors[int(rng.randrange(len(factors)))]
        if float(rng.random()) < 0.5:
            left, right = right, left

    base[pair_index] = int(left)
    base[pair_index + 1] = int(right)
    return True


def build_number_wall_case(
    rng, *, answer_value: int, answer_support: Sequence[int], wall_kind: str
) -> ArithmeticCase:
    """Build an addition number wall with one hidden brick."""

    if str(wall_kind) != "addition":
        raise ValueError(f"unsupported number-wall kind: {wall_kind!r}")
    operator = "sum"

    prefer_upper_target = float(rng.random()) < 0.75
    for attempt_index in range(800):
        base_count = int(rng.choice((3, 4)))
        base = _random_number_wall_base(
            rng,
            base_count=base_count,
            operator=operator,
        )
        placed_upper = prefer_upper_target and _place_upper_number_wall_answer(
            rng,
            base=base,
            answer_value=int(answer_value),
            operator=operator,
        )
        if prefer_upper_target and not placed_upper:
            continue
        if not prefer_upper_target and (attempt_index % 3) == 0:
            base[int(rng.randrange(base_count))] = int(answer_value)
        levels = _number_wall_levels(base, operator)
        candidates = _number_wall_candidate_positions(
            levels=levels,
            answer_value=int(answer_value),
            answer_support=answer_support,
            operator=operator,
        )
        if candidates:
            upper_candidates = [
                position for position in candidates if int(position[0]) > 0
            ]
            if prefer_upper_target and not upper_candidates:
                continue
            eligible = upper_candidates if prefer_upper_target else candidates
            target_level, target_index = eligible[int(rng.randrange(len(eligible)))]
            break
    else:
        raise ValueError("could not construct a unique number-wall arithmetic case")

    return _case(
        kind=f"number_wall_{wall_kind}",
        layout_style="stacked_wall",
        answer_value=int(answer_value),
        answer_support=answer_support,
        data={
            "levels": levels,
            "target_level": target_level,
            "target_index": target_index,
            "operator": operator,
        },
    )


__all__ = [
    "build_equal_sum_case",
    "build_number_wall_case",
    "build_operation_table_case",
    "build_row_column_total_case",
    "build_vertical_arithmetic_case",
]
