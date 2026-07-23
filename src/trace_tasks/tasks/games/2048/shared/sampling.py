"""Sampling helpers for the 2048 games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import SCENE_CONFIG_NAMESPACE
from .rules import board_key, line_coords, simulate_2048_move
from .state import (
    Board,
    EMPTY,
    SIZE,
    SUPPORTED_2048_DIRECTIONS,
    SUPPORTED_2048_RESULT_BOARD_LABELS,
    SUPPORTED_2048_SCENE_VARIANTS,
    SUPPORTED_2048_STYLE_VARIANTS,
    Move2048Result,
)

_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    "2048",
)


@dataclass(frozen=True)
class Resolved2048Axes:
    """Resolved semantic and visual axes for one 2048 instance."""

    scene_variant: str
    style_variant: str
    move_direction: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    move_direction_probabilities: Dict[str, float]


@dataclass(frozen=True)
class Resolved2048IntegerTarget:
    """Resolved integer target support for one public 2048 objective."""

    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class Resolved2048ResultBoardTarget:
    """Resolved target label and option-count axes for the result-board task."""

    target_label: str
    target_label_support: Tuple[str, ...]
    result_board_option_count: int
    result_board_option_count_support: Tuple[int, ...]
    target_label_probabilities: Dict[str, float]
    result_board_option_count_probabilities: Dict[str, float]


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
    gen_defaults: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_CONFIG_NAMESPACE}.{str(namespace)}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=[str(value) for value in supported],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=[str(value) for value in supported],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{SCENE_CONFIG_NAMESPACE}.{str(namespace)}",
    )
    return str(selected), dict(probabilities)


def _string_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[str],
) -> Tuple[str, ...]:
    raw = params.get(str(key), group_default(gen_defaults, str(key), tuple(fallback)))
    if raw is None:
        raw = tuple(fallback)
    values = (str(raw),) if isinstance(raw, str) else tuple(str(value) for value in raw)
    values = tuple(value for value in values if value)
    if not values:
        raise ValueError(f"{key} must contain at least one label")
    return values


def _resolve_label_choice(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[str],
    namespace: str,
    balanced_flag_key: str,
    gen_defaults: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    support = _string_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=fallback_support,
    )
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"{explicit_key}={value!r} is not in {support_key}")
        return value, {str(item): (1.0 if str(item) == value else 0.0) for item in support}
    probabilities = {str(item): 1.0 / float(len(support)) for item in support}
    del balanced_flag_key
    rng = spawn_rng(int(instance_seed), str(namespace))
    return str(uniform_choice(rng, tuple(support))), probabilities


def resolve_2048_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any] | None = None,
) -> Resolved2048Axes:
    """Resolve all semantic and visual axes for one 2048 instance."""

    defaults = dict(gen_defaults or _GEN_DEFAULTS)
    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_2048_SCENE_VARIANTS,
        gen_defaults=defaults,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_2048_STYLE_VARIANTS,
        gen_defaults=defaults,
    )
    move_direction, move_direction_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="move_direction",
        explicit_key="move_direction",
        weights_key="move_direction_weights",
        balance_flag_key="balanced_move_direction_sampling",
        supported=SUPPORTED_2048_DIRECTIONS,
        gen_defaults=defaults,
    )
    return Resolved2048Axes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        move_direction=str(move_direction),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        move_direction_probabilities=dict(move_direction_probabilities),
    )


def resolve_2048_integer_target(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> Resolved2048IntegerTarget:
    """Resolve an integer target for one public 2048 objective."""

    fallback = tuple(int(value) for value in fallback_support)
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=fallback,
    )
    target, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=fallback,
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return Resolved2048IntegerTarget(
        target_answer=int(target),
        target_answer_support=tuple(int(value) for value in support),
        target_answer_probabilities=dict(probabilities),
    )


def resolve_2048_result_board_target(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_labels: Sequence[str] = SUPPORTED_2048_RESULT_BOARD_LABELS,
    fallback_option_counts: Sequence[int] = (4, 6),
    namespace: str,
) -> Resolved2048ResultBoardTarget:
    """Resolve label and option-count axes for the result-board objective."""

    full_label_support = _string_support(
        params,
        gen_defaults=gen_defaults,
        key="result_board_label_support",
        fallback=fallback_labels,
    )
    option_fallback = tuple(int(value) for value in fallback_option_counts)
    option_count, option_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="result_board_option_count_support",
        explicit_key="result_board_option_count",
        fallback_support=option_fallback,
        namespace=f"{str(namespace)}.result_board_option_count",
        balanced_flag_key="balanced_result_board_option_count_sampling",
        namespace_support_permutation=True,
    )
    option_count_support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key="result_board_option_count_support",
        fallback=option_fallback,
    )
    if int(option_count) < 2:
        raise ValueError("result_board_option_count must be at least 2")
    if int(option_count) > len(full_label_support):
        raise ValueError("result_board_option_count exceeds result board label support")
    explicit_target_label = params.get("target_label")
    if explicit_target_label is not None and str(explicit_target_label) in full_label_support:
        required_count = int(full_label_support.index(str(explicit_target_label)) + 1)
        if required_count > int(option_count):
            if params.get("result_board_option_count") is not None:
                raise ValueError("target_label is outside the explicit result_board_option_count label range")
            feasible_counts = [
                int(value)
                for value in option_count_support
                if int(value) >= int(required_count)
            ]
            if not feasible_counts:
                raise ValueError("target_label is outside the configured result board label range")
            option_count = int(min(feasible_counts))
            option_probabilities = {
                str(value): (1.0 if int(value) == int(option_count) else 0.0)
                for value in option_count_support
            }
    target_label_support = tuple(str(label) for label in full_label_support[: int(option_count)])
    label_params = dict(params)
    label_params["result_board_label_support"] = list(target_label_support)
    target_label, label_probabilities = _resolve_label_choice(
        instance_seed=int(instance_seed),
        params=label_params,
        support_key="result_board_label_support",
        explicit_key="target_label",
        fallback_support=target_label_support,
        namespace=f"{str(namespace)}.target_label",
        balanced_flag_key="balanced_target_label_sampling",
        gen_defaults=gen_defaults,
    )
    return Resolved2048ResultBoardTarget(
        target_label=str(target_label),
        target_label_support=tuple(str(value) for value in target_label_support),
        result_board_option_count=int(option_count),
        result_board_option_count_support=tuple(int(value) for value in option_count_support),
        target_label_probabilities=dict(label_probabilities),
        result_board_option_count_probabilities=dict(option_probabilities),
    )


def board_from_move_order(lines: Sequence[Sequence[int]], *, direction: str) -> Board:
    """Build a board from rows/columns expressed in movement order."""

    if len(lines) != SIZE or any(len(line) != SIZE for line in lines):
        raise ValueError("2048 move-order lines must be 4 x 4")
    rows = [[EMPTY for _col in range(SIZE)] for _row in range(SIZE)]
    for line_index, line in enumerate(lines):
        for slot_index, value in enumerate(line):
            row, col = line_coords(int(line_index), str(direction))[int(slot_index)]
            rows[int(row)][int(col)] = int(value)
    return tuple(tuple(int(value) for value in row) for row in rows)


def score_decomposition(score: int) -> Tuple[int, ...]:
    """Return merged tile values that sum to a supported score."""

    target = int(score)
    if target == 0:
        return tuple()
    values: list[int] = []
    remaining = int(target)
    for value in (32, 16, 8, 4):
        while remaining >= int(value) and len(values) < 4:
            values.append(int(value))
            remaining -= int(value)
    if remaining != 0 or len(values) > 4:
        raise ValueError(f"unsupported 2048 score target: {score}")
    return tuple(values)


def _non_merging_line(rng, *, force_slide: bool = False, max_value: int = 64) -> Tuple[int, ...]:
    values = [2, 4, 8, 16, 32, 64, 128]
    values = [value for value in values if int(value) <= int(max_value)]
    fill_count = min(len(values), SIZE, int(rng.randint(2, SIZE)))
    filled = [int(value) for value in rng.sample(values, int(fill_count))]
    if bool(force_slide):
        line = [EMPTY] + list(filled[:3])
        while len(line) < SIZE:
            line.append(EMPTY)
        return tuple(int(value) for value in line[:SIZE])
    line = list(filled[:SIZE])
    while len(line) < SIZE:
        line.append(EMPTY)
    if rng.random() < 0.45:
        rng.shuffle(line)
    return tuple(int(value) for value in line)


def board_for_merge_values(
    *,
    rng,
    direction: str,
    merge_values: Sequence[int],
    max_clutter_value: int = 64,
    force_slide_when_no_merge: bool = False,
) -> Board:
    """Construct a board whose shown move creates the requested merge values."""

    line_values: list[list[int]] = [[] for _ in range(SIZE)]
    line_order = list(range(SIZE))
    rng.shuffle(line_order)
    for index, merged_value in enumerate(merge_values):
        line = line_values[line_order[int(index) % SIZE]]
        if len(line) > SIZE - 2:
            line = line_values[line_order[(int(index) + 1) % SIZE]]
        line.extend([int(merged_value) // 2, int(merged_value) // 2])

    lines: list[Tuple[int, ...]] = []
    used_slide_line = False
    for line in line_values:
        if line:
            padded = list(line[:SIZE])
            while len(padded) < SIZE:
                padded.append(EMPTY)
            if rng.random() < 0.40 and padded.count(EMPTY) > 0:
                zero_indices = [idx for idx, value in enumerate(padded) if int(value) == EMPTY]
                if zero_indices:
                    first_zero = int(zero_indices[0])
                    padded.insert(0, padded.pop(first_zero))
            lines.append(tuple(int(value) for value in padded[:SIZE]))
            continue
        force_slide = bool(force_slide_when_no_merge and not used_slide_line)
        lines.append(_non_merging_line(rng, force_slide=force_slide, max_value=int(max_clutter_value)))
        used_slide_line = bool(used_slide_line or force_slide)
    return board_from_move_order(lines, direction=str(direction))


def _try_add_unique_board(
    out: list[Board],
    seen: set[Tuple[Tuple[int, ...], ...]],
    board: Board,
) -> None:
    key = board_key(board)
    if key in seen:
        return
    seen.add(key)
    out.append(key)


def _mutated_result_board(*, rng, board: Board, mutation_index: int) -> Board:
    rows = [list(int(value) for value in row) for row in board]
    nonempty = [(row, col) for row in range(SIZE) for col in range(SIZE) if int(rows[row][col]) != EMPTY]
    empty = [(row, col) for row in range(SIZE) for col in range(SIZE) if int(rows[row][col]) == EMPTY]
    mode = int(mutation_index) % 5
    if mode == 0 and len(nonempty) >= 2:
        a, b = rng.sample(nonempty, 2)
        rows[a[0]][a[1]], rows[b[0]][b[1]] = rows[b[0]][b[1]], rows[a[0]][a[1]]
    elif mode == 1 and nonempty:
        row, col = rng.choice(nonempty)
        rows[row][col] = max(2, int(rows[row][col]) // 2)
    elif mode == 2 and nonempty:
        row, col = rng.choice(nonempty)
        rows[row][col] = min(512, int(rows[row][col]) * 2)
    elif mode == 3 and empty:
        row, col = rng.choice(empty)
        rows[row][col] = int(rng.choice((2, 4)))
    elif nonempty:
        row, col = rng.choice(nonempty)
        rows[row][col] = EMPTY
    return tuple(tuple(int(value) for value in row) for row in rows)


def build_result_board_options(
    *,
    rng,
    board: Board,
    result: Move2048Result,
    all_results: Mapping[str, Move2048Result],
    target_label: str,
    labels: Sequence[str],
) -> Dict[str, Board]:
    """Build labeled candidate post-move boards with one correct answer."""

    label_list = [str(label) for label in labels]
    if str(target_label) not in label_list:
        raise ValueError("target result-board label must be in label support")
    if len(label_list) < 2:
        raise ValueError("result-board option construction requires at least two labels")
    distractor_count = int(len(label_list) - 1)

    distractors: list[Board] = []
    seen = {board_key(result.after)}
    for direction in SUPPORTED_2048_DIRECTIONS:
        if str(direction) == str(result.direction):
            continue
        candidate = all_results[str(direction)].after
        _try_add_unique_board(distractors, seen, candidate)
    _try_add_unique_board(distractors, seen, board)
    for direction in SUPPORTED_2048_DIRECTIONS:
        candidate = simulate_2048_move(result.after, str(direction)).after
        _try_add_unique_board(distractors, seen, candidate)
    mutation_index = 0
    while len(distractors) < distractor_count and mutation_index < 80:
        candidate = _mutated_result_board(rng=rng, board=result.after, mutation_index=mutation_index)
        _try_add_unique_board(distractors, seen, candidate)
        mutation_index += 1
    if len(distractors) < distractor_count:
        raise ValueError("failed to build enough distinct 2048 result-board distractors")

    out: Dict[str, Board] = {}
    out[str(target_label)] = result.after
    distractor_iter = iter(distractors[:distractor_count])
    for label in label_list:
        if str(label) == str(target_label):
            continue
        out[str(label)] = next(distractor_iter)
    return {str(label): out[str(label)] for label in label_list}


__all__ = [
    "Resolved2048Axes",
    "Resolved2048IntegerTarget",
    "Resolved2048ResultBoardTarget",
    "board_for_merge_values",
    "board_from_move_order",
    "build_result_board_options",
    "resolve_2048_axes",
    "resolve_2048_integer_target",
    "resolve_2048_result_board_target",
    "score_decomposition",
]
