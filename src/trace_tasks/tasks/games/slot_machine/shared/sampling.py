"""Sampling primitives for slot-machine games tasks."""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import (
    PAYLINE_IDS,
    PAYTABLE_SCORE_VALUES,
    REEL_COUNT,
    ROW_COUNT,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
    SYMBOL_KEYS,
)
from .state import (
    PaytableEntry,
    SlotCell,
    SlotCompletionOption,
    SlotCompletionScene,
    SlotMachineAxes,
    SlotMachineScene,
    validate_slot_completion_scene,
    validate_slot_machine_scene,
    winning_payline_ids_for_grid,
)


def _uniform_probability(values: Sequence[str]) -> dict[str, float]:
    """Return a JSON-friendly uniform probability map for finite axes."""

    items = tuple(str(value) for value in values)
    return {str(value): 1.0 / float(len(items)) for value in items}


def _sample_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    values: Sequence[str],
    namespace: str,
) -> str:
    """Resolve a scene axis from explicit params, sample cursor, or RNG."""

    explicit = params.get(str(key))
    choices = tuple(str(value) for value in values)
    if explicit is not None:
        value = str(explicit)
        if value not in choices:
            raise ValueError(f"unsupported slot-machine {key}: {explicit}")
        return value
    weights = params.get(f"{key}_weights", group_default(defaults, f"{key}_weights", {}))
    rng = spawn_rng(int(instance_seed), str(namespace))
    parsed = [max(0.0, float(dict(weights or {}).get(str(value), 1.0))) for value in choices]
    if not sum(parsed):
        return str(uniform_choice(rng, choices))
    total = sum(parsed)
    threshold = rng.random() * total
    cursor_value = 0.0
    for value, weight in zip(choices, parsed):
        cursor_value += float(weight)
        if threshold <= cursor_value:
            return str(value)
    return str(choices[-1])


def resolve_slot_machine_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> SlotMachineAxes:
    """Resolve nonsemantic scene and style axes for a slot-machine instance."""

    scene_variant = _sample_axis(
        instance_seed=int(instance_seed),
        params=params,
        defaults=gen_defaults,
        key="scene_variant",
        values=SUPPORTED_SCENE_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.scene_variant",
    )
    style_variant = _sample_axis(
        instance_seed=int(instance_seed),
        params=params,
        defaults=gen_defaults,
        key="style_variant",
        values=SUPPORTED_STYLE_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.style_variant",
    )
    return SlotMachineAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=_uniform_probability(SUPPORTED_SCENE_VARIANTS),
        style_variant_probabilities=_uniform_probability(SUPPORTED_STYLE_VARIANTS),
    )


@lru_cache(maxsize=1)
def _patterns_by_winning_payline_count() -> dict[int, tuple[tuple[int, ...], ...]]:
    """Enumerate compact 3x3 symbol patterns by conceptual payline count."""

    patterns: dict[int, list[tuple[int, ...]]] = {count: [] for count in range(len(PAYLINE_IDS) + 1)}
    for values in product(range(3), repeat=REEL_COUNT * ROW_COUNT):
        grid = tuple(
            tuple(int(values[row * REEL_COUNT + col]) for col in range(REEL_COUNT))
            for row in range(ROW_COUNT)
        )
        count = len(winning_payline_ids_for_grid(grid))
        patterns[int(count)].append(tuple(int(value) for value in values))
    return {count: tuple(values) for count, values in patterns.items()}


def _grid_from_two_reels_and_option(
    base_pattern: Sequence[int],
    option_pattern: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    """Return a compact 3x3 symbol grid from first-two-reel and third-reel codes."""

    base = tuple(int(value) for value in base_pattern)
    option = tuple(int(value) for value in option_pattern)
    return tuple(
        (
            int(base[row * 2]),
            int(base[row * 2 + 1]),
            int(option[row]),
        )
        for row in range(ROW_COUNT)
    )


@lru_cache(maxsize=1)
def _reel_completion_cases() -> tuple[tuple[tuple[int, ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]], ...]:
    """Enumerate first-two-reel states with valid winning and non-winning options."""

    cases = []
    for base_pattern in product(range(3), repeat=ROW_COUNT * (REEL_COUNT - 1)):
        winning_options: list[tuple[int, ...]] = []
        nonwinning_options: list[tuple[int, ...]] = []
        for option_pattern in product(range(3), repeat=ROW_COUNT):
            grid = _grid_from_two_reels_and_option(base_pattern, option_pattern)
            winning_count = len(winning_payline_ids_for_grid(grid))
            if winning_count == 1:
                winning_options.append(tuple(int(value) for value in option_pattern))
            elif winning_count == 0:
                nonwinning_options.append(tuple(int(value) for value in option_pattern))
        if winning_options and len(nonwinning_options) >= 3:
            cases.append(
                (
                    tuple(int(value) for value in base_pattern),
                    tuple(winning_options),
                    tuple(nonwinning_options),
                )
            )
    return tuple(cases)


def sample_slot_machine_grid(
    *,
    rng: Any,
    axes: SlotMachineAxes,
    target_winning_count: int,
    paytable_entries: Sequence[PaytableEntry] = (),
) -> SlotMachineScene:
    """Sample a 3 x 3 reel window with exactly the requested winning paylines."""

    target = int(target_winning_count)
    if target < 0 or target > len(PAYLINE_IDS):
        raise ValueError("target_winning_count must be between 0 and 5")
    candidate_patterns = _patterns_by_winning_payline_count().get(target, ())
    if not candidate_patterns:
        raise ValueError(f"no slot-machine pattern can produce {target} winning paylines")
    pattern = tuple(int(value) for value in rng.choice(candidate_patterns))
    labels = tuple(sorted(set(pattern)))
    sampled_symbols = list(rng.sample(list(SYMBOL_KEYS), len(labels)))
    symbol_by_label = {int(label): str(symbol) for label, symbol in zip(labels, sampled_symbols)}
    cells: list[SlotCell] = []
    for row in range(ROW_COUNT):
        for col in range(REEL_COUNT):
            symbol = symbol_by_label[int(pattern[row * REEL_COUNT + col])]
            cells.append(SlotCell(row=int(row), col=int(col), symbol_key=str(symbol)))
    grid = tuple(
        tuple(symbol_by_label[int(pattern[row * REEL_COUNT + col])] for col in range(REEL_COUNT))
        for row in range(ROW_COUNT)
    )
    scene = SlotMachineScene(
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        cells=tuple(cells),
        winning_payline_ids=winning_payline_ids_for_grid(grid),
        paytable_entries=tuple(paytable_entries),
    )
    validate_slot_machine_scene(scene)
    return scene


def sample_slot_paytable(rng: Any) -> tuple[PaytableEntry, ...]:
    """Sample a complete visible paytable for all slot symbols."""

    values = list(rng.sample(list(PAYTABLE_SCORE_VALUES), len(SYMBOL_KEYS)))
    return tuple(
        PaytableEntry(symbol_key=str(symbol_key), score_value=int(score_value))
        for symbol_key, score_value in zip(SYMBOL_KEYS, values)
    )


def sample_slot_reel_completion_scene(
    *,
    rng: Any,
    axes: SlotMachineAxes,
    option_labels: Sequence[str] = ("A", "B", "C", "D"),
) -> SlotCompletionScene:
    """Sample two fixed reels and four third-reel options with one winning choice."""

    labels = tuple(str(label) for label in option_labels)
    if len(labels) != 4 or len(set(labels)) != 4:
        raise ValueError("slot reel completion requires exactly four unique option labels")
    cases = _reel_completion_cases()
    if not cases:
        raise ValueError("slot reel completion has no valid compact cases")
    base_pattern, winning_options, nonwinning_options = rng.choice(cases)
    correct_option_pattern = tuple(int(value) for value in rng.choice(winning_options))
    distractor_patterns = [tuple(int(value) for value in pattern) for pattern in rng.sample(list(nonwinning_options), 3)]
    pattern_items = [correct_option_pattern, *distractor_patterns]
    rng.shuffle(pattern_items)
    sampled_symbols = list(rng.sample(list(SYMBOL_KEYS), 3))
    symbol_by_code = {code: str(symbol) for code, symbol in enumerate(sampled_symbols)}
    base_cells = tuple(
        SlotCell(
            row=int(row),
            col=int(col),
            symbol_key=symbol_by_code[int(base_pattern[row * 2 + col])],
        )
        for row in range(ROW_COUNT)
        for col in range(REEL_COUNT - 1)
    )
    options: list[SlotCompletionOption] = []
    answer_label = ""
    answer_completed_paylines: tuple[str, ...] = ()
    for label, option_pattern in zip(labels, pattern_items):
        cells = tuple(
            SlotCell(
                row=int(row),
                col=REEL_COUNT - 1,
                symbol_key=symbol_by_code[int(option_pattern[row])],
            )
            for row in range(ROW_COUNT)
        )
        grid = _grid_from_two_reels_and_option(base_pattern, option_pattern)
        completed_paylines = winning_payline_ids_for_grid(grid)
        if tuple(option_pattern) == correct_option_pattern:
            answer_label = str(label)
            answer_completed_paylines = tuple(str(payline_id) for payline_id in completed_paylines)
        options.append(
            SlotCompletionOption(
                label=str(label),
                cells=cells,
                completed_payline_ids=tuple(str(payline_id) for payline_id in completed_paylines),
            )
        )
    scene = SlotCompletionScene(
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        base_cells=base_cells,
        options=tuple(options),
        answer_label=str(answer_label),
        answer_completed_payline_ids=answer_completed_paylines,
    )
    validate_slot_completion_scene(scene)
    return scene


__all__ = [
    "resolve_slot_machine_axes",
    "sample_slot_reel_completion_scene",
    "sample_slot_paytable",
    "sample_slot_machine_grid",
]
