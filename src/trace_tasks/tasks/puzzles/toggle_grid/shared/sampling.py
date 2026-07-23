"""Sampling helpers for toggle-grid puzzle scenes."""

from __future__ import annotations

from string import ascii_uppercase
from typing import Any, Mapping

from trace_tasks.core.sampling import sample_without_replacement, uniform_choice

from .defaults import (
    resolve_grid_size,
    resolve_option_count,
    resolve_press_count,
    resolve_scene_variant,
)
from .rules import all_cells, apply_toggles, state_signature, toggle_once
from .state import GridState, ResultOption, SwitchOption, ToggleDataset


def random_state(*, rows: int, cols: int, rng) -> GridState:
    """Sample a non-degenerate binary grid state."""

    state = tuple(
        tuple(int(rng.randrange(2)) for _col in range(int(cols)))
        for _row in range(int(rows))
    )
    total_on = sum(sum(row) for row in state)
    if total_on in {0, int(rows) * int(cols)}:
        mutable = [list(row) for row in state]
        mutable[0][0] = 1 - int(mutable[0][0])
        state = tuple(tuple(row) for row in mutable)
    return state_signature(state)


def _result_options(
    *,
    target_state: GridState,
    option_count: int,
    rng,
) -> tuple[tuple[ResultOption, ...], str]:
    """Build unique visual result-grid options with one correct label."""

    rows = len(target_state)
    cols = len(target_state[0])
    labels = tuple(ascii_uppercase[index] for index in range(int(option_count)))
    correct_label = str(uniform_choice(rng, labels))
    distractor_cells = list(all_cells(rows, cols))
    rng.shuffle(distractor_cells)
    states: list[GridState] = [target_state]
    for cell in distractor_cells:
        mutated = toggle_once(target_state, cell)
        if mutated not in states:
            states.append(mutated)
        if len(states) >= int(option_count):
            break
    if len(states) < int(option_count):
        raise ValueError("not enough unique toggle result distractors")
    correct = states.pop(0)
    rng.shuffle(states)
    correct_index = labels.index(correct_label)
    states.insert(int(correct_index), correct)
    options = tuple(
        ResultOption(
            option_label=str(label),
            state=state,
            is_correct=state == target_state,
        )
        for label, state in zip(labels, states)
    )
    return options, str(correct_label)


def sample_result_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> ToggleDataset:
    """Construct one result-option task dataset.

    The invariant is one simulated target grid and one matching visual option;
    all other options are unique one-toggle distractor states.
    """

    scene_variant, _scene_probs = resolve_scene_variant(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )
    rows, cols, _row_range, _col_range = resolve_grid_size(
        params,
        generation_defaults=generation_defaults,
        rng=rng,
    )
    press_count, _press_range = resolve_press_count(
        params,
        generation_defaults=generation_defaults,
        rng=rng,
    )
    option_count = resolve_option_count(
        params,
        generation_defaults=generation_defaults,
    )
    start_state = random_state(rows=rows, cols=cols, rng=rng)
    pressed_cells = sample_without_replacement(
        rng,
        all_cells(rows, cols),
        int(press_count),
    )
    target_state = apply_toggles(start_state, tuple(pressed_cells))
    result_options, correct_label = _result_options(
        target_state=target_state,
        option_count=int(option_count),
        rng=rng,
    )
    return ToggleDataset(
        rows=int(rows),
        cols=int(cols),
        start_state=start_state,
        target_state=target_state,
        pressed_cells=tuple((int(row), int(col)) for row, col in pressed_cells),
        result_options=tuple(result_options),
        switch_options=tuple(),
        correct_option_label=str(correct_label),
        scene_variant=str(scene_variant),
        target_answer_support=tuple(
            ascii_uppercase[index] for index in range(int(option_count))
        ),
    )


def sample_repair_dataset(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rng,
) -> ToggleDataset:
    """Construct one single-switch repair task dataset."""

    scene_variant, _scene_probs = resolve_scene_variant(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )
    rows, cols, _row_range, _col_range = resolve_grid_size(
        params,
        generation_defaults=generation_defaults,
        rng=rng,
    )
    option_count = resolve_option_count(
        params,
        generation_defaults=generation_defaults,
    )
    start_state = random_state(rows=rows, cols=cols, rng=rng)
    cells = list(all_cells(rows, cols))
    correct_cell = tuple(uniform_choice(rng, tuple(cells)))
    target_state = toggle_once(start_state, correct_cell)
    candidates = [correct_cell]
    rng.shuffle(cells)
    for cell in cells:
        normalized = (int(cell[0]), int(cell[1]))
        if normalized not in candidates:
            candidates.append(normalized)
        if len(candidates) >= int(option_count):
            break
    if len(candidates) < int(option_count):
        raise ValueError("not enough toggle switch candidates")
    labels = tuple(ascii_uppercase[index] for index in range(int(option_count)))
    correct_label = str(uniform_choice(rng, labels))
    correct = candidates.pop(0)
    rng.shuffle(candidates)
    candidates.insert(labels.index(correct_label), correct)
    switch_options = tuple(
        SwitchOption(
            option_label=str(label),
            row=int(cell[0]),
            col=int(cell[1]),
            is_correct=tuple(cell) == tuple(correct_cell),
        )
        for label, cell in zip(labels, candidates)
    )
    return ToggleDataset(
        rows=int(rows),
        cols=int(cols),
        start_state=start_state,
        target_state=target_state,
        pressed_cells=(tuple(correct_cell),),
        result_options=tuple(),
        switch_options=tuple(switch_options),
        correct_option_label=str(correct_label),
        scene_variant=str(scene_variant),
        target_answer_support=tuple(labels),
    )
