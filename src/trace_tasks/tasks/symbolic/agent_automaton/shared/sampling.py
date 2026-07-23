"""Sampling primitives for symbolic agent automata."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .....core.sampling import uniform_choice
from ....shared.mcq import option_label_for_index
from ...shared.common import get_int_range as _get_range

from .rules import DIRECTIONS, pose_text, state_count_for_rule, simulate_agent
from .state import AgentGridOption, AgentPoseOption, AgentSimulationSample, AgentStepTrace


def sample_grid(rng, *, rows: int, cols: int, state_count: int, live_prob: float = 0.38) -> Tuple[Tuple[int, ...], ...]:
    """Sample an initial state grid."""

    values: List[Tuple[int, ...]] = []
    for _row in range(int(rows)):
        row_values: List[int] = []
        for _col in range(int(cols)):
            if int(state_count) == 2:
                row_values.append(1 if float(rng.random()) < float(live_prob) else 0)
            else:
                row_values.append(int(rng.randrange(int(state_count))))
        values.append(tuple(row_values))
    return tuple(values)


def build_agent_simulation(
    *,
    rng,
    rows: int,
    cols: int,
    steps: int,
    rule_variant: str,
) -> tuple[
    Tuple[Tuple[int, ...], ...],
    int,
    int,
    int,
    Tuple[Tuple[int, ...], ...],
    int,
    int,
    int,
    Tuple[AgentStepTrace, ...],
]:
    """Sample a start state and run the agent simulation."""

    state_count = state_count_for_rule(str(rule_variant))
    initial_grid = sample_grid(rng, rows=int(rows), cols=int(cols), state_count=int(state_count), live_prob=0.38)
    start_row = int(rng.randrange(int(rows)))
    start_col = int(rng.randrange(int(cols)))
    start_direction = int(uniform_choice(rng, tuple(range(len(DIRECTIONS))), sort_keys=True))
    final_grid, final_row, final_col, final_direction, traces = simulate_agent(
        initial_grid,
        start_row=int(start_row),
        start_col=int(start_col),
        start_direction=int(start_direction),
        steps=int(steps),
        rule_variant=str(rule_variant),
    )
    return (
        initial_grid,
        int(start_row),
        int(start_col),
        int(start_direction),
        final_grid,
        int(final_row),
        int(final_col),
        int(final_direction),
        tuple(traces),
    )


def sample_agent_run(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sample_scope: str,
    step_min_key: str,
    step_max_key: str,
    fallback_step_min: int,
    fallback_step_max: int,
    rule_variant: str,
) -> AgentSimulationSample:
    """Sample grid dimensions, start pose, and simulated agent trajectory."""

    rng = spawn_rng(int(instance_seed), str(sample_scope))
    rows_min, rows_max = _get_range(params, gen_defaults, min_key="agent_rows_min", max_key="agent_rows_max", fallback_min=6, fallback_max=9)
    cols_min, cols_max = _get_range(params, gen_defaults, min_key="agent_cols_min", max_key="agent_cols_max", fallback_min=6, fallback_max=9)
    steps_min, steps_max = _get_range(
        params,
        gen_defaults,
        min_key=str(step_min_key),
        max_key=str(step_max_key),
        fallback_min=int(fallback_step_min),
        fallback_max=int(fallback_step_max),
    )
    rows = int(rng.randint(int(rows_min), int(rows_max)))
    cols = int(rng.randint(int(cols_min), int(cols_max)))
    steps = int(params.get("steps", rng.randint(int(steps_min), int(steps_max))))
    (
        initial_grid,
        start_row,
        start_col,
        start_direction,
        final_grid,
        final_row,
        final_col,
        final_direction,
        traces,
    ) = build_agent_simulation(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        steps=int(steps),
        rule_variant=str(rule_variant),
    )
    return AgentSimulationSample(
        rows=int(rows),
        cols=int(cols),
        steps=int(steps),
        initial_grid=tuple(initial_grid),
        final_grid=tuple(final_grid),
        start_row=int(start_row),
        start_col=int(start_col),
        start_direction=int(start_direction),
        final_row=int(final_row),
        final_col=int(final_col),
        final_direction=int(final_direction),
        traces=tuple(traces),
    )


def choose_pose_options(
    *,
    rng,
    params: dict,
    instance_seed: int,
    namespace: str,
    rows: int,
    cols: int,
    final_row: int,
    final_col: int,
    final_direction: int,
    option_count: int,
) -> tuple[Tuple[AgentPoseOption, ...], str]:
    """Build unique final-pose options and return the correct label."""

    correct = (int(final_row), int(final_col), int(final_direction))
    options = {correct}
    while len(options) < int(option_count):
        kind = int(rng.randrange(4))
        if kind == 0:
            candidate = (int((final_row + rng.choice([-2, -1, 1, 2])) % rows), int(final_col), int(final_direction))
        elif kind == 1:
            candidate = (int(final_row), int((final_col + rng.choice([-2, -1, 1, 2])) % cols), int(final_direction))
        elif kind == 2:
            candidate = (int(final_row), int(final_col), int((final_direction + rng.choice([1, 2, 3])) % 4))
        else:
            candidate = (int(rng.randrange(rows)), int(rng.randrange(cols)), int(rng.randrange(4)))
        options.add(candidate)
    distractors = [option for option in options if option != correct]
    rng.shuffle(distractors)
    correct_index = int(instance_seed) % int(option_count)
    ordered: List[Tuple[int, int, int]] = []
    for index in range(int(option_count)):
        if index == correct_index:
            ordered.append(correct)
        else:
            ordered.append(distractors.pop())
    specs: List[AgentPoseOption] = []
    answer_label = "A"
    for index, (row, col, direction) in enumerate(ordered):
        label = option_label_for_index(index)
        is_correct = bool((row, col, direction) == correct)
        if is_correct:
            answer_label = str(label)
        specs.append(
            AgentPoseOption(
                option_id=f"option_{label}",
                label=str(label),
                row=int(row),
                col=int(col),
                direction=int(direction),
                pose_text=pose_text(int(row), int(col), int(direction)),
                is_correct=is_correct,
            )
        )
    return tuple(specs), str(answer_label)


def _grid_key(grid: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(value) for value in row) for row in grid)


def _mutated_grid(
    rng,
    *,
    grid: Sequence[Sequence[int]],
    state_count: int,
    mutation_count: int,
) -> Tuple[Tuple[int, ...], ...]:
    rows = len(grid)
    cols = len(grid[0])
    candidate = [list(int(value) for value in row) for row in grid]
    cells = [(row, col) for row in range(rows) for col in range(cols)]
    rng.shuffle(cells)
    for row, col in cells[: max(1, int(mutation_count))]:
        current = int(candidate[row][col])
        alternatives = [value for value in range(int(state_count)) if int(value) != current]
        candidate[row][col] = int(rng.choice(alternatives))
    return _grid_key(candidate)


def choose_grid_options(
    *,
    rng,
    params: dict,
    instance_seed: int,
    namespace: str,
    final_grid: Sequence[Sequence[int]],
    state_count: int,
    option_count: int,
) -> tuple[Tuple[AgentGridOption, ...], str]:
    """Build unique future-grid options and return the correct label."""

    correct = _grid_key(final_grid)
    options = {correct}
    rows = len(correct)
    cols = len(correct[0])
    max_mutations = max(1, min(3, int(rows) * int(cols)))
    attempts = 0
    while len(options) < int(option_count) and attempts < 300:
        attempts += 1
        mutation_count = 1 + (attempts % max_mutations)
        options.add(
            _mutated_grid(
                rng,
                grid=correct,
                state_count=int(state_count),
                mutation_count=int(mutation_count),
            )
        )
    if len(options) < int(option_count):
        for row in range(rows):
            for col in range(cols):
                for value in range(int(state_count)):
                    candidate = [list(values) for values in correct]
                    if int(candidate[row][col]) == int(value):
                        continue
                    candidate[row][col] = int(value)
                    options.add(_grid_key(candidate))
                    if len(options) >= int(option_count):
                        break
                if len(options) >= int(option_count):
                    break
            if len(options) >= int(option_count):
                break
    if len(options) < int(option_count):
        raise RuntimeError("could not build enough unique future-grid options")

    distractors = [option for option in options if option != correct]
    rng.shuffle(distractors)
    correct_index = int(instance_seed) % int(option_count)
    ordered: List[Tuple[Tuple[int, ...], ...]] = []
    for index in range(int(option_count)):
        if index == correct_index:
            ordered.append(correct)
        else:
            ordered.append(distractors.pop())

    specs: List[AgentGridOption] = []
    answer_label = "A"
    for index, grid in enumerate(ordered):
        label = option_label_for_index(index)
        is_correct = bool(grid == correct)
        if is_correct:
            answer_label = str(label)
        specs.append(
            AgentGridOption(
                option_id=f"option_{label}",
                label=str(label),
                grid=tuple(tuple(int(value) for value in row) for row in grid),
                is_correct=bool(is_correct),
            )
        )
    return tuple(specs), str(answer_label)
