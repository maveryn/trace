"""Rules and constants for symbolic agent automata."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .state import AgentStepTrace


SCENE_ID = "agent_automaton"
SCENE_VARIANTS: Tuple[str, ...] = ("clean_grid", "lab_panel", "notebook_grid")
RULE_VARIANTS: Tuple[str, ...] = ("binary_rule", "three_state_rule")
BOARD_STYLES: Tuple[str, ...] = (
    "classic_grid",
    "rounded_tiles",
    "inset_cells",
    "lab_matrix",
    "notebook_cells",
)
DIRECTIONS: Tuple[str, ...] = ("up", "right", "down", "left")
DIR_VEC: Tuple[Tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
AGENT_RGB: Tuple[int, int, int] = (218, 58, 74)


def state_count_for_rule(rule_variant: str) -> int:
    """Return the number of visible cell states for one rule variant."""

    return 3 if str(rule_variant) == "three_state_rule" else 2


def turn_offsets_for_rule(rule_variant: str) -> Tuple[int, ...]:
    """Return direction updates indexed by current cell state."""

    if str(rule_variant) == "three_state_rule":
        return (1, 0, -1)
    return (1, -1)


def state_label(*, state: int, state_count: int) -> str:
    """Human-readable state label used in traces."""

    if int(state_count) == 2:
        return "light" if int(state) == 0 else "colored state"
    return f"state {int(state)}"


def pose_text(row: int, col: int, direction: int) -> str:
    """Return compact text for one row/column/direction pose."""

    return f"r{int(row) + 1}, c{int(col) + 1}, {DIRECTIONS[int(direction)]}"


def simulate_agent(
    grid: Sequence[Sequence[int]],
    *,
    start_row: int,
    start_col: int,
    start_direction: int,
    steps: int,
    rule_variant: str,
) -> Tuple[Tuple[Tuple[int, ...], ...], int, int, int, Tuple[AgentStepTrace, ...]]:
    """Run the turning-agent rule and return the final grid, pose, and trace."""

    turn_offsets = turn_offsets_for_rule(str(rule_variant))
    state_count = len(turn_offsets)
    rows = len(grid)
    cols = len(grid[0])
    current = [list(row) for row in grid]
    row = int(start_row)
    col = int(start_col)
    direction = int(start_direction)
    traces: List[AgentStepTrace] = []
    for step in range(1, int(steps) + 1):
        state_before = int(current[row][col])
        direction = int((direction + int(turn_offsets[state_before])) % 4)
        state_after = int((state_before + 1) % state_count)
        current[row][col] = state_after
        traces.append(
            AgentStepTrace(
                step=int(step),
                row=int(row),
                col=int(col),
                direction=int(direction),
                state_before=int(state_before),
                state_after=int(state_after),
            )
        )
        dr, dc = DIR_VEC[direction]
        row = int((row + dr) % rows)
        col = int((col + dc) % cols)
    final_grid = tuple(tuple(int(value) for value in row_values) for row_values in current)
    return final_grid, int(row), int(col), int(direction), tuple(traces)


def visit_counts(traces: Sequence[AgentStepTrace]) -> Dict[Tuple[int, int], int]:
    """Return update counts by visited cell."""

    counts: Dict[Tuple[int, int], int] = {}
    for trace in traces:
        key = (int(trace.row), int(trace.col))
        counts[key] = int(counts.get(key, 0)) + 1
    return counts
