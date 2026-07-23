"""Passive state contracts for slot-machine games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from .defaults import PAYLINE_COORDS, PAYLINE_IDS, REEL_COUNT, ROW_COUNT, SUPPORTED_SCENE_VARIANTS, SUPPORTED_STYLE_VARIANTS, SYMBOL_KEYS


@dataclass(frozen=True)
class SlotCell:
    """One visible symbol cell in the reel window."""

    row: int
    col: int
    symbol_key: str


@dataclass(frozen=True)
class PaytableEntry:
    """One visible score value for a slot symbol."""

    symbol_key: str
    score_value: int


@dataclass(frozen=True)
class SlotMachineScene:
    """Generated slot-machine state before rendering."""

    scene_variant: str
    style_variant: str
    cells: Tuple[SlotCell, ...]
    winning_payline_ids: Tuple[str, ...]
    paytable_entries: Tuple[PaytableEntry, ...] = ()


@dataclass(frozen=True)
class SlotCompletionOption:
    """One candidate third reel for a reel-completion choice."""

    label: str
    cells: Tuple[SlotCell, ...]
    completed_payline_ids: Tuple[str, ...]


@dataclass(frozen=True)
class SlotCompletionScene:
    """Two fixed reels plus labeled candidate third reels."""

    scene_variant: str
    style_variant: str
    base_cells: Tuple[SlotCell, ...]
    options: Tuple[SlotCompletionOption, ...]
    answer_label: str
    answer_completed_payline_ids: Tuple[str, ...]


@dataclass(frozen=True)
class SlotMachineAxes:
    """Resolved nonsemantic slot-machine scene axes."""

    scene_variant: str
    style_variant: str
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]


def slot_cell_id(row: int, col: int) -> str:
    """Return the stable rendered entity id for one reel cell."""

    return f"cell_{int(row)}_{int(col)}"


PAYLINE_CELLS_BY_ID = {
    str(payline_id): tuple((int(row), int(col)) for row, col in coords)
    for payline_id, coords in zip(PAYLINE_IDS, PAYLINE_COORDS)
}


def payline_entity_id(payline_key: str) -> str:
    """Return the stable rendered entity id for one conceptual payline."""

    key = str(payline_key)
    if key not in PAYLINE_CELLS_BY_ID:
        raise ValueError(f"unsupported slot-machine payline: {key}")
    return f"payline_{key}"


def cell_grid(scene: SlotMachineScene) -> tuple[tuple[str, ...], ...]:
    """Return a row-major symbol grid for trace/debug output."""

    grid = [["" for _ in range(REEL_COUNT)] for _ in range(ROW_COUNT)]
    for cell in scene.cells:
        grid[int(cell.row)][int(cell.col)] = str(cell.symbol_key)
    return tuple(tuple(row) for row in grid)


def completion_option_grid(
    base_cells: Sequence[SlotCell],
    option_cells: Sequence[SlotCell],
) -> tuple[tuple[str, ...], ...]:
    """Return the conceptual 3x3 grid made by fixed reels plus one option."""

    grid = [["" for _ in range(REEL_COUNT)] for _ in range(ROW_COUNT)]
    for cell in tuple(base_cells) + tuple(option_cells):
        grid[int(cell.row)][int(cell.col)] = str(cell.symbol_key)
    if any(not value for row in grid for value in row):
        raise ValueError("slot completion grid must cover all 3 x 3 positions")
    return tuple(tuple(row) for row in grid)


def winning_payline_ids_for_grid(grid: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Return conceptual payline ids whose three symbols match."""

    winners: list[str] = []
    for payline_key in PAYLINE_IDS:
        cells = PAYLINE_CELLS_BY_ID[str(payline_key)]
        symbols = {str(grid[int(row)][int(col)]) for row, col in cells}
        if len(symbols) == 1:
            winners.append(str(payline_key))
    return tuple(winners)


def payline_symbol_key(grid: tuple[tuple[str, ...], ...], payline_key: str) -> str:
    """Return the matching symbol for a winning payline."""

    key = str(payline_key)
    cells = PAYLINE_CELLS_BY_ID[key]
    symbols = tuple(str(grid[int(row)][int(col)]) for row, col in cells)
    if len(set(symbols)) != 1:
        raise ValueError(f"payline is not winning: {key}")
    return str(symbols[0])


def paytable_score_map(scene: SlotMachineScene) -> dict[str, int]:
    """Return the scene paytable as a symbol-keyed score map."""

    return {str(entry.symbol_key): int(entry.score_value) for entry in scene.paytable_entries}


def winning_payline_score_details(scene: SlotMachineScene) -> tuple[dict[str, object], ...]:
    """Return per-winning-payline score details from the visible paytable."""

    grid = cell_grid(scene)
    score_by_symbol = paytable_score_map(scene)
    details: list[dict[str, object]] = []
    for payline_key in scene.winning_payline_ids:
        symbol_key = payline_symbol_key(grid, str(payline_key))
        if symbol_key not in score_by_symbol:
            raise ValueError(f"paytable has no score for winning symbol: {symbol_key}")
        details.append(
            {
                "payline_id": str(payline_key),
                "symbol_key": str(symbol_key),
                "score_value": int(score_by_symbol[symbol_key]),
            }
        )
    return tuple(details)


def total_winning_payline_score(scene: SlotMachineScene) -> int:
    """Return the total score over all winning paylines."""

    return sum(int(detail["score_value"]) for detail in winning_payline_score_details(scene))


def validate_slot_machine_scene(scene: SlotMachineScene) -> None:
    """Validate scene consistency independent of one public objective."""

    if scene.scene_variant not in SUPPORTED_SCENE_VARIANTS:
        raise ValueError(f"unsupported slot-machine scene variant: {scene.scene_variant}")
    if scene.style_variant not in SUPPORTED_STYLE_VARIANTS:
        raise ValueError(f"unsupported slot-machine style variant: {scene.style_variant}")
    if len(scene.cells) != REEL_COUNT * ROW_COUNT:
        raise ValueError("slot-machine scene must contain exactly 3 x 3 visible cells")
    seen = {(int(cell.row), int(cell.col)) for cell in scene.cells}
    expected = {(row, col) for row in range(ROW_COUNT) for col in range(REEL_COUNT)}
    if seen != expected:
        raise ValueError("slot-machine cells must cover every visible reel position exactly once")
    if any(str(cell.symbol_key) not in SYMBOL_KEYS for cell in scene.cells):
        raise ValueError("slot-machine cells use unsupported symbols")
    grid = cell_grid(scene)
    actual_winning_paylines = winning_payline_ids_for_grid(grid)
    if tuple(scene.winning_payline_ids) != actual_winning_paylines:
        raise ValueError("slot-machine winning paylines must match the symbol grid")
    if scene.paytable_entries:
        symbols = tuple(str(entry.symbol_key) for entry in scene.paytable_entries)
        if tuple(symbols) != tuple(SYMBOL_KEYS):
            raise ValueError("slot-machine paytable must list every supported symbol in stable order")
        if any(int(entry.score_value) <= 0 for entry in scene.paytable_entries):
            raise ValueError("slot-machine paytable scores must be positive integers")
        total_winning_payline_score(scene)


def validate_slot_completion_scene(scene: SlotCompletionScene) -> None:
    """Validate two-reel completion state independent of one task."""

    if scene.scene_variant not in SUPPORTED_SCENE_VARIANTS:
        raise ValueError(f"unsupported slot-machine scene variant: {scene.scene_variant}")
    if scene.style_variant not in SUPPORTED_STYLE_VARIANTS:
        raise ValueError(f"unsupported slot-machine style variant: {scene.style_variant}")
    base_positions = {(int(cell.row), int(cell.col)) for cell in scene.base_cells}
    expected_base = {(row, col) for row in range(ROW_COUNT) for col in range(REEL_COUNT - 1)}
    if base_positions != expected_base:
        raise ValueError("slot completion base must contain the first two reels exactly")
    if any(str(cell.symbol_key) not in SYMBOL_KEYS for cell in scene.base_cells):
        raise ValueError("slot completion base uses unsupported symbols")
    if len(scene.options) != 4:
        raise ValueError("slot completion scene must contain exactly four options")
    labels = tuple(str(option.label) for option in scene.options)
    if len(set(labels)) != len(labels):
        raise ValueError("slot completion option labels must be unique")
    if str(scene.answer_label) not in labels:
        raise ValueError("slot completion answer label must be one of the options")
    expected_option_positions = {(row, REEL_COUNT - 1) for row in range(ROW_COUNT)}
    for option in scene.options:
        option_positions = {(int(cell.row), int(cell.col)) for cell in option.cells}
        if option_positions != expected_option_positions:
            raise ValueError("each slot completion option must contain one complete third reel")
        if any(str(cell.symbol_key) not in SYMBOL_KEYS for cell in option.cells):
            raise ValueError("slot completion option uses unsupported symbols")
        grid = completion_option_grid(scene.base_cells, option.cells)
        actual = winning_payline_ids_for_grid(grid)
        if tuple(option.completed_payline_ids) != actual:
            raise ValueError("slot completion option payline ids must match its completed grid")
        if str(option.label) == str(scene.answer_label) and tuple(scene.answer_completed_payline_ids) != actual:
            raise ValueError("slot completion answer payline ids must match the answer option")


__all__ = [
    "PaytableEntry",
    "SlotCompletionOption",
    "SlotCompletionScene",
    "SlotCell",
    "SlotMachineAxes",
    "SlotMachineScene",
    "PAYLINE_CELLS_BY_ID",
    "cell_grid",
    "completion_option_grid",
    "payline_entity_id",
    "payline_symbol_key",
    "paytable_score_map",
    "slot_cell_id",
    "total_winning_payline_score",
    "validate_slot_completion_scene",
    "validate_slot_machine_scene",
    "winning_payline_score_details",
    "winning_payline_ids_for_grid",
]
