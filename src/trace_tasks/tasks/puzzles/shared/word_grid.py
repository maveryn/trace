"""Identity-free helpers for letter-grid puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from string import ascii_uppercase
from typing import Iterable, Sequence

Cell = tuple[int, int]

WORD_POOL: tuple[str, ...] = (
    "ABLE",
    "ACID",
    "AGED",
    "ALOE",
    "ARCH",
    "BARK",
    "BEAM",
    "BIRD",
    "BOLT",
    "CANE",
    "CAVE",
    "COLD",
    "COVE",
    "DART",
    "DIAL",
    "DUNE",
    "ECHO",
    "FERN",
    "FISH",
    "FORK",
    "GATE",
    "GLOW",
    "GOLD",
    "HARP",
    "HAZE",
    "HILL",
    "IRON",
    "IVY",
    "JADE",
    "KITE",
    "LACE",
    "LAKE",
    "LAMP",
    "LEAF",
    "LIME",
    "MARS",
    "MINT",
    "MOON",
    "NEST",
    "NODE",
    "NOVA",
    "OPAL",
    "ORCA",
    "PALM",
    "PEAR",
    "PINE",
    "POND",
    "RING",
    "ROCK",
    "ROOT",
    "SAGE",
    "SAND",
    "SEED",
    "SHIP",
    "SILK",
    "SNOW",
    "STAR",
    "TIDE",
    "TREE",
    "VALE",
    "VINE",
    "WAVE",
    "WIND",
    "WOLF",
    "YARN",
    "ZINC",
)

WORD_DIRECTIONS: tuple[tuple[str, int, int], ...] = (
    ("right", 0, 1),
    ("down", 1, 0),
    ("down-right", 1, 1),
    ("up-right", -1, 1),
    ("left", 0, -1),
    ("up", -1, 0),
    ("down-left", 1, -1),
    ("up-left", -1, -1),
)

DIRECTION_CODES: dict[str, str] = {
    "right": "R",
    "left": "L",
    "up": "U",
    "down": "D",
    "down-right": "DR",
    "up-right": "UR",
    "down-left": "DL",
    "up-left": "UL",
}

DIRECTION_LEGEND_LINES: tuple[str, ...] = (
    "R=right, L=left, U=up, D=down",
    "DR=down-right, UR=up-right",
    "DL=down-left, UL=up-left",
)


@dataclass(frozen=True)
class WordPlacement:
    """One contiguous word placement in a letter grid."""

    word: str
    row: int
    col: int
    direction: str
    dr: int
    dc: int
    cells: tuple[Cell, ...]


def cell_key(cell: Cell) -> str:
    """Return the stable render-map id for a grid cell."""

    return f"cell_{int(cell[0])}_{int(cell[1])}"


def coordinate_token(cell: Cell) -> str:
    """Return row-letter/column-number coordinate text for a zero-based cell."""

    row, col = int(cell[0]), int(cell[1])
    return f"{ascii_uppercase[row]}{col + 1}"


def option_key(label: str) -> str:
    """Return the stable render-map id for an option card."""

    return f"option_{str(label)}"


def word_chip_key(word: str) -> str:
    """Return the stable render-map id for one word-bank chip."""

    return f"word_{str(word)}"


def direction_code(direction: str) -> str:
    """Return the compact prompt-facing code for one word direction."""

    return DIRECTION_CODES[str(direction)]


def cells_for_word(
    row: int, col: int, dr: int, dc: int, length: int
) -> tuple[Cell, ...]:
    """Return the grid cells covered by a word starting at one cell."""

    return tuple(
        (int(row) + (index * int(dr)), int(col) + (index * int(dc)))
        for index in range(int(length))
    )


def fits_grid(rows: int, cols: int, cells: Iterable[Cell]) -> bool:
    """Return whether every cell lies inside a grid of the given size."""

    return all(
        0 <= int(row) < int(rows) and 0 <= int(col) < int(cols) for row, col in cells
    )


def choose_words(
    rng,
    *,
    count: int,
    min_len: int,
    max_len: int,
    pool: Sequence[str] = WORD_POOL,
) -> list[str]:
    """Sample unique uppercase words from a fixed readable word pool."""

    candidates = [
        str(word).upper()
        for word in pool
        if int(min_len) <= len(str(word)) <= int(max_len)
    ]
    if len(candidates) < int(count):
        raise RuntimeError("word pool too small for requested sample")
    rng.shuffle(candidates)
    return [str(word) for word in candidates[: int(count)]]


def fill_random_letters(
    grid: list[list[str]],
    rng,
    *,
    excluded_letters: set[str] | None = None,
) -> None:
    """Fill empty grid cells with random uppercase letters."""

    excluded = {str(letter).upper() for letter in (excluded_letters or set())}
    choices = [letter for letter in ascii_uppercase if letter not in excluded]
    if not choices:
        raise ValueError("random letter fill requires at least one allowed letter")
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if not grid[row][col]:
                grid[row][col] = str(choices[int(rng.randrange(len(choices)))])


def place_word(grid: list[list[str]], word: str, rng) -> WordPlacement:
    """Place one word into empty or matching cells and return its placement."""

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    directions = list(WORD_DIRECTIONS)
    for _attempt in range(250):
        direction, dr, dc = directions[int(rng.randrange(len(directions)))]
        row = int(rng.randrange(rows))
        col = int(rng.randrange(cols))
        cells = cells_for_word(row, col, int(dr), int(dc), len(str(word)))
        if not fits_grid(rows, cols, cells):
            continue
        if any(
            grid[rr][cc] not in {"", str(word)[index]}
            for index, (rr, cc) in enumerate(cells)
        ):
            continue
        for index, (rr, cc) in enumerate(cells):
            grid[rr][cc] = str(word)[index]
        return WordPlacement(
            word=str(word),
            row=int(row),
            col=int(col),
            direction=str(direction),
            dr=int(dr),
            dc=int(dc),
            cells=tuple(cells),
        )
    raise RuntimeError("failed to place word")


def scan_word(grid: Sequence[Sequence[str]], word: str) -> list[WordPlacement]:
    """Find every contiguous placement of a word in all supported directions."""

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    hits: list[WordPlacement] = []
    for row in range(rows):
        for col in range(cols):
            for direction, dr, dc in WORD_DIRECTIONS:
                cells = cells_for_word(row, col, int(dr), int(dc), len(str(word)))
                if not fits_grid(rows, cols, cells):
                    continue
                letters = "".join(str(grid[rr][cc]) for rr, cc in cells)
                if letters == str(word):
                    hits.append(
                        WordPlacement(
                            word=str(word),
                            row=int(row),
                            col=int(col),
                            direction=str(direction),
                            dr=int(dr),
                            dc=int(dc),
                            cells=tuple(cells),
                        )
                    )
    return hits


__all__ = [
    "Cell",
    "DIRECTION_LEGEND_LINES",
    "WORD_DIRECTIONS",
    "WORD_POOL",
    "WordPlacement",
    "cell_key",
    "choose_words",
    "coordinate_token",
    "direction_code",
    "fill_random_letters",
    "fits_grid",
    "option_key",
    "place_word",
    "scan_word",
    "word_chip_key",
]
