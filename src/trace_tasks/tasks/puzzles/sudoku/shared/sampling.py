"""Scene-neutral Sudoku sampling helpers and axis resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.puzzles.shared.common import resolve_puzzle_axis_variant
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import (
    support_probability_map,
    uniform_choice_with_probabilities,
)
from trace_tasks.tasks.puzzles.shared.layout import resolve_puzzle_layout_jitter
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
    with_puzzle_unit_size_jitter,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.support_sampling import (
    resolve_integer_choice,
    resolve_integer_support,
)

from .state import (
    BOX_SIZE,
    DIGITS,
    SIZE,
    SUPPORTED_SUDOKU_SCENE_VARIANTS,
    SUPPORTED_SUDOKU_UNIT_TYPES,
    Board,
    Coord,
    SudokuSample,
)
from .rules import peer_coords
from .styles import SUPPORTED_SUDOKU_STYLE_VARIANTS


@dataclass(frozen=True)
class SudokuDefaults:
    """Stable fallback generation and rendering defaults for Sudoku grids."""

    marked_cell_value_support: tuple[int, ...] = tuple(DIGITS)
    marked_cell_candidate_count_support: tuple[int, ...] = (1, 2, 3, 4, 5)
    option_label_support: tuple[str, ...] = ("A", "B", "C", "D")
    sparse_min_visible_count: int = 18
    sparse_max_visible_count: int = 26
    filled_min_visible_count: int = 28
    filled_max_visible_count: int = 42
    canvas_width: int = 900
    canvas_height: int = 900
    panel_margin_px: int = 48
    max_board_size_px: int = 760
    board_border_width_px: int = 5
    grid_line_width_px: int = 2
    box_line_width_px: int = 5
    cell_padding_px: int = 5
    digit_font_size_px: int = 48
    marked_cell_outline_width_px: int = 7


@dataclass(frozen=True)
class SudokuAxes:
    """Resolved non-objective Sudoku scene axes."""

    scene_variant: str
    style_variant: str
    target_answer: int
    target_answer_support: tuple[int, ...]
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]
    target_answer_probabilities: dict[str, float]
    unit_type: str | None = None
    unit_type_probabilities: dict[str, float] | None = None


@dataclass(frozen=True)
class SudokuRenderParams:
    """Resolved render controls for one Sudoku scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_size_px: int
    board_border_width_px: int
    grid_line_width_px: int
    box_line_width_px: int
    cell_padding_px: int
    digit_font_size_px: int
    marked_cell_outline_width_px: int
    font_family: str = ""
    layout_jitter_meta: dict[str, Any] | None = None
    instance_seed: int = 0


@dataclass(frozen=True)
class MarkedPeerConstruction:
    """Intermediate board state for a marked cell constrained by peer digits."""

    board: Board
    filled_peer_coords: tuple[Coord, ...]
    visible_peer_coords: tuple[Coord, ...]
    visible_count: int


def freeze_board(board: Sequence[Sequence[int]]) -> Board:
    """Freeze a mutable 9 by 9 board into the canonical tuple form."""

    frozen = tuple(tuple(int(cell) for cell in row) for row in board)
    if len(frozen) != SIZE or any(len(row) != SIZE for row in frozen):
        raise ValueError("Sudoku board must be 9 by 9")
    return frozen


def mutable_empty_board() -> list[list[int]]:
    """Return one mutable empty Sudoku board."""

    return [[0 for _ in range(SIZE)] for _ in range(SIZE)]


def visible_cell_count(board: Board | Sequence[Sequence[int]]) -> int:
    """Return the number of non-empty cells on one board."""

    return sum(1 for row in board for cell in row if int(cell) != 0)


def _shuffled_groups(rng) -> list[int]:
    """Return a Sudoku-valid shuffled row/column ordering."""

    bands = list(range(BOX_SIZE))
    rng.shuffle(bands)
    order: list[int] = []
    for band in bands:
        offsets = list(range(BOX_SIZE))
        rng.shuffle(offsets)
        order.extend((int(band) * BOX_SIZE) + int(offset) for offset in offsets)
    return order


def build_sudoku_solution(rng) -> Board:
    """Build one fully solved Sudoku board by permuting a canonical pattern."""

    row_order = _shuffled_groups(rng)
    col_order = _shuffled_groups(rng)
    digits = list(DIGITS)
    rng.shuffle(digits)

    def pattern(row: int, col: int) -> int:
        return int((BOX_SIZE * (row % BOX_SIZE) + row // BOX_SIZE + col) % SIZE)

    return freeze_board(
        [[int(digits[pattern(row, col)]) for col in col_order] for row in row_order]
    )


def coords_with_solution_value(solution: Board, digit: int) -> tuple[Coord, ...]:
    """Return every coordinate whose solution digit equals `digit`."""

    target = int(digit)
    return tuple(
        (row, col)
        for row in range(SIZE)
        for col in range(SIZE)
        if int(solution[row][col]) == int(target)
    )


def add_random_solution_givens(
    *,
    rng,
    board: list[list[int]],
    solution: Board,
    excluded_coords: Sequence[Coord] | set[Coord],
    target_visible_count: int,
) -> None:
    """Fill random empty cells from the solution to the requested count."""

    excluded = {(int(row), int(col)) for row, col in excluded_coords}
    candidates = [
        (row, col)
        for row in range(SIZE)
        for col in range(SIZE)
        if (row, col) not in excluded and int(board[row][col]) == 0
    ]
    rng.shuffle(candidates)
    for row, col in candidates:
        if visible_cell_count(board) >= int(target_visible_count):
            break
        board[row][col] = int(solution[row][col])


def build_marked_peer_construction(
    *,
    rng,
    solution: Board,
    marked_cell: Coord,
    peer_digits: Sequence[int],
    scene_variant: str,
    defaults: SudokuDefaults,
    exclude_all_peers: bool,
) -> MarkedPeerConstruction:
    """Place requested peer digits, then add neutral solution givens."""

    board = mutable_empty_board()
    filled_peer_coords: set[Coord] = set()
    peers = peer_coords(marked_cell)
    for digit in sorted({int(value) for value in peer_digits}):
        candidates = [
            coord for coord in peers if int(solution[coord[0]][coord[1]]) == int(digit)
        ]
        if not candidates:
            raise ValueError(f"missing Sudoku peer witness for digit {digit}")
        coord = tuple(rng.choice(candidates))
        board[coord[0]][coord[1]] = int(digit)
        filled_peer_coords.add(coord)

    excluded = set(peers) if bool(exclude_all_peers) else set(filled_peer_coords)
    excluded.add(marked_cell)
    target_visible = target_visible_count(
        rng=rng,
        scene_variant=str(scene_variant),
        defaults=defaults,
        minimum_floor=len(filled_peer_coords),
    )
    add_random_solution_givens(
        rng=rng,
        board=board,
        solution=solution,
        excluded_coords=excluded,
        target_visible_count=int(target_visible),
    )
    frozen = freeze_board(board)
    visible_peers = tuple(
        coord for coord in peers if int(frozen[coord[0]][coord[1]]) != 0
    )
    return MarkedPeerConstruction(
        board=frozen,
        filled_peer_coords=tuple(sorted(filled_peer_coords)),
        visible_peer_coords=tuple(sorted(visible_peers)),
        visible_count=int(visible_cell_count(frozen)),
    )


def finalize_highlighted_unit_board(
    *,
    rng,
    board: list[list[int]],
    solution: Board,
    unit: Sequence[Coord],
    scene_variant: str,
    defaults: SudokuDefaults,
) -> Board:
    """Add neutral solution givens while keeping the highlighted unit fixed."""

    target_visible = target_visible_count(
        rng=rng,
        scene_variant=str(scene_variant),
        defaults=defaults,
        minimum_floor=int(visible_cell_count(board)),
    )
    add_random_solution_givens(
        rng=rng,
        board=board,
        solution=solution,
        excluded_coords=unit,
        target_visible_count=int(target_visible),
    )
    return freeze_board(board)


def populate_unit_with_missing_digits(
    *,
    rng,
    solution: Board,
    unit: Sequence[Coord],
    target_count: int,
) -> tuple[list[list[int]], tuple[int, ...]]:
    """Return a mutable board whose unit omits target_count digit values."""

    missing_digits = tuple(sorted(rng.sample(list(DIGITS), k=int(target_count))))
    missing_digit_set = {int(digit) for digit in missing_digits}
    board = mutable_empty_board()
    for row, col in unit:
        value = int(solution[row][col])
        if value not in missing_digit_set:
            board[row][col] = int(value)
    return board, tuple(int(value) for value in missing_digits)


def populate_unit_with_repeated_digits(
    *,
    rng,
    unit: Sequence[Coord],
    target_count: int,
) -> tuple[list[list[int]], tuple[int, ...]]:
    """Return a mutable board whose unit repeats target_count digit values."""

    shuffled_unit = list(unit)
    rng.shuffle(shuffled_unit)
    repeated_digits = tuple(sorted(rng.sample(list(DIGITS), k=int(target_count))))
    board = mutable_empty_board()
    cursor = 0
    for digit in repeated_digits:
        for _ in range(2):
            row, col = shuffled_unit[cursor]
            cursor += 1
            board[row][col] = int(digit)

    remaining_cells = shuffled_unit[cursor:]
    singleton_digits = [
        digit for digit in DIGITS if int(digit) not in set(repeated_digits)
    ]
    rng.shuffle(singleton_digits)
    max_singletons = min(len(remaining_cells), len(singleton_digits))
    min_singletons = 0 if int(target_count) >= 4 else 1
    singleton_count = (
        int(rng.randint(int(min_singletons), int(max_singletons)))
        if int(max_singletons) > 0
        else 0
    )
    for index in range(singleton_count):
        row, col = remaining_cells[index]
        board[row][col] = int(singleton_digits[index])
    return board, tuple(int(value) for value in repeated_digits)


def make_sudoku_sample(
    *,
    board: Board,
    solution: Board,
    answer: int | str,
    annotation_coords: Sequence[Coord],
    construction_mode: str,
    marked_cell: Coord | None = None,
    highlighted_unit_type: str | None = None,
    highlighted_unit_index: int | None = None,
    repeated_digit_values: Sequence[int] = (),
    missing_digit_values: Sequence[int] = (),
    option_specs: Sequence[Mapping[str, Any]] = (),
    correct_option_label: str | None = None,
    target_digit: int | None = None,
) -> SudokuSample:
    """Build the canonical scene sample record from task-owned witnesses."""

    return SudokuSample(
        board=board,
        solution=solution,
        answer=int(answer) if isinstance(answer, int) else str(answer),
        annotation_coords=tuple(annotation_coords),
        marked_cell=marked_cell,
        highlighted_unit_type=(
            str(highlighted_unit_type) if highlighted_unit_type is not None else None
        ),
        highlighted_unit_index=(
            int(highlighted_unit_index) if highlighted_unit_index is not None else None
        ),
        repeated_digit_values=tuple(int(value) for value in repeated_digit_values),
        missing_digit_values=tuple(int(value) for value in missing_digit_values),
        option_specs=tuple(dict(spec) for spec in option_specs),
        correct_option_label=(
            str(correct_option_label) if correct_option_label is not None else None
        ),
        target_digit=int(target_digit) if target_digit is not None else None,
        visible_count=int(visible_cell_count(board)),
        construction_mode=str(construction_mode),
    )


def resolve_sudoku_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
) -> tuple[str, dict[str, float]]:
    """Resolve the sparse/filled board scene variant."""

    return resolve_puzzle_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SUDOKU_SCENE_VARIANTS,
        task_id=str(namespace_root),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def resolve_sudoku_unit_type(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
) -> tuple[str, dict[str, float]]:
    """Resolve a row/column/box unit type for highlighted-unit tasks."""

    return resolve_puzzle_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SUDOKU_UNIT_TYPES,
        task_id=str(namespace_root),
        explicit_key="unit_type",
        weights_key="unit_type_weights",
        balance_flag_key="balanced_unit_type_sampling",
        axis_namespace="unit_type",
    )


def resolve_sudoku_style_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
) -> tuple[str, dict[str, float]]:
    """Resolve one Sudoku board style variant."""

    return resolve_puzzle_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SUDOKU_STYLE_VARIANTS,
        task_id=str(namespace_root),
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        axis_namespace="style_variant",
    )


def resolve_sudoku_target_answer(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
    support_key: str,
    fallback_support: Sequence[int],
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve one task-owned integer target answer and its support."""

    answer, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=fallback_support,
        namespace=f"{namespace_root}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=fallback_support,
    )
    return int(answer), tuple(int(value) for value in support), dict(probabilities)


def resolve_sudoku_target_digit(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
    support_key: str,
    fallback_support: Sequence[int],
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve one target digit for an option-letter Sudoku task."""

    digit, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_digit",
        fallback_support=fallback_support,
        namespace=f"{namespace_root}.target_digit",
        balanced_flag_key="balanced_target_digit_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=fallback_support,
    )
    return int(digit), tuple(int(value) for value in support), dict(probabilities)


def resolve_sudoku_option_label_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    fallback: Sequence[str],
) -> tuple[str, ...]:
    """Resolve the explicit option-label support for Sudoku option tasks."""

    raw_support = params.get(
        "option_label_support",
        group_default(
            gen_defaults,
            "option_label_support",
            tuple(str(value) for value in fallback),
        ),
    )
    support: list[str] = []
    for raw_value in raw_support:
        value = str(raw_value).strip()
        if value and value not in support:
            support.append(value)
    if len(support) != 4:
        raise ValueError("Sudoku option tasks require exactly four option labels")
    return tuple(support)


def resolve_sudoku_answer_label(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
    fallback: Sequence[str],
) -> tuple[str, tuple[str, ...], dict[str, float]]:
    """Resolve the correct option letter for Sudoku option tasks."""

    support = resolve_sudoku_option_label_support(
        params,
        gen_defaults=gen_defaults,
        fallback=fallback,
    )
    explicit = params.get("answer_label", params.get("target_answer"))
    if explicit is not None:
        selected = str(explicit).strip()
        if selected not in set(support):
            raise ValueError(f"unsupported Sudoku answer_label: {selected}")
        return selected, support, support_probability_map(support, selected=selected)

    rng = spawn_rng(int(instance_seed), f"{namespace_root}.answer_label")
    selected, probabilities = uniform_choice_with_probabilities(rng, support)
    return str(selected), support, dict(probabilities)


def resolve_sudoku_axes(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
    support_key: str,
    fallback_support: Sequence[int],
    include_unit_type: bool = False,
) -> SudokuAxes:
    """Resolve scene/style/target axes and optional highlighted-unit type."""

    scene_variant, scene_probs = resolve_sudoku_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(namespace_root),
    )
    style_variant, style_probs = resolve_sudoku_style_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(namespace_root),
    )
    target_answer, support, answer_probs = resolve_sudoku_target_answer(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=str(namespace_root),
        support_key=str(support_key),
        fallback_support=fallback_support,
    )
    unit_type = None
    unit_probs = None
    if include_unit_type:
        unit_type, unit_probs = resolve_sudoku_unit_type(
            params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            namespace_root=str(namespace_root),
        )
    return SudokuAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        target_answer=int(target_answer),
        target_answer_support=tuple(int(value) for value in support),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
        target_answer_probabilities=dict(answer_probs),
        unit_type=str(unit_type) if unit_type is not None else None,
        unit_type_probabilities=dict(unit_probs) if unit_probs is not None else None,
    )


def visible_count_bounds(
    scene_variant: str, defaults: SudokuDefaults
) -> tuple[int, int]:
    """Return total visible-cell bounds for one Sudoku scene variant."""

    if str(scene_variant) == "filled_grid":
        return int(defaults.filled_min_visible_count), int(
            defaults.filled_max_visible_count
        )
    return int(defaults.sparse_min_visible_count), int(
        defaults.sparse_max_visible_count
    )


def target_visible_count(
    *,
    rng,
    scene_variant: str,
    defaults: SudokuDefaults,
    minimum_floor: int = 0,
) -> int:
    """Sample one visible-cell count while respecting required witnesses."""

    low, high = visible_count_bounds(str(scene_variant), defaults)
    lower = max(int(low), int(minimum_floor))
    upper = max(int(lower), int(high))
    return int(rng.randint(int(lower), int(upper)))


def resolve_sudoku_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    defaults: SudokuDefaults,
) -> SudokuRenderParams:
    """Resolve Sudoku rendering parameters from scene config and params."""

    unit_scale, unit_scale_meta = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.sudoku.unit_size",
    )
    layout_jitter = with_puzzle_unit_size_jitter(
        resolve_puzzle_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace="puzzles.sudoku.layout",
        ),
        unit_scale_meta,
    )
    max_board_size_px = scale_puzzle_px(
        params.get(
            "max_board_size_px",
            group_default(
                render_defaults,
                "max_board_size_px",
                defaults.max_board_size_px,
            ),
        ),
        unit_scale,
        min_px=380,
    )
    default_canvas_width = int(
        group_default(render_defaults, "canvas_width", defaults.canvas_width)
    )
    default_canvas_height = int(
        group_default(render_defaults, "canvas_height", defaults.canvas_height)
    )
    canvas_size = int(
        max(
            540,
            min(
                max(default_canvas_width, default_canvas_height),
                int(max_board_size_px) + 160,
            ),
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace="puzzles.sudoku.font_family",
        params=params,
    )
    return SudokuRenderParams(
        canvas_width=int(params.get("canvas_width", canvas_size)),
        canvas_height=int(params.get("canvas_height", canvas_size)),
        panel_margin_px=int(
            params.get(
                "panel_margin_px",
                group_default(
                    render_defaults,
                    "panel_margin_px",
                    defaults.panel_margin_px,
                ),
            )
        ),
        max_board_size_px=int(max_board_size_px),
        board_border_width_px=scale_puzzle_px(
            params.get(
                "board_border_width_px",
                group_default(
                    render_defaults,
                    "board_border_width_px",
                    defaults.board_border_width_px,
                ),
            ),
            unit_scale,
            min_px=2,
        ),
        grid_line_width_px=scale_puzzle_px(
            params.get(
                "grid_line_width_px",
                group_default(
                    render_defaults,
                    "grid_line_width_px",
                    defaults.grid_line_width_px,
                ),
            ),
            unit_scale,
            min_px=1,
        ),
        box_line_width_px=scale_puzzle_px(
            params.get(
                "box_line_width_px",
                group_default(
                    render_defaults,
                    "box_line_width_px",
                    defaults.box_line_width_px,
                ),
            ),
            unit_scale,
            min_px=2,
        ),
        cell_padding_px=scale_puzzle_px(
            params.get(
                "cell_padding_px",
                group_default(
                    render_defaults, "cell_padding_px", defaults.cell_padding_px
                ),
            ),
            unit_scale,
            min_px=3,
        ),
        digit_font_size_px=scale_puzzle_px(
            params.get(
                "digit_font_size_px",
                group_default(
                    render_defaults,
                    "digit_font_size_px",
                    defaults.digit_font_size_px,
                ),
            ),
            unit_scale,
            min_px=18,
        ),
        marked_cell_outline_width_px=scale_puzzle_px(
            params.get(
                "marked_cell_outline_width_px",
                group_default(
                    render_defaults,
                    "marked_cell_outline_width_px",
                    defaults.marked_cell_outline_width_px,
                ),
            ),
            unit_scale,
            min_px=3,
        ),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
        instance_seed=int(instance_seed),
    )


__all__ = [
    "MarkedPeerConstruction",
    "SudokuAxes",
    "SudokuDefaults",
    "SudokuRenderParams",
    "add_random_solution_givens",
    "build_sudoku_solution",
    "build_marked_peer_construction",
    "coords_with_solution_value",
    "finalize_highlighted_unit_board",
    "freeze_board",
    "make_sudoku_sample",
    "mutable_empty_board",
    "populate_unit_with_missing_digits",
    "populate_unit_with_repeated_digits",
    "resolve_sudoku_answer_label",
    "resolve_sudoku_axes",
    "resolve_sudoku_option_label_support",
    "resolve_sudoku_render_params",
    "resolve_sudoku_scene_variant",
    "resolve_sudoku_style_variant",
    "resolve_sudoku_target_answer",
    "resolve_sudoku_target_digit",
    "resolve_sudoku_unit_type",
    "target_visible_count",
    "visible_cell_count",
    "visible_count_bounds",
]
