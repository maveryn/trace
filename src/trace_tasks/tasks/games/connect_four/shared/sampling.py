"""Identity-free sampling helpers for Connect Four games tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.style import SUPPORTED_CONNECT_FOUR_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .rules import (
    Board,
    COLUMNS,
    EMPTY,
    RED,
    ROWS,
    YELLOW,
    Coord,
    board_dimensions,
    coord_to_cell_id,
    drop_disc,
    empty_board,
    has_connect_four,
    legal_drop_rows,
    occupied_cell_count,
    opponent,
    winning_drop_map,
)
from .defaults import (
    COLUMN_LABELS,
    DEFAULT_SAFE_BOARD_SIZE_VARIANTS,
    FALLBACK_GENERATION_DEFAULTS,
    SCENE_ID,
    SUPPORTED_BOARD_SIZE_VARIANTS,
    SUPPORTED_SAFE_BOARD_SIZE_VARIANTS,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_WINNING_MOVE_LABEL_THREAT_KINDS,
)
from .state import (
    ConnectFourColumnProfileSample,
    ConnectFourCountSample,
    ConnectFourEvaluation,
    ConnectFourLabelSample,
    ConnectFourSceneAxes,
)


_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def _generation_default(key: str) -> Any:
    return group_default(_GEN_DEFAULTS, str(key), FALLBACK_GENERATION_DEFAULTS[str(key)])


def _full_probability_map(supported: Sequence[str], probabilities: Mapping[str, float]) -> dict[str, float]:
    return {str(key): float(probabilities.get(str(key), 0.0)) for key in supported}


def resolve_scene_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported_values: Sequence[str],
    gen_defaults: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, float]]:
    """Resolve one scene-semantic axis without public task identity."""

    active_defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=active_defaults,
        supported_variants=tuple(str(value) for value in supported_values),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported_values),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), _full_probability_map(supported_values, probabilities)


def column_labels_for_columns(columns: int) -> tuple[str, ...]:
    """Return visible column labels for a Connect Four board width."""

    if int(columns) > len(COLUMN_LABELS):
        raise ValueError(f"Connect Four column labels only support up to {len(COLUMN_LABELS)} columns")
    return tuple(str(label) for label in COLUMN_LABELS[: int(columns)])


def _board_size_variant_to_dimensions(board_size_variant: str) -> tuple[int, int]:
    if str(board_size_variant) == "square_5x5":
        return int(_generation_default("square_5_board_rows")), int(_generation_default("square_5_board_columns"))
    if str(board_size_variant) == "square_6x6":
        return int(_generation_default("square_6_board_rows")), int(_generation_default("square_6_board_columns"))
    if str(board_size_variant) == "small_6x5":
        return int(_generation_default("small_board_rows")), int(_generation_default("small_board_columns"))
    if str(board_size_variant) == "standard_7x6":
        return int(_generation_default("standard_board_rows")), int(_generation_default("standard_board_columns"))
    raise ValueError(f"unsupported board_size_variant: {board_size_variant}")


def _normalize_board_size_variant(raw_value: Any) -> str:
    text = str(raw_value).strip().lower().replace(" ", "")
    aliases = {
        "7x6": "standard_7x6",
        "standard": "standard_7x6",
        "standard_7x6": "standard_7x6",
        "6x5": "small_6x5",
        "small": "small_6x5",
        "small_6x5": "small_6x5",
        "5x5": "square_5x5",
        "square5x5": "square_5x5",
        "square_5x5": "square_5x5",
        "6x6": "square_6x6",
        "square6x6": "square_6x6",
        "square_6x6": "square_6x6",
    }
    if text not in aliases:
        raise ValueError(f"unsupported board_size: {raw_value}")
    return str(aliases[text])


def resolve_connect_four_scene_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    safe_board_defaults: bool = False,
    target_answer: int | None = None,
    gen_defaults: Mapping[str, Any] | None = None,
    namespace_suffix: str = "",
) -> ConnectFourSceneAxes:
    """Resolve scene, board-size, and style axes for one Connect Four task."""

    active_defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    suffix = f".{namespace_suffix}" if str(namespace_suffix) else ""
    scene_variant, scene_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        namespace=f"games.connect_four.scene_variant{suffix}",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_values=SUPPORTED_SCENE_VARIANTS,
    )

    board_params = dict(params)
    explicit_board = board_params.get("board_size_variant") or board_params.get("board_size")
    explicit_safe_board = board_params.get("safe_board_size_variant")
    if explicit_board is not None and board_params.get("board_size_variant") is None:
        board_params["board_size_variant"] = _normalize_board_size_variant(explicit_board)
    if explicit_safe_board is not None:
        board_params["safe_board_size_variant"] = _normalize_board_size_variant(explicit_safe_board)

    if bool(safe_board_defaults):
        if board_params.get("safe_board_size_variant") is None and board_params.get("board_size_variant") is not None:
            board_params["safe_board_size_variant"] = board_params["board_size_variant"]
        if board_params.get("safe_board_size_variant_weights") is None:
            board_params["safe_board_size_variant_weights"] = group_default(
                active_defaults,
                "safe_board_size_variant_weights",
                {key: 1.0 for key in DEFAULT_SAFE_BOARD_SIZE_VARIANTS},
            )
        if board_params.get("balanced_safe_board_size_variant_sampling") is None:
            board_params["balanced_safe_board_size_variant_sampling"] = bool(
                group_default(active_defaults, "balanced_safe_board_size_variant_sampling", True)
            )
        if explicit_board is None and explicit_safe_board is None and target_answer is not None and int(target_answer) > 5:
            board_params["safe_board_size_variant_weights"] = {"square_6x6": 1.0}
        selected_board, board_probs = resolve_scene_axis(
            instance_seed=int(instance_seed),
            params=board_params,
            gen_defaults=active_defaults,
            namespace=f"games.connect_four.safe_board_size_variant{suffix}",
            explicit_key="safe_board_size_variant",
            weights_key="safe_board_size_variant_weights",
            balance_flag_key="balanced_safe_board_size_variant_sampling",
            supported_values=SUPPORTED_SAFE_BOARD_SIZE_VARIANTS,
        )
    else:
        selected_board, board_probs = resolve_scene_axis(
            instance_seed=int(instance_seed),
            params=board_params,
            gen_defaults=active_defaults,
            namespace=f"games.connect_four.board_size_variant{suffix}",
            explicit_key="board_size_variant",
            weights_key="board_size_variant_weights",
            balance_flag_key="balanced_board_size_variant_sampling",
            supported_values=SUPPORTED_BOARD_SIZE_VARIANTS,
        )
    rows, columns = _board_size_variant_to_dimensions(str(selected_board))
    if bool(safe_board_defaults) and target_answer is not None and int(target_answer) > int(columns):
        raise ValueError(f"safe_move_count target_answer={int(target_answer)} is infeasible for {selected_board}")

    style_variant, style_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=active_defaults,
        namespace=f"games.connect_four.style_variant{suffix}",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_values=SUPPORTED_CONNECT_FOUR_STYLE_VARIANTS,
    )
    return ConnectFourSceneAxes(
        scene_variant=str(scene_variant),
        board_size_variant=str(selected_board),
        board_rows=int(rows),
        board_columns=int(columns),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        board_size_variant_probabilities=dict(board_probs),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve task-owned integer answer support."""

    target_answer, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(v) for v in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(v) for v in fallback_support),
    )
    return int(target_answer), tuple(int(v) for v in support), dict(probabilities)


def _mutable_board(board: Board | None = None, *, rows: int = ROWS, columns: int = COLUMNS) -> list[list[int]]:
    if board is None:
        board = empty_board(rows=int(rows), columns=int(columns))
    return [list(int(cell) for cell in row) for row in board]


def _freeze_board(board: Sequence[Sequence[int]]) -> Board:
    return tuple(tuple(int(cell) for cell in row) for row in board)


def resolve_current_player(rng, *, params: Mapping[str, Any]) -> int:
    """Resolve the current player for one scene."""

    explicit = params.get("current_player")
    if explicit is None:
        return int(RED if int(rng.randrange(2)) == 0 else YELLOW)
    text = str(explicit).strip().lower()
    if text in {"red", "r", "1"}:
        return int(RED)
    if text in {"yellow", "y", "-1"}:
        return int(YELLOW)
    raise ValueError(f"unsupported current_player: {explicit}")


def safe_move_coords(board: Board, *, current_player: int) -> tuple[Coord, ...]:
    """Return every legal landing square that leaves the opponent without an immediate win."""

    coords: list[Coord] = []
    opposing_player = int(opponent(int(current_player)))
    for col in sorted(legal_drop_rows(board).keys()):
        next_board, landing_coord = drop_disc(board, int(current_player), int(col))
        if has_connect_four(next_board, int(current_player)):
            continue
        if not winning_drop_map(next_board, int(opposing_player)):
            coords.append(tuple(landing_coord))
    return tuple(sorted(tuple(coord) for coord in coords))


def evaluate_count_query(*, board: Board, current_player: int, count_mode: str) -> ConnectFourEvaluation:
    """Evaluate one count mode on one visible board."""

    current_wins = winning_drop_map(board, int(current_player))
    winning_coords = tuple(sorted(tuple(coord) for coord, _ in current_wins.values()))
    safe_coords = safe_move_coords(board, current_player=int(current_player))
    if str(count_mode) == "winning":
        annotation_coords = winning_coords
    elif str(count_mode) == "safe":
        annotation_coords = safe_coords
    else:
        raise ValueError(f"unsupported Connect Four count mode: {count_mode}")
    return ConnectFourEvaluation(
        answer=int(len(annotation_coords)),
        annotation_coords=tuple(tuple(coord) for coord in annotation_coords),
        annotation_entity_ids=tuple(coord_to_cell_id(coord) for coord in annotation_coords),
        winning_move_coords=winning_coords,
        safe_move_coords=safe_coords,
    )


def _construct_vertical_threat_base(
    *,
    rng,
    current_player: int,
    target_answer: int,
    rows: int,
    columns: int,
) -> tuple[Board, str]:
    if int(rows) < 4:
        raise ValueError("Connect Four immediate-win construction requires at least 4 rows")
    board = _mutable_board(rows=int(rows), columns=int(columns))
    candidate_columns = (0, 1, 4, 5) if int(columns) == 6 else tuple(range(0, int(columns), 2))
    if int(target_answer) > len(candidate_columns):
        raise ValueError("target answer exceeds the supported vertical-threat columns")
    selected_columns = [] if int(target_answer) == 0 else list(rng.sample(candidate_columns, k=int(target_answer)))
    for col in selected_columns:
        for row in range(int(rows) - 1, int(rows) - 4, -1):
            board[int(row)][int(col)] = int(current_player)
    return _freeze_board(board), "vertical_threats"


def _horizontal_segment_for_target(*, rng, target_column: int, columns: int) -> tuple[int, int]:
    start_min = max(0, int(target_column) - 3)
    start_max = min(int(target_column), int(columns) - 4)
    if int(start_min) > int(start_max):
        raise ValueError("failed to place a horizontal Connect Four threat segment")
    start = int(rng.randint(int(start_min), int(start_max)))
    return int(start), int(start + 3)


def _construct_winning_label_base(
    *,
    rng,
    current_player: int,
    target_column: int,
    rows: int,
    columns: int,
    threat_kind: str,
) -> tuple[Board, str]:
    if int(rows) < 4 or int(columns) < 4:
        raise ValueError("Connect Four winning-column labels require at least 4 rows and 4 columns")
    if not 0 <= int(target_column) < int(columns):
        raise ValueError("target_column is outside the board")
    board = _mutable_board(rows=int(rows), columns=int(columns))
    if str(threat_kind) == "vertical_threat":
        for row in range(int(rows) - 1, int(rows) - 4, -1):
            board[int(row)][int(target_column)] = int(current_player)
        return _freeze_board(board), "single_vertical_threat"
    if str(threat_kind) == "horizontal_threat":
        start, end = _horizontal_segment_for_target(rng=rng, target_column=int(target_column), columns=int(columns))
        bottom_row = int(rows) - 1
        for col in range(int(start), int(end) + 1):
            if int(col) != int(target_column):
                board[int(bottom_row)][int(col)] = int(current_player)
        return _freeze_board(board), "single_horizontal_threat"
    raise ValueError(f"unsupported winning_move_label_threat_kind: {threat_kind}")


def _occupancy_bounds(scene_variant: str, *, rows: int, columns: int) -> tuple[int, int]:
    if str(scene_variant) == "crowded_board":
        minimum = int(_generation_default("crowded_min_occupied_count"))
        maximum = int(_generation_default("crowded_max_occupied_count"))
    else:
        minimum = int(_generation_default("midgame_min_occupied_count"))
        maximum = int(_generation_default("midgame_max_occupied_count"))
    capacity = int(rows) * int(columns)
    return min(int(minimum), int(capacity) - 1), min(int(maximum), int(capacity) - 1)


def _safe_occupancy_bounds(scene_variant: str, *, rows: int, columns: int, gen_defaults: Mapping[str, Any]) -> tuple[int, int]:
    if str(scene_variant) == "crowded_board":
        minimum = int(group_default(gen_defaults, "safe_crowded_min_occupied_count", 16))
        maximum = int(group_default(gen_defaults, "safe_crowded_max_occupied_count", 24))
    else:
        minimum = int(group_default(gen_defaults, "safe_midgame_min_occupied_count", 8))
        maximum = int(group_default(gen_defaults, "safe_midgame_max_occupied_count", 16))
    capacity = int(rows) * int(columns)
    return min(int(minimum), int(capacity) - 1), min(int(maximum), int(capacity) - 1)


def _augment_board_density(
    *,
    rng,
    board: Board,
    current_player: int,
    count_mode: str,
    target_answer: int,
    scene_variant: str,
) -> Board:
    """Add legal filler discs while preserving the task's exact answer count."""

    rows, columns = board_dimensions(board)
    minimum_occupied, maximum_occupied = _occupancy_bounds(str(scene_variant), rows=int(rows), columns=int(columns))
    current_board = board
    if int(occupied_cell_count(current_board)) > int(maximum_occupied):
        raise ValueError("base board already exceeds the requested scene occupancy bound")
    if int(occupied_cell_count(current_board)) >= int(minimum_occupied):
        return current_board
    opposing_player = int(opponent(int(current_player)))
    for _ in range(320):
        if int(occupied_cell_count(current_board)) >= int(minimum_occupied):
            break
        candidate_columns = [int(col) for col in sorted(legal_drop_rows(current_board).keys())]
        if not candidate_columns:
            break
        success = False
        for _ in range(96):
            candidate_column = int(candidate_columns[int(rng.randrange(len(candidate_columns)))])
            filler_player = int(current_player) if float(rng.random()) < 0.35 else int(opposing_player)
            next_board, _ = drop_disc(current_board, int(filler_player), int(candidate_column))
            if int(occupied_cell_count(next_board)) > int(maximum_occupied):
                continue
            if has_connect_four(next_board, int(current_player)) or has_connect_four(next_board, int(opposing_player)):
                continue
            evaluation = evaluate_count_query(board=next_board, current_player=int(current_player), count_mode=str(count_mode))
            if int(evaluation.answer) != int(target_answer):
                continue
            current_board = next_board
            success = True
            break
        if not success:
            break
    if int(occupied_cell_count(current_board)) < int(minimum_occupied):
        raise ValueError("failed to densify Connect Four board into the requested scene range")
    return current_board


def _sample_column_heights(*, rng, occupied: int, rows: int, columns: int) -> list[int]:
    heights = [0] * int(columns)
    column_indices = list(range(int(columns)))
    rng.shuffle(column_indices)
    remaining = int(occupied)
    for index, col in enumerate(column_indices):
        remaining_columns = int(columns - index - 1)
        min_height = max(0, int(remaining - (remaining_columns * int(rows))))
        max_height = min(int(rows), int(remaining))
        height = int(rng.randint(min_height, max_height))
        heights[int(col)] = int(height)
        remaining -= int(height)
    return heights


def _random_gravity_board(
    *,
    rng,
    minimum_occupied: int,
    maximum_occupied: int,
    current_player: int,
    rows: int,
    columns: int,
    search_attempts: int = 1024,
) -> Board:
    """Sample a gravity-valid board with turn-count parity and no existing win."""

    feasible_occupied: list[int] = []
    for occupied in range(int(minimum_occupied), int(maximum_occupied) + 1):
        if occupied >= int(rows * columns):
            continue
        if int(current_player) == int(RED) and int(occupied) % 2 == 0:
            feasible_occupied.append(int(occupied))
        if int(current_player) != int(RED) and int(occupied) % 2 == 1:
            feasible_occupied.append(int(occupied))
    if not feasible_occupied:
        raise ValueError("no feasible occupied counts match the current-player parity")
    for _ in range(max(1, int(search_attempts))):
        occupied = int(feasible_occupied[int(rng.randrange(len(feasible_occupied)))])
        heights = _sample_column_heights(rng=rng, occupied=int(occupied), rows=int(rows), columns=int(columns))
        if max(heights) == 0:
            continue
        if int(current_player) == int(RED):
            red_count = yellow_count = int(occupied // 2)
        else:
            red_count = int((occupied + 1) // 2)
            yellow_count = int((occupied - 1) // 2)
        colors = [int(RED)] * int(red_count) + [int(YELLOW)] * int(yellow_count)
        rng.shuffle(colors)
        board = _mutable_board(rows=int(rows), columns=int(columns))
        color_index = 0
        for col in range(int(columns)):
            for offset in range(int(heights[col])):
                row = int(rows - 1 - offset)
                board[row][col] = int(colors[color_index])
                color_index += 1
        frozen = _freeze_board(board)
        if has_connect_four(frozen, int(RED)) or has_connect_four(frozen, int(YELLOW)):
            continue
        return frozen
    raise ValueError("failed to sample one gravity-consistent Connect Four board")


def _construct_safe_move_board(
    *,
    rng,
    current_player: int,
    target_answer: int,
    scene_variant: str,
    rows: int,
    columns: int,
    gen_defaults: Mapping[str, Any],
) -> tuple[Board, str]:
    minimum_occupied, maximum_occupied = _safe_occupancy_bounds(str(scene_variant), rows=int(rows), columns=int(columns), gen_defaults=gen_defaults)
    for _ in range(72):
        board = _random_gravity_board(
            rng=rng,
            minimum_occupied=int(minimum_occupied),
            maximum_occupied=int(maximum_occupied),
            current_player=int(current_player),
            rows=int(rows),
            columns=int(columns),
            search_attempts=48,
        )
        if winning_drop_map(board, int(current_player)):
            continue
        evaluation = evaluate_count_query(board=board, current_player=int(current_player), count_mode="safe")
        if int(evaluation.answer) != int(target_answer):
            continue
        return board, "safe_move_search"
    raise ValueError("failed to construct one safe-move Connect Four board")


def sample_count_scene(
    *,
    rng,
    axes: ConnectFourSceneAxes,
    params: Mapping[str, Any],
    count_mode: str,
    target_answer: int,
    gen_defaults: Mapping[str, Any],
) -> ConnectFourCountSample:
    """Construct one Connect Four scene for a count objective."""

    current_player = resolve_current_player(rng, params=params)
    if str(count_mode) == "winning":
        base_board, construction_mode = _construct_vertical_threat_base(
            rng=rng,
            current_player=int(current_player),
            target_answer=int(target_answer),
            rows=int(axes.board_rows),
            columns=int(axes.board_columns),
        )
        board = _augment_board_density(
            rng=rng,
            board=base_board,
            current_player=int(current_player),
            count_mode="winning",
            target_answer=int(target_answer),
            scene_variant=str(axes.scene_variant),
        )
    elif str(count_mode) == "safe":
        board, construction_mode = _construct_safe_move_board(
            rng=rng,
            current_player=int(current_player),
            target_answer=int(target_answer),
            scene_variant=str(axes.scene_variant),
            rows=int(axes.board_rows),
            columns=int(axes.board_columns),
            gen_defaults=gen_defaults,
        )
    else:
        raise ValueError(f"unsupported Connect Four count mode: {count_mode}")
    evaluation = evaluate_count_query(board=board, current_player=int(current_player), count_mode=str(count_mode))
    if int(evaluation.answer) != int(target_answer):
        raise ValueError("final Connect Four board does not match target answer")
    return ConnectFourCountSample(
        board=board,
        current_player=int(current_player),
        evaluation=evaluation,
        occupied_count=int(occupied_cell_count(board)),
        construction_mode=str(construction_mode),
        scene_variant=str(axes.scene_variant),
        board_size_variant=str(axes.board_size_variant),
        board_rows=int(axes.board_rows),
        board_columns=int(axes.board_columns),
        style_variant=str(axes.style_variant),
    )


def resolve_winning_label_threat_kind(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    """Resolve the constructed immediate-win pattern for the label task."""

    return resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="games.connect_four.winning_move_column_label.threat_kind",
        explicit_key="winning_move_label_threat_kind",
        weights_key="winning_move_label_threat_kind_weights",
        balance_flag_key="balanced_winning_move_label_threat_kind_sampling",
        supported_values=SUPPORTED_WINNING_MOVE_LABEL_THREAT_KINDS,
    )


def resolve_blocking_label_threat_kind(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[str, dict[str, float]]:
    """Resolve the opponent threat pattern for the blocking-column label task."""

    return resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        namespace="games.connect_four.blocking_move_column_label.threat_kind",
        explicit_key="blocking_move_label_threat_kind",
        weights_key="blocking_move_label_threat_kind_weights",
        balance_flag_key="balanced_blocking_move_label_threat_kind_sampling",
        supported_values=SUPPORTED_WINNING_MOVE_LABEL_THREAT_KINDS,
    )


def _target_column_for_label_task(*, rng, params: Mapping[str, Any], columns: int) -> tuple[int, str, tuple[str, ...]]:
    column_labels = column_labels_for_columns(int(columns))
    label_to_col = {str(label): int(index) for index, label in enumerate(column_labels)}
    explicit_label = params.get("target_column_label", params.get("answer_label"))
    if explicit_label is not None:
        label = str(explicit_label).strip().upper()
        if label not in label_to_col:
            raise ValueError(f"unsupported target_column_label={explicit_label!r} for {int(columns)} columns")
        return int(label_to_col[str(label)]), str(label), tuple(column_labels)
    explicit_column = params.get("target_column_index")
    if explicit_column is not None:
        column = int(explicit_column)
        if not 0 <= int(column) < int(columns):
            raise ValueError(f"target_column_index={int(column)} is outside a {int(columns)}-column board")
        return int(column), str(column_labels[int(column)]), tuple(column_labels)
    column = int(uniform_choice(rng, tuple(range(int(columns)))))
    return int(column), str(column_labels[int(column)]), tuple(column_labels)


def sample_blocking_column_label_scene(
    *,
    rng,
    axes: ConnectFourSceneAxes,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
) -> ConnectFourLabelSample:
    """Construct one scene with exactly one column that blocks the opponent's immediate win."""

    current_player = resolve_current_player(rng, params=params)
    opposing_player = int(opponent(int(current_player)))
    threat_kind, threat_kind_probabilities = resolve_blocking_label_threat_kind(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
    )
    target_column, answer_label, column_labels = _target_column_for_label_task(rng=rng, params=params, columns=int(axes.board_columns))
    base_board, construction_mode = _construct_winning_label_base(
        rng=rng,
        current_player=int(opposing_player),
        target_column=int(target_column),
        rows=int(axes.board_rows),
        columns=int(axes.board_columns),
        threat_kind=str(threat_kind),
    )
    board = _augment_board_density(
        rng=rng,
        board=base_board,
        current_player=int(opposing_player),
        count_mode="winning",
        target_answer=1,
        scene_variant=str(axes.scene_variant),
    )
    if winning_drop_map(board, int(current_player)):
        raise ValueError("blocking scene should not give the current player an immediate win")
    opponent_wins = winning_drop_map(board, int(opposing_player))
    if set(opponent_wins.keys()) != {int(target_column)}:
        raise ValueError("blocking scene must preserve exactly one opponent winning column")
    opponent_landing_coord, completed_lines = opponent_wins[int(target_column)]
    next_board, blocking_coord = drop_disc(board, int(current_player), int(target_column))
    if tuple(blocking_coord) != tuple(opponent_landing_coord):
        raise ValueError("blocking move must occupy the opponent's winning landing square")
    if has_connect_four(next_board, int(current_player)):
        raise ValueError("blocking move should not also be an immediate win")
    if winning_drop_map(next_board, int(opposing_player)):
        raise ValueError("blocking move must remove the opponent's immediate win")
    evaluation = ConnectFourEvaluation(
        answer=int(target_column),
        annotation_coords=(tuple(blocking_coord),),
        annotation_entity_ids=(coord_to_cell_id(tuple(blocking_coord)),),
        winning_move_coords=tuple(),
        safe_move_coords=tuple(),
    )
    return ConnectFourLabelSample(
        board=board,
        current_player=int(current_player),
        evaluation=evaluation,
        occupied_count=int(occupied_cell_count(board)),
        construction_mode=f"opponent_{str(construction_mode)}",
        scene_variant=str(axes.scene_variant),
        board_size_variant=str(axes.board_size_variant),
        board_rows=int(axes.board_rows),
        board_columns=int(axes.board_columns),
        style_variant=str(axes.style_variant),
        threat_kind=str(threat_kind),
        threat_kind_probabilities=dict(threat_kind_probabilities),
        column_labels=tuple(str(label) for label in column_labels),
        answer_label=str(answer_label),
        answer_column=int(target_column),
        winning_line_coords=tuple(tuple(coord) for coord in completed_lines[0]),
    )


def sample_winning_column_label_scene(
    *,
    rng,
    axes: ConnectFourSceneAxes,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
) -> ConnectFourLabelSample:
    """Construct one Connect Four scene with exactly one immediate winning column."""

    current_player = resolve_current_player(rng, params=params)
    threat_kind, threat_kind_probabilities = resolve_winning_label_threat_kind(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
    )
    target_column, answer_label, column_labels = _target_column_for_label_task(rng=rng, params=params, columns=int(axes.board_columns))
    base_board, construction_mode = _construct_winning_label_base(
        rng=rng,
        current_player=int(current_player),
        target_column=int(target_column),
        rows=int(axes.board_rows),
        columns=int(axes.board_columns),
        threat_kind=str(threat_kind),
    )
    board = _augment_board_density(
        rng=rng,
        board=base_board,
        current_player=int(current_player),
        count_mode="winning",
        target_answer=1,
        scene_variant=str(axes.scene_variant),
    )
    evaluation = evaluate_count_query(board=board, current_player=int(current_player), count_mode="winning")
    if int(evaluation.answer) != 1:
        raise ValueError("final Connect Four board does not have exactly one winning move")
    winning_map = winning_drop_map(board, int(current_player))
    if set(winning_map.keys()) != {int(target_column)}:
        raise ValueError("final Connect Four board did not preserve the target winning column")
    _landing_coord, completed_lines = winning_map[int(target_column)]
    if not completed_lines:
        raise ValueError("target winning column has no completed line")
    return ConnectFourLabelSample(
        board=board,
        current_player=int(current_player),
        evaluation=evaluation,
        occupied_count=int(occupied_cell_count(board)),
        construction_mode=str(construction_mode),
        scene_variant=str(axes.scene_variant),
        board_size_variant=str(axes.board_size_variant),
        board_rows=int(axes.board_rows),
        board_columns=int(axes.board_columns),
        style_variant=str(axes.style_variant),
        threat_kind=str(threat_kind),
        threat_kind_probabilities=dict(threat_kind_probabilities),
        column_labels=tuple(str(label) for label in column_labels),
        answer_label=str(answer_label),
        answer_column=int(target_column),
        winning_line_coords=tuple(tuple(coord) for coord in completed_lines[0]),
    )


def _column_disc_profile(board: Board, column: int) -> tuple[int, int]:
    """Return red and yellow disc counts for one board column."""

    red_count = 0
    yellow_count = 0
    for row in board:
        value = int(row[int(column)])
        if value == int(RED):
            red_count += 1
        elif value == int(YELLOW):
            yellow_count += 1
    return int(red_count), int(yellow_count)


def _column_disc_coords(board: Board, column: int) -> tuple[Coord, ...]:
    """Return occupied cell coordinates in one board column."""

    coords: list[Coord] = []
    for row_index, row in enumerate(board):
        if int(row[int(column)]) != int(EMPTY):
            coords.append((int(row_index), int(column)))
    return tuple(coords)


def _mixed_column_values(*, rng, red_count: int, yellow_count: int) -> list[int]:
    """Return one shuffled bottom-up stack with the requested color counts."""

    values = [int(RED)] * int(red_count) + [int(YELLOW)] * int(yellow_count)
    rng.shuffle(values)
    return [int(value) for value in values]


def _fill_column_profile(
    board: list[list[int]],
    *,
    column: int,
    rows: int,
    values_bottom_up: Sequence[int],
) -> None:
    """Write a gravity-valid bottom-up column stack into one mutable board."""

    if len(values_bottom_up) > int(rows):
        raise ValueError("column profile height exceeds the board row count")
    for offset, value in enumerate(values_bottom_up):
        board[int(rows) - 1 - int(offset)][int(column)] = int(value)


def _random_nonmatching_profile(
    *,
    rng,
    rows: int,
    target_red_count: int,
    target_yellow_count: int,
) -> tuple[int, int]:
    """Sample one column profile that does not equal the target profile."""

    max_height = min(int(rows), 5)
    for _ in range(64):
        height = int(rng.randint(0, int(max_height)))
        red_count = int(rng.randint(0, int(height))) if height > 0 else 0
        yellow_count = int(height - red_count)
        if (int(red_count), int(yellow_count)) != (int(target_red_count), int(target_yellow_count)):
            return int(red_count), int(yellow_count)
    if int(target_red_count) != 0:
        return 0, int(target_red_count + target_yellow_count)
    return int(target_red_count + target_yellow_count), 0


def sample_column_disc_profile_label_scene(
    *,
    rng,
    axes: ConnectFourSceneAxes,
    params: Mapping[str, Any],
    target_red_count: int,
    target_yellow_count: int,
) -> ConnectFourColumnProfileSample:
    """Construct a board with exactly one column matching a red/yellow count profile."""

    rows = int(axes.board_rows)
    columns = int(axes.board_columns)
    total_target_height = int(target_red_count) + int(target_yellow_count)
    if int(target_red_count) <= 0 or int(target_yellow_count) <= 0:
        raise ValueError("Connect Four column profile labels require both colors in the target column")
    if int(total_target_height) > int(rows):
        raise ValueError("target column profile exceeds the board row count")

    current_player = resolve_current_player(rng, params=params)
    target_column, answer_label, column_labels = _target_column_for_label_task(rng=rng, params=params, columns=int(columns))
    board = _mutable_board(rows=int(rows), columns=int(columns))
    for column in range(int(columns)):
        if int(column) == int(target_column):
            red_count, yellow_count = int(target_red_count), int(target_yellow_count)
        else:
            red_count, yellow_count = _random_nonmatching_profile(
                rng=rng,
                rows=int(rows),
                target_red_count=int(target_red_count),
                target_yellow_count=int(target_yellow_count),
            )
        values = _mixed_column_values(rng=rng, red_count=int(red_count), yellow_count=int(yellow_count))
        _fill_column_profile(board, column=int(column), rows=int(rows), values_bottom_up=values)

    frozen = _freeze_board(board)
    matching_columns = [
        int(column)
        for column in range(int(columns))
        if _column_disc_profile(frozen, int(column)) == (int(target_red_count), int(target_yellow_count))
    ]
    if matching_columns != [int(target_column)]:
        raise ValueError("failed to construct a unique matching Connect Four column profile")
    annotation_coords = _column_disc_coords(frozen, int(target_column))
    if not annotation_coords:
        raise ValueError("target column profile must have visible discs for annotation")
    evaluation = ConnectFourEvaluation(
        answer=int(target_column),
        annotation_coords=tuple(tuple(coord) for coord in annotation_coords),
        annotation_entity_ids=tuple(coord_to_cell_id(coord) for coord in annotation_coords),
        winning_move_coords=tuple(),
        safe_move_coords=tuple(),
    )
    return ConnectFourColumnProfileSample(
        board=frozen,
        current_player=int(current_player),
        evaluation=evaluation,
        occupied_count=int(occupied_cell_count(frozen)),
        construction_mode="unique_column_disc_profile",
        scene_variant=str(axes.scene_variant),
        board_size_variant=str(axes.board_size_variant),
        board_rows=int(axes.board_rows),
        board_columns=int(axes.board_columns),
        style_variant=str(axes.style_variant),
        column_labels=tuple(str(label) for label in column_labels),
        answer_label=str(answer_label),
        answer_column=int(target_column),
        target_red_count=int(target_red_count),
        target_yellow_count=int(target_yellow_count),
    )


__all__ = [
    "column_labels_for_columns",
    "evaluate_count_query",
    "resolve_connect_four_scene_axes",
    "resolve_current_player",
    "resolve_scene_axis",
    "resolve_target_answer",
    "resolve_blocking_label_threat_kind",
    "resolve_winning_label_threat_kind",
    "safe_move_coords",
    "sample_blocking_column_label_scene",
    "sample_column_disc_profile_label_scene",
    "sample_count_scene",
    "sample_winning_column_label_scene",
]
