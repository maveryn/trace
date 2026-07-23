"""Sampling helpers for the Checkers games scene."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .rules import (
    BLACK,
    BOARD_SIZE,
    RED,
    Board,
    Coord,
    allowed_non_king_row,
    coord_to_cell_id,
    empty_board,
    enumerate_king_capture_chains,
    enumerate_legal_moves,
    freeze_board,
    occupied_piece_count,
    opponent,
    piece_to_entity_id,
    playable_coords,
)
from .defaults import FALLBACK_GENERATION_DEFAULTS
from .state import (
    SCENE_ID,
    SUPPORTED_CHECKERS_SCENE_VARIANTS,
    SUPPORTED_CHECKERS_STYLE_VARIANTS,
    CheckersEvaluation,
    ResolvedCheckersSceneAxes,
    SampledCheckersScene,
    TargetAnswerAxis,
)

_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def resolve_scene_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported_values: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one shared scene/style axis without public task identity."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=tuple(str(value) for value in supported_values),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=tuple(str(value) for value in supported_values),
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), dict(probabilities)


def resolve_checkers_scene_axes(instance_seed: int, *, params: Mapping[str, Any]) -> ResolvedCheckersSceneAxes:
    """Resolve shared scene and style axes."""

    scene_variant, scene_probs = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.checkers.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_values=SUPPORTED_CHECKERS_SCENE_VARIANTS,
    )
    style_variant, style_probs = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.checkers.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_values=SUPPORTED_CHECKERS_STYLE_VARIANTS,
    )
    return ResolvedCheckersSceneAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_probs),
    )


def resolve_checkers_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    gen_defaults: Mapping[str, Any] | None = None,
) -> TargetAnswerAxis:
    """Resolve a task-owned target answer from an integer support."""

    defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    answer, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    return TargetAnswerAxis(
        target_answer=int(answer),
        target_answer_support=tuple(int(value) for value in support),
        target_answer_probabilities=dict(probabilities),
    )


def resolve_current_player(rng, *, params: Mapping[str, Any]) -> int:
    """Resolve the current player for one scene."""

    explicit = params.get("current_player")
    if explicit is None:
        return int(RED if int(rng.randrange(2)) == 0 else BLACK)
    text = str(explicit).strip().lower()
    if text in {"red", "r", "1"}:
        return int(RED)
    if text in {"black", "b", "-1"}:
        return int(BLACK)
    raise ValueError(f"unsupported current_player: {explicit}")


def scene_object_description(scene_variant: str) -> str:
    """Return prompt-facing object text for one Checkers scene variant."""

    if str(scene_variant) == "crowded_board":
        return "a crowded 8 by 8 checkers board with many red and black pieces"
    return "an 8 by 8 checkers board with red and black pieces"


def movement_rule_text(current_player: int) -> str:
    """Return the prompt-facing forward-movement rule text."""

    if int(current_player) == int(RED):
        return "Red moves upward and Black moves downward."
    return "Black moves downward and Red moves upward."


def scene_occupied_range(scene_variant: str) -> Tuple[int, int]:
    """Return the target occupied-piece range for one scene family."""

    if str(scene_variant) == "crowded_board":
        return (
            int(FALLBACK_GENERATION_DEFAULTS.get("crowded_min_occupied_count", 13)),
            int(FALLBACK_GENERATION_DEFAULTS.get("crowded_max_occupied_count", 17)),
        )
    return (
        int(FALLBACK_GENERATION_DEFAULTS.get("midgame_min_occupied_count", 8)),
        int(FALLBACK_GENERATION_DEFAULTS.get("midgame_max_occupied_count", 12)),
    )


def resolve_task_occupied_range(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[int, int] | None:
    """Return an optional task-local occupied-piece range override."""

    raw_min = params.get("min_occupied_count", gen_defaults.get("min_occupied_count"))
    raw_max = params.get("max_occupied_count", gen_defaults.get("max_occupied_count"))
    if raw_min is None and raw_max is None:
        return None
    if raw_min is None or raw_max is None:
        raise ValueError("Checkers occupied-piece override requires both min and max values")
    min_occupied = int(raw_min)
    max_occupied = int(raw_max)
    if min_occupied < 0 or max_occupied < min_occupied:
        raise ValueError("invalid Checkers occupied-piece override")
    return min_occupied, max_occupied


def is_edge_coord(coord: Coord) -> bool:
    """Return true when a playable coordinate lies on the board perimeter."""

    row, col = int(coord[0]), int(coord[1])
    return row in {0, BOARD_SIZE - 1} or col in {0, BOARD_SIZE - 1}


def piece_state_candidate_coords(player: int, *, edge_only: bool) -> Tuple[Coord, ...]:
    """Return legal ordinary-piece coordinates that could satisfy a piece-state request."""

    coords = [
        coord
        for coord in playable_coords()
        if allowed_non_king_row(int(player), int(coord[0]))
    ]
    if bool(edge_only):
        coords = [coord for coord in coords if is_edge_coord(coord)]
    return tuple(coords)


def piece_state_target_coords(board: Sequence[Sequence[int]], *, player: int, edge_only: bool) -> Tuple[Coord, ...]:
    """Return target piece coordinates for one piece-state request."""

    coords: list[Coord] = []
    for coord in playable_coords():
        row, col = int(coord[0]), int(coord[1])
        if int(board[row][col]) != int(player):
            continue
        if bool(edge_only) and not is_edge_coord((row, col)):
            continue
        coords.append((row, col))
    return tuple(coords)


def _quiet_slots(player: int) -> Tuple[Tuple[Coord, Coord], ...]:
    """Return non-overlapping one-step edge move templates for one player."""

    if int(player) == int(RED):
        return (
            ((1, 0), (0, 1)),
            ((1, 6), (0, 7)),
            ((3, 0), (2, 1)),
            ((3, 6), (2, 7)),
            ((5, 0), (4, 1)),
            ((5, 6), (4, 7)),
        )
    return (
        ((0, 7), (1, 6)),
        ((1, 0), (2, 1)),
        ((2, 7), (3, 6)),
        ((3, 0), (4, 1)),
        ((4, 7), (5, 6)),
        ((5, 0), (6, 1)),
    )


def _capture_slots(player: int) -> Tuple[Tuple[Coord, Coord, Coord], ...]:
    """Return non-overlapping single-jump edge capture templates for one player."""

    if int(player) == int(RED):
        return (
            ((3, 0), (2, 1), (1, 2)),
            ((5, 0), (4, 1), (3, 2)),
            ((7, 0), (6, 1), (5, 2)),
            ((2, 7), (1, 6), (0, 5)),
            ((4, 7), (3, 6), (2, 5)),
            ((6, 7), (5, 6), (4, 5)),
        )
    return (
        ((0, 7), (1, 6), (2, 5)),
        ((1, 0), (2, 1), (3, 2)),
        ((2, 7), (3, 6), (4, 5)),
        ((3, 0), (4, 1), (5, 2)),
        ((4, 7), (5, 6), (6, 5)),
        ((5, 0), (6, 1), (7, 2)),
    )


_KING_CHAIN_TEMPLATE: Tuple[Coord, ...] = (
    (0, 1),
    (2, 3),
    (4, 1),
    (6, 3),
    (4, 5),
    (6, 7),
)


def _transform_king_chain_template(rng) -> Tuple[Coord, ...]:
    """Return one reflected king-capture template with five possible jumps."""

    flip_rows = bool(rng.randrange(2))
    flip_cols = bool(rng.randrange(2))
    coords: list[Coord] = []
    for row, col in _KING_CHAIN_TEMPLATE:
        out_row = BOARD_SIZE - 1 - int(row) if flip_rows else int(row)
        out_col = BOARD_SIZE - 1 - int(col) if flip_cols else int(col)
        coords.append((int(out_row), int(out_col)))
    return tuple(coords)


def evaluate_move_destinations(
    *,
    board: Board,
    current_player: int,
    capture_only: bool,
) -> CheckersEvaluation | None:
    """Evaluate landing-square count semantics for one finalized board."""

    legal_moves = tuple(enumerate_legal_moves(board, int(current_player)))
    capture_moves = tuple(move for move in legal_moves if move.captured is not None)
    relevant_moves = capture_moves if bool(capture_only) else legal_moves
    destinations = tuple((int(move.landing[0]), int(move.landing[1])) for move in relevant_moves)
    if len(set(destinations)) != len(destinations):
        return None
    annotation_coords = tuple(sorted(set(destinations)))
    return CheckersEvaluation(
        answer=int(len(relevant_moves)),
        legal_moves=legal_moves,
        capture_moves=capture_moves,
        annotation_coords=annotation_coords,
        annotation_entity_ids=tuple(coord_to_cell_id(coord) for coord in annotation_coords),
    )


def evaluate_piece_mobility(
    *,
    board: Board,
    current_player: int,
    capture_only: bool,
) -> CheckersEvaluation:
    """Evaluate source-piece mobility count semantics for one finalized board."""

    legal_moves = tuple(enumerate_legal_moves(board, int(current_player)))
    capture_moves = tuple(move for move in legal_moves if move.captured is not None)
    relevant_moves = capture_moves if bool(capture_only) else legal_moves
    source_coords = tuple(sorted({(int(move.origin[0]), int(move.origin[1])) for move in relevant_moves}))
    return CheckersEvaluation(
        answer=int(len(source_coords)),
        legal_moves=legal_moves,
        capture_moves=capture_moves,
        annotation_coords=source_coords,
        annotation_entity_ids=tuple(piece_to_entity_id(coord, player=int(current_player)) for coord in source_coords),
        annotation_kind="piece_point",
    )


def evaluate_piece_state(*, board: Board, player: int, edge_only: bool) -> CheckersEvaluation:
    """Evaluate visible piece-state counting semantics for one finalized board."""

    target_coords = piece_state_target_coords(board, player=int(player), edge_only=bool(edge_only))
    legal_moves = tuple(enumerate_legal_moves(board, int(player)))
    capture_moves = tuple(move for move in legal_moves if move.captured is not None)
    return CheckersEvaluation(
        answer=int(len(target_coords)),
        legal_moves=legal_moves,
        capture_moves=capture_moves,
        annotation_coords=target_coords,
        annotation_entity_ids=tuple(piece_to_entity_id(coord, player=int(player)) for coord in target_coords),
        annotation_kind="piece_point",
    )


def evaluate_king_capture_chain(
    *,
    board: Board,
    current_player: int,
    marked_coord: Coord,
) -> CheckersEvaluation | None:
    """Evaluate longest marked-king capture-chain semantics."""

    marked = (int(marked_coord[0]), int(marked_coord[1]))
    if int(board[marked[0]][marked[1]]) != int(current_player):
        return None
    legal_moves = tuple(enumerate_legal_moves(board, int(current_player)))
    capture_moves = tuple(move for move in legal_moves if move.captured is not None)
    chains = tuple(enumerate_king_capture_chains(board, player=int(current_player), origin=marked))
    if not chains:
        return None
    max_length = max(len(chain.captured) for chain in chains)
    max_chains = tuple(chain for chain in chains if len(chain.captured) == int(max_length))
    if len(max_chains) != 1:
        return None
    selected = max_chains[0]
    annotation_coords = tuple(selected.captured)
    return CheckersEvaluation(
        answer=int(max_length),
        legal_moves=legal_moves,
        capture_moves=capture_moves,
        annotation_coords=annotation_coords,
        annotation_entity_ids=tuple(piece_to_entity_id(coord, player=opponent(int(current_player))) for coord in annotation_coords),
        annotation_kind="piece",
        marked_coord=marked,
        max_capture_chains=max_chains,
        selected_capture_chain=selected,
    )


def _base_move_destination_board(*, rng, current_player: int, target_answer: int, capture_only: bool) -> Tuple[Board, str]:
    """Construct a sparse board for landing-square count semantics."""

    mutable = [list(int(cell) for cell in row) for row in empty_board()]
    if not bool(capture_only):
        if int(target_answer) > 0:
            selected_slots = list(rng.sample(_quiet_slots(int(current_player)), k=int(target_answer)))
            for origin, _landing in selected_slots:
                mutable[int(origin[0])][int(origin[1])] = int(current_player)
            return freeze_board(mutable), "quiet_edge_templates"
        return freeze_board(mutable), "empty_zero_legal"

    if int(target_answer) > 0:
        selected_slots = list(rng.sample(_capture_slots(int(current_player)), k=int(target_answer)))
        for origin, captured, _landing in selected_slots:
            mutable[int(origin[0])][int(origin[1])] = int(current_player)
            mutable[int(captured[0])][int(captured[1])] = int(opponent(int(current_player)))
        return freeze_board(mutable), "capture_edge_templates"

    quiet_count = min(len(_quiet_slots(int(current_player))), max(1, int(rng.randint(2, 4))))
    for origin, _landing in list(rng.sample(_quiet_slots(int(current_player)), k=int(quiet_count))):
        mutable[int(origin[0])][int(origin[1])] = int(current_player)
    return freeze_board(mutable), "quiet_zero_capture"


def _base_piece_mobility_board(*, rng, current_player: int, target_answer: int, capture_only: bool) -> Tuple[Board, str]:
    """Construct a sparse board for source-piece mobility semantics."""

    mutable = [list(int(cell) for cell in row) for row in empty_board()]
    if not bool(capture_only):
        if int(target_answer) > 0:
            selected_slots = list(rng.sample(_quiet_slots(int(current_player)), k=int(target_answer)))
            for origin, _landing in selected_slots:
                mutable[int(origin[0])][int(origin[1])] = int(current_player)
            return freeze_board(mutable), "piece_quiet_edge_templates"
        return freeze_board(mutable), "empty_zero_piece_legal"

    if int(target_answer) > 0:
        selected_slots = list(rng.sample(_capture_slots(int(current_player)), k=int(target_answer)))
        for origin, captured, _landing in selected_slots:
            mutable[int(origin[0])][int(origin[1])] = int(current_player)
            mutable[int(captured[0])][int(captured[1])] = int(opponent(int(current_player)))
        return freeze_board(mutable), "piece_capture_edge_templates"

    quiet_count = min(len(_quiet_slots(int(current_player))), max(1, int(rng.randint(2, 4))))
    for origin, _landing in list(rng.sample(_quiet_slots(int(current_player)), k=int(quiet_count))):
        mutable[int(origin[0])][int(origin[1])] = int(current_player)
    return freeze_board(mutable), "quiet_zero_piece_capture"


def _base_piece_state_board(*, rng, player: int, edge_only: bool, target_answer: int) -> Board:
    """Construct a sparse board with an exact piece-state answer."""

    candidates = list(piece_state_candidate_coords(int(player), edge_only=bool(edge_only)))
    if int(target_answer) > len(candidates):
        raise ValueError("piece-state target exceeds candidate count")
    selected = set(rng.sample(candidates, k=int(target_answer)))
    mutable = [list(int(cell) for cell in row) for row in empty_board()]
    for row, col in selected:
        mutable[int(row)][int(col)] = int(player)
    return freeze_board(mutable)


def _base_king_chain_board(*, rng, current_player: int, target_answer: int) -> Tuple[Board, Coord]:
    """Construct a board with one unique marked-king capture chain prefix."""

    if not (1 <= int(target_answer) <= 5):
        raise ValueError("king capture-chain target must be in 1..5")
    landing_path = _transform_king_chain_template(rng)[: int(target_answer) + 1]
    mutable = [list(int(cell) for cell in row) for row in empty_board()]
    marked_coord = landing_path[0]
    mutable[int(marked_coord[0])][int(marked_coord[1])] = int(current_player)
    for origin, landing in zip(landing_path, landing_path[1:]):
        captured = ((int(origin[0]) + int(landing[0])) // 2, (int(origin[1]) + int(landing[1])) // 2)
        mutable[int(captured[0])][int(captured[1])] = int(opponent(int(current_player)))
    return freeze_board(mutable), marked_coord


def _try_add_fillers(
    *,
    rng,
    board: Board,
    current_player: int,
    target_answer: int,
    scene_variant: str,
    occupied_range: Tuple[int, int] | None = None,
    evaluator: Callable[[Board], CheckersEvaluation | None],
) -> Tuple[Board, CheckersEvaluation, int]:
    """Add non-semantic filler pieces while preserving the requested answer."""

    min_occupied, max_occupied = (
        occupied_range
        if occupied_range is not None
        else scene_occupied_range(str(scene_variant))
    )
    desired_occupied = int(rng.randint(int(min_occupied), int(max_occupied)))
    mutable = [list(int(cell) for cell in row) for row in board]
    current_occupied = int(occupied_piece_count(mutable))
    evaluation = evaluator(freeze_board(mutable))
    if evaluation is None or int(evaluation.answer) != int(target_answer):
        raise ValueError("base board did not satisfy the requested checkers answer")

    playable = list(playable_coords())
    attempts = 0
    while int(current_occupied) < int(desired_occupied) and attempts < 640:
        attempts += 1
        row, col = playable[int(rng.randrange(len(playable)))]
        if int(mutable[row][col]) != 0:
            continue
        piece_player = int(current_player if float(rng.random()) < 0.36 else opponent(int(current_player)))
        if not allowed_non_king_row(int(piece_player), int(row)):
            continue
        mutable[row][col] = int(piece_player)
        candidate = evaluator(freeze_board(mutable))
        if candidate is None or int(candidate.answer) != int(target_answer):
            mutable[row][col] = 0
            continue
        evaluation = candidate
        current_occupied = int(occupied_piece_count(mutable))

    frozen = freeze_board(mutable)
    evaluation = evaluator(frozen)
    final_occupied = int(occupied_piece_count(frozen))
    if evaluation is None or int(evaluation.answer) != int(target_answer):
        raise ValueError("failed to preserve the requested checkers answer after filler placement")
    if not (int(min_occupied) <= int(final_occupied) <= int(max_occupied)):
        raise ValueError("failed to reach the requested scene-density range for checkers")
    return frozen, evaluation, final_occupied


def sample_move_destination_scene(
    *,
    rng,
    axes: ResolvedCheckersSceneAxes,
    params: Mapping[str, Any],
    target_answer: int,
    capture_only: bool,
    occupied_range: Tuple[int, int] | None = None,
) -> SampledCheckersScene:
    """Construct one Checkers landing-square count scene."""

    current_player = resolve_current_player(rng, params=params)
    board, mode = _base_move_destination_board(
        rng=rng,
        current_player=int(current_player),
        target_answer=int(target_answer),
        capture_only=bool(capture_only),
    )
    evaluator = lambda candidate: evaluate_move_destinations(  # noqa: E731
        board=candidate,
        current_player=int(current_player),
        capture_only=bool(capture_only),
    )
    board, evaluation, occupied_count = _try_add_fillers(
        rng=rng,
        board=board,
        current_player=int(current_player),
        target_answer=int(target_answer),
        scene_variant=str(axes.scene_variant),
        occupied_range=occupied_range,
        evaluator=evaluator,
    )
    return SampledCheckersScene(
        board=board,
        current_player=int(current_player),
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        evaluation=evaluation,
        occupied_count=int(occupied_count),
        construction_mode=str(mode),
    )


def sample_piece_mobility_scene(
    *,
    rng,
    axes: ResolvedCheckersSceneAxes,
    params: Mapping[str, Any],
    target_answer: int,
    capture_only: bool,
    occupied_range: Tuple[int, int] | None = None,
) -> SampledCheckersScene:
    """Construct one Checkers source-piece mobility count scene."""

    current_player = resolve_current_player(rng, params=params)
    board, mode = _base_piece_mobility_board(
        rng=rng,
        current_player=int(current_player),
        target_answer=int(target_answer),
        capture_only=bool(capture_only),
    )
    evaluator = lambda candidate: evaluate_piece_mobility(  # noqa: E731
        board=candidate,
        current_player=int(current_player),
        capture_only=bool(capture_only),
    )
    board, evaluation, occupied_count = _try_add_fillers(
        rng=rng,
        board=board,
        current_player=int(current_player),
        target_answer=int(target_answer),
        scene_variant=str(axes.scene_variant),
        occupied_range=occupied_range,
        evaluator=evaluator,
    )
    return SampledCheckersScene(
        board=board,
        current_player=int(current_player),
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        evaluation=evaluation,
        occupied_count=int(occupied_count),
        construction_mode=str(mode),
    )


def sample_piece_state_scene(
    *,
    rng,
    axes: ResolvedCheckersSceneAxes,
    params: Mapping[str, Any],
    target_answer: int,
    player: int,
    edge_only: bool,
    occupied_range: Tuple[int, int] | None = None,
) -> SampledCheckersScene:
    """Construct one Checkers visible piece-state count scene."""

    current_player = resolve_current_player(rng, params=params)
    board = _base_piece_state_board(
        rng=rng,
        player=int(player),
        edge_only=bool(edge_only),
        target_answer=int(target_answer),
    )
    evaluator = lambda candidate: evaluate_piece_state(  # noqa: E731
        board=candidate,
        player=int(player),
        edge_only=bool(edge_only),
    )
    board, evaluation, occupied_count = _try_add_fillers(
        rng=rng,
        board=board,
        current_player=int(current_player),
        target_answer=int(target_answer),
        scene_variant=str(axes.scene_variant),
        occupied_range=occupied_range,
        evaluator=evaluator,
    )
    return SampledCheckersScene(
        board=board,
        current_player=int(current_player),
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        evaluation=evaluation,
        occupied_count=int(occupied_count),
        construction_mode="piece_state_count_templates",
        extra={"target_player": "red" if int(player) == int(RED) else "black", "edge_only": bool(edge_only)},
    )


def sample_king_capture_chain_scene(
    *,
    rng,
    axes: ResolvedCheckersSceneAxes,
    params: Mapping[str, Any],
    target_answer: int,
    occupied_range: Tuple[int, int] | None = None,
) -> SampledCheckersScene:
    """Construct one marked-king capture-chain scene."""

    current_player = resolve_current_player(rng, params=params)
    board, marked_coord = _base_king_chain_board(
        rng=rng,
        current_player=int(current_player),
        target_answer=int(target_answer),
    )
    evaluator = lambda candidate: evaluate_king_capture_chain(  # noqa: E731
        board=candidate,
        current_player=int(current_player),
        marked_coord=marked_coord,
    )
    board, evaluation, occupied_count = _try_add_fillers(
        rng=rng,
        board=board,
        current_player=int(current_player),
        target_answer=int(target_answer),
        scene_variant=str(axes.scene_variant),
        occupied_range=occupied_range,
        evaluator=evaluator,
    )
    return SampledCheckersScene(
        board=board,
        current_player=int(current_player),
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        evaluation=evaluation,
        occupied_count=int(occupied_count),
        construction_mode="marked_king_capture_chain",
    )


__all__ = [
    "evaluate_king_capture_chain",
    "evaluate_move_destinations",
    "evaluate_piece_mobility",
    "evaluate_piece_state",
    "movement_rule_text",
    "resolve_checkers_scene_axes",
    "resolve_checkers_target_answer",
    "resolve_task_occupied_range",
    "sample_king_capture_chain_scene",
    "sample_move_destination_scene",
    "sample_piece_mobility_scene",
    "sample_piece_state_scene",
    "scene_object_description",
]
