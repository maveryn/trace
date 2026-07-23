"""Identity-free sampling helpers for circular-chess games tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.piece_board_rules import (
    BLACK,
    CIRCULAR_CHESS_MATERIAL_CAPS,
    WHITE,
    ChessPiece,
    material_count,
    opponent,
    sample_material_piece,
    validate_circular_chess_material,
)
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import (
    FALLBACK_GENERATION_DEFAULTS,
    SCENE_ID,
    SUPPORTED_NON_KING_PIECE_KINDS,
    SUPPORTED_PIECE_KINDS,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
)
from .rules import (
    all_coords,
    empty_board,
    evaluate_marked_destinations,
    evaluate_target_reachers,
    freeze_board,
    legal_destinations,
    movement_units_for_piece,
    occupied_coords,
)
from .state import Board, CircularChessSample, CircularChessSceneAxes, Coord, MarkedPieceAxes


_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def _generation_default(key: str) -> Any:
    return group_default(_GEN_DEFAULTS, str(key), FALLBACK_GENERATION_DEFAULTS[str(key)])


def resolve_scene_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported_values: Sequence[str],
) -> tuple[str, dict[str, float]]:
    """Resolve one scene-semantic axis without public task identity."""

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


def resolve_circular_chess_scene_axes(instance_seed: int, *, params: Mapping[str, Any]) -> CircularChessSceneAxes:
    """Resolve shared scene and style axes."""

    scene_variant, scene_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.circular_chess.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_values=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.circular_chess.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_values=SUPPORTED_STYLE_VARIANTS,
    )
    return CircularChessSceneAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
    )


def resolve_marked_piece_axes(instance_seed: int, *, params: Mapping[str, Any]) -> MarkedPieceAxes:
    """Resolve visible marked-piece kind and color axes."""

    piece_kind, piece_kind_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.circular_chess.marked_piece_kind",
        explicit_key="marked_piece_kind",
        weights_key="marked_piece_kind_weights",
        balance_flag_key="balanced_marked_piece_kind_sampling",
        supported_values=SUPPORTED_PIECE_KINDS,
    )
    piece_color, piece_color_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.circular_chess.marked_piece_color",
        explicit_key="marked_piece_color",
        weights_key="marked_piece_color_weights",
        balance_flag_key="balanced_marked_piece_color_sampling",
        supported_values=(WHITE, BLACK),
    )
    return MarkedPieceAxes(
        piece_kind=str(piece_kind),
        piece_color=str(piece_color),
        piece_kind_probabilities=dict(piece_kind_probabilities),
        piece_color_probabilities=dict(piece_color_probabilities),
    )


def resolve_task_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any] | None = None,
    support_key: str,
    fallback_support: Sequence[int],
    possible_max: int,
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve a task-owned answer support after scene feasibility filtering."""

    active_defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    configured_support = resolve_integer_support(
        params,
        gen_defaults=active_defaults,
        key=str(support_key),
        fallback=tuple(int(v) for v in fallback_support),
    )
    target_support = tuple(int(v) for v in configured_support if int(v) <= int(possible_max))
    if not target_support:
        target_support = tuple(int(v) for v in fallback_support if int(v) <= int(possible_max))
    if not target_support:
        raise ValueError(f"no feasible target answers for {support_key}")
    target_answer, target_answer_probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults={**dict(active_defaults), str(support_key): list(target_support)},
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=target_support,
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return int(target_answer), tuple(int(v) for v in target_support), dict(target_answer_probabilities)


def _occupancy_bounds(scene_variant: str) -> tuple[int, int]:
    if str(scene_variant) == "crowded_board":
        return int(_generation_default("crowded_min_occupied_count")), int(_generation_default("crowded_max_occupied_count"))
    return int(_generation_default("sparse_min_occupied_count")), int(_generation_default("sparse_max_occupied_count"))


def _opposite_color(color: str) -> str:
    return opponent(str(color))


def _place_circular_piece(
    *,
    rng,
    mutable: list[list[ChessPiece | None]],
    coord: Coord,
    color: str | None = None,
    kinds: Sequence[str] = SUPPORTED_PIECE_KINDS,
) -> bool:
    """Place one material-capped piece on a circular-chess cell."""

    ring, sector = int(coord[0]), int(coord[1])
    if mutable[ring][sector] is not None:
        return False
    colors = (str(color),) if color is not None else (WHITE, BLACK)
    piece = sample_material_piece(
        rng,
        freeze_board(mutable),
        colors=colors,
        kinds=kinds,
        caps=CIRCULAR_CHESS_MATERIAL_CAPS,
    )
    if piece is None:
        return False
    mutable[ring][sector] = piece
    return True


def _place_circular_exact_piece(
    *,
    mutable: list[list[ChessPiece | None]],
    coord: Coord,
    piece: ChessPiece,
) -> bool:
    """Place an exact circular-chess piece if material caps allow it."""

    ring, sector = int(coord[0]), int(coord[1])
    if mutable[ring][sector] is not None:
        return False
    candidate = [list(row) for row in mutable]
    candidate[ring][sector] = piece
    if not validate_circular_chess_material(freeze_board(candidate)):
        return False
    mutable[ring][sector] = piece
    return True


def _circular_kings_valid(board: Board) -> bool:
    return material_count(board, color=WHITE, kind="king") == 1 and material_count(board, color=BLACK, kind="king") == 1


def _missing_circular_king_count(board: Board) -> int:
    return sum(1 for color in (WHITE, BLACK) if material_count(board, color=str(color), kind="king") == 0)


def _add_required_kings_preserving(
    *,
    rng,
    board: Board,
    evaluate,
    target_answer: int,
    target_coord: Coord | None,
) -> Board:
    """Add one king per color without changing the active answer."""

    mutable = [list(row) for row in board]
    for color in (WHITE, BLACK):
        if material_count(freeze_board(mutable), color=str(color), kind="king") >= 1:
            continue
        candidates = [
            coord
            for coord in all_coords()
            if mutable[int(coord[0])][int(coord[1])] is None and (target_coord is None or tuple(coord) != tuple(target_coord))
        ]
        rng.shuffle(candidates)
        placed = False
        for coord in candidates:
            if not _place_circular_exact_piece(
                mutable=mutable,
                coord=coord,
                piece=ChessPiece(str(color), "king"),
            ):
                continue
            candidate_board = freeze_board(mutable)
            evaluation = evaluate(candidate_board)
            if int(evaluation.answer) != int(target_answer):
                mutable[int(coord[0])][int(coord[1])] = None
                continue
            placed = True
            break
        if not placed:
            raise ValueError("failed to add required circular chess king without changing answer")
    final = freeze_board(mutable)
    if not validate_circular_chess_material(final) or not _circular_kings_valid(final):
        raise ValueError("circular chess board is not material-plausible")
    return final


def _allocate_move_destinations(
    units: Sequence[Sequence[Coord]],
    target_answer: int,
    rng,
) -> tuple[tuple[Coord, ...], tuple[Coord, ...]]:
    """Choose exact legal destination cells and friendly blockers for a move target."""

    unit_list = [tuple(unit) for unit in units if unit]
    rng.shuffle(unit_list)
    desired: list[Coord] = []
    blockers: list[Coord] = []
    remaining = int(target_answer)
    for unit in unit_list:
        if int(remaining) <= 0:
            blockers.append(tuple(unit[0]))
            continue
        take = min(int(remaining), len(unit))
        desired.extend(tuple(coord) for coord in unit[:take])
        remaining -= int(take)
        if int(take) < len(unit):
            blockers.append(tuple(unit[int(take)]))
    if int(remaining) > 0:
        raise ValueError("target move count exceeds available Circular Chess movement cells")
    return tuple(sorted(set(desired))), tuple(sorted(set(blockers) - set(desired)))


def sample_marked_destination_scene(
    *,
    rng,
    scene_axes: CircularChessSceneAxes,
    marked_axes: MarkedPieceAxes,
    destination_mode: str,
    target_answer: int,
) -> CircularChessSample:
    """Construct a board with an exact marked-piece move/capture answer."""

    capture_mode = str(destination_mode) == "capture"
    if str(destination_mode) not in {"move", "capture"}:
        raise ValueError(f"unsupported destination mode: {destination_mode}")
    candidate_origins = list(all_coords())
    rng.shuffle(candidate_origins)
    for origin in candidate_origins:
        units = movement_units_for_piece(str(marked_axes.piece_kind), origin)
        if capture_mode:
            if int(target_answer) > sum(1 for unit in units if unit):
                continue
            unit_list = [tuple(unit) for unit in units if unit]
            rng.shuffle(unit_list)
            selected_units = tuple(unit_list[: int(target_answer)])
            desired = tuple(sorted(unit[0] for unit in selected_units))
            blockers = tuple(sorted(unit[0] for unit in unit_list[int(target_answer) :]))
        else:
            if int(target_answer) > sum(len(unit) for unit in units):
                continue
            desired, blockers = _allocate_move_destinations(units, target_answer, rng)

        mutable = [list(row) for row in empty_board()]
        marked_piece = ChessPiece(color=str(marked_axes.piece_color), kind=str(marked_axes.piece_kind))
        if not _place_circular_exact_piece(mutable=mutable, coord=origin, piece=marked_piece):
            continue
        for coord in desired:
            if capture_mode and not _place_circular_piece(
                rng=rng,
                mutable=mutable,
                coord=coord,
                color=_opposite_color(str(marked_axes.piece_color)),
            ):
                raise ValueError("failed to place circular chess capture target")
        for coord in blockers:
            if mutable[int(coord[0])][int(coord[1])] is None and not _place_circular_piece(
                rng=rng,
                mutable=mutable,
                coord=coord,
                color=str(marked_axes.piece_color),
            ):
                raise ValueError("failed to place circular chess blocker")

        board = freeze_board(mutable)
        protected = {tuple(origin), *desired, *blockers}
        movement_cells = {coord for unit in units for coord in unit}
        minimum, maximum = _occupancy_bounds(str(scene_axes.scene_variant))
        missing_kings = _missing_circular_king_count(board)
        target_occupied_max = max(len(occupied_coords(board)), min(int(maximum) - int(missing_kings), 26))
        target_occupied = int(
            rng.randint(
                max(int(minimum) - int(missing_kings), len(occupied_coords(board))),
                max(int(minimum), target_occupied_max),
            )
        )
        available = [
            coord
            for coord in all_coords()
            if coord not in protected
            and coord not in movement_cells
            and board[int(coord[0])][int(coord[1])] is None
        ]
        rng.shuffle(available)
        mutable = [list(row) for row in board]
        for coord in available:
            if len(occupied_coords(freeze_board(mutable))) >= int(target_occupied):
                break
            _place_circular_piece(
                rng=rng,
                mutable=mutable,
                coord=coord,
                kinds=SUPPORTED_NON_KING_PIECE_KINDS,
            )
        board = freeze_board(mutable)
        evaluation = evaluate_marked_destinations(
            board,
            destination_mode=str(destination_mode),
            marked_coord=origin,
        )
        if int(evaluation.answer) != int(target_answer):
            continue

        board = _add_required_kings_preserving(
            rng=rng,
            board=board,
            evaluate=lambda candidate: evaluate_marked_destinations(
                candidate,
                destination_mode=str(destination_mode),
                marked_coord=origin,
            ),
            target_answer=int(target_answer),
            target_coord=None,
        )
        evaluation = evaluate_marked_destinations(
            board,
            destination_mode=str(destination_mode),
            marked_coord=origin,
        )
        if int(evaluation.answer) == int(target_answer):
            return CircularChessSample(
                board=board,
                evaluation=evaluation,
                occupied_count=len(occupied_coords(board)),
                construction_mode="constructed_marked_piece_exact_answer",
                scene_variant=str(scene_axes.scene_variant),
                style_variant=str(scene_axes.style_variant),
            )
    raise ValueError("failed to construct marked Circular Chess sample")


def _candidate_reacher_sources(target_coord: Coord) -> tuple[tuple[Coord, str], ...]:
    candidates: list[tuple[Coord, str]] = []
    board_rows = [list(row) for row in empty_board()]
    for coord in all_coords():
        if tuple(coord) == tuple(target_coord):
            continue
        for kind in SUPPORTED_PIECE_KINDS:
            mutable = [list(row) for row in board_rows]
            mutable[int(coord[0])][int(coord[1])] = ChessPiece(color=WHITE, kind=str(kind))
            board = freeze_board(mutable)
            if tuple(target_coord) in legal_destinations(board, coord):
                candidates.append((tuple(coord), str(kind)))
    return tuple(candidates)


def sample_target_reacher_scene(
    *,
    rng,
    scene_axes: CircularChessSceneAxes,
    target_color: str,
    target_answer: int,
) -> CircularChessSample:
    """Construct a board with an exact target-cell reacher answer."""

    target_coords = [(ring, sector) for ring in (1, 2) for sector in range(16)]
    rng.shuffle(target_coords)
    for target_coord in target_coords:
        candidates = list(_candidate_reacher_sources(target_coord))
        rng.shuffle(candidates)
        mutable = [list(row) for row in empty_board()]
        selected: list[tuple[Coord, str]] = []
        used_coords: set[Coord] = set()
        for coord, kind in candidates:
            if len(selected) >= int(target_answer):
                break
            if tuple(coord) in used_coords:
                continue
            if not _place_circular_exact_piece(
                mutable=mutable,
                coord=coord,
                piece=ChessPiece(color=str(target_color), kind=str(kind)),
            ):
                continue
            selected.append((tuple(coord), str(kind)))
            used_coords.add(tuple(coord))
        if len(selected) != int(target_answer):
            continue

        board = freeze_board(mutable)
        minimum, maximum = _occupancy_bounds(str(scene_axes.scene_variant))
        missing_kings = _missing_circular_king_count(board)
        target_occupied_max = max(len(occupied_coords(board)), min(int(maximum) - int(missing_kings), 24))
        target_occupied = int(
            rng.randint(
                max(int(minimum) - int(missing_kings), len(occupied_coords(board))),
                max(int(minimum), target_occupied_max),
            )
        )
        all_available = [
            coord
            for coord in all_coords()
            if tuple(coord) != tuple(target_coord) and board[int(coord[0])][int(coord[1])] is None
        ]
        rng.shuffle(all_available)
        mutable = [list(row) for row in board]
        for coord in all_available:
            if len(occupied_coords(freeze_board(mutable))) >= int(target_occupied):
                break
            old = mutable[int(coord[0])][int(coord[1])]
            if not _place_circular_piece(
                rng=rng,
                mutable=mutable,
                coord=coord,
                kinds=SUPPORTED_NON_KING_PIECE_KINDS,
            ):
                continue
            candidate_board = freeze_board(mutable)
            evaluation = evaluate_target_reachers(
                candidate_board,
                target_coord=target_coord,
                target_color=str(target_color),
            )
            if int(evaluation.answer) != int(target_answer):
                mutable[int(coord[0])][int(coord[1])] = old
        board = freeze_board(mutable)
        evaluation = evaluate_target_reachers(
            board,
            target_coord=target_coord,
            target_color=str(target_color),
        )
        if int(evaluation.answer) != int(target_answer):
            continue

        board = _add_required_kings_preserving(
            rng=rng,
            board=board,
            evaluate=lambda candidate: evaluate_target_reachers(
                candidate,
                target_coord=target_coord,
                target_color=str(target_color),
            ),
            target_answer=int(target_answer),
            target_coord=target_coord,
        )
        evaluation = evaluate_target_reachers(
            board,
            target_coord=target_coord,
            target_color=str(target_color),
        )
        if int(evaluation.answer) == int(target_answer):
            return CircularChessSample(
                board=board,
                evaluation=evaluation,
                occupied_count=len(occupied_coords(board)),
                construction_mode="constructed_target_reacher_exact_answer",
                scene_variant=str(scene_axes.scene_variant),
                style_variant=str(scene_axes.style_variant),
            )
    raise ValueError("failed to construct target-reacher Circular Chess sample")


__all__ = [
    "resolve_circular_chess_scene_axes",
    "resolve_marked_piece_axes",
    "resolve_scene_axis",
    "resolve_task_target_answer",
    "sample_marked_destination_scene",
    "sample_target_reacher_scene",
]
