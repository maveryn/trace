"""Sampling helpers for the Chess games scene."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.piece_board_rules import (
    BLACK,
    BOARD_SIZE,
    WHITE,
    Board,
    ChessPiece,
    Coord,
    apply_chess_move,
    empty_board,
    find_king,
    freeze_board,
    in_bounds,
    is_king_in_check,
    legal_chess_moves_for_color,
    move_checkmates,
    occupied_piece_count,
    opponent,
    piece_to_entity_id,
    serialize_board,
    validate_square_chess_material,
)
from trace_tasks.tasks.games.shared.style import SUPPORTED_CHESS_STYLE_VARIANTS
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import FALLBACK_GENERATION_DEFAULTS
from .rules import (
    add_material_fillers_preserving,
    attacker_slots_for_square,
    coords_between,
    evaluate_king_escapes,
    evaluate_line_blockers,
    evaluate_marked_destinations,
    evaluate_piece_matches,
    evaluate_player_captures,
    evaluate_target_attackers,
    line_allowed_for_slider,
    opponent_attackers_after_king_move,
    piece_entity_ids,
    place_capped_piece,
    place_capped_random_piece,
    place_display_piece,
    place_non_adjacent_kings,
    random_empty_coord,
    same_color_slider_points_to_any,
    slider_directions,
)
from .state import (
    CHESS_OPTION_LABELS,
    PIECE_COUNT_COLOR_SUPPORT,
    PIECE_COUNT_KIND_SUPPORT,
    SCENE_ID,
    SUPPORTED_CHESS_SCENE_VARIANTS,
    ChessCheckmateSample,
    ChessMoveOption,
    ChessSceneSample,
    ResolvedChessSceneAxes,
    cell_entity_ids,
)

_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def string_probability_map(values: Sequence[str], selected: str) -> Dict[str, float]:
    """Return a one-hot probability map for string values."""

    return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in values}


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


def resolve_chess_scene_axes(instance_seed: int, *, params: Mapping[str, Any]) -> ResolvedChessSceneAxes:
    """Resolve shared scene and style axes."""

    scene_variant, scene_probs = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.chess.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_values=SUPPORTED_CHESS_SCENE_VARIANTS,
    )
    style_variant, style_probs = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.chess.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_values=SUPPORTED_CHESS_STYLE_VARIANTS,
    )
    return ResolvedChessSceneAxes(
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_probs),
    )


def resolve_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    gen_defaults: Mapping[str, Any] | None = None,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
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
    return int(answer), tuple(int(value) for value in support), dict(probabilities)


def resolve_string_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported_values: Sequence[str],
    namespace: str,
    gen_defaults: Mapping[str, Any] | None = None,
) -> Tuple[str, Dict[str, float]]:
    """Resolve a task-owned string axis."""

    defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    supported = tuple(str(value) for value in supported_values)
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=defaults,
        supported_variants=supported,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=str(namespace),
    )
    return str(selected), dict(probabilities)


def resolve_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balance_flag_key: str,
    gen_defaults: Mapping[str, Any] | None = None,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve a task-owned integer axis not named target_answer."""

    defaults = _GEN_DEFAULTS if gen_defaults is None else gen_defaults
    value, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(item) for item in fallback_support),
        namespace=str(namespace),
        balanced_flag_key=str(balance_flag_key),
        namespace_support_permutation=True,
    )
    support = resolve_integer_support(
        params,
        gen_defaults=defaults,
        key=str(support_key),
        fallback=tuple(int(item) for item in fallback_support),
    )
    return int(value), tuple(int(item) for item in support), dict(probabilities)


def resolve_player_color(rng, *, params: Mapping[str, Any]) -> str:
    """Resolve a side color from params or local randomness."""

    explicit = params.get("player_color")
    if explicit is None:
        return WHITE if int(rng.randrange(2)) == 0 else BLACK
    text = str(explicit).strip().lower()
    if text in {"white", "w"}:
        return WHITE
    if text in {"black", "b"}:
        return BLACK
    raise ValueError(f"unsupported player_color: {explicit}")


def _ray_coords_from(origin: Coord, direction: Coord) -> Tuple[Coord, ...]:
    """Return in-bounds coordinates along one ray from a source square."""

    row, col = int(origin[0]), int(origin[1])
    delta_row, delta_col = int(direction[0]), int(direction[1])
    coords: list[Coord] = []
    row += int(delta_row)
    col += int(delta_col)
    while in_bounds(row, col):
        coords.append((int(row), int(col)))
        row += int(delta_row)
        col += int(delta_col)
    return tuple(coords)


def _allocate_ray_lengths(rng, ray_lengths: Sequence[int], target_answer: int) -> Tuple[int, ...]:
    """Randomly split a target destination count across bounded movement rays."""

    lengths = [int(value) for value in ray_lengths]
    if int(target_answer) < 0 or int(target_answer) > sum(lengths):
        raise ValueError("target answer is not feasible for marked-piece rays")
    allocations = [0 for _ in lengths]
    remaining = int(target_answer)
    order = list(range(len(lengths)))
    rng.shuffle(order)
    for order_index, ray_index in enumerate(order):
        remaining_ray_indices = order[order_index + 1 :]
        remaining_capacity = sum(lengths[index] for index in remaining_ray_indices)
        min_count = max(0, int(remaining) - int(remaining_capacity))
        max_count = min(int(lengths[ray_index]), int(remaining))
        allocations[ray_index] = int(rng.randint(int(min_count), int(max_count)))
        remaining -= int(allocations[ray_index])
    if int(remaining) != 0:
        raise ValueError("failed to allocate marked-piece ray lengths")
    return tuple(int(value) for value in allocations)


def _piece_count_target_rng_color(rng) -> str:
    return str(rng.choice(PIECE_COUNT_COLOR_SUPPORT))


def sample_piece_count_scene(
    *,
    rng,
    axes: ResolvedChessSceneAxes,
    target_answer: int,
    target_kind: str,
    target_color: str | None,
    distractor_count: int,
) -> ChessSceneSample:
    """Construct a display board with an exact visible piece-kind count."""

    if str(target_kind) not in PIECE_COUNT_KIND_SUPPORT:
        raise ValueError(f"unsupported target piece kind: {target_kind}")
    if target_color is not None and str(target_color) not in PIECE_COUNT_COLOR_SUPPORT:
        raise ValueError(f"unsupported target piece color: {target_color}")
    mutable = [list(row) for row in empty_board()]
    for _ in range(int(target_answer)):
        color = str(target_color) if target_color is not None else _piece_count_target_rng_color(rng)
        place_display_piece(rng=rng, mutable=mutable, color=color, kind=str(target_kind))
    distractor_specs = []
    for color in PIECE_COUNT_COLOR_SUPPORT:
        for kind in PIECE_COUNT_KIND_SUPPORT:
            if target_color is None and str(kind) == str(target_kind):
                continue
            if target_color is not None and str(kind) == str(target_kind) and str(color) == str(target_color):
                continue
            distractor_specs.append((str(color), str(kind)))
    for _ in range(int(distractor_count)):
        color, kind = tuple(rng.choice(distractor_specs))
        place_display_piece(rng=rng, mutable=mutable, color=str(color), kind=str(kind))
    board = freeze_board(mutable)
    matches = evaluate_piece_matches(board, target_kind=str(target_kind), target_color=None if target_color is None else str(target_color))
    if len(matches) != int(target_answer):
        raise ValueError("constructed chess piece-count board has wrong answer")
    return ChessSceneSample(
        board=board,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        construction_mode="display_piece_count",
        player_color=str(target_color or WHITE),
        target_piece_kind=str(target_kind),
        target_piece_color="" if target_color is None else str(target_color),
        annotation_coords=matches,
        annotation_entity_ids=piece_entity_ids(board, matches),
        annotation_kind="piece",
        occupied_count=int(occupied_piece_count(board)),
    )


def _sample_marked_slider_destination_scene(
    *,
    rng,
    axes: ResolvedChessSceneAxes,
    destination_mode: str,
    target_answer: int,
    marked_piece_kind: str,
) -> ChessSceneSample:
    """Construct an exact destination/capture scene for a marked sliding piece."""

    if str(marked_piece_kind) not in {"bishop", "rook", "queen"}:
        raise ValueError(f"unsupported slider marked piece kind: {marked_piece_kind}")
    marked_coord: Coord = (3, 3)
    piece_color = WHITE if int(rng.randrange(2)) == 0 else BLACK
    opponent_color = opponent(piece_color)
    directions = tuple(slider_directions(str(marked_piece_kind)))
    rays = tuple(_ray_coords_from(marked_coord, direction) for direction in directions)
    ray_cells = tuple(coord for ray in rays for coord in ray)

    for _attempt in range(80):
        mutable = [list(row) for row in empty_board()]
        mutable[marked_coord[0]][marked_coord[1]] = ChessPiece(piece_color, str(marked_piece_kind))

        if str(destination_mode) == "move":
            allocations = _allocate_ray_lengths(rng, tuple(len(ray) for ray in rays), int(target_answer))
            for ray, allowed_count in zip(rays, allocations):
                if int(allowed_count) >= len(ray):
                    continue
                blocker_coord = ray[int(allowed_count)]
                if not place_capped_random_piece(
                    rng=rng,
                    mutable=mutable,
                    coord=blocker_coord,
                    colors=(piece_color,),
                    kinds=("pawn", "knight", "bishop", "rook", "queen"),
                ):
                    raise ValueError("failed to place friendly ray blocker")
        elif str(destination_mode) == "capture":
            if int(target_answer) > len(rays):
                raise ValueError("capture target exceeds slider ray count")
            selected_indices = list(range(len(rays)))
            rng.shuffle(selected_indices)
            capture_ray_indices = set(selected_indices[: int(target_answer)])
            for ray_index, ray in enumerate(rays):
                if not ray:
                    continue
                if ray_index in capture_ray_indices:
                    capture_coord = tuple(rng.choice(tuple(ray)))
                    if not place_capped_random_piece(
                        rng=rng,
                        mutable=mutable,
                        coord=capture_coord,
                        colors=(opponent_color,),
                        kinds=("pawn", "knight", "bishop", "rook", "queen"),
                    ):
                        raise ValueError("failed to place slider capture target")
                elif int(rng.randrange(2)) == 0:
                    blocker_coord = tuple(rng.choice(tuple(ray)))
                    place_capped_random_piece(
                        rng=rng,
                        mutable=mutable,
                        coord=blocker_coord,
                        colors=(piece_color,),
                        kinds=("pawn", "knight", "bishop", "rook", "queen"),
                    )
        else:
            raise ValueError(f"unsupported destination mode: {destination_mode}")

        try:
            place_non_adjacent_kings(
                rng=rng,
                mutable=mutable,
                first_color=piece_color,
                second_color=opponent_color,
                forbidden=(marked_coord,) + tuple(ray_cells),
            )
        except ValueError:
            continue
        board = freeze_board(mutable)
        coords = evaluate_marked_destinations(board, marked_coord, destination_mode=str(destination_mode))
        if len(coords) != int(target_answer):
            continue
        board = add_material_fillers_preserving(
            rng=rng,
            board=board,
            scene_variant=str(axes.scene_variant),
            preserved_coords=(marked_coord,) + tuple(ray_cells),
            expected_coords=coords,
            evaluator=lambda candidate: evaluate_marked_destinations(candidate, marked_coord, destination_mode=str(destination_mode)),
        )
        coords = evaluate_marked_destinations(board, marked_coord, destination_mode=str(destination_mode))
        piece = board[marked_coord[0]][marked_coord[1]]
        return ChessSceneSample(
            board=board,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            construction_mode=f"direct_marked_{str(marked_piece_kind)}_{str(destination_mode)}_count",
            player_color=str(piece_color),
            marked_coord=marked_coord,
            marked_piece=piece,
            destination_coords=coords,
            capture_coords=coords if str(destination_mode) == "capture" else (),
            annotation_coords=coords,
            annotation_entity_ids=cell_entity_ids(coords),
            annotation_kind="cell",
            occupied_count=int(occupied_piece_count(board)),
            extra={"marked_piece_kind": str(marked_piece_kind)},
        )
    raise ValueError("failed to construct marked-slider destination scene")


def sample_marked_piece_destination_scene(
    *,
    rng,
    axes: ResolvedChessSceneAxes,
    destination_mode: str,
    target_answer: int,
    marked_piece_kind: str = "knight",
) -> ChessSceneSample:
    """Construct a marked-piece destination or capture-count scene."""

    if str(marked_piece_kind) in {"bishop", "rook", "queen"}:
        return _sample_marked_slider_destination_scene(
            rng=rng,
            axes=axes,
            destination_mode=str(destination_mode),
            target_answer=int(target_answer),
            marked_piece_kind=str(marked_piece_kind),
        )
    if str(marked_piece_kind) != "knight":
        raise ValueError(f"unsupported marked piece kind: {marked_piece_kind}")

    marked_coord: Coord = (3, 3)
    piece_color = WHITE if int(rng.randrange(2)) == 0 else BLACK
    opponent_color = opponent(piece_color)
    for _attempt in range(80):
        mutable = [list(row) for row in empty_board()]
        mutable[marked_coord[0]][marked_coord[1]] = ChessPiece(piece_color, "knight")
        knight_dests = list(evaluate_marked_destinations(freeze_board(mutable), marked_coord, destination_mode="move"))
        rng.shuffle(knight_dests)
        if str(destination_mode) == "move":
            if int(target_answer) > len(knight_dests):
                raise ValueError("marked knight cannot expose requested move count")
            blocked = knight_dests[int(target_answer) :]
            for coord in blocked:
                if not place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, colors=(piece_color,), kinds=("pawn", "knight", "bishop", "rook", "queen")):
                    raise ValueError("failed to place friendly blocker")
        elif str(destination_mode) == "capture":
            if int(target_answer) > len(knight_dests):
                raise ValueError("marked knight cannot expose requested capture count")
            for coord in knight_dests[: int(target_answer)]:
                if not place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, colors=(opponent_color,), kinds=("pawn", "knight", "bishop", "rook", "queen")):
                    raise ValueError("failed to place capture target")
            for coord in knight_dests[int(target_answer) :]:
                if int(rng.randrange(2)) == 0:
                    place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, colors=(piece_color,), kinds=("pawn", "knight", "bishop", "rook", "queen"))
        else:
            raise ValueError(f"unsupported destination mode: {destination_mode}")
        place_non_adjacent_kings(rng=rng, mutable=mutable, first_color=piece_color, second_color=opponent_color, forbidden=(marked_coord,))
        board = freeze_board(mutable)
        coords = evaluate_marked_destinations(board, marked_coord, destination_mode=str(destination_mode))
        if len(coords) != int(target_answer):
            continue
        board = add_material_fillers_preserving(
            rng=rng,
            board=board,
            scene_variant=str(axes.scene_variant),
            preserved_coords=(marked_coord,) + tuple(knight_dests),
            expected_coords=coords,
            evaluator=lambda candidate: evaluate_marked_destinations(candidate, marked_coord, destination_mode=str(destination_mode)),
        )
        coords = evaluate_marked_destinations(board, marked_coord, destination_mode=str(destination_mode))
        piece = board[marked_coord[0]][marked_coord[1]]
        return ChessSceneSample(
            board=board,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            construction_mode=f"direct_marked_knight_{str(destination_mode)}_count",
            player_color=str(piece_color),
            marked_coord=marked_coord,
            marked_piece=piece,
            destination_coords=coords,
            capture_coords=coords if str(destination_mode) == "capture" else (),
            annotation_coords=coords,
            annotation_entity_ids=cell_entity_ids(coords),
            annotation_kind="cell",
            occupied_count=int(occupied_piece_count(board)),
            extra={"marked_piece_kind": str(marked_piece_kind)},
        )
    raise ValueError("failed to construct marked-piece destination scene")


def sample_player_capture_scene(
    *,
    rng,
    axes: ResolvedChessSceneAxes,
    player_color: str,
    target_answer: int,
) -> ChessSceneSample:
    """Construct an exact side-wide capture-count scene."""

    queen_coord = (3, 3)
    opponent_color = opponent(str(player_color))
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    candidate_targets = [(queen_coord[0] + dr, queen_coord[1] + dc) for dr, dc in directions if in_bounds(queen_coord[0] + dr, queen_coord[1] + dc)]
    if int(target_answer) > len(candidate_targets):
        raise ValueError("target capture count exceeds construction support")
    mutable = [list(row) for row in empty_board()]
    mutable[queen_coord[0]][queen_coord[1]] = ChessPiece(str(player_color), "queen")
    rng.shuffle(candidate_targets)
    for coord in candidate_targets[: int(target_answer)]:
        if not place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, colors=(opponent_color,), kinds=("pawn", "knight", "bishop", "rook")):
            raise ValueError("failed to place capturable target")
    place_non_adjacent_kings(rng=rng, mutable=mutable, first_color=str(player_color), second_color=opponent_color, forbidden=(queen_coord,))
    board = freeze_board(mutable)
    coords = evaluate_player_captures(board, str(player_color))
    if len(coords) != int(target_answer):
        raise ValueError("constructed player-capture board has wrong answer")
    board = add_material_fillers_preserving(
        rng=rng,
        board=board,
        scene_variant=str(axes.scene_variant),
        preserved_coords=(queen_coord,) + tuple(candidate_targets),
        expected_coords=coords,
        evaluator=lambda candidate: evaluate_player_captures(candidate, str(player_color)),
    )
    coords = evaluate_player_captures(board, str(player_color))
    return ChessSceneSample(
        board=board,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        construction_mode="direct_player_capture_count",
        player_color=str(player_color),
        capture_coords=coords,
        annotation_coords=coords,
        annotation_entity_ids=piece_entity_ids(board, coords),
        annotation_kind="piece",
        occupied_count=int(occupied_piece_count(board)),
    )


def sample_target_square_attacker_scene(
    *,
    rng,
    axes: ResolvedChessSceneAxes,
    attacker_color: str,
    target_answer: int,
    target_has_king: bool,
) -> ChessSceneSample:
    """Construct a marked target-square attacker-count scene."""

    for _attempt in range(120):
        mutable = [list(row) for row in empty_board()]
        target_coord = (int(rng.randint(2, 5)), int(rng.randint(2, 5)))
        defender_color = opponent(str(attacker_color))
        if bool(target_has_king):
            mutable[target_coord[0]][target_coord[1]] = ChessPiece(defender_color, "king")
        slots = list(attacker_slots_for_square(target_coord, str(attacker_color)))
        rng.shuffle(slots)
        selected: list[Tuple[Coord, str]] = []
        for coord, kind in slots:
            if len(selected) >= int(target_answer):
                break
            if mutable[int(coord[0])][int(coord[1])] is not None:
                continue
            if any(coord == other for other, _kind in selected):
                continue
            if not place_capped_piece(mutable=mutable, coord=coord, piece=ChessPiece(str(attacker_color), str(kind))):
                continue
            selected.append((coord, str(kind)))
        if len(selected) != int(target_answer):
            continue
        if bool(target_has_king):
            placed = False
            for _king_attempt in range(256):
                king_coord = random_empty_coord(rng, mutable)
                if king_coord == target_coord or (
                    abs(king_coord[0] - target_coord[0]) <= 1
                    and abs(king_coord[1] - target_coord[1]) <= 1
                ):
                    continue
                mutable[int(king_coord[0])][int(king_coord[1])] = ChessPiece(str(attacker_color), "king")
                placed = True
                break
            if not placed:
                continue
        else:
            try:
                place_non_adjacent_kings(rng=rng, mutable=mutable, first_color=str(attacker_color), second_color=defender_color, forbidden=(target_coord,))
            except ValueError:
                continue
        board = freeze_board(mutable)
        if bool(target_has_king):
            if not validate_square_chess_material(board):
                continue
        else:
            if not validate_square_chess_material(board):
                continue
        coords = evaluate_target_attackers(board, target_coord, str(attacker_color))
        if len(coords) != int(target_answer):
            continue
        board = add_material_fillers_preserving(
            rng=rng,
            board=board,
            scene_variant=str(axes.scene_variant),
            preserved_coords=(target_coord,) + tuple(coord for coord, _kind in selected),
            expected_coords=coords,
            evaluator=lambda candidate: evaluate_target_attackers(candidate, target_coord, str(attacker_color)),
        )
        coords = evaluate_target_attackers(board, target_coord, str(attacker_color))
        marked_piece = board[target_coord[0]][target_coord[1]]
        return ChessSceneSample(
            board=board,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            construction_mode="direct_target_square_attacker_count",
            player_color=str(defender_color if target_has_king else attacker_color),
            marked_coord=target_coord,
            target_coord=target_coord,
            marked_piece=marked_piece,
            attacker_coords=coords,
            annotation_coords=coords,
            annotation_entity_ids=piece_entity_ids(board, coords),
            annotation_kind="piece",
            occupied_count=int(occupied_piece_count(board)),
        )
    raise ValueError("failed to construct target-square attacker scene")


def sample_line_blocker_scene(
    *,
    rng,
    axes: ResolvedChessSceneAxes,
    slider_kind: str,
    target_answer: int,
) -> ChessSceneSample:
    """Construct an exact line-blocker scene for a marked sliding piece."""

    candidate_pairs: list[Tuple[Coord, Coord, Tuple[Coord, ...]]] = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            source = (int(row), int(col))
            for dr, dc in slider_directions(str(slider_kind)):
                for distance in range(max(2, int(target_answer) + 1), BOARD_SIZE):
                    target = (int(row + (dr * distance)), int(col + (dc * distance)))
                    if not in_bounds(*target):
                        break
                    between = coords_between(source, target)
                    if len(between) >= int(target_answer):
                        candidate_pairs.append((source, target, between))
    rng.shuffle(candidate_pairs)
    for source_coord, target_coord, between_coords in candidate_pairs[:160]:
        mutable = [list(row) for row in empty_board()]
        source_color = WHITE if int(rng.randrange(2)) == 0 else BLACK
        if not place_capped_piece(mutable=mutable, coord=source_coord, piece=ChessPiece(source_color, str(slider_kind))):
            continue
        blocker_positions = list(between_coords)
        rng.shuffle(blocker_positions)
        for coord in blocker_positions[: int(target_answer)]:
            if not place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, kinds=("queen", "rook", "bishop", "knight", "pawn")):
                break
        else:
            try:
                place_non_adjacent_kings(
                    rng=rng,
                    mutable=mutable,
                    first_color=source_color,
                    second_color=opponent(source_color),
                    forbidden=(source_coord, target_coord) + tuple(between_coords),
                )
            except ValueError:
                continue
            board = freeze_board(mutable)
            coords = evaluate_line_blockers(board, source_coord, target_coord, slider_kind=str(slider_kind))
            if len(coords) != int(target_answer):
                continue
            board = add_material_fillers_preserving(
                rng=rng,
                board=board,
                scene_variant=str(axes.scene_variant),
                preserved_coords=(source_coord, target_coord) + tuple(between_coords),
                expected_coords=coords,
                evaluator=lambda candidate: evaluate_line_blockers(candidate, source_coord, target_coord, slider_kind=str(slider_kind)),
            )
            coords = evaluate_line_blockers(board, source_coord, target_coord, slider_kind=str(slider_kind))
            return ChessSceneSample(
                board=board,
                scene_variant=str(axes.scene_variant),
                style_variant=str(axes.style_variant),
                construction_mode="direct_line_blocker_count",
                player_color=str(source_color),
                marked_coord=source_coord,
                target_coord=target_coord,
                marked_piece=board[int(source_coord[0])][int(source_coord[1])],
                blocker_coords=coords,
                annotation_coords=coords,
                annotation_entity_ids=piece_entity_ids(board, coords),
                annotation_kind="piece",
                occupied_count=int(occupied_piece_count(board)),
            )
    raise ValueError("failed to construct line-blocker scene")


def _place_knight_attacker_for_escape(
    *,
    rng,
    mutable: list[list[ChessPiece | None]],
    target_coord: Coord,
    king_coord: Coord,
    attacker_color: str,
    safe_coords: Tuple[Coord, ...],
    adjacent_coords: Tuple[Coord, ...],
) -> bool:
    candidates: list[Coord] = []
    for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
        origin = (int(target_coord[0]) - int(dr), int(target_coord[1]) - int(dc))
        if not in_bounds(*origin):
            continue
        if origin == king_coord or origin in adjacent_coords:
            continue
        if mutable[int(origin[0])][int(origin[1])] is not None:
            continue
        candidates.append(origin)
    rng.shuffle(candidates)
    for origin in candidates:
        if not place_capped_piece(mutable=mutable, coord=origin, piece=ChessPiece(str(attacker_color), "knight")):
            continue
        board = freeze_board(mutable)
        attacks_target = bool(evaluate_target_attackers(board, target_coord, str(attacker_color)))
        attacks_safe = any(origin in evaluate_target_attackers(board, safe_coord, str(attacker_color)) for safe_coord in safe_coords)
        if attacks_target and not attacks_safe:
            return True
        mutable[int(origin[0])][int(origin[1])] = None
    return False


def sample_king_escape_scene(
    *,
    rng,
    axes: ResolvedChessSceneAxes,
    target_answer: int,
) -> ChessSceneSample:
    """Construct a king-escape count scene with blocked and attacked unsafe cells."""

    for _attempt in range(160):
        mutable = [list(row) for row in empty_board()]
        king_color = WHITE if int(rng.randrange(2)) == 0 else BLACK
        opponent_color = opponent(king_color)
        king_coord = (int(rng.randint(2, 5)), int(rng.randint(2, 5)))
        mutable[king_coord[0]][king_coord[1]] = ChessPiece(king_color, "king")
        adjacent = [
            (king_coord[0] + dr, king_coord[1] + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]
        rng.shuffle(adjacent)
        safe_coords = tuple(adjacent[: int(target_answer)])
        unsafe_coords = [coord for coord in adjacent if coord not in safe_coords]
        attacked_unsafe = tuple(unsafe_coords[:1]) if unsafe_coords else ()
        blocked_unsafe = [coord for coord in unsafe_coords if coord not in attacked_unsafe]
        failed = False
        for coord in attacked_unsafe:
            if not _place_knight_attacker_for_escape(
                rng=rng,
                mutable=mutable,
                target_coord=coord,
                king_coord=king_coord,
                attacker_color=opponent_color,
                safe_coords=safe_coords,
                adjacent_coords=tuple(adjacent),
            ):
                failed = True
                break
        if failed:
            continue
        for coord in blocked_unsafe:
            if not place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, colors=(king_color,), kinds=("pawn", "knight")):
                failed = True
                break
        if failed:
            continue
        possible = [
            (row, col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if mutable[row][col] is None
            and not any((row, col) == c for c in adjacent)
            and (row, col) != king_coord
        ]
        rng.shuffle(possible)
        placed = False
        for coord in possible:
            if coord in adjacent or coord == king_coord:
                continue
            if any(abs(coord[0] - other[0]) <= 1 and abs(coord[1] - other[1]) <= 1 for other in (king_coord,)):
                continue
            mutable[int(coord[0])][int(coord[1])] = ChessPiece(opponent_color, "king")
            placed = True
            break
        if not placed:
            continue
        board = freeze_board(mutable)
        coords = evaluate_king_escapes(board, king_coord)
        if len(coords) != int(target_answer):
            continue
        if unsafe_coords and not any(
            opponent_attackers_after_king_move(board, king_coord, coord, king_color)
            for coord in unsafe_coords
            if board[int(coord[0])][int(coord[1])] is None
        ):
            continue
        if same_color_slider_points_to_any(board, color=king_color, targets=coords):
            continue
        board = add_material_fillers_preserving(
            rng=rng,
            board=board,
            scene_variant=str(axes.scene_variant),
            preserved_coords=(king_coord,) + tuple(adjacent),
            expected_coords=coords,
            evaluator=lambda candidate: evaluate_king_escapes(candidate, king_coord),
            avoid_adjacent_to=king_coord,
        )
        coords = evaluate_king_escapes(board, king_coord)
        if same_color_slider_points_to_any(board, color=king_color, targets=coords):
            continue
        return ChessSceneSample(
            board=board,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            construction_mode="direct_king_escape_count",
            player_color=str(king_color),
            marked_coord=king_coord,
            marked_piece=board[king_coord[0]][king_coord[1]],
            destination_coords=coords,
            annotation_coords=coords,
            annotation_entity_ids=cell_entity_ids(coords),
            annotation_kind="cell",
            occupied_count=int(occupied_piece_count(board)),
        )
    raise ValueError("failed to construct king-escape scene")


def resolve_checkmate_answer_label(rng, *, params: Mapping[str, Any], labels: Sequence[str]) -> Tuple[str, Dict[str, float]]:
    """Resolve the correct visible option label for a checkmate scene."""

    labels_tuple = tuple(str(label) for label in labels)
    explicit = params.get("answer_option_label", params.get("target_label"))
    if explicit is not None:
        label = str(explicit).strip().upper()
        if label not in labels_tuple:
            raise ValueError(f"answer_option_label={label!r} is not available for labels {labels_tuple}")
        return label, string_probability_map(labels_tuple, label)
    label = str(uniform_choice(rng, labels_tuple))
    return label, string_probability_map(labels_tuple, label)


def base_checkmate_board(*, player_color: str, mirror_columns: bool) -> Tuple[Board, Coord, Coord]:
    """Return a compact queen mate-in-one pattern and the mating move."""

    def maybe_mirror(coord: Coord) -> Coord:
        return (int(coord[0]), int(BOARD_SIZE - 1 - int(coord[1]))) if bool(mirror_columns) else (int(coord[0]), int(coord[1]))

    mutable = [list(row) for row in empty_board()]
    if str(player_color) == WHITE:
        defender_color = BLACK
        defender_king = maybe_mirror((0, 7))
        attacker_king = maybe_mirror((2, 5))
        queen_source = maybe_mirror((2, 6))
        queen_destination = maybe_mirror((1, 6))
    else:
        defender_color = WHITE
        defender_king = maybe_mirror((7, 7))
        attacker_king = maybe_mirror((5, 5))
        queen_source = maybe_mirror((5, 6))
        queen_destination = maybe_mirror((6, 6))
    mutable[defender_king[0]][defender_king[1]] = ChessPiece(defender_color, "king")
    mutable[attacker_king[0]][attacker_king[1]] = ChessPiece(str(player_color), "king")
    mutable[queen_source[0]][queen_source[1]] = ChessPiece(str(player_color), "queen")
    return freeze_board(mutable), queen_source, queen_destination


def checkmate_board_valid(board: Board, *, player_color: str, correct_source: Coord, correct_destination: Coord) -> bool:
    """Return whether the position preserves the intended mate-in-one contract."""

    defender_color = opponent(str(player_color))
    if find_king(board, str(player_color)) is None or find_king(board, defender_color) is None:
        return False
    if is_king_in_check(board, str(player_color)) or is_king_in_check(board, defender_color):
        return False
    return move_checkmates(board, correct_source, correct_destination)


def add_checkmate_fillers(
    *,
    rng,
    board: Board,
    player_color: str,
    correct_source: Coord,
    correct_destination: Coord,
    scene_variant: str,
) -> Board:
    """Add filler material while preserving one checkmate move."""

    lower, upper = (6, 9) if str(scene_variant) == "sparse_board" else (9, 13)
    desired_count = int(rng.randint(int(lower), int(upper)))
    mutable = [list(row) for row in board]
    protected = {tuple(correct_source), tuple(correct_destination)}
    attempts = 0
    while occupied_piece_count(mutable) < desired_count and attempts < 360:
        attempts += 1
        coord = random_empty_coord(rng, mutable)
        if coord in protected:
            continue
        if not place_capped_random_piece(rng=rng, mutable=mutable, coord=coord, kinds=("rook", "bishop", "knight", "pawn")):
            continue
        candidate = freeze_board(mutable)
        if not validate_square_chess_material(candidate) or not checkmate_board_valid(
            candidate,
            player_color=str(player_color),
            correct_source=correct_source,
            correct_destination=correct_destination,
        ):
            mutable[int(coord[0])][int(coord[1])] = None
    final_board = freeze_board(mutable)
    if not validate_square_chess_material(final_board):
        raise ValueError("checkmate board is not material-plausible")
    return final_board


def option_from_move(board: Board, *, label: str, move: Tuple[Coord, Coord]) -> ChessMoveOption:
    """Build one visible option record from a legal move."""

    source, destination = move
    piece = board[int(source[0])][int(source[1])]
    if piece is None:
        raise ValueError("cannot build a move option from an empty source square")
    return ChessMoveOption(label=str(label), source=tuple(source), destination=tuple(destination), piece=piece)


def sample_checkmate_scene(
    *,
    instance_seed: int,
    rng,
    params: Mapping[str, Any],
    axes: ResolvedChessSceneAxes,
    option_count: int,
    option_label_support: Sequence[str],
    answer_label: str,
) -> ChessCheckmateSample:
    """Construct one chess mate-in-one option-selection scene."""

    player_color = resolve_player_color(rng, params=params)
    mirror_columns = bool(rng.randrange(2))
    for _attempt in range(96):
        board, correct_source, correct_destination = base_checkmate_board(player_color=str(player_color), mirror_columns=bool(mirror_columns))
        if not checkmate_board_valid(board, player_color=str(player_color), correct_source=correct_source, correct_destination=correct_destination):
            mirror_columns = not bool(mirror_columns)
            continue
        board = add_checkmate_fillers(
            rng=rng,
            board=board,
            player_color=str(player_color),
            correct_source=correct_source,
            correct_destination=correct_destination,
            scene_variant=str(axes.scene_variant),
        )
        legal_moves = list(legal_chess_moves_for_color(board, str(player_color)))
        correct_move = (tuple(correct_source), tuple(correct_destination))
        if correct_move not in legal_moves or not move_checkmates(board, correct_source, correct_destination):
            mirror_columns = not bool(mirror_columns)
            continue
        distractor_moves = [move for move in legal_moves if tuple(move) != correct_move and not move_checkmates(board, move[0], move[1])]
        rng.shuffle(distractor_moves)
        if len(distractor_moves) < int(option_count) - 1:
            mirror_columns = not bool(mirror_columns)
            continue
        labels = list(str(label) for label in option_label_support)
        correct_index = labels.index(str(answer_label))
        options: list[ChessMoveOption] = []
        distractor_iter = iter(distractor_moves[: int(option_count) - 1])
        for index, label in enumerate(labels):
            if int(index) == int(correct_index):
                options.append(option_from_move(board, label=str(label), move=correct_move))
            else:
                options.append(option_from_move(board, label=str(label), move=next(distractor_iter)))
        mate_labels = [option.label for option in options if move_checkmates(board, option.source, option.destination)]
        if mate_labels != [str(answer_label)]:
            mirror_columns = not bool(mirror_columns)
            continue
        defender_color = opponent(str(player_color))
        defender_king = find_king(board, defender_color)
        if defender_king is None:
            raise ValueError("checkmate scene lacks defender king")
        return ChessCheckmateSample(
            board=board,
            player_color=str(player_color),
            defender_color=str(defender_color),
            correct_option=options[int(correct_index)],
            options=tuple(options),
            defender_king_coord=defender_king,
            occupied_count=int(occupied_piece_count(board)),
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            construction_mode="queen_corner_mate_in_one_with_legal_distractors",
            option_count=int(option_count),
            option_label_support=tuple(str(label) for label in option_label_support),
            extra={"mirror_columns": bool(mirror_columns)},
        )
    raise ValueError("failed to construct chess checkmate move-label scene")

__all__ = [
    "resolve_chess_scene_axes",
    "resolve_checkmate_answer_label",
    "resolve_integer_axis",
    "resolve_player_color",
    "resolve_scene_axis",
    "resolve_string_axis",
    "resolve_target_answer",
    "sample_checkmate_scene",
    "sample_king_escape_scene",
    "sample_line_blocker_scene",
    "sample_marked_piece_destination_scene",
    "sample_piece_count_scene",
    "sample_player_capture_scene",
    "sample_target_square_attacker_scene",
    "serialize_board",
    "string_probability_map",
    "uniform_probability_map",
]
