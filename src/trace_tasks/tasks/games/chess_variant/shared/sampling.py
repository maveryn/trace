"""Identity-free sampling helpers for chess-variant games tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.piece_board_rules import (
    BLACK,
    BOARD_SIZE,
    NON_KING_PIECE_KINDS,
    WHITE,
    Board,
    ChessPiece,
    Coord,
    empty_board,
    freeze_board,
    occupied_piece_count,
    opponent,
    sample_material_piece,
    validate_square_chess_material,
)
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .defaults import (
    FALLBACK_GENERATION_DEFAULTS,
    SCENE_ID,
    SUPPORTED_RULE_FAMILIES,
    SUPPORTED_SCENE_VARIANTS,
    SUPPORTED_STYLE_VARIANTS,
)
from .rules import (
    all_empty_board_destinations,
    directions_for_rule,
    evaluate_by_semantic_query,
    evaluate_marked_piece_board,
    max_possible_marked_destination_answer,
    ray_coords,
    with_destination_annotation,
)
from .state import ChessVariantSample, ChessVariantSceneAxes


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


def range_support_for_rule(rule_family: str) -> tuple[int, ...]:
    """Return valid range values for one visible rule family."""

    if str(rule_family) == "straight_or_diagonal_range":
        return tuple(int(v) for v in _generation_default("queen_range_k_support"))
    if str(rule_family).endswith("_range"):
        return tuple(int(v) for v in _generation_default("range_k_support"))
    return (0,)


def resolve_chess_variant_scene_axes(instance_seed: int, *, params: Mapping[str, Any]) -> ChessVariantSceneAxes:
    """Resolve shared scene, style, rule, and range axes."""

    rule_family, rule_family_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.chess_variant.rule_family",
        explicit_key="rule_family",
        weights_key="rule_family_weights",
        balance_flag_key="balanced_rule_family_sampling",
        supported_values=SUPPORTED_RULE_FAMILIES,
    )
    scene_variant, scene_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.chess_variant.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_values=SUPPORTED_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = resolve_scene_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="games.chess_variant.style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_values=SUPPORTED_STYLE_VARIANTS,
    )
    range_support = range_support_for_rule(str(rule_family))
    if len(range_support) == 1:
        range_k = int(range_support[0])
        range_k_probabilities = {str(range_k): 1.0}
    else:
        range_k, range_k_probabilities = resolve_integer_choice(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults={**dict(_GEN_DEFAULTS), "range_k_support": list(range_support)},
            support_key="range_k_support",
            explicit_key="range_k",
            fallback_support=range_support,
            namespace=f"games.chess_variant.range_k.{str(rule_family)}",
            balanced_flag_key="balanced_range_k_sampling",
            namespace_support_permutation=True,
        )
    return ChessVariantSceneAxes(
        rule_family=str(rule_family),
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        range_k=int(range_k),
        rule_family_probabilities=dict(rule_family_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        range_k_probabilities=dict(range_k_probabilities),
    )


def resolve_task_target_answer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support_key: str,
    fallback_support: Sequence[int],
    possible_max: int,
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve a task-owned answer support after scene feasibility filtering."""

    configured_support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
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
        gen_defaults={**dict(_GEN_DEFAULTS), str(support_key): list(target_support)},
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=target_support,
        namespace=str(namespace),
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return int(target_answer), tuple(int(v) for v in target_support), dict(target_answer_probabilities)


def _random_marked_coord(rng, *, rule_family: str, range_k: int, minimum_destinations: int) -> Coord:
    coords = [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE)]
    rng.shuffle(coords)
    for coord in coords:
        if len(all_empty_board_destinations(str(rule_family), int(range_k), coord)) >= int(minimum_destinations):
            return tuple(coord)
    raise ValueError("no marked coordinate has enough destinations")


def _place_variant_piece(
    *,
    rng,
    mutable: list[list[ChessPiece | None]],
    coord: Coord,
    color: str | None = None,
    kinds: Sequence[str] = NON_KING_PIECE_KINDS,
) -> bool:
    """Place one material-capped chess symbol for a rule-card variant board."""

    row, col = int(coord[0]), int(coord[1])
    if mutable[row][col] is not None:
        return False
    colors = (str(color),) if color is not None else (WHITE, BLACK)
    piece = sample_material_piece(
        rng,
        freeze_board(mutable),
        colors=colors,
        kinds=kinds,
        row=int(row),
        enforce_standard_pawn_rows=True,
    )
    if piece is None:
        return False
    mutable[row][col] = piece
    return True


def _add_required_kings_preserving(
    *,
    rng,
    board: Board,
    axes: ChessVariantSceneAxes,
    target_answer: int,
    destination_mode: str | None,
    marked: Coord | None,
) -> Board:
    """Add one king per side without changing the active answer."""

    mutable = [list(row) for row in board]
    for color in (WHITE, BLACK):
        if any(piece is not None and str(piece.color) == str(color) and str(piece.kind) == "king" for row in mutable for piece in row):
            continue
        candidates = [
            (row, col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if mutable[row][col] is None
        ]
        rng.shuffle(candidates)
        placed = False
        for coord in candidates:
            mutable[int(coord[0])][int(coord[1])] = ChessPiece(str(color), "king")
            candidate = freeze_board(mutable)
            if not validate_square_chess_material(
                candidate,
                require_both_kings=False,
                enforce_standard_pawn_rows=True,
                enforce_non_adjacent_kings=True,
            ):
                mutable[int(coord[0])][int(coord[1])] = None
                continue
            try:
                evaluation = evaluate_by_semantic_query(
                    candidate,
                    destination_mode=destination_mode,
                    marked_coord=marked,
                    rule_family=str(axes.rule_family),
                    range_k=int(axes.range_k),
                )
            except ValueError:
                evaluation = None
            if evaluation is None or int(evaluation.answer) != int(target_answer):
                mutable[int(coord[0])][int(coord[1])] = None
                continue
            placed = True
            break
        if not placed:
            raise ValueError("failed to add required chess-variant kings without changing answer")
    final = freeze_board(mutable)
    if not validate_square_chess_material(final):
        raise ValueError("chess-variant board is not material-plausible")
    return final


def _random_composition_with_caps(rng, *, total: int, caps: Sequence[int]) -> tuple[int, ...] | None:
    counts = [0 for _ in caps]
    remaining = int(total)
    order = list(range(len(caps)))
    rng.shuffle(order)
    for index in order:
        if remaining <= 0:
            break
        max_here = min(int(caps[index]), int(remaining))
        if max_here <= 0:
            continue
        value = int(rng.randint(0, max_here))
        counts[index] = int(value)
        remaining -= int(value)
    while remaining > 0:
        candidates = [i for i, cap in enumerate(caps) if counts[i] < int(cap)]
        if not candidates:
            return None
        index = int(rng.choice(candidates))
        counts[index] += 1
        remaining -= 1
    return tuple(int(v) for v in counts)


def _construct_move_board(*, rng, axes: ChessVariantSceneAxes, target_answer: int) -> tuple[Board, Coord]:
    marked_color = WHITE if int(rng.randrange(2)) == 0 else BLACK
    marked = _random_marked_coord(
        rng,
        rule_family=str(axes.rule_family),
        range_k=int(axes.range_k),
        minimum_destinations=int(target_answer),
    )
    mutable = [list(row) for row in empty_board()]
    if not _place_variant_piece(rng=rng, mutable=mutable, coord=marked, color=marked_color):
        raise ValueError("failed to place marked chess-variant piece")
    if str(axes.rule_family).endswith("_range"):
        rays = [ray_coords(marked, direction, max_steps=int(axes.range_k)) for direction in directions_for_rule(str(axes.rule_family))]
        caps = [len(ray) for ray in rays]
        counts = _random_composition_with_caps(rng, total=int(target_answer), caps=caps)
        if counts is None:
            raise ValueError("failed to distribute move destinations")
        for ray, legal_count in zip(rays, counts):
            for index, coord in enumerate(ray):
                if index < int(legal_count):
                    continue
                if not _place_variant_piece(rng=rng, mutable=mutable, coord=coord, color=marked_color):
                    raise ValueError("failed to place chess-variant range blocker")
                break
    else:
        potentials = list(all_empty_board_destinations(str(axes.rule_family), int(axes.range_k), marked))
        if len(potentials) < int(target_answer):
            raise ValueError("not enough leaper destinations")
        rng.shuffle(potentials)
        legal = set(tuple(coord) for coord in potentials[: int(target_answer)])
        for coord in potentials:
            if tuple(coord) in legal:
                continue
            if not _place_variant_piece(rng=rng, mutable=mutable, coord=coord, color=marked_color):
                raise ValueError("failed to place chess-variant leaper blocker")
    return freeze_board(mutable), marked


def _construct_capture_board(*, rng, axes: ChessVariantSceneAxes, target_answer: int) -> tuple[Board, Coord]:
    """Build a minimal board with exactly the requested capture destinations."""

    marked_color = WHITE if int(rng.randrange(2)) == 0 else BLACK
    marked = _random_marked_coord(
        rng,
        rule_family=str(axes.rule_family),
        range_k=int(axes.range_k),
        minimum_destinations=int(target_answer),
    )
    mutable = [list(row) for row in empty_board()]
    if not _place_variant_piece(rng=rng, mutable=mutable, coord=marked, color=marked_color):
        raise ValueError("failed to place marked chess-variant piece")
    if str(axes.rule_family).endswith("_range"):
        rays = [ray_coords(marked, direction, max_steps=int(axes.range_k)) for direction in directions_for_rule(str(axes.rule_family))]
        available = [ray for ray in rays if ray]
        if len(available) < int(target_answer):
            raise ValueError("not enough capture rays")
        rng.shuffle(available)
        capture_rays = set(tuple(ray[0]) for ray in available[: int(target_answer)])
        for ray in rays:
            if not ray:
                continue
            if tuple(ray[0]) in capture_rays:
                distance = int(rng.randint(1, len(ray)))
                target = ray[distance - 1]
                if not _place_variant_piece(rng=rng, mutable=mutable, coord=target, color=opponent(marked_color)):
                    raise ValueError("failed to place chess-variant capture target")
                continue
            if int(rng.randrange(2)) == 0:
                blocker = ray[int(rng.randint(0, len(ray) - 1))]
                if not _place_variant_piece(rng=rng, mutable=mutable, coord=blocker, color=marked_color):
                    raise ValueError("failed to place chess-variant capture blocker")
    else:
        potentials = list(all_empty_board_destinations(str(axes.rule_family), int(axes.range_k), marked))
        if len(potentials) < int(target_answer):
            raise ValueError("not enough leaper capture squares")
        rng.shuffle(potentials)
        captures = set(tuple(coord) for coord in potentials[: int(target_answer)])
        for coord in potentials:
            piece_color = opponent(marked_color) if tuple(coord) in captures else marked_color
            if not _place_variant_piece(rng=rng, mutable=mutable, coord=coord, color=piece_color):
                raise ValueError("failed to place chess-variant leaper capture piece")
    return freeze_board(mutable), marked


def _desired_piece_count(rng, scene_variant: str) -> int:
    if str(scene_variant) == "crowded_board":
        return int(
            rng.randint(
                int(_generation_default("crowded_min_occupied_count")),
                int(_generation_default("crowded_max_occupied_count")),
            )
        )
    return int(
        rng.randint(
            int(_generation_default("sparse_min_occupied_count")),
            int(_generation_default("sparse_max_occupied_count")),
        )
    )


def _add_fillers_preserving(
    *,
    rng,
    board: Board,
    marked: Coord,
    axes: ChessVariantSceneAxes,
    destination_mode: str,
    target_answer: int,
    desired_count: int,
) -> Board:
    """Add visual clutter while preserving the marked-piece destination answer."""

    mutable = [list(row) for row in board]
    attempts = 0
    while occupied_piece_count(mutable) < int(desired_count) and attempts < 640:
        attempts += 1
        empties = [(row, col) for row in range(BOARD_SIZE) for col in range(BOARD_SIZE) if mutable[row][col] is None]
        if not empties:
            break
        coord = tuple(rng.choice(empties))
        if not _place_variant_piece(rng=rng, mutable=mutable, coord=coord):
            break
        frozen = freeze_board(mutable)
        if not validate_square_chess_material(
            frozen,
            require_both_kings=False,
            enforce_standard_pawn_rows=True,
            enforce_non_adjacent_kings=True,
        ):
            mutable[int(coord[0])][int(coord[1])] = None
            continue
        candidate = with_destination_annotation(
            evaluate_marked_piece_board(
                frozen,
                marked_coord=marked,
                rule_family=str(axes.rule_family),
                range_k=int(axes.range_k),
            ),
            destination_mode=str(destination_mode),
        )
        if int(candidate.answer) != int(target_answer):
            mutable[int(coord[0])][int(coord[1])] = None
    return freeze_board(mutable)


def sample_marked_destination_scene(
    *,
    rng,
    axes: ChessVariantSceneAxes,
    destination_mode: str,
    target_answer: int,
) -> ChessVariantSample:
    """Sample a marked-piece board, then validate the exact destination-count answer."""

    for _ in range(240):
        try:
            if str(destination_mode) in {"move", "empty"}:
                board, marked = _construct_move_board(rng=rng, axes=axes, target_answer=int(target_answer))
            elif str(destination_mode) == "capture":
                board, marked = _construct_capture_board(rng=rng, axes=axes, target_answer=int(target_answer))
            else:
                raise ValueError(f"unsupported destination mode: {destination_mode}")
            board = _add_required_kings_preserving(
                rng=rng,
                board=board,
                axes=axes,
                target_answer=int(target_answer),
                destination_mode=str(destination_mode),
                marked=marked,
            )
            desired_count = max(_desired_piece_count(rng, axes.scene_variant), occupied_piece_count(board))
            board = _add_fillers_preserving(
                rng=rng,
                board=board,
                marked=marked,
                axes=axes,
                destination_mode=str(destination_mode),
                target_answer=int(target_answer),
                desired_count=desired_count,
            )
            evaluation = with_destination_annotation(
                evaluate_marked_piece_board(
                    board,
                    marked_coord=marked,
                    rule_family=str(axes.rule_family),
                    range_k=int(axes.range_k),
                ),
                destination_mode=str(destination_mode),
            )
            if int(evaluation.answer) != int(target_answer):
                continue
            if not validate_square_chess_material(board):
                continue
            return ChessVariantSample(
                board=board,
                evaluation=evaluation,
                occupied_count=occupied_piece_count(board),
                construction_mode="direct_rule_constrained_board",
                scene_variant=str(axes.scene_variant),
                style_variant=str(axes.style_variant),
            )
        except ValueError:
            continue
    raise ValueError("failed to sample requested chess-variant marked-piece board")


__all__ = [
    "max_possible_marked_destination_answer",
    "range_support_for_rule",
    "resolve_chess_variant_scene_axes",
    "resolve_task_target_answer",
    "sample_marked_destination_scene",
]
