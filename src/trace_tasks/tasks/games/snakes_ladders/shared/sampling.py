"""Scene-local sampling primitives for Snakes and Ladders tasks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support

from .defaults import GEN_DEFAULTS
from .rules import board_last_square, square_to_cell_id, square_to_coord
from .state import SnakesLaddersJump


def select_integer_axis(
    params: Mapping[str, Any],
    *,
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    instance_seed: int,
    namespace: str,
    balanced_flag_key: str,
    max_value: int | None = None,
    use_instance_seed_cycle: bool = False,
) -> tuple[int, dict[str, float]]:
    """Select one integer from config/params, optionally capped by board size."""

    support = resolve_integer_support(params, gen_defaults=GEN_DEFAULTS, key=str(support_key), fallback=fallback_support)
    if max_value is not None:
        support = tuple(int(value) for value in support if int(value) <= int(max_value))
    if not support:
        raise ValueError(f"{support_key} has no values compatible with max_value={max_value}")
    choice_params = dict(params)
    choice_params[str(support_key)] = list(int(value) for value in support)
    return resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=choice_params,
        gen_defaults=GEN_DEFAULTS,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=support,
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        use_instance_seed_cycle=bool(use_instance_seed_cycle),
    )


def move_jump_probability(params: Mapping[str, Any], fallback: float) -> float:
    """Resolve the chance that a move-outcome sample uses a forced jump."""

    return float(params.get("move_outcome_jump_probability", group_default(GEN_DEFAULTS, "move_outcome_jump_probability", float(fallback))))


def make_jump(*, kind: str, start: int, end: int) -> SnakesLaddersJump:
    """Return one validated jump."""

    kind_text = str(kind)
    return SnakesLaddersJump(
        jump_id=f"{kind_text}_{int(start)}_{int(end)}",
        kind=kind_text,
        start_square=int(start),
        end_square=int(end),
    )


def random_jump_endpoint(rng, *, kind: str, start: int, board_side: int) -> int | None:
    """Return one endpoint for a random visual jump, if feasible."""

    side = int(board_side)
    last_square = board_last_square(side)
    min_span = max(4, int(side) - 1)
    max_span = max(min_span, int(side) * 4)
    if str(kind) == "ladder":
        low = int(start) + int(min_span)
        high = min(int(last_square) - 1, int(start) + int(max_span))
        if low > high:
            return None
        return int(rng.randint(low, high))
    high = int(start) - int(min_span)
    low = max(2, int(start) - int(max_span))
    if low > high:
        return None
    return int(rng.randint(low, high))


def target_jump_counts(board_side: int) -> Tuple[int, int]:
    """Return the desired ladder/snake counts for a board side length."""

    if int(board_side) <= 7:
        return (1, 1)
    return (2, 2)


def jump_visual_segment(jump: SnakesLaddersJump, *, board_side: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the board-coordinate segment used to draw one jump."""

    start_row, start_col = square_to_coord(int(jump.start_square), board_side=int(board_side))
    end_row, end_col = square_to_coord(int(jump.end_square), board_side=int(board_side))
    return ((float(start_col), float(start_row)), (float(end_col), float(end_row)))


def _orientation(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    eps = 1e-9
    return (
        min(a[0], c[0]) - eps <= b[0] <= max(a[0], c[0]) + eps
        and min(a[1], c[1]) - eps <= b[1] <= max(a[1], c[1]) + eps
        and abs(_orientation(a, b, c)) <= eps
    )


def segments_intersect(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
) -> bool:
    """Return true if two board-coordinate segments intersect."""

    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    eps = 1e-9
    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True
    return _on_segment(a, c, b) or _on_segment(a, d, b) or _on_segment(c, a, d) or _on_segment(c, b, d)


def jump_conflicts_with_existing(
    candidate: SnakesLaddersJump,
    *,
    existing: Sequence[SnakesLaddersJump],
    board_side: int,
) -> bool:
    """Return true if a candidate would make connector layout ambiguous."""

    candidate_squares = {int(candidate.start_square), int(candidate.end_square)}
    candidate_start, candidate_end = jump_visual_segment(candidate, board_side=int(board_side))
    for jump in existing:
        if candidate_squares & {int(jump.start_square), int(jump.end_square)}:
            return True
        existing_start, existing_end = jump_visual_segment(jump, board_side=int(board_side))
        if segments_intersect(candidate_start, candidate_end, existing_start, existing_end):
            return True
    return False


def add_random_jumps(
    *,
    rng,
    jumps: Sequence[SnakesLaddersJump],
    board_side: int,
    protected_starts: Sequence[int],
    protected_range: Sequence[int] = tuple(),
    target_ladders: int = 3,
    target_snakes: int = 3,
) -> Tuple[SnakesLaddersJump, ...]:
    """Add harmless visual snakes/ladders while avoiding protected starts."""

    last_square = board_last_square(int(board_side))
    result = list(jumps)
    used_starts = {int(jump.start_square) for jump in result}
    protected = set(int(value) for value in protected_starts) | set(int(value) for value in protected_range)

    def _count(kind: str) -> int:
        return sum(1 for jump in result if str(jump.kind) == str(kind))

    for kind, target in (("ladder", int(target_ladders)), ("snake", int(target_snakes))):
        for _attempt in range(140):
            if _count(kind) >= int(target):
                break
            if str(kind) == "ladder":
                start = int(rng.randint(2, max(2, int(last_square) - max(2, int(board_side)))))
            else:
                start = int(rng.randint(max(3, int(board_side) + 1), int(last_square) - 1))
            if start in used_starts or start in protected:
                continue
            end = random_jump_endpoint(rng, kind=kind, start=start, board_side=int(board_side))
            if end is None:
                continue
            candidate = make_jump(kind=kind, start=start, end=int(end))
            if jump_conflicts_with_existing(candidate, existing=tuple(result), board_side=int(board_side)):
                continue
            result.append(candidate)
            used_starts.add(int(start))
    return tuple(result)


def valid_jump_starts(*, kind: str, board_side: int) -> Tuple[int, ...]:
    """Return jump-start squares with at least one valid endpoint."""

    side = int(board_side)
    last_square = board_last_square(side)
    min_span = max(4, int(side) - 1)
    if str(kind) == "ladder":
        return tuple(range(2, max(1, int(last_square) - int(min_span)) + 1))
    return tuple(range(max(3, int(min_span) + 2), int(last_square)))


def jump_start_entity_ids(
    *,
    jumps: Sequence[SnakesLaddersJump],
    kind: str,
) -> Tuple[str, ...]:
    """Return square ids for every visible jump start of the requested kind."""

    squares = sorted(
        int(jump.start_square)
        for jump in jumps
        if str(jump.kind) == str(kind)
    )
    return tuple(square_to_cell_id(int(square)) for square in squares)


def append_jumps_from_allowed_starts(
    *,
    rng,
    jumps: Sequence[SnakesLaddersJump],
    board_side: int,
    kind: str,
    allowed_starts: Sequence[int],
    count: int,
    required: bool,
) -> Tuple[SnakesLaddersJump, ...] | None:
    """Append up to ``count`` non-conflicting jumps from allowed starts."""

    result = list(jumps)
    used_starts = {int(jump.start_square) for jump in result}
    candidates = [int(value) for value in allowed_starts if int(value) not in used_starts]
    rng.shuffle(candidates)
    target_count = max(0, int(count))
    for start in candidates:
        if target_count <= 0:
            break
        for _endpoint_attempt in range(12):
            end = random_jump_endpoint(rng, kind=str(kind), start=int(start), board_side=int(board_side))
            if end is None:
                continue
            candidate = make_jump(kind=str(kind), start=int(start), end=int(end))
            if jump_conflicts_with_existing(candidate, existing=tuple(result), board_side=int(board_side)):
                continue
            result.append(candidate)
            used_starts.add(int(start))
            target_count -= 1
            break
    if target_count > 0 and bool(required):
        return None
    return tuple(result)


def nearby_die_region(start_square: int, *, board_side: int) -> Tuple[int, ...]:
    """Return local squares kept visually clear around token and die landings."""

    last_square = board_last_square(int(board_side))
    low = max(2, int(start_square) - 2)
    high = min(int(last_square) - 1, int(start_square) + max(4, int(board_side)))
    return tuple(range(low, high + 1))


def sample_start_for_direct_target(rng, *, target_final: int, die_value: int, board_side: int) -> int | None:
    """Choose a start square that directly lands on the target final square."""

    last_square = board_last_square(int(board_side))
    start = int(target_final) - int(die_value)
    if start >= 1:
        return int(start)
    candidates = [
        square
        for square in range(1, int(last_square) + 1)
        if square + int(die_value) > int(last_square) and square == int(target_final)
    ]
    if candidates:
        return int(rng.choice(tuple(candidates)))
    return None


def construct_single_roll_outcome_sample(
    *,
    rng,
    axes,
    target_final: int,
    die_value: int,
    jump_probability: float,
):
    """Construct a board where one shown die roll ends on the target square."""

    from .rules import apply_die_roll, square_to_cell_id, validate_snakes_ladders_sample
    from .state import SnakesLaddersJump, SnakesLaddersSample

    board_side = int(axes.board_side)
    last_square = board_last_square(board_side)
    for _attempt in range(180):
        use_jump = bool(rng.random() < max(0.0, min(1.0, float(jump_probability))))
        jumps: list[SnakesLaddersJump] = []
        start_square: int | None = None
        landing_square: int | None = None
        if use_jump:
            possible_sources: list[int] = []
            for source in range(2, int(last_square)):
                if int(source) == int(target_final):
                    continue
                if abs(int(source) - int(target_final)) < max(4, int(board_side) - 1):
                    continue
                start = int(source) - int(die_value)
                if start >= 1 and start + int(die_value) == int(source):
                    possible_sources.append(int(source))
            if possible_sources:
                landing_square = int(rng.choice(tuple(possible_sources)))
                start_square = int(landing_square) - int(die_value)
                kind = "ladder" if int(target_final) > int(landing_square) else "snake"
                jumps.append(make_jump(kind=kind, start=int(landing_square), end=int(target_final)))
        if start_square is None:
            direct_start = sample_start_for_direct_target(
                rng,
                target_final=int(target_final),
                die_value=int(die_value),
                board_side=int(board_side),
            )
            if direct_start is None:
                continue
            start_square = int(direct_start)
            landing_square = int(target_final)

        target_ladders, target_snakes = target_jump_counts(int(board_side))
        jumps_tuple = add_random_jumps(
            rng=rng,
            jumps=tuple(jumps),
            board_side=int(board_side),
            protected_starts=(int(landing_square), int(start_square)),
            protected_range=nearby_die_region(int(start_square), board_side=int(board_side)),
            target_ladders=int(target_ladders),
            target_snakes=int(target_snakes),
        )
        move = apply_die_roll(int(start_square), int(die_value), jumps_tuple, board_side=int(board_side))
        if int(move.final_square) != int(target_final):
            continue
        sample = SnakesLaddersSample(
            mode="single_die_move",
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            board_side=int(board_side),
            answer=int(target_final),
            start_square=int(start_square),
            jumps=tuple(jumps_tuple),
            move=move,
            horizon_roll_count=None,
            optimal_route=tuple(),
            annotation_entity_ids=tuple(dict.fromkeys((square_to_cell_id(int(start_square)), square_to_cell_id(int(move.final_square))))),
            construction_mode="single_die_move_outcome",
        )
        validate_snakes_ladders_sample(sample)
        return sample
    raise ValueError("failed to sample Snakes and Ladders move-outcome scene")


__all__ = [
    "add_random_jumps",
    "append_jumps_from_allowed_starts",
    "construct_single_roll_outcome_sample",
    "jump_start_entity_ids",
    "jump_conflicts_with_existing",
    "jump_visual_segment",
    "make_jump",
    "move_jump_probability",
    "nearby_die_region",
    "random_jump_endpoint",
    "sample_start_for_direct_target",
    "select_integer_axis",
    "segments_intersect",
    "target_jump_counts",
    "valid_jump_starts",
]
