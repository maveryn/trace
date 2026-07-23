"""Sampling helpers for the Backgammon games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis

from .rules import (
    compute_single_die_destinations,
    destination_for_player,
    opponent_for_player,
    pip_count_contributions_for_player,
    point_matches_stack_state,
    target_destinations_for_status,
    target_points_for_stack_state,
    validate_backgammon_sample,
)
from .state import (
    ACTIVE_PLAYERS,
    DESTINATION_STATUS_BLOCKED,
    DESTINATION_STATUS_HIT,
    DESTINATION_STATUS_LEGAL,
    DESTINATION_STATUSES,
    POINT_IDS,
    PLAYER_BLACK,
    PLAYER_WHITE,
    STACK_STATE_SINGLE,
    STACK_STATE_TWO_OR_MORE,
    STACK_STATES,
    SUPPORTED_BACKGAMMON_SCENE_VARIANTS,
    SUPPORTED_BACKGAMMON_STYLE_VARIANTS,
    BackgammonPoint,
    BackgammonSample,
    empty_points,
)

_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    "backgammon",
)


@dataclass(frozen=True)
class ResolvedBackgammonAxes:
    """Resolved semantic and visual axes for one Backgammon instance."""

    scene_variant: str
    style_variant: str
    active_player: str
    scene_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    active_player_probabilities: Dict[str, float]


def _resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=[str(value) for value in supported],
    )


def resolve_backgammon_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
) -> ResolvedBackgammonAxes:
    """Resolve visual and active-player axes for one Backgammon instance."""

    scene_variant, scene_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_BACKGAMMON_SCENE_VARIANTS,
    )
    style_variant, style_variant_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_BACKGAMMON_STYLE_VARIANTS,
    )
    active_player, active_player_probabilities = _resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        namespace="active_player",
        explicit_key="active_player",
        weights_key="active_player_weights",
        balance_flag_key="balanced_active_player_sampling",
        supported=ACTIVE_PLAYERS,
    )
    return ResolvedBackgammonAxes(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        active_player=str(active_player),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant_probabilities=dict(style_variant_probabilities),
        active_player_probabilities=dict(active_player_probabilities),
    )


def choose_backgammon_dice(rng: Any) -> Tuple[int, int]:
    """Choose two non-double dice values."""

    values = list(range(1, 7))
    rng.shuffle(values)
    return int(values[0]), int(values[1])


def _choose_source_points(
    rng: Any,
    *,
    dice: Tuple[int, int],
    source_count: int,
    active_player: str,
) -> Tuple[int, ...]:
    if str(active_player) == PLAYER_BLACK:
        candidates = [point for point in POINT_IDS if int(point) > max(int(value) for value in dice)]
    else:
        candidates = [point for point in POINT_IDS if int(point) <= 24 - max(int(value) for value in dice)]
    rng.shuffle(candidates)
    selected: list[int] = []
    for candidate in candidates:
        blocked_by_source_conflict = False
        for existing in selected:
            for die in dice:
                if destination_for_player(int(candidate), int(die), active_player=str(active_player)) == int(existing):
                    blocked_by_source_conflict = True
                if destination_for_player(int(existing), int(die), active_player=str(active_player)) == int(candidate):
                    blocked_by_source_conflict = True
        if blocked_by_source_conflict:
            continue
        selected.append(int(candidate))
        if len(selected) >= int(source_count):
            return tuple(sorted(selected))
    raise ValueError("could not choose enough Backgammon source points")


def _candidate_destinations(
    *,
    sources: Sequence[int],
    dice: Tuple[int, int],
    active_player: str,
) -> Tuple[int, ...]:
    destinations = {
        destination_for_player(int(source), int(die), active_player=str(active_player))
        for source in sources
        for die in dice
        if destination_for_player(int(source), int(die), active_player=str(active_player)) in POINT_IDS
    }
    return tuple(sorted(destinations))


def _target_state_for_destination_status(
    rng: Any,
    *,
    destination_status: str,
    is_target: bool,
    active_player: str,
) -> BackgammonPoint:
    status = str(destination_status)
    opponent = opponent_for_player(str(active_player))
    if status == DESTINATION_STATUS_LEGAL:
        if bool(is_target):
            return BackgammonPoint(owner=None, count=0)
        return BackgammonPoint(owner=opponent, count=int(rng.randint(2, 4)))
    if status == DESTINATION_STATUS_HIT:
        if bool(is_target):
            return BackgammonPoint(owner=opponent, count=1)
        if float(rng.random()) < 0.48:
            return BackgammonPoint(owner=opponent, count=int(rng.randint(2, 4)))
        return BackgammonPoint(owner=None, count=0)
    if status == DESTINATION_STATUS_BLOCKED:
        if bool(is_target):
            return BackgammonPoint(owner=opponent, count=int(rng.randint(2, 4)))
        if float(rng.random()) < 0.38:
            return BackgammonPoint(owner=opponent, count=1)
        return BackgammonPoint(owner=None, count=0)
    raise ValueError(f"unsupported Backgammon destination status: {status}")


def sample_destination_count_scene(
    rng: Any,
    *,
    axes: ResolvedBackgammonAxes,
    destination_status: str,
    target_answer: int,
    source_count_min: int = 2,
    source_count_max: int = 4,
    distractor_stack_probability: float = 0.08,
) -> BackgammonSample:
    """Construct one exact-answer Backgammon destination-count position."""

    status = str(destination_status)
    if status not in DESTINATION_STATUSES:
        raise ValueError(f"unsupported destination-count status: {status}")
    active_player = str(axes.active_player)
    opponent = opponent_for_player(active_player)
    target = int(target_answer)
    for _inner_attempt in range(1500):
        dice = choose_backgammon_dice(rng)
        min_sources = max(int(source_count_min), int((target + 1) // 2))
        max_sources = min(int(source_count_max), max(min_sources, int(target) + 2))
        source_count = int(rng.randint(int(min_sources), int(max_sources)))
        try:
            sources = _choose_source_points(
                rng,
                dice=dice,
                source_count=source_count,
                active_player=active_player,
            )
        except ValueError:
            continue
        candidates = _candidate_destinations(sources=sources, dice=dice, active_player=active_player)
        if len(candidates) < int(target):
            continue
        target_destinations = tuple(sorted(rng.sample(list(candidates), int(target))))
        target_set = set(int(point) for point in target_destinations)

        points = empty_points()
        for source in sources:
            points[int(source)] = BackgammonPoint(owner=active_player, count=int(rng.randint(1, 4)))
        for destination in candidates:
            points[int(destination)] = _target_state_for_destination_status(
                rng,
                destination_status=status,
                is_target=int(destination) in target_set,
                active_player=active_player,
            )

        protected_points = set(int(point) for point in sources) | set(int(point) for point in candidates)
        for point in POINT_IDS:
            if int(point) in protected_points:
                continue
            if float(rng.random()) < float(distractor_stack_probability):
                points[int(point)] = BackgammonPoint(owner=opponent, count=int(rng.randint(1, 4)))

        outcome = compute_single_die_destinations(points, dice=dice, active_player=active_player)
        expected_targets = target_destinations_for_status(outcome, destination_status=status)
        if tuple(expected_targets) != tuple(target_destinations):
            continue
        sample = BackgammonSample(
            points=dict(points),
            dice=(int(dice[0]), int(dice[1])),
            active_player=active_player,
            answer=int(target),
            target_destinations=tuple(int(point) for point in target_destinations),
            outcome=outcome,
            style_variant=str(axes.style_variant),
            target_answer=int(target),
            target_points=tuple(int(point) for point in target_destinations),
            destination_status=status,
        )
        validate_backgammon_sample(sample)
        return sample
    raise ValueError(f"could not construct Backgammon destination-count sample for {status} answer {target}")


def _target_state_for_point_state(rng: Any, *, checker_color: str, stack_state: str) -> BackgammonPoint:
    color = str(checker_color)
    state = str(stack_state)
    if color not in {PLAYER_BLACK, PLAYER_WHITE}:
        raise ValueError(f"unsupported Backgammon checker color: {color}")
    if state == STACK_STATE_SINGLE:
        return BackgammonPoint(owner=color, count=1)
    if state == STACK_STATE_TWO_OR_MORE:
        return BackgammonPoint(owner=color, count=int(rng.randint(2, 4)))
    raise ValueError(f"unsupported Backgammon stack state: {state}")


def _non_target_state_for_point_state(rng: Any, *, checker_color: str, stack_state: str) -> BackgammonPoint:
    candidates = [
        BackgammonPoint(owner=None, count=0),
        BackgammonPoint(owner=PLAYER_BLACK, count=1),
        BackgammonPoint(owner=PLAYER_BLACK, count=2),
        BackgammonPoint(owner=PLAYER_BLACK, count=3),
        BackgammonPoint(owner=PLAYER_BLACK, count=4),
        BackgammonPoint(owner=PLAYER_WHITE, count=1),
        BackgammonPoint(owner=PLAYER_WHITE, count=2),
        BackgammonPoint(owner=PLAYER_WHITE, count=3),
        BackgammonPoint(owner=PLAYER_WHITE, count=4),
    ]
    non_matching = [
        candidate
        for candidate in candidates
        if not point_matches_stack_state(
            candidate,
            checker_color=str(checker_color),
            stack_state=str(stack_state),
        )
    ]
    selected = rng.choice(non_matching)
    return BackgammonPoint(owner=selected.owner, count=int(selected.count))


def sample_point_state_count_scene(
    rng: Any,
    *,
    axes: ResolvedBackgammonAxes,
    checker_color: str,
    stack_state: str,
    target_answer: int,
) -> BackgammonSample:
    """Construct one exact-answer Backgammon point-state count position."""

    color = str(checker_color)
    state = str(stack_state)
    if color not in {PLAYER_BLACK, PLAYER_WHITE}:
        raise ValueError(f"unsupported Backgammon checker color: {color}")
    if state not in STACK_STATES:
        raise ValueError(f"unsupported Backgammon stack state: {state}")
    active_player = str(axes.active_player)
    target = int(target_answer)
    for _inner_attempt in range(500):
        dice = choose_backgammon_dice(rng)
        target_points = tuple(sorted(rng.sample(list(POINT_IDS), int(target))))
        target_set = {int(point) for point in target_points}
        points = empty_points()
        for point in target_points:
            points[int(point)] = _target_state_for_point_state(
                rng,
                checker_color=color,
                stack_state=state,
            )

        min_occupied = max(int(target), 8)
        max_occupied = min(20, max(min_occupied, int(target) + 12))
        occupied_count = int(rng.randint(int(min_occupied), int(max_occupied)))
        distractor_count = max(0, int(occupied_count) - int(target))
        available_points = [int(point) for point in POINT_IDS if int(point) not in target_set]
        rng.shuffle(available_points)
        for point in available_points[:distractor_count]:
            points[int(point)] = _non_target_state_for_point_state(
                rng,
                checker_color=color,
                stack_state=state,
            )

        expected_points = target_points_for_stack_state(
            points,
            checker_color=color,
            stack_state=state,
        )
        if tuple(expected_points) != tuple(target_points):
            continue
        outcome = compute_single_die_destinations(points, dice=dice, active_player=active_player)
        sample = BackgammonSample(
            points=dict(points),
            dice=(int(dice[0]), int(dice[1])),
            active_player=active_player,
            answer=int(target),
            target_destinations=(),
            outcome=outcome,
            style_variant=str(axes.style_variant),
            target_answer=int(target),
            target_points=tuple(int(point) for point in target_points),
            checker_color=color,
            stack_state=state,
        )
        validate_backgammon_sample(sample)
        return sample
    raise ValueError(f"could not construct Backgammon point-state sample for {color}/{state} answer {target}")


def _point_for_pip_distance(distance: int, *, active_player: str) -> int:
    """Return the board point whose checker has the requested pip distance."""

    value = int(distance)
    if value < 1 or value > 24:
        raise ValueError(f"pip distance out of range: {distance}")
    if str(active_player) == PLAYER_BLACK:
        return int(value)
    if str(active_player) == PLAYER_WHITE:
        return int(25 - value)
    raise ValueError(f"unsupported Backgammon player: {active_player!r}")


def _pip_term_combinations(
    target_answer: int,
    *,
    max_distance: int = 12,
    min_points: int = 1,
    max_points: int = 5,
    max_stack_count: int = 3,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return unique sparse stack decompositions for one small pip count."""

    target = int(target_answer)
    combinations: list[tuple[tuple[int, int], ...]] = []

    def visit(distance: int, remaining: int, chosen: tuple[tuple[int, int], ...]) -> None:
        if remaining == 0:
            if int(min_points) <= len(chosen) <= int(max_points):
                combinations.append(tuple(chosen))
            return
        if distance > int(max_distance) or len(chosen) >= int(max_points):
            return
        if remaining < 0:
            return
        visit(distance + 1, remaining, chosen)
        for count in range(1, int(max_stack_count) + 1):
            contribution = int(distance) * int(count)
            if contribution > remaining:
                break
            visit(distance + 1, remaining - contribution, (*chosen, (int(distance), int(count))))

    visit(1, target, tuple())
    return tuple(combinations)


def sample_pip_count_scene(
    rng: Any,
    *,
    axes: ResolvedBackgammonAxes,
    target_answer: int,
    opponent_distractor_min: int = 0,
    opponent_distractor_max: int = 2,
) -> BackgammonSample:
    """Construct a sparse exact-answer Backgammon pip-count race position."""

    active_player = str(axes.active_player)
    opponent = opponent_for_player(active_player)
    target = int(target_answer)
    combinations = list(_pip_term_combinations(target))
    if not combinations:
        raise ValueError(f"unsupported Backgammon pip-count target: {target}")
    terms = tuple(rng.choice(combinations))
    points = empty_points()
    active_points: list[int] = []
    for distance, checker_count in terms:
        point = _point_for_pip_distance(int(distance), active_player=active_player)
        points[int(point)] = BackgammonPoint(owner=active_player, count=int(checker_count))
        active_points.append(int(point))

    available = [int(point) for point in POINT_IDS if int(point) not in set(active_points)]
    rng.shuffle(available)
    distractor_count = min(
        len(available),
        int(rng.randint(int(opponent_distractor_min), int(opponent_distractor_max))),
    )
    for point in available[:distractor_count]:
        points[int(point)] = BackgammonPoint(owner=opponent, count=int(rng.randint(1, 4)))

    dice = choose_backgammon_dice(rng)
    outcome = compute_single_die_destinations(points, dice=dice, active_player=active_player)
    target_points = tuple(sorted(active_points))
    contributions = pip_count_contributions_for_player(points, active_player=active_player)
    sample = BackgammonSample(
        points=dict(points),
        dice=(int(dice[0]), int(dice[1])),
        active_player=active_player,
        answer=int(target),
        target_destinations=(),
        outcome=outcome,
        style_variant=str(axes.style_variant),
        target_answer=int(target),
        target_points=target_points,
        pip_count_contributions=dict(contributions),
        use_dice_for_moves=False,
    )
    validate_backgammon_sample(sample)
    return sample


__all__ = [
    "ResolvedBackgammonAxes",
    "resolve_backgammon_axes",
    "sample_destination_count_scene",
    "sample_pip_count_scene",
    "sample_point_state_count_scene",
]
