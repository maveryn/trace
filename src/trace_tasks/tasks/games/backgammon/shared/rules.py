"""Backgammon rule and predicate helpers for visible-board tasks."""

from __future__ import annotations

from typing import Mapping, Tuple

from .state import (
    DESTINATION_STATUSES,
    DESTINATION_STATUS_BLOCKED,
    DESTINATION_STATUS_HIT,
    DESTINATION_STATUS_LEGAL,
    POINT_IDS,
    PLAYER_BLACK,
    PLAYER_WHITE,
    STACK_STATES,
    STACK_STATE_SINGLE,
    STACK_STATE_TWO_OR_MORE,
    BackgammonOutcome,
    BackgammonPoint,
    BackgammonSample,
    stack_at,
)


def opponent_for_player(active_player: str) -> str:
    """Return the opposing checker color for one active player."""

    player = str(active_player)
    if player == PLAYER_BLACK:
        return PLAYER_WHITE
    if player == PLAYER_WHITE:
        return PLAYER_BLACK
    raise ValueError(f"unsupported Backgammon player: {active_player!r}")


def destination_for_player(source: int, die: int, *, active_player: str) -> int:
    """Return one single-die destination point for the active player."""

    if str(active_player) == PLAYER_BLACK:
        return int(source) - int(die)
    if str(active_player) == PLAYER_WHITE:
        return int(source) + int(die)
    raise ValueError(f"unsupported Backgammon player: {active_player!r}")


def is_blocked_by_opponent(points: Mapping[int, BackgammonPoint], point_id: int, *, active_player: str) -> bool:
    """Return true when a destination has two or more opposing checkers."""

    stack = stack_at(points, int(point_id))
    return str(stack.owner) == opponent_for_player(str(active_player)) and int(stack.count) >= 2


def is_opponent_blot(points: Mapping[int, BackgammonPoint], point_id: int, *, active_player: str) -> bool:
    """Return true when a destination has exactly one opposing checker."""

    stack = stack_at(points, int(point_id))
    return str(stack.owner) == opponent_for_player(str(active_player)) and int(stack.count) == 1


def compute_single_die_destinations(
    points: Mapping[int, BackgammonPoint],
    *,
    dice: Tuple[int, int],
    active_player: str,
) -> BackgammonOutcome:
    """Compute destination-point sets for the active player and shown dice."""

    player = str(active_player)
    candidate_destinations: set[int] = set()
    for source in POINT_IDS:
        stack = stack_at(points, int(source))
        if str(stack.owner) != player or int(stack.count) <= 0:
            continue
        for die in dice:
            dest = destination_for_player(int(source), int(die), active_player=player)
            if int(dest) in POINT_IDS:
                candidate_destinations.add(int(dest))

    blocked = tuple(sorted(point for point in candidate_destinations if is_blocked_by_opponent(points, point, active_player=player)))
    legal = tuple(sorted(point for point in candidate_destinations if point not in set(blocked)))
    hit = tuple(sorted(point for point in legal if is_opponent_blot(points, point, active_player=player)))
    return BackgammonOutcome(
        legal_destinations=legal,
        hit_destinations=hit,
        blocked_destinations=blocked,
    )


def target_destinations_for_status(outcome: BackgammonOutcome, *, destination_status: str) -> Tuple[int, ...]:
    """Return the destination set for one semantic destination status."""

    status = str(destination_status)
    if status == DESTINATION_STATUS_LEGAL:
        return tuple(outcome.legal_destinations)
    if status == DESTINATION_STATUS_HIT:
        return tuple(outcome.hit_destinations)
    if status == DESTINATION_STATUS_BLOCKED:
        return tuple(outcome.blocked_destinations)
    raise ValueError(f"unsupported Backgammon destination status: {status}")


def point_matches_stack_state(point: BackgammonPoint, *, checker_color: str, stack_state: str) -> bool:
    """Return true when one point stack satisfies a color and stack-state predicate."""

    color = str(checker_color)
    state = str(stack_state)
    if color not in {PLAYER_BLACK, PLAYER_WHITE}:
        raise ValueError(f"unsupported Backgammon checker color: {color}")
    if state == STACK_STATE_SINGLE:
        return str(point.owner) == color and int(point.count) == 1
    if state == STACK_STATE_TWO_OR_MORE:
        return str(point.owner) == color and int(point.count) >= 2
    raise ValueError(f"unsupported Backgammon stack state: {state}")


def target_points_for_stack_state(
    points: Mapping[int, BackgammonPoint],
    *,
    checker_color: str,
    stack_state: str,
) -> Tuple[int, ...]:
    """Return numbered board points satisfying one color and stack-state predicate."""

    return tuple(
        int(point)
        for point in POINT_IDS
        if point_matches_stack_state(
            stack_at(points, int(point)),
            checker_color=str(checker_color),
            stack_state=str(stack_state),
        )
    )


def pip_distance_for_player(point_id: int, *, active_player: str) -> int:
    """Return standard pips-to-bear-off for one checker on a numbered point."""

    point = int(point_id)
    if point not in POINT_IDS:
        raise ValueError(f"Backgammon point out of range: {point_id}")
    player = str(active_player)
    if player == PLAYER_BLACK:
        return int(point)
    if player == PLAYER_WHITE:
        return int(25 - point)
    raise ValueError(f"unsupported Backgammon player: {active_player!r}")


def pip_count_points_for_player(
    points: Mapping[int, BackgammonPoint],
    *,
    active_player: str,
) -> Tuple[int, ...]:
    """Return active-player occupied points that contribute to pip count."""

    player = str(active_player)
    if player not in {PLAYER_BLACK, PLAYER_WHITE}:
        raise ValueError(f"unsupported Backgammon player: {active_player!r}")
    return tuple(
        int(point)
        for point in POINT_IDS
        if str(stack_at(points, int(point)).owner) == player
        and int(stack_at(points, int(point)).count) > 0
    )


def pip_count_contributions_for_player(
    points: Mapping[int, BackgammonPoint],
    *,
    active_player: str,
) -> dict[int, int]:
    """Return per-point pip-count contributions for active-player stacks."""

    player = str(active_player)
    return {
        int(point): int(stack_at(points, int(point)).count)
        * pip_distance_for_player(int(point), active_player=player)
        for point in pip_count_points_for_player(points, active_player=player)
    }


def pip_count_for_player(
    points: Mapping[int, BackgammonPoint],
    *,
    active_player: str,
) -> int:
    """Return standard Backgammon pip count for the active player."""

    return int(sum(pip_count_contributions_for_player(points, active_player=str(active_player)).values()))


def validate_backgammon_sample(sample: BackgammonSample) -> None:
    """Validate the generated Backgammon sample against the public contract."""

    if tuple(int(value) for value in sample.dice) != tuple(sample.dice):
        raise ValueError("dice must be integers")
    if len(set(int(value) for value in sample.dice)) != len(sample.dice):
        raise ValueError("Backgammon calibration scenes avoid doubles")
    if any(int(value) < 1 or int(value) > 6 for value in sample.dice):
        raise ValueError("dice values must be in 1..6")
    for point in POINT_IDS:
        stack = stack_at(sample.points, point)
        if stack.owner is None:
            if int(stack.count) != 0:
                raise ValueError("empty points must have count 0")
            continue
        if str(stack.owner) not in {PLAYER_BLACK, PLAYER_WHITE}:
            raise ValueError(f"unsupported checker owner: {stack.owner}")
        if int(stack.count) < 1:
            raise ValueError("occupied points must have positive checker count")
    if str(sample.active_player) not in {PLAYER_BLACK, PLAYER_WHITE}:
        raise ValueError(f"unsupported active player: {sample.active_player}")
    if str(sample.destination_status):
        if str(sample.destination_status) not in DESTINATION_STATUSES:
            raise ValueError(f"unsupported Backgammon destination status: {sample.destination_status}")
        outcome = compute_single_die_destinations(
            sample.points,
            dice=sample.dice,
            active_player=str(sample.active_player),
        )
        expected = target_destinations_for_status(outcome, destination_status=str(sample.destination_status))
        if tuple(expected) != tuple(sample.target_destinations):
            raise ValueError("target destinations do not match recomputed outcome")
        if tuple(sample.target_points or sample.target_destinations) != tuple(expected):
            raise ValueError("target points do not match target destinations")
        if int(sample.answer) != len(expected):
            raise ValueError("answer does not match target destination count")
        return
    if str(sample.checker_color) or str(sample.stack_state):
        if str(sample.checker_color) not in {PLAYER_BLACK, PLAYER_WHITE}:
            raise ValueError(f"unsupported Backgammon checker color: {sample.checker_color}")
        if str(sample.stack_state) not in STACK_STATES:
            raise ValueError(f"unsupported Backgammon stack state: {sample.stack_state}")
        expected_points = target_points_for_stack_state(
            sample.points,
            checker_color=str(sample.checker_color),
            stack_state=str(sample.stack_state),
        )
        if tuple(expected_points) != tuple(sample.target_points):
            raise ValueError("target points do not match recomputed point-state predicate")
        if tuple(sample.target_destinations):
            raise ValueError("point-state predicates must not report target destinations")
        if int(sample.answer) != len(expected_points):
            raise ValueError("answer does not match target point count")
        return
    if tuple(sample.target_points) or sample.pip_count_contributions:
        if tuple(sample.target_destinations):
            raise ValueError("pip-count samples must not report target destinations")
        expected_points = pip_count_points_for_player(
            sample.points,
            active_player=str(sample.active_player),
        )
        if tuple(expected_points) != tuple(sample.target_points):
            raise ValueError("target points do not match active-player pip-count points")
        expected_contributions = pip_count_contributions_for_player(
            sample.points,
            active_player=str(sample.active_player),
        )
        if dict(sample.pip_count_contributions) and {
            int(key): int(value) for key, value in sample.pip_count_contributions.items()
        } != expected_contributions:
            raise ValueError("pip-count contributions do not match board state")
        if int(sample.answer) != int(sum(expected_contributions.values())):
            raise ValueError("answer does not match active-player pip count")
        return
    raise ValueError("Backgammon sample must define a destination status or point-state predicate")


__all__ = [
    "compute_single_die_destinations",
    "destination_for_player",
    "is_blocked_by_opponent",
    "is_opponent_blot",
    "opponent_for_player",
    "pip_count_contributions_for_player",
    "pip_count_for_player",
    "pip_count_points_for_player",
    "pip_distance_for_player",
    "point_matches_stack_state",
    "target_destinations_for_status",
    "target_points_for_stack_state",
    "validate_backgammon_sample",
]
