"""Lane-runner movement, entity ids, validation, and trace serializers."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from .state import (
    SUPPORTED_LANE_RUNNER_SCENE_VARIANTS,
    SUPPORTED_LANE_RUNNER_STYLE_VARIANTS,
    LaneRunnerCoin,
    LaneRunnerHazard,
    LaneRunnerPathCoinSample,
    LaneRunnerPathOption,
    LaneRunnerSafePathSample,
)


def coin_entity_id(row: int, lane: int) -> str:
    """Return stable entity id for one coin cell."""

    return f"coin_r{int(row)}_l{int(lane)}"


def hazard_entity_id(row: int, lane: int) -> str:
    """Return stable entity id for one hazard cell."""

    return f"hazard_r{int(row)}_l{int(lane)}"


def runner_entity_id() -> str:
    """Return stable entity id for the runner marker."""

    return "runner_start"


def cell_entity_id(row: int, lane: int) -> str:
    """Return stable entity id for one lane grid cell."""

    return f"cell_r{int(row)}_l{int(lane)}"


def path_option_entity_id(label: str) -> str:
    """Return stable entity id for one displayed path option."""

    return f"path_option_{str(label).lower()}"


def _coin_lookup(coins: Sequence[LaneRunnerCoin]) -> Dict[Tuple[int, int], LaneRunnerCoin]:
    lookup: Dict[Tuple[int, int], LaneRunnerCoin] = {}
    for coin in coins:
        key = (int(coin.row), int(coin.lane))
        if key in lookup:
            raise ValueError(f"duplicate lane-runner coin at {key}")
        lookup[key] = coin
    return lookup


def path_coin_collection(
    *,
    coins: Sequence[LaneRunnerCoin],
    shown_path_lanes: Sequence[int],
    row_count: int,
    lane_count: int,
    start_lane: int,
) -> Tuple[int, Tuple[str, ...]]:
    """Return the coins collected by a displayed one-cell-per-row path."""

    rows = int(row_count)
    lanes = int(lane_count)
    start = int(start_lane)
    if rows <= 0:
        raise ValueError("row_count must be positive")
    if lanes <= 1:
        raise ValueError("lane_count must be at least 2")
    if len(shown_path_lanes) != rows:
        raise ValueError("shown_path_lanes must contain one lane per row")
    if not (0 <= start < lanes):
        raise ValueError("start_lane out of range")
    lookup = _coin_lookup(coins)
    previous = start
    collected: list[str] = []
    for row, lane_value in enumerate(shown_path_lanes):
        lane = int(lane_value)
        if not (0 <= lane < lanes):
            raise ValueError("shown path lane out of range")
        if abs(lane - int(previous)) > 1:
            raise ValueError("shown path can move at most one lane per row")
        coin = lookup.get((int(row), int(lane)))
        if coin is not None:
            collected.append(str(coin.coin_id))
        previous = int(lane)
    return len(collected), tuple(collected)


def path_hits_hazard(
    *,
    lanes_by_row: Sequence[int],
    hazards: Sequence[LaneRunnerHazard],
) -> bool:
    """Return true when a route enters at least one hazard cell."""

    hazard_cells = {(int(hazard.row), int(hazard.lane)) for hazard in hazards}
    return any((int(row), int(lane)) in hazard_cells for row, lane in enumerate(lanes_by_row))


def visible_coin_trace(coins: Sequence[LaneRunnerCoin]) -> Tuple[dict[str, int | str], ...]:
    """Serialize visible coins for trace metadata."""

    return tuple(
        {
            "coin_id": str(coin.coin_id),
            "row": int(coin.row),
            "lane": int(coin.lane),
        }
        for coin in coins
    )


def visible_hazard_trace(hazards: Sequence[LaneRunnerHazard]) -> Tuple[dict[str, int | str], ...]:
    """Serialize visible hazards for trace metadata."""

    return tuple(
        {
            "hazard_id": str(hazard.hazard_id),
            "row": int(hazard.row),
            "lane": int(hazard.lane),
        }
        for hazard in hazards
    )


def visible_path_option_trace(options: Sequence[LaneRunnerPathOption]) -> Tuple[dict[str, object], ...]:
    """Serialize visible path options for trace metadata."""

    return tuple(
        {
            "label": str(option.label),
            "lanes_by_row": [int(value) for value in option.lanes_by_row],
        }
        for option in options
    )


def validate_lane_runner_path_coin_sample(sample: LaneRunnerPathCoinSample) -> None:
    """Validate one shown-path coin-count symbolic sample."""

    if str(sample.scene_variant) not in SUPPORTED_LANE_RUNNER_SCENE_VARIANTS:
        raise ValueError(f"unsupported lane-runner scene_variant: {sample.scene_variant}")
    if str(sample.style_variant) not in SUPPORTED_LANE_RUNNER_STYLE_VARIANTS:
        raise ValueError(f"unsupported lane-runner style_variant: {sample.style_variant}")
    if int(sample.lane_count) != 2:
        raise ValueError("lane-runner requires exactly two lanes")
    if len(sample.shown_path_lanes) != int(sample.row_count):
        raise ValueError("shown_path_lanes must contain one lane per row")
    seen_ids: set[str] = set()
    for coin in sample.coins:
        if str(coin.coin_id) in seen_ids:
            raise ValueError(f"duplicate lane-runner coin id: {coin.coin_id}")
        seen_ids.add(str(coin.coin_id))
        if not (0 <= int(coin.row) < int(sample.row_count)):
            raise ValueError("coin row out of range")
        if not (0 <= int(coin.lane) < int(sample.lane_count)):
            raise ValueError("coin lane out of range")
    answer, annotation_ids = path_coin_collection(
        coins=sample.coins,
        shown_path_lanes=sample.shown_path_lanes,
        row_count=int(sample.row_count),
        lane_count=int(sample.lane_count),
        start_lane=int(sample.start_lane),
    )
    if int(answer) != int(sample.answer):
        raise ValueError("lane-runner answer does not match shown-path coin count")
    if tuple(str(value) for value in annotation_ids) != tuple(str(value) for value in sample.annotation_entity_ids):
        raise ValueError("lane-runner annotation ids do not match shown-path coins")
    if len(sample.coins) <= int(sample.answer):
        raise ValueError("shown-path task requires off-path coin distractors")
    coin_cells = {(int(coin.row), int(coin.lane)) for coin in sample.coins}
    has_parallel_coin_row = any(
        (int(row), 1 - int(path_lane)) in coin_cells and (int(row), int(path_lane)) in coin_cells
        for row, path_lane in enumerate(sample.shown_path_lanes)
    )
    if not has_parallel_coin_row:
        raise ValueError("shown-path task requires at least one same-row parallel coin distractor")


def validate_lane_runner_safe_path_sample(sample: LaneRunnerSafePathSample) -> None:
    """Validate one safe-path option-card symbolic sample."""

    if str(sample.scene_variant) not in SUPPORTED_LANE_RUNNER_SCENE_VARIANTS:
        raise ValueError(f"unsupported lane-runner scene_variant: {sample.scene_variant}")
    if str(sample.style_variant) not in SUPPORTED_LANE_RUNNER_STYLE_VARIANTS:
        raise ValueError(f"unsupported lane-runner style_variant: {sample.style_variant}")
    if int(sample.lane_count) != 2:
        raise ValueError("lane-runner requires exactly two lanes")
    if not (0 <= int(sample.start_lane) < int(sample.lane_count)):
        raise ValueError("start_lane out of range")
    hazard_cells: set[Tuple[int, int]] = set()
    for hazard in sample.hazards:
        key = (int(hazard.row), int(hazard.lane))
        if key in hazard_cells:
            raise ValueError(f"duplicate lane-runner hazard at {key}")
        hazard_cells.add(key)
        if str(hazard.hazard_id) != hazard_entity_id(int(hazard.row), int(hazard.lane)):
            raise ValueError("lane-runner hazard id does not match row/lane")
        if not (0 <= int(hazard.row) < int(sample.row_count)):
            raise ValueError("hazard row out of range")
        if not (0 <= int(hazard.lane) < int(sample.lane_count)):
            raise ValueError("hazard lane out of range")
    labels: set[str] = set()
    safe_labels: list[str] = []
    for option in sample.path_options:
        label = str(option.label)
        if label in labels:
            raise ValueError(f"duplicate lane-runner path label: {label}")
        labels.add(label)
        if len(option.lanes_by_row) != int(sample.row_count):
            raise ValueError("path option must contain one lane per row")
        for lane in option.lanes_by_row:
            if not (0 <= int(lane) < int(sample.lane_count)):
                raise ValueError("path option lane out of range")
        if not path_hits_hazard(lanes_by_row=option.lanes_by_row, hazards=sample.hazards):
            safe_labels.append(label)
    if safe_labels != [str(sample.answer_label)]:
        raise ValueError("lane-runner safe-path sample must have exactly one safe answer label")
    expected_cells = tuple(
        cell_entity_id(row, lane)
        for row, lane in enumerate(
            next(option.lanes_by_row for option in sample.path_options if str(option.label) == str(sample.answer_label))
        )
    )
    if tuple(str(value) for value in sample.safe_path_cell_ids) != tuple(str(value) for value in expected_cells):
        raise ValueError("lane-runner safe-path cell ids do not match answer route cells")


__all__ = [
    "cell_entity_id",
    "coin_entity_id",
    "hazard_entity_id",
    "path_coin_collection",
    "path_hits_hazard",
    "path_option_entity_id",
    "runner_entity_id",
    "validate_lane_runner_path_coin_sample",
    "validate_lane_runner_safe_path_sample",
    "visible_coin_trace",
    "visible_hazard_trace",
    "visible_path_option_trace",
]
