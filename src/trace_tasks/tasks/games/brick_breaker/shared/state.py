"""Identity-free Brick-breaker scene state helpers for games-domain tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SUPPORTED_BRICK_BREAKER_SCENE_VARIANTS: Tuple[str, ...] = ("brick_wall",)
SUPPORTED_BRICK_BREAKER_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "neon",
    "paper",
    "blueprint",
    "arcade",
)


@dataclass(frozen=True)
class BrickBreakerBrick:
    """One visible labeled brick."""

    brick_id: str
    label: str
    row: int
    col: int
    color_index: int


@dataclass(frozen=True)
class BrickBreakerSample:
    """Generated Brick-breaker scene state."""

    brick_rows: int
    brick_cols: int
    lane_count: int
    scene_variant: str
    bricks: Tuple[BrickBreakerBrick, ...]
    target_brick_id: str | None
    target_brick_label: str | None
    target_row_remaining_brick_ids: Tuple[str, ...]
    target_row_remaining_count: int | None
    target_lane_index: int | None
    target_lane_label: str | None
    ball_start_lane_index: int | None
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str


def brick_entity_id(row: int, col: int) -> str:
    """Return the stable render/entity id for one brick cell."""

    return f"brick_{int(row)}_{int(col)}"


def lane_entity_id(lane: int) -> str:
    """Return the stable render/entity id for one bottom catch lane pad."""

    return f"lane_{int(lane)}"


def lane_label(lane: int) -> str:
    """Return the prompt-facing label for one catch lane."""

    return chr(ord("A") + int(lane))


def validate_brick_breaker_scene_state(sample: BrickBreakerSample) -> None:
    """Validate Brick-breaker entity ids, labels, targets, and annotation references together."""

    if int(sample.brick_rows) <= 0 or int(sample.brick_cols) <= 0:
        raise ValueError("brick breaker brick grid dimensions must be positive")
    if int(sample.lane_count) <= 0:
        raise ValueError("brick breaker lane_count must be positive")
    brick_ids = [str(brick.brick_id) for brick in sample.bricks]
    brick_labels = [str(brick.label) for brick in sample.bricks]
    if len(brick_ids) != len(set(brick_ids)):
        raise ValueError("brick breaker brick ids must be unique")
    if len(brick_labels) != len(set(brick_labels)):
        raise ValueError("brick breaker brick labels must be unique")
    for brick in sample.bricks:
        if not (0 <= int(brick.row) < int(sample.brick_rows)):
            raise ValueError("brick breaker brick row out of range")
        if not (0 <= int(brick.col) < int(sample.brick_cols)):
            raise ValueError("brick breaker brick col out of range")

    known_entities = set(brick_ids) | {lane_entity_id(lane) for lane in range(int(sample.lane_count))}
    if not set(sample.annotation_entity_ids) <= known_entities:
        raise ValueError("brick breaker annotation references unknown entities")

    if sample.target_brick_id is not None:
        if str(sample.target_brick_id) not in set(brick_ids):
            raise ValueError("target brick id must reference a visible brick")
        if sample.target_brick_label is not None:
            if str(sample.target_brick_label) not in set(brick_labels):
                raise ValueError("target brick label must reference a visible brick")
    if sample.target_row_remaining_brick_ids:
        if sample.target_brick_id is None:
            raise ValueError("row remaining targets require a hit brick")
        remaining_ids = tuple(str(value) for value in sample.target_row_remaining_brick_ids)
        if sample.target_row_remaining_count is None:
            raise ValueError("row remaining targets require a row remaining count")
        if len(remaining_ids) != int(sample.target_row_remaining_count):
            raise ValueError("row remaining id count must match target_row_remaining_count")
        if str(sample.target_brick_id) in set(remaining_ids):
            raise ValueError("row remaining annotation must exclude the brick that is hit")
        known_by_id = {str(brick.brick_id): brick for brick in sample.bricks}
        target_row = int(known_by_id[str(sample.target_brick_id)].row)
        for brick_id in remaining_ids:
            if brick_id not in known_by_id:
                raise ValueError("row remaining annotation references unknown brick")
            if int(known_by_id[brick_id].row) != target_row:
                raise ValueError("row remaining annotation must stay in the target row")
    if sample.target_lane_index is not None:
        if not (0 <= int(sample.target_lane_index) < int(sample.lane_count)):
            raise ValueError("target lane index out of range")
        valid_lane_labels = {lane_label(lane) for lane in range(int(sample.lane_count))}
        if sample.target_lane_label is None or str(sample.target_lane_label) not in valid_lane_labels:
            raise ValueError("target lane label must reference a visible lane")


__all__ = [
    "SUPPORTED_BRICK_BREAKER_SCENE_VARIANTS",
    "SUPPORTED_BRICK_BREAKER_STYLE_VARIANTS",
    "BrickBreakerBrick",
    "BrickBreakerSample",
    "brick_entity_id",
    "lane_entity_id",
    "lane_label",
    "validate_brick_breaker_scene_state",
]
