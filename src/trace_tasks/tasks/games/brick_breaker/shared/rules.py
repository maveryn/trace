"""Identity-free Brick-breaker scene construction helpers."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from .state import (
    BrickBreakerBrick,
    BrickBreakerSample,
    brick_entity_id,
    lane_entity_id,
    lane_label,
    validate_brick_breaker_scene_state,
)


def label_pool(count: int) -> list[str]:
    """Return enough short unique uppercase labels for visible bricks."""

    if int(count) > 26:
        raise ValueError("brick breaker scenes must keep visible brick labels to A..Z")
    labels = [chr(ord("A") + index) for index in range(26)]
    return labels[: int(count)]


def cap_cells_for_single_letter_labels(
    *,
    cells: Sequence[Tuple[int, int]],
    rng: Any,
    protected_cells: Sequence[Tuple[int, int]] = (),
) -> list[Tuple[int, int]]:
    """Remove non-critical bricks until the scene can use only `A` through `Z` labels."""

    keep = set((int(row), int(col)) for row, col in cells)
    protected = set((int(row), int(col)) for row, col in protected_cells)
    removable = [cell for cell in keep if cell not in protected]
    rng.shuffle(removable)
    for cell in removable:
        if len(keep) <= 26:
            break
        keep.remove(cell)
    if len(keep) > 26:
        raise ValueError("not enough removable bricks to keep labels single-letter")
    return sorted(keep)


def make_bricks(*, cells: Sequence[Tuple[int, int]], rng: Any) -> Tuple[BrickBreakerBrick, ...]:
    """Create labeled bricks from row/column cells."""

    labels = label_pool(len(cells))
    rng.shuffle(labels)
    bricks: list[BrickBreakerBrick] = []
    for index, (row, col) in enumerate(cells):
        bricks.append(
            BrickBreakerBrick(
                brick_id=brick_entity_id(int(row), int(col)),
                label=str(labels[index]),
                row=int(row),
                col=int(col),
                color_index=int((int(row) + int(col) + index) % 5),
            )
        )
    return tuple(bricks)


def sample_next_hit_scene(
    *,
    rng: Any,
    scene_variant: str,
    brick_rows: int,
    brick_cols: int,
    lane_count: int,
) -> BrickBreakerSample:
    """Construct a scene where the motion path first reaches one labeled brick."""

    rows = int(brick_rows)
    cols = int(brick_cols)
    target_row = rows - 1
    target_col = int(rng.randrange(cols))
    cells: list[Tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            if int(row) != int(target_row) or int(col) != int(target_col):
                if int(row) == int(target_row) and abs(int(col) - int(target_col)) == 1 and rng.random() < 0.55:
                    continue
                if rng.random() < 0.18:
                    continue
            cells.append((int(row), int(col)))
    if (target_row, target_col) not in set(cells):
        cells.append((target_row, target_col))
    cells = cap_cells_for_single_letter_labels(
        cells=sorted(set(cells)),
        rng=rng,
        protected_cells=((target_row, target_col),),
    )
    bricks = make_bricks(cells=cells, rng=rng)
    target_brick_id = brick_entity_id(target_row, target_col)
    target_brick = next(brick for brick in bricks if str(brick.brick_id) == str(target_brick_id))
    lane_total = int(lane_count)
    target_lane_hint = int(round((((float(target_col) + 0.5) / float(cols)) * float(lane_total)) - 0.5))
    target_lane_hint = max(0, min(lane_total - 1, int(target_lane_hint)))
    start_lane_candidates = [
        lane
        for lane in (target_lane_hint - 1, target_lane_hint + 1)
        if 0 <= int(lane) < lane_total
    ]
    if not start_lane_candidates:
        start_lane_candidates = [lane for lane in range(lane_total) if int(lane) != int(target_lane_hint)] or [target_lane_hint]
    ball_start_lane = int(rng.choice(start_lane_candidates))
    sample = BrickBreakerSample(
        brick_rows=rows,
        brick_cols=cols,
        lane_count=lane_total,
        scene_variant=str(scene_variant),
        bricks=bricks,
        target_brick_id=str(target_brick.brick_id),
        target_brick_label=str(target_brick.label),
        target_row_remaining_brick_ids=tuple(),
        target_row_remaining_count=None,
        target_lane_index=None,
        target_lane_label=None,
        ball_start_lane_index=ball_start_lane,
        annotation_entity_ids=(str(target_brick.brick_id),),
        construction_mode="angled_path_first_lower_row_brick",
    )
    validate_brick_breaker_scene_state(sample)
    return sample


def sample_paddle_catch_scene(
    *,
    rng: Any,
    scene_variant: str,
    brick_rows: int,
    brick_cols: int,
    lane_count: int,
) -> BrickBreakerSample:
    """Construct a scene where the ball path ends in one bottom catch lane."""

    rows = int(brick_rows)
    cols = int(brick_cols)
    cells = [
        (row, col)
        for row in range(rows)
        for col in range(cols)
        if rng.random() >= 0.10
    ]
    if len(cells) < max(8, cols * 2):
        cells = [(row, col) for row in range(rows) for col in range(cols)]
    cells = cap_cells_for_single_letter_labels(cells=sorted(cells), rng=rng)
    bricks = make_bricks(cells=sorted(cells), rng=rng)
    lane_total = int(lane_count)
    target_lane = int(rng.randrange(lane_total))
    candidate_starts = [
        lane
        for lane in range(lane_total)
        if abs(int(lane) - int(target_lane)) <= 2
    ]
    start_lane = int(rng.choice(candidate_starts))
    sample = BrickBreakerSample(
        brick_rows=rows,
        brick_cols=cols,
        lane_count=lane_total,
        scene_variant=str(scene_variant),
        bricks=bricks,
        target_brick_id=None,
        target_brick_label=None,
        target_row_remaining_brick_ids=tuple(),
        target_row_remaining_count=None,
        target_lane_index=int(target_lane),
        target_lane_label=lane_label(target_lane),
        ball_start_lane_index=int(start_lane),
        annotation_entity_ids=(lane_entity_id(target_lane),),
        construction_mode="straight_path_to_catch_lane",
    )
    validate_brick_breaker_scene_state(sample)
    return sample


def sample_hit_row_remaining_scene(
    *,
    rng: Any,
    scene_variant: str,
    brick_rows: int,
    brick_cols: int,
    lane_count: int,
    row_remaining_count: int,
) -> BrickBreakerSample:
    """Construct a scene where a shot removes one brick, then count row survivors."""

    rows = int(brick_rows)
    cols = int(brick_cols)
    target_row = rows - 1
    target_col = int(rng.randrange(cols))
    possible_remaining_cols = [col for col in range(cols) if int(col) != int(target_col)]
    rng.shuffle(possible_remaining_cols)
    remaining_count = int(row_remaining_count)
    if int(remaining_count) > len(possible_remaining_cols):
        raise ValueError("row_remaining_count requires more same-row bricks than this grid can support")
    remaining_cols = sorted(possible_remaining_cols[:remaining_count])
    protected_cells = [(target_row, target_col)] + [(target_row, col) for col in remaining_cols]
    cells: list[Tuple[int, int]] = list(protected_cells)
    for row in range(rows - 1):
        for col in range(cols):
            if rng.random() < 0.24:
                continue
            cells.append((int(row), int(col)))
    cells = cap_cells_for_single_letter_labels(
        cells=sorted(set(cells)),
        rng=rng,
        protected_cells=protected_cells,
    )
    bricks = make_bricks(cells=cells, rng=rng)
    target_brick_id = brick_entity_id(target_row, target_col)
    target_brick = next(brick for brick in bricks if str(brick.brick_id) == str(target_brick_id))
    remaining_ids = tuple(brick_entity_id(target_row, col) for col in remaining_cols)
    lane_total = int(lane_count)
    target_lane_hint = int(round((((float(target_col) + 0.5) / float(cols)) * float(lane_total)) - 0.5))
    target_lane_hint = max(0, min(lane_total - 1, int(target_lane_hint)))
    start_lane_candidates = [
        lane
        for lane in (target_lane_hint - 1, target_lane_hint + 1)
        if 0 <= int(lane) < lane_total
    ]
    if not start_lane_candidates:
        start_lane_candidates = [lane for lane in range(lane_total) if int(lane) != int(target_lane_hint)] or [target_lane_hint]
    ball_start_lane = int(rng.choice(start_lane_candidates))
    sample = BrickBreakerSample(
        brick_rows=rows,
        brick_cols=cols,
        lane_count=lane_total,
        scene_variant=str(scene_variant),
        bricks=bricks,
        target_brick_id=str(target_brick.brick_id),
        target_brick_label=str(target_brick.label),
        target_row_remaining_brick_ids=remaining_ids,
        target_row_remaining_count=int(len(remaining_ids)),
        target_lane_index=None,
        target_lane_label=None,
        ball_start_lane_index=ball_start_lane,
        annotation_entity_ids=remaining_ids,
        construction_mode="brick_hit_then_same_row_remaining_count",
    )
    validate_brick_breaker_scene_state(sample)
    return sample


__all__ = [
    "cap_cells_for_single_letter_labels",
    "label_pool",
    "make_bricks",
    "sample_hit_row_remaining_scene",
    "sample_next_hit_scene",
    "sample_paddle_catch_scene",
]
