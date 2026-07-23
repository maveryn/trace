"""Identity-free Bowling lane rules and scene construction helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from .state import (
    BowlingPathOption,
    BowlingPin,
    BowlingSample,
    option_label,
    path_entity_id,
    pin_entity_id,
    validate_bowling_scene_state,
)


PIN_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(10))
SPARE_GROUPS: Tuple[Tuple[int, ...], ...] = (
    (9,),
    (8,),
    (7,),
    (5,),
    (6,),
    (5, 9),
    (6, 3),
    (4, 0),
    (8, 2),
    (7, 1),
)
PIN_X_NORM: Tuple[float, ...] = (
    0.407,
    0.469,
    0.531,
    0.593,
    0.438,
    0.500,
    0.562,
    0.469,
    0.531,
    0.500,
)
PIN_Y_NORM: Tuple[float, ...] = (
    0.145,
    0.145,
    0.145,
    0.145,
    0.205,
    0.205,
    0.205,
    0.265,
    0.265,
    0.325,
)
BALL_Y_NORM = 0.875
PATH_AIM_Y_NORM = 0.09
LANE_WIDTH_FOR_HIT = 760.0
LANE_HEIGHT_FOR_HIT = 660.0
PIN_HIT_RADIUS_PX = 27.0
PIN_VISUAL_CLEARANCE_RADIUS_PX = 44.0
PATH_HIT_CORE_RADIUS_PX = 12.0
PATH_NON_HIT_CLEARANCE_PX = 58.0
PATH_HIT_MIN_SEPARATION_PX = 44.0


def pin_row_col(rack_index: int) -> Tuple[int, int]:
    """Return row/column within the visible rack for a 0-based rack index."""

    remaining = int(rack_index)
    for row, count in enumerate((4, 3, 2, 1)):
        if remaining < count:
            return int(row), int(remaining)
        remaining -= count
    raise ValueError(f"unsupported rack_index: {rack_index}")


def jitter_pin_positions(*, rng: Any) -> Dict[int, Tuple[float, float]]:
    """Return mildly haphazard visible pin positions in lane-normalized coordinates."""

    positions: Dict[int, Tuple[float, float]] = {}
    for rack_index, (base_x, base_y) in enumerate(zip(PIN_X_NORM, PIN_Y_NORM)):
        x = float(base_x + rng.uniform(-0.014, 0.014))
        y = float(base_y + rng.uniform(-0.012, 0.012))
        positions[int(rack_index)] = (max(0.34, min(0.66, x)), max(0.105, min(0.355, y)))
    return positions


def pins_with_labels(
    *,
    target_label: str,
    target_rack_index: int,
    rng: Any,
) -> Tuple[str, ...]:
    """Assign labels haphazardly while keeping the requested answer label on the hit pin."""

    labels = [str(label) for label in PIN_LABELS if str(label) != str(target_label)]
    rng.shuffle(labels)
    assigned: list[str] = []
    for rack_index in range(10):
        if int(rack_index) == int(target_rack_index):
            assigned.append(str(target_label))
        else:
            assigned.append(str(labels.pop()))
    return tuple(assigned)


def make_pins(
    *,
    standing_ids: set[int] | None = None,
    include_ids: set[int] | None = None,
    positions_norm: Mapping[int, Tuple[float, float]] | None = None,
    labels: Sequence[str] | None = None,
) -> Tuple[BowlingPin, ...]:
    """Create visible labeled bowling pins."""

    include = set(range(10)) if include_ids is None else {int(value) for value in include_ids}
    standing = set(include) if standing_ids is None else ({int(value) for value in standing_ids} & set(include))
    label_values = tuple(str(label) for label in (labels if labels is not None else PIN_LABELS))
    if len(label_values) != 10:
        raise ValueError("Bowling label assignment must contain exactly 10 labels")
    pins: list[BowlingPin] = []
    for rack_index in sorted(include):
        row, col = pin_row_col(rack_index)
        pos = None if positions_norm is None else positions_norm.get(int(rack_index))
        pins.append(
            BowlingPin(
                pin_id=pin_entity_id(rack_index),
                label=label_values[rack_index],
                rack_index=int(rack_index),
                row=int(row),
                col=int(col),
                color_index=int(rack_index % 5),
                standing=int(rack_index) in standing,
                x_norm=None if pos is None else float(pos[0]),
                y_norm=None if pos is None else float(pos[1]),
            )
        )
    return tuple(pins)


def first_intersected_pin_id(
    *,
    pins: Sequence[BowlingPin],
    ball_x_norm: float,
    aim_x_norm: float,
    aim_y_norm: float,
) -> str | None:
    """Return the first standing pin intersected by the ball ray."""

    sx = float(ball_x_norm) * LANE_WIDTH_FOR_HIT
    sy = BALL_Y_NORM * LANE_HEIGHT_FOR_HIT
    ex = float(aim_x_norm) * LANE_WIDTH_FOR_HIT
    ey = float(aim_y_norm) * LANE_HEIGHT_FOR_HIT
    dx = ex - sx
    dy = ey - sy
    length_sq = (dx * dx) + (dy * dy)
    if length_sq <= 1e-6:
        return None
    length = math.sqrt(length_sq)
    best: tuple[float, str] | None = None
    for pin in pins:
        if not bool(pin.standing) or pin.x_norm is None or pin.y_norm is None:
            continue
        px = float(pin.x_norm) * LANE_WIDTH_FOR_HIT
        py = float(pin.y_norm) * LANE_HEIGHT_FOR_HIT
        t = (((px - sx) * dx) + ((py - sy) * dy)) / length_sq
        if t < 0.0 or t > 1.06:
            continue
        closest_x = sx + (t * dx)
        closest_y = sy + (t * dy)
        distance = math.hypot(px - closest_x, py - closest_y)
        if distance > PIN_HIT_RADIUS_PX:
            continue
        entry_offset = math.sqrt(max(0.0, (PIN_HIT_RADIUS_PX * PIN_HIT_RADIUS_PX) - (distance * distance))) / length
        entry_t = float(t - entry_offset)
        if best is None or entry_t < best[0]:
            best = (entry_t, str(pin.pin_id))
    return None if best is None else str(best[1])


def non_target_path_clearance_px(
    *,
    pins: Sequence[BowlingPin],
    target_pin_id: str,
    ball_x_norm: float,
    aim_x_norm: float,
    aim_y_norm: float,
) -> float | None:
    """Return the closest non-target pin distance to the target ray in lane pixels."""

    sx = float(ball_x_norm) * LANE_WIDTH_FOR_HIT
    sy = BALL_Y_NORM * LANE_HEIGHT_FOR_HIT
    ex = float(aim_x_norm) * LANE_WIDTH_FOR_HIT
    ey = float(aim_y_norm) * LANE_HEIGHT_FOR_HIT
    dx = ex - sx
    dy = ey - sy
    length_sq = (dx * dx) + (dy * dy)
    if length_sq <= 1e-6:
        return None

    nearest: float | None = None
    for pin in pins:
        if str(pin.pin_id) == str(target_pin_id) or not bool(pin.standing) or pin.x_norm is None or pin.y_norm is None:
            continue
        px = float(pin.x_norm) * LANE_WIDTH_FOR_HIT
        py = float(pin.y_norm) * LANE_HEIGHT_FOR_HIT
        t = (((px - sx) * dx) + ((py - sy) * dy)) / length_sq
        if t < -0.02 or t > 1.04:
            continue
        closest_x = sx + (t * dx)
        closest_y = sy + (t * dy)
        distance = math.hypot(px - closest_x, py - closest_y)
        nearest = float(distance) if nearest is None else min(float(nearest), float(distance))
    return nearest


def _path_metric_for_point(
    *,
    point_x_norm: float,
    point_y_norm: float,
    ball_x_norm: float,
    aim_x_norm: float,
    aim_y_norm: float,
) -> tuple[float, float] | None:
    """Return ray parameter and lane-pixel distance from one point to a path."""

    sx = float(ball_x_norm) * LANE_WIDTH_FOR_HIT
    sy = BALL_Y_NORM * LANE_HEIGHT_FOR_HIT
    ex = float(aim_x_norm) * LANE_WIDTH_FOR_HIT
    ey = float(aim_y_norm) * LANE_HEIGHT_FOR_HIT
    px = float(point_x_norm) * LANE_WIDTH_FOR_HIT
    py = float(point_y_norm) * LANE_HEIGHT_FOR_HIT
    dx = ex - sx
    dy = ey - sy
    length_sq = (dx * dx) + (dy * dy)
    if length_sq <= 1e-6:
        return None
    t = (((px - sx) * dx) + ((py - sy) * dy)) / length_sq
    closest_x = sx + (t * dx)
    closest_y = sy + (t * dy)
    return float(t), float(math.hypot(px - closest_x, py - closest_y))


def pin_distance_to_path_px(
    pin: BowlingPin,
    *,
    ball_x_norm: float,
    aim_x_norm: float,
    aim_y_norm: float,
) -> float | None:
    """Return one standing pin center's lane-pixel distance to a straight path."""

    if not bool(pin.standing) or pin.x_norm is None or pin.y_norm is None:
        return None
    metric = _path_metric_for_point(
        point_x_norm=float(pin.x_norm),
        point_y_norm=float(pin.y_norm),
        ball_x_norm=float(ball_x_norm),
        aim_x_norm=float(aim_x_norm),
        aim_y_norm=float(aim_y_norm),
    )
    return None if metric is None else float(metric[1])


def path_intersected_pin_ids(
    *,
    pins: Sequence[BowlingPin],
    ball_x_norm: float,
    aim_x_norm: float,
    aim_y_norm: float,
    hit_radius_px: float = PIN_HIT_RADIUS_PX,
    max_t: float = 1.0,
) -> Tuple[str, ...]:
    """Return standing pins whose body is intersected by the ball path segment."""

    hits: list[tuple[float, str]] = []
    for pin in pins:
        if not bool(pin.standing) or pin.x_norm is None or pin.y_norm is None:
            continue
        metric = _path_metric_for_point(
            point_x_norm=float(pin.x_norm),
            point_y_norm=float(pin.y_norm),
            ball_x_norm=float(ball_x_norm),
            aim_x_norm=float(aim_x_norm),
            aim_y_norm=float(aim_y_norm),
        )
        if metric is None:
            continue
        t, distance = metric
        if float(t) < 0.0 or float(t) > float(max_t):
            continue
        if float(distance) <= float(hit_radius_px):
            hits.append((float(t), str(pin.pin_id)))
    return tuple(pin_id for _t, pin_id in sorted(hits))


def _center_distance_px(left: tuple[float, float], right: tuple[float, float]) -> float:
    return math.hypot(
        (float(left[0]) - float(right[0])) * LANE_WIDTH_FOR_HIT,
        (float(left[1]) - float(right[1])) * LANE_HEIGHT_FOR_HIT,
    )


def _line_x_at_y_norm(*, ball_x_norm: float, aim_x_norm: float, y_norm: float) -> float:
    t = (float(y_norm) - BALL_Y_NORM) / (PATH_AIM_Y_NORM - BALL_Y_NORM)
    return float(ball_x_norm) + (float(t) * (float(aim_x_norm) - float(ball_x_norm)))


def make_path_options(*, rng: Any, option_count: int, target_index: int, target_aim_x: float) -> Tuple[BowlingPathOption, ...]:
    """Create path options numbered by their left-to-right visual order."""

    candidate_offsets = [-0.29, -0.24, -0.19, -0.14, -0.095, 0.095, 0.14, 0.19, 0.24, 0.29]
    candidate_positions: list[float] = []
    for offset in candidate_offsets:
        candidate = max(0.28, min(0.72, float(target_aim_x + offset)))
        if abs(candidate - float(target_aim_x)) < 0.055:
            continue
        if all(abs(candidate - other) >= 0.045 for other in candidate_positions):
            candidate_positions.append(float(candidate))
    for candidate in (0.28, 0.34, 0.40, 0.46, 0.54, 0.60, 0.66, 0.72):
        if abs(float(candidate) - float(target_aim_x)) < 0.055:
            continue
        if all(abs(float(candidate) - other) >= 0.045 for other in candidate_positions):
            candidate_positions.append(float(candidate))
    left_candidates = [value for value in candidate_positions if value < float(target_aim_x)]
    right_candidates = [value for value in candidate_positions if value > float(target_aim_x)]
    rng.shuffle(left_candidates)
    rng.shuffle(right_candidates)

    left_needed = int(target_index)
    right_needed = int(option_count) - int(target_index) - 1
    if len(left_candidates) < left_needed or len(right_candidates) < right_needed:
        raise ValueError("not enough distinct Bowling path aim positions")

    aim_values = (
        sorted(float(value) for value in left_candidates[:left_needed])
        + [float(target_aim_x)]
        + sorted(float(value) for value in right_candidates[:right_needed])
    )
    return tuple(
        BowlingPathOption(
            path_id=path_entity_id(index),
            label=option_label(index),
            aim_x_norm=float(aim_values[index]),
            color_index=int(index),
        )
        for index in range(int(option_count))
    )


def path_target_aim_x_range(*, option_count: int, target_index: int) -> Tuple[float, float]:
    """Return an aim-x range with enough room for left/right ranked options."""

    left_needed = int(target_index)
    right_needed = int(option_count) - int(target_index) - 1
    min_x = float(0.34 + (0.058 * left_needed))
    max_x = float(0.66 - (0.058 * right_needed))
    if min_x > max_x:
        raise ValueError("Bowling path rank cannot fit within lane option bounds")
    return max(0.36, min_x), min(0.64, max_x)


def sample_first_pin_hit_scene(
    *,
    rng: Any,
    scene_variant: str,
    style_variant: str,
    target_pin_label_index: int,
    visible_pin_count: int,
) -> BowlingSample:
    """Construct a Bowling scene where the shown path first reaches one pin."""

    target_label = str(PIN_LABELS[int(target_pin_label_index)])
    for _layout_attempt in range(96):
        positions = jitter_pin_positions(rng=rng)
        target_rack_indices = list(range(10))
        rng.shuffle(target_rack_indices)
        ball_candidates = [0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74]
        ball_candidates.extend(float(rng.uniform(0.24, 0.76)) for _ in range(18))
        rng.shuffle(ball_candidates)
        for target_rack_index in target_rack_indices:
            label_order = pins_with_labels(
                target_label=target_label,
                target_rack_index=int(target_rack_index),
                rng=rng,
            )
            visible_ids = {int(target_rack_index)}
            other_rack_indices = [int(index) for index in range(10) if int(index) != int(target_rack_index)]
            rng.shuffle(other_rack_indices)
            visible_ids.update(other_rack_indices[: max(0, int(visible_pin_count) - 1)])
            pins = make_pins(
                positions_norm=positions,
                labels=label_order,
                include_ids=visible_ids,
                standing_ids=visible_ids,
            )
            target_pin = next(pin for pin in pins if int(pin.rack_index) == int(target_rack_index))
            if target_pin.x_norm is None or target_pin.y_norm is None:
                continue
            for ball_x in ball_candidates:
                first_pin_id = first_intersected_pin_id(
                    pins=pins,
                    ball_x_norm=float(ball_x),
                    aim_x_norm=float(target_pin.x_norm),
                    aim_y_norm=float(target_pin.y_norm),
                )
                if str(first_pin_id) != str(target_pin.pin_id):
                    continue
                clearance_px = non_target_path_clearance_px(
                    pins=pins,
                    target_pin_id=str(target_pin.pin_id),
                    ball_x_norm=float(ball_x),
                    aim_x_norm=float(target_pin.x_norm),
                    aim_y_norm=float(target_pin.y_norm),
                )
                if clearance_px is not None and float(clearance_px) < PIN_VISUAL_CLEARANCE_RADIUS_PX:
                    continue
                sample = BowlingSample(
                    scene_variant=str(scene_variant),
                    style_variant=str(style_variant),
                    pins=pins,
                    path_options=tuple(),
                    ball_x_norm=float(ball_x),
                    target_pin_id=str(target_pin.pin_id),
                    target_pin_label=str(target_pin.label),
                    target_path_id=None,
                    target_path_label=None,
                    remaining_pin_ids=tuple(pin.pin_id for pin in pins if bool(pin.standing)),
                    annotation_entity_ids=(str(target_pin.pin_id),),
                    construction_mode="verified_first_collision_path",
                    path_visible_fraction=None,
                    path_clearance_px=None if clearance_px is None else float(clearance_px),
                )
                validate_bowling_scene_state(sample)
                return sample
    raise ValueError("failed to construct a first-pin Bowling path with unique first contact")


def sample_spare_path_scene(
    *,
    rng: Any,
    scene_variant: str,
    style_variant: str,
    path_option_count: int,
    target_path_index: int,
) -> BowlingSample:
    """Construct a Bowling scene where one labeled path covers all standing spare pins."""

    option_count = int(path_option_count)
    target_index = int(target_path_index)
    if target_index < 0 or target_index >= option_count:
        raise ValueError("target path index must be visible")
    group = tuple(int(value) for value in SPARE_GROUPS[int(rng.randrange(len(SPARE_GROUPS)))])
    target_aim_min, target_aim_max = path_target_aim_x_range(
        option_count=option_count,
        target_index=target_index,
    )
    target_aim_x = float(rng.uniform(float(target_aim_min), float(target_aim_max)))
    positions = jitter_pin_positions(rng=rng)
    line_t_values = [0.79, 0.86]
    perpendicular_jitter = float(rng.uniform(-0.004, 0.004))
    for group_index, pin_index in enumerate(group):
        t = float(line_t_values[min(group_index, len(line_t_values) - 1)])
        x = float(0.50 + (t * (target_aim_x - 0.50)) + perpendicular_jitter)
        y = float(BALL_Y_NORM + (t * (PATH_AIM_Y_NORM - BALL_Y_NORM)))
        positions[int(pin_index)] = (max(0.34, min(0.66, x)), max(0.115, min(0.36, y)))
    path_options = make_path_options(
        rng=rng,
        option_count=option_count,
        target_index=target_index,
        target_aim_x=float(target_aim_x),
    )
    target_path = path_options[target_index]
    sample = BowlingSample(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        pins=make_pins(standing_ids=set(group), include_ids=set(group), positions_norm=positions),
        path_options=path_options,
        ball_x_norm=0.50,
        target_pin_id=None,
        target_pin_label=None,
        target_path_id=str(target_path.path_id),
        target_path_label=str(target_path.label),
        remaining_pin_ids=tuple(pin_entity_id(index) for index in group),
        annotation_entity_ids=(str(target_path.path_id),),
        construction_mode="unique_path_through_remaining_pins",
        path_visible_fraction=float(rng.uniform(0.42, 0.56)),
        path_clearance_px=None,
    )
    validate_bowling_scene_state(sample)
    return sample


def sample_path_hit_count_scene(
    *,
    rng: Any,
    scene_variant: str,
    style_variant: str,
    target_hit_count: int,
    distractor_count_min: int = 3,
    distractor_count_max: int = 5,
) -> BowlingSample:
    """Construct a Bowling scene where a straight path clearly intersects N pins."""

    target = int(target_hit_count)
    if target < 1 or target > 5:
        raise ValueError("Bowling path-hit count support is 1..5")

    hit_y_values = (0.135, 0.200, 0.265, 0.330, 0.395)[:target]
    candidate_non_hit_positions = [
        (x, y)
        for y in (0.155, 0.215, 0.275, 0.335, 0.395, 0.455)
        for x in (0.345, 0.395, 0.445, 0.555, 0.605, 0.655)
    ]

    for _attempt in range(384):
        ball_x = float(rng.uniform(0.39, 0.61))
        aim_x = float(rng.uniform(0.36, 0.64))
        if abs(float(aim_x) - float(ball_x)) < 0.035:
            aim_x = max(0.36, min(0.64, float(aim_x + (0.07 if aim_x <= ball_x else -0.07))))

        hit_positions = tuple(
            (_line_x_at_y_norm(ball_x_norm=ball_x, aim_x_norm=aim_x, y_norm=y_norm), float(y_norm))
            for y_norm in hit_y_values
        )
        if any(float(x) < 0.34 or float(x) > 0.66 for x, _y in hit_positions):
            continue
        if any(
            _center_distance_px(left, right) < PATH_HIT_MIN_SEPARATION_PX
            for index, left in enumerate(hit_positions)
            for right in hit_positions[index + 1 :]
        ):
            continue

        rack_indices = list(range(10))
        rng.shuffle(rack_indices)
        hit_indices = tuple(int(value) for value in rack_indices[:target])
        positions: dict[int, tuple[float, float]] = {
            int(rack_index): (float(position[0]), float(position[1]))
            for rack_index, position in zip(hit_indices, hit_positions)
        }

        top_hit_index = int(hit_indices[0])
        top_hit_x, top_hit_y = positions[top_hit_index]
        non_hit_indices = [int(value) for value in rack_indices[target:]]
        rng.shuffle(non_hit_indices)
        candidate_positions = list(candidate_non_hit_positions)
        rng.shuffle(candidate_positions)
        selected_non_hits: list[int] = []
        occupied_positions = [tuple(position) for position in hit_positions]
        max_distractors = min(int(distractor_count_max), len(non_hit_indices))
        desired_distractors = int(rng.randint(int(distractor_count_min), max_distractors))

        for candidate_x, candidate_y in candidate_positions:
            if len(selected_non_hits) >= desired_distractors:
                break
            jittered = (
                max(0.335, min(0.665, float(candidate_x + rng.uniform(-0.007, 0.007)))),
                max(0.145, min(0.465, float(candidate_y + rng.uniform(-0.007, 0.007)))),
            )
            metric = _path_metric_for_point(
                point_x_norm=float(jittered[0]),
                point_y_norm=float(jittered[1]),
                ball_x_norm=float(ball_x),
                aim_x_norm=float(top_hit_x),
                aim_y_norm=float(top_hit_y),
            )
            if metric is None:
                continue
            t, distance = metric
            if 0.0 <= float(t) <= 1.06 and float(distance) < PATH_NON_HIT_CLEARANCE_PX:
                continue
            if any(_center_distance_px(jittered, existing) < 54.0 for existing in occupied_positions):
                continue
            rack_index = int(non_hit_indices[len(selected_non_hits)])
            positions[rack_index] = (float(jittered[0]), float(jittered[1]))
            occupied_positions.append(jittered)
            selected_non_hits.append(rack_index)

        if len(selected_non_hits) < desired_distractors:
            continue

        labels = list(PIN_LABELS)
        rng.shuffle(labels)
        visible_indices = set(hit_indices) | set(selected_non_hits)
        pins = make_pins(
            standing_ids=visible_indices,
            include_ids=visible_indices,
            positions_norm=positions,
            labels=labels,
        )
        hit_ids = path_intersected_pin_ids(
            pins=pins,
            ball_x_norm=float(ball_x),
            aim_x_norm=float(top_hit_x),
            aim_y_norm=float(top_hit_y),
            hit_radius_px=PIN_HIT_RADIUS_PX,
        )
        core_hit_ids = path_intersected_pin_ids(
            pins=pins,
            ball_x_norm=float(ball_x),
            aim_x_norm=float(top_hit_x),
            aim_y_norm=float(top_hit_y),
            hit_radius_px=PATH_HIT_CORE_RADIUS_PX,
        )
        expected_hit_ids = {pin_entity_id(index) for index in hit_indices}
        if set(hit_ids) != expected_hit_ids or set(core_hit_ids) != expected_hit_ids or len(hit_ids) != target:
            continue

        non_hit_clearances = [
            pin_distance_to_path_px(
                pin,
                ball_x_norm=float(ball_x),
                aim_x_norm=float(top_hit_x),
                aim_y_norm=float(top_hit_y),
            )
            for pin in pins
            if str(pin.pin_id) not in expected_hit_ids
        ]
        min_non_hit_clearance = min(
            float(value)
            for value in non_hit_clearances
            if value is not None
        )
        if float(min_non_hit_clearance) < PATH_NON_HIT_CLEARANCE_PX:
            continue

        anchor_pin = next(pin for pin in pins if str(pin.pin_id) == pin_entity_id(top_hit_index))
        sample = BowlingSample(
            scene_variant=str(scene_variant),
            style_variant=str(style_variant),
            pins=pins,
            path_options=tuple(),
            ball_x_norm=float(ball_x),
            target_pin_id=str(anchor_pin.pin_id),
            target_pin_label=str(anchor_pin.label),
            target_path_id=None,
            target_path_label=None,
            remaining_pin_ids=tuple(pin.pin_id for pin in pins if bool(pin.standing)),
            annotation_entity_ids=tuple(str(pin_id) for pin_id in hit_ids),
            construction_mode="exact_path_hit_count_with_clearance",
            path_visible_fraction=None,
            path_clearance_px=float(min_non_hit_clearance),
        )
        validate_bowling_scene_state(sample)
        return sample
    raise ValueError(f"failed to construct Bowling path-hit count sample for answer {target}")


__all__ = [
    "PIN_LABELS",
    "PATH_HIT_CORE_RADIUS_PX",
    "PATH_HIT_MIN_SEPARATION_PX",
    "PATH_NON_HIT_CLEARANCE_PX",
    "first_intersected_pin_id",
    "make_path_options",
    "make_pins",
    "path_intersected_pin_ids",
    "path_target_aim_x_range",
    "pin_distance_to_path_px",
    "sample_first_pin_hit_scene",
    "sample_path_hit_count_scene",
    "sample_spare_path_scene",
]
