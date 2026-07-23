"""Construction primitives for coordinate-panel scenes."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

from trace_tasks.tasks.geometry.shared.quadrilateral_prototypes import classify_quadrilateral_kind

from .state import GraphPoint, GraphSegment

ALL_EXACT_SHAPE_KINDS: Tuple[str, ...] = (
    "square",
    "rectangle_non_square",
    "rhombus_non_square",
    "parallelogram_only",
)

SEGMENT_RELATION_KINDS: Tuple[str, ...] = ("parallel", "perpendicular", "equal_length")
POINT_SET_TRANSFORM_KINDS: Tuple[str, ...] = ("translation", "reflection_x", "reflection_y", "rotation_180")

SQUARE_VECTORS: Tuple[GraphPoint, ...] = (
    (2, 0),
    (3, 0),
    (0, 2),
    (0, 3),
    (2, 1),
    (1, 2),
    (2, -1),
    (1, -2),
    (3, 1),
    (1, 3),
    (3, -1),
    (1, -3),
)
RECTANGLE_VECTOR_PAIRS: Tuple[Tuple[GraphPoint, GraphPoint], ...] = (
    ((4, 0), (0, 2)),
    ((3, 0), (0, 2)),
    ((2, 0), (0, 4)),
    ((2, 1), (-2, 4)),
    ((1, 2), (-4, 2)),
    ((2, -1), (2, 4)),
    ((1, -2), (4, 2)),
)
RHOMBUS_VECTOR_PAIRS: Tuple[Tuple[GraphPoint, GraphPoint], ...] = (
    ((2, 1), (1, 2)),
    ((3, 1), (1, 3)),
    ((3, 2), (2, 3)),
    ((2, -1), (1, -2)),
    ((3, -1), (1, -3)),
    ((3, -2), (2, -3)),
)
PARALLELOGRAM_VECTOR_PAIRS: Tuple[Tuple[GraphPoint, GraphPoint], ...] = (
    ((4, 0), (1, 2)),
    ((3, 0), (1, 2)),
    ((2, 1), (3, -1)),
    ((3, 1), (1, 2)),
    ((4, 1), (-1, 2)),
    ((2, -1), (3, 1)),
)


def transform_vector(vector: GraphPoint, transform_index: int) -> GraphPoint:
    x_value, y_value = int(vector[0]), int(vector[1])
    variants = (
        (x_value, y_value),
        (-x_value, y_value),
        (x_value, -y_value),
        (-x_value, -y_value),
        (y_value, x_value),
        (-y_value, x_value),
        (y_value, -x_value),
        (-y_value, -x_value),
    )
    selected_index = int(transform_index)
    if selected_index < 0 or selected_index >= len(variants):
        raise ValueError("transform_index is outside quadrilateral transform support")
    return tuple(int(value) for value in variants[selected_index])  # type: ignore[return-value]


def vector_pair_for_kind(kind: str, rng) -> Tuple[GraphPoint, GraphPoint]:
    if str(kind) == "square":
        u = tuple(int(value) for value in rng.choice(SQUARE_VECTORS))
        v = (-int(u[1]), int(u[0]))
    elif str(kind) == "rectangle_non_square":
        u, v = rng.choice(RECTANGLE_VECTOR_PAIRS)
    elif str(kind) == "rhombus_non_square":
        u, v = rng.choice(RHOMBUS_VECTOR_PAIRS)
    elif str(kind) == "parallelogram_only":
        u, v = rng.choice(PARALLELOGRAM_VECTOR_PAIRS)
    else:
        raise ValueError(f"unsupported quadrilateral kind: {kind}")

    transform = int(rng.randrange(8))
    u_t = transform_vector(tuple(u), int(transform))
    v_t = transform_vector(tuple(v), int(transform))
    if bool(rng.randrange(2)):
        u_t, v_t = v_t, u_t
    return u_t, v_t


def translate_points_within(points: Sequence[GraphPoint], *, rng, max_abs: int) -> Tuple[GraphPoint, ...]:
    min_x = min(int(point[0]) for point in points)
    max_x = max(int(point[0]) for point in points)
    min_y = min(int(point[1]) for point in points)
    max_y = max(int(point[1]) for point in points)
    shift_x_min = int(-int(max_abs) - int(min_x))
    shift_x_max = int(int(max_abs) - int(max_x))
    shift_y_min = int(-int(max_abs) - int(min_y))
    shift_y_max = int(int(max_abs) - int(max_y))
    if shift_x_min > shift_x_max or shift_y_min > shift_y_max:
        raise ValueError("shape does not fit inside graph bounds")
    shift = (int(rng.randint(shift_x_min, shift_x_max)), int(rng.randint(shift_y_min, shift_y_max)))
    return tuple((int(x_value) + int(shift[0]), int(y_value) + int(shift[1])) for x_value, y_value in points)


def signed_area(vertices: Sequence[GraphPoint]) -> float:
    total = 0.0
    for index, point in enumerate(vertices):
        nxt = vertices[(int(index) + 1) % len(vertices)]
        total += (float(point[0]) * float(nxt[1])) - (float(nxt[0]) * float(point[1]))
    return 0.5 * float(total)


def order_points_around_centroid(points: Sequence[GraphPoint]) -> Tuple[GraphPoint, ...] | None:
    unique = tuple((int(x_value), int(y_value)) for x_value, y_value in points)
    if len(unique) != 4 or len(set(unique)) != 4:
        return None
    cx = sum(float(point[0]) for point in unique) / 4.0
    cy = sum(float(point[1]) for point in unique) / 4.0
    ordered = tuple(
        sorted(
            unique,
            key=lambda point: math.atan2(float(point[1]) - float(cy), float(point[0]) - float(cx)),
        )
    )
    if abs(float(signed_area(ordered))) <= 1e-9:
        return None
    return ordered


def is_convex(vertices: Sequence[GraphPoint]) -> bool:
    signs: List[int] = []
    for index in range(len(vertices)):
        a = vertices[index]
        b = vertices[(index + 1) % len(vertices)]
        c = vertices[(index + 2) % len(vertices)]
        cross = ((int(b[0]) - int(a[0])) * (int(c[1]) - int(b[1]))) - (
            (int(b[1]) - int(a[1])) * (int(c[0]) - int(b[0]))
        )
        if int(cross) == 0:
            return False
        signs.append(1 if int(cross) > 0 else -1)
    return len(set(signs)) == 1


def classify_point_set(points: Sequence[GraphPoint]) -> str:
    ordered = order_points_around_centroid(points)
    if ordered is None or not is_convex(ordered):
        return "other"
    return str(classify_quadrilateral_kind(tuple((float(x_value), float(y_value)) for x_value, y_value in ordered)))


def is_ambiguous_for_prompt(kind: str, target_kind: str) -> bool:
    if str(kind) == str(target_kind):
        return True
    if str(target_kind) == "parallelogram_only" and str(kind) in ALL_EXACT_SHAPE_KINDS:
        return True
    if str(target_kind) in {"rectangle_non_square", "rhombus_non_square"} and str(kind) == "square":
        return True
    return False


def sample_ordered_quadrilateral(kind: str, *, rng, max_abs: int) -> Tuple[GraphPoint, GraphPoint, GraphPoint, GraphPoint]:
    for _ in range(400):
        u, v = vector_pair_for_kind(str(kind), rng)
        base_points: Tuple[GraphPoint, GraphPoint, GraphPoint, GraphPoint] = (
            (0, 0),
            (int(u[0]), int(u[1])),
            (int(u[0]) + int(v[0]), int(u[1]) + int(v[1])),
            (int(v[0]), int(v[1])),
        )
        try:
            translated = translate_points_within(base_points, rng=rng, max_abs=int(max_abs))
        except ValueError:
            continue
        start = int(rng.randrange(4))
        ordered = tuple(translated[(start + index) % 4] for index in range(4))
        if bool(rng.randrange(2)):
            ordered = (ordered[0], ordered[3], ordered[2], ordered[1])
        if classify_point_set(ordered) == str(kind):
            return ordered  # type: ignore[return-value]
    raise RuntimeError(f"failed to sample {kind} quadrilateral")


def sample_other_quadrilateral(*, rng, max_abs: int) -> Tuple[GraphPoint, GraphPoint, GraphPoint, GraphPoint]:
    for _ in range(1200):
        base_kind = str(rng.choice(ALL_EXACT_SHAPE_KINDS))
        points = list(sample_ordered_quadrilateral(base_kind, rng=rng, max_abs=int(max_abs)))
        index = int(rng.randrange(4))
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)):
            mutated = list(points)
            mutated[index] = (int(mutated[index][0]) + int(delta[0]), int(mutated[index][1]) + int(delta[1]))
            if any(max(abs(int(coord)) for coord in point) > int(max_abs) for point in mutated):
                continue
            if len(set(mutated)) != 4:
                continue
            if classify_point_set(mutated) == "other":
                return tuple(mutated)  # type: ignore[return-value]
    raise RuntimeError("failed to sample other quadrilateral point set")


def sample_panel_points(kind: str, *, rng, max_abs: int) -> Tuple[GraphPoint, GraphPoint, GraphPoint, GraphPoint]:
    if str(kind) == "other":
        return sample_other_quadrilateral(rng=rng, max_abs=int(max_abs))
    return sample_ordered_quadrilateral(str(kind), rng=rng, max_abs=int(max_abs))


def shape_distractor_kinds(target_kind: str, *, rng, count: int) -> List[str]:
    if str(target_kind) == "square":
        base = ["rectangle_non_square", "rhombus_non_square", "parallelogram_only", "other", "other"]
    elif str(target_kind) == "rectangle_non_square":
        base = ["rhombus_non_square", "parallelogram_only", "other", "other", "other"]
    elif str(target_kind) == "rhombus_non_square":
        base = ["rectangle_non_square", "parallelogram_only", "other", "other", "other"]
    elif str(target_kind) == "parallelogram_only":
        base = ["other", "other", "other", "other", "other"]
    else:
        base = ["other"] * int(count)
    while len(base) < int(count):
        base.append("other")
    rng.shuffle(base)
    return list(base[: int(count)])


def _all_grid_points(max_abs: int) -> Tuple[GraphPoint, ...]:
    limit = int(max_abs)
    return tuple((int(x), int(y)) for x in range(-limit, limit + 1) for y in range(-limit, limit + 1))


def segment_vector(segment: GraphSegment) -> GraphPoint:
    return (
        int(segment[1][0]) - int(segment[0][0]),
        int(segment[1][1]) - int(segment[0][1]),
    )


def segment_squared_length(segment: GraphSegment) -> int:
    vx, vy = segment_vector(segment)
    return int(vx) * int(vx) + int(vy) * int(vy)


def _segment_relation_flags(segments: Sequence[GraphSegment]) -> Dict[str, bool]:
    if len(segments) != 2:
        raise ValueError("segment relation panels require exactly two segments")
    first, second = tuple(segments)
    v1 = segment_vector(first)
    v2 = segment_vector(second)
    cross = int(v1[0]) * int(v2[1]) - int(v1[1]) * int(v2[0])
    dot = int(v1[0]) * int(v2[0]) + int(v1[1]) * int(v2[1])
    return {
        "parallel": int(cross) == 0,
        "perpendicular": int(dot) == 0,
        "equal_length": int(segment_squared_length(first)) == int(segment_squared_length(second)),
    }


def classify_segment_pair_relation(segments: Sequence[GraphSegment]) -> Dict[str, bool]:
    return dict(_segment_relation_flags(segments))


def _segments_are_visually_distinct(first: GraphSegment, second: GraphSegment) -> bool:
    first_points = {tuple(first[0]), tuple(first[1])}
    second_points = {tuple(second[0]), tuple(second[1])}
    if len(first_points) != 2 or len(second_points) != 2:
        return False
    if first_points == second_points:
        return False
    v1 = segment_vector(first)
    v2 = segment_vector(second)
    if int(v1[0]) == 0 and int(v1[1]) == 0:
        return False
    if int(v2[0]) == 0 and int(v2[1]) == 0:
        return False
    if _segment_relation_flags((first, second))["parallel"]:
        offset = (int(second[0][0]) - int(first[0][0]), int(second[0][1]) - int(first[0][1]))
        if int(v1[0]) * int(offset[1]) - int(v1[1]) * int(offset[0]) == 0:
            return False
    return True


def _all_candidate_segments(max_abs: int) -> Tuple[GraphSegment, ...]:
    points = _all_grid_points(int(max_abs))
    segments: List[GraphSegment] = []
    for index, first in enumerate(points):
        for second in points[index + 1 :]:
            segment = (tuple(first), tuple(second))
            length_sq = segment_squared_length(segment)
            if 4 <= int(length_sq) <= 72:
                segments.append(segment)
    return tuple(segments)


def sample_segment_pair(
    relation_kind: str,
    *,
    rng,
    max_abs: int,
    should_match: bool,
) -> Tuple[GraphSegment, GraphSegment]:
    if str(relation_kind) not in set(SEGMENT_RELATION_KINDS):
        raise ValueError(f"unsupported segment relation kind: {relation_kind}")
    candidates = list(_all_candidate_segments(int(max_abs)))
    for _ in range(2500):
        first = rng.choice(candidates)
        second = rng.choice(candidates)
        if not _segments_are_visually_distinct(first, second):
            continue
        flags = _segment_relation_flags((first, second))
        matches = bool(flags[str(relation_kind)])
        if bool(matches) == bool(should_match):
            if bool(should_match) and str(relation_kind) == "equal_length":
                if flags["parallel"] or flags["perpendicular"]:
                    continue
            return (first, second)
    raise RuntimeError(f"failed to sample segment pair for {relation_kind} should_match={should_match}")


def transform_point(point: GraphPoint, transform_kind: str, *, translation: GraphPoint = (0, 0)) -> GraphPoint:
    x_value, y_value = int(point[0]), int(point[1])
    if str(transform_kind) == "translation":
        return (x_value + int(translation[0]), y_value + int(translation[1]))
    if str(transform_kind) == "reflection_x":
        return (x_value, -y_value)
    if str(transform_kind) == "reflection_y":
        return (-x_value, y_value)
    if str(transform_kind) == "rotation_180":
        return (-x_value, -y_value)
    raise ValueError(f"unsupported transform kind: {transform_kind}")


def transform_point_set(
    points: Sequence[GraphPoint],
    transform_kind: str,
    *,
    translation: GraphPoint = (0, 0),
) -> Tuple[GraphPoint, ...]:
    return tuple(transform_point(tuple(point), str(transform_kind), translation=translation) for point in points)


def _point_set_transform_flags(
    source_points: Sequence[GraphPoint],
    candidate_points: Sequence[GraphPoint],
    *,
    translation: GraphPoint,
) -> Dict[str, bool]:
    source = tuple((int(x), int(y)) for x, y in source_points)
    candidate = {tuple((int(x), int(y))) for x, y in candidate_points}
    return {
        "translation": set(transform_point_set(source, "translation", translation=translation)) == candidate,
        "reflection_x": set(transform_point_set(source, "reflection_x")) == candidate,
        "reflection_y": set(transform_point_set(source, "reflection_y")) == candidate,
        "rotation_180": set(transform_point_set(source, "rotation_180")) == candidate,
    }


def classify_point_set_transform(
    source_points: Sequence[GraphPoint],
    candidate_points: Sequence[GraphPoint],
    *,
    translation: GraphPoint,
) -> Dict[str, bool]:
    return dict(_point_set_transform_flags(source_points, candidate_points, translation=translation))


def _is_non_collinear_point_set(points: Sequence[GraphPoint]) -> bool:
    unique = tuple((int(x), int(y)) for x, y in points)
    if len(unique) < 3 or len(set(unique)) != len(unique):
        return False
    a, b, c = unique[0], unique[1], unique[2]
    return ((int(b[0]) - int(a[0])) * (int(c[1]) - int(a[1]))) != (
        (int(b[1]) - int(a[1])) * (int(c[0]) - int(a[0]))
    )


def _sample_translation_vector(rng) -> GraphPoint:
    choices = (
        (-3, -2),
        (-3, 1),
        (-2, 3),
        (-1, -3),
        (1, 3),
        (2, -3),
        (3, -1),
        (3, 2),
    )
    return tuple(int(value) for value in rng.choice(choices))  # type: ignore[return-value]


def _points_within_bounds(points: Sequence[GraphPoint], *, max_abs: int) -> bool:
    return all(max(abs(int(coord)) for coord in point) <= int(max_abs) for point in points)


def sample_transform_panel_points(
    target_transform: str,
    *,
    rng,
    max_abs: int,
    point_count: int,
    should_match: bool,
) -> Tuple[Tuple[GraphPoint, ...], Tuple[GraphPoint, ...], GraphPoint, Dict[str, bool]]:
    """Sample source/candidate sets while preserving a unique transform relation."""

    if str(target_transform) not in set(POINT_SET_TRANSFORM_KINDS):
        raise ValueError(f"unsupported point-set transform kind: {target_transform}")
    if int(point_count) < 3:
        raise ValueError("point-set transform panels require at least three points")

    points = list(_all_grid_points(int(max_abs)))
    alternate_transforms = [kind for kind in POINT_SET_TRANSFORM_KINDS if kind != str(target_transform)]
    for _ in range(2500):
        translation = _sample_translation_vector(rng)
        source = tuple(tuple(point) for point in rng.sample(points, int(point_count)))
        if not _is_non_collinear_point_set(source):
            continue
        if str(target_transform) != "translation" and not set(source).isdisjoint(
            set(transform_point_set(source, str(target_transform)))
        ):
            continue
        winner_candidate = transform_point_set(source, str(target_transform), translation=translation)
        if not _points_within_bounds(winner_candidate, max_abs=int(max_abs)):
            continue
        if not set(source).isdisjoint(set(winner_candidate)):
            continue

        if bool(should_match):
            candidate = winner_candidate
        else:
            distractor_kind = str(rng.choice(alternate_transforms))
            candidate = transform_point_set(source, distractor_kind, translation=translation)
            if not _points_within_bounds(candidate, max_abs=int(max_abs)) or not set(source).isdisjoint(set(candidate)):
                candidate = tuple(tuple(point) for point in rng.sample(points, int(point_count)))
            if len(set(candidate)) != int(point_count) or not _points_within_bounds(candidate, max_abs=int(max_abs)):
                continue

        flags = _point_set_transform_flags(source, candidate, translation=translation)
        if bool(flags[str(target_transform)]) == bool(should_match):
            return tuple(source), tuple(candidate), tuple(translation), dict(flags)
    raise RuntimeError(f"failed to sample transform panel for {target_transform} should_match={should_match}")


__all__ = [
    "ALL_EXACT_SHAPE_KINDS",
    "POINT_SET_TRANSFORM_KINDS",
    "SEGMENT_RELATION_KINDS",
    "classify_point_set",
    "classify_point_set_transform",
    "classify_segment_pair_relation",
    "is_ambiguous_for_prompt",
    "sample_panel_points",
    "sample_segment_pair",
    "sample_transform_panel_points",
    "shape_distractor_kinds",
    "transform_point_set",
]
