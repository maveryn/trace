"""Scene, predicate, and rendering helpers for coordinate-relation tasks."""

from __future__ import annotations

from math import gcd
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.sampling import normalize_positive_weights, weighted_choice
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import uniform_probability_map
from trace_tasks.tasks.shared.variant_sampling import has_non_null_param, is_uniform_probability_map
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_units_to_pixel, scale_point
from trace_tasks.tasks.geometry.shared.labeled_point_annotation import (
    empty_graph_point_set_annotation_artifacts,
    graph_point_set_annotation_artifacts,
)
from trace_tasks.tasks.geometry.shared.point_labels import draw_labeled_points
from trace_tasks.tasks.geometry.shared.polygon_transformations import apply_rigid_transform_recipe, translate_polygon
from trace_tasks.tasks.geometry.shared.render_variation import sample_int_render_param
from trace_tasks.tasks.geometry.shared.single_object_scene import GraphSceneContext

from .state import (
    GraphPoint,
    PixelSegment,
    PixelPoint,
    Segment,
    _DEFAULTS,
    _GEN_DEFAULTS,
    _RENDER_DEFAULTS,
    _ResolvedQuery,
    _RenderedCoordinateScene,
)

def _normalize_direction_key(vector: GraphPoint) -> GraphPoint:
    """Return one canonical direction key modulo scale and sign."""

    dx = int(vector[0])
    dy = int(vector[1])
    if dx == 0 and dy == 0:
        raise ValueError("direction vector cannot be zero")
    divisor = gcd(abs(dx), abs(dy))
    norm_dx = int(dx // max(1, divisor))
    norm_dy = int(dy // max(1, divisor))
    if norm_dx < 0 or (norm_dx == 0 and norm_dy < 0):
        norm_dx *= -1
        norm_dy *= -1
    return (int(norm_dx), int(norm_dy))


def _is_parallel(vector_a: GraphPoint, vector_b: GraphPoint) -> bool:
    """Return whether two segment vectors are parallel."""

    return _normalize_direction_key(vector_a) == _normalize_direction_key(vector_b)


def _is_perpendicular(vector_a: GraphPoint, vector_b: GraphPoint) -> bool:
    """Return whether two segment vectors are perpendicular."""

    return int(vector_a[0]) * int(vector_b[0]) + int(vector_a[1]) * int(vector_b[1]) == 0


def _segment_from_center(center: GraphPoint, half_vector: GraphPoint) -> Segment:
    """Return one integer-lattice segment from a center plus half-vector."""

    cx, cy = int(center[0]), int(center[1])
    hx, hy = int(half_vector[0]), int(half_vector[1])
    return ((cx - hx, cy - hy), (cx + hx, cy + hy))


def _segment_within_endpoint_limit(segment: Segment, *, endpoint_abs_max: int) -> bool:
    """Return whether every endpoint stays within the visible coordinate window."""

    return all(abs(int(coord)) <= int(endpoint_abs_max) for point in segment for coord in point)


def _safe_segment_centers(half_vector: GraphPoint, *, endpoint_abs_max: int) -> Tuple[GraphPoint, ...]:
    """Enumerate lattice centers whose segment endpoints stay off the graph border."""

    hx = abs(int(half_vector[0]))
    hy = abs(int(half_vector[1]))
    x_min = int(-int(endpoint_abs_max) + hx)
    x_max = int(int(endpoint_abs_max) - hx)
    y_min = int(-int(endpoint_abs_max) + hy)
    y_max = int(int(endpoint_abs_max) - hy)
    return tuple(
        (int(x_value), int(y_value))
        for x_value in range(int(x_min), int(x_max) + 1)
        for y_value in range(int(y_min), int(y_max) + 1)
    )


def _line_support_points(
    *,
    anchor: GraphPoint,
    direction: GraphPoint,
    point_abs_max: int,
) -> Tuple[GraphPoint, ...]:
    """Enumerate all lattice points on one line inside the visible graph window."""

    dx = int(direction[0])
    dy = int(direction[1])
    if gcd(abs(dx), abs(dy)) != 1:
        raise ValueError("line direction must be primitive")
    support: List[GraphPoint] = []
    for step in range(-24, 25):
        point = (int(anchor[0]) + (int(step) * dx), int(anchor[1]) + (int(step) * dy))
        if max(abs(int(point[0])), abs(int(point[1]))) <= int(point_abs_max):
            support.append(point)
    support = sorted(set(support), key=lambda point: ((point[0] * dx) + (point[1] * dy), point[0], point[1]))
    return tuple((int(point[0]), int(point[1])) for point in support)


def _quadrant_name(point: GraphPoint) -> str:
    """Return the Cartesian quadrant name for one non-axis lattice point."""

    x_value = int(point[0])
    y_value = int(point[1])
    if x_value == 0 or y_value == 0:
        raise ValueError("quadrant_name requires a point strictly off both axes")
    if x_value > 0 and y_value > 0:
        return "I"
    if x_value < 0 and y_value > 0:
        return "II"
    if x_value < 0 and y_value < 0:
        return "III"
    return "IV"


def _point_on_segment(point: GraphPoint, segment: Segment) -> bool:
    """Return whether one lattice point lies on one lattice segment."""

    px, py = int(point[0]), int(point[1])
    (ax, ay), (bx, by) = segment
    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if cross != 0:
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def _point_on_polygon_boundary(point: GraphPoint, vertices: Sequence[GraphPoint]) -> bool:
    """Return whether one lattice point lies on the polygon boundary."""

    if len(vertices) < 2:
        return False
    for index in range(len(vertices)):
        if _point_on_segment(point, (vertices[index], vertices[(index + 1) % len(vertices)])):
            return True
    return False


def _segment_orientation(a: GraphPoint, b: GraphPoint, c: GraphPoint) -> int:
    """Return the signed orientation of one ordered triplet."""

    return ((int(b[0]) - int(a[0])) * (int(c[1]) - int(a[1]))) - (
        (int(b[1]) - int(a[1])) * (int(c[0]) - int(a[0]))
    )


def _segments_intersect(segment_a: Segment, segment_b: Segment) -> bool:
    """Return whether two closed lattice segments intersect."""

    a1, a2 = segment_a
    b1, b2 = segment_b
    o1 = _segment_orientation(a1, a2, b1)
    o2 = _segment_orientation(a1, a2, b2)
    o3 = _segment_orientation(b1, b2, a1)
    o4 = _segment_orientation(b1, b2, a2)

    if o1 == 0 and _point_on_segment(b1, segment_a):
        return True
    if o2 == 0 and _point_on_segment(b2, segment_a):
        return True
    if o3 == 0 and _point_on_segment(a1, segment_b):
        return True
    if o4 == 0 and _point_on_segment(a2, segment_b):
        return True
    return bool((o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0))


def _count_target_support(*, operation_key: str, params: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve the supported answer counts for one count-style coordinate variant."""

    normalized_query = str(operation_key).strip().lower()
    if normalized_query in {"parallel", "perpendicular"}:
        support_key = "segment_target_support"
        fallback_support = _DEFAULTS.segment_target_support
        max_supported = 6
    elif normalized_query == "collinear":
        support_key = "collinear_target_support"
        fallback_support = _DEFAULTS.collinear_target_support
        max_supported = 6
    elif normalized_query == "same_quadrant":
        support_key = "same_quadrant_target_support"
        fallback_support = _DEFAULTS.same_quadrant_target_support
        max_supported = 6
    elif normalized_query == "polygon_interior":
        support_key = "point_in_shape_target_support"
        fallback_support = _DEFAULTS.point_in_shape_target_support
        max_supported = 8
    else:
        raise ValueError(f"unsupported coordinate count operation_key: {operation_key}")

    raw_support = params.get(
        support_key,
        group_default(_GEN_DEFAULTS, support_key, fallback_support),
    )
    support = []
    for value in raw_support:
        normalized = int(value)
        if 0 <= normalized <= int(max_supported) and normalized not in support:
            support.append(normalized)
    if not support:
        raise ValueError(f"{support_key} must contain at least one value in 0..{max_supported}")
    return tuple(sorted(int(value) for value in support))


def _resolve_count_target(
    rng,
    *,
    instance_seed: int,
    scene_variant: str,
    operation_key: str,
    params: Mapping[str, Any],
) -> Tuple[int, Dict[str, float]]:
    """Resolve a balanced target count for one count-style coordinate variant."""

    support = _count_target_support(operation_key=operation_key, params=params)
    explicit = params.get("target_count")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"unsupported target_count: {selected}")
        return int(selected), uniform_probability_map(support, selected=int(selected))

    raw_weights = params.get("target_count_weights", {str(value): 1.0 for value in support})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("target_count_weights must be a mapping when provided")
    weights = {
        str(key): float(value)
        for key, value in raw_weights.items()
        if int(key) in set(support)
    }
    probabilities = normalize_positive_weights(weights, default_keys=[str(value) for value in support])
    selected = int(weighted_choice(rng, probabilities, sort_keys=True))

    return int(selected), {
        str(key): float(value)
        for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))
    }


def _pixel_point(point: GraphPoint, *, context: GraphSceneContext) -> PixelPoint:
    """Project one lattice point into canonical pixel coordinates."""

    return graph_units_to_pixel(
        (int(point[0]), int(point[1])),
        graph_origin=context.graph_origin,
        spacing=int(context.graph_spacing),
    )


def _render_point(point_px: PixelPoint, *, context: GraphSceneContext) -> PixelPoint:
    """Scale one canonical pixel point into the supersampled render space."""

    return scale_point(point_px, int(context.scene_scale))


def _resolve_scene_render_params(scene_variant: str, *, params: Mapping[str, Any]) -> Dict[str, int]:
    """Resolve scene-specific graph-paper sizing defaults.

    The coordinate family now uses fixed 20x20 graph-paper windows so the
    visible geometry occupies more of the canvas without changing task
    difficulty across scene variants.
    """

    normalized_scene = str(scene_variant).strip().lower()
    if normalized_scene == "segment_set":
        return {
            "canvas_size_min": int(params.get("segment_canvas_size_min", group_default(_RENDER_DEFAULTS, "segment_canvas_size_min", 640))),
            "canvas_size_max": int(params.get("segment_canvas_size_max", group_default(_RENDER_DEFAULTS, "segment_canvas_size_max", 720))),
            "graph_cells_min": int(params.get("segment_graph_cells_min", group_default(_RENDER_DEFAULTS, "segment_graph_cells_min", 20))),
            "graph_cells_max": int(params.get("segment_graph_cells_max", group_default(_RENDER_DEFAULTS, "segment_graph_cells_max", 20))),
            "graph_origin_fraction_x": float(
                params.get("segment_graph_origin_fraction_x", group_default(_RENDER_DEFAULTS, "segment_graph_origin_fraction_x", 0.50))
            ),
            "graph_origin_fraction_y": float(
                params.get("segment_graph_origin_fraction_y", group_default(_RENDER_DEFAULTS, "segment_graph_origin_fraction_y", 0.50))
            ),
        }
    if normalized_scene == "line_points":
        return {
            "canvas_size_min": int(params.get("collinear_canvas_size_min", group_default(_RENDER_DEFAULTS, "collinear_canvas_size_min", 640))),
            "canvas_size_max": int(params.get("collinear_canvas_size_max", group_default(_RENDER_DEFAULTS, "collinear_canvas_size_max", 700))),
            "graph_cells_min": int(params.get("collinear_graph_cells_min", group_default(_RENDER_DEFAULTS, "collinear_graph_cells_min", 20))),
            "graph_cells_max": int(params.get("collinear_graph_cells_max", group_default(_RENDER_DEFAULTS, "collinear_graph_cells_max", 20))),
            "point_radius_scale": float(
                params.get("collinear_point_radius_scale", group_default(_RENDER_DEFAULTS, "collinear_point_radius_scale", 1.4))
            ),
        }
    if normalized_scene == "quadrant_points":
        return {
            "canvas_size_min": int(params.get("quadrant_canvas_size_min", group_default(_RENDER_DEFAULTS, "quadrant_canvas_size_min", 640))),
            "canvas_size_max": int(params.get("quadrant_canvas_size_max", group_default(_RENDER_DEFAULTS, "quadrant_canvas_size_max", 700))),
            "graph_cells_min": int(params.get("quadrant_graph_cells_min", group_default(_RENDER_DEFAULTS, "quadrant_graph_cells_min", 20))),
            "graph_cells_max": int(params.get("quadrant_graph_cells_max", group_default(_RENDER_DEFAULTS, "quadrant_graph_cells_max", 20))),
            "point_radius_scale": float(
                params.get("quadrant_point_radius_scale", group_default(_RENDER_DEFAULTS, "quadrant_point_radius_scale", 1.5))
            ),
            "reference_cross_scale": float(
                params.get("quadrant_reference_cross_scale", group_default(_RENDER_DEFAULTS, "quadrant_reference_cross_scale", 2.6))
            ),
        }
    if normalized_scene == "polygon_lattice":
        return {
            "canvas_size_min": int(params.get("polygon_canvas_size_min", group_default(_RENDER_DEFAULTS, "polygon_canvas_size_min", 640))),
            "canvas_size_max": int(params.get("polygon_canvas_size_max", group_default(_RENDER_DEFAULTS, "polygon_canvas_size_max", 700))),
            "graph_cells_min": int(params.get("polygon_graph_cells_min", group_default(_RENDER_DEFAULTS, "polygon_graph_cells_min", 20))),
            "graph_cells_max": int(params.get("polygon_graph_cells_max", group_default(_RENDER_DEFAULTS, "polygon_graph_cells_max", 20))),
        }
    raise ValueError(f"unsupported coordinate scene_variant: {scene_variant}")


def _draw_line_segment(draw, *, segment_px: PixelSegment, line_width: int, color: Tuple[int, int, int]) -> None:
    """Draw one rendered line segment."""

    draw.line(
        [segment_px[0], segment_px[1]],
        fill=tuple(int(value) for value in color),
        width=max(1, int(line_width)),
    )


def _draw_point_marker(
    draw,
    *,
    point_px: PixelPoint,
    radius_px: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw one filled point marker."""

    radius = max(1, int(radius_px))
    draw.ellipse(
        [
            float(point_px[0]) - float(radius),
            float(point_px[1]) - float(radius),
            float(point_px[0]) + float(radius),
            float(point_px[1]) + float(radius),
        ],
        fill=tuple(int(value) for value in color),
    )


def _draw_cross_marker(
    draw,
    *,
    point_px: PixelPoint,
    half_span_px: int,
    line_width: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw one X-shaped reference marker centered at one graph point."""

    span = max(2, int(half_span_px))
    width = max(1, int(line_width))
    px = float(point_px[0])
    py = float(point_px[1])
    draw.line(
        [(px - float(span), py - float(span)), (px + float(span), py + float(span))],
        fill=tuple(int(value) for value in color),
        width=width,
    )
    draw.line(
        [(px - float(span), py + float(span)), (px + float(span), py - float(span))],
        fill=tuple(int(value) for value in color),
        width=width,
    )


def _draw_polygon_outline(
    draw,
    *,
    vertices_px: Sequence[PixelPoint],
    line_width: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw one closed polygon outline."""

    if not vertices_px:
        return
    draw.line(
        [*vertices_px, vertices_px[0]],
        fill=tuple(int(value) for value in color),
        width=max(1, int(line_width)),
        joint="curve",
    )


def _sample_segment_count_scene(
    rng,
    *,
    query: _ResolvedQuery,
    context: GraphSceneContext,
    draw,
    line_width: int,
    point_radius_px: int,
    label_font_size_px: int,
    label_stroke_width: int,
    label_offset_px: float,
    shape_style,
    params: Mapping[str, Any],
    render_canvas_size: int,
) -> _RenderedCoordinateScene:
    """Render reference segment AB and relation-matching candidate segments."""

    half_vectors = tuple(
        (int(value[0]), int(value[1]))
        for value in params.get(
            "segment_half_vectors",
            group_default(_GEN_DEFAULTS, "segment_half_vectors", _DEFAULTS.segment_half_vectors),
        )
    )
    candidate_count = int(
        params.get(
            "segment_candidate_count",
            group_default(_GEN_DEFAULTS, "segment_candidate_count", _DEFAULTS.segment_candidate_count),
        )
    )
    endpoint_abs_max = int(
        params.get(
            "segment_endpoint_abs_max",
            group_default(_GEN_DEFAULTS, "segment_endpoint_abs_max", _DEFAULTS.segment_endpoint_abs_max),
        )
    )
    if int(query.target_count or 0) > int(candidate_count):
        raise ValueError("segment target_count exceeds the available candidate segment count")

    reference_half_vector = tuple(rng.choice(list(half_vectors)))
    reference_centers = list(_safe_segment_centers(reference_half_vector, endpoint_abs_max=int(endpoint_abs_max)))
    rng.shuffle(reference_centers)
    if not reference_centers:
        raise ValueError("segment support cannot place the target segment inside the graph window")
    reference_segment = _segment_from_center(tuple(reference_centers[0]), reference_half_vector)

    compatible_vectors = [
        vector
        for vector in half_vectors
        if (
            _is_parallel(reference_half_vector, vector)
            if str(query.operation_key) == "parallel"
            else _is_perpendicular(reference_half_vector, vector)
        )
    ]
    incompatible_vectors = [
        vector
        for vector in half_vectors
        if (
            not _is_parallel(reference_half_vector, vector)
            if str(query.operation_key) == "parallel"
            else not _is_perpendicular(reference_half_vector, vector)
        )
    ]
    if not compatible_vectors or not incompatible_vectors:
        raise ValueError("segment direction support cannot realize the requested count scene")

    candidate_ids = tuple(f"segment_{int(index) + 1}" for index in range(int(candidate_count)))
    shuffled_ids = list(candidate_ids)
    rng.shuffle(shuffled_ids)
    matching_ids = tuple(sorted(shuffled_ids[: int(query.target_count or 0)]))
    matching_id_set = set(matching_ids)

    segment_graph_by_id: Dict[str, Segment] = {}
    placed_segments: List[Segment] = [reference_segment]
    for segment_id in candidate_ids:
        desired_pool = compatible_vectors if str(segment_id) in matching_id_set else incompatible_vectors
        candidate_vectors = list(desired_pool)
        rng.shuffle(candidate_vectors)
        placed_segment: Segment | None = None
        for half_vector in candidate_vectors:
            centers = list(_safe_segment_centers(tuple(half_vector), endpoint_abs_max=int(endpoint_abs_max)))
            rng.shuffle(centers)
            for center in centers:
                candidate_segment = _segment_from_center(tuple(center), tuple(half_vector))
                if not _segment_within_endpoint_limit(candidate_segment, endpoint_abs_max=int(endpoint_abs_max)):
                    continue
                if any(_segments_intersect(candidate_segment, existing_segment) for existing_segment in placed_segments):
                    continue
                placed_segment = candidate_segment
                break
            if placed_segment is not None:
                break
        if placed_segment is None:
            raise ValueError("segment layout could not place a non-intersecting candidate inside the graph window")
        segment_graph_by_id[str(segment_id)] = placed_segment
        placed_segments.append(placed_segment)

    reference_segment_px = tuple(_pixel_point(point, context=context) for point in reference_segment)
    candidate_segment_px_by_id = {
        str(segment_id): tuple(_pixel_point(point, context=context) for point in segment)
        for segment_id, segment in segment_graph_by_id.items()
    }

    reference_segment_render_px = tuple(_render_point(point, context=context) for point in reference_segment_px)
    candidate_segment_render_px_by_id = {
        str(segment_id): tuple(_render_point(point, context=context) for point in segment_px)
        for segment_id, segment_px in candidate_segment_px_by_id.items()
    }

    blocked_segments: List[PixelSegment] = [tuple(reference_segment_render_px)] + [
        tuple(segment_px) for segment_px in candidate_segment_render_px_by_id.values()
    ]
    for segment_px in blocked_segments:
        _draw_line_segment(
            draw,
            segment_px=segment_px,
            line_width=int(line_width),
            color=shape_style.line_color,
        )
        for point_px in segment_px:
            _draw_point_marker(
                draw,
                point_px=point_px,
                radius_px=int(point_radius_px),
                color=shape_style.line_color,
            )

    draw_labeled_points(
        draw,
        points=[reference_segment_render_px[0], reference_segment_render_px[1]],
        labels=["A", "B"],
        label_offset_px=float(label_offset_px),
        font_size_px=int(label_font_size_px),
        text_stroke_width=int(label_stroke_width),
        blocked_segments=blocked_segments,
        blocked_points=[point for segment_px in candidate_segment_render_px_by_id.values() for point in segment_px],
        marker_radius_px=0,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        canvas_size=int(render_canvas_size),
    )

    annotation_value = [
        [
            [float(candidate_segment_px_by_id[str(segment_id)][0][0]), float(candidate_segment_px_by_id[str(segment_id)][0][1])],
            [float(candidate_segment_px_by_id[str(segment_id)][1][0]), float(candidate_segment_px_by_id[str(segment_id)][1][1])],
        ]
        for segment_id in matching_ids
    ]
    projected_annotation = {
        "type": "segment_set",
        "segment_set": list(annotation_value),
        "pixel_segment_set": list(annotation_value),
    }
    witness_symbolic = {
        "type": "matching_segments",
        "matching_segment_ids": list(matching_ids),
        "reference_segment_graph": [list(reference_segment[0]), list(reference_segment[1])],
        "candidate_segments_graph": {
            str(segment_id): [list(segment_graph_by_id[str(segment_id)][0]), list(segment_graph_by_id[str(segment_id)][1])]
            for segment_id in candidate_ids
        },
    }

    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "reference_segment_ab",
            "entity_type": "reference_segment",
            "label": "AB",
            "endpoints_graph": [list(reference_segment[0]), list(reference_segment[1])],
        }
    ]
    for segment_id in candidate_ids:
        segment_graph = segment_graph_by_id[str(segment_id)]
        scene_entities.append(
            {
                "entity_id": str(segment_id),
                "entity_type": "candidate_segment",
                "endpoints_graph": [list(segment_graph[0]), list(segment_graph[1])],
                "matches_query": bool(str(segment_id) in matching_id_set),
            }
        )

    return _RenderedCoordinateScene(
        scene_entities=scene_entities,
        render_map={
            "image_id": "img0",
            "reference_segment_graph": [list(reference_segment[0]), list(reference_segment[1])],
            "candidate_segments_graph": {
                str(segment_id): [list(segment_graph_by_id[str(segment_id)][0]), list(segment_graph_by_id[str(segment_id)][1])]
                for segment_id in candidate_ids
            },
            "matching_segment_ids": list(matching_ids),
        },
        answer_value=int(len(matching_ids)),
        annotation_type="segment_set",
        annotation_value=list(annotation_value),
        projected_annotation=dict(projected_annotation),
        witness_symbolic=dict(witness_symbolic),
        required_annotation_labels=list(matching_ids),
        object_count=len(candidate_ids),
        matching_labels=tuple(matching_ids),
    )


def _quadrant_support() -> Dict[str, Tuple[GraphPoint, ...]]:
    """Return reusable lattice point supports for the four quadrants."""

    return {
        "I": ((3, 2), (5, 3), (6, 5), (3, 6), (7, 2), (2, 7), (6, 3), (4, 7), (7, 6)),
        "II": ((-3, 2), (-5, 3), (-6, 5), (-3, 6), (-7, 2), (-2, 7), (-6, 3), (-4, 7), (-7, 6)),
        "III": ((-3, -2), (-5, -3), (-6, -5), (-3, -6), (-7, -2), (-2, -7), (-6, -3), (-4, -7), (-7, -6)),
        "IV": ((3, -2), (5, -3), (6, -5), (3, -6), (7, -2), (2, -7), (6, -3), (4, -7), (7, -6)),
    }


def _sample_quadrant_count_scene(
    rng,
    *,
    query: _ResolvedQuery,
    context: GraphSceneContext,
    draw,
    point_radius_px: int,
    label_font_size_px: int,
    label_stroke_width: int,
    label_offset_px: float,
    point_radius_scale: float,
    reference_cross_scale: float,
    shape_style,
    render_canvas_size: int,
) -> _RenderedCoordinateScene:
    """Render one same-quadrant counting scene with one marked reference point."""

    quadrant_points = {key: list(value) for key, value in _quadrant_support().items()}
    reference_quadrant = str(rng.choice(list(quadrant_points.keys())))
    reference_label = str(
        group_default(_GEN_DEFAULTS, "quadrant_reference_label", _DEFAULTS.quadrant_reference_label)
    ).upper()
    reference_point = tuple(rng.choice(quadrant_points[reference_quadrant]))
    same_quadrant_pool = [point for point in quadrant_points[reference_quadrant] if tuple(point) != tuple(reference_point)]
    if int(query.target_count or 0) > len(same_quadrant_pool):
        raise ValueError("same_quadrant_count target_count exceeds feasible same-quadrant support")

    candidate_labels = list(query.label_pool)
    shuffled_labels = list(candidate_labels)
    rng.shuffle(shuffled_labels)
    matching_labels = tuple(sorted(shuffled_labels[: int(query.target_count or 0)]))

    candidate_point_by_label: Dict[str, GraphPoint] = {}
    matching_positions = list(same_quadrant_pool)
    rng.shuffle(matching_positions)
    remaining_positions = [
        point
        for quadrant, points in quadrant_points.items()
        if str(quadrant) != str(reference_quadrant)
        for point in points
    ]
    rng.shuffle(remaining_positions)
    for label in candidate_labels:
        if str(label) in set(matching_labels):
            candidate_point_by_label[str(label)] = tuple(matching_positions.pop())
        else:
            candidate_point_by_label[str(label)] = tuple(remaining_positions.pop())

    reference_point_px = _pixel_point(reference_point, context=context)
    reference_point_render_px = _render_point(reference_point_px, context=context)
    candidate_points_px = {
        str(label): _pixel_point(candidate_point_by_label[str(label)], context=context)
        for label in candidate_labels
    }
    candidate_points_render_px = {
        str(label): _render_point(candidate_points_px[str(label)], context=context)
        for label in candidate_labels
    }
    point_radius = max(4, int(round(float(point_radius_px) * float(point_radius_scale))))
    reference_cross_half_span = max(
        point_radius + int(context.scene_scale),
        int(round(float(point_radius_px) * float(reference_cross_scale))),
    )

    for label in candidate_labels:
        _draw_point_marker(
            draw,
            point_px=candidate_points_render_px[str(label)],
            radius_px=int(point_radius),
            color=shape_style.line_color,
        )
    _draw_cross_marker(
        draw,
        point_px=reference_point_render_px,
        half_span_px=int(reference_cross_half_span),
        line_width=max(2, int(point_radius_px)),
        color=shape_style.line_color,
    )

    annotation = (
        graph_point_set_annotation_artifacts(
            points_by_label={
                f"match_{int(index) + 1}": candidate_points_px[str(label)]
                for index, label in enumerate(matching_labels)
            },
            graph_origin=context.graph_origin,
            graph_spacing=int(context.graph_spacing),
            witness_type="same_quadrant_points",
            ordered_labels=tuple(f"match_{int(index) + 1}" for index in range(len(matching_labels))),
        )
        if matching_labels
        else empty_graph_point_set_annotation_artifacts(witness_type="same_quadrant_points")
    )
    scene_entities = [
        {
            "entity_id": f"point_{reference_label}",
            "entity_type": "reference_point",
            "label": reference_label,
            "point_graph": list(reference_point),
            "quadrant": reference_quadrant,
        }
    ]
    for label in candidate_labels:
        point = candidate_point_by_label[str(label)]
        scene_entities.append(
            {
                "entity_id": f"point_{label}",
                "entity_type": "candidate_point",
                "label": str(label),
                "point_graph": list(point),
                "quadrant": _quadrant_name(point),
                "matches_query": bool(str(label) in set(matching_labels)),
            }
        )

    return _RenderedCoordinateScene(
        scene_entities=scene_entities,
        render_map={
            "image_id": "img0",
            "reference_point_graph": list(reference_point),
            "candidate_points_graph_by_label": {str(label): list(candidate_point_by_label[str(label)]) for label in candidate_labels},
            "matching_points_graph": [list(candidate_point_by_label[str(label)]) for label in matching_labels],
        },
        answer_value=int(len(matching_labels)),
        annotation_type=str(annotation["annotation_type"]),
        annotation_value=list(annotation["annotation_value"]),
        projected_annotation=dict(annotation["projected_annotation"]),
        witness_symbolic=dict(annotation["witness_symbolic"]),
        required_annotation_labels=list(annotation["required_labels"]),
        object_count=len(candidate_labels),
        matching_labels=tuple(matching_labels),
    )


def _sample_collinear_count_scene(
    rng,
    *,
    query: _ResolvedQuery,
    context: GraphSceneContext,
    draw,
    point_radius_px: int,
    label_font_size_px: int,
    label_stroke_width: int,
    label_offset_px: float,
    point_radius_scale: float,
    shape_style,
    params: Mapping[str, Any],
    render_canvas_size: int,
) -> _RenderedCoordinateScene:
    """Render one coordinate scene that counts dot points collinear with A and B."""

    candidate_count = int(
        params.get(
            "collinear_candidate_count",
            group_default(_GEN_DEFAULTS, "collinear_candidate_count", _DEFAULTS.collinear_candidate_count),
        )
    )
    point_abs_max = int(
        params.get(
            "collinear_point_abs_max",
            group_default(_GEN_DEFAULTS, "collinear_point_abs_max", _DEFAULTS.collinear_point_abs_max),
        )
    )
    direction_vectors = tuple(
        (int(value[0]), int(value[1]))
        for value in params.get(
            "collinear_direction_vectors",
            group_default(_GEN_DEFAULTS, "collinear_direction_vectors", _DEFAULTS.collinear_direction_vectors),
        )
    )
    target_count = int(query.target_count or 0)
    if target_count > int(candidate_count):
        raise ValueError("collinear_count target_count exceeds available candidate points")

    line_support: Tuple[GraphPoint, ...] | None = None
    reference_points: Tuple[GraphPoint, GraphPoint] | None = None
    direction = None
    for _ in range(400):
        direction = tuple(rng.choice(list(direction_vectors)))
        anchor = (
            int(rng.randint(-int(point_abs_max), int(point_abs_max))),
            int(rng.randint(-int(point_abs_max), int(point_abs_max))),
        )
        support = _line_support_points(anchor=anchor, direction=direction, point_abs_max=int(point_abs_max))
        safe_reference_points = [point for point in support if max(abs(int(point[0])), abs(int(point[1]))) <= int(point_abs_max - 1)]
        if len(safe_reference_points) < 2:
            continue
        available_count = len([point for point in support if point not in {safe_reference_points[0], safe_reference_points[-1]}])
        if int(available_count) < int(target_count):
            continue
        reference_points = (tuple(safe_reference_points[0]), tuple(safe_reference_points[-1]))
        line_support = tuple(point for point in support if point not in set(reference_points))
        break
    if line_support is None or reference_points is None or direction is None:
        raise ValueError("collinear_count could not sample a valid reference line")

    matching_points = list(line_support)
    rng.shuffle(matching_points)
    matching_points = [tuple(point) for point in matching_points[: int(target_count)]]

    distractor_pool = [
        (int(x_value), int(y_value))
        for x_value in range(-int(point_abs_max), int(point_abs_max) + 1)
        for y_value in range(-int(point_abs_max), int(point_abs_max) + 1)
        if (int(x_value), int(y_value)) not in set(line_support)
        and (int(x_value), int(y_value)) not in set(reference_points)
    ]
    rng.shuffle(distractor_pool)
    distractor_points = [
        tuple(point)
        for point in distractor_pool[: max(0, int(candidate_count) - int(target_count))]
    ]
    if len(distractor_points) != max(0, int(candidate_count) - int(target_count)):
        raise ValueError("collinear_count distractor pool exhausted unexpectedly")

    candidate_points = [*matching_points, *distractor_points]
    rng.shuffle(candidate_points)

    reference_points_px = [_pixel_point(point, context=context) for point in reference_points]
    reference_points_render_px = [_render_point(point, context=context) for point in reference_points_px]
    candidate_points_px = [_pixel_point(point, context=context) for point in candidate_points]
    candidate_points_render_px = [_render_point(point, context=context) for point in candidate_points_px]
    point_radius = max(4, int(round(float(point_radius_px) * float(point_radius_scale))))

    for point_px in candidate_points_render_px:
        _draw_point_marker(
            draw,
            point_px=point_px,
            radius_px=int(point_radius),
            color=shape_style.line_color,
        )
    for point_px in reference_points_render_px:
        _draw_point_marker(
            draw,
            point_px=point_px,
            radius_px=max(int(point_radius) + 1, int(point_radius_px)),
            color=shape_style.line_color,
        )

    draw_labeled_points(
        draw,
        points=[reference_points_render_px[0], reference_points_render_px[1]],
        labels=["A", "B"],
        label_offset_px=float(label_offset_px),
        font_size_px=int(label_font_size_px),
        text_stroke_width=int(label_stroke_width),
        blocked_segments=[],
        blocked_points=list(candidate_points_render_px),
        marker_radius_px=0,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
        canvas_size=int(render_canvas_size),
    )

    annotation = (
        graph_point_set_annotation_artifacts(
            points_by_label={
                f"match_{int(index) + 1}": _pixel_point(point, context=context)
                for index, point in enumerate(matching_points)
            },
            graph_origin=context.graph_origin,
            graph_spacing=int(context.graph_spacing),
            witness_type="collinear_points",
            ordered_labels=tuple(f"match_{int(index) + 1}" for index in range(len(matching_points))),
        )
        if matching_points
        else empty_graph_point_set_annotation_artifacts(witness_type="collinear_points")
    )

    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "reference_point_a",
            "entity_type": "reference_point",
            "label": "A",
            "point_graph": list(reference_points[0]),
        },
        {
            "entity_id": "reference_point_b",
            "entity_type": "reference_point",
            "label": "B",
            "point_graph": list(reference_points[1]),
        },
    ]
    matching_set = {tuple(point) for point in matching_points}
    for index, point in enumerate(candidate_points):
        scene_entities.append(
            {
                "entity_id": f"candidate_point_{int(index) + 1}",
                "entity_type": "candidate_point",
                "point_graph": list(point),
                "matches_query": bool(tuple(point) in matching_set),
            }
        )

    return _RenderedCoordinateScene(
        scene_entities=scene_entities,
        render_map={
            "image_id": "img0",
            "reference_points_graph": [list(reference_points[0]), list(reference_points[1])],
            "candidate_points_graph": [list(point) for point in candidate_points],
            "matching_points_graph": [list(point) for point in matching_points],
            "line_direction_graph": [int(direction[0]), int(direction[1])],
        },
        answer_value=int(len(matching_points)),
        annotation_type=str(annotation["annotation_type"]),
        annotation_value=list(annotation["annotation_value"]),
        projected_annotation=dict(annotation["projected_annotation"]),
        witness_symbolic=dict(annotation["witness_symbolic"]),
        required_annotation_labels=list(annotation["required_labels"]),
        object_count=len(candidate_points),
        matching_labels=tuple(f"match_{int(index) + 1}" for index in range(len(matching_points))),
    )


def _polygon_lattice_templates() -> Dict[int, Tuple[Tuple[GraphPoint, ...], ...]]:
    """Return one reusable lattice-polygon catalog keyed by strict interior count."""

    return {
        0: (
            ((0, 0), (2, 0), (1, 1)),
            ((0, 0), (3, 0), (2, 1), (0, 2)),
        ),
        1: (
            ((0, 0), (3, 0), (1, 2)),
            ((0, 0), (2, 0), (3, 1), (0, 2)),
        ),
        2: (
            ((0, 0), (4, 0), (1, 2)),
            ((0, 0), (3, 0), (4, 1), (0, 2)),
        ),
        3: (
            ((0, 0), (4, 0), (4, 1), (0, 2)),
            ((0, 0), (2, 0), (4, 1), (3, 2), (0, 2)),
        ),
        4: (
            ((0, 0), (4, 0), (2, 3)),
            ((0, 0), (5, 0), (2, 3)),
        ),
        5: (
            ((0, 0), (0, 2), (6, 1)),
            ((0, 0), (6, 0), (4, 1), (0, 2)),
        ),
        6: (
            ((0, 0), (0, 3), (5, 1)),
            ((0, 0), (5, 0), (5, 1), (1, 3)),
        ),
        7: (
            ((0, 0), (5, 0), (3, 4)),
            ((0, 0), (5, 0), (4, 2), (0, 3)),
        ),
        8: (
            ((0, 0), (0, 4), (5, 1)),
            ((0, 0), (5, 0), (5, 1), (2, 4)),
        ),
    }


def _point_strictly_inside_polygon(point: GraphPoint, vertices: Sequence[GraphPoint]) -> bool:
    """Return whether one lattice point lies strictly inside one simple polygon."""

    if _point_on_polygon_boundary(point, vertices):
        return False
    x_value = float(point[0])
    y_value = float(point[1])
    inside = False
    for index in range(len(vertices)):
        x_left, y_left = vertices[index]
        x_right, y_right = vertices[(index + 1) % len(vertices)]
        if (float(y_left) > float(y_value)) == (float(y_right) > float(y_value)):
            continue
        intersection_x = float(x_left) + (
            ((float(y_value) - float(y_left)) * (float(x_right) - float(x_left)))
            / (float(y_right) - float(y_left))
        )
        if float(intersection_x) > float(x_value):
            inside = not inside
    return bool(inside)


def _strict_interior_lattice_points(vertices: Sequence[GraphPoint]) -> Tuple[GraphPoint, ...]:
    """Enumerate all integer lattice points strictly inside one simple polygon."""

    min_x = min(int(point[0]) for point in vertices)
    max_x = max(int(point[0]) for point in vertices)
    min_y = min(int(point[1]) for point in vertices)
    max_y = max(int(point[1]) for point in vertices)
    support: List[GraphPoint] = []
    for x_value in range(int(min_x), int(max_x) + 1):
        for y_value in range(int(min_y), int(max_y) + 1):
            point = (int(x_value), int(y_value))
            if _point_strictly_inside_polygon(point, vertices):
                support.append(point)
    return tuple(sorted(support, key=lambda point: (int(point[0]), int(point[1]))))


def _transform_lattice_polygon(rng, vertices: Sequence[GraphPoint]) -> Tuple[GraphPoint, ...]:
    """Apply one random rigid lattice transform plus translation within the 20x20 frame."""

    transformed = tuple((float(point[0]), float(point[1])) for point in vertices)
    transformed = apply_rigid_transform_recipe(
        transformed,
        recipe=str(
            rng.choice(
                (
                    "identity",
                    "reflect_vertical",
                    "reflect_horizontal",
                    "rotate_90_cw",
                    "rotate_90_ccw",
                    "rotate_180",
                )
            )
        ),
    )
    min_x = min(int(round(point[0])) for point in transformed)
    max_x = max(int(round(point[0])) for point in transformed)
    min_y = min(int(round(point[1])) for point in transformed)
    max_y = max(int(round(point[1])) for point in transformed)
    dx_min = -8 - int(min_x)
    dx_max = 8 - int(max_x)
    dy_min = -8 - int(min_y)
    dy_max = 8 - int(max_y)
    dx = int(rng.randint(int(dx_min), int(dx_max)))
    dy = int(rng.randint(int(dy_min), int(dy_max)))
    translated = translate_polygon(transformed, dx=int(dx), dy=int(dy))
    return tuple((int(round(point[0])), int(round(point[1]))) for point in translated)


def _draw_polygon_scene_outline(
    draw,
    *,
    vertices_px: Sequence[PixelPoint],
    line_width: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw one polygon outline without fill so interior lattice points stay visible."""

    _draw_polygon_outline(
        draw,
        vertices_px=vertices_px,
        line_width=int(line_width),
        color=color,
    )


def _sample_point_in_shape_scene(
    rng,
    *,
    query: _ResolvedQuery,
    context: GraphSceneContext,
    draw,
    line_width: int,
    point_radius_px: int,
    label_font_size_px: int,
    label_stroke_width: int,
    label_offset_px: float,
    shape_style,
    params: Mapping[str, Any],
    render_canvas_size: int,
) -> _RenderedCoordinateScene:
    """Render one lattice-polygon scene that counts strict interior graph points."""

    target_count = int(query.target_count or 0)
    template_pool = _polygon_lattice_templates().get(int(target_count))
    if not template_pool:
        raise ValueError(f"unsupported point_in_shape_count target_count: {target_count}")

    polygon_vertices = _transform_lattice_polygon(rng, tuple(rng.choice(list(template_pool))))
    interior_points = _strict_interior_lattice_points(polygon_vertices)
    if len(interior_points) != int(target_count):
        raise ValueError("polygon lattice template did not realize the requested strict interior count")

    polygon_vertices_px = [_pixel_point(point, context=context) for point in polygon_vertices]
    polygon_vertices_render_px = [_render_point(point, context=context) for point in polygon_vertices_px]
    _draw_polygon_scene_outline(
        draw,
        vertices_px=polygon_vertices_render_px,
        line_width=int(line_width),
        color=shape_style.line_color,
    )

    annotation = graph_point_set_annotation_artifacts(
        points_by_label={
            f"point_{int(index) + 1}": _pixel_point(point, context=context)
            for index, point in enumerate(interior_points)
        },
        graph_origin=context.graph_origin,
        graph_spacing=int(context.graph_spacing),
        witness_type="strict_interior_lattice_points",
        ordered_labels=tuple(f"point_{int(index) + 1}" for index in range(len(interior_points))),
    ) if interior_points else empty_graph_point_set_annotation_artifacts(witness_type="strict_interior_lattice_points")

    return _RenderedCoordinateScene(
        scene_entities=[
            {
                "entity_id": "polygon_region",
                "entity_type": "polygon",
                "vertices_graph": [list(point) for point in polygon_vertices],
                "strict_interior_points_graph": [list(point) for point in interior_points],
            }
        ],
        render_map={
            "image_id": "img0",
            "scene_variant": str(query.scene_variant),
            "polygon_vertices_graph": [list(point) for point in polygon_vertices],
            "strict_interior_points_graph": [list(point) for point in interior_points],
        },
        answer_value=int(len(interior_points)),
        annotation_type=str(annotation["annotation_type"]),
        annotation_value=list(annotation["annotation_value"]),
        projected_annotation=dict(annotation["projected_annotation"]),
        witness_symbolic=dict(annotation["witness_symbolic"]),
        required_annotation_labels=list(annotation["required_labels"]),
        object_count=len(polygon_vertices),
    )


def _selection_params_for_trace(query: _ResolvedQuery) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "scene_variant": str(query.scene_variant),
        "operation_key": str(query.operation_key),
        "operation_key_probabilities": dict(query.operation_key_probabilities),
        "scene_variant_probabilities": dict(query.scene_variant_probabilities),
    }
    if query.target_count is not None:
        params["target_count"] = int(query.target_count)
        params["target_count_probabilities"] = dict(query.target_count_probabilities)
    if query.label_pool:
        params["candidate_label_pool"] = list(query.label_pool)
    return dict(params)


def _question_format_for_scene_variant(scene_variant: str) -> str:
    return {
        "segment_set": "count_segments_satisfying_relation",
        "line_points": "count_points_collinear_with_reference_line",
        "quadrant_points": "count_points_satisfying_relation",
        "polygon_lattice": "count_points_in_polygon_interior",
    }[str(scene_variant)]


def _matching_label_fields(scene_variant: str, labels: Sequence[str]) -> Dict[str, Any]:
    if not labels:
        return {}
    if str(scene_variant) == "segment_set":
        return {"matching_segment_ids": list(labels)}
    if str(scene_variant) == "quadrant_points":
        return {"matching_labels": list(labels)}
    return {}


def _execution_trace_for_trace(query: _ResolvedQuery, rendered_scene: _RenderedCoordinateScene) -> Dict[str, Any]:
    execution_trace: Dict[str, Any] = {
        "scene_variant": str(query.scene_variant),
        "operation_key": str(query.operation_key),
        "scene_variant_probabilities": dict(query.scene_variant_probabilities),
        "operation_key_probabilities": dict(query.operation_key_probabilities),
        "required_annotation_labels": list(rendered_scene.required_annotation_labels),
        "question_format": _question_format_for_scene_variant(str(query.scene_variant)),
    }
    if query.target_count is not None:
        execution_trace["target_count"] = int(query.target_count)
        execution_trace["target_count_probabilities"] = dict(query.target_count_probabilities)
    execution_trace.update(_matching_label_fields(str(query.scene_variant), rendered_scene.matching_labels))
    return dict(execution_trace)


def _scene_relations_for_trace(query: _ResolvedQuery, rendered_scene: _RenderedCoordinateScene) -> Dict[str, Any]:
    relations: Dict[str, Any] = {
        "scene_variant": str(query.scene_variant),
        "operation_key": str(query.operation_key),
    }
    relations.update(_matching_label_fields(str(query.scene_variant), rendered_scene.matching_labels))
    return dict(relations)

__all__ = [
    "_resolve_scene_render_params",
    "_sample_segment_count_scene",
    "_sample_collinear_count_scene",
    "_sample_quadrant_count_scene",
    "_sample_point_in_shape_scene",
    "_selection_params_for_trace",
    "_execution_trace_for_trace",
    "_scene_relations_for_trace",
    "_segments_intersect",
]
