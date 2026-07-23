"""Object placement and projection dataset assembly for dense clusters."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.three_d.shared.color_variation import resolve_three_d_object_fill_rgb
from trace_tasks.tasks.three_d.shared.object_scene import (
    CONTEXT_OBJECT_COLORS,
    ObjectSceneRenderParams,
    _bbox_intersection_area,
    _build_projection_frame,
    _camera_yaw_band_for_instance,
    _make_object_spec,
    _min_pairwise,
    _object_reference_points,
    _object_screen_bbox,
    _project_screen,
    _sample_camera,
)

from .defaults import (
    CLUSTER_COUNT_WEIGHTS,
    CLUSTER_DIMENSION_SCALE,
    MAX_PAIRWISE_OVERLAP_FRACTION,
    MAX_PAIRWISE_OVERLAP_PX,
    MAX_RENDERED_CUMULATIVE_OCCLUSION_FRACTION,
    MAX_RENDERED_PAIRWISE_OVERLAP_FRACTION,
    MAX_RENDERED_PAIRWISE_OVERLAP_PX,
    MIN_RENDERED_BBOX_SIDE_PX,
    MIN_RENDERED_VISIBLE_BBOX_FRACTION,
    MIN_PROJECTED_OBJECT_AREA_PX,
    OBJECT_CLUSTER_ORIENTATION_DEGREES,
    PLACEMENT_FOOTPRINT_SEPARATION_FACTOR,
    PROMPT_COLOR_RGB,
    cluster_dimensions,
    object_name_for_shape,
)
from .state import ClusterSequenceItem


def scale_dimensions(dimensions_xyz: Sequence[float], scale: float) -> Tuple[float, float, float]:
    """Scale one resource footprint for dense-cluster placement."""

    return tuple(round(float(value) * float(scale), 4) for value in dimensions_xyz)  # type: ignore[return-value]


def bbox_area(bbox: Sequence[float]) -> float:
    """Return the pixel area of one projected object box."""

    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def bbox_visible_fraction(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
) -> float:
    """Return the fraction of one object bbox that remains inside the canvas."""

    full_area = max(1.0, bbox_area(bbox))
    visible = _bbox_intersection_area(
        list(bbox),
        [0.0, 0.0, float(width), float(height)],
    )
    return float(visible) / float(full_area)


def _bbox_intersection(bbox_a: Sequence[float], bbox_b: Sequence[float]) -> List[float] | None:
    """Return the axis-aligned intersection rectangle, if non-empty."""

    x0 = max(float(bbox_a[0]), float(bbox_b[0]))
    y0 = max(float(bbox_a[1]), float(bbox_b[1]))
    x1 = min(float(bbox_a[2]), float(bbox_b[2]))
    y1 = min(float(bbox_a[3]), float(bbox_b[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def rectangle_union_area(rectangles: Sequence[Sequence[float]]) -> float:
    """Return exact union area for a small set of axis-aligned rectangles."""

    rects = [tuple(float(value) for value in rect) for rect in rectangles if bbox_area(rect) > 0.0]
    if not rects:
        return 0.0
    x_values = sorted({coord for x0, _y0, x1, _y1 in rects for coord in (x0, x1)})
    total = 0.0
    for left, right in zip(x_values, x_values[1:]):
        if right <= left:
            continue
        intervals: List[Tuple[float, float]] = []
        for x0, y0, x1, y1 in rects:
            if x0 < right and x1 > left:
                intervals.append((float(y0), float(y1)))
        if not intervals:
            continue
        merged_height = 0.0
        current_start, current_end = sorted(intervals)[0]
        for start, end in sorted(intervals)[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
            else:
                merged_height += current_end - current_start
                current_start, current_end = start, end
        merged_height += current_end - current_start
        total += (right - left) * merged_height
    return float(total)


def clip_bbox_to_canvas(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
) -> List[float]:
    """Clip one bbox to the public image coordinate range."""

    return [
        round(max(0.0, min(float(width), float(bbox[0]))), 3),
        round(max(0.0, min(float(height), float(bbox[1]))), 3),
        round(max(0.0, min(float(width), float(bbox[2]))), 3),
        round(max(0.0, min(float(height), float(bbox[3]))), 3),
    ]


def clip_object_bboxes_to_canvas(
    object_bboxes_px: Mapping[str, Sequence[float]],
    *,
    width: int,
    height: int,
) -> Dict[str, List[float]]:
    """Clip rendered object bboxes before exposing them as public annotations."""

    return {
        str(object_id): clip_bbox_to_canvas(bbox, width=int(width), height=int(height))
        for object_id, bbox in object_bboxes_px.items()
    }


def object_draw_order_ranks(object_specs: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Return renderer draw-order ranks keyed by object id.

    Rank 0 is drawn first; larger ranks are drawn later and can visually occlude
    smaller-rank objects.
    """

    ordered = sorted(
        [dict(spec) for spec in object_specs],
        key=lambda item: float(item["camera_distance"]) + float(item.get("render_order_bias", 0.0)),
        reverse=True,
    )
    return {str(spec["object_id"]): int(index) for index, spec in enumerate(ordered)}


def depth_aware_occlusion_stats(
    object_bboxes_px: Mapping[str, Sequence[float]],
    *,
    object_specs: Sequence[Mapping[str, Any]],
    width: int,
    height: int,
) -> Dict[str, Any]:
    """Measure cumulative bbox occlusion from objects painted later in draw order."""

    bboxes = {str(key): list(value) for key, value in object_bboxes_px.items()}
    ranks = object_draw_order_ranks(object_specs)
    canvas_bbox = [0.0, 0.0, float(width), float(height)]
    missing = sorted(str(object_id) for object_id in bboxes if str(object_id) not in ranks)
    per_object: Dict[str, Dict[str, float]] = {}
    max_occlusion_fraction = 0.0
    min_final_visible_fraction = 1.0
    if missing:
        return {
            "missing_draw_order_object_ids": list(missing),
            "max_depth_aware_occlusion_fraction": 1.0,
            "min_depth_aware_final_visible_bbox_fraction": 0.0,
            "per_object": {},
        }
    for object_id, bbox in bboxes.items():
        full_area = max(1.0, bbox_area(bbox))
        visible_bbox = _bbox_intersection(bbox, canvas_bbox)
        canvas_visible_area = bbox_area(visible_bbox) if visible_bbox is not None else 0.0
        occlusion_rects: List[List[float]] = []
        for other_id, other_bbox in bboxes.items():
            if str(other_id) == str(object_id):
                continue
            if int(ranks[str(other_id)]) <= int(ranks[str(object_id)]):
                continue
            if visible_bbox is None:
                continue
            intersection = _bbox_intersection(visible_bbox, other_bbox)
            if intersection is not None:
                occlusion_rects.append(intersection)
        occluded_area = min(canvas_visible_area, rectangle_union_area(occlusion_rects))
        final_visible_area = max(0.0, canvas_visible_area - occluded_area)
        canvas_visible_fraction = float(canvas_visible_area) / full_area
        occlusion_fraction = float(occluded_area) / full_area
        final_visible_fraction = float(final_visible_area) / full_area
        max_occlusion_fraction = max(max_occlusion_fraction, occlusion_fraction)
        min_final_visible_fraction = min(min_final_visible_fraction, final_visible_fraction)
        per_object[str(object_id)] = {
            "canvas_visible_fraction": round(float(canvas_visible_fraction), 5),
            "depth_aware_occlusion_fraction": round(float(occlusion_fraction), 5),
            "depth_aware_final_visible_bbox_fraction": round(float(final_visible_fraction), 5),
        }
    return {
        "missing_draw_order_object_ids": [],
        "max_depth_aware_occlusion_fraction": round(float(max_occlusion_fraction), 5),
        "min_depth_aware_final_visible_bbox_fraction": round(float(min_final_visible_fraction), 5),
        "per_object": dict(sorted(per_object.items())),
    }


def bbox_is_readable(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    min_side_px: float = 12.0,
) -> bool:
    """Check that a projected object remains visible inside the canvas."""

    box_width = float(bbox[2]) - float(bbox[0])
    box_height = float(bbox[3]) - float(bbox[1])
    if box_width < float(min_side_px) or box_height < float(min_side_px):
        return False
    return float(bbox[2]) > 4.0 and float(bbox[3]) > 4.0 and float(bbox[0]) < float(width - 4) and float(bbox[1]) < float(height - 4)


def sample_scaled_dimensions(*, rng, shape_type: str) -> Tuple[Tuple[float, float, float], float]:
    """Sample small per-instance scale while preserving object-resource proportions."""

    base = cluster_dimensions(str(shape_type))
    scale = float(rng.uniform(0.92, 1.16)) * float(CLUSTER_DIMENSION_SCALE)
    return scale_dimensions(base, scale), round(float(scale), 4)


def make_cluster_object(
    *,
    rng,
    object_id: str,
    item: ClusterSequenceItem,
    xy: Tuple[float, float],
) -> Dict[str, Any]:
    """Create a renderable object spec from a semantic count item."""

    return move_cluster_object(
        make_cluster_object_template(rng=rng, object_id=str(object_id), item=item),
        xy=xy,
    )


def make_cluster_object_template(
    *,
    rng,
    object_id: str,
    item: ClusterSequenceItem,
) -> Dict[str, Any]:
    """Create a reusable object spec whose dimensions/style are fixed before placement."""

    dimensions_xyz, dimension_scale = sample_scaled_dimensions(rng=rng, shape_type=str(item.shape_type))
    object_name = object_name_for_shape(str(item.shape_type))
    spec = _make_object_spec(
        object_id=str(object_id),
        shape_type=str(item.shape_type),
        object_role="candidate",
        xy=(0.0, 0.0),
        dimensions_xyz=dimensions_xyz,
        dimension_scale=float(dimension_scale),
        label=None,
    )
    color_name = str(item.color_name)
    fill_rgb = (
        [int(channel) for channel in PROMPT_COLOR_RGB[color_name]]
        if color_name in PROMPT_COLOR_RGB
        else [
            int(channel)
            for channel in resolve_three_d_object_fill_rgb(
                spec,
                palette=CONTEXT_OBJECT_COLORS,
                salt="object_cluster.semantic_fill",
                variation_strength=0.34,
            )
        ]
    )
    spec.update(
        {
            "object_name": str(object_name),
            "prompt_name": str(object_name),
            "nameable_for_prompt": True,
            "is_answer_candidate": False,
            "is_countable_object": True,
            "matches_query": bool(item.matches_query),
            "count_role": str(item.count_role),
            "color_name": str(color_name),
            "prompt_color_name": str(color_name),
            "fill_rgb": list(fill_rgb),
            "semantic_color": color_name in PROMPT_COLOR_RGB,
            "orientation_deg": round(
                float(rng.uniform(-float(OBJECT_CLUSTER_ORIENTATION_DEGREES), float(OBJECT_CLUSTER_ORIENTATION_DEGREES))),
                3,
            ),
            "render_order_bias": round(float(rng.uniform(-0.035, 0.035)), 5),
        }
    )
    return spec


def move_cluster_object(spec: Mapping[str, Any], *, xy: Tuple[float, float]) -> Dict[str, Any]:
    """Return one pre-sampled object spec moved to a new floor coordinate."""

    moved = dict(spec)
    _width, _depth, height = (float(value) for value in moved["dimensions_xyz"])
    moved.update(
        {
            "world_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(height * 0.5), 4)],
            "base_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), 0.0],
        }
    )
    return moved


def _weighted_cluster_count(rng) -> int:
    """Sample how many local centers compose this cluster scene."""

    roll = float(rng.random())
    cumulative = 0.0
    for count, weight in sorted(CLUSTER_COUNT_WEIGHTS.items()):
        cumulative += float(weight)
        if roll <= cumulative:
            return int(count)
    return int(max(CLUSTER_COUNT_WEIGHTS))


def sample_cluster_layout(
    rng,
    *,
    scene_variant: str,
    object_count: int,
) -> Dict[str, Any]:
    """Sample continuous cluster compactness and one or more local centers."""

    count = max(1, int(object_count))
    count_t = max(0.0, min(1.0, (float(count) - 6.0) / 14.0))
    compactness = float(rng.random())
    cluster_count = _weighted_cluster_count(rng)
    variant_base_radius = {"tabletop_pile": 2.70, "shallow_tray": 2.52, "cluster_mat": 2.84}.get(str(scene_variant), 2.70)
    variant_y_scale = {"tabletop_pile": 0.88, "shallow_tray": 0.82, "cluster_mat": 0.94}.get(str(scene_variant), 0.88)
    count_radius_factor = 1.0 + 0.15 * count_t
    compact_radius_factor = 1.12 - 0.17 * compactness
    cluster_count_radius_factor = 1.0 + 0.12 * float(cluster_count - 1)
    local_radius = float(variant_base_radius * count_radius_factor * compact_radius_factor / cluster_count_radius_factor)
    y_scale = float(variant_y_scale * (1.04 - 0.08 * compactness))
    radius_exponent = float(0.72 + 0.34 * compactness)

    center_spread = 0.0
    centers: List[Dict[str, Any]] = []
    if cluster_count == 1:
        centers.append(
            {
                "center_xy": [round(float(rng.uniform(-0.22, 0.22)), 4), round(float(rng.uniform(-0.18, 0.18)), 4)],
                "weight": 1.0,
            }
        )
    else:
        center_spread = float(variant_base_radius * (0.40 + 0.24 * (1.0 - compactness)) * (1.0 + 0.12 * count_t))
        start_angle = float(rng.uniform(0.0, 2.0 * math.pi))
        raw_weights = [float(rng.uniform(0.82, 1.18)) for _ in range(cluster_count)]
        weight_total = max(1e-9, sum(raw_weights))
        for index in range(cluster_count):
            angle = float(start_angle + (2.0 * math.pi * float(index) / float(cluster_count)) + rng.uniform(-0.34, 0.34))
            radius = float(center_spread * rng.uniform(0.84, 1.08))
            centers.append(
                {
                    "center_xy": [
                        round(float(radius * math.cos(angle)), 4),
                        round(float(radius * math.sin(angle) * 0.82), 4),
                    ],
                    "weight": round(float(raw_weights[index] / weight_total), 5),
                }
            )

    return {
        "cluster_count": int(cluster_count),
        "compactness": round(float(compactness), 5),
        "local_radius": round(float(local_radius), 5),
        "y_scale": round(float(y_scale), 5),
        "radius_exponent": round(float(radius_exponent), 5),
        "center_spread": round(float(center_spread), 5),
        "centers": list(centers),
    }


def _select_cluster_center(rng, centers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Select one local center using layout weights."""

    roll = float(rng.random())
    cumulative = 0.0
    for center in centers:
        cumulative += float(center.get("weight", 0.0))
        if roll <= cumulative:
            return center
    return centers[-1]


def sample_cluster_xy(rng, *, layout: Mapping[str, Any]) -> Tuple[float, float]:
    """Sample an elliptical floor coordinate from the current cluster layout."""

    center = _select_cluster_center(rng, list(layout["centers"]))
    center_x, center_y = (float(value) for value in center["center_xy"])
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    radius = float(rng.random() ** float(layout["radius_exponent"])) * float(layout["local_radius"])
    return (
        float(center_x + radius * math.cos(angle) + rng.uniform(-0.14, 0.14)),
        float(center_y + radius * math.sin(angle) * float(layout["y_scale"]) + rng.uniform(-0.12, 0.12)),
    )


def can_place_cluster(candidate: Mapping[str, Any], placed: Sequence[Mapping[str, Any]]) -> bool:
    """Check floor-space separation before camera projection."""

    cx, cy, _cz = (float(value) for value in candidate["world_xyz"])
    for item in placed:
        ix, iy, _iz = (float(value) for value in item["world_xyz"])
        min_distance = float(PLACEMENT_FOOTPRINT_SEPARATION_FACTOR) * (
            float(candidate["footprint_radius"]) + float(item["footprint_radius"])
        )
        if math.hypot(float(cx - ix), float(cy - iy)) < float(min_distance):
            return False
    return True


def place_cluster_objects(
    *,
    rng,
    sequence: Sequence[ClusterSequenceItem],
    scene_variant: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Place every semantic object item into a dense but separable 3D cluster."""

    layout = sample_cluster_layout(rng, scene_variant=str(scene_variant), object_count=len(sequence))
    placed: List[Dict[str, Any]] = []
    for index, item in enumerate(sequence):
        for _ in range(520):
            candidate = make_cluster_object(
                rng=rng,
                object_id=f"cluster_object_{int(index):02d}",
                item=item,
                xy=sample_cluster_xy(rng, layout=layout),
            )
            if can_place_cluster(candidate, placed):
                candidate["render_order_bias"] = round(float(rng.uniform(-0.035, 0.035)), 5)
                placed.append(candidate)
                break
        else:
            raise ValueError("could not place enough clustered 3D objects")
    return list(placed), dict(layout)


def _stable_projection_frame(
    *,
    camera,
    render_params: ObjectSceneRenderParams,
    composition_offset: Mapping[str, Any],
):
    """Build one candidate-placement frame for a camera and offset."""

    frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=[])
    return apply_composition_offset(frame, composition_offset)


def projected_object_bboxes(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> Dict[str, List[float]]:
    """Project approximate object bboxes before the expensive rendered-glyph pass."""

    return {
        str(spec["object_id"]): list(_object_screen_bbox(spec, camera, frame, pad_px=5.0))
        for spec in specs
    }


def projected_specs_are_valid(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    render_params: ObjectSceneRenderParams,
    annotation_object_ids: Sequence[str] = (),
    require_spread: bool = False,
) -> bool:
    """Validate projected placement estimates with the same hard readability caps."""

    bboxes_by_id = projected_object_bboxes(specs, camera=camera, frame=frame)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    for bbox in bboxes_by_id.values():
        if not bbox_is_readable(bbox, width=width, height=height):
            return False
        if bbox_area(bbox) < float(MIN_PROJECTED_OBJECT_AREA_PX):
            return False

    for object_id in [str(value) for value in annotation_object_ids]:
        bbox = bboxes_by_id.get(str(object_id))
        if bbox is None:
            return False
        if not bbox_is_readable(bbox, width=width, height=height, min_side_px=float(MIN_RENDERED_BBOX_SIDE_PX)):
            return False

    bbox_values = list(bboxes_by_id.values())
    for index, bbox_a in enumerate(bbox_values):
        area_a = bbox_area(bbox_a)
        for bbox_b in bbox_values[index + 1 :]:
            overlap = _bbox_intersection_area(bbox_a, bbox_b)
            if overlap > float(MAX_PAIRWISE_OVERLAP_PX):
                return False
            if overlap > float(MAX_PAIRWISE_OVERLAP_FRACTION) * min(area_a, bbox_area(bbox_b)):
                return False

    if require_spread:
        centers = [
            _project_screen(spec["world_xyz"], camera, frame)
            for spec in specs
        ]
        center_x_values = [float(point[0]) for point in centers]
        center_y_values = [float(point[1]) for point in centers]
        min_x_span, min_y_span = screen_span_requirements(len(specs), width=width, height=height)
        if center_x_values and float(max(center_x_values) - min(center_x_values)) < float(min_x_span):
            return False
        if center_y_values and float(max(center_y_values) - min(center_y_values)) < float(min_y_span):
            return False

    return True


def projected_candidate_bbox_if_valid(
    candidate: Mapping[str, Any],
    *,
    placed_bboxes: Mapping[str, Sequence[float]],
    camera,
    frame,
    render_params: ObjectSceneRenderParams,
) -> List[float] | None:
    """Return candidate projected bbox when it passes incremental readability checks."""

    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    candidate_bbox = list(_object_screen_bbox(candidate, camera, frame, pad_px=5.0))
    if not bbox_is_readable(candidate_bbox, width=width, height=height):
        return None
    if bbox_area(candidate_bbox) < float(MIN_PROJECTED_OBJECT_AREA_PX):
        return None
    if bool(candidate.get("matches_query", False)) and not bbox_is_readable(
        candidate_bbox,
        width=width,
        height=height,
        min_side_px=float(MIN_RENDERED_BBOX_SIDE_PX),
    ):
        return None

    candidate_area = bbox_area(candidate_bbox)
    for placed_bbox in placed_bboxes.values():
        overlap = _bbox_intersection_area(candidate_bbox, placed_bbox)
        if overlap > float(MAX_PAIRWISE_OVERLAP_PX):
            return None
        if overlap > float(MAX_PAIRWISE_OVERLAP_FRACTION) * min(candidate_area, bbox_area(placed_bbox)):
            return None
    return list(candidate_bbox)


def place_cluster_objects_projection_aware(
    *,
    rng,
    sequence: Sequence[ClusterSequenceItem],
    scene_variant: str,
    camera,
    render_params: ObjectSceneRenderParams,
    composition_offset: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Place clustered objects while checking projected readability incrementally."""

    layout = sample_cluster_layout(rng, scene_variant=str(scene_variant), object_count=len(sequence))
    templates = [
        make_cluster_object_template(
            rng=rng,
            object_id=f"cluster_object_{int(index):02d}",
            item=item,
        )
        for index, item in enumerate(sequence)
    ]
    ordered_indices = sorted(
        range(len(sequence)),
        key=lambda index: (
            not bool(sequence[index].matches_query),
            -float(templates[index]["footprint_radius"]),
            int(index),
        ),
    )
    placement_frame = _stable_projection_frame(
        camera=camera,
        render_params=render_params,
        composition_offset=composition_offset,
    )
    placed: List[Dict[str, Any]] = []
    placed_bboxes: Dict[str, List[float]] = {}
    candidate_attempts_total = 0
    candidate_attempts_max = 0
    candidate_attempts_by_object: Dict[str, int] = {}

    for original_index in ordered_indices:
        object_id = f"cluster_object_{int(original_index):02d}"
        for attempt_index in range(720):
            candidate_attempts_total += 1
            candidate = move_cluster_object(templates[int(original_index)], xy=sample_cluster_xy(rng, layout=layout))
            if not can_place_cluster(candidate, placed):
                continue
            candidate_bbox = projected_candidate_bbox_if_valid(
                candidate,
                placed_bboxes=placed_bboxes,
                camera=camera,
                frame=placement_frame,
                render_params=render_params,
            )
            if candidate_bbox is None:
                continue
            placed.append(candidate)
            placed_bboxes[str(object_id)] = list(candidate_bbox)
            attempts_for_object = int(attempt_index + 1)
            candidate_attempts_by_object[str(object_id)] = attempts_for_object
            candidate_attempts_max = max(candidate_attempts_max, attempts_for_object)
            break
        else:
            raise ValueError("could not place enough projected-readable clustered 3D objects")

    placed_sorted = sorted(placed, key=lambda spec: str(spec["object_id"]))
    annotation_ids = [str(spec["object_id"]) for spec in placed_sorted if bool(spec.get("matches_query", False))]
    if not projected_specs_are_valid(
        placed_sorted,
        camera=camera,
        frame=placement_frame,
        render_params=render_params,
        annotation_object_ids=annotation_ids,
        require_spread=True,
    ):
        raise ValueError("projected object cluster failed final readability constraints")

    placement_meta = {
        "placement_strategy": "projection_aware_v2",
        "placement_candidate_attempts_total": int(candidate_attempts_total),
        "placement_candidate_attempts_max_per_object": int(candidate_attempts_max),
        "placement_layout_attempts": 1,
        "placement_candidate_attempts_by_object": dict(sorted(candidate_attempts_by_object.items())),
    }
    return list(placed_sorted), dict(layout), dict(placement_meta)


def finalize_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> List[Dict[str, Any]]:
    """Attach projected screen and camera coordinates to object specs."""

    finalized_specs: List[Dict[str, Any]] = []
    for spec in specs:
        screen = _project_screen(spec["world_xyz"], camera, frame)
        finalized = dict(spec)
        finalized.update(
            {
                "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_specs.append(finalized)
    return list(finalized_specs)


def sample_composition_offset(rng, *, render_params: ObjectSceneRenderParams) -> Dict[str, Any]:
    """Sample a screen-space composition offset so clusters are not always centered."""

    roll = float(rng.random())
    if roll < 0.35:
        offset_kind = "horizontal_edge"
        dx_frac = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.18, 0.30))
        dy_frac = float(rng.uniform(-0.16, 0.16))
    elif roll < 0.60:
        offset_kind = "vertical_edge"
        dx_frac = float(rng.uniform(-0.20, 0.20))
        dy_frac = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 0.26))
    elif roll < 0.75:
        offset_kind = "corner_bias"
        dx_frac = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.16, 0.27))
        dy_frac = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.13, 0.23))
    else:
        offset_kind = "mild"
        dx_frac = float(rng.uniform(-0.16, 0.16))
        dy_frac = float(rng.uniform(-0.12, 0.12))
    usable_width = float(
        int(render_params.canvas_width)
        - int(render_params.scene_margin_left_px)
        - int(render_params.scene_margin_right_px)
    )
    usable_height = float(
        int(render_params.canvas_height)
        - int(render_params.scene_margin_top_px)
        - int(render_params.scene_margin_bottom_px)
    )
    dx_px = float(dx_frac * usable_width)
    dy_px = float(dy_frac * usable_height)
    return {
        "offset_kind": str(offset_kind),
        "dx_frac": round(float(dx_frac), 5),
        "dy_frac": round(float(dy_frac), 5),
        "dx_px": round(float(dx_px), 3),
        "dy_px": round(float(dy_px), 3),
    }


def apply_composition_offset(frame, composition_offset: Mapping[str, Any]):
    """Shift the projection frame center by one sampled screen-space offset."""

    return replace(
        frame,
        center_x=float(frame.center_x) + float(composition_offset["dx_px"]),
        center_y=float(frame.center_y) + float(composition_offset["dy_px"]),
    )


def view_is_valid(
    *,
    specs: Sequence[Mapping[str, Any]],
    camera,
    frame,
    render_params: ObjectSceneRenderParams,
) -> bool:
    """Validate projection readability and cap pairwise object occlusion."""

    bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=5.0) for spec in specs]
    if any(
        not bbox_is_readable(
            bbox,
            width=int(render_params.canvas_width),
            height=int(render_params.canvas_height),
        )
        for bbox in bboxes
    ):
        return False
    if any(bbox_area(bbox) < MIN_PROJECTED_OBJECT_AREA_PX for bbox in bboxes):
        return False
    for index, bbox_a in enumerate(bboxes):
        area_a = bbox_area(bbox_a)
        for bbox_b in bboxes[index + 1 :]:
            overlap = _bbox_intersection_area(bbox_a, bbox_b)
            if overlap > MAX_PAIRWISE_OVERLAP_PX:
                return False
            if overlap > float(MAX_PAIRWISE_OVERLAP_FRACTION) * min(area_a, bbox_area(bbox_b)):
                return False
    return True


def rendered_bboxes_are_valid(
    object_bboxes_px: Mapping[str, Sequence[float]],
    object_centers_px: Mapping[str, Sequence[float]],
    *,
    width: int,
    height: int,
    object_count: int,
    object_specs: Sequence[Mapping[str, Any]] = (),
    annotation_object_ids: Sequence[str] = (),
) -> bool:
    """Validate the final rendered object boxes that reviewers actually inspect."""

    stats = rendered_layout_stats(
        object_bboxes_px=object_bboxes_px,
        object_centers_px=object_centers_px,
        width=int(width),
        height=int(height),
        object_specs=object_specs,
    )
    if float(stats["min_visible_bbox_fraction"]) < float(MIN_RENDERED_VISIBLE_BBOX_FRACTION):
        return False
    if int(stats["center_inside_canvas_count"]) != int(stats["object_count"]):
        return False
    min_x_span, min_y_span = screen_span_requirements(int(object_count), width=int(width), height=int(height))
    if float(stats["center_x_span_px"]) < float(min_x_span) or float(stats["center_y_span_px"]) < float(min_y_span):
        return False

    clipped_bboxes = clip_object_bboxes_to_canvas(object_bboxes_px, width=int(width), height=int(height))
    bboxes = [list(bbox) for bbox in clipped_bboxes.values()]
    if any(
        not bbox_is_readable(
            bbox,
            width=int(width),
            height=int(height),
            min_side_px=12.0,
        )
        for bbox in bboxes
    ):
        return False
    annotation_ids = [str(object_id) for object_id in annotation_object_ids]
    if any(
        not bbox_is_readable(
            clipped_bboxes[str(object_id)],
            width=int(width),
            height=int(height),
            min_side_px=float(MIN_RENDERED_BBOX_SIDE_PX),
        )
        for object_id in annotation_ids
    ):
        return False
    if object_specs:
        occlusion_stats = depth_aware_occlusion_stats(
            object_bboxes_px,
            object_specs=object_specs,
            width=int(width),
            height=int(height),
        )
        if occlusion_stats["missing_draw_order_object_ids"]:
            return False
        if float(occlusion_stats["max_depth_aware_occlusion_fraction"]) > float(MAX_RENDERED_CUMULATIVE_OCCLUSION_FRACTION):
            return False
        if float(occlusion_stats["min_depth_aware_final_visible_bbox_fraction"]) < float(MIN_RENDERED_VISIBLE_BBOX_FRACTION):
            return False
    for index, bbox_a in enumerate(bboxes):
        area_a = bbox_area(bbox_a)
        for bbox_b in bboxes[index + 1 :]:
            overlap = _bbox_intersection_area(bbox_a, bbox_b)
            if overlap > float(MAX_RENDERED_PAIRWISE_OVERLAP_PX):
                return False
            if overlap > float(MAX_RENDERED_PAIRWISE_OVERLAP_FRACTION) * min(area_a, bbox_area(bbox_b)):
                return False
    return True


def screen_span_requirements(object_count: int, *, width: int, height: int) -> Tuple[float, float]:
    """Return minimum screen-space center spread for a rendered cluster."""

    count = int(object_count)
    if count <= 8:
        return float(0.24 * float(width)), float(0.16 * float(height))
    if count <= 14:
        return float(0.32 * float(width)), float(0.22 * float(height))
    return float(0.40 * float(width)), float(0.28 * float(height))


def rendered_layout_stats(
    object_bboxes_px: Mapping[str, Sequence[float]],
    object_centers_px: Mapping[str, Sequence[float]],
    *,
    width: int,
    height: int,
    object_specs: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    """Summarize final rendered visibility and spread for trace/debug audits."""

    bboxes = {str(key): list(value) for key, value in object_bboxes_px.items()}
    centers = {str(key): list(value) for key, value in object_centers_px.items()}
    visible_fractions = [
        bbox_visible_fraction(bbox, width=int(width), height=int(height))
        for bbox in bboxes.values()
    ]
    center_x_values = [float(point[0]) for point in centers.values()]
    center_y_values = [float(point[1]) for point in centers.values()]
    center_inside_count = sum(
        1
        for point in centers.values()
        if 0.0 <= float(point[0]) <= float(width) and 0.0 <= float(point[1]) <= float(height)
    )
    max_overlap_fraction = 0.0
    max_overlap_pixels = 0.0
    bbox_values = list(bboxes.values())
    for index, bbox_a in enumerate(bbox_values):
        area_a = bbox_area(bbox_a)
        for bbox_b in bbox_values[index + 1 :]:
            overlap = _bbox_intersection_area(bbox_a, bbox_b)
            max_overlap_pixels = max(max_overlap_pixels, float(overlap))
            max_overlap_fraction = max(max_overlap_fraction, float(overlap) / max(1.0, min(area_a, bbox_area(bbox_b))))
    stats = {
        "object_count": int(len(bboxes)),
        "center_inside_canvas_count": int(center_inside_count),
        "center_x_span_px": round(float(max(center_x_values) - min(center_x_values)) if center_x_values else 0.0, 3),
        "center_y_span_px": round(float(max(center_y_values) - min(center_y_values)) if center_y_values else 0.0, 3),
        "min_visible_bbox_fraction": round(float(min(visible_fractions)) if visible_fractions else 1.0, 5),
        "max_pairwise_overlap_fraction": round(float(max_overlap_fraction), 5),
        "max_pairwise_overlap_px": round(float(max_overlap_pixels), 3),
    }
    if object_specs:
        occlusion_stats = depth_aware_occlusion_stats(
            object_bboxes_px,
            object_specs=object_specs,
            width=int(width),
            height=int(height),
        )
        stats.update(
            {
                "max_depth_aware_occlusion_fraction": float(occlusion_stats["max_depth_aware_occlusion_fraction"]),
                "min_depth_aware_final_visible_bbox_fraction": float(
                    occlusion_stats["min_depth_aware_final_visible_bbox_fraction"]
                ),
            }
        )
    return stats


def camera_record(camera, *, yaw_band: Sequence[float]) -> Dict[str, Any]:
    """Serialize camera vectors and angles for deterministic trace replay."""

    return {
        "camera_position": [round(float(value), 4) for value in camera.camera_position],
        "target": [round(float(value), 4) for value in camera.target],
        "yaw_degrees": round(float(camera.yaw_degrees), 4),
        "yaw_band_degrees": [round(float(value), 4) for value in yaw_band],
        "pitch_degrees": round(float(camera.pitch_degrees), 4),
        "distance": round(float(camera.distance), 4),
        "right": [round(float(value), 5) for value in camera.right],
        "up": [round(float(value), 5) for value in camera.up],
        "forward": [round(float(value), 5) for value in camera.forward],
    }


def frame_record(frame) -> Dict[str, Any]:
    """Serialize projection-frame parameters for deterministic trace replay."""

    return {
        "scale": round(float(frame.scale), 5),
        "center_x": round(float(frame.center_x), 3),
        "center_y": round(float(frame.center_y), 3),
        "normalized_center_u": round(float(frame.normalized_center_u), 6),
        "normalized_center_v": round(float(frame.normalized_center_v), 6),
    }


def build_dataset_from_sequence(
    *,
    source_namespace: str,
    scene_kind: str,
    prompt_query_key: str,
    scene_variant: str,
    render_params: ObjectSceneRenderParams,
    instance_seed: int,
    sequence: Sequence[ClusterSequenceItem],
    target_spec: Mapping[str, Any],
    answer_value: int,
    expected_annotation_count: int,
    extra_trace: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Project one semantic object sequence into a valid rendered dataset."""

    rng = spawn_rng(int(instance_seed), f"{source_namespace}.dataset")
    selected_camera_yaw_band = _camera_yaw_band_for_instance(int(instance_seed))
    for _attempt in range(720):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        composition_offset = sample_composition_offset(rng, render_params=render_params)
        try:
            object_specs, cluster_layout, placement_meta = place_cluster_objects_projection_aware(
                rng=rng,
                sequence=sequence,
                scene_variant=str(scene_variant),
                camera=camera,
                render_params=render_params,
                composition_offset=composition_offset,
            )
        except ValueError:
            continue
        placement_meta["placement_layout_attempts"] = int(_attempt + 1)
        reference_points = [point for spec in object_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        frame = apply_composition_offset(frame, composition_offset)
        if not view_is_valid(specs=object_specs, camera=camera, frame=frame, render_params=render_params):
            continue
        finalized_specs = finalize_specs(object_specs, camera=camera, frame=frame)
        target_specs = [spec for spec in finalized_specs if bool(spec.get("matches_query", False))]
        if len(target_specs) != int(expected_annotation_count):
            continue

        distances = [float(spec["camera_distance"]) for spec in finalized_specs]
        shape_counts = Counter(str(spec["shape_type"]) for spec in finalized_specs)
        color_counts = Counter(str(spec["color_name"]) for spec in finalized_specs)
        property_counts = Counter((str(spec["shape_type"]), str(spec["color_name"])) for spec in finalized_specs)
        count_role_counts = Counter(str(spec["count_role"]) for spec in finalized_specs)
        target_object_ids = [str(spec["object_id"]) for spec in sorted(target_specs, key=lambda item: str(item["object_id"]))]
        role_object_ids: Dict[str, List[str]] = {}
        for spec in sorted(finalized_specs, key=lambda item: str(item["object_id"])):
            role = str(spec.get("count_role", ""))
            if role:
                role_object_ids.setdefault(role, []).append(str(spec["object_id"]))
        trace_extra = dict(extra_trace or {})
        trace_extra.update(dict(placement_meta))
        dataset_target_spec = dict(target_spec)
        return {
            "scene_kind": str(scene_kind),
            "query_key": str(prompt_query_key),
            "scene_variant": str(scene_variant),
            "object_count": len(finalized_specs),
            "countable_object_count": len(finalized_specs),
            "target_count": int(expected_annotation_count),
            "answer_value": int(answer_value),
            "target_spec": dict(dataset_target_spec),
            "target_shape_type": dataset_target_spec.get("target_shape_type"),
            "target_shape_types": list(dataset_target_spec.get("target_shape_types", [])),
            "target_object_name": dataset_target_spec.get("target_object_name"),
            "target_object_plural": dataset_target_spec.get("target_object_plural"),
            "target_object_union_phrase": dataset_target_spec.get("target_object_union_phrase"),
            "target_color_name": dataset_target_spec.get("target_color_name"),
            "target_color_names": list(dataset_target_spec.get("target_color_names", [])),
            "singleton_shape_types": list(dataset_target_spec.get("singleton_shape_types", [])),
            "left_operand_phrase": dataset_target_spec.get("left_operand_phrase"),
            "right_operand_phrase": dataset_target_spec.get("right_operand_phrase"),
            "arithmetic_operation": dataset_target_spec.get("arithmetic_operation"),
            "target_property_phrase": str(dataset_target_spec.get("target_property_phrase", "counted objects")),
            "target_object_ids": list(target_object_ids),
            "counted_object_ids": list(trace_extra.get("counted_object_ids", target_object_ids)),
            "role_object_ids": {str(role): list(ids) for role, ids in sorted(role_object_ids.items())},
            "cluster_layout": dict(cluster_layout),
            "cluster_count": int(cluster_layout["cluster_count"]),
            "cluster_compactness": float(cluster_layout["compactness"]),
            "composition_offset": dict(composition_offset),
            "object_specs": sorted(finalized_specs, key=lambda spec: str(spec["object_id"])),
            "point_specs": sorted(finalized_specs, key=lambda spec: str(spec["object_id"])),
            "context_object_specs": [],
            "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
            "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
            "property_counts": {
                f"{color_name}_{shape_type}": int(count)
                for (shape_type, color_name), count in sorted(property_counts.items())
            },
            "count_role_counts": {str(key): int(value) for key, value in sorted(count_role_counts.items())},
            "camera": camera_record(camera, yaw_band=selected_camera_yaw_band),
            "projection_frame": frame_record(frame),
            "solver_trace": {
                "count_predicate": str(dataset_target_spec.get("mode", "object_cluster_count")),
                "target_spec": dict(dataset_target_spec),
                "target_property_phrase": str(dataset_target_spec.get("target_property_phrase", "counted objects")),
                "target_count": int(expected_annotation_count),
                "answer_value": int(answer_value),
                "target_object_ids": list(target_object_ids),
                "role_object_ids": {str(role): list(ids) for role, ids in sorted(role_object_ids.items())},
                "cluster_layout": dict(cluster_layout),
                "cluster_count": int(cluster_layout["cluster_count"]),
                "cluster_compactness": float(cluster_layout["compactness"]),
                "composition_offset": dict(composition_offset),
                "shape_counts": {str(key): int(value) for key, value in sorted(shape_counts.items())},
                "color_counts": {str(key): int(value) for key, value in sorted(color_counts.items())},
                "property_counts": {
                    f"{color_name}_{shape_type}": int(count)
                    for (shape_type, color_name), count in sorted(property_counts.items())
                },
                "count_role_counts": {str(key): int(value) for key, value in sorted(count_role_counts.items())},
                "semantic_color_palette": {str(key): list(value) for key, value in sorted(PROMPT_COLOR_RGB.items())},
                "unique_integer_answer": True,
                "minimum_pairwise_camera_distance_margin": round(float(_min_pairwise(distances)), 4),
                **trace_extra,
            },
            **trace_extra,
        }
    raise ValueError("could not construct a valid 3D object-cluster dataset")


__all__ = [
    "build_dataset_from_sequence",
    "apply_composition_offset",
    "clip_bbox_to_canvas",
    "clip_object_bboxes_to_canvas",
    "rendered_bboxes_are_valid",
    "rendered_layout_stats",
    "sample_composition_offset",
    "screen_span_requirements",
]
