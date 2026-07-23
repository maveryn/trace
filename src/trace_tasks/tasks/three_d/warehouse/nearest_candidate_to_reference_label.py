"""Robot-to-reference nearest-object task for a synthetic 3D warehouse scene."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
)
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import split_generation_rendering_prompt_defaults
from ..shared.task_support import normalize_unit as _normalize_unit
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.object_resources import (
    WAREHOUSE_NEAREST_OBJECT_CANDIDATE_TYPES,
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_DIMENSIONS,
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_NAME,
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB,
    WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE,
)
from ..shared.object_scene import (
    POINT_LABELS,
    _bbox_intersection_area,
    _build_projection_frame,
    _canvas_floor_polygon_xy,
    _object_reference_points,
    _object_screen_bbox,
    _sample_camera,
)
from ._lifecycle import build_warehouse_option_label_task_output
from .shared.state import (
    MAX_CANDIDATE_BBOX_INTERSECTION_PX,
    MIN_CANDIDATE_CENTER_SEPARATION_PX,
    MIN_CANDIDATE_VISIBLE_PX,
    ROBOT_ACCENT_COLORS,
    ROBOT_BASE_COLORS,
    SCENE_ID,
    SUPPORTED_ROBOT_DESIGNS,
    SUPPORTED_ROBOT_HEADINGS,
    SUPPORTED_SCENE_VARIANTS,
    WAREHOUSE_CAMERA_YAW_BANDS_DEGREES,
    _WarehouseRenderParams,
    _bbox_area,
    _dimensions_for_object,
    _finalize_specs,
    _heading_vector,
    _heading_axis,
    _local_to_world,
    _make_object_spec,
    _resolve_camera_yaw_band,
    _resolve_render_params,
    _sample_reference_and_objects,
)
from .shared.rendering import render_warehouse_robot_nearest_scene_3d


TASK_ID = "task_three_d__warehouse__nearest_candidate_to_reference_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("closest_object_to_reference", "closest_object_to_robot")
SUPPORTED_AISLE_HEADINGS: Tuple[str, ...] = tuple(SUPPORTED_ROBOT_HEADINGS)
REFERENCE_OBJECT_TYPE = WAREHOUSE_NEAREST_REFERENCE_OBJECT_TYPE
REFERENCE_OBJECT_NAME = WAREHOUSE_NEAREST_REFERENCE_OBJECT_NAME
REFERENCE_OBJECT_RGB: Tuple[int, int, int] = WAREHOUSE_NEAREST_REFERENCE_OBJECT_RGB
MIN_NEAREST_REFERENCE_OBJECT_MARGIN = 0.42
MIN_NEAREST_OBJECT_MARGIN = 0.42
MIN_REFERENCE_VISIBLE_PX = 26.0
OBJECT_CANDIDATE_TYPES: Tuple[str, ...] = WAREHOUSE_NEAREST_OBJECT_CANDIDATE_TYPES
def _surface_gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax, ay, _az = (float(value) for value in a["world_xyz"])
    bx, by, _bz = (float(value) for value in b["world_xyz"])
    center_distance_xy = math.hypot(float(ax - bx), float(ay - by))
    return max(0.0, float(center_distance_xy) - float(a["footprint_radius"]) - float(b["footprint_radius"]))


def _can_place(candidate: Mapping[str, Any], placed: Sequence[Mapping[str, Any]], *, clearance: float = 0.16) -> bool:
    for item in placed:
        cx, cy, _cz = (float(value) for value in candidate["world_xyz"])
        ix, iy, _iz = (float(value) for value in item["world_xyz"])
        min_distance = float(candidate["footprint_radius"]) + float(item["footprint_radius"]) + float(clearance)
        if math.hypot(float(cx - ix), float(cy - iy)) < min_distance:
            return False
    return True


def _make_reference_object(*, xy: Tuple[float, float], scale: float) -> Dict[str, Any]:
    dimensions = tuple(round(float(value) * float(scale), 4) for value in WAREHOUSE_NEAREST_REFERENCE_OBJECT_DIMENSIONS)
    return {
        "object_id": "warehouse_reference_red_sphere",
        "object_type": REFERENCE_OBJECT_TYPE,
        "object_name": REFERENCE_OBJECT_NAME,
        "prompt_name": REFERENCE_OBJECT_NAME,
        "object_role": "warehouse_reference_object",
        "orientation_axis": "x",
        "is_answer_candidate": False,
        "dimension_scale": round(float(scale), 4),
        "world_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(dimensions[2] * 0.5), 4)],
        "base_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), 0.0],
        "dimensions_xyz": [round(float(value), 4) for value in dimensions],
        "footprint_radius": round(float(0.5 * math.sqrt(dimensions[0] * dimensions[0] + dimensions[1] * dimensions[1])), 4),
        "fill_rgb": [int(channel) for channel in REFERENCE_OBJECT_RGB],
    }


def _make_robot_candidate(
    *,
    rng,
    object_id: str,
    xy: Tuple[float, float],
    heading: str,
    label: str | None,
    robot_design: str | None = None,
) -> Dict[str, Any]:
    """Build one robot option while preserving footprint metadata."""
    orientation_axis = "x" if str(heading) in {"east", "west"} else "y"
    robot_design = str(robot_design) if robot_design is not None else str(rng.choice(SUPPORTED_ROBOT_DESIGNS))
    if robot_design not in set(SUPPORTED_ROBOT_DESIGNS):
        raise ValueError(f"unsupported robot_design: {robot_design}")
    dimensions = _dimensions_for_object("warehouse_robot", orientation_axis=str(orientation_axis), scale=float(rng.uniform(0.94, 1.10)))
    width, depth, height = dimensions
    if robot_design == "sensor_tower":
        dimensions = (round(width * 0.96, 4), round(depth * 0.96, 4), round(height * 1.22, 4))
    elif robot_design == "stacker_bot":
        dimensions = (round(width * 0.94, 4), round(depth * 0.98, 4), round(height * 1.36, 4))
    spec = _make_object_spec(
        object_id=str(object_id),
        object_type="warehouse_robot",
        object_role="warehouse_robot_candidate",
        xy=xy,
        orientation_axis=str(orientation_axis),
        dimensions_xyz=dimensions,
        dimension_scale=1.0,
        label=label,
    )
    heading_xy = _heading_vector(str(heading))
    width, depth, height = (float(value) for value in dimensions)
    gripper_extent = width * 0.76 if str(orientation_axis) == "x" else depth * 0.76
    spec.update(
        {
            "robot_design": str(robot_design),
            "robot_heading": str(heading),
            "robot_base_rgb": [int(channel) for channel in ROBOT_BASE_COLORS[int(rng.randrange(len(ROBOT_BASE_COLORS)))]],
            "robot_accent_rgb": [int(channel) for channel in ROBOT_ACCENT_COLORS[int(rng.randrange(len(ROBOT_ACCENT_COLORS)))]],
            "gripper_tip_xyz": [
                round(float(xy[0]) + float(heading_xy[0]) * float(gripper_extent), 4),
                round(float(xy[1]) + float(heading_xy[1]) * float(gripper_extent), 4),
                round(float(height * 0.58), 4),
            ],
        }
    )
    return spec


def _make_object_candidate(
    *,
    rng,
    object_id: str,
    object_type: str,
    xy: Tuple[float, float],
    orientation_axis: str,
    label: str | None,
) -> Dict[str, Any]:
    scale = float(rng.uniform(0.88, 1.14))
    dimensions = _dimensions_for_object(str(object_type), orientation_axis=str(orientation_axis), scale=float(scale))
    return _make_object_spec(
        object_id=str(object_id),
        object_type=str(object_type),
        object_role="warehouse_object_candidate",
        xy=xy,
        orientation_axis=str(orientation_axis),
        dimensions_xyz=dimensions,
        dimension_scale=float(scale),
        label=label,
    )


def _attach_object_nearest_answers(
    candidate_specs: Sequence[Mapping[str, Any]],
    reference_spec: Mapping[str, Any],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    candidate_count: int,
    nearest_flag_key: str,
    distance_key: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Relabel objects so exactly one nearest option is correct."""
    ordered = sorted(candidate_specs, key=lambda spec: (_surface_gap(spec, reference_spec), str(spec["object_id"])))
    first = dict(ordered[0])
    second = dict(ordered[1])
    margin = float(_surface_gap(second, reference_spec) - _surface_gap(first, reference_spec))
    if margin < MIN_NEAREST_OBJECT_MARGIN:
        raise ValueError("nearest warehouse object margin too small")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label")
    answer_label = str(rng.choice(tuple(POINT_LABELS[: int(candidate_count)])))
    remaining_labels = [str(label) for label in POINT_LABELS[: int(candidate_count)] if str(label) != str(answer_label)]
    assignment_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.object_answer_label_assignment")
    assignment_rng.shuffle(remaining_labels)
    relabeled: List[Dict[str, Any]] = []
    for spec in candidate_specs:
        updated = dict(spec)
        is_answer = str(updated["object_id"]) == str(first["object_id"])
        label = answer_label if is_answer else remaining_labels.pop()
        distance = _surface_gap(updated, reference_spec)
        updated.update(
            {
                "object_id": f"warehouse_object_{label}",
                "point_id": f"warehouse_object_{label}",
                "point_label": str(label),
                "object_label": str(label),
                "is_answer_candidate": True,
                str(nearest_flag_key): bool(is_answer),
                str(distance_key): round(float(distance), 4),
            }
        )
        relabeled.append(updated)
    answer_spec = next(spec for spec in relabeled if bool(spec[str(nearest_flag_key)]))
    distance_order = [
        str(spec["point_label"])
        for spec in sorted(relabeled, key=lambda item: (float(item[str(distance_key)]), str(item["point_label"])))
    ]
    return list(sorted(relabeled, key=lambda spec: str(spec["point_label"]))), {
        "answer_label": str(answer_label),
        "answer_spec": dict(answer_spec),
        "nearest_margin": round(float(margin), 4),
        "distance_order": list(distance_order),
    }


def _sample_reference_and_object_candidates(
    *,
    rng,
    candidate_count: int,
    context_object_count: int,
    aisle_heading: str,
    render_params: _WarehouseRenderParams,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Place a red reference object and small object candidates for nearest queries."""
    forward_xy = _heading_vector(str(aisle_heading))
    orientation_axis = _heading_axis(str(aisle_heading))
    _unused_reference, _unused_candidates, context_specs, scene_geometry = _sample_reference_and_objects(
        rng=rng,
        candidate_count=max(4, int(candidate_count)),
        context_object_count=int(context_object_count),
        robot_heading=str(aisle_heading),
        render_params=render_params,
    )
    origin_xy = scene_geometry["origin_xy"]
    reference_xy = _local_to_world(
        forward_s=float(rng.uniform(0.18, 0.72)),
        lateral_l=float(rng.uniform(-0.16, 0.16)),
        origin_xy=origin_xy,
        forward_xy=forward_xy,
    )
    reference_spec = _make_reference_object(xy=reference_xy, scale=float(rng.uniform(0.92, 1.10)))
    placed: List[Dict[str, Any]] = [dict(reference_spec)]
    ref_x, ref_y, _ref_z = (float(value) for value in reference_spec["world_xyz"])

    object_types = [str(object_type) for object_type in OBJECT_CANDIDATE_TYPES]
    rng.shuffle(object_types)

    answer_type = object_types.pop()
    answer_probe = _make_object_candidate(
        rng=rng,
        object_id="object_answer_probe",
        object_type=str(answer_type),
        xy=(0.0, 0.0),
        orientation_axis=str(orientation_axis),
        label=None,
    )
    angle = float(rng.choice([0.25, 1.10, 2.05, 2.95, 3.85, 4.80, 5.65]) + rng.uniform(-0.16, 0.16))
    answer_radius = float(answer_probe["footprint_radius"])
    answer_distance = float(reference_spec["footprint_radius"]) + float(answer_radius) + float(rng.uniform(0.18, 0.30))
    answer_xy = (float(ref_x + math.cos(angle) * answer_distance), float(ref_y + math.sin(angle) * answer_distance))
    answer_spec = _make_object_candidate(
        rng=rng,
        object_id="object_answer_slot",
        object_type=str(answer_type),
        xy=answer_xy,
        orientation_axis=str(orientation_axis),
        label="?",
    )
    if not _can_place(answer_spec, placed, clearance=0.12):
        raise ValueError("could not place nearest warehouse object")
    placed.append(answer_spec)

    candidate_specs: List[Dict[str, Any]] = [answer_spec]
    ring_specs = [
        (1.78, -2.18),
        (2.18, -1.18),
        (2.16, 1.18),
        (1.46, 2.08),
        (-0.18, 2.26),
        (-1.46, 2.02),
        (-2.04, 0.92),
        (-2.10, -0.92),
        (-1.16, -2.08),
        (0.38, -2.34),
    ]
    rng.shuffle(ring_specs)
    for index in range(int(candidate_count) - 1):
        placed_spec: Dict[str, Any] | None = None
        object_type = str(object_types.pop() if object_types else rng.choice(OBJECT_CANDIDATE_TYPES))
        for radial_distance, base_angle in ring_specs[index:] + ring_specs[:index]:
            for _jitter_attempt in range(8):
                theta = float(base_angle + rng.uniform(-0.20, 0.20))
                distance = float(radial_distance + rng.uniform(0.00, 0.42))
                xy = (float(ref_x + math.cos(theta) * distance), float(ref_y + math.sin(theta) * distance))
                spec = _make_object_candidate(
                    rng=rng,
                    object_id=f"object_distractor_slot_{index}",
                    object_type=str(object_type),
                    xy=xy,
                    orientation_axis=str(orientation_axis),
                    label="?",
                )
                if _surface_gap(spec, reference_spec) <= _surface_gap(answer_spec, reference_spec) + MIN_NEAREST_REFERENCE_OBJECT_MARGIN:
                    continue
                if not _can_place(spec, placed, clearance=0.14):
                    continue
                placed_spec = spec
                break
            if placed_spec is not None:
                break
        if placed_spec is None:
            raise ValueError("could not place warehouse object distractor")
        placed.append(placed_spec)
        candidate_specs.append(placed_spec)

    candidate_specs, answer_meta = _attach_object_nearest_answers(
        candidate_specs,
        reference_spec,
        params=params,
        instance_seed=int(instance_seed),
        candidate_count=int(candidate_count),
        nearest_flag_key="is_nearest_object_to_reference_object",
        distance_key="distance_to_reference_object",
    )
    return [reference_spec], candidate_specs, context_specs, scene_geometry, answer_meta


def _sample_robot_and_object_candidates(
    *,
    rng,
    candidate_count: int,
    context_object_count: int,
    aisle_heading: str,
    render_params: _WarehouseRenderParams,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """Place reference robot and object candidates for nearest queries."""
    forward_xy = _heading_vector(str(aisle_heading))
    orientation_axis = _heading_axis(str(aisle_heading))
    _unused_reference_specs, _unused_candidates, context_specs, scene_geometry = _sample_reference_and_objects(
        rng=rng,
        candidate_count=max(4, int(candidate_count)),
        context_object_count=int(context_object_count),
        robot_heading=str(aisle_heading),
        render_params=render_params,
    )
    robot_xy = tuple(float(value) for value in scene_geometry["robot_xy"])
    robot_spec = _make_robot_candidate(
        rng=rng,
        object_id="warehouse_robot_reference",
        xy=(float(robot_xy[0]), float(robot_xy[1])),
        heading=str(aisle_heading),
        label=None,
        robot_design=str(rng.choice(("sensor_tower", "stacker_bot"))),
    )
    robot_spec.update(
        {
            "object_role": "warehouse_reference_robot",
            "is_answer_candidate": False,
            "point_id": None,
            "point_label": None,
            "object_label": None,
        }
    )
    placed: List[Dict[str, Any]] = [dict(robot_spec)]
    robot_x, robot_y, _robot_z = (float(value) for value in robot_spec["world_xyz"])

    object_types = [str(object_type) for object_type in OBJECT_CANDIDATE_TYPES]
    rng.shuffle(object_types)

    answer_type = object_types.pop()
    answer_probe = _make_object_candidate(
        rng=rng,
        object_id="object_answer_probe",
        object_type=str(answer_type),
        xy=(0.0, 0.0),
        orientation_axis=str(orientation_axis),
        label=None,
    )
    angle = float(rng.choice([0.18, 0.92, 1.72, 2.56, 3.32, 4.10, 5.12]) + rng.uniform(-0.16, 0.16))
    answer_distance = float(robot_spec["footprint_radius"]) + float(answer_probe["footprint_radius"]) + float(rng.uniform(0.22, 0.36))
    answer_xy = (float(robot_x + math.cos(angle) * answer_distance), float(robot_y + math.sin(angle) * answer_distance))
    answer_spec = _make_object_candidate(
        rng=rng,
        object_id="object_answer_slot",
        object_type=str(answer_type),
        xy=answer_xy,
        orientation_axis=str(orientation_axis),
        label="?",
    )
    if not _can_place(answer_spec, placed, clearance=0.10):
        raise ValueError("could not place nearest warehouse object")
    placed.append(answer_spec)

    candidate_specs: List[Dict[str, Any]] = [answer_spec]
    ring_specs = [
        (1.60, -2.34),
        (1.92, -1.26),
        (2.12, -0.12),
        (1.78, 1.08),
        (2.22, 2.06),
        (1.48, 2.86),
        (2.36, 3.72),
        (1.86, 4.70),
        (2.18, 5.56),
    ]
    rng.shuffle(ring_specs)
    for index in range(int(candidate_count) - 1):
        placed_spec: Dict[str, Any] | None = None
        object_type = str(object_types.pop() if object_types else rng.choice(OBJECT_CANDIDATE_TYPES))
        for radial_distance, base_angle in ring_specs[index:] + ring_specs[:index]:
            for _jitter_attempt in range(9):
                theta = float(base_angle + rng.uniform(-0.22, 0.22))
                distance = float(radial_distance + rng.uniform(0.06, 0.48))
                xy = (float(robot_x + math.cos(theta) * distance), float(robot_y + math.sin(theta) * distance))
                spec = _make_object_candidate(
                    rng=rng,
                    object_id=f"object_distractor_slot_{index}",
                    object_type=str(object_type),
                    xy=xy,
                    orientation_axis=str(orientation_axis),
                    label="?",
                )
                if _surface_gap(spec, robot_spec) <= _surface_gap(answer_spec, robot_spec) + MIN_NEAREST_OBJECT_MARGIN:
                    continue
                if not _can_place(spec, placed, clearance=0.14):
                    continue
                placed_spec = spec
                break
            if placed_spec is not None:
                break
        if placed_spec is None:
            raise ValueError("could not place warehouse object distractor")
        placed.append(placed_spec)
        candidate_specs.append(placed_spec)

    candidate_specs, answer_meta = _attach_object_nearest_answers(
        candidate_specs,
        robot_spec,
        params=params,
        instance_seed=int(instance_seed),
        candidate_count=int(candidate_count),
        nearest_flag_key="is_nearest_object_to_reference_robot",
        distance_key="distance_to_reference_robot",
    )
    robot_spec["nearest_object_labels"] = [str(answer_meta["answer_label"])]
    return [robot_spec], candidate_specs, context_specs, scene_geometry, answer_meta


def _visibility_ok_nearest(
    candidate_specs: Sequence[Mapping[str, Any]],
    reference_specs: Sequence[Mapping[str, Any]],
    context_specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    render_params: _WarehouseRenderParams,
) -> bool:
    """Reject nearest-reference layouts with ambiguous projected candidates."""
    candidate_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=18.0) for spec in candidate_specs]
    candidate_centers = [(float(spec["screen_xy"][0]), float(spec["screen_xy"][1])) for spec in candidate_specs]
    for bbox in candidate_bboxes:
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width < MIN_CANDIDATE_VISIBLE_PX or height < MIN_CANDIDATE_VISIBLE_PX:
            return False
        if (
            float(bbox[0]) < -28.0
            or float(bbox[1]) < -28.0
            or float(bbox[2]) > float(render_params.canvas_width + 28)
            or float(bbox[3]) > float(render_params.canvas_height + 28)
        ):
            return False
    for index, center in enumerate(candidate_centers):
        for other_index in range(index + 1, len(candidate_centers)):
            other = candidate_centers[other_index]
            if math.hypot(center[0] - other[0], center[1] - other[1]) < MIN_CANDIDATE_CENTER_SEPARATION_PX:
                return False
            if _bbox_intersection_area(candidate_bboxes[index], candidate_bboxes[other_index]) > MAX_CANDIDATE_BBOX_INTERSECTION_PX:
                return False
    reference_bboxes = []
    for reference in reference_specs:
        ref_bbox = _object_screen_bbox(reference, camera, frame, pad_px=10.0)
        width = float(ref_bbox[2]) - float(ref_bbox[0])
        height = float(ref_bbox[3]) - float(ref_bbox[1])
        if width < MIN_REFERENCE_VISIBLE_PX or height < MIN_REFERENCE_VISIBLE_PX:
            return False
        if (
            float(ref_bbox[0]) < -28.0
            or float(ref_bbox[1]) < -28.0
            or float(ref_bbox[2]) > float(render_params.canvas_width + 28)
            or float(ref_bbox[3]) > float(render_params.canvas_height + 28)
        ):
            return False
        reference_bboxes.append(list(ref_bbox))
    for candidate_index, candidate_bbox in enumerate(candidate_bboxes):
        candidate_area = max(1.0, _bbox_area(candidate_bbox))
        for ref_bbox in reference_bboxes:
            overlap = _bbox_intersection_area(candidate_bbox, ref_bbox)
            if overlap > 1800.0 and overlap / candidate_area > 0.20:
                return False
    context_bboxes = {str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=8.0) for spec in context_specs}
    checked_specs = [*candidate_specs, *reference_specs]
    checked_bboxes = candidate_bboxes + [_object_screen_bbox(spec, camera, frame, pad_px=10.0) for spec in reference_specs]
    for checked_index, checked in enumerate(checked_specs):
        checked_bbox = checked_bboxes[checked_index]
        checked_area = max(1.0, _bbox_area(checked_bbox))
        for context in context_specs:
            if str(context.get("object_type")) == "shelf_rack":
                continue
            if float(context["camera_distance"]) >= float(checked["camera_distance"]) - 0.05:
                continue
            overlap = _bbox_intersection_area(checked_bbox, context_bboxes[str(context["object_id"])])
            if overlap > 2200.0 and overlap / checked_area > 0.24:
                return False
    return True


def _build_dataset(
    *,
    params: Mapping[str, Any],
    query_id: str,
    scene_variant: str,
    aisle_heading: str,
    candidate_count: int,
    context_object_count: int,
    camera_yaw_band: Tuple[float, float],
    camera_yaw_band_index: int,
    render_params: _WarehouseRenderParams,
    instance_seed: int,
) -> Dict[str, Any]:
    """Sample a visible nearest-reference scene with bound answers."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    for _attempt in range(520):
        camera = _sample_camera(rng, yaw_band_degrees=tuple(float(value) for value in camera_yaw_band))
        if str(query_id) == "closest_object_to_reference":
            reference_specs, candidate_specs, context_specs, scene_geometry, answer_meta = _sample_reference_and_object_candidates(
                rng=rng,
                candidate_count=int(candidate_count),
                context_object_count=int(context_object_count),
                aisle_heading=str(aisle_heading),
                render_params=render_params,
                params=params,
                instance_seed=int(instance_seed),
            )
        elif str(query_id) == "closest_object_to_robot":
            reference_specs, candidate_specs, context_specs, scene_geometry, answer_meta = _sample_robot_and_object_candidates(
                rng=rng,
                candidate_count=int(candidate_count),
                context_object_count=int(context_object_count),
                aisle_heading=str(aisle_heading),
                render_params=render_params,
                params=params,
                instance_seed=int(instance_seed),
            )
        else:
            raise ValueError(f"unsupported query_id: {query_id}")
        all_specs = [*reference_specs, *candidate_specs, *context_specs]
        reference_points: List[Tuple[float, float, float]] = []
        for spec in all_specs:
            if str(spec.get("object_type")) == "shelf_rack":
                continue
            reference_points.extend(_object_reference_points(spec))
            if str(spec.get("object_type")) == "warehouse_robot" and isinstance(spec.get("gripper_tip_xyz"), Sequence):
                reference_points.append(tuple(float(value) for value in spec["gripper_tip_xyz"]))
        reference_points.extend(tuple(float(value) for value in point) for point in scene_geometry["main_aisle_polygon"])
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        if not _canvas_floor_polygon_xy(camera=camera, frame=frame, render_params=render_params):
            continue
        finalized_reference = _finalize_specs(reference_specs, camera=camera, frame=frame)
        finalized_candidates = _finalize_specs(candidate_specs, camera=camera, frame=frame)
        finalized_context = _finalize_specs(context_specs, camera=camera, frame=frame)
        if not _visibility_ok_nearest(finalized_candidates, finalized_reference, finalized_context, camera=camera, frame=frame, render_params=render_params):
            continue
        answer_label = str(answer_meta["answer_label"])
        answer_spec = next(spec for spec in finalized_candidates if str(spec["point_label"]) == answer_label)
        if str(query_id) == "closest_object_to_reference":
            if not bool(answer_spec.get("is_nearest_object_to_reference_object", False)):
                continue
            distance_by_label = {str(spec["point_label"]): round(float(spec["distance_to_reference_object"]), 4) for spec in finalized_candidates}
            nearest_by_label = {str(spec["point_label"]): bool(spec.get("is_nearest_object_to_reference_object", False)) for spec in finalized_candidates}
            predicate = f"option-panel warehouse object with the smallest ground-plane surface gap to the {REFERENCE_OBJECT_NAME}"
            reference_object_name = REFERENCE_OBJECT_NAME
            nearest_robot_by_label: Dict[str, bool] = {}
            nearest_object_by_label = dict(sorted(nearest_by_label.items()))
            answer_robot_id = ""
            answer_object_id = str(answer_spec["object_id"])
            nearest_robot_candidate_labels: List[str] = []
            nearest_object_candidate_labels = [str(answer_label)]
        else:
            if not bool(answer_spec.get("is_nearest_object_to_reference_robot", False)):
                continue
            distance_by_label = {str(spec["point_label"]): round(float(spec["distance_to_reference_robot"]), 4) for spec in finalized_candidates}
            nearest_by_label = {str(spec["point_label"]): bool(spec.get("is_nearest_object_to_reference_robot", False)) for spec in finalized_candidates}
            predicate = "option-panel warehouse object with the smallest ground-plane surface gap to the robot"
            reference_object_name = "robot"
            nearest_robot_by_label = {}
            nearest_object_by_label = dict(sorted(nearest_by_label.items()))
            answer_robot_id = ""
            answer_object_id = str(answer_spec["object_id"])
            nearest_robot_candidate_labels = []
            nearest_object_candidate_labels = [str(answer_label)]
        candidate_projected_bboxes = {
            str(spec["point_label"]): [round(float(value), 3) for value in _object_screen_bbox(spec, camera, frame, pad_px=0.0)]
            for spec in finalized_candidates
        }
        candidate_object_types = {str(spec["point_label"]): str(spec["object_type"]) for spec in finalized_candidates}
        object_type_counts = Counter(str(spec["object_type"]) for spec in [*finalized_reference, *finalized_candidates, *finalized_context])
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "candidate_count": int(candidate_count),
            "context_object_count": int(context_object_count),
            "aisle_heading": str(aisle_heading),
            "reference_object": dict(finalized_reference[0]),
            "reference_object_name": str(reference_object_name),
            "reference_specs": list(finalized_reference),
            "reference_object_specs": list(finalized_reference),
            "reference_robot_specs": list(finalized_reference) if str(query_id) == "closest_object_to_robot" else [],
            "candidate_specs": sorted(finalized_candidates, key=lambda spec: str(spec["point_label"])),
            "candidate_robot_specs": [],
            "candidate_object_specs": sorted(finalized_candidates, key=lambda spec: str(spec["point_label"])),
            "context_object_specs": sorted(finalized_context, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted([*finalized_reference, *finalized_candidates, *finalized_context], key=lambda spec: str(spec["object_id"])),
            "target_object_ids": [str(answer_spec["object_id"])],
            "answer_label": str(answer_label),
            "answer_object_id": str(answer_object_id),
            "answer_robot_id": str(answer_robot_id),
            "answer_object_type": str(answer_spec["object_type"]),
            "nearest_candidate_labels": [str(answer_label)],
            "nearest_robot_candidate_labels": list(nearest_robot_candidate_labels),
            "nearest_object_candidate_labels": list(nearest_object_candidate_labels),
            "nearest_candidate_by_label": dict(sorted(nearest_by_label.items())),
            "nearest_robot_by_label": dict(nearest_robot_by_label),
            "nearest_object_by_label": dict(nearest_object_by_label),
            "distance_to_reference_by_label": dict(sorted(distance_by_label.items())),
            "distance_to_reference_object_by_label": dict(sorted(distance_by_label.items())) if str(query_id) == "closest_object_to_reference" else {},
            "distance_to_reference_robot_by_label": dict(sorted(distance_by_label.items())) if str(query_id) == "closest_object_to_robot" else {},
            "distance_order_near_to_far": [str(label) for label in answer_meta["distance_order"]],
            "nearest_margin": float(answer_meta["nearest_margin"]),
            "nearest_reference_object_margin": float(answer_meta["nearest_margin"]) if str(query_id) == "closest_object_to_reference" else 0.0,
            "nearest_object_margin": float(answer_meta["nearest_margin"]),
            "candidate_projected_bboxes_by_label": dict(sorted(candidate_projected_bboxes.items())),
            "candidate_object_types_by_label": dict(sorted(candidate_object_types.items())),
            "object_type_counts": dict(sorted(object_type_counts.items())),
            "object_count": int(1 + len(finalized_candidates) + len(finalized_context)),
            "main_aisle_polygon_world": list(scene_geometry["main_aisle_polygon"]),
            "shelf_zone_polygons_world": list(scene_geometry["shelf_zone_polygons"]),
            "camera": {
                "camera_position": [round(float(value), 4) for value in camera.camera_position],
                "target": [round(float(value), 4) for value in camera.target],
                "yaw_degrees": round(float(camera.yaw_degrees), 4),
                "yaw_band_index": int(camera_yaw_band_index),
                "yaw_band_degrees": [round(float(value), 4) for value in camera_yaw_band],
                "pitch_degrees": round(float(camera.pitch_degrees), 4),
                "distance": round(float(camera.distance), 4),
                "right": [round(float(value), 5) for value in camera.right],
                "up": [round(float(value), 5) for value in camera.up],
                "forward": [round(float(value), 5) for value in camera.forward],
            },
            "projection_frame": {
                "scale": round(float(frame.scale), 5),
                "center_x": round(float(frame.center_x), 3),
                "center_y": round(float(frame.center_y), 3),
                "normalized_center_u": round(float(frame.normalized_center_u), 6),
                "normalized_center_v": round(float(frame.normalized_center_v), 6),
            },
            "solver_trace": {
                "predicate": str(predicate),
                "reference_object_id": str(finalized_reference[0]["object_id"]),
                "reference_object_name": str(reference_object_name),
                "nearest_candidate_by_label": dict(sorted(nearest_by_label.items())),
                "nearest_robot_by_label": dict(nearest_robot_by_label),
                "nearest_object_by_label": dict(nearest_object_by_label),
                "distance_to_reference_by_label": dict(sorted(distance_by_label.items())),
                "distance_to_reference_object_by_label": dict(sorted(distance_by_label.items())) if str(query_id) == "closest_object_to_reference" else {},
                "distance_to_reference_robot_by_label": dict(sorted(distance_by_label.items())) if str(query_id) == "closest_object_to_robot" else {},
                "distance_order_near_to_far": [str(label) for label in answer_meta["distance_order"]],
                "answer_label": str(answer_label),
                "answer_object_id": str(answer_object_id),
                "answer_robot_id": str(answer_robot_id),
                "unique_answer": True,
            },
        }
    raise ValueError("could not construct a visible warehouse nearest-reference scene")






_TASK_GROUP_DEFAULTS = get_scene_defaults("three_d", "warehouse")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _build_retry_locked_params(instance_seed: int, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Lock generation axes once for stable retry semantics."""
    locked_params = dict(params)
    query_id, _query_probabilities = _shared_resolve_axis_variant(
        params=params,
        task_id=TASK_ID,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_QUERY_IDS,
        explicit_key="query_id",
        weights_key="query_id_weights",
        balance_flag_key="balanced_query_id_sampling",
        axis_namespace="query_id",
        allow_locked=True,
    )
    scene_variant, _scene_probabilities = _shared_resolve_axis_variant(
        params=params,
        task_id=TASK_ID,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
        allow_locked=True,
    )
    aisle_heading, _heading_probabilities = _shared_resolve_axis_variant(
        params=params,
        task_id=TASK_ID,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_AISLE_HEADINGS,
        explicit_key="aisle_heading",
        weights_key="aisle_heading_weights",
        balance_flag_key="balanced_aisle_heading_sampling",
        axis_namespace="aisle_heading",
        allow_locked=True,
    )
    candidate_count, _candidate_probabilities = _shared_resolve_count(
        params=params,
        task_id=TASK_ID,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        key="candidate_count",
        default_min=4,
        default_max=4,
        lower=4,
        upper=4,
        allow_locked=True,
    )
    context_object_count, _context_probabilities = _shared_resolve_count(
        params=params,
        task_id=TASK_ID,
        gen_defaults=_GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        key="context_object_count",
        default_min=10,
        default_max=10,
        lower=8,
        upper=13,
        allow_locked=True,
    )
    _camera_yaw_band, _camera_probabilities, camera_yaw_band_index = _resolve_camera_yaw_band(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.camera_yaw_band_index",
    )
    locked_params.update(
        {
            "_locked_query_id": str(query_id),
            "_locked_scene_variant": str(scene_variant),
            "_locked_aisle_heading": str(aisle_heading),
            "_locked_candidate_count": int(candidate_count),
            "_locked_context_object_count": int(context_object_count),
            "_locked_camera_yaw_band_index": int(camera_yaw_band_index),
        }
    )
    return locked_params


@register_task
class ThreeDWarehouseNearestCandidateToReferenceLabelTask:
    """Choose the option-panel warehouse item closest to a reference item."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
    domain = "three_d"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        last_error: Exception | None = None
        retry_params = _build_retry_locked_params(int(instance_seed), params)
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = int(instance_seed) if attempt_index == 0 else int(spawn_rng(int(instance_seed), f"{TASK_ID}.attempt_seed.{attempt_index}").randrange(1, 2**62))
            try:
                return self._generate_once(int(attempt_seed), params=retry_params)
            except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
                last_error = exc
        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}")

    def _generate_once(self, instance_seed: int, *, params: Dict[str, Any]) -> TaskOutput:
        """Resolve axes, sample a scene, and assemble verifier output."""
        query_id, query_probabilities = _shared_resolve_axis_variant(
            params=params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            supported_variants=SUPPORTED_QUERY_IDS,
            explicit_key="query_id",
            weights_key="query_id_weights",
            balance_flag_key="balanced_query_id_sampling",
            axis_namespace="query_id",
            allow_locked=True,
        )
        scene_variant, scene_probabilities = _shared_resolve_axis_variant(
            params=params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            supported_variants=SUPPORTED_SCENE_VARIANTS,
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
            balance_flag_key="balanced_scene_variant_sampling",
            axis_namespace="scene_variant",
            allow_locked=True,
        )
        aisle_heading, aisle_heading_probabilities = _shared_resolve_axis_variant(
            params=params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            supported_variants=SUPPORTED_AISLE_HEADINGS,
            explicit_key="aisle_heading",
            weights_key="aisle_heading_weights",
            balance_flag_key="balanced_aisle_heading_sampling",
            axis_namespace="aisle_heading",
            allow_locked=True,
        )
        candidate_count, candidate_count_probabilities = _shared_resolve_count(
            params=params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="candidate_count",
            default_min=4,
            default_max=4,
            lower=4,
            upper=4,
            allow_locked=True,
        )
        context_object_count, context_count_probabilities = _shared_resolve_count(
            params=params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="context_object_count",
            default_min=10,
            default_max=10,
            lower=8,
            upper=13,
            allow_locked=True,
        )
        camera_yaw_band, camera_yaw_probabilities, camera_yaw_band_index = _resolve_camera_yaw_band(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.camera_yaw_band_index",
        )
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_dataset(
            params=params,
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            aisle_heading=str(aisle_heading),
            candidate_count=int(candidate_count),
            context_object_count=int(context_object_count),
            camera_yaw_band=tuple(camera_yaw_band),
            camera_yaw_band_index=int(camera_yaw_band_index),
            render_params=render_params,
            instance_seed=int(instance_seed),
        )
        return build_warehouse_option_label_task_output(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_query_key=str(query_id),
            dynamic_prompt_slots={"reference_object_name": str(dataset["reference_object_name"])},
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            params=params,
            instance_seed=int(instance_seed),
            query_id=str(query_id),
            query_probabilities=query_probabilities,
            scene_variant=str(scene_variant),
            scene_probabilities=scene_probabilities,
            candidate_count=int(candidate_count),
            candidate_count_probabilities=candidate_count_probabilities,
            context_object_count=int(context_object_count),
            context_object_count_probabilities=context_count_probabilities,
            camera_yaw_band_index=int(camera_yaw_band_index),
            camera_yaw_probabilities=camera_yaw_probabilities,
            render_params=render_params,
            dataset=dataset,
            render_scene=render_warehouse_robot_nearest_scene_3d,
            candidate_specs=dataset["candidate_specs"],
            scene_kind="three_d_warehouse_robot",
            view_family="synthetic_perspective_3d_warehouse_robot",
            scene_relation_fields={
                "aisle_heading": str(dataset["aisle_heading"]),
                "reference_object_name": str(dataset["reference_object_name"]),
                "nearest_candidate_by_label": dict(dataset["nearest_candidate_by_label"]),
                "nearest_robot_by_label": dict(dataset["nearest_robot_by_label"]),
                "nearest_object_by_label": dict(dataset["nearest_object_by_label"]),
                "answer_robot_id": str(dataset["answer_robot_id"]),
            },
            query_param_fields={
                "aisle_heading": str(aisle_heading),
                "aisle_heading_probabilities": dict(aisle_heading_probabilities),
            },
            render_spec_fields={"aisle_heading": str(dataset["aisle_heading"])},
            execution_trace_fields={
                "answer_robot_id": str(dataset["answer_robot_id"]),
                "reference_object": dict(dataset["reference_object"]),
                "reference_specs": [dict(spec) for spec in dataset["reference_specs"]],
                "reference_object_specs": [dict(spec) for spec in dataset["reference_object_specs"]],
                "reference_robot_specs": [dict(spec) for spec in dataset["reference_robot_specs"]],
                "candidate_specs": [dict(spec) for spec in dataset["candidate_specs"]],
                "candidate_robot_specs": [dict(spec) for spec in dataset["candidate_robot_specs"]],
                "candidate_object_specs": [dict(spec) for spec in dataset["candidate_object_specs"]],
                "aisle_heading": str(dataset["aisle_heading"]),
                "reference_object_name": str(dataset["reference_object_name"]),
                "main_aisle_polygon_world": list(dataset["main_aisle_polygon_world"]),
                "shelf_zone_polygons_world": list(dataset["shelf_zone_polygons_world"]),
                "nearest_candidate_labels": list(dataset["nearest_candidate_labels"]),
                "nearest_robot_candidate_labels": list(dataset["nearest_robot_candidate_labels"]),
                "nearest_object_candidate_labels": list(dataset["nearest_object_candidate_labels"]),
                "nearest_candidate_by_label": dict(dataset["nearest_candidate_by_label"]),
                "nearest_robot_by_label": dict(dataset["nearest_robot_by_label"]),
                "nearest_object_by_label": dict(dataset["nearest_object_by_label"]),
                "distance_to_reference_by_label": dict(dataset["distance_to_reference_by_label"]),
                "distance_to_reference_object_by_label": dict(dataset["distance_to_reference_object_by_label"]),
                "distance_to_reference_robot_by_label": dict(dataset["distance_to_reference_robot_by_label"]),
                "distance_order_near_to_far": list(dataset["distance_order_near_to_far"]),
                "nearest_margin": float(dataset["nearest_margin"]),
                "nearest_reference_object_margin": float(dataset["nearest_reference_object_margin"]),
                "nearest_object_margin": float(dataset["nearest_object_margin"]),
                "candidate_projected_bboxes_by_label": dict(dataset["candidate_projected_bboxes_by_label"]),
                "candidate_object_types_by_label": dict(dataset["candidate_object_types_by_label"]),
            },
        )


__all__ = [
    "MIN_NEAREST_OBJECT_MARGIN",
    "MIN_NEAREST_REFERENCE_OBJECT_MARGIN",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDWarehouseNearestCandidateToReferenceLabelTask",
]
