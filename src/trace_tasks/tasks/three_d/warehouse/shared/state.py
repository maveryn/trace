"""Shared warehouse scene specs, sampling helpers, and geometry utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.color_distance import coerce_rgb as _rgb
from ...shared.object_resources import (
    WAREHOUSE_CONTEXT_OBJECT_TYPES,
    WAREHOUSE_OBJECT_BASE_DIMENSIONS,
    WAREHOUSE_OBJECT_COLORS,
    WAREHOUSE_OBJECT_NAMES,
    WAREHOUSE_SMALL_OBJECT_CANDIDATE_TYPES,
    WAREHOUSE_RADIAL_OBJECT_TYPES,
    WAREHOUSE_ROBOT_ACCENT_COLORS,
    WAREHOUSE_ROBOT_BASE_COLORS,
    WAREHOUSE_ROBOT_DESIGNS,
    WAREHOUSE_ROBOT_HEADINGS,
    WAREHOUSE_SHELF_FRAME_COLORS,
    WAREHOUSE_SHELF_LOAD_COLORS,
    WAREHOUSE_SHELF_RACK_STYLES,
)
from ...shared.object_scene import _CameraSpec, _ProjectionFrame, _project_screen
from ...shared.canvas import resolve_three_d_canvas_spec
from ...shared.task_support import resolve_support_choice_for_namespace
from ...shared.task_support import float_value as _float_value
from ...shared.task_support import int_value as _int_value
from ...shared.visual_styles import resolve_three_d_surface_tone


SCENE_ID = "warehouse"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("storage_aisle", "loading_zone", "packing_floor")
SUPPORTED_ROBOT_HEADINGS: Tuple[str, ...] = WAREHOUSE_ROBOT_HEADINGS
SUPPORTED_ROBOT_DESIGNS: Tuple[str, ...] = WAREHOUSE_ROBOT_DESIGNS
SUPPORTED_SHELF_RACK_STYLES: Tuple[str, ...] = WAREHOUSE_SHELF_RACK_STYLES
WAREHOUSE_CAMERA_YAW_BANDS_DEGREES: Tuple[Tuple[float, float], ...] = (
    (-54.0, -28.0),
    (28.0, 54.0),
    (-146.0, -116.0),
    (116.0, 146.0),
)
CONTEXT_OBJECT_TYPES: Tuple[str, ...] = WAREHOUSE_CONTEXT_OBJECT_TYPES
OBJECT_NAMES: Dict[str, str] = dict(WAREHOUSE_OBJECT_NAMES)
OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = dict(WAREHOUSE_OBJECT_COLORS)
ROBOT_BASE_COLORS: Tuple[Tuple[int, int, int], ...] = WAREHOUSE_ROBOT_BASE_COLORS
ROBOT_ACCENT_COLORS: Tuple[Tuple[int, int, int], ...] = WAREHOUSE_ROBOT_ACCENT_COLORS
SHELF_FRAME_COLORS: Tuple[Tuple[int, int, int], ...] = WAREHOUSE_SHELF_FRAME_COLORS
SHELF_LOAD_COLORS: Tuple[Tuple[int, int, int], ...] = WAREHOUSE_SHELF_LOAD_COLORS
PATH_CORRIDOR_HALF_WIDTH = 0.42
MIN_FORWARD_DISTANCE = 0.72
MIN_CANDIDATE_VISIBLE_PX = 24.0
MIN_CANDIDATE_CENTER_SEPARATION_PX = 30.0
MAX_CANDIDATE_BBOX_INTERSECTION_PX = 9000.0


@dataclass(frozen=True)
class _WarehouseRenderParams:
    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    room_extent: float
    grid_step: float
    marker_radius_px: int
    label_font_size_px: int
    line_width_px: int
    floor_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    aisle_rgb: Tuple[int, int, int]
    shelf_zone_rgb: Tuple[int, int, int]
    path_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    background_tone_id: str = "custom"
    background_tone_rgb: Tuple[int, int, int] = (221, 226, 220)
    surface_accent_rgb: Tuple[int, int, int] = (207, 215, 211)
    canvas_preset: str = "explicit"
    canvas_policy: str = "explicit_dimensions"


def _resolve_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int = 0,
    namespace: str = "three_d.warehouse.canvas",
) -> _WarehouseRenderParams:
    """Resolve warehouse canvas, surface tone, and projection styling together.

    The returned params are shared by sampling and rendering, so projected
    object visibility checks use the same margins and dimensions as the image.
    """

    merged = dict(render_defaults)
    merged.update(dict(params))
    canvas = resolve_three_d_canvas_spec(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        fallback_width=_int_value(merged, "canvas_width", 1200),
        fallback_height=_int_value(merged, "canvas_height", 800),
    )
    tone = resolve_three_d_surface_tone(
        params=params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.surface_tone",
    )
    return _WarehouseRenderParams(
        canvas_width=int(canvas.canvas_width),
        canvas_height=int(canvas.canvas_height),
        canvas_preset=str(canvas.preset_id),
        canvas_policy=str(canvas.policy),
        scene_margin_left_px=_int_value(merged, "scene_margin_left_px", 48),
        scene_margin_right_px=_int_value(merged, "scene_margin_right_px", 48),
        scene_margin_top_px=_int_value(merged, "scene_margin_top_px", 42),
        scene_margin_bottom_px=_int_value(merged, "scene_margin_bottom_px", 52),
        room_extent=_float_value(merged, "room_extent", 4.6),
        grid_step=_float_value(merged, "grid_step", 0.72),
        marker_radius_px=_int_value(merged, "marker_radius_px", 20),
        label_font_size_px=_int_value(merged, "label_font_size_px", 25),
        line_width_px=_int_value(merged, "line_width_px", 2),
        floor_rgb=_rgb(params.get("floor_rgb", tone.floor_rgb), tone.floor_rgb),
        grid_rgb=_rgb(params.get("grid_rgb", tone.grid_rgb), tone.grid_rgb),
        aisle_rgb=_rgb(params.get("aisle_rgb", tone.surface_accent_rgb), tone.surface_accent_rgb),
        shelf_zone_rgb=_rgb(params.get("shelf_zone_rgb", tone.grid_rgb), tone.grid_rgb),
        path_rgb=_rgb(merged.get("path_rgb", (236, 195, 72)), (236, 195, 72)),
        text_rgb=_rgb(params.get("text_rgb", tone.text_rgb), tone.text_rgb),
        text_stroke_rgb=_rgb(params.get("text_stroke_rgb", tone.text_stroke_rgb), tone.text_stroke_rgb),
        background_tone_id=str(tone.tone_id),
        background_tone_rgb=tuple(int(value) for value in tone.floor_rgb),
        surface_accent_rgb=tuple(int(value) for value in tone.surface_accent_rgb),
    )


def _resolve_camera_yaw_band(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> Tuple[Tuple[float, float], Dict[str, float], int]:
    """Resolve one supported warehouse camera-yaw band for task-owned sampling."""

    support = tuple(range(len(WAREHOUSE_CAMERA_YAW_BANDS_DEGREES)))
    explicit = params.get("camera_yaw_band_index")
    locked = params.get("_locked_camera_yaw_band_index")
    if explicit is not None:
        selected = int(explicit)
        probabilities = {
            str(key): (1.0 if int(key) == int(selected) else 0.0)
            for key in support
        }
    elif locked is not None:
        selected = int(locked)
        probabilities = {str(key): 1.0 / float(len(support)) for key in support}
    else:
        selected, raw_probabilities = resolve_support_choice_for_namespace(
            params=params,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            support_values=support,
            explicit_key="camera_yaw_band_index",
            locked_key="_locked_camera_yaw_band_index",
        )
        probabilities = {
            str(key): float(value) for key, value in raw_probabilities.items()
        }
    if int(selected) not in set(support):
        raise ValueError(f"unsupported camera_yaw_band_index: {selected}")
    return (
        tuple(float(value) for value in WAREHOUSE_CAMERA_YAW_BANDS_DEGREES[int(selected)]),
        {
            str(key): float(value)
            for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))
        },
        int(selected),
    )


def _heading_vector(robot_heading: str) -> Tuple[float, float]:
    if str(robot_heading) == "east":
        return (1.0, 0.0)
    if str(robot_heading) == "west":
        return (-1.0, 0.0)
    if str(robot_heading) == "north":
        return (0.0, 1.0)
    if str(robot_heading) == "south":
        return (0.0, -1.0)
    raise ValueError(f"unsupported robot_heading: {robot_heading}")


def _heading_axis(robot_heading: str) -> str:
    return "x" if str(robot_heading) in {"east", "west"} else "y"


def _local_to_world(
    *,
    forward_s: float,
    lateral_l: float,
    origin_xy: Sequence[float],
    forward_xy: Sequence[float],
) -> Tuple[float, float]:
    fx, fy = float(forward_xy[0]), float(forward_xy[1])
    lx, ly = -fy, fx
    return (
        round(float(origin_xy[0]) + fx * float(forward_s) + lx * float(lateral_l), 4),
        round(float(origin_xy[1]) + fy * float(forward_s) + ly * float(lateral_l), 4),
    )


def _world_to_robot_path(
    xy: Sequence[float],
    *,
    robot_xy: Sequence[float],
    forward_xy: Sequence[float],
) -> Tuple[float, float]:
    fx, fy = float(forward_xy[0]), float(forward_xy[1])
    lx, ly = -fy, fx
    rx = float(xy[0]) - float(robot_xy[0])
    ry = float(xy[1]) - float(robot_xy[1])
    return (round(rx * fx + ry * fy, 4), round(rx * lx + ry * ly, 4))


def _dimensions_for_object(object_type: str, *, orientation_axis: str, scale: float) -> Tuple[float, float, float]:
    length, width, height = WAREHOUSE_OBJECT_BASE_DIMENSIONS.get(str(object_type), (0.64, 0.52, 0.52))
    if str(object_type) in WAREHOUSE_RADIAL_OBJECT_TYPES:
        return (round(width * scale, 4), round(width * scale, 4), round(height * scale, 4))
    if str(orientation_axis) == "y":
        return (round(width * scale, 4), round(length * scale, 4), round(height * scale, 4))
    return (round(length * scale, 4), round(width * scale, 4), round(height * scale, 4))


def _make_object_spec(
    *,
    object_id: str,
    object_type: str,
    object_role: str,
    xy: Tuple[float, float],
    orientation_axis: str,
    dimensions_xyz: Tuple[float, float, float],
    dimension_scale: float,
    label: str | None,
) -> Dict[str, Any]:
    width, depth, height = (float(value) for value in dimensions_xyz)
    footprint_radius = 0.5 * math.sqrt(width * width + depth * depth)
    spec: Dict[str, Any] = {
        "object_id": str(object_id),
        "object_type": str(object_type),
        "object_name": str(OBJECT_NAMES.get(str(object_type), str(object_type).replace("_", " "))),
        "prompt_name": str(OBJECT_NAMES.get(str(object_type), str(object_type).replace("_", " "))),
        "object_role": str(object_role),
        "orientation_axis": str(orientation_axis),
        "is_answer_candidate": bool(label),
        "dimension_scale": round(float(dimension_scale), 4),
        "world_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(height * 0.5), 4)],
        "base_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), 0.0],
        "dimensions_xyz": [round(width, 4), round(depth, 4), round(height, 4)],
        "footprint_radius": round(float(footprint_radius), 4),
    }
    if label is not None:
        spec.update({"point_id": f"warehouse_object_{label}", "point_label": str(label), "object_label": str(label)})
    return spec


def _finalize_specs(specs: Sequence[Mapping[str, Any]], *, camera: _CameraSpec, frame: _ProjectionFrame) -> List[Dict[str, Any]]:
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


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _scene_palette(scene_variant: str, render_params: _WarehouseRenderParams) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    del scene_variant
    return render_params.floor_rgb, render_params.grid_rgb, render_params.aisle_rgb, render_params.shelf_zone_rgb


def _projected_bbox(points: Sequence[Sequence[float]]) -> List[float]:
    return [
        round(float(min(float(point[0]) for point in points)), 3),
        round(float(min(float(point[1]) for point in points)), 3),
        round(float(max(float(point[0]) for point in points)), 3),
        round(float(max(float(point[1]) for point in points)), 3),
    ]


def _camera_from_dataset(dataset: Mapping[str, Any]) -> _CameraSpec:
    raw = dataset["camera"]
    return _CameraSpec(
        camera_position=tuple(float(value) for value in raw["camera_position"]),
        target=tuple(float(value) for value in raw["target"]),
        right=tuple(float(value) for value in raw["right"]),
        up=tuple(float(value) for value in raw["up"]),
        forward=tuple(float(value) for value in raw["forward"]),
        yaw_degrees=float(raw["yaw_degrees"]),
        pitch_degrees=float(raw["pitch_degrees"]),
        distance=float(raw["distance"]),
    )


def _frame_from_dataset(dataset: Mapping[str, Any]) -> _ProjectionFrame:
    raw = dataset["projection_frame"]
    return _ProjectionFrame(
        scale=float(raw["scale"]),
        center_x=float(raw["center_x"]),
        center_y=float(raw["center_y"]),
        normalized_center_u=float(raw["normalized_center_u"]),
        normalized_center_v=float(raw["normalized_center_v"]),
    )


def _make_path_polygon(
    *,
    robot_xy: Sequence[float],
    forward_xy: Sequence[float],
    start_s: float,
    end_s: float,
    half_width: float,
) -> List[Tuple[float, float, float]]:
    fx, fy = float(forward_xy[0]), float(forward_xy[1])
    lx, ly = -fy, fx
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    return [
        (rx + fx * start_s + lx * half_width, ry + fy * start_s + ly * half_width, 0.022),
        (rx + fx * end_s + lx * half_width, ry + fy * end_s + ly * half_width, 0.022),
        (rx + fx * end_s - lx * half_width, ry + fy * end_s - ly * half_width, 0.022),
        (rx + fx * start_s - lx * half_width, ry + fy * start_s - ly * half_width, 0.022),
    ]


def _sample_reference_and_objects(
    *,
    rng,
    candidate_count: int,
    context_object_count: int,
    robot_heading: str,
    render_params: _WarehouseRenderParams,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Sample robot reference, candidates, context, and aisle geometry."""
    del render_params
    forward_xy = _heading_vector(str(robot_heading))
    orientation_axis = _heading_axis(str(robot_heading))
    origin_xy = (float(rng.uniform(-0.18, 0.18)), float(rng.uniform(-0.18, 0.18)))
    robot_s = -1.72
    robot_xy = _local_to_world(forward_s=robot_s, lateral_l=0.0, origin_xy=origin_xy, forward_xy=forward_xy)
    robot_design = str(rng.choice(SUPPORTED_ROBOT_DESIGNS))
    robot_dims = _dimensions_for_object("warehouse_robot", orientation_axis=str(orientation_axis), scale=float(rng.uniform(0.96, 1.08)))
    robot_width, robot_depth, robot_height = robot_dims
    if robot_design == "sensor_tower":
        robot_dims = (round(robot_width * 0.96, 4), round(robot_depth * 0.96, 4), round(robot_height * 1.26, 4))
    elif robot_design == "stacker_bot":
        robot_dims = (round(robot_width * 0.94, 4), round(robot_depth * 0.98, 4), round(robot_height * 1.42, 4))
    robot_spec = _make_object_spec(
        object_id="warehouse_robot_reference",
        object_type="warehouse_robot",
        object_role="warehouse_reference_robot",
        xy=robot_xy,
        orientation_axis=str(orientation_axis),
        dimensions_xyz=robot_dims,
        dimension_scale=1.0,
        label=None,
    )
    robot_base_rgb = ROBOT_BASE_COLORS[int(rng.randrange(len(ROBOT_BASE_COLORS)))]
    robot_accent_rgb = ROBOT_ACCENT_COLORS[int(rng.randrange(len(ROBOT_ACCENT_COLORS)))]
    robot_spec.update(
        {
            "robot_design": str(robot_design),
            "robot_heading": str(robot_heading),
            "robot_base_rgb": [int(channel) for channel in robot_base_rgb],
            "robot_accent_rgb": [int(channel) for channel in robot_accent_rgb],
        }
    )

    required_slots = [
        ("first_path", robot_s + float(rng.uniform(1.42, 1.72)), float(rng.uniform(-0.08, 0.08))),
        ("later_path", robot_s + float(rng.uniform(2.78, 3.22)), float(rng.uniform(-0.10, 0.10))),
    ]
    distractor_slots = [
        ("side_close", robot_s + float(rng.uniform(1.02, 1.42)), float(rng.choice((-1.0, 1.0))) * float(rng.uniform(1.14, 1.42))),
        ("behind_near", robot_s - float(rng.uniform(1.10, 1.52)), float(rng.choice((-1.0, 1.0))) * float(rng.uniform(1.42, 1.92))),
        ("adjacent_aisle", robot_s + float(rng.uniform(1.54, 2.22)), float(rng.choice((-1.0, 1.0))) * float(rng.uniform(1.70, 2.08))),
        ("side_far", robot_s + float(rng.uniform(2.42, 3.06)), float(rng.choice((-1.0, 1.0))) * float(rng.uniform(1.12, 1.54))),
    ]
    rng.shuffle(distractor_slots)
    candidate_local_slots = [*required_slots, *distractor_slots[: max(0, int(candidate_count) - len(required_slots))]]
    rng.shuffle(candidate_local_slots)
    object_types = list(WAREHOUSE_SMALL_OBJECT_CANDIDATE_TYPES)
    path_object_types = [
        "crate_stack",
        "barrel",
        "box_stack",
        "tire_stack",
        "storage_bin",
        "rolling_bin",
        "wrapped_bundle",
        "traffic_cone",
    ]
    rng.shuffle(object_types)
    rng.shuffle(path_object_types)
    candidate_specs: List[Dict[str, Any]] = []
    for index, (slot_role, forward_s, lateral_l) in enumerate(candidate_local_slots[: int(candidate_count)]):
        object_type = str(
            path_object_types.pop()
            if slot_role in {"first_path", "later_path"} and path_object_types
            else rng.choice(object_types)
        )
        xy = _local_to_world(forward_s=float(forward_s), lateral_l=float(lateral_l), origin_xy=origin_xy, forward_xy=forward_xy)
        scale = float(rng.uniform(0.90, 1.12))
        dimensions = _dimensions_for_object(str(object_type), orientation_axis=str(orientation_axis), scale=float(scale))
        spec = _make_object_spec(
            object_id=f"candidate_slot_{slot_role}_{index}",
            object_type=str(object_type),
            object_role="warehouse_candidate",
            xy=xy,
            orientation_axis=str(orientation_axis),
            dimensions_xyz=dimensions,
            dimension_scale=float(scale),
            label="?",
        )
        forward_from_robot, lateral_from_robot = _world_to_robot_path(xy, robot_xy=robot_xy, forward_xy=forward_xy)
        in_corridor = float(forward_from_robot) >= MIN_FORWARD_DISTANCE and abs(float(lateral_from_robot)) <= PATH_CORRIDOR_HALF_WIDTH
        spec.update(
            {
                "slot_role": str(slot_role),
                "forward_distance_from_robot": round(float(forward_from_robot), 4),
                "lateral_offset_from_robot": round(float(lateral_from_robot), 4),
                "is_in_forward_path_corridor": bool(in_corridor),
            }
        )
        candidate_specs.append(spec)

    shelf_specs: List[Dict[str, Any]] = []
    shelf_slots = [
        (1.72, -2.82),
        (3.66, -2.70),
        (1.82, 2.82),
        (3.78, 2.70),
    ]
    for index, (forward_s, lateral_l) in enumerate(shelf_slots):
        xy = _local_to_world(
            forward_s=float(forward_s + rng.uniform(-0.12, 0.12)),
            lateral_l=float(lateral_l + rng.uniform(-0.16, 0.16)),
            origin_xy=origin_xy,
            forward_xy=forward_xy,
        )
        shelf_style = str(rng.choice(SUPPORTED_SHELF_RACK_STYLES))
        if shelf_style == "loaded_bins":
            level_fracs = [0.10, 0.34, 0.58, 0.84]
            load_count = int(rng.randint(4, 8))
            height_scale = float(rng.uniform(0.84, 1.08))
        elif shelf_style == "mixed_crates":
            level_fracs = [0.12, 0.48, 0.82]
            load_count = int(rng.randint(2, 5))
            height_scale = float(rng.uniform(0.74, 0.98))
        elif shelf_style == "tall_sparse":
            level_fracs = [0.08, 0.30, 0.56, 0.82]
            load_count = int(rng.randint(1, 3))
            height_scale = float(rng.uniform(1.04, 1.24))
        elif shelf_style == "heavy_low":
            level_fracs = [0.14, 0.62]
            load_count = int(rng.randint(1, 4))
            height_scale = float(rng.uniform(0.56, 0.76))
        else:
            level_fracs = [0.12, 0.54, 0.86] if rng.random() < 0.55 else [0.16, 0.76]
            load_count = int(rng.randint(0, 2))
            height_scale = float(rng.uniform(0.68, 1.04))
        scale = float(rng.uniform(1.00, 1.18))
        base_width, base_depth, base_height = _dimensions_for_object("shelf_rack", orientation_axis=str(orientation_axis), scale=float(scale))
        dimensions = (base_width, base_depth, round(float(base_height * height_scale), 4))
        load_slots: List[Dict[str, Any]] = []
        loadable_levels = level_fracs[:-1] if len(level_fracs) > 1 else level_fracs
        for load_index in range(load_count):
            level_frac = float(rng.choice(loadable_levels))
            load_slots.append(
                {
                    "x_frac": round(float(rng.uniform(-0.30, 0.30)), 4),
                    "y_frac": round(float(rng.uniform(-0.18, 0.18)), 4),
                    "z_frac": round(min(0.86, level_frac + float(rng.uniform(0.035, 0.060))), 4),
                    "w_frac": round(float(rng.uniform(0.12, 0.24)), 4),
                    "d_frac": round(float(rng.uniform(0.38, 0.66)), 4),
                    "h_frac": round(float(rng.uniform(0.10, 0.18)), 4),
                    "color_index": int(rng.randrange(len(SHELF_LOAD_COLORS))),
                }
            )
        shelf_spec = _make_object_spec(
            object_id=f"context_shelf_rack_{index}",
            object_type="shelf_rack",
            object_role="warehouse_context",
            xy=xy,
            orientation_axis=str(orientation_axis),
            dimensions_xyz=dimensions,
            dimension_scale=float(scale),
            label=None,
        )
        shelf_spec.update(
            {
                "shelf_style": str(shelf_style),
                "shelf_height_scale": round(float(height_scale), 4),
                "shelf_level_fracs": [round(float(value), 4) for value in level_fracs],
                "shelf_levels": int(len(level_fracs)),
                "shelf_beam_height": round(float(rng.uniform(0.075, 0.125)), 4),
                "shelf_post_width": round(float(rng.uniform(0.070, 0.115)), 4),
                "shelf_frame_rgb": [int(channel) for channel in SHELF_FRAME_COLORS[int(rng.randrange(len(SHELF_FRAME_COLORS)))]],
                "shelf_load_slots": list(load_slots),
            }
        )
        shelf_specs.append(shelf_spec)
    context_specs: List[Dict[str, Any]] = list(shelf_specs)
    optional_context_slots = [
        ("charging_dock", robot_s - 0.90, -1.70),
        ("conveyor", 2.96, 1.28),
        ("pallet", -2.58, 1.72),
        ("crate_stack", -2.72, -1.62),
        ("barrel", 3.34, -1.18),
        ("storage_bin", 3.24, 1.46),
        ("tool_cart", 0.18, 2.46),
        ("traffic_cone", 0.32, -2.46),
        ("pallet_load", 3.68, 1.88),
        ("workbench", -0.32, -2.60),
        ("rolling_bin", 1.64, 2.54),
        ("trash_can", -2.30, -2.28),
        ("warning_bollard", -0.08, 1.48),
        ("wrapped_bundle", 2.30, -2.54),
        ("fire_extinguisher", -1.84, 2.42),
        ("stacked_pipes", 3.54, -2.08),
    ]
    rng.shuffle(optional_context_slots)
    target_optional = max(0, int(context_object_count) - len(context_specs))
    for index, (object_type, forward_s, lateral_l) in enumerate(optional_context_slots[:target_optional]):
        xy = _local_to_world(
            forward_s=float(forward_s + rng.uniform(-0.12, 0.12)),
            lateral_l=float(lateral_l + rng.uniform(-0.10, 0.10)),
            origin_xy=origin_xy,
            forward_xy=forward_xy,
        )
        scale = float(rng.uniform(0.90, 1.12))
        dimensions = _dimensions_for_object(str(object_type), orientation_axis=str(orientation_axis), scale=float(scale))
        context_specs.append(
            _make_object_spec(
                object_id=f"context_{index}_{object_type}",
                object_type=str(object_type),
                object_role="warehouse_context",
                xy=xy,
                orientation_axis=str(orientation_axis),
                dimensions_xyz=dimensions,
                dimension_scale=float(scale),
                label=None,
            )
        )
    path_corridor_polygon = _make_path_polygon(robot_xy=robot_xy, forward_xy=forward_xy, start_s=0.20, end_s=3.50, half_width=PATH_CORRIDOR_HALF_WIDTH)
    main_aisle_polygon = _make_path_polygon(robot_xy=robot_xy, forward_xy=forward_xy, start_s=-1.06, end_s=4.30, half_width=1.28)
    left_shelf_zone = _make_path_polygon(robot_xy=robot_xy, forward_xy=forward_xy, start_s=1.04, end_s=4.72, half_width=0.42)
    right_shelf_zone = _make_path_polygon(robot_xy=robot_xy, forward_xy=forward_xy, start_s=1.04, end_s=4.72, half_width=0.42)
    fx, fy = forward_xy
    lx, ly = -fy, fx
    shifted_left = [(x + lx * 2.76, y + ly * 2.76, z) for x, y, z in left_shelf_zone]
    shifted_right = [(x - lx * 2.76, y - ly * 2.76, z) for x, y, z in right_shelf_zone]
    scene_geometry = {
        "origin_xy": [round(float(value), 4) for value in origin_xy],
        "robot_xy": [round(float(value), 4) for value in robot_xy],
        "forward_xy": [round(float(value), 4) for value in forward_xy],
        "path_corridor_polygon": [[round(float(value), 4) for value in point] for point in path_corridor_polygon],
        "main_aisle_polygon": [[round(float(value), 4) for value in point] for point in main_aisle_polygon],
        "shelf_zone_polygons": [
            [[round(float(value), 4) for value in point] for point in shifted_left],
            [[round(float(value), 4) for value in point] for point in shifted_right],
        ],
    }
    return [robot_spec], candidate_specs, context_specs, scene_geometry


__all__ = [
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_ROBOT_HEADINGS",
    "SUPPORTED_ROBOT_DESIGNS",
    "SUPPORTED_SHELF_RACK_STYLES",
    "WAREHOUSE_CAMERA_YAW_BANDS_DEGREES",
    "CONTEXT_OBJECT_TYPES",
    "OBJECT_NAMES",
    "OBJECT_COLORS",
    "ROBOT_BASE_COLORS",
    "ROBOT_ACCENT_COLORS",
    "SHELF_FRAME_COLORS",
    "SHELF_LOAD_COLORS",
    "PATH_CORRIDOR_HALF_WIDTH",
    "MIN_FORWARD_DISTANCE",
    "MIN_CANDIDATE_VISIBLE_PX",
    "MIN_CANDIDATE_CENTER_SEPARATION_PX",
    "MAX_CANDIDATE_BBOX_INTERSECTION_PX",
    "_WarehouseRenderParams",
    "_resolve_camera_yaw_band",
    "_resolve_render_params",
    "_heading_vector",
    "_heading_axis",
    "_local_to_world",
    "_world_to_robot_path",
    "_dimensions_for_object",
    "_make_object_spec",
    "_finalize_specs",
    "_bbox_area",
    "_scene_palette",
    "_projected_bbox",
    "_camera_from_dataset",
    "_frame_from_dataset",
    "_make_path_polygon",
    "_sample_reference_and_objects",
]
