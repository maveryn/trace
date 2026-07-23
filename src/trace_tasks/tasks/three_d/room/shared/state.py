"""Scene-local constants and geometry helpers for the 3D room scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ...shared.object_resources import (
    ROOM_EXTRA_WALL_TYPES,
    ROOM_FLOOR_DISTRACTOR_SPECS,
    ROOM_FLOOR_DISTRACTOR_TYPES,
    ROOM_FLOOR_PROP_SHAPES,
    ROOM_FLOOR_PROP_SPECS,
    ROOM_FRONT_FLOOR_PROP_SHAPES,
    ROOM_OBJECT_PROMPT_NAMES,
    ROOM_SURFACE_DISTRACTOR_SPECS,
    ROOM_SURFACE_DISTRACTOR_TYPES,
    ROOM_SURFACE_PROP_SHAPES_BY_SCENE,
    ROOM_SURFACE_PROP_TYPES,
    ROOM_WALL_BASE_DIMENSIONS,
)
from ...shared.camera_projection import (
    CameraSpec,
    project_screen,
    vec_cross,
    vec_norm,
    vec_sub,
)
from ...shared.object_scene import (
    make_object_spec,
    object_screen_bbox,
)
from ...shared.room_wall_rendering_geometry import (
    _add_vec,
    _projected_polygon_bbox,
    _wall_axes,
    _wall_rect_points,
)

SCENE_ID = "room"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("living_room", "office_room", "studio_room")
OBJECT_PROMPT_NAMES: Dict[str, Tuple[str, str]] = dict(ROOM_OBJECT_PROMPT_NAMES)
EXTRA_WALL_TYPES: Tuple[str, ...] = ROOM_EXTRA_WALL_TYPES
FLOOR_DISTRACTOR_TYPES: Tuple[str, ...] = ROOM_FLOOR_DISTRACTOR_TYPES
SURFACE_DISTRACTOR_TYPES: Tuple[str, ...] = ROOM_SURFACE_DISTRACTOR_TYPES
SURFACE_PROP_SHAPES_BY_SCENE: Dict[str, Tuple[str, ...]] = dict(ROOM_SURFACE_PROP_SHAPES_BY_SCENE)
SURFACE_PROP_TYPES: Tuple[str, ...] = ROOM_SURFACE_PROP_TYPES
FLOOR_PROP_SHAPES: Tuple[str, ...] = ROOM_FLOOR_PROP_SHAPES
PICTURE_SCENERY_VARIANTS: Tuple[str, ...] = ("mountains", "lake", "forest", "sunset", "city")
WALL_X = 3.35
WALL_BACK_Y = 2.85
ROOM_FRONT_Y = -2.95
ROOM_HEIGHT = 3.05
ROOM_RENDER_FRONT_MIN_EXTENSION = 1.8
ROOM_RENDER_FRONT_MAX_EXTENSION = 5.2
ROOM_RENDER_SIDE_WALL_MAX_EXTENSION = 2.15
ROOM_CAMERA_PITCH_DEGREES: Tuple[float, float] = (9.0, 18.0)
ROOM_CAMERA_DISTANCE_RANGE: Tuple[float, float] = (6.8, 8.2)
ROOM_CAMERA_TARGET_Z = 1.05
SIDE_WALL_OBJECT_HPOS_MAX = 1.15
FRONT_FLOOR_PROP_SLOTS: Tuple[Tuple[float, float], ...] = (
    (-2.36, -2.56),
    (-0.86, -2.66),
    (0.82, -2.64),
    (2.28, -2.46),
)
FRONT_FLOOR_PROP_SHAPES: Tuple[str, ...] = ROOM_FRONT_FLOOR_PROP_SHAPES
ROOM_VIEW_YAW_BANDS: Dict[str, Tuple[float, float]] = {
    "living_room": (-24.0, -10.0),
    "office_room": (10.0, 24.0),
    "studio_room": (-8.0, 8.0),
}


def _object_name(object_type: str) -> str:
    return str(OBJECT_PROMPT_NAMES.get(str(object_type), (str(object_type).replace("_", " "), ""))[0])


def _object_plural(object_type: str) -> str:
    singular, plural = OBJECT_PROMPT_NAMES.get(str(object_type), (str(object_type).replace("_", " "), ""))
    return str(plural or f"{singular}s")

def _sample_room_camera(
    rng,
    *,
    scene_variant: str,
    yaw_band_degrees: Tuple[float, float] | None = None,
) -> CameraSpec:
    yaw_lower, yaw_upper = (
        tuple(float(value) for value in yaw_band_degrees)
        if yaw_band_degrees is not None
        else ROOM_VIEW_YAW_BANDS.get(str(scene_variant), ROOM_VIEW_YAW_BANDS["studio_room"])
    )
    yaw_degrees = float(rng.uniform(float(yaw_lower), float(yaw_upper)))
    pitch_degrees = float(rng.uniform(float(ROOM_CAMERA_PITCH_DEGREES[0]), float(ROOM_CAMERA_PITCH_DEGREES[1])))
    distance = float(rng.uniform(float(ROOM_CAMERA_DISTANCE_RANGE[0]), float(ROOM_CAMERA_DISTANCE_RANGE[1])))
    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    target = (0.0, 0.0, float(ROOM_CAMERA_TARGET_Z))
    camera_position = (
        float(distance * math.cos(pitch) * math.sin(yaw)),
        float(-distance * math.cos(pitch) * math.cos(yaw)),
        float(target[2] + distance * math.sin(pitch)),
    )
    forward = vec_norm(vec_sub(target, camera_position))
    world_up = (0.0, 0.0, 1.0)
    right = vec_norm(vec_cross(forward, world_up))
    up = vec_norm(vec_cross(right, forward))
    return CameraSpec(
        camera_position=tuple(camera_position),
        target=tuple(target),
        right=tuple(right),
        up=tuple(up),
        forward=tuple(forward),
        yaw_degrees=float(yaw_degrees),
        pitch_degrees=float(pitch_degrees),
        distance=float(distance),
    )


def _wall_center(wall: str, hpos: float, z: float) -> Tuple[float, float, float]:
    if str(wall) == "back":
        return (float(hpos), float(WALL_BACK_Y), float(z))
    if str(wall) == "left":
        return (-float(WALL_X), float(hpos), float(z))
    return (float(WALL_X), float(hpos), float(z))


def _wall_reference_points(spec: Mapping[str, Any]) -> List[Tuple[float, float, float]]:
    points = list(_wall_rect_points(spec, normal_offset=0.04)) + [tuple(float(value) for value in spec["world_xyz"])]
    object_type = str(spec.get("object_type", ""))
    if object_type not in {"tv", "wall_shelf"}:
        return points

    horizontal_axis, vertical_axis, normal_axis = _wall_axes(str(spec["wall"]))
    center = tuple(float(value) for value in spec["world_xyz"])

    def physical_point(h: float, v: float, normal_offset: float) -> Tuple[float, float, float]:
        return _add_vec(
            center,
            horizontal_axis,
            vertical_axis,
            normal_axis,
            hw=float(h),
            hh=float(v),
            normal_offset=float(normal_offset),
        )

    half_w = float(spec["wall_width"]) * 0.5
    half_h = float(spec["wall_height"]) * 0.5
    normal_offsets = (0.04, 0.18) if object_type == "tv" else (0.04, 0.43)
    points.extend(
        physical_point(h, v, normal_offset)
        for h in (-half_w, half_w)
        for v in (-half_h, half_h)
        for normal_offset in normal_offsets
    )
    if object_type == "wall_shelf":
        for hpos in (-half_w * 0.36, half_w * 0.36):
            points.extend(
                [
                    physical_point(hpos, -half_h, 0.35),
                    physical_point(hpos, -half_h - 0.30, 0.055),
                ]
            )
    return points


def _floor_spec(
    *,
    object_id: str,
    object_type: str,
    prompt_name: str,
    xy: Tuple[float, float],
    dimensions_xyz: Tuple[float, float, float],
    color_role: str = "furniture",
    base_z: float = 0.0,
    mounting: str = "floor",
    support_object_id: str | None = None,
    support_surface_type: str | None = None,
    draw_order: int = 10,
) -> Dict[str, Any]:
    """Create a floor-mounted room object spec with normalized render metadata.

    Every returned spec uses the same footprint/support fields so layout,
    rendering, and annotation projection can treat room objects uniformly.
    """

    spec = make_object_spec(
        object_id=str(object_id),
        shape_type="rectangular_prism",
        object_role="context",
        xy=(float(xy[0]), float(xy[1])),
        dimensions_xyz=tuple(float(value) for value in dimensions_xyz),
        dimension_scale=1.0,
    )
    height = float(dimensions_xyz[2])
    spec.update(
        {
            "world_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(base_z) + height * 0.5, 4)],
            "base_xyz": [round(float(xy[0]), 4), round(float(xy[1]), 4), round(float(base_z), 4)],
            "object_type": str(object_type),
            "object_name": str(prompt_name),
            "prompt_name": str(prompt_name),
            "object_role": "floor_object",
            "is_wall_mounted": False,
            "mounting": str(mounting),
            "counts_for_query": False,
            "color_role": str(color_role),
            "draw_order": int(draw_order),
        }
    )
    if support_object_id is not None:
        spec["support_object_id"] = str(support_object_id)
    if support_surface_type is not None:
        spec["support_surface_type"] = str(support_surface_type)
    return spec


def _wall_spec(
    *,
    object_id: str,
    object_type: str,
    wall: str,
    hpos: float,
    z: float,
    width: float,
    height: float,
    counts_for_query: bool,
) -> Dict[str, Any]:
    center = _wall_center(str(wall), float(hpos), float(z))
    return {
        "object_id": str(object_id),
        "object_type": str(object_type),
        "shape_type": str(object_type),
        "object_name": _object_name(str(object_type)),
        "prompt_name": _object_name(str(object_type)),
        "object_role": "wall_object",
        "world_xyz": [round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4)],
        "base_xyz": [round(float(center[0]), 4), round(float(center[1]), 4), round(max(0.0, float(center[2]) - float(height) * 0.5), 4)],
        "dimensions_xyz": [round(float(width), 4), 0.06, round(float(height), 4)],
        "wall": str(wall),
        "wall_hpos": round(float(hpos), 4),
        "wall_width": round(float(width), 4),
        "wall_height": round(float(height), 4),
        "is_wall_mounted": True,
        "mounting": "wall_mounted",
        "counts_for_query": bool(counts_for_query),
        "nameable_for_prompt": True,
    }


def _with_picture_scenery(spec: Mapping[str, Any], rng) -> Dict[str, Any]:
    updated = dict(spec)
    if str(updated.get("object_type")) == "picture_frame":
        scenery_variant = str(rng.choice(PICTURE_SCENERY_VARIANTS))
        updated["scenery_variant"] = scenery_variant
        updated["picture_content"] = f"simple {scenery_variant} painting"
    return updated


def _top_z(spec: Mapping[str, Any]) -> float:
    base = spec.get("base_xyz", (0.0, 0.0, 0.0))
    base_z = float(base[2]) if isinstance(base, Sequence) and len(base) >= 3 else 0.0
    return round(float(base_z) + float(spec["dimensions_xyz"][2]), 4)


def _support_can_hold(object_type: str, support_spec: Mapping[str, Any]) -> bool:
    support_type = str(support_spec.get("object_type", ""))
    support_width = float(support_spec["dimensions_xyz"][0])
    if str(object_type) == "tv":
        return support_type in {"media_console", "desk", "bed"} and support_width >= 0.90
    if str(object_type) in {"clock", "picture_frame"}:
        return support_type in set(SURFACE_PROP_TYPES)
    return False


def _surface_xy(
    rng,
    support_spec: Mapping[str, Any],
    dimensions_xyz: Tuple[float, float, float],
) -> Tuple[float, float]:
    support_width, support_depth, _support_height = (float(value) for value in support_spec["dimensions_xyz"])
    object_width, object_depth, _object_height = (float(value) for value in dimensions_xyz)
    max_dx = max(0.0, support_width * 0.5 - object_width * 0.5 - 0.06)
    max_dy = max(0.0, support_depth * 0.5 - object_depth * 0.5 - 0.06)
    support_x = float(support_spec["world_xyz"][0])
    support_y = float(support_spec["world_xyz"][1])
    return (
        float(support_x + rng.uniform(-max_dx, max_dx)),
        float(support_y + rng.uniform(-max_dy, max_dy)),
    )


def _surface_distractor_for_type(
    *,
    rng,
    object_id: str,
    object_type: str,
    support_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    spec_data = ROOM_SURFACE_DISTRACTOR_SPECS.get(str(object_type), ROOM_SURFACE_DISTRACTOR_SPECS["picture_frame"])
    dimensions = tuple(float(value) for value in spec_data["dimensions_xyz"])
    spec = _floor_spec(
        object_id=str(object_id),
        object_type=str(object_type),
        prompt_name=str(spec_data["prompt_name"]),
        xy=_surface_xy(rng, support_spec, dimensions),
        dimensions_xyz=tuple(float(value) for value in dimensions),
        color_role=str(spec_data["color_role"]),
        base_z=_top_z(support_spec),
        mounting="on_furniture",
        support_object_id=str(support_spec["object_id"]),
        support_surface_type=str(support_spec.get("object_type", "")),
        draw_order=24,
    )
    return _with_picture_scenery(spec, rng)


def _floor_distractor_for_type(
    *,
    rng,
    object_id: str,
    object_type: str,
    xy: Tuple[float, float],
) -> Dict[str, Any]:
    spec_data = ROOM_FLOOR_DISTRACTOR_SPECS.get(str(object_type), ROOM_FLOOR_DISTRACTOR_SPECS["wall_shelf"])
    spec = _floor_spec(
        object_id=object_id,
        object_type=str(object_type) if str(object_type) in ROOM_FLOOR_DISTRACTOR_SPECS else "wall_shelf",
        prompt_name=str(spec_data["prompt_name"]),
        xy=xy,
        dimensions_xyz=tuple(float(value) for value in spec_data["dimensions_xyz"]),
        color_role=str(spec_data["color_role"]),
        base_z=0.0,
    )
    return _with_picture_scenery(spec, rng)


def _make_floor_prop(
    *,
    rng,
    object_id: str,
    prop_shape: str,
    xy: Tuple[float, float],
) -> Dict[str, Any]:
    spec_data = ROOM_FLOOR_PROP_SPECS.get(str(prop_shape), ROOM_FLOOR_PROP_SPECS["box"])
    spec = _floor_spec(
        object_id=object_id,
        object_type=str(spec_data["object_type"]),
        prompt_name=str(spec_data["prompt_name"]),
        xy=xy,
        dimensions_xyz=tuple(float(value) for value in spec_data["dimensions_xyz"]),
        color_role=str(spec_data["color_role"]),
    )
    if "shape_type" in spec_data:
        spec["shape_type"] = str(spec_data["shape_type"])
    return spec


def _finalize_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> List[Dict[str, Any]]:
    finalized_specs: List[Dict[str, Any]] = []
    for spec in specs:
        screen = project_screen(spec["world_xyz"], camera, frame)
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


def _slot_is_compatible(slot: Tuple[str, float, float], *, object_type: str) -> bool:
    wall, _hpos, z = slot
    if str(object_type) == "wall_shelf" and str(wall) != "back":
        return False
    if str(object_type) in {"tv", "mirror"} and float(z) < 1.15:
        return False
    return True


def _wall_dimensions_for_type(object_type: str, rng) -> Tuple[float, float]:
    scale = float(rng.uniform(0.88, 1.12))
    base = ROOM_WALL_BASE_DIMENSIONS.get(str(object_type), (0.50, 0.50))
    return round(float(base[0]) * scale, 4), round(float(base[1]) * scale, 4)

def _room_object_bbox(spec: Mapping[str, Any], camera, frame) -> List[float]:
    if str(spec.get("object_role")) == "wall_object":
        return _projected_polygon_bbox(_wall_reference_points(spec), camera, frame, pad_px=8.0)
    return object_screen_bbox(spec, camera, frame, pad_px=8.0)


def _wall_object_visible_bbox(spec: Mapping[str, Any], camera, frame) -> List[float]:
    return _projected_polygon_bbox(_wall_reference_points(spec), camera, frame, pad_px=0.0)


def _bbox_pixel_width_height(bbox: Sequence[float]) -> Tuple[float, float]:
    return float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1])


def _wall_object_visible_size_ok(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    min_width_px: float,
    min_height_px: float,
) -> bool:
    for spec in specs:
        bbox = _wall_object_visible_bbox(spec, camera, frame)
        width, height = _bbox_pixel_width_height(bbox)
        if width < float(min_width_px) or height < float(min_height_px):
            return False
    return True
