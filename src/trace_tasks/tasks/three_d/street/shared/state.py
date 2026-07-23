"""Shared street-intersection scene specs, sampling helpers, and geometry utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.color_distance import coerce_rgb as _rgb
from ....shared.config_defaults import group_default
from ...shared.task_support import float_value as _float_value
from ...shared.task_support import int_value as _int_value
from ...shared.canvas import resolve_three_d_canvas_spec
from ...shared.visual_styles import resolve_three_d_surface_tone
from ...shared.object_resources import (
    BUILDING_STYLE_DIMENSION_FACTORS,
    BUILDING_STYLE_POOLS,
    BUILDING_STYLES,
)
from ...shared.camera_projection import (
    canvas_floor_polygon_xy as _canvas_floor_polygon_xy,
    project_screen as _project_screen,
)
from ...shared.object_scene import (
    _bbox_intersection_area,
    _object_reference_points,
    _object_screen_bbox,
)
from ...shared.street_object_rendering_common import (
    PEDESTRIAN_OBJECT_TYPES,
    _street_object_name,
    _base_street_object_dimensions,
    _fixed_building_style_for_street_object,
    _apply_street_building_style,
    _dimensions_for_orientation,
    _orientation_axis_for_xy,
    _missing_arm_for_layout,
    _arm_is_present,
    _stable_palette_index,
)


SCENE_ID = "street"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "downtown_intersection",
    "neighborhood_intersection",
    "transit_intersection",
)
SUPPORTED_INTERSECTION_LAYOUTS: Tuple[str, ...] = (
    "four_way",
    "t_missing_north",
    "t_missing_south",
    "t_missing_east",
    "t_missing_west",
)
STREET_CAMERA_YAW_BANDS_DEGREES: Tuple[Tuple[float, float], ...] = (
    (-54.0, -28.0),
    (28.0, 54.0),
    (-146.0, -116.0),
    (116.0, 146.0),
)
MIN_CANDIDATE_VISIBLE_PX = 20.0
MIN_CANDIDATE_CENTER_SEPARATION_PX = 42.0
MAX_CANDIDATE_BBOX_INTERSECTION_PX = 6500.0
INTERSECTION_CENTER_XY: Tuple[float, float] = (0.0, 0.0)
DEFAULT_INTERSECTION_CENTER_JITTER_X = 0.62
DEFAULT_INTERSECTION_CENTER_JITTER_Y = 0.54
STREET_FULL_BLEED_FALLBACK_EXTENT_MULTIPLIER = 3.2


@dataclass(frozen=True)
class _StreetRenderParams:
    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    room_extent: float
    street_extent: float
    road_half_width: float
    marker_radius_px: int
    label_font_size_px: int
    line_width_px: int
    sidewalk_rgb: Tuple[int, int, int]
    asphalt_rgb: Tuple[int, int, int]
    road_mark_rgb: Tuple[int, int, int]
    crosswalk_rgb: Tuple[int, int, int]
    curb_rgb: Tuple[int, int, int]
    text_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    background_tone_id: str = "custom"
    background_tone_rgb: Tuple[int, int, int] = (214, 222, 218)
    surface_accent_rgb: Tuple[int, int, int] = (207, 218, 207)
    canvas_preset: str = "explicit"
    canvas_policy: str = "explicit_dimensions"


def _min_pairwise(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 999.0
    return min(
        abs(float(a) - float(b))
        for index, a in enumerate(values)
        for b in values[index + 1 :]
    )






def _sample_intersection_center(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_params: _StreetRenderParams,
) -> Tuple[float, float]:
    explicit = params.get("intersection_center_xy")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)) and len(explicit) >= 2:
        return (
            round(max(-1.0, min(1.0, float(explicit[0]))), 4),
            round(max(-1.0, min(1.0, float(explicit[1]))), 4),
        )
    max_x = min(
        float(DEFAULT_INTERSECTION_CENTER_JITTER_X),
        float(group_default(gen_defaults, "intersection_center_jitter_x", DEFAULT_INTERSECTION_CENTER_JITTER_X)),
        max(0.0, float(render_params.street_extent) - float(render_params.road_half_width) - 2.75),
    )
    max_y = min(
        float(DEFAULT_INTERSECTION_CENTER_JITTER_Y),
        float(group_default(gen_defaults, "intersection_center_jitter_y", DEFAULT_INTERSECTION_CENTER_JITTER_Y)),
        max(0.0, float(render_params.street_extent) - float(render_params.road_half_width) - 2.85),
    )
    return (
        round(float(rng.uniform(-max_x, max_x)), 4),
        round(float(rng.uniform(-max_y, max_y)), 4),
    )


def _resolve_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int = 0,
    namespace: str = "three_d.street.canvas",
) -> _StreetRenderParams:
    """Resolve deterministic canvas and street render parameters for one sample."""

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
    street_extent = _float_value(merged, "street_extent", 4.45)
    tone = resolve_three_d_surface_tone(
        params=params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.surface_tone",
    )
    return _StreetRenderParams(
        canvas_width=int(canvas.canvas_width),
        canvas_height=int(canvas.canvas_height),
        canvas_preset=str(canvas.preset_id),
        canvas_policy=str(canvas.policy),
        scene_margin_left_px=_int_value(merged, "scene_margin_left_px", 48),
        scene_margin_right_px=_int_value(merged, "scene_margin_right_px", 48),
        scene_margin_top_px=_int_value(merged, "scene_margin_top_px", 42),
        scene_margin_bottom_px=_int_value(merged, "scene_margin_bottom_px", 52),
        room_extent=float(street_extent),
        street_extent=float(street_extent),
        road_half_width=_float_value(merged, "road_half_width", 1.02),
        marker_radius_px=_int_value(merged, "marker_radius_px", 20),
        label_font_size_px=_int_value(merged, "label_font_size_px", 25),
        line_width_px=_int_value(merged, "line_width_px", 2),
        sidewalk_rgb=_rgb(params.get("sidewalk_rgb", tone.floor_rgb), tone.floor_rgb),
        asphalt_rgb=_rgb(merged.get("asphalt_rgb", (86, 93, 101)), (86, 93, 101)),
        road_mark_rgb=_rgb(merged.get("road_mark_rgb", (236, 210, 86)), (236, 210, 86)),
        crosswalk_rgb=_rgb(merged.get("crosswalk_rgb", (240, 243, 238)), (240, 243, 238)),
        curb_rgb=_rgb(params.get("curb_rgb", tone.edge_rgb), tone.edge_rgb),
        text_rgb=_rgb(params.get("text_rgb", tone.text_rgb), tone.text_rgb),
        text_stroke_rgb=_rgb(params.get("text_stroke_rgb", tone.text_stroke_rgb), tone.text_stroke_rgb),
        background_tone_id=str(tone.tone_id),
        background_tone_rgb=tuple(int(value) for value in tone.floor_rgb),
        surface_accent_rgb=tuple(int(value) for value in tone.surface_accent_rgb),
    )


def _slot_allowed_for_layout(
    relative_xy: Sequence[float],
    *,
    intersection_layout: str,
    road_half_width: float,
) -> bool:
    missing = _missing_arm_for_layout(str(intersection_layout))
    if missing is None:
        return True
    x, y = float(relative_xy[0]), float(relative_xy[1])
    corridor = float(road_half_width) * 1.04
    offset = float(road_half_width) * 1.20
    if missing == "north":
        return not (abs(x) < corridor and y > offset)
    if missing == "south":
        return not (abs(x) < corridor and y < -offset)
    if missing == "east":
        return not (x > offset and abs(y) < corridor)
    if missing == "west":
        return not (x < -offset and abs(y) < corridor)
    return True


def _translate_scene_xy(
    relative_xy: Sequence[float],
    *,
    center_xy: Sequence[float],
    extent: float,
    margin: float,
    rng=None,
    jitter: float = 0.0,
) -> Tuple[float, float]:
    x = float(center_xy[0]) + float(relative_xy[0])
    y = float(center_xy[1]) + float(relative_xy[1])
    if rng is not None and float(jitter) > 0.0:
        x += float(rng.uniform(-float(jitter), float(jitter)))
        y += float(rng.uniform(-float(jitter), float(jitter)))
    limit = max(0.1, float(extent) - float(margin))
    return (
        round(max(-limit, min(limit, x)), 4),
        round(max(-limit, min(limit, y)), 4),
    )


def _make_street_object_spec(
    *,
    object_id: str,
    object_type: str,
    object_role: str,
    xy: Tuple[float, float],
    intersection_center_xy: Tuple[float, float],
    orientation_axis: str,
    dimensions_xyz: Tuple[float, float, float],
    label: str | None,
    dimension_scale: float,
) -> Dict[str, Any]:
    """Create canonical object metadata shared by street task samplers."""

    width, depth, height = (float(value) for value in dimensions_xyz)
    x, y = float(xy[0]), float(xy[1])
    center_x, center_y = float(intersection_center_xy[0]), float(intersection_center_xy[1])
    ground_distance = math.hypot(x - center_x, y - center_y)
    footprint_radius = 0.5 * math.sqrt(float(width) * float(width) + float(depth) * float(depth))
    spec: Dict[str, Any] = {
        "object_id": str(object_id),
        "object_type": str(object_type),
        "object_name": _street_object_name(str(object_type)),
        "prompt_name": _street_object_name(str(object_type)),
        "object_role": str(object_role),
        "orientation_axis": str(orientation_axis),
        "is_answer_candidate": bool(label),
        "dimension_scale": round(float(dimension_scale), 4),
        "world_xyz": [round(x, 4), round(y, 4), round(float(height * 0.5), 4)],
        "base_xyz": [round(x, 4), round(y, 4), 0.0],
        "dimensions_xyz": [round(width, 4), round(depth, 4), round(height, 4)],
        "footprint_radius": round(float(footprint_radius), 4),
        "intersection_center_xy": [round(float(center_x), 4), round(float(center_y), 4)],
        "ground_distance_to_intersection": round(float(ground_distance), 4),
    }
    if label is not None:
        spec.update(
            {
                "point_id": f"street_object_{label}",
                "point_label": str(label),
                "object_label": str(label),
            }
        )
    if str(object_type) in PEDESTRIAN_OBJECT_TYPES:
        if str(object_type) == "female_pedestrian":
            gender_id = "female"
        elif str(object_type) == "male_pedestrian":
            gender_id = "male"
        else:
            gender_id = "female" if _stable_palette_index(str(object_id), 2) == 1 else "male"
        spec.update(
            {
                "pedestrian_gender_id": str(gender_id),
                "pedestrian_appearance_id": f"pedestrian_{gender_id}",
            }
        )
    return spec


def _sample_context_specs(
    *,
    rng,
    scene_variant: str,
    context_object_count: int,
    intersection_center_xy: Tuple[float, float],
    intersection_layout: str,
    road_half_width: float,
    street_extent: float,
) -> List[Dict[str, Any]]:
    """Sample background buildings and street furniture away from answer grammar."""

    building_height_ranges = {
        "downtown_intersection": (1.12, 1.78),
        "neighborhood_intersection": (0.76, 1.22),
        "transit_intersection": (0.94, 1.52),
    }
    building_slots = [
        (-3.36, -3.20),
        (3.28, -3.26),
        (-3.30, 3.24),
        (3.36, 3.18),
        (-3.66, -1.55),
        (-3.66, 1.55),
        (3.66, -1.55),
        (3.66, 1.55),
        (-3.72, -2.58),
        (3.72, -2.58),
        (-3.72, 2.58),
        (3.72, 2.58),
        (-1.55, -3.66),
        (1.55, -3.66),
        (-1.55, 3.66),
        (1.55, 3.66),
        (-2.58, -3.72),
        (2.58, -3.72),
        (-2.58, 3.72),
        (2.58, 3.72),
    ]
    missing_arm = _missing_arm_for_layout(str(intersection_layout))
    if missing_arm == "north":
        building_slots.extend([(-0.58, 3.24), (0.64, 3.34)])
    elif missing_arm == "south":
        building_slots.extend([(-0.58, -3.24), (0.64, -3.34)])
    elif missing_arm == "east":
        building_slots.extend([(3.28, -0.62), (3.38, 0.62)])
    elif missing_arm == "west":
        building_slots.extend([(-3.28, -0.62), (-3.38, 0.62)])
    rng.shuffle(building_slots)
    context_specs: List[Dict[str, Any]] = []
    min_h, max_h = building_height_ranges.get(str(scene_variant), (0.92, 1.42))
    building_count = min(max(5, int(round(rng.uniform(5.2, 6.8)))), int(context_object_count), len(building_slots))
    building_styles = list(BUILDING_STYLE_POOLS.get(str(scene_variant), BUILDING_STYLES))
    rng.shuffle(building_styles)
    for index, relative_xy in enumerate(building_slots[:building_count]):
        xy = _translate_scene_xy(
            relative_xy,
            center_xy=intersection_center_xy,
            extent=float(street_extent),
            margin=0.62,
            rng=rng,
            jitter=0.18,
        )
        building_style = str(rng.choice(building_styles))
        width_factor, depth_factor, height_factor = (
            float(value)
            for value in BUILDING_STYLE_DIMENSION_FACTORS.get(str(building_style), (1.0, 1.0, 1.0))
        )
        scale = float(rng.uniform(0.92, 1.16))
        base_w, base_d, _base_h = _base_street_object_dimensions("building")
        height = float(rng.uniform(float(min_h), float(max_h))) * float(height_factor)
        dimensions = (
            round(float(base_w * scale * width_factor), 4),
            round(float(base_d * scale * depth_factor), 4),
            round(float(height), 4),
        )
        spec = _make_street_object_spec(
            object_id=f"context_building_{index}",
            object_type="building",
            object_role="street_context",
            xy=(float(xy[0]), float(xy[1])),
            intersection_center_xy=tuple(float(value) for value in intersection_center_xy),
            orientation_axis="x",
            dimensions_xyz=dimensions,
            label=None,
            dimension_scale=float(scale),
        )
        context_specs.append(_apply_street_building_style(spec, style=str(building_style)))
    corner = float(road_half_width) + 0.34
    optional_context = [
        ("tree", (-2.34, -2.74), "x", 0.20),
        ("tree", (2.36, 2.74), "x", 0.20),
        ("tree", (-2.58, 2.42), "x", 0.20),
        ("tree", (2.72, -2.38), "x", 0.20),
        ("tree", (-3.28, 0.92), "x", 0.18),
        ("tree", (3.24, -0.92), "x", 0.18),
        ("shrub", (-2.92, -2.12), "x", 0.18),
        ("shrub", (2.92, 2.12), "x", 0.18),
        ("shrub", (-2.96, 2.02), "x", 0.18),
        ("shrub", (2.96, -2.02), "x", 0.18),
        ("shrub", (-3.40, -0.82), "x", 0.16),
        ("shrub", (3.40, 0.82), "x", 0.16),
        ("traffic_light", (-corner, -corner), "x", 0.08),
        ("traffic_light", (corner, corner), "x", 0.08),
        ("traffic_light", (-corner, corner), "x", 0.08),
        ("street_sign", (corner + 0.20, -corner), "y", 0.11),
        ("street_sign", (-corner - 0.20, corner), "y", 0.11),
        ("street_sign", (corner + 0.18, corner + 0.10), "y", 0.11),
        ("bench", (-2.78, 1.44), "x", 0.18),
        ("bench", (2.78, -1.44), "x", 0.18),
        ("bench", (2.58, 1.70), "x", 0.18),
        ("store", (-3.34, 1.56), "x", 0.12),
        ("store", (3.34, -1.56), "x", 0.12),
        ("office_building", (-3.58, -1.70), "x", 0.12),
        ("office_building", (3.58, 1.70), "x", 0.12),
    ]
    if str(scene_variant) == "transit_intersection":
        optional_context.append(("street_sign", (-2.65, -1.36), "x", 0.12))
    optional_context = [
        item for item in optional_context
        if _slot_allowed_for_layout(item[1], intersection_layout=str(intersection_layout), road_half_width=float(road_half_width))
        or str(item[0]) not in {"traffic_light", "street_sign"}
    ]
    rng.shuffle(optional_context)
    target_optional = max(0, int(context_object_count) - len(context_specs))
    for index, (object_type, relative_xy, orientation_axis, jitter) in enumerate(optional_context[:target_optional]):
        xy = _translate_scene_xy(
            relative_xy,
            center_xy=intersection_center_xy,
            extent=float(street_extent),
            margin=0.48,
            rng=rng,
            jitter=float(jitter),
        )
        scale = float(rng.uniform(0.92, 1.14))
        dimensions = _dimensions_for_orientation(
            str(object_type),
            orientation_axis=str(orientation_axis),
            scale=float(scale),
        )
        spec = _make_street_object_spec(
            object_id=f"context_{index}_{object_type}",
            object_type=str(object_type),
            object_role="street_context",
            xy=(float(xy[0]), float(xy[1])),
            intersection_center_xy=tuple(float(value) for value in intersection_center_xy),
            orientation_axis=str(orientation_axis),
            dimensions_xyz=dimensions,
            label=None,
            dimension_scale=float(scale),
        )
        context_specs.append(_apply_street_building_style(spec))
    return list(context_specs)


def _finalize_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> List[Dict[str, Any]]:
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


def _candidate_screen_separation_ok(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    render_params: _StreetRenderParams,
) -> bool:
    bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=16.0) for spec in specs]
    centers = [(float(spec["screen_xy"][0]), float(spec["screen_xy"][1])) for spec in specs]
    for bbox in bboxes:
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if width < MIN_CANDIDATE_VISIBLE_PX or height < MIN_CANDIDATE_VISIBLE_PX:
            return False
        if (
            float(bbox[0]) < -24.0
            or float(bbox[1]) < -24.0
            or float(bbox[2]) > float(render_params.canvas_width + 24)
            or float(bbox[3]) > float(render_params.canvas_height + 24)
        ):
            return False
    for index, center in enumerate(centers):
        for other_index in range(index + 1, len(centers)):
            other = centers[other_index]
            if math.hypot(center[0] - other[0], center[1] - other[1]) < MIN_CANDIDATE_CENTER_SEPARATION_PX:
                return False
            if _bbox_intersection_area(bboxes[index], bboxes[other_index]) > MAX_CANDIDATE_BBOX_INTERSECTION_PX:
                return False
    return True


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _candidate_context_visibility_ok(
    candidate_specs: Sequence[Mapping[str, Any]],
    context_specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> bool:
    candidate_bboxes = {
        str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=22.0)
        for spec in candidate_specs
    }
    context_bboxes = {
        str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=10.0)
        for spec in context_specs
    }
    for candidate in candidate_specs:
        candidate_bbox = candidate_bboxes[str(candidate["object_id"])]
        candidate_area = max(1.0, _bbox_area(candidate_bbox))
        for context in context_specs:
            if float(context["camera_distance"]) >= float(candidate["camera_distance"]) - 0.05:
                continue
            context_bbox = context_bboxes[str(context["object_id"])]
            overlap = _bbox_intersection_area(candidate_bbox, context_bbox)
            if overlap > 900.0 and overlap / candidate_area > 0.10:
                return False
    return True


def _reference_visibility_ok(
    reference_spec: Mapping[str, Any],
    candidate_specs: Sequence[Mapping[str, Any]],
    context_specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    render_params: _StreetRenderParams,
    min_center_separation_px: float,
    max_reference_candidate_bbox_intersection_px: float,
    min_visible_px: float = MIN_CANDIDATE_VISIBLE_PX,
) -> bool:
    """Validate that one marked reference object remains readable and separated."""

    reference_bbox = _object_screen_bbox(reference_spec, camera, frame, pad_px=16.0)
    width = float(reference_bbox[2]) - float(reference_bbox[0])
    height = float(reference_bbox[3]) - float(reference_bbox[1])
    if width < float(min_visible_px) or height < float(min_visible_px):
        return False
    if (
        float(reference_bbox[0]) < -24.0
        or float(reference_bbox[1]) < -24.0
        or float(reference_bbox[2]) > float(render_params.canvas_width + 24)
        or float(reference_bbox[3]) > float(render_params.canvas_height + 24)
    ):
        return False
    ref_center = (float(reference_spec["screen_xy"][0]), float(reference_spec["screen_xy"][1]))
    for candidate in candidate_specs:
        cand_center = (float(candidate["screen_xy"][0]), float(candidate["screen_xy"][1]))
        if math.hypot(ref_center[0] - cand_center[0], ref_center[1] - cand_center[1]) < float(min_center_separation_px):
            return False
        candidate_bbox = _object_screen_bbox(candidate, camera, frame, pad_px=16.0)
        if _bbox_intersection_area(reference_bbox, candidate_bbox) > float(max_reference_candidate_bbox_intersection_px):
            return False
    for context in context_specs:
        if float(context["camera_distance"]) >= float(reference_spec["camera_distance"]) - 0.05:
            continue
        context_bbox = _object_screen_bbox(context, camera, frame, pad_px=10.0)
        overlap = _bbox_intersection_area(reference_bbox, context_bbox)
        if overlap > 900.0 and overlap / max(1.0, (width * height)) > 0.10:
            return False
    return True


def _canvas_floor_polygon_available(
    *,
    camera,
    frame,
    render_params: _StreetRenderParams,
) -> bool:
    return len(_canvas_floor_polygon_xy(camera=camera, frame=frame, render_params=render_params)) >= 3


def _camera_from_dataset(dataset: Mapping[str, Any]):
    camera = type("CameraTuple", (), {})()
    raw = dataset["camera"]
    camera.camera_position = tuple(float(value) for value in raw["camera_position"])
    camera.target = tuple(float(value) for value in raw["target"])
    camera.right = tuple(float(value) for value in raw["right"])
    camera.up = tuple(float(value) for value in raw["up"])
    camera.forward = tuple(float(value) for value in raw["forward"])
    camera.yaw_degrees = float(raw["yaw_degrees"])
    camera.pitch_degrees = float(raw["pitch_degrees"])
    camera.distance = float(raw["distance"])
    return camera


def _frame_from_dataset(dataset: Mapping[str, Any]):
    frame = type("FrameTuple", (), {})()
    raw = dataset["projection_frame"]
    frame.scale = float(raw["scale"])
    frame.center_x = float(raw["center_x"])
    frame.center_y = float(raw["center_y"])
    frame.normalized_center_u = float(raw["normalized_center_u"])
    frame.normalized_center_v = float(raw["normalized_center_v"])
    return frame

__all__ = [
    "DEFAULT_INTERSECTION_CENTER_JITTER_X",
    "DEFAULT_INTERSECTION_CENTER_JITTER_Y",
    "INTERSECTION_CENTER_XY",
    "MAX_CANDIDATE_BBOX_INTERSECTION_PX",
    "MIN_CANDIDATE_CENTER_SEPARATION_PX",
    "MIN_CANDIDATE_VISIBLE_PX",
    "SCENE_ID",
    "STREET_CAMERA_YAW_BANDS_DEGREES",
    "STREET_FULL_BLEED_FALLBACK_EXTENT_MULTIPLIER",
    "SUPPORTED_INTERSECTION_LAYOUTS",
    "SUPPORTED_SCENE_VARIANTS",
    "_StreetRenderParams",
    "_arm_is_present",
    "_bbox_area",
    "_bbox_intersection_area",
    "_candidate_context_visibility_ok",
    "_candidate_screen_separation_ok",
    "_canvas_floor_polygon_available",
    "_dimensions_for_orientation",
    "_finalize_specs",
    "_frame_from_dataset",
    "_make_street_object_spec",
    "_min_pairwise",
    "_missing_arm_for_layout",
    "_object_reference_points",
    "_object_screen_bbox",
    "_orientation_axis_for_xy",
    "_reference_visibility_ok",
    "_resolve_render_params",
    "_sample_context_specs",
    "_sample_intersection_center",
    "_slot_allowed_for_layout",
    "_translate_scene_xy",
]
