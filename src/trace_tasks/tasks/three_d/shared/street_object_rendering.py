"""Street object dispatch helpers for street-intersection scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from .camera_projection import (
    canvas_floor_polygon_xy as _canvas_floor_polygon_xy,
    polygon_axis_line_segment as _polygon_axis_line_segment,
    project_screen as _project_screen,
    project_xy as _project_xy,
    screen_to_floor_xy as _screen_to_floor_xy,
)
from .color_variation import resolve_three_d_object_fill_rgb
from .object_resources import (
    BUILDING_STYLE_BASE_COLORS,
    BUILDING_STYLE_DIMENSION_FACTORS,
    BUILDING_STYLE_DISPLAY_NAMES,
    STREET_OBJECT_BASE_DIMENSIONS,
    STREET_OBJECT_COLORS,
    STREET_OBJECT_NAMES,
    STREET_RADIAL_OBJECT_TYPES,
    STREET_VEHICLE_OBJECT_TYPES,
)
from .object_scene_rendering import (
    _bbox_union,
    _draw_box_object,
    _draw_box_parts_object,
    _draw_cone_object,
    _draw_cylinder_object,
    _draw_line,
    _draw_sphere_object,
    _shade,
    _sub_box_spec,
    _tint,
)
from .projected_object_geometry import _object_screen_bbox

from .street_object_rendering_common import *  # noqa: F403
from .street_fixture_object_rendering import (
    _draw_construction_barrier_object,
    _draw_fire_hydrant_object,
    _draw_mailbox_object,
    _draw_road_barrel_object,
    _draw_street_sign_context_object,
    _draw_traffic_cone_object,
    _draw_traffic_light_context_object,
    _draw_trash_bin_object,
)
from .street_landscape_object_rendering import (
    _draw_street_bench_object,
    _draw_street_bush_object,
    _draw_street_evergreen_tree_object,
)
from .street_pedestrian_object_rendering import _draw_pedestrian_object
from .street_vehicle_object_rendering import (
    _draw_bicycle_object,
    _draw_motorcycle_object,
    _draw_scooter_object,
    _draw_vehicle_object,
)

def _draw_context_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    object_type = str(spec["object_type"])
    fill = _street_object_fill_rgb(spec)
    if object_type in STREET_BUILDING_CONTEXT_OBJECT_TYPES:
        raise ValueError(f"street building context is scene-local support, not a reusable object: {object_type}")
    if object_type == "tree":
        return _draw_street_evergreen_tree_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "shrub":
        return _draw_street_bush_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "traffic_light":
        return _draw_traffic_light_context_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "street_sign":
        return _draw_street_sign_context_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "bench":
        return _draw_street_bench_object(draw, spec, camera=camera, frame=frame, fill=fill)
    return _draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)

def _draw_candidate_object(
    draw: ImageDraw.ImageDraw,
    spec: Mapping[str, Any],
    *,
    camera,
    frame,
) -> List[float]:
    object_type = str(spec["object_type"])
    fill = _street_object_fill_rgb(spec)
    if object_type in VEHICLE_OBJECT_TYPES:
        return _draw_vehicle_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "scooter":
        return _draw_scooter_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "motorcycle":
        return _draw_motorcycle_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "bicycle":
        return _draw_bicycle_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type in PEDESTRIAN_OBJECT_TYPES:
        return _draw_pedestrian_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "traffic_cone":
        return _draw_traffic_cone_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "fire_hydrant":
        return _draw_fire_hydrant_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "trash_bin":
        return _draw_trash_bin_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "mailbox":
        return _draw_mailbox_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "construction_barrier":
        return _draw_construction_barrier_object(draw, spec, camera=camera, frame=frame, fill=fill)
    if object_type == "road_barrel":
        return _draw_road_barrel_object(draw, spec, camera=camera, frame=frame, fill=fill)
    return _draw_box_object(draw, spec, camera=camera, frame=frame, fill=fill)


__all__ = [
    '_draw_context_object',
    '_draw_candidate_object',
]
