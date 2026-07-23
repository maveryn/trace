"""Shared 3D camera and projection helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_CAMERA_YAW_BANDS_DEGREES: Tuple[Tuple[float, float], ...] = (
    (-145.0, -108.0),
    (-82.0, -48.0),
    (-42.0, -20.0),
    (20.0, 42.0),
    (48.0, 82.0),
    (108.0, 145.0),
)


@dataclass(frozen=True)
class CameraSpec:
    camera_position: Tuple[float, float, float]
    target: Tuple[float, float, float]
    right: Tuple[float, float, float]
    up: Tuple[float, float, float]
    forward: Tuple[float, float, float]
    yaw_degrees: float
    pitch_degrees: float
    distance: float


@dataclass(frozen=True)
class ProjectionFrame:
    scale: float
    center_x: float
    center_y: float
    normalized_center_u: float
    normalized_center_v: float



def vec_sub(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def vec_cross(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def vec_norm(v: Sequence[float]) -> Tuple[float, float, float]:
    length = max(1e-9, math.sqrt(sum(float(component) * float(component) for component in v)))
    return (float(v[0]) / length, float(v[1]) / length, float(v[2]) / length)


def distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def min_pairwise(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 999.0
    return min(abs(float(a) - float(b)) for index, a in enumerate(values) for b in values[index + 1 :])



def sample_camera(rng, *, yaw_band_degrees: Tuple[float, float] | None = None) -> CameraSpec:
    yaw_lower, yaw_upper = (
        tuple(float(value) for value in yaw_band_degrees)
        if yaw_band_degrees is not None
        else tuple(float(value) for value in rng.choice(DEFAULT_CAMERA_YAW_BANDS_DEGREES))
    )
    yaw_degrees = float(rng.uniform(float(yaw_lower), float(yaw_upper)))
    pitch_degrees = float(rng.uniform(18.0, 33.0))
    distance = float(rng.uniform(7.2, 8.8))
    yaw = math.radians(float(yaw_degrees))
    pitch = math.radians(float(pitch_degrees))
    target = (0.0, 0.0, 0.72)
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


def project_normalized(point: Sequence[float], camera: CameraSpec) -> Tuple[float, float, float, float, float, float]:
    rel = vec_sub(point, camera.camera_position)
    cx = vec_dot(rel, camera.right)
    cy = vec_dot(rel, camera.up)
    cz = max(1e-6, vec_dot(rel, camera.forward))
    return (float(cx / cz), float(cy / cz), float(cz), float(cx), float(cy), float(distance(point, camera.camera_position)))


def stage_reference_points(extent: float) -> List[Tuple[float, float, float]]:
    e = float(extent)
    return [
        (x, y, 0.0)
        for x in (-e, e)
        for y in (-e, e)
    ] + [(0.0, 0.0, 0.0), (0.0, e, 0.0), (0.0, -e, 0.0)]


def build_projection_frame(
    *,
    camera: CameraSpec,
    render_params: Any,
    point_worlds: Sequence[Sequence[float]],
) -> ProjectionFrame:
    points = list(point_worlds) + stage_reference_points(render_params.room_extent)
    normalized = [project_normalized(point, camera) for point in points]
    min_u = min(value[0] for value in normalized)
    max_u = max(value[0] for value in normalized)
    min_v = min(value[1] for value in normalized)
    max_v = max(value[1] for value in normalized)
    usable_width = float(render_params.canvas_width - render_params.scene_margin_left_px - render_params.scene_margin_right_px)
    usable_height = float(render_params.canvas_height - render_params.scene_margin_top_px - render_params.scene_margin_bottom_px)
    scale = min(usable_width / max(0.01, max_u - min_u), usable_height / max(0.01, max_v - min_v))
    return ProjectionFrame(
        scale=float(scale),
        center_x=float(render_params.scene_margin_left_px + 0.5 * usable_width),
        center_y=float(render_params.scene_margin_top_px + 0.5 * usable_height),
        normalized_center_u=float(0.5 * (min_u + max_u)),
        normalized_center_v=float(0.5 * (min_v + max_v)),
    )


def project_screen(point: Sequence[float], camera: CameraSpec, frame: ProjectionFrame) -> Tuple[float, float, float, float, float, float, float, float]:
    u, v, cz, cx, cy, distance = project_normalized(point, camera)
    x = float(frame.center_x + (u - frame.normalized_center_u) * frame.scale)
    y = float(frame.center_y - (v - frame.normalized_center_v) * frame.scale)
    return (float(x), float(y), float(u), float(v), float(cz), float(cx), float(cy), float(distance))


def project_xy(point: Sequence[float], camera: CameraSpec, frame: ProjectionFrame) -> Tuple[float, float]:
    projected = project_screen(point, camera, frame)
    return (float(projected[0]), float(projected[1]))


def screen_to_normalized(screen_x: float, screen_y: float, frame: ProjectionFrame) -> Tuple[float, float]:
    u = (float(screen_x) - float(frame.center_x)) / max(1e-9, float(frame.scale)) + float(frame.normalized_center_u)
    v = (float(frame.center_y) - float(screen_y)) / max(1e-9, float(frame.scale)) + float(frame.normalized_center_v)
    return (float(u), float(v))


def screen_to_floor_xy(
    screen_x: float,
    screen_y: float,
    *,
    camera: CameraSpec,
    frame: ProjectionFrame,
) -> Tuple[float, float] | None:
    u, v = screen_to_normalized(float(screen_x), float(screen_y), frame)
    direction = (
        float(camera.forward[0]) + float(u) * float(camera.right[0]) + float(v) * float(camera.up[0]),
        float(camera.forward[1]) + float(u) * float(camera.right[1]) + float(v) * float(camera.up[1]),
        float(camera.forward[2]) + float(u) * float(camera.right[2]) + float(v) * float(camera.up[2]),
    )
    if float(direction[2]) >= -1e-7:
        return None
    t = -float(camera.camera_position[2]) / float(direction[2])
    if float(t) <= 1e-7:
        return None
    return (
        float(camera.camera_position[0]) + float(t) * float(direction[0]),
        float(camera.camera_position[1]) + float(t) * float(direction[1]),
    )


def canvas_floor_polygon_xy(
    *,
    camera: CameraSpec,
    frame: ProjectionFrame,
    render_params: Any,
) -> List[Tuple[float, float]]:
    width = float(render_params.canvas_width)
    height = float(render_params.canvas_height)
    polygon: List[Tuple[float, float]] = []
    for screen_x, screen_y in ((0.0, 0.0), (width, 0.0), (width, height), (0.0, height)):
        floor_xy = screen_to_floor_xy(screen_x, screen_y, camera=camera, frame=frame)
        if floor_xy is None:
            return []
        polygon.append(floor_xy)
    return polygon


def grid_values_for_range(min_value: float, max_value: float, step: float) -> List[float]:
    grid_step = max(0.05, float(step))
    start = math.floor(float(min_value) / grid_step) * grid_step
    end = math.ceil(float(max_value) / grid_step) * grid_step
    count = int(max(1, math.ceil((float(end) - float(start)) / grid_step))) + 1
    return [round(float(start) + index * grid_step, 6) for index in range(count + 1)]


def dedupe_line_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    unique: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for point in points:
        unique[(round(float(point[0]) * 1_000_000), round(float(point[1]) * 1_000_000))] = (
            float(point[0]),
            float(point[1]),
        )
    return list(unique.values())


def polygon_axis_line_segment(
    polygon_xy: Sequence[Tuple[float, float]],
    *,
    axis: str,
    value: float,
) -> Tuple[Tuple[float, float], Tuple[float, float]] | None:
    intersections: List[Tuple[float, float]] = []
    eps = 1e-8
    for index, point_a in enumerate(polygon_xy):
        next_index = index + 1
        if next_index >= len(polygon_xy):
            next_index = 0
        point_b = polygon_xy[next_index]
        x1, y1 = float(point_a[0]), float(point_a[1])
        x2, y2 = float(point_b[0]), float(point_b[1])
        if str(axis) == "x":
            delta_1 = x1 - float(value)
            delta_2 = x2 - float(value)
            if abs(delta_1) <= eps and abs(delta_2) <= eps:
                intersections.extend([(float(value), y1), (float(value), y2)])
            elif delta_1 * delta_2 <= eps and abs(x2 - x1) > eps:
                t = (float(value) - x1) / (x2 - x1)
                if -eps <= t <= 1.0 + eps:
                    intersections.append((float(value), y1 + t * (y2 - y1)))
        else:
            delta_1 = y1 - float(value)
            delta_2 = y2 - float(value)
            if abs(delta_1) <= eps and abs(delta_2) <= eps:
                intersections.extend([(x1, float(value)), (x2, float(value))])
            elif delta_1 * delta_2 <= eps and abs(y2 - y1) > eps:
                t = (float(value) - y1) / (y2 - y1)
                if -eps <= t <= 1.0 + eps:
                    intersections.append((x1 + t * (x2 - x1), float(value)))
    unique_points = dedupe_line_points(intersections)
    if len(unique_points) < 2:
        return None
    if str(axis) == "x":
        sorted_points = sorted(unique_points, key=lambda point: float(point[1]))
    else:
        sorted_points = sorted(unique_points, key=lambda point: float(point[0]))
    return (sorted_points[0], sorted_points[-1])


__all__ = [
    "CameraSpec",
    "DEFAULT_CAMERA_YAW_BANDS_DEGREES",
    "ProjectionFrame",
    "build_projection_frame",
    "canvas_floor_polygon_xy",
    "dedupe_line_points",
    "distance",
    "grid_values_for_range",
    "min_pairwise",
    "polygon_axis_line_segment",
    "project_normalized",
    "project_screen",
    "project_xy",
    "sample_camera",
    "screen_to_floor_xy",
    "screen_to_normalized",
    "stage_reference_points",
    "vec_cross",
    "vec_dot",
    "vec_norm",
    "vec_sub",
]
