"""Shared non-semantic transforms for geometry scene renderers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "probability": 0.0,
    "angle_abs_min": 15.0,
    "angle_abs_max": 75.0,
    "margin_px": 48.0,
    "min_scale_after_fit": 0.86,
    "max_attempts": 6,
}


@dataclass(frozen=True)
class SceneTransform:
    """One affine transform sampled for a single geometry scene."""

    enabled: bool
    applied: bool
    angle_degrees: float
    center_px: Point
    scale: float
    translation_px: Point
    margin_px: float
    reason: str = ""

    @classmethod
    def identity(cls, *, enabled: bool = False, margin_px: float = 0.0, reason: str = "identity") -> "SceneTransform":
        return cls(
            enabled=bool(enabled),
            applied=False,
            angle_degrees=0.0,
            center_px=(0.0, 0.0),
            scale=1.0,
            translation_px=(0.0, 0.0),
            margin_px=float(margin_px),
            reason=str(reason),
        )

    def point(self, point: Point) -> Point:
        if not self.applied:
            return (float(point[0]), float(point[1]))
        theta = math.radians(float(self.angle_degrees))
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        dx = float(point[0]) - float(self.center_px[0])
        dy = float(point[1]) - float(self.center_px[1])
        sx = dx * float(self.scale)
        sy = dy * float(self.scale)
        return (
            float(self.center_px[0]) + (sx * cos_theta - sy * sin_theta) + float(self.translation_px[0]),
            float(self.center_px[1]) + (sx * sin_theta + sy * cos_theta) + float(self.translation_px[1]),
        )

    def points(self, points: Sequence[Point]) -> tuple[Point, ...]:
        return tuple(self.point(point) for point in points)

    def keyed_points(self, points: Mapping[str, Point]) -> dict[str, Point]:
        return {str(key): self.point(point) for key, point in points.items()}

    def bbox_from_points(self, points: Sequence[Point], *, width: int, height: int, pad: float = 0.0) -> BBox:
        transformed = self.points(points)
        return bbox_from_points(transformed, width=int(width), height=int(height), pad=float(pad))

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "applied": bool(self.applied),
            "angle_degrees": round(float(self.angle_degrees), 3),
            "center_px": [round(float(self.center_px[0]), 3), round(float(self.center_px[1]), 3)],
            "scale": round(float(self.scale), 6),
            "translation_px": [round(float(self.translation_px[0]), 3), round(float(self.translation_px[1]), 3)],
            "margin_px": round(float(self.margin_px), 3),
            "reason": str(self.reason),
            "applied_before_annotation_projection": True,
        }


class LazySceneTransform:
    """Resolve one scene transform on first use, then reuse it consistently."""

    def __init__(
        self,
        rng: Any,
        *,
        params: Mapping[str, Any],
        render_defaults: Mapping[str, Any],
        canvas_width: int,
        canvas_height: int,
    ) -> None:
        self._rng = rng
        self._params = params
        self._render_defaults = render_defaults
        self._canvas_width = int(canvas_width)
        self._canvas_height = int(canvas_height)
        self._transform: SceneTransform | None = None

    @property
    def resolved(self) -> bool:
        return self._transform is not None

    @property
    def transform(self) -> SceneTransform:
        if self._transform is None:
            self._transform = sample_single_object_scene_transform(
                self._rng,
                content_points=(),
                params=self._params,
                render_defaults=self._render_defaults,
                canvas_width=self._canvas_width,
                canvas_height=self._canvas_height,
            )
        return self._transform

    def resolve(self, content_points: Sequence[Point]) -> SceneTransform:
        if self._transform is None:
            self._transform = sample_single_object_scene_transform(
                self._rng,
                content_points=content_points,
                params=self._params,
                render_defaults=self._render_defaults,
                canvas_width=self._canvas_width,
                canvas_height=self._canvas_height,
            )
        return self._transform

    def points(self, points: Sequence[Point]) -> tuple[Point, ...]:
        return self.resolve(points).points(points)

    def keyed_points(self, points: Mapping[str, Point]) -> dict[str, Point]:
        self.resolve(tuple(points.values()))
        return self.transform.keyed_points(points)

    def point(self, point: Point) -> Point:
        if self._transform is None:
            return self.resolve((point,)).point(point)
        return self._transform.point(point)

    def metadata(self) -> dict[str, Any]:
        return self.transform.metadata()


def scene_rotation_config(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> dict[str, Any]:
    raw = render_defaults.get("single_object_scene_rotation", {})
    config = dict(_DEFAULT_CONFIG)
    if isinstance(raw, Mapping):
        config.update(dict(raw))
    prefix = "single_object_scene_rotation_"
    for key in tuple(config.keys()):
        param_key = f"{prefix}{key}"
        if param_key in params:
            config[str(key)] = params[param_key]
    if "scene_rotation_degrees" in params:
        config["enabled"] = True
        config["probability"] = 1.0
        config["angle_degrees"] = params["scene_rotation_degrees"]
    return config


def sample_single_object_scene_transform(
    rng: Any,
    *,
    content_points: Sequence[Point],
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    canvas_width: int,
    canvas_height: int,
) -> SceneTransform:
    config = scene_rotation_config(params, render_defaults)
    enabled = bool(config.get("enabled", False))
    margin = float(config.get("margin_px", _DEFAULT_CONFIG["margin_px"]))
    if not enabled:
        return SceneTransform.identity(enabled=False, margin_px=margin, reason="disabled")
    points = tuple((float(point[0]), float(point[1])) for point in content_points)
    if len(points) < 2:
        return SceneTransform.identity(enabled=True, margin_px=margin, reason="insufficient_content_points")
    probability = max(0.0, min(1.0, float(config.get("probability", _DEFAULT_CONFIG["probability"]))))
    if float(rng.random()) >= probability and "angle_degrees" not in config:
        return SceneTransform.identity(enabled=True, margin_px=margin, reason="probability_skip")

    angle_min = abs(float(config.get("angle_abs_min", _DEFAULT_CONFIG["angle_abs_min"])))
    angle_max = abs(float(config.get("angle_abs_max", _DEFAULT_CONFIG["angle_abs_max"])))
    if angle_max < angle_min:
        angle_min, angle_max = angle_max, angle_min
    max_attempts = max(1, int(config.get("max_attempts", _DEFAULT_CONFIG["max_attempts"])))
    min_scale = float(config.get("min_scale_after_fit", _DEFAULT_CONFIG["min_scale_after_fit"]))
    min_x, min_y, max_x, max_y = _point_bounds(points)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    available_width = max(1.0, float(canvas_width) - (2.0 * margin))
    available_height = max(1.0, float(canvas_height) - (2.0 * margin))

    for attempt in range(max_attempts):
        if "angle_degrees" in config:
            angle = float(config["angle_degrees"])
        else:
            sign = -1.0 if int(rng.randrange(2)) == 0 else 1.0
            angle = sign * float(rng.uniform(angle_min, angle_max))
        if abs(float(angle)) <= 1e-6:
            return SceneTransform.identity(enabled=True, margin_px=margin, reason="zero_angle")

        unit = SceneTransform(
            enabled=True,
            applied=True,
            angle_degrees=float(angle),
            center_px=center,
            scale=1.0,
            translation_px=(0.0, 0.0),
            margin_px=margin,
            reason="candidate",
        )
        rotated = unit.points(points)
        r_min_x, r_min_y, r_max_x, r_max_y = _point_bounds(rotated)
        bbox_width = max(1e-6, r_max_x - r_min_x)
        bbox_height = max(1e-6, r_max_y - r_min_y)
        scale = min(1.0, available_width / bbox_width, available_height / bbox_height)
        if float(scale) < float(min_scale):
            if "angle_degrees" in config:
                break
            continue

        scaled_candidate = SceneTransform(
            enabled=True,
            applied=True,
            angle_degrees=float(angle),
            center_px=center,
            scale=float(scale),
            translation_px=(0.0, 0.0),
            margin_px=margin,
            reason="candidate",
        )
        scaled = scaled_candidate.points(points)
        s_min_x, s_min_y, s_max_x, s_max_y = _point_bounds(scaled)
        dx = 0.0
        dy = 0.0
        if s_min_x < margin:
            dx = margin - s_min_x
        elif s_max_x > float(canvas_width) - margin:
            dx = (float(canvas_width) - margin) - s_max_x
        if s_min_y < margin:
            dy = margin - s_min_y
        elif s_max_y > float(canvas_height) - margin:
            dy = (float(canvas_height) - margin) - s_max_y
        return SceneTransform(
            enabled=True,
            applied=True,
            angle_degrees=float(angle),
            center_px=center,
            scale=float(scale),
            translation_px=(dx, dy),
            margin_px=margin,
            reason="sampled" if attempt == 0 else f"sampled_after_{attempt + 1}_attempts",
        )

    return SceneTransform.identity(enabled=True, margin_px=margin, reason="fit_scale_below_threshold")


def bbox_from_points(points: Sequence[Point], *, width: int, height: int, pad: float = 0.0) -> BBox:
    pts = tuple((float(point[0]), float(point[1])) for point in points)
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    min_x, min_y, max_x, max_y = _point_bounds(pts)
    return (
        round(max(0.0, min(float(width), min_x - float(pad))), 3),
        round(max(0.0, min(float(height), min_y - float(pad))), 3),
        round(max(0.0, min(float(width), max_x + float(pad))), 3),
        round(max(0.0, min(float(height), max_y + float(pad))), 3),
    )


def _point_bounds(points: Sequence[Point]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return min(xs), min(ys), max(xs), max(ys)


__all__ = [
    "BBox",
    "LazySceneTransform",
    "Point",
    "SceneTransform",
    "bbox_from_points",
    "sample_single_object_scene_transform",
    "scene_rotation_config",
]
