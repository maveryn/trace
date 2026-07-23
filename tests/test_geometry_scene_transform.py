from __future__ import annotations

import random

from trace_tasks.tasks.geometry.shared.scene_transform import (
    LazySceneTransform,
    sample_single_object_scene_transform,
)


def test_scene_transform_rotates_points_inside_canvas() -> None:
    transform = sample_single_object_scene_transform(
        random.Random(7),
        content_points=((200.0, 180.0), (420.0, 180.0), (420.0, 360.0), (200.0, 360.0)),
        params={"scene_rotation_degrees": 45.0},
        render_defaults={
            "single_object_scene_rotation": {
                "enabled": True,
                "probability": 1.0,
                "margin_px": 40,
                "min_scale_after_fit": 0.80,
            }
        },
        canvas_width=640,
        canvas_height=520,
    )
    assert transform.applied is True
    assert transform.angle_degrees == 45.0
    points = transform.points(((200.0, 180.0), (420.0, 360.0)))
    for x_value, y_value in points:
        assert 0.0 <= x_value <= 640.0
        assert 0.0 <= y_value <= 520.0


def test_scene_transform_skips_when_fit_would_shrink_too_much() -> None:
    transform = sample_single_object_scene_transform(
        random.Random(11),
        content_points=((10.0, 10.0), (630.0, 10.0), (630.0, 510.0), (10.0, 510.0)),
        params={"scene_rotation_degrees": 45.0},
        render_defaults={
            "single_object_scene_rotation": {
                "enabled": True,
                "probability": 1.0,
                "margin_px": 60,
                "min_scale_after_fit": 0.95,
            }
        },
        canvas_width=640,
        canvas_height=520,
    )
    assert transform.applied is False
    assert transform.reason == "fit_scale_below_threshold"


def test_lazy_scene_transform_preserves_keyed_point_bindings() -> None:
    resolver = LazySceneTransform(
        random.Random(13),
        params={"scene_rotation_degrees": -30.0},
        render_defaults={
            "single_object_scene_rotation": {
                "enabled": True,
                "probability": 1.0,
                "margin_px": 24,
                "min_scale_after_fit": 0.75,
            }
        },
        canvas_width=512,
        canvas_height=512,
    )
    keys = ("A", "B", "C")
    values = ((120.0, 120.0), (320.0, 120.0), (220.0, 300.0))
    transformed = dict(zip(keys, resolver.points(values)))
    assert tuple(transformed.keys()) == keys
    assert resolver.metadata()["applied"] is True
    for point in transformed.values():
        assert len(point) == 2
