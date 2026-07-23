from __future__ import annotations

import random
from types import SimpleNamespace

from trace_tasks.tasks.three_d.shared import camera_projection
from trace_tasks.tasks.three_d.object_scene import camera_distance_extremum_label as camera_distance


def test_camera_distance_reexports_shared_projection_helpers() -> None:
    assert camera_distance._CameraSpec is camera_projection.CameraSpec
    assert camera_distance._ProjectionFrame is camera_projection.ProjectionFrame
    assert camera_distance._sample_camera is camera_projection.sample_camera
    assert camera_distance._build_projection_frame is camera_projection.build_projection_frame
    assert camera_distance._project_xy is camera_projection.project_xy


def test_screen_floor_projection_round_trip_for_floor_point() -> None:
    camera = camera_projection.sample_camera(
        random.Random(7),
        yaw_band_degrees=(36.0, 36.0),
    )
    render_params = SimpleNamespace(
        canvas_width=640,
        canvas_height=480,
        scene_margin_left_px=40,
        scene_margin_right_px=40,
        scene_margin_top_px=32,
        scene_margin_bottom_px=48,
        room_extent=3.0,
    )
    point = (0.55, -0.35, 0.0)
    frame = camera_projection.build_projection_frame(
        camera=camera,
        render_params=render_params,
        point_worlds=[point],
    )
    screen_x, screen_y = camera_projection.project_xy(point, camera, frame)

    floor_xy = camera_projection.screen_to_floor_xy(
        screen_x,
        screen_y,
        camera=camera,
        frame=frame,
    )

    assert floor_xy is not None
    assert abs(floor_xy[0] - point[0]) < 1e-6
    assert abs(floor_xy[1] - point[1]) < 1e-6
