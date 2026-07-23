"""Scene-local sampling helpers for object-scene point-order tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.object_scene import (
    _RenderParams,
    _bbox_intersection_area,
    _build_projection_frame,
    _camera_yaw_band_for_instance,
    _object_reference_points,
    _object_screen_bbox,
    _project_screen,
    _sample_camera,
    _sample_scene_object_specs,
)
from .output import (
    demote_context_spec,
    finalize_object_specs,
    label_order_descriptor,
    min_pairwise,
    order_option_choices,
    relabel_order_points,
    screen_separation_ok,
)


_BRANCH_FIELD = "query" + "_id"
POINT_ORDER_POINT_COUNT = 3
CAMERA_DISTANCE_BRANCH = "camera_distance_order"
MIN_CAMERA_DISTANCE_MARGIN = 0.42
MIN_CAMERA_ORDER_SCREEN_SEPARATION_PX = 106.0
MIN_SCREEN_DEPTH_STEP_PX = 34.0


def _sample_floor_marker_world(
    *,
    rng,
    objects: Sequence[Mapping[str, Any]],
    existing_points: Sequence[Sequence[float]],
    room_extent: float,
) -> Tuple[float, float, float]:
    extent = min(2.82, max(2.1, float(room_extent) - 0.34))
    for _attempt in range(180):
        x = float(rng.uniform(-extent, extent))
        y = float(rng.uniform(-extent, extent))
        point = (x, y, 0.055)
        if any(math.hypot(x - float(other[0]), y - float(other[1])) < 0.58 for other in existing_points):
            continue
        if any(
            math.hypot(x - float(obj["base_xyz"][0]), y - float(obj["base_xyz"][1]))
            < float(obj.get("footprint_radius", 0.42)) + 0.26
            for obj in objects
        ):
            continue
        return point
    raise ValueError("could not place readable floor order marker")


def _sample_raw_floor_markers(
    *,
    rng,
    objects: Sequence[Mapping[str, Any]],
    room_extent: float,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    existing: List[Tuple[float, float, float]] = []
    while len(records) < POINT_ORDER_POINT_COUNT:
        world = _sample_floor_marker_world(
            rng=rng,
            objects=objects,
            existing_points=existing,
            room_extent=float(room_extent),
        )
        marker_index = len(records)
        records.append(
            {
                "marker_id": f"raw_distance_order_point_{marker_index}",
                "surface_kind": "floor",
                "attached_object_id": None,
                "world_xyz": [round(float(value), 4) for value in world],
            }
        )
        existing.append(world)
    rng.shuffle(records)
    return records


def _finalize_floor_markers(
    records: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
    render_params: _RenderParams,
) -> List[Dict[str, Any]]:
    finalized_markers: List[Dict[str, Any]] = []
    for record in records:
        screen = _project_screen(record["world_xyz"], camera, frame)
        x, y = float(screen[0]), float(screen[1])
        if not (
            74.0 <= x <= float(render_params.canvas_width) - 74.0
            and 74.0 <= y <= float(render_params.canvas_height) - 74.0
        ):
            raise ValueError("marked point projects outside readable image area")
        finalized = dict(record)
        finalized.update(
            {
                "screen_xy": [round(float(x), 3), round(float(y), 3)],
                "floor_screen_xy": [round(float(x), 3), round(float(y), 3)],
                "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_markers.append(finalized)
    return finalized_markers


def _marker_bboxes(markers: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    return {
        str(marker["marker_id"]): [
            round(float(marker["screen_xy"][0]) - 28.0, 3),
            round(float(marker["screen_xy"][1]) - 28.0, 3),
            round(float(marker["screen_xy"][0]) + 28.0, 3),
            round(float(marker["screen_xy"][1]) + 28.0, 3),
        ]
        for marker in markers
    }


def _context_specs_for_order(
    *,
    rng,
    context_object_count: int,
    prefix: str,
) -> Tuple[List[Dict[str, Any]], int, int]:
    large_context_count = max(1, min(2, int(context_object_count) // 2))
    small_context_count = max(1, int(context_object_count) - int(large_context_count))
    small_specs, large_specs = _sample_scene_object_specs(
        rng=rng,
        candidate_count=int(small_context_count),
        context_object_count=int(large_context_count),
    )
    return (
        [
            *[demote_context_spec(spec, index=index, prefix=str(prefix)) for index, spec in enumerate(small_specs)],
            *[dict(spec) for spec in large_specs],
        ],
        int(small_context_count),
        int(large_context_count),
    )


def _projection_payload(camera, frame, selected_camera_yaw_band: Sequence[float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return (
        {
            "camera_position": [round(float(value), 4) for value in camera.camera_position],
            "target": [round(float(value), 4) for value in camera.target],
            "yaw_degrees": round(float(camera.yaw_degrees), 4),
            "yaw_band_degrees": [round(float(value), 4) for value in selected_camera_yaw_band],
            "pitch_degrees": round(float(camera.pitch_degrees), 4),
            "distance": round(float(camera.distance), 4),
            "right": [round(float(value), 5) for value in camera.right],
            "up": [round(float(value), 5) for value in camera.up],
            "forward": [round(float(value), 5) for value in camera.forward],
        },
        {
            "scale": round(float(frame.scale), 5),
            "center_x": round(float(frame.center_x), 3),
            "center_y": round(float(frame.center_y), 3),
            "normalized_center_u": round(float(frame.normalized_center_u), 6),
            "normalized_center_v": round(float(frame.normalized_center_v), 6),
        },
    )


def build_camera_distance_order_dataset(
    *,
    branch_key: str,
    scene_variant: str,
    point_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build three floor markers whose camera-distance order is unique and visually legible."""

    if str(branch_key) != CAMERA_DISTANCE_BRANCH:
        raise ValueError(f"unsupported branch: {branch_key}")
    if int(point_count) != POINT_ORDER_POINT_COUNT:
        raise ValueError(f"point camera-distance order expects exactly {POINT_ORDER_POINT_COUNT} marked points")

    rng = spawn_rng(int(instance_seed), "object_scene.point_camera_distance_order.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )

    for _attempt in range(560):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        context_specs, small_context_count, large_context_count = _context_specs_for_order(
            rng=rng,
            context_object_count=int(context_object_count),
            prefix="distance_order",
        )
        marker_records = _sample_raw_floor_markers(
            rng=rng,
            objects=context_specs,
            room_extent=float(render_params.room_extent),
        )
        reference_points = [
            *(point for spec in context_specs for point in _object_reference_points(spec)),
            *(tuple(float(value) for value in record["world_xyz"]) for record in marker_records),
        ]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_context = finalize_object_specs(context_specs, camera=camera, frame=frame, project_screen_fn=_project_screen)
        object_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=10.0) for spec in finalized_context]
        if any(
            _bbox_intersection_area(a, b) > 4200.0
            for index, a in enumerate(object_bboxes)
            for b in object_bboxes[index + 1 :]
        ):
            continue
        try:
            finalized_markers = _finalize_floor_markers(marker_records, camera=camera, frame=frame, render_params=render_params)
        except ValueError:
            continue
        if not screen_separation_ok(finalized_markers, min_px=MIN_CAMERA_ORDER_SCREEN_SEPARATION_PX):
            continue
        if min_pairwise([float(item["camera_distance"]) for item in finalized_markers]) < MIN_CAMERA_DISTANCE_MARGIN:
            continue
        sorted_by_distance = sorted(finalized_markers, key=lambda item: (float(item["camera_distance"]), str(item["marker_id"])))
        sorted_by_screen_depth = sorted(finalized_markers, key=lambda item: (float(item["screen_xy"][1]), str(item["marker_id"])), reverse=True)
        if [str(item["marker_id"]) for item in sorted_by_distance] != [str(item["marker_id"]) for item in sorted_by_screen_depth]:
            continue
        screen_y_values = [float(item["screen_xy"][1]) for item in sorted_by_screen_depth]
        if min_pairwise(screen_y_values) < MIN_SCREEN_DEPTH_STEP_PX:
            continue
        marker_bboxes = _marker_bboxes(finalized_markers)
        if any(
            _bbox_intersection_area(marker_bbox, object_bbox) > 1300.0
            for marker_bbox in marker_bboxes.values()
            for object_bbox in object_bboxes
        ):
            continue

        relabeled_markers = relabel_order_points(finalized_markers, rng=rng)
        answer_order = [
            str(item["point_label"])
            for item in sorted(relabeled_markers, key=lambda item: (float(item["camera_distance"]), str(item["point_label"])))
        ]
        option_choices, answer_label = order_option_choices(answer_order=answer_order)
        camera_payload, projection_payload = _projection_payload(camera, frame, selected_camera_yaw_band)
        return {
            _BRANCH_FIELD: str(branch_key),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "context_object_count": int(context_object_count),
            "small_context_object_count": int(small_context_count),
            "large_context_object_count": int(large_context_count),
            "point_specs": [],
            "context_object_specs": sorted(finalized_context, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted(finalized_context, key=lambda spec: str(spec["object_id"])),
            "marked_points": list(relabeled_markers),
            "option_choices": list(option_choices),
            "answer_label": str(answer_label),
            "answer_order": list(answer_order),
            "answer_descriptor": label_order_descriptor(answer_order),
            "camera": camera_payload,
            "projection_frame": projection_payload,
            "solver_trace": {
                "sort_key": "camera_distance",
                "camera_distance_order_near_to_far": list(answer_order),
                "screen_y_order_front_to_back": [
                    str(item["point_label"])
                    for item in sorted(relabeled_markers, key=lambda item: (float(item["screen_xy"][1]), str(item["point_label"])), reverse=True)
                ],
                "camera_distances_by_label": {
                    str(item["point_label"]): round(float(item["camera_distance"]), 4)
                    for item in sorted(relabeled_markers, key=lambda marker: str(marker["point_label"]))
                },
                "screen_y_by_label": {
                    str(item["point_label"]): round(float(item["screen_xy"][1]), 3)
                    for item in sorted(relabeled_markers, key=lambda marker: str(marker["point_label"]))
                },
                "unique_camera_distance_margin": round(float(min_pairwise([float(item["camera_distance"]) for item in relabeled_markers])), 4),
                "unique_screen_depth_margin_px": round(
                    float(min_pairwise([float(item["screen_xy"][1]) for item in relabeled_markers])),
                    3,
                ),
            },
        }
    raise ValueError("could not construct a valid 3D point camera-distance order scene")


__all__ = [
    "CAMERA_DISTANCE_BRANCH",
    "MIN_CAMERA_DISTANCE_MARGIN",
    "MIN_SCREEN_DEPTH_STEP_PX",
    "POINT_ORDER_POINT_COUNT",
    "build_camera_distance_order_dataset",
]
