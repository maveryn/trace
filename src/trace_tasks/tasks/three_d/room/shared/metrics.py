"""Dataset builder for closest-camera wall-object room objectives."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.camera_projection import build_projection_frame
from ...shared.object_resources import (
    ROOM_CAMERA_DISTANCE_CANDIDATE_WALL_OBJECT_TYPES,
    ROOM_CAMERA_DISTANCE_CONTEXT_WALL_OBJECT_TYPES,
)
from ...shared.object_scene import POINT_LABELS, bbox_intersection_area, object_reference_points
from ...shared.task_support import shuffled_repeated_support
from .state import (
    FLOOR_PROP_SHAPES,
    ROOM_FRONT_Y,
    ROOM_HEIGHT,
    SIDE_WALL_OBJECT_HPOS_MAX,
    SUPPORTED_SCENE_VARIANTS,
    WALL_BACK_Y,
    WALL_X,
    _finalize_specs,
    _make_floor_prop,
    _room_object_bbox,
    _sample_room_camera,
    _wall_object_visible_bbox,
    _wall_object_visible_size_ok,
    _wall_dimensions_for_type,
    _wall_reference_points,
    _wall_spec,
    _with_picture_scenery,
)


CANDIDATE_WALL_OBJECT_TYPES: Tuple[str, ...] = ROOM_CAMERA_DISTANCE_CANDIDATE_WALL_OBJECT_TYPES
CONTEXT_WALL_OBJECT_TYPES: Tuple[str, ...] = ROOM_CAMERA_DISTANCE_CONTEXT_WALL_OBJECT_TYPES
ROOM_CAMERA_DISTANCE_YAW_BANDS: Dict[str, Tuple[float, float]] = {
    "living_room": (-8.0, -3.0),
    "office_room": (3.0, 8.0),
    "studio_room": (-4.0, 4.0),
}
CANDIDATE_WALL_SLOTS: Tuple[Tuple[str, float, float], ...] = (
    ("left", -2.05, 1.22),
    ("left", -0.38, 2.04),
    ("back", -2.08, 1.28),
    ("back", 1.18, 2.12),
    ("right", -1.05, 1.26),
    ("right", 0.38, 2.06),
)
CONTEXT_WALL_SLOTS: Tuple[Tuple[str, float, float], ...] = (
    ("left", -1.56, 1.82),
    ("left", 0.26, 2.22),
    ("back", -0.42, 1.55),
    ("back", 2.36, 1.34),
    ("right", -1.46, 1.82),
    ("right", 0.34, 1.42),
)
LETTERED_WALL_OBJECT_SIZE_SCALE = 1.35
LETTERED_WALL_OBJECT_MIN_VISIBLE_PX = 34.0
CAMERA_DISTANCE_MIN_MARGIN = 0.45
CAMERA_DISTANCE_MIN_ROOM_DEPTH_MARGIN = 0.65


def _candidate_slots(*, candidate_count: int, instance_seed: int, namespace: str) -> List[Tuple[str, float, float]]:
    if int(candidate_count) > len(CANDIDATE_WALL_SLOTS):
        raise ValueError("room wall camera-distance task supports at most six candidates")
    slots = list(CANDIDATE_WALL_SLOTS[: int(candidate_count)])
    if spawn_rng(int(instance_seed), f"{namespace}.front_side_wall").random() < 0.5:
        return slots
    mirrored: List[Tuple[str, float, float]] = []
    for wall, hpos, z in slots:
        if str(wall) == "left":
            mirrored.append(("right", float(hpos), float(z)))
        elif str(wall) == "right":
            mirrored.append(("left", float(hpos), float(z)))
        else:
            mirrored.append((str(wall), float(hpos), float(z)))
    return mirrored


def _sample_candidate_types(rng, candidate_count: int) -> List[str]:
    types = [str(item) for item in CANDIDATE_WALL_OBJECT_TYPES]
    rng.shuffle(types)
    return list(types[: int(candidate_count)])


def _wall_spec_for_type(
    *,
    rng,
    object_id: str,
    object_type: str,
    wall: str,
    hpos: float,
    z: float,
    counts_for_query: bool,
    size_scale: float = 1.0,
) -> Dict[str, Any]:
    width, height = _wall_dimensions_for_type(str(object_type), rng)
    width = float(width) * float(size_scale)
    height = float(height) * float(size_scale)
    return _with_picture_scenery(
        _wall_spec(
            object_id=str(object_id),
            object_type=str(object_type),
            wall=str(wall),
            hpos=float(hpos + rng.uniform(-0.045, 0.045)),
            z=float(z + rng.uniform(-0.035, 0.035)),
            width=float(width),
            height=float(height),
            counts_for_query=bool(counts_for_query),
        ),
        rng,
    )


def _candidate_screen_separation_ok(candidate_specs: Sequence[Mapping[str, Any]], *, camera, frame) -> bool:
    bboxes = [_room_object_bbox(spec, camera, frame) for spec in candidate_specs]
    centers = [(float(spec["screen_xy"][0]), float(spec["screen_xy"][1])) for spec in candidate_specs]
    for index, bbox in enumerate(bboxes):
        for other_index in range(index + 1, len(bboxes)):
            if math.hypot(centers[index][0] - centers[other_index][0], centers[index][1] - centers[other_index][1]) < 48.0:
                return False
            if bbox_intersection_area(bbox, bboxes[other_index]) > 2100.0:
                return False
    return True


def _candidate_wall_visibility_ok(candidate_specs: Sequence[Mapping[str, Any]], *, camera, frame) -> bool:
    if not _wall_object_visible_size_ok(
        candidate_specs,
        camera=camera,
        frame=frame,
        min_width_px=LETTERED_WALL_OBJECT_MIN_VISIBLE_PX,
        min_height_px=LETTERED_WALL_OBJECT_MIN_VISIBLE_PX,
    ):
        return False
    for spec in candidate_specs:
        bbox = _wall_object_visible_bbox(spec, camera, frame)
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        if str(spec.get("wall")) in {"left", "right"}:
            aspect = min(width, height) / max(width, height)
            if width < LETTERED_WALL_OBJECT_MIN_VISIBLE_PX or aspect < 0.36:
                return False
    return True


def _build_floor_context(*, rng, floor_context_count: int) -> List[Dict[str, Any]]:
    floor_slots = [
        (-2.38, -1.95),
        (-1.15, -2.12),
        (0.18, -2.18),
        (1.48, -1.94),
        (2.48, -1.18),
        (-2.52, -0.35),
        (-1.08, -0.68),
        (0.62, -0.56),
        (1.98, -0.34),
        (-2.18, 1.08),
        (-0.62, 0.84),
        (1.02, 0.86),
        (2.34, 0.98),
    ]
    rng.shuffle(floor_slots)
    prop_shapes = list(FLOOR_PROP_SHAPES)
    rng.shuffle(prop_shapes)
    floor_specs: List[Dict[str, Any]] = []
    prop_shape_order = shuffled_repeated_support(rng, prop_shapes, int(floor_context_count))
    for index, prop_shape in enumerate(prop_shape_order):
        xy = floor_slots.pop()
        floor_specs.append(
            _make_floor_prop(
                rng=rng,
                object_id=f"floor_context_{index}_{prop_shape}",
                prop_shape=str(prop_shape),
                xy=(float(xy[0] + rng.uniform(-0.08, 0.08)), float(xy[1] + rng.uniform(-0.08, 0.08))),
            )
        )
    return list(floor_specs)


def _room_depth_to_camera(spec: Mapping[str, Any]) -> float:
    return float(spec["world_xyz"][1])


def build_room_wall_camera_distance_dataset(
    *,
    scene_variant: str,
    candidate_count: int,
    context_wall_count: int,
    floor_context_count: int,
    render_params,
    namespace: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Build one room dataset whose unique answer is the nearest candidate wall object."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    for _attempt in range(360):
        camera = _sample_room_camera(
            rng,
            scene_variant=str(scene_variant),
            yaw_band_degrees=tuple(float(value) for value in ROOM_CAMERA_DISTANCE_YAW_BANDS[str(scene_variant)]),
        )
        candidate_types = _sample_candidate_types(rng, int(candidate_count))
        slots = _candidate_slots(candidate_count=int(candidate_count), instance_seed=int(instance_seed), namespace=str(namespace))
        rng.shuffle(slots)
        candidate_wall_specs: List[Dict[str, Any]] = []
        for index, (wall, hpos, z) in enumerate(slots):
            object_type = str(candidate_types[index])
            spec = _wall_spec_for_type(
                rng=rng,
                object_id=f"candidate_wall_object_{index}_{object_type}",
                object_type=str(object_type),
                wall=str(wall),
                hpos=float(hpos),
                z=float(z),
                counts_for_query=True,
                size_scale=LETTERED_WALL_OBJECT_SIZE_SCALE,
            )
            spec["is_answer_candidate"] = True
            candidate_wall_specs.append(spec)

        context_types = [str(item) for item in CONTEXT_WALL_OBJECT_TYPES]
        rng.shuffle(context_types)
        context_slots = list(CONTEXT_WALL_SLOTS)
        rng.shuffle(context_slots)
        context_wall_specs: List[Dict[str, Any]] = []
        context_slot_order = shuffled_repeated_support(rng, context_slots, int(context_wall_count))
        context_type_order = shuffled_repeated_support(rng, context_types, int(context_wall_count))
        for index, ((wall, hpos, z), object_type) in enumerate(zip(context_slot_order, context_type_order)):
            context_wall_specs.append(
                _wall_spec_for_type(
                    rng=rng,
                    object_id=f"context_wall_object_{index}_{object_type}",
                    object_type=str(object_type),
                    wall=str(wall),
                    hpos=float(hpos),
                    z=float(z),
                    counts_for_query=False,
                )
            )

        floor_specs = _build_floor_context(rng=rng, floor_context_count=int(floor_context_count))
        all_reference_points: List[Tuple[float, float, float]] = [
            (-WALL_X, ROOM_FRONT_Y, 0.0),
            (WALL_X, ROOM_FRONT_Y, 0.0),
            (-WALL_X, WALL_BACK_Y, 0.0),
            (WALL_X, WALL_BACK_Y, 0.0),
            (-WALL_X, WALL_BACK_Y, ROOM_HEIGHT),
            (WALL_X, WALL_BACK_Y, ROOM_HEIGHT),
            (-WALL_X, ROOM_FRONT_Y, ROOM_HEIGHT),
            (WALL_X, ROOM_FRONT_Y, ROOM_HEIGHT),
        ]
        for spec in [*candidate_wall_specs, *context_wall_specs]:
            all_reference_points.extend(_wall_reference_points(spec))
        for spec in floor_specs:
            all_reference_points.extend(object_reference_points(spec))

        frame = build_projection_frame(camera=camera, render_params=render_params, point_worlds=all_reference_points)
        finalized_candidates = _finalize_specs(candidate_wall_specs, camera=camera, frame=frame)
        finalized_context_wall = _finalize_specs(context_wall_specs, camera=camera, frame=frame)
        finalized_floor = _finalize_specs(floor_specs, camera=camera, frame=frame)
        if any(
            str(spec.get("wall")) in {"left", "right"} and float(spec["world_xyz"][1]) > float(SIDE_WALL_OBJECT_HPOS_MAX)
            for spec in [*finalized_candidates, *finalized_context_wall]
        ):
            continue
        if not _candidate_wall_visibility_ok(finalized_candidates, camera=camera, frame=frame):
            continue
        if not _candidate_screen_separation_ok(finalized_candidates, camera=camera, frame=frame):
            continue

        sorted_by_distance = sorted(finalized_candidates, key=lambda spec: (float(spec["camera_distance"]), str(spec["object_id"])))
        distances = [float(spec["camera_distance"]) for spec in sorted_by_distance]
        if len(distances) >= 2 and abs(float(distances[1]) - float(distances[0])) < CAMERA_DISTANCE_MIN_MARGIN:
            continue
        sorted_by_room_depth = sorted(finalized_candidates, key=lambda spec: (_room_depth_to_camera(spec), str(spec["object_id"])))
        room_depths = [_room_depth_to_camera(spec) for spec in sorted_by_room_depth]
        if str(sorted_by_distance[0]["object_id"]) != str(sorted_by_room_depth[0]["object_id"]):
            continue
        if len(room_depths) >= 2 and float(room_depths[1]) - float(room_depths[0]) < CAMERA_DISTANCE_MIN_ROOM_DEPTH_MARGIN:
            continue
        answer_object_id = str(sorted_by_distance[0]["object_id"])
        label_support = tuple(POINT_LABELS[: int(candidate_count)])
        answer_label = str(spawn_rng(int(instance_seed), f"{namespace}.answer_label").choice(label_support))
        remaining_labels = [str(label) for label in POINT_LABELS[: int(candidate_count)] if str(label) != str(answer_label)]
        rng.shuffle(remaining_labels)
        relabeled_candidates: List[Dict[str, Any]] = []
        for spec in finalized_candidates:
            updated = dict(spec)
            label = str(answer_label) if str(updated["object_id"]) == answer_object_id else str(remaining_labels.pop())
            updated.update(
                {
                    "object_id": f"wall_object_{label}",
                    "point_id": f"wall_object_{label}",
                    "point_label": str(label),
                    "object_label": str(label),
                    "is_answer_candidate": True,
                }
            )
            relabeled_candidates.append(updated)

        relabeled_by_distance = sorted(relabeled_candidates, key=lambda spec: (float(spec["camera_distance"]), str(spec["point_label"])))
        answer_spec = next(spec for spec in relabeled_candidates if str(spec["point_label"]) == str(answer_label))
        finalized_wall = [*relabeled_candidates, *finalized_context_wall]
        all_finalized = [*finalized_wall, *finalized_floor]
        wall_object_type_counts = Counter(str(spec["object_type"]) for spec in finalized_wall)
        floor_object_type_counts = Counter(str(spec["object_type"]) for spec in finalized_floor)
        object_type_counts = Counter(str(spec["object_type"]) for spec in all_finalized)
        candidate_camera_distances = {str(spec["point_label"]): round(float(spec["camera_distance"]), 4) for spec in relabeled_candidates}
        candidate_walls = {str(spec["point_label"]): str(spec["wall"]) for spec in relabeled_candidates}
        candidate_projected_bboxes = {str(spec["point_label"]): [round(float(value), 3) for value in _room_object_bbox(spec, camera, frame)] for spec in relabeled_candidates}
        candidate_visible_bboxes = {str(spec["point_label"]): [round(float(value), 3) for value in _wall_object_visible_bbox(spec, camera, frame)] for spec in relabeled_candidates}
        camera_distance_margin = float(relabeled_by_distance[1]["camera_distance"]) - float(answer_spec["camera_distance"])
        relabeled_by_room_depth = sorted(relabeled_candidates, key=lambda spec: (_room_depth_to_camera(spec), str(spec["point_label"])))
        room_depth_margin = _room_depth_to_camera(relabeled_by_room_depth[1]) - _room_depth_to_camera(answer_spec)
        return {
            "scene_variant": str(scene_variant),
            "candidate_count": int(candidate_count),
            "context_wall_count": int(context_wall_count),
            "floor_context_count": int(floor_context_count),
            "wall_object_specs": list(sorted(finalized_wall, key=lambda spec: str(spec["object_id"]))),
            "floor_object_specs": list(sorted(finalized_floor, key=lambda spec: str(spec["object_id"]))),
            "object_specs": list(sorted(all_finalized, key=lambda spec: str(spec["object_id"]))),
            "candidate_object_specs": list(sorted(relabeled_candidates, key=lambda spec: str(spec["point_label"]))),
            "target_object_ids": [str(answer_spec["object_id"])],
            "answer_label": str(answer_label),
            "answer_object_id": str(answer_spec["object_id"]),
            "answer_object_type": str(answer_spec["object_type"]),
            "answer_wall": str(answer_spec["wall"]),
            "wall_object_count": int(len(finalized_wall)),
            "floor_object_count": int(len(finalized_floor)),
            "object_count": int(len(all_finalized)),
            "candidate_camera_distances_by_label": dict(sorted(candidate_camera_distances.items())),
            "candidate_walls_by_label": dict(sorted(candidate_walls.items())),
            "candidate_projected_bboxes_by_label": dict(sorted(candidate_projected_bboxes.items())),
            "candidate_visible_bboxes_by_label": dict(sorted(candidate_visible_bboxes.items())),
            "camera_distance_order_near_to_far": [str(spec["point_label"]) for spec in relabeled_by_distance],
            "room_depth_order_front_to_back": [str(spec["point_label"]) for spec in relabeled_by_room_depth],
            "camera_distance_margin": round(float(camera_distance_margin), 4),
            "room_depth_margin": round(float(room_depth_margin), 4),
            "object_type_counts": dict(sorted(object_type_counts.items())),
            "wall_object_type_counts": dict(sorted(wall_object_type_counts.items())),
            "floor_object_type_counts": dict(sorted(floor_object_type_counts.items())),
            "camera": {
                "camera_position": [round(float(value), 4) for value in camera.camera_position],
                "target": [round(float(value), 4) for value in camera.target],
                "yaw_degrees": round(float(camera.yaw_degrees), 4),
                "yaw_band_degrees": [round(float(value), 4) for value in ROOM_CAMERA_DISTANCE_YAW_BANDS[str(scene_variant)]],
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
                "predicate": "minimum finalized camera_distance among option-panel wall-mounted candidates",
                "candidate_camera_distances_by_label": dict(sorted(candidate_camera_distances.items())),
                "candidate_walls_by_label": dict(sorted(candidate_walls.items())),
                "camera_distance_order_near_to_far": [str(spec["point_label"]) for spec in relabeled_by_distance],
                "room_depth_order_front_to_back": [str(spec["point_label"]) for spec in relabeled_by_room_depth],
                "answer_label": str(answer_label),
                "answer_object_id": str(answer_spec["object_id"]),
                "answer_wall": str(answer_spec["wall"]),
                "camera_distance_margin": round(float(camera_distance_margin), 4),
                "room_depth_margin": round(float(room_depth_margin), 4),
                "unique_answer": True,
            },
        }
    raise ValueError("could not construct a room wall camera-distance scene with a unique visible closest candidate")


__all__ = [
    "CANDIDATE_WALL_OBJECT_TYPES",
    "CAMERA_DISTANCE_MIN_MARGIN",
    "CAMERA_DISTANCE_MIN_ROOM_DEPTH_MARGIN",
    "CONTEXT_WALL_OBJECT_TYPES",
    "CONTEXT_WALL_SLOTS",
    "LETTERED_WALL_OBJECT_MIN_VISIBLE_PX",
    "LETTERED_WALL_OBJECT_SIZE_SCALE",
    "ROOM_CAMERA_DISTANCE_YAW_BANDS",
    "_build_floor_context",
    "_candidate_screen_separation_ok",
    "_candidate_wall_visibility_ok",
    "_wall_spec_for_type",
    "build_room_wall_camera_distance_dataset",
]
