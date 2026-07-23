"""Dataset builder for wall-plane side-relation room objectives."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.camera_projection import build_projection_frame
from ...shared.object_resources import ROOM_SIDE_RELATION_REFERENCE_OBJECT_TYPE
from ...shared.object_scene import POINT_LABELS, bbox_intersection_area, object_reference_points
from ...shared.task_support import resolve_support_choice_for_namespace, shuffled_repeated_support
from .metrics import (
    CANDIDATE_WALL_OBJECT_TYPES,
    CONTEXT_WALL_OBJECT_TYPES,
    CONTEXT_WALL_SLOTS,
    ROOM_CAMERA_DISTANCE_YAW_BANDS,
    _build_floor_context,
    _candidate_wall_visibility_ok,
    _wall_spec_for_type,
)
from .state import (
    ROOM_FRONT_Y,
    ROOM_HEIGHT,
    WALL_BACK_Y,
    WALL_X,
    _finalize_specs,
    _room_object_bbox,
    _sample_room_camera,
    _wall_object_visible_bbox,
    _wall_reference_points,
)


REFERENCE_OBJECT_TYPE = ROOM_SIDE_RELATION_REFERENCE_OBJECT_TYPE
REFERENCE_WALL_SLOTS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "back": ((-0.95, 1.78),),
    "left": ((-0.95, 1.78),),
    "right": ((-0.55, 1.78),),
}
ANSWER_SLOTS_BY_SIDE_AND_WALL: Dict[str, Dict[str, Tuple[Tuple[float, float], ...]]] = {
    "left": {
        "back": ((-2.22, 1.24), (-2.02, 2.18)),
        "left": ((-2.30, 1.24), (-1.78, 2.18)),
        "right": ((0.12, 1.24), (0.95, 2.18)),
    },
    "right": {
        "back": ((0.42, 1.24), (1.16, 2.18)),
        "left": ((-0.32, 1.24), (0.44, 2.18)),
        "right": ((-1.42, 1.08), (-2.24, 2.42)),
    },
}
DISTRACTOR_SLOTS_BY_SIDE_AND_WALL: Dict[str, Dict[str, Tuple[Tuple[float, float], ...]]] = {
    "left": {
        "back": ((-0.28, 1.20), (0.42, 2.14), (1.14, 1.30), (1.88, 2.02), (2.54, 1.66)),
        "left": ((-0.48, 1.20), (-0.10, 2.14), (0.28, 1.32), (0.66, 2.02), (1.02, 1.66)),
        "right": ((-2.80, 1.74), (-2.30, 1.20), (-1.80, 2.12), (-1.28, 1.32), (-0.78, 2.02)),
    },
    "right": {
        "back": ((-3.05, 1.66), (-2.85, 1.20), (-2.30, 2.12), (-1.75, 1.32), (-1.30, 2.02)),
        "left": ((-2.70, 1.20), (-2.20, 2.12), (-1.70, 1.32), (-1.28, 2.02), (-0.96, 1.66)),
        "right": ((-0.50, 2.62), (-0.10, 2.20), (0.32, 1.78), (0.72, 1.34), (1.10, 0.94)),
    },
}
OTHER_WALL_CANDIDATE_SLOTS_BY_WALL: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "back": ((-2.20, 1.24), (-0.78, 2.12), (0.68, 1.28), (1.92, 2.06), (2.54, 1.66), (-2.85, 1.20)),
    "left": ((-2.20, 1.22), (-1.42, 2.04), (-0.42, 1.30), (0.42, 2.04), (-2.70, 1.20), (-0.96, 1.66)),
    "right": ((-2.20, 1.22), (-1.42, 2.04), (-0.42, 1.30), (0.42, 2.04), (-2.30, 1.20), (0.72, 1.34)),
}
REFERENCE_WALL_OBJECT_SIZE_SCALE = 1.28
SIDE_RELATION_LETTERED_WALL_OBJECT_SIZE_SCALE = 1.35
SIDE_RELATION_SCREEN_MIN_CENTER_DISTANCE_PX = 30.0
SIDE_RELATION_SCREEN_MAX_INTERSECTION_AREA_PX = 6500.0


def _reference_slot(*, instance_seed: int, reference_wall: str, namespace: str) -> Tuple[float, float]:
    slots = tuple(REFERENCE_WALL_SLOTS[str(reference_wall)])
    hpos, z = resolve_support_choice_for_namespace(
        params={},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.reference_slot",
        support_values=slots,
    )[0]
    return float(hpos), float(z)


def _candidate_slots(*, rng, candidate_count: int, reference_wall: str, side_relation: str) -> List[Tuple[str, float, float, bool]]:
    answer_slots = list(ANSWER_SLOTS_BY_SIDE_AND_WALL[str(side_relation)][str(reference_wall)])
    distractor_slots = list(DISTRACTOR_SLOTS_BY_SIDE_AND_WALL[str(side_relation)][str(reference_wall)])
    other_wall_slots = [
        (str(wall), float(hpos), float(z))
        for wall, slots in OTHER_WALL_CANDIDATE_SLOTS_BY_WALL.items()
        if str(wall) != str(reference_wall)
        for hpos, z in slots
    ]
    rng.shuffle(answer_slots)
    rng.shuffle(distractor_slots)
    rng.shuffle(other_wall_slots)
    answer_hpos, answer_z = answer_slots[0]
    slots: List[Tuple[str, float, float, bool]] = [(str(reference_wall), float(answer_hpos), float(answer_z), True)]
    if int(candidate_count) >= 2:
        opposite_hpos, opposite_z = distractor_slots[0]
        slots.append((str(reference_wall), float(opposite_hpos), float(opposite_z), False))
    remaining_count = max(0, int(candidate_count) - len(slots))
    if remaining_count > len(other_wall_slots):
        raise ValueError("could not sample enough other-wall side-relation distractors")
    slots.extend((str(wall), float(hpos), float(z), False) for wall, hpos, z in other_wall_slots[:remaining_count])
    if len(slots) < int(candidate_count):
        raise ValueError("could not sample enough wall-side candidates")
    rng.shuffle(slots)
    return list(slots[: int(candidate_count)])


def _sample_candidate_types(*, rng, candidate_count: int) -> List[str]:
    candidate_types = [str(item) for item in CANDIDATE_WALL_OBJECT_TYPES if str(item) != REFERENCE_OBJECT_TYPE]
    if int(candidate_count) > len(candidate_types):
        raise ValueError("not enough unique candidate wall object types")
    rng.shuffle(candidate_types)
    return list(candidate_types[: int(candidate_count)])


def _build_context_wall_specs(*, rng, context_wall_count: int, excluded_wall: str) -> List[Dict[str, Any]]:
    context_types = [str(item) for item in CONTEXT_WALL_OBJECT_TYPES]
    rng.shuffle(context_types)
    context_slots = [slot for slot in CONTEXT_WALL_SLOTS if str(slot[0]) != str(excluded_wall)]
    rng.shuffle(context_slots)
    specs: List[Dict[str, Any]] = []
    context_slot_order = shuffled_repeated_support(rng, context_slots, int(context_wall_count))
    context_type_order = shuffled_repeated_support(rng, context_types, int(context_wall_count))
    for index, ((wall, hpos, z), object_type) in enumerate(zip(context_slot_order, context_type_order)):
        specs.append(
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
    return list(specs)


def _wall_axis_hpos(spec: Mapping[str, Any]) -> float:
    world = spec.get("world_xyz", (0.0, 0.0, 0.0))
    if str(spec.get("wall")) == "back":
        return float(world[0])
    return float(world[1])


def _wall_left_coordinate(wall: str, hpos: float) -> float:
    if str(wall) == "right":
        return float(hpos)
    return -float(hpos)


def _is_left_of_reference_on_wall(spec: Mapping[str, Any], reference_spec: Mapping[str, Any], *, margin: float = 0.28) -> bool:
    if str(spec.get("wall")) != str(reference_spec.get("wall")):
        return False
    wall = str(reference_spec["wall"])
    candidate_coord = _wall_left_coordinate(wall, _wall_axis_hpos(spec))
    reference_coord = _wall_left_coordinate(wall, _wall_axis_hpos(reference_spec))
    return bool(candidate_coord > reference_coord + float(margin))


def _is_right_of_reference_on_wall(spec: Mapping[str, Any], reference_spec: Mapping[str, Any], *, margin: float = 0.28) -> bool:
    if str(spec.get("wall")) != str(reference_spec.get("wall")):
        return False
    wall = str(reference_spec["wall"])
    candidate_coord = _wall_left_coordinate(wall, _wall_axis_hpos(spec))
    reference_coord = _wall_left_coordinate(wall, _wall_axis_hpos(reference_spec))
    return bool(candidate_coord < reference_coord - float(margin))


def _is_selected_side_of_reference_on_wall(spec: Mapping[str, Any], reference_spec: Mapping[str, Any], *, side_relation: str) -> bool:
    if str(side_relation) == "left":
        return _is_left_of_reference_on_wall(spec, reference_spec)
    if str(side_relation) == "right":
        return _is_right_of_reference_on_wall(spec, reference_spec)
    raise ValueError(f"unsupported wall side relation: {side_relation}")


def _side_relation_screen_separation_ok(candidate_specs: Sequence[Mapping[str, Any]], *, camera, frame) -> bool:
    bboxes = [_room_object_bbox(spec, camera, frame) for spec in candidate_specs]
    centers = [(float(spec["screen_xy"][0]), float(spec["screen_xy"][1])) for spec in candidate_specs]
    for index, bbox in enumerate(bboxes):
        for other_index in range(index + 1, len(bboxes)):
            center_distance = math.hypot(centers[index][0] - centers[other_index][0], centers[index][1] - centers[other_index][1])
            if center_distance < SIDE_RELATION_SCREEN_MIN_CENTER_DISTANCE_PX:
                return False
            if bbox_intersection_area(bbox, bboxes[other_index]) > SIDE_RELATION_SCREEN_MAX_INTERSECTION_AREA_PX:
                return False
    return True


def build_room_wall_side_relation_dataset(
    *,
    side_relation: str,
    scene_variant: str,
    candidate_count: int,
    context_wall_count: int,
    floor_context_count: int,
    reference_wall: str,
    render_params,
    namespace: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Build a room dataset with one candidate on the requested wall-plane side."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    reference_hpos, reference_z = _reference_slot(instance_seed=int(instance_seed), reference_wall=str(reference_wall), namespace=str(namespace))
    for _attempt in range(420):
        camera = _sample_room_camera(
            rng,
            scene_variant=str(scene_variant),
            yaw_band_degrees=tuple(float(value) for value in ROOM_CAMERA_DISTANCE_YAW_BANDS[str(scene_variant)]),
        )
        reference_spec = _wall_spec_for_type(
            rng=rng,
            object_id=f"reference_wall_object_{REFERENCE_OBJECT_TYPE}",
            object_type=REFERENCE_OBJECT_TYPE,
            wall=str(reference_wall),
            hpos=float(reference_hpos),
            z=float(reference_z),
            counts_for_query=False,
            size_scale=REFERENCE_WALL_OBJECT_SIZE_SCALE,
        )
        reference_spec["is_named_reference"] = True
        candidate_types = _sample_candidate_types(rng=rng, candidate_count=int(candidate_count))
        slots = _candidate_slots(rng=rng, candidate_count=int(candidate_count), reference_wall=str(reference_wall), side_relation=str(side_relation))
        candidate_wall_specs: List[Dict[str, Any]] = []
        for index, (wall, hpos, z, intended_answer) in enumerate(slots):
            object_type = str(candidate_types[index])
            spec = _wall_spec_for_type(
                rng=rng,
                object_id=f"candidate_wall_object_{index}_{object_type}",
                object_type=str(object_type),
                wall=str(wall),
                hpos=float(hpos),
                z=float(z),
                counts_for_query=True,
                size_scale=SIDE_RELATION_LETTERED_WALL_OBJECT_SIZE_SCALE,
            )
            spec["is_answer_candidate"] = True
            spec["intended_side_relation"] = bool(intended_answer)
            spec["intended_left_of_reference"] = bool(str(side_relation) == "left" and intended_answer)
            spec["intended_right_of_reference"] = bool(str(side_relation) == "right" and intended_answer)
            candidate_wall_specs.append(spec)

        context_wall_specs = _build_context_wall_specs(rng=rng, context_wall_count=int(context_wall_count), excluded_wall=str(reference_wall))
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
        for spec in [reference_spec, *candidate_wall_specs, *context_wall_specs]:
            all_reference_points.extend(_wall_reference_points(spec))
        for spec in floor_specs:
            all_reference_points.extend(object_reference_points(spec))

        frame = build_projection_frame(camera=camera, render_params=render_params, point_worlds=all_reference_points)
        finalized_reference = _finalize_specs([reference_spec], camera=camera, frame=frame)[0]
        finalized_candidates = _finalize_specs(candidate_wall_specs, camera=camera, frame=frame)
        finalized_context_wall = _finalize_specs(context_wall_specs, camera=camera, frame=frame)
        finalized_floor = _finalize_specs(floor_specs, camera=camera, frame=frame)
        if not _candidate_wall_visibility_ok([finalized_reference, *finalized_candidates], camera=camera, frame=frame):
            continue
        if not _side_relation_screen_separation_ok([finalized_reference, *finalized_candidates], camera=camera, frame=frame):
            continue

        satisfying = [spec for spec in finalized_candidates if _is_selected_side_of_reference_on_wall(spec, finalized_reference, side_relation=str(side_relation))]
        if len(satisfying) != 1:
            continue
        answer_object_id = str(satisfying[0]["object_id"])
        label_support = tuple(POINT_LABELS[: int(candidate_count)])
        answer_label = str(spawn_rng(int(instance_seed), f"{namespace}.answer_label").choice(label_support))
        remaining_labels = [str(label) for label in POINT_LABELS[: int(candidate_count)] if str(label) != str(answer_label)]
        rng.shuffle(remaining_labels)
        relabeled_candidates: List[Dict[str, Any]] = []
        for spec in finalized_candidates:
            updated = dict(spec)
            label = str(answer_label) if str(updated["object_id"]) == answer_object_id else str(remaining_labels.pop())
            updated.update({"object_id": f"wall_object_{label}", "point_id": f"wall_object_{label}", "point_label": str(label), "object_label": str(label), "is_answer_candidate": True})
            relabeled_candidates.append(updated)

        answer_spec = next(spec for spec in relabeled_candidates if str(spec["point_label"]) == str(answer_label))
        finalized_wall = [finalized_reference, *relabeled_candidates, *finalized_context_wall]
        all_finalized = [*finalized_wall, *finalized_floor]
        reference_prompt_name = str(finalized_reference["prompt_name"])
        reference_prompt_name_count = sum(1 for spec in all_finalized if str(spec.get("prompt_name")) == reference_prompt_name)
        if int(reference_prompt_name_count) != 1:
            continue
        wall_object_type_counts = Counter(str(spec["object_type"]) for spec in finalized_wall)
        floor_object_type_counts = Counter(str(spec["object_type"]) for spec in finalized_floor)
        object_type_counts = Counter(str(spec["object_type"]) for spec in all_finalized)
        left_relation_flags = {str(spec["point_label"]): _is_left_of_reference_on_wall(spec, finalized_reference) for spec in relabeled_candidates}
        right_relation_flags = {str(spec["point_label"]): _is_right_of_reference_on_wall(spec, finalized_reference) for spec in relabeled_candidates}
        selected_relation_flags = left_relation_flags if str(side_relation) == "left" else right_relation_flags
        candidate_walls = {str(spec["point_label"]): str(spec["wall"]) for spec in relabeled_candidates}
        candidate_wall_hpos = {str(spec["point_label"]): round(float(_wall_axis_hpos(spec)), 4) for spec in relabeled_candidates}
        candidate_wall_left_coordinates = {str(spec["point_label"]): round(float(_wall_left_coordinate(str(spec["wall"]), _wall_axis_hpos(spec))), 4) for spec in relabeled_candidates}
        candidate_projected_bboxes = {str(spec["point_label"]): [round(float(value), 3) for value in _room_object_bbox(spec, camera, frame)] for spec in relabeled_candidates}
        candidate_visible_bboxes = {str(spec["point_label"]): [round(float(value), 3) for value in _wall_object_visible_bbox(spec, camera, frame)] for spec in relabeled_candidates}
        reference_bbox = [round(float(value), 3) for value in _room_object_bbox(finalized_reference, camera, frame)]
        return {
            "scene_variant": str(scene_variant),
            "candidate_count": int(candidate_count),
            "context_wall_count": int(context_wall_count),
            "floor_context_count": int(floor_context_count),
            "reference_object": {
                "object_id": str(finalized_reference["object_id"]),
                "object_type": str(finalized_reference["object_type"]),
                "prompt_name": reference_prompt_name,
                "wall": str(finalized_reference["wall"]),
                "wall_axis_hpos": round(float(_wall_axis_hpos(finalized_reference)), 4),
                "wall_left_coordinate": round(float(_wall_left_coordinate(str(finalized_reference["wall"]), _wall_axis_hpos(finalized_reference))), 4),
                "world_xyz": list(finalized_reference["world_xyz"]),
                "screen_xy": list(finalized_reference["screen_xy"]),
                "bbox_px": list(reference_bbox),
                "prompt_name_count": int(reference_prompt_name_count),
            },
            "wall_object_specs": list(sorted(finalized_wall, key=lambda spec: str(spec["object_id"]))),
            "floor_object_specs": list(sorted(finalized_floor, key=lambda spec: str(spec["object_id"]))),
            "object_specs": list(sorted(all_finalized, key=lambda spec: str(spec["object_id"]))),
            "candidate_object_specs": list(sorted(relabeled_candidates, key=lambda spec: str(spec["point_label"]))),
            "target_object_ids": [str(answer_spec["object_id"])],
            "answer_label": str(answer_label),
            "answer_object_id": str(answer_spec["object_id"]),
            "answer_object_type": str(answer_spec["object_type"]),
            "answer_wall": str(answer_spec["wall"]),
            "left_of_reference_on_wall_by_label": dict(sorted(left_relation_flags.items())),
            "right_of_reference_on_wall_by_label": dict(sorted(right_relation_flags.items())),
            "selected_side_relation_by_label": dict(sorted(selected_relation_flags.items())),
            "candidate_walls_by_label": dict(sorted(candidate_walls.items())),
            "candidate_wall_hpos_by_label": dict(sorted(candidate_wall_hpos.items())),
            "candidate_wall_left_coordinates_by_label": dict(sorted(candidate_wall_left_coordinates.items())),
            "candidate_projected_bboxes_by_label": dict(sorted(candidate_projected_bboxes.items())),
            "candidate_visible_bboxes_by_label": dict(sorted(candidate_visible_bboxes.items())),
            "wall_object_count": int(len(finalized_wall)),
            "floor_object_count": int(len(finalized_floor)),
            "object_count": int(len(all_finalized)),
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
                "predicate": "option-panel wall-mounted candidate on the requested side of the TV reference along the same wall",
                "reference_object": {
                    "object_id": str(finalized_reference["object_id"]),
                    "object_type": str(finalized_reference["object_type"]),
                    "prompt_name": reference_prompt_name,
                    "wall": str(finalized_reference["wall"]),
                    "prompt_name_count": int(reference_prompt_name_count),
                    "wall_axis_hpos": round(float(_wall_axis_hpos(finalized_reference)), 4),
                    "wall_left_coordinate": round(float(_wall_left_coordinate(str(finalized_reference["wall"]), _wall_axis_hpos(finalized_reference))), 4),
                },
                "left_of_reference_on_wall_by_label": dict(sorted(left_relation_flags.items())),
                "right_of_reference_on_wall_by_label": dict(sorted(right_relation_flags.items())),
                "selected_side_relation_by_label": dict(sorted(selected_relation_flags.items())),
                "candidate_walls_by_label": dict(sorted(candidate_walls.items())),
                "candidate_wall_hpos_by_label": dict(sorted(candidate_wall_hpos.items())),
                "candidate_wall_left_coordinates_by_label": dict(sorted(candidate_wall_left_coordinates.items())),
                "answer_label": str(answer_label),
                "answer_object_id": str(answer_spec["object_id"]),
                "answer_wall": str(answer_spec["wall"]),
                "unique_answer": True,
            },
        }
    raise ValueError("could not construct a room side-relation scene with a unique visible answer")


__all__ = ["REFERENCE_OBJECT_TYPE", "build_room_wall_side_relation_dataset"]
