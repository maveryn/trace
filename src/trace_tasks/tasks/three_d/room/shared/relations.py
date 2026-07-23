"""Dataset builder for same-wall reference room objectives."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ...shared.camera_projection import build_projection_frame
from ...shared.object_resources import ROOM_SAME_WALL_REFERENCE_WALL_OBJECT_TYPES
from ...shared.object_scene import POINT_LABELS, object_reference_points
from ...shared.task_support import resolve_support_choice_for_namespace, shuffled_repeated_support
from .metrics import (
    CANDIDATE_WALL_OBJECT_TYPES,
    CONTEXT_WALL_OBJECT_TYPES,
    CONTEXT_WALL_SLOTS,
    LETTERED_WALL_OBJECT_SIZE_SCALE,
    ROOM_CAMERA_DISTANCE_YAW_BANDS,
    _build_floor_context,
    _candidate_screen_separation_ok,
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


REFERENCE_WALL_OBJECT_TYPES: Tuple[str, ...] = ROOM_SAME_WALL_REFERENCE_WALL_OBJECT_TYPES
REFERENCE_WALL_SLOTS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "back": ((-2.18, 1.82), (0.0, 1.76), (2.18, 1.82)),
    "left": ((-1.42, 1.74), (0.18, 1.88)),
    "right": ((-1.42, 1.74), (0.18, 1.88)),
}
WALL_CANDIDATE_SLOTS_BY_WALL: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "back": ((-2.20, 1.24), (-0.78, 2.12), (0.68, 1.28), (1.92, 2.06)),
    "left": ((-2.20, 1.22), (-1.42, 2.04), (-0.42, 1.30), (0.42, 2.04)),
    "right": ((-2.20, 1.22), (-1.42, 2.04), (-0.42, 1.30), (0.42, 2.04)),
}
REFERENCE_WALL_OBJECT_SIZE_SCALE = 1.25


def _slot_distance(slot_a: Tuple[float, float], slot_b: Tuple[float, float]) -> float:
    return ((float(slot_a[0]) - float(slot_b[0])) ** 2 + (float(slot_a[1]) - float(slot_b[1])) ** 2) ** 0.5


def _reference_slot(*, instance_seed: int, reference_wall: str, namespace: str) -> Tuple[float, float]:
    slots = tuple(REFERENCE_WALL_SLOTS[str(reference_wall)])
    hpos, z = resolve_support_choice_for_namespace(
        params={},
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.reference_slot",
        support_values=slots,
    )[0]
    return float(hpos), float(z)


def _candidate_slots(
    *,
    rng,
    candidate_count: int,
    reference_wall: str,
    reference_slot: Tuple[float, float],
) -> List[Tuple[str, float, float]]:
    same_wall_slots = [slot for slot in WALL_CANDIDATE_SLOTS_BY_WALL[str(reference_wall)] if _slot_distance(slot, reference_slot) >= 0.68]
    if not same_wall_slots:
        raise ValueError("could not place answer candidate away from reference")
    rng.shuffle(same_wall_slots)
    answer_hpos, answer_z = same_wall_slots[0]
    slots: List[Tuple[str, float, float]] = [(str(reference_wall), float(answer_hpos), float(answer_z))]
    other_slots: List[Tuple[str, float, float]] = []
    for wall, wall_slots in WALL_CANDIDATE_SLOTS_BY_WALL.items():
        if str(wall) == str(reference_wall):
            continue
        for hpos, z in wall_slots:
            other_slots.append((str(wall), float(hpos), float(z)))
    rng.shuffle(other_slots)
    slots.extend(other_slots[: max(0, int(candidate_count) - 1)])
    if len(slots) < int(candidate_count):
        raise ValueError("could not sample enough wall candidates")
    return list(slots[: int(candidate_count)])


def _sample_candidate_types(*, rng, reference_object_type: str, candidate_count: int) -> List[str]:
    candidate_types = [str(item) for item in CANDIDATE_WALL_OBJECT_TYPES if str(item) != str(reference_object_type)]
    if int(candidate_count) > len(candidate_types):
        raise ValueError("not enough unique candidate wall object types")
    rng.shuffle(candidate_types)
    return list(candidate_types[: int(candidate_count)])


def _build_context_wall_specs(*, rng, reference_object_type: str, context_wall_count: int) -> List[Dict[str, Any]]:
    context_types = [str(item) for item in CONTEXT_WALL_OBJECT_TYPES if str(item) != str(reference_object_type)]
    rng.shuffle(context_types)
    context_slots = list(CONTEXT_WALL_SLOTS)
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


def build_room_wall_same_wall_reference_dataset(
    *,
    scene_variant: str,
    candidate_count: int,
    context_wall_count: int,
    floor_context_count: int,
    reference_wall: str,
    reference_object_type: str,
    render_params,
    namespace: str,
    instance_seed: int,
) -> Dict[str, Any]:
    """Build a room dataset with one candidate sharing the reference object's wall."""

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
            object_id=f"reference_wall_object_{reference_object_type}",
            object_type=str(reference_object_type),
            wall=str(reference_wall),
            hpos=float(reference_hpos),
            z=float(reference_z),
            counts_for_query=False,
            size_scale=REFERENCE_WALL_OBJECT_SIZE_SCALE,
        )
        reference_spec["is_named_reference"] = True
        candidate_types = _sample_candidate_types(rng=rng, reference_object_type=str(reference_object_type), candidate_count=int(candidate_count))
        slots = _candidate_slots(rng=rng, candidate_count=int(candidate_count), reference_wall=str(reference_wall), reference_slot=(float(reference_hpos), float(reference_z)))
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

        context_wall_specs = _build_context_wall_specs(rng=rng, reference_object_type=str(reference_object_type), context_wall_count=int(context_wall_count))
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
        if not _candidate_screen_separation_ok([finalized_reference, *finalized_candidates], camera=camera, frame=frame):
            continue

        satisfying = [spec for spec in finalized_candidates if str(spec["wall"]) == str(finalized_reference["wall"])]
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
        same_wall_flags = {str(spec["point_label"]): str(spec["wall"]) == str(finalized_reference["wall"]) for spec in relabeled_candidates}
        candidate_walls = {str(spec["point_label"]): str(spec["wall"]) for spec in relabeled_candidates}
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
            "same_wall_as_reference_by_label": dict(sorted(same_wall_flags.items())),
            "candidate_walls_by_label": dict(sorted(candidate_walls.items())),
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
                "predicate": "option-panel wall-mounted candidate whose wall equals the uniquely named reference wall object wall",
                "reference_object": {
                    "object_id": str(finalized_reference["object_id"]),
                    "object_type": str(finalized_reference["object_type"]),
                    "prompt_name": reference_prompt_name,
                    "wall": str(finalized_reference["wall"]),
                    "prompt_name_count": int(reference_prompt_name_count),
                },
                "same_wall_as_reference_by_label": dict(sorted(same_wall_flags.items())),
                "candidate_walls_by_label": dict(sorted(candidate_walls.items())),
                "answer_label": str(answer_label),
                "answer_object_id": str(answer_spec["object_id"]),
                "answer_wall": str(answer_spec["wall"]),
                "unique_answer": True,
            },
        }
    raise ValueError("could not construct a room same-wall-reference scene with a unique visible answer")


__all__ = ["REFERENCE_WALL_OBJECT_TYPES", "build_room_wall_same_wall_reference_dataset"]
