"""Landmark definitions for object-scene correspondence tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class ObjectLandmarkSpec:
    """One stable local landmark on a prompt-safe object-scene shape."""

    landmark_id: str
    display_name: str
    coordinate_mode: str
    local_xyz: Tuple[float, float, float]


LANDMARK_CORRESPONDENCE_SHAPE_TYPES: Tuple[str, ...] = (
    "fish",
    "key",
    "hammer",
    "sword",
    "glove",
    "leaf",
    "plug",
    "open_book",
)


LANDMARKS_BY_SHAPE_TYPE: Dict[str, Tuple[ObjectLandmarkSpec, ...]] = {
    "fish": (
        ObjectLandmarkSpec("mouth", "mouth", "upright_profile", (0.92, 0.0, 0.0)),
        ObjectLandmarkSpec("tail_tip", "tail tip", "upright_profile", (-1.08, 0.0, 0.0)),
        ObjectLandmarkSpec("top_fin", "top fin", "upright_profile", (0.18, 0.0, 0.82)),
        ObjectLandmarkSpec("bottom_fin", "bottom fin", "upright_profile", (0.26, 0.0, -0.78)),
    ),
    "key": (
        ObjectLandmarkSpec("ring_center", "ring center", "floor_plan", (0.0, -0.36, 0.36)),
        ObjectLandmarkSpec("bow_bottom", "bow bottom", "floor_plan", (0.0, -0.78, 0.34)),
        ObjectLandmarkSpec("shaft_center", "shaft center", "floor_plan", (0.0, 0.18, 0.56)),
        ObjectLandmarkSpec("tooth_tip", "tooth tip", "floor_plan", (0.48, 0.58, 0.58)),
    ),
    "hammer": (
        ObjectLandmarkSpec("left_head", "left head", "floor_plan", (-0.58, 0.38, 0.76)),
        ObjectLandmarkSpec("right_head", "right head", "floor_plan", (0.58, 0.38, 0.76)),
        ObjectLandmarkSpec("handle_end", "handle end", "floor_plan", (0.0, -0.56, 0.62)),
        ObjectLandmarkSpec("neck", "neck", "floor_plan", (0.0, 0.18, 0.66)),
    ),
    "sword": (
        ObjectLandmarkSpec("blade_tip", "blade tip", "floor_plan", (0.0, 0.98, 0.66)),
        ObjectLandmarkSpec("pommel", "pommel", "floor_plan", (0.0, -0.92, 0.62)),
        ObjectLandmarkSpec("left_guard", "left guard", "floor_plan", (-0.62, -0.20, 0.68)),
        ObjectLandmarkSpec("right_guard", "right guard", "floor_plan", (0.62, -0.20, 0.68)),
    ),
    "glove": (
        ObjectLandmarkSpec("thumb_tip", "thumb tip", "upright_profile", (0.78, 0.0, 0.18)),
        ObjectLandmarkSpec("index_finger_tip", "index fingertip", "upright_profile", (0.36, 0.0, 0.94)),
        ObjectLandmarkSpec("pinky_tip", "pinky fingertip", "upright_profile", (-0.50, 0.0, 0.86)),
        ObjectLandmarkSpec("wrist", "wrist", "upright_profile", (-0.12, 0.0, -0.88)),
    ),
    "leaf": (
        ObjectLandmarkSpec("left_tip", "left tip", "upright_profile", (-0.88, 0.0, 0.0)),
        ObjectLandmarkSpec("right_tip", "right tip", "upright_profile", (0.88, 0.0, 0.0)),
        ObjectLandmarkSpec("top_edge", "top edge", "upright_profile", (0.0, 0.0, 0.82)),
        ObjectLandmarkSpec("bottom_edge", "bottom edge", "upright_profile", (0.0, 0.0, -0.82)),
    ),
    "plug": (
        ObjectLandmarkSpec("prong_tip", "prong tip", "upright_profile", (1.04, 0.0, 0.20)),
        ObjectLandmarkSpec("cord_end", "cord end", "upright_profile", (-1.08, 0.0, -0.28)),
        ObjectLandmarkSpec("body_top", "body top", "upright_profile", (0.04, 0.0, 0.54)),
        ObjectLandmarkSpec("body_bottom", "body bottom", "upright_profile", (0.04, 0.0, -0.54)),
    ),
    "open_book": (
        ObjectLandmarkSpec("left_page_corner", "left page corner", "upright_profile", (-0.84, 0.0, 0.78)),
        ObjectLandmarkSpec("right_page_corner", "right page corner", "upright_profile", (0.84, 0.0, 0.78)),
        ObjectLandmarkSpec("spine_bottom", "spine bottom", "upright_profile", (0.0, 0.0, -0.86)),
        ObjectLandmarkSpec("spine_top", "spine top", "upright_profile", (0.0, 0.0, 0.76)),
    ),
}


def object_landmarks_for_shape(shape_type: str) -> Tuple[ObjectLandmarkSpec, ...]:
    """Return the supported landmarks for one shape type."""

    landmarks = LANDMARKS_BY_SHAPE_TYPE.get(str(shape_type), ())
    if not landmarks:
        raise ValueError(f"unsupported landmark correspondence shape type: {shape_type}")
    return tuple(landmarks)


def landmark_world_xyz(
    spec: Mapping[str, Any],
    landmark: ObjectLandmarkSpec,
    *,
    camera: Any,
) -> Tuple[float, float, float]:
    """Project a local object landmark into world coordinates for the active view."""

    x, y, _z = (float(value) for value in spec["world_xyz"])
    raw_base = spec.get("base_xyz", (x, y, 0.0))
    base_z = float(raw_base[2]) if isinstance(raw_base, Sequence) and len(raw_base) >= 3 else 0.0
    width, depth, height = (float(value) for value in spec["dimensions_xyz"])
    local_x, local_y, local_z = (float(value) for value in landmark.local_xyz)
    if str(landmark.coordinate_mode) == "upright_profile":
        return (
            x + float(camera.right[0]) * local_x * width * 0.5,
            y + float(camera.right[1]) * local_x * width * 0.5,
            base_z + (local_z + 1.0) * height * 0.5,
        )
    if str(landmark.coordinate_mode) == "floor_plan":
        return (
            x + local_x * width * 0.5,
            y + local_y * depth * 0.5,
            base_z + max(0.0, min(1.0, local_z)) * height,
        )
    raise ValueError(f"unsupported landmark coordinate mode: {landmark.coordinate_mode}")


__all__ = [
    "LANDMARK_CORRESPONDENCE_SHAPE_TYPES",
    "LANDMARKS_BY_SHAPE_TYPE",
    "ObjectLandmarkSpec",
    "landmark_world_xyz",
    "object_landmarks_for_shape",
]
