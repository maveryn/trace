"""View-relative and camera-depth counting task for a synthetic 3D object scene."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .canvas import render_params_canvas_metadata
from ....core.seed import spawn_rng
from ....core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
)
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from .color_variation import resolve_three_d_object_fill_rgb
from .annotation_geometry import normalize_annotation_bboxes
from .task_support import normalize_unit as _normalize_unit
from .task_support import resolve_axis_variant as _shared_resolve_axis_variant
from .task_support import resolve_count as _shared_resolve_count
from .object_scene import (
    CONTEXT_OBJECT_COLORS,
    NAMED_SMALL_OBJECT_SHAPE_TYPES,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    _RenderParams,
    _bbox_intersection_area,
    _build_projection_frame,
    _camera_yaw_band_for_instance,
    _make_object_spec,
    _min_pairwise,
    _object_reference_points,
    _object_screen_bbox,
    _project_screen,
    _resolve_render_params,
    _sample_camera,
    _sample_shape_dimensions,
    render_object_scene_3d,
)


IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID = "task_three_d__object_scene__image_plane_lateral_relation_count"
CAMERA_DEPTH_RELATION_COUNT_TASK_ID = "task_three_d__object_scene__camera_depth_relation_count"
TASK_ID = IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID
SOURCE_ID = "three_d_object_scene_view_relation_count_source"
SCREEN_SIDE_QUERY_IDS: Tuple[str, ...] = (
    "left_of_reference_in_view_count",
    "right_of_reference_in_view_count",
)
CAMERA_DEPTH_QUERY_IDS: Tuple[str, ...] = (
    "closer_to_camera_than_reference_count",
    "farther_from_camera_than_reference_count",
)
SUPPORTED_QUERY_IDS: Tuple[str, ...] = SCREEN_SIDE_QUERY_IDS + CAMERA_DEPTH_QUERY_IDS
COUNTABLE_SHAPE_TYPES: Tuple[str, ...] = tuple(NAMED_SMALL_OBJECT_SHAPE_TYPES)
VIEW_SCENE_SLOTS: Tuple[Tuple[float, float], ...] = tuple(
    (x, y)
    for y in (-2.42, -1.34, -0.26, 0.82, 1.90)
    for x in (-2.58, -1.48, -0.38, 0.72, 1.82, 2.70)
)
SCREEN_SIDE_AXIS_EXTENT = 3.05
SCREEN_SIDE_DEPTH_EXTENT = 0.52
COUNTABLE_DIMENSION_SCALE = 1.08
SCREEN_SIDE_COUNTABLE_DIMENSION_SCALE = 0.90
MIN_PROJECTED_OBJECT_AREA_PX = 520.0
MAX_PAIRWISE_OVERLAP_PX = 3600.0
MIN_REFERENCE_X_MARGIN_PX = 60.0
MIN_REFERENCE_X_BBOX_GAP_PX = 14.0
MIN_REFERENCE_DEPTH_MARGIN = 0.48


def _uniform_string_probability_map(values: Sequence[str]) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    probability = 1.0 / max(1, len(support))
    return {str(value): float(probability) for value in support}


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_is_readable(bbox: Sequence[float], *, width: int, height: int, min_side_px: float = 18.0) -> bool:
    box_width = float(bbox[2]) - float(bbox[0])
    box_height = float(bbox[3]) - float(bbox[1])
    if box_width < float(min_side_px) or box_height < float(min_side_px):
        return False
    return float(bbox[2]) > 4.0 and float(bbox[3]) > 4.0 and float(bbox[0]) < float(width - 4) and float(bbox[1]) < float(height - 4)


def _screen_bbox(spec: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    raw_bbox = spec.get("screen_bbox_px")
    if not isinstance(raw_bbox, Sequence) or isinstance(raw_bbox, (str, bytes)) or len(raw_bbox) != 4:
        raise ValueError("view-relation object is missing screen_bbox_px")
    return tuple(float(value) for value in raw_bbox)  # type: ignore[return-value]


def _scale_dimensions(dimensions_xyz: Sequence[float], scale: float) -> Tuple[float, float, float]:
    return tuple(round(float(value) * float(scale), 4) for value in dimensions_xyz)  # type: ignore[return-value]


def _make_countable_object(
    *,
    rng,
    object_id: str,
    shape_type: str,
    xy: Tuple[float, float],
    dimension_scale_factor: float = COUNTABLE_DIMENSION_SCALE,
) -> Dict[str, Any]:
    dimensions_xyz, dimension_scale = _sample_shape_dimensions(str(shape_type), object_role="candidate", rng=rng)
    scaled_dimensions = _scale_dimensions(dimensions_xyz, float(dimension_scale_factor))
    spec = _make_object_spec(
        object_id=str(object_id),
        shape_type=str(shape_type),
        object_role="candidate",
        xy=tuple(float(value) for value in xy),
        dimensions_xyz=scaled_dimensions,
        dimension_scale=float(dimension_scale) * float(dimension_scale_factor),
        label=None,
    )
    spec.update(
        {
            "is_answer_candidate": False,
            "is_countable_object": True,
            "is_reference_object": False,
            "matches_query": False,
            "count_role": "distractor",
        }
    )
    spec["fill_rgb"] = [
        int(channel)
        for channel in resolve_three_d_object_fill_rgb(
            spec,
            palette=CONTEXT_OBJECT_COLORS,
            salt=f"{SOURCE_ID}.countable",
            variation_strength=0.32,
        )
    ]
    return spec


def _can_place(candidate: Mapping[str, Any], placed: Sequence[Mapping[str, Any]], *, clearance: float = 0.16) -> bool:
    cx, cy, _cz = (float(value) for value in candidate["world_xyz"])
    for item in placed:
        ix, iy, _iz = (float(value) for value in item["world_xyz"])
        min_distance = float(candidate["footprint_radius"]) + float(item["footprint_radius"]) + float(clearance)
        if math.hypot(float(cx - ix), float(cy - iy)) < float(min_distance):
            return False
    return True


def _sample_unique_shapes(*, rng, object_count: int) -> List[str]:
    shape_pool = [str(shape) for shape in COUNTABLE_SHAPE_TYPES]
    rng.shuffle(shape_pool)
    if int(object_count) > len(shape_pool):
        raise ValueError("view relation count needs no more objects than unique named small shapes")
    return list(shape_pool[: int(object_count)])


def _place_countable_objects(*, rng, shape_types: Sequence[str]) -> List[Dict[str, Any]]:
    slots = list(VIEW_SCENE_SLOTS)
    rng.shuffle(slots)
    placed: List[Dict[str, Any]] = []
    for index, shape_type in enumerate(shape_types):
        for slot_index, (slot_x, slot_y) in enumerate(list(slots)):
            candidate_xy = (
                float(slot_x + rng.uniform(-0.14, 0.14)),
                float(slot_y + rng.uniform(-0.14, 0.14)),
            )
            spec = _make_countable_object(
                rng=rng,
                object_id=f"view_object_{int(index):02d}",
                shape_type=str(shape_type),
                xy=candidate_xy,
            )
            if _can_place(spec, placed):
                placed.append(spec)
                slots.pop(int(slot_index))
                break
        else:
            raise ValueError("could not place enough view-relation 3D objects")
    return list(placed)


def _place_screen_side_countable_objects(
    *,
    rng,
    camera,
    shape_types: Sequence[str],
) -> List[Dict[str, Any]]:
    """Place objects along the camera's screen-right floor axis for clear left/right counts."""

    object_count = len(shape_types)
    if object_count < 3:
        raise ValueError("screen-side relation needs at least three objects")
    right_xy = (float(camera.right[0]), float(camera.right[1]))
    forward_xy = (float(camera.forward[0]), float(camera.forward[1]))
    axis_step = 0.0 if object_count == 1 else (2.0 * float(SCREEN_SIDE_AXIS_EXTENT)) / float(object_count - 1)
    depth_offsets = [float(value) for value in (-SCREEN_SIDE_DEPTH_EXTENT, -0.18, 0.18, SCREEN_SIDE_DEPTH_EXTENT)]
    depth_sequence: List[float] = []
    while len(depth_sequence) < int(object_count):
        shuffled_offsets = list(depth_offsets)
        rng.shuffle(shuffled_offsets)
        depth_sequence.extend(shuffled_offsets)

    placed: List[Dict[str, Any]] = []
    for index, shape_type in enumerate(shape_types):
        axis_position = -float(SCREEN_SIDE_AXIS_EXTENT) + float(index) * float(axis_step)
        depth_position = float(depth_sequence[int(index)]) + rng.uniform(-0.05, 0.05)
        axis_jitter = rng.uniform(-0.035, 0.035)
        candidate_xy = (
            float((axis_position + axis_jitter) * right_xy[0] + depth_position * forward_xy[0]),
            float((axis_position + axis_jitter) * right_xy[1] + depth_position * forward_xy[1]),
        )
        spec = _make_countable_object(
            rng=rng,
            object_id=f"view_object_{int(index):02d}",
            shape_type=str(shape_type),
            xy=candidate_xy,
            dimension_scale_factor=SCREEN_SIDE_COUNTABLE_DIMENSION_SCALE,
        )
        if not _can_place(spec, placed, clearance=0.02):
            raise ValueError("could not place screen-side view-relation objects with enough spacing")
        placed.append(spec)
    return list(placed)


def _finalize_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> List[Dict[str, Any]]:
    finalized_specs: List[Dict[str, Any]] = []
    for spec in specs:
        screen = _project_screen(spec["world_xyz"], camera, frame)
        screen_bbox = _object_screen_bbox(spec, camera, frame, pad_px=8.0)
        finalized = dict(spec)
        finalized.update(
            {
                "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                "screen_bbox_px": [round(float(value), 3) for value in screen_bbox],
                "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_specs.append(finalized)
    return list(finalized_specs)


def _view_is_valid(
    *,
    specs: Sequence[Mapping[str, Any]],
    camera,
    frame,
    render_params: _RenderParams,
) -> bool:
    bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=8.0) for spec in specs]
    if any(not _bbox_is_readable(bbox, width=int(render_params.canvas_width), height=int(render_params.canvas_height)) for bbox in bboxes):
        return False
    if any(_bbox_area(bbox) < MIN_PROJECTED_OBJECT_AREA_PX for bbox in bboxes):
        return False
    for index, bbox_a in enumerate(bboxes):
        for bbox_b in bboxes[index + 1 :]:
            overlap = _bbox_intersection_area(bbox_a, bbox_b)
            if overlap > MAX_PAIRWISE_OVERLAP_PX:
                return False
            if overlap > 0.42 * min(_bbox_area(bbox_a), _bbox_area(bbox_b)):
                return False
    return True


def _select_reference_and_targets(
    specs: Sequence[Mapping[str, Any]],
    *,
    query_id: str,
    target_count: int,
    rng,
) -> Tuple[Dict[str, Any], List[str], Dict[str, bool], float]:
    if str(query_id) in SCREEN_SIDE_QUERY_IDS:
        ordered = sorted(specs, key=lambda spec: (float(spec["screen_xy"][0]), str(spec["object_id"])))
        valid_choices: List[Tuple[Mapping[str, Any], List[Mapping[str, Any]], Dict[str, bool], float]] = []
        for candidate_reference in ordered[1:-1]:
            reference = dict(candidate_reference)
            reference_bbox = _screen_bbox(reference)
            relation_status_by_object_id: Dict[str, bool] = {}
            target_specs: List[Mapping[str, Any]] = []
            min_relation_margin = float("inf")
            left_count = 0
            right_count = 0
            ambiguous = False
            for spec in ordered:
                object_id = str(spec["object_id"])
                if object_id == str(reference["object_id"]):
                    relation_status_by_object_id[object_id] = False
                    continue
                object_bbox = _screen_bbox(spec)
                left_gap = float(reference_bbox[0]) - float(object_bbox[2])
                right_gap = float(object_bbox[0]) - float(reference_bbox[2])
                if left_gap >= MIN_REFERENCE_X_BBOX_GAP_PX:
                    is_target = str(query_id) == "left_of_reference_in_view_count"
                    relation_margin = float(left_gap)
                    left_count += 1
                elif right_gap >= MIN_REFERENCE_X_BBOX_GAP_PX:
                    is_target = str(query_id) == "right_of_reference_in_view_count"
                    relation_margin = float(right_gap)
                    right_count += 1
                else:
                    ambiguous = True
                    break
                min_relation_margin = min(float(min_relation_margin), float(relation_margin))
                relation_status_by_object_id[object_id] = bool(is_target)
                if is_target:
                    target_specs.append(spec)
            if ambiguous or left_count <= 0 or right_count <= 0:
                continue
            if len(target_specs) == int(target_count):
                valid_choices.append((reference, list(target_specs), relation_status_by_object_id, float(min_relation_margin)))
        if not valid_choices:
            raise ValueError("screen-side bbox relation could not satisfy requested target count")
        reference, target_specs, relation_status_by_object_id, min_relation_margin = rng.choice(valid_choices)
        return (
            dict(reference),
            [str(spec["object_id"]) for spec in sorted(target_specs, key=lambda item: str(item["object_id"]))],
            relation_status_by_object_id,
            float(min_relation_margin),
        )

    if str(query_id) in CAMERA_DEPTH_QUERY_IDS:
        ordered = sorted(specs, key=lambda spec: (float(spec["camera_distance"]), str(spec["object_id"])))
        if str(query_id) == "closer_to_camera_than_reference_count":
            reference_index = int(target_count)
        else:
            reference_index = len(ordered) - int(target_count) - 1
        if reference_index <= 0 or reference_index >= len(ordered) - 1:
            raise ValueError("reference would be too close to the depth edge of the object set")

        reference = dict(ordered[reference_index])
        reference_distance = float(reference["camera_distance"])
        relation_status_by_object_id = {}
        target_specs = []
        min_relation_margin = float("inf")
        for spec in ordered:
            object_id = str(spec["object_id"])
            if object_id == str(reference["object_id"]):
                relation_status_by_object_id[object_id] = False
                continue
            distance_delta = float(spec["camera_distance"]) - reference_distance
            min_relation_margin = min(float(min_relation_margin), abs(float(distance_delta)))
            if abs(float(distance_delta)) < MIN_REFERENCE_DEPTH_MARGIN:
                raise ValueError("object camera distance too close to reference camera distance")
            is_target = (
                bool(distance_delta < 0.0)
                if str(query_id) == "closer_to_camera_than_reference_count"
                else bool(distance_delta > 0.0)
            )
            relation_status_by_object_id[object_id] = bool(is_target)
            if is_target:
                target_specs.append(spec)
        if len(target_specs) != int(target_count):
            raise ValueError("camera-depth target count does not match requested target count")
        return (
            reference,
            [str(spec["object_id"]) for spec in sorted(target_specs, key=lambda item: str(item["object_id"]))],
            relation_status_by_object_id,
            float(min_relation_margin),
        )

    raise ValueError(f"unsupported query_id: {query_id}")


def _relation_metadata(query_id: str) -> Dict[str, str]:
    if str(query_id) == "left_of_reference_in_view_count":
        return {
            "view_relation": "left",
            "relation_frame": "image_view",
            "relation_axis": "rendered_bbox_x",
            "count_predicate": "rendered object bbox is fully image-left of the red-boxed reference object in the final image",
        }
    if str(query_id) == "right_of_reference_in_view_count":
        return {
            "view_relation": "right",
            "relation_frame": "image_view",
            "relation_axis": "rendered_bbox_x",
            "count_predicate": "rendered object bbox is fully image-right of the red-boxed reference object in the final image",
        }
    if str(query_id) == "closer_to_camera_than_reference_count":
        return {
            "view_relation": "closer_to_camera",
            "relation_frame": "camera_distance",
            "relation_axis": "camera_distance",
            "count_predicate": "object camera distance is smaller than the red-boxed reference object's camera distance",
        }
    if str(query_id) == "farther_from_camera_than_reference_count":
        return {
            "view_relation": "farther_from_camera",
            "relation_frame": "camera_distance",
            "relation_axis": "camera_distance",
            "count_predicate": "object camera distance is larger than the red-boxed reference object's camera distance",
        }
    else:
        raise ValueError(f"unsupported query_id: {query_id}")


def _validate_rendered_screen_side_relation(
    *,
    rendered_bboxes: Mapping[str, Sequence[float]],
    reference_object_id: str,
    target_object_ids: Sequence[str],
    query_id: str,
) -> None:
    reference_bbox = rendered_bboxes[str(reference_object_id)]
    target_set = {str(object_id) for object_id in target_object_ids}
    for object_id, bbox in rendered_bboxes.items():
        object_id = str(object_id)
        if object_id == str(reference_object_id):
            continue
        left_gap = float(reference_bbox[0]) - float(bbox[2])
        right_gap = float(bbox[0]) - float(reference_bbox[2])
        if left_gap >= MIN_REFERENCE_X_BBOX_GAP_PX:
            expected = str(query_id) == "left_of_reference_in_view_count"
        elif right_gap >= MIN_REFERENCE_X_BBOX_GAP_PX:
            expected = str(query_id) == "right_of_reference_in_view_count"
        else:
            raise ValueError("rendered bbox is too close to or overlaps the reference bbox in x")
        if bool(object_id in target_set) != bool(expected):
            raise ValueError("rendered bbox side relation does not match dataset target ids")


def _build_view_relation_count_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    target_count: int,
    object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
) -> Dict[str, Any]:
    rng = spawn_rng(int(instance_seed), f"{SOURCE_ID}.dataset")
    selected_camera_yaw_band = _camera_yaw_band_for_instance(int(instance_seed))
    for _attempt in range(520):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        shape_types = _sample_unique_shapes(rng=rng, object_count=int(object_count))
        object_specs = (
            _place_screen_side_countable_objects(rng=rng, camera=camera, shape_types=shape_types)
            if str(query_id) in SCREEN_SIDE_QUERY_IDS
            else _place_countable_objects(rng=rng, shape_types=shape_types)
        )
        prompt_name_counts = Counter(str(spec["prompt_name"]) for spec in object_specs)
        if any(int(count) != 1 for count in prompt_name_counts.values()):
            continue
        reference_points = [point for spec in object_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        if not _view_is_valid(specs=object_specs, camera=camera, frame=frame, render_params=render_params):
            continue
        finalized_specs = _finalize_specs(object_specs, camera=camera, frame=frame)
        try:
            reference_spec, target_object_ids, relation_status_by_object_id, min_relation_margin = _select_reference_and_targets(
                finalized_specs,
                query_id=str(query_id),
                target_count=int(target_count),
                rng=rng,
            )
        except ValueError:
            continue
        finalized_by_id = {str(spec["object_id"]): dict(spec) for spec in finalized_specs}
        for object_id, is_target in relation_status_by_object_id.items():
            finalized_by_id[str(object_id)]["matches_query"] = bool(is_target)
            finalized_by_id[str(object_id)]["count_role"] = "target" if bool(is_target) else "distractor"
        finalized_by_id[str(reference_spec["object_id"])]["is_reference_object"] = True
        finalized_by_id[str(reference_spec["object_id"])]["count_role"] = "reference"
        finalized_specs = sorted(finalized_by_id.values(), key=lambda spec: str(spec["object_id"]))
        distances = [float(spec["camera_distance"]) for spec in finalized_specs]
        relation_meta = _relation_metadata(str(query_id))
        solver_trace = {
            "count_predicate": str(relation_meta["count_predicate"]),
            "relation_axis": str(relation_meta["relation_axis"]),
            "relation_frame": str(relation_meta["relation_frame"]),
            "view_relation": str(relation_meta["view_relation"]),
            "reference_object_id": str(reference_spec["object_id"]),
            "reference_object_name": str(reference_spec["prompt_name"]),
            "reference_shape_type": str(reference_spec["shape_type"]),
            "reference_screen_xy": list(reference_spec["screen_xy"]),
            "reference_camera_distance": float(reference_spec["camera_distance"]),
            "reference_prompt_name_count": int(prompt_name_counts[str(reference_spec["prompt_name"])]),
            "target_count": int(target_count),
            "target_object_ids": list(target_object_ids),
            "view_relation_status_by_object_id": dict(sorted(relation_status_by_object_id.items())),
            "minimum_reference_relation_margin": round(float(min_relation_margin), 4),
            "minimum_pairwise_camera_distance_margin": round(float(_min_pairwise(distances)), 4),
            "unique_integer_answer": True,
        }
        if str(relation_meta["relation_axis"]) == "rendered_bbox_x":
            solver_trace["screen_axis"] = "x"
            solver_trace["minimum_reference_bbox_x_gap_px"] = round(float(min_relation_margin), 3)
        else:
            solver_trace["minimum_reference_camera_distance_margin"] = round(float(min_relation_margin), 4)
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "object_count": int(object_count),
            "countable_object_count": int(object_count),
            "target_count": int(target_count),
            "answer_value": int(target_count),
            "view_relation": str(relation_meta["view_relation"]),
            "view_relation_frame": str(relation_meta["relation_frame"]),
            "view_relation_axis": str(relation_meta["relation_axis"]),
            "reference_object_id": str(reference_spec["object_id"]),
            "reference_object_name": str(reference_spec["prompt_name"]),
            "reference_shape_type": str(reference_spec["shape_type"]),
            "reference_prompt_name_count": int(prompt_name_counts[str(reference_spec["prompt_name"])]),
            "target_object_ids": list(target_object_ids),
            "object_specs": list(finalized_specs),
            "point_specs": list(finalized_specs),
            "context_object_specs": [],
            "view_relation_status_by_object_id": dict(sorted(relation_status_by_object_id.items())),
            "prompt_name_counts": {str(key): int(value) for key, value in sorted(prompt_name_counts.items())},
            "camera": _camera_record(camera, yaw_band=selected_camera_yaw_band),
            "projection_frame": _frame_record(frame),
            "solver_trace": dict(solver_trace),
        }
    raise ValueError("could not construct a valid 3D view-relation count scene")


def _camera_record(camera, *, yaw_band: Sequence[float]) -> Dict[str, Any]:
    return {
        "camera_position": [round(float(value), 4) for value in camera.camera_position],
        "target": [round(float(value), 4) for value in camera.target],
        "yaw_degrees": round(float(camera.yaw_degrees), 4),
        "yaw_band_degrees": [round(float(value), 4) for value in yaw_band],
        "pitch_degrees": round(float(camera.pitch_degrees), 4),
        "distance": round(float(camera.distance), 4),
        "right": [round(float(value), 5) for value in camera.right],
        "up": [round(float(value), 5) for value in camera.up],
        "forward": [round(float(value), 5) for value in camera.forward],
    }


def _frame_record(frame) -> Dict[str, Any]:
    return {
        "scale": round(float(frame.scale), 5),
        "center_x": round(float(frame.center_x), 3),
        "center_y": round(float(frame.center_y), 3),
        "normalized_center_u": round(float(frame.normalized_center_u), 6),
        "normalized_center_v": round(float(frame.normalized_center_v), 6),
    }




_SCENE_DEFAULTS = get_scene_defaults("three_d", SCENE_ID)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _resolve_task_defaults(task_id: str) -> Tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
        task_id=str(task_id),
    )
    visual_defaults = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
    return dict(gen_defaults), dict(render_defaults), dict(prompt_defaults), dict(visual_defaults)


class _ThreeDSpatialViewRelationCountBase:
    """Count objects by image-side or camera-depth relation to a named small reference."""

    task_id = TASK_ID
    supported_query_ids: Tuple[str, ...] = SUPPORTED_QUERY_IDS
    domain = "three_d"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        task_id = str(self.task_id)
        last_error: Exception | None = None
        axis_seed = int(instance_seed)
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt_index == 0
                else int(spawn_rng(int(instance_seed), f"{task_id}.attempt_seed.{attempt_index}").randrange(1, 2**62))
            )
            try:
                return self._generate_once(int(attempt_seed), params=params, axis_seed=axis_seed)
            except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
                last_error = exc
        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}")

    def _generate_once(self, instance_seed: int, *, params: Dict[str, Any], axis_seed: int | None = None) -> TaskOutput:
        task_id = str(self.task_id)
        sampling_seed = int(axis_seed) if axis_seed is not None else int(instance_seed)
        gen_defaults, render_defaults, prompt_defaults_config, _visual_defaults = _resolve_task_defaults(task_id)
        query_id, query_probabilities = _shared_resolve_axis_variant(
            params,
            task_id=task_id,
            gen_defaults=gen_defaults,
            instance_seed=int(sampling_seed),
            supported_variants=self.supported_query_ids,
            explicit_key="query_id",
            weights_key="query_id_weights",
            balance_flag_key="balanced_query_id_sampling",
            axis_namespace="query_id",
        )
        scene_variant, scene_probabilities = _shared_resolve_axis_variant(
            params,
            task_id=task_id,
            gen_defaults=gen_defaults,
            instance_seed=int(sampling_seed),
            supported_variants=SUPPORTED_SCENE_VARIANTS,
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
            balance_flag_key="balanced_scene_variant_sampling",
            axis_namespace="scene_variant",
        )
        object_count, object_count_probabilities = _shared_resolve_count(
            params,
            task_id=task_id,
            gen_defaults=gen_defaults,
            instance_seed=int(sampling_seed),
            prefix="object_count",
            minimum_default=int(group_default(gen_defaults, "object_count_min", 10)),
            maximum_default=int(group_default(gen_defaults, "object_count_max", 13)),
            lower=7,
            upper=16,
        )
        target_count, target_count_probabilities = _shared_resolve_count(
            params,
            task_id=task_id,
            gen_defaults=gen_defaults,
            instance_seed=int(sampling_seed),
            prefix="target_count",
            minimum_default=int(group_default(gen_defaults, "target_count_min", 2)),
            maximum_default=int(group_default(gen_defaults, "target_count_max", 5)),
            lower=1,
            upper=max(1, min(6, int(object_count) - 3)),
        )

        render_params = _resolve_render_params(
            params,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.canvas",
        )
        dataset = _build_view_relation_count_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            target_count=int(target_count),
            object_count=int(object_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
        )
        background, background_meta = make_background_canvas(
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            instance_seed=int(instance_seed),
            params=params,
            default_config=_BACKGROUND_DEFAULTS,
        )
        rendered = render_object_scene_3d(
            background,
            dataset=dataset,
            render_params=render_params,
            draw_candidate_labels=False,
            compute_single_annotation=False,
            highlight_object_ids=[str(dataset["reference_object_id"])],
        )
        if str(query_id) in SCREEN_SIDE_QUERY_IDS:
            _validate_rendered_screen_side_relation(
                rendered_bboxes=rendered.object_bboxes_px,
                reference_object_id=str(dataset["reference_object_id"]),
                target_object_ids=[str(object_id) for object_id in dataset["target_object_ids"]],
                query_id=str(query_id),
            )
        image, post_noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=_NOISE_DEFAULTS,
        )
        target_object_ids = [str(object_id) for object_id in dataset["target_object_ids"]]
        raw_annotation_bboxes = [list(rendered.object_bboxes_px[str(object_id)]) for object_id in target_object_ids]
        annotation_bboxes, annotation_bbox_normalization = normalize_annotation_bboxes(
            raw_annotation_bboxes,
            bounds_px=[0.0, 0.0, float(image.width), float(image.height)],
        )
        annotation_payload = bbox_set_annotation_artifacts(annotation_bboxes)
        annotation_bbox_by_object_id = {
            str(object_id): list(bbox)
            for object_id, bbox in zip(target_object_ids, annotation_bboxes)
        }
        raw_annotation_bbox_by_object_id = {
            str(object_id): list(bbox)
            for object_id, bbox in zip(target_object_ids, raw_annotation_bboxes)
        }

        prompt_defaults = required_group_defaults(
            prompt_defaults_config,
            (
                "bundle_id",
                "scene_key",
                "task_key",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            query_key=str(query_id),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "reference_name": str(dataset["reference_object_name"]),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        answer_value = int(dataset["answer_value"])
        answer_gt = TypedValue(type="integer", value=int(answer_value))
        annotation_gt = annotation_payload.annotation_gt
        solver_trace = dict(dataset["solver_trace"])

        trace_payload = {
            "scene_ir": {
                "scene_kind": "three_d_object_scene_view_relation_count",
                "entities": [dict(entity) for entity in rendered.entities],
                "relations": {
                    "scene_variant": str(scene_variant),
                    "object_count": int(dataset["object_count"]),
                    "countable_object_count": int(dataset["countable_object_count"]),
                    "target_count": int(answer_value),
                    "target_object_ids": list(target_object_ids),
                    "view_relation": str(dataset["view_relation"]),
                    "view_relation_frame": str(dataset["view_relation_frame"]),
                    "view_relation_axis": str(dataset["view_relation_axis"]),
                    "reference_object_id": str(dataset["reference_object_id"]),
                    "reference_object_name": str(dataset["reference_object_name"]),
                    "reference_shape_type": str(dataset["reference_shape_type"]),
                    "view_relation_status_by_object_id": dict(dataset["view_relation_status_by_object_id"]),
                },
            },
            "query_spec": {
                "query_id": str(query_id),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(query_id),
                    "query_id_probabilities": dict(query_probabilities),
                    "scene_variant": str(scene_variant),
                    "scene_variant_probabilities": dict(scene_probabilities),
                    "object_count": int(object_count),
                    "object_count_probabilities": dict(object_count_probabilities),
                    "target_count": int(answer_value),
                    "target_count_probabilities": dict(target_count_probabilities),
                    "reference_shape_type": str(dataset["reference_shape_type"]),
                    "reference_shape_probabilities": _uniform_string_probability_map(COUNTABLE_SHAPE_TYPES),
                },
            },
            "render_spec": {
                "canvas_width": int(render_params.canvas_width),
                "canvas_height": int(render_params.canvas_height),
                "scene_canvas_preset": str(render_params.canvas_preset),
                "scene_canvas_width": int(render_params.canvas_width),
                "scene_canvas_height": int(render_params.canvas_height),
                "scene_canvas_policy": str(render_params.canvas_policy),
                **render_params_canvas_metadata(render_params),
                "final_canvas_width": int(image.width),
                "final_canvas_height": int(image.height),
                "final_canvas_pixels": int(image.width) * int(image.height),
                "coord_space": "pixel",
                "scene_variant": str(scene_variant),
                "background_style": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
                "camera": dict(dataset["camera"]),
                "projection_frame": dict(dataset["projection_frame"]),
                "room_extent": float(render_params.room_extent),
                "full_bleed_floor": bool(render_params.full_bleed_floor),
                "annotation_bbox_normalization": dict(annotation_bbox_normalization),
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": list(rendered.scene_bbox_px),
                "room_bbox_px": list(rendered.room_bbox_px),
                "object_bboxes_px": dict(rendered.object_bboxes_px),
                "object_centers_px": dict(rendered.object_centers_px),
                "target_object_bboxes_px": {
                    str(object_id): list(annotation_bbox_by_object_id[str(object_id)])
                    for object_id in target_object_ids
                },
                "target_object_raw_bboxes_px": {
                    str(object_id): list(raw_annotation_bbox_by_object_id[str(object_id)])
                    for object_id in target_object_ids
                },
                "target_object_centers_px": {
                    str(object_id): list(rendered.object_centers_px[str(object_id)])
                    for object_id in target_object_ids
                },
                "reference_object_bbox_px": list(rendered.object_bboxes_px[str(dataset["reference_object_id"])]),
                "reference_object_center_px": list(rendered.object_centers_px[str(dataset["reference_object_id"])]),
                "reference_highlight_entity_id": f"red_reference_box_{str(dataset['reference_object_id'])}",
                "annotation_raw_bboxes_px": [list(bbox) for bbox in raw_annotation_bboxes],
                "annotation_bboxes_px": [list(bbox) for bbox in annotation_bboxes],
                "annotation_bbox_normalization": dict(annotation_bbox_normalization),
            },
            "execution_trace": {
                "query_id": str(query_id),
                "scene_variant": str(scene_variant),
                "object_count": int(dataset["object_count"]),
                "countable_object_count": int(dataset["countable_object_count"]),
                "target_count": int(answer_value),
                "answer_value": int(answer_value),
                "view_relation": str(dataset["view_relation"]),
                "view_relation_frame": str(dataset["view_relation_frame"]),
                "view_relation_axis": str(dataset["view_relation_axis"]),
                "target_object_ids": list(target_object_ids),
                "target_object_bboxes_px": dict(annotation_bbox_by_object_id),
                "target_object_raw_bboxes_px": dict(raw_annotation_bbox_by_object_id),
                "reference_object_id": str(dataset["reference_object_id"]),
                "reference_object_name": str(dataset["reference_object_name"]),
                "reference_shape_type": str(dataset["reference_shape_type"]),
                "reference_prompt_name_count": int(dataset["reference_prompt_name_count"]),
                "view_relation_status_by_object_id": dict(dataset["view_relation_status_by_object_id"]),
                "object_specs": [dict(spec) for spec in dataset["object_specs"]],
                "prompt_name_counts": dict(dataset["prompt_name_counts"]),
                "camera": dict(dataset["camera"]),
                "projection_frame": dict(dataset["projection_frame"]),
                "question_format": str(query_id),
                "solver_trace": dict(solver_trace),
            },
            "witness_symbolic": {
                "type": "counted_view_relation_object_set",
                "object_ids": list(target_object_ids),
                "reference_object_id": str(dataset["reference_object_id"]),
                "view_relation": str(dataset["view_relation"]),
                "view_relation_frame": str(dataset["view_relation_frame"]),
                "answer_value": int(answer_value),
            },
            "projected_annotation": dict(annotation_payload.projected_annotation),
            "background": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
        }

        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )


__all__ = [
    "CAMERA_DEPTH_QUERY_IDS",
    "CAMERA_DEPTH_RELATION_COUNT_TASK_ID",
    "IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID",
    "SCREEN_SIDE_QUERY_IDS",
    "MIN_REFERENCE_X_BBOX_GAP_PX",
    "MIN_REFERENCE_DEPTH_MARGIN",
    "MIN_REFERENCE_X_MARGIN_PX",
    "_ThreeDSpatialViewRelationCountBase",
]
