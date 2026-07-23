"""Marked-point vertical relation task for a synthetic 3D object scene."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
)
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from ..shared.object_scene_marked_point_output import build_marked_point_object_scene_output as _build_marked_point_object_scene_output
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.object_scene import (
    NAMED_SMALL_OBJECT_SHAPE_TYPES,
    POINT_LABELS,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    _RenderParams,
    _bbox_intersection_area,
    _build_projection_frame,
    _camera_yaw_band_for_instance,
    _object_reference_points,
    _object_screen_bbox,
    _project_screen,
    _resolve_render_params,
    _sample_camera,
    _sample_scene_object_specs,
)
from .shared.annotations import draw_marked_points as _render_marked_point_overlay
from .shared.labels import assign_answer_label as _assign_answer_label
from .shared.labels import bbox_union as _bbox_union


TASK_ID = "task_three_d__object_scene__marked_point_vertical_relation_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("directly_above_reference",)
_REQUESTED_REFERENCE_SHAPE_TYPES: Tuple[str, ...] = (
    "sphere",
    "cube",
    "cylinder",
    "cone",
    "pyramid",
    "torus",
    "half_cylinder",
    "cup",
    "bottle",
    "drum",
    "dice",
    "cactus",
    "helmet",
    "hat",
    "bell",
    "trophy",
    "lantern",
    "candle",
    "goblet",
    "flask",
    "clock",
    "apple",
)
REFERENCE_SHAPE_TYPES: Tuple[str, ...] = tuple(
    shape for shape in _REQUESTED_REFERENCE_SHAPE_TYPES if shape in set(NAMED_SMALL_OBJECT_SHAPE_TYPES)
)
MIN_DISTRACTOR_REFERENCE_XY_OFFSET = 0.48
MIN_MARKER_SCREEN_SEPARATION_PX = 78.0
MIN_ANSWER_SCREEN_ABOVE_REFERENCE_PX = 26.0
MIN_READABLE_OBJECT_AREA_PX = 520.0
MAX_PAIRWISE_OBJECT_OVERLAP_PX = 4200.0


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_is_readable(bbox: Sequence[float], *, width: int, height: int, min_side_px: float = 17.0) -> bool:
    box_width = float(bbox[2]) - float(bbox[0])
    box_height = float(bbox[3]) - float(bbox[1])
    if box_width < float(min_side_px) or box_height < float(min_side_px):
        return False
    return float(bbox[2]) > 4.0 and float(bbox[3]) > 4.0 and float(bbox[0]) < float(width - 4) and float(bbox[1]) < float(height - 4)


def _demote_small_context_spec(spec: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    updated = dict(spec)
    shape_type = str(updated["shape_type"])
    updated["object_id"] = f"vertical_context_{int(index):02d}_{shape_type}"
    updated["object_role"] = "small_context"
    updated["is_answer_candidate"] = False
    for key in ("point_id", "point_label", "object_label"):
        updated.pop(key, None)
    return updated


def _finalize_object_specs(
    specs: Sequence[Mapping[str, Any]],
    *,
    camera,
    frame,
) -> List[Dict[str, Any]]:
    finalized_specs: List[Dict[str, Any]] = []
    for spec in specs:
        screen = _project_screen(spec["world_xyz"], camera, frame)
        finalized = dict(spec)
        finalized.update(
            {
                "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_specs.append(finalized)
    return finalized_specs


def _sample_small_object_specs(*, rng, object_count: int) -> List[Dict[str, Any]]:
    small_specs, large_specs = _sample_scene_object_specs(
        rng=rng,
        candidate_count=int(object_count),
        context_object_count=0,
    )
    if large_specs:
        raise ValueError("vertical marked-point scene should not include large context props")
    return [_demote_small_context_spec(spec, index=index) for index, spec in enumerate(small_specs)]


def _choose_reference_spec(*, rng, specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    candidates = [dict(spec) for spec in specs if str(spec["shape_type"]) in set(REFERENCE_SHAPE_TYPES)]
    if not candidates:
        raise ValueError("no suitable vertical-reference object sampled")
    rng.shuffle(candidates)
    return dict(candidates[0])


def _make_marker_record(
    *,
    marker_id: str,
    world_xyz: Sequence[float],
    surface_kind: str,
    attached_object_id: str | None,
) -> Dict[str, Any]:
    return {
        "marker_id": str(marker_id),
        "surface_kind": str(surface_kind),
        "attached_object_id": None if attached_object_id is None else str(attached_object_id),
        "world_xyz": [round(float(value), 4) for value in world_xyz],
    }


def _sample_vertical_marker_records(
    *,
    rng,
    reference_spec: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    point_count: int,
    room_extent: float,
    room_height: float,
) -> Tuple[List[Dict[str, Any]], str, float]:
    """Sample vertically related marked points while keeping projected labels readable and relation membership unique."""
    if int(point_count) != 6:
        raise ValueError("vertical marked-point task expects exactly six point options")

    ref_x, ref_y, ref_base_z = (float(value) for value in reference_spec["base_xyz"])
    ref_height = float(reference_spec["dimensions_xyz"][2])
    ref_top_z = float(ref_base_z + ref_height)
    answer_gap = float(rng.uniform(0.58, 0.84))
    answer_marker_id = "raw_vertical_answer"
    records = [
        _make_marker_record(
            marker_id=answer_marker_id,
            world_xyz=(ref_x, ref_y, min(float(room_height) - 0.20, ref_top_z + answer_gap)),
            surface_kind="directly_above_reference",
            attached_object_id=str(reference_spec["object_id"]),
        )
    ]
    extent = min(2.84, max(2.10, float(room_extent) - 0.34))
    max_z = max(1.25, float(room_height) - 0.22)
    min_xy_offset = float("inf")
    attempts = 0
    other_objects = [dict(spec) for spec in objects if str(spec["object_id"]) != str(reference_spec["object_id"])]

    while len(records) < int(point_count) and attempts < 520:
        attempts += 1
        use_other_object = bool(other_objects) and rng.random() < 0.36
        if use_other_object:
            obj = dict(other_objects[int(rng.randrange(len(other_objects)))])
            base_x, base_y, base_z = (float(value) for value in obj["base_xyz"])
            obj_top = float(base_z + float(obj["dimensions_xyz"][2]))
            x = base_x + rng.uniform(-0.13, 0.13)
            y = base_y + rng.uniform(-0.13, 0.13)
            z = min(max_z, obj_top + rng.uniform(0.34, 0.80))
        else:
            angle = float(rng.uniform(0.0, math.tau))
            radius = float(rng.uniform(0.62, 1.46))
            x = max(-extent, min(extent, ref_x + math.cos(angle) * radius))
            y = max(-extent, min(extent, ref_y + math.sin(angle) * radius))
            z = min(max_z, ref_top_z + rng.uniform(0.18, 1.16))
        xy_offset = math.hypot(float(x - ref_x), float(y - ref_y))
        if xy_offset < MIN_DISTRACTOR_REFERENCE_XY_OFFSET:
            continue
        if any(math.hypot(float(x - float(record["world_xyz"][0])), float(y - float(record["world_xyz"][1]))) < 0.34 for record in records):
            continue
        min_xy_offset = min(float(min_xy_offset), float(xy_offset))
        records.append(
            _make_marker_record(
                marker_id=f"raw_vertical_distractor_{len(records)}",
                world_xyz=(x, y, max(0.12, z)),
                surface_kind="offset_floating_point",
                attached_object_id=None,
            )
        )

    if len(records) != int(point_count):
        raise ValueError("could not place six vertical-relation marked points")
    return records, str(answer_marker_id), float(min_xy_offset)


def _finalize_marker_records(
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
            58.0 <= x <= float(render_params.canvas_width) - 58.0
            and 58.0 <= y <= float(render_params.canvas_height) - 58.0
        ):
            raise ValueError("marked point projects outside the readable image area")
        finalized = dict(record)
        finalized.update(
            {
                "screen_xy": [round(float(x), 3), round(float(y), 3)],
                "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                "camera_distance": round(float(screen[7]), 4),
            }
        )
        finalized_markers.append(finalized)
    return finalized_markers


def _validate_projection(
    *,
    finalized_objects: Sequence[Mapping[str, Any]],
    finalized_markers: Sequence[Mapping[str, Any]],
    reference_spec: Mapping[str, Any],
    answer_marker_id: str,
    camera,
    frame,
    render_params: _RenderParams,
) -> Tuple[Dict[str, List[float]], float]:
    """Validate projected marked-point geometry so the vertical relation is visually legible and unambiguous."""
    object_bboxes_by_id = {
        str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=10.0)
        for spec in finalized_objects
    }
    if any(
        not _bbox_is_readable(
            bbox,
            width=int(render_params.canvas_width),
            height=int(render_params.canvas_height),
        )
        for bbox in object_bboxes_by_id.values()
    ):
        raise ValueError("small object bbox is not readable")
    if any(_bbox_area(bbox) < MIN_READABLE_OBJECT_AREA_PX for bbox in object_bboxes_by_id.values()):
        raise ValueError("small object projected area is too small")
    object_bboxes = list(object_bboxes_by_id.values())
    if any(
        _bbox_intersection_area(a, b) > MAX_PAIRWISE_OBJECT_OVERLAP_PX
        for index, a in enumerate(object_bboxes)
        for b in object_bboxes[index + 1 :]
    ):
        raise ValueError("small object overlap is too high")

    marker_centers = [(float(item["screen_xy"][0]), float(item["screen_xy"][1])) for item in finalized_markers]
    if any(
        math.hypot(a[0] - b[0], a[1] - b[1]) < MIN_MARKER_SCREEN_SEPARATION_PX
        for index, a in enumerate(marker_centers)
        for b in marker_centers[index + 1 :]
    ):
        raise ValueError("marked point centers are too close")

    reference_id = str(reference_spec["object_id"])
    reference_bbox = object_bboxes_by_id[reference_id]
    answer_marker = next(marker for marker in finalized_markers if str(marker["marker_id"]) == str(answer_marker_id))
    answer_x, answer_y = (float(answer_marker["screen_xy"][0]), float(answer_marker["screen_xy"][1]))
    reference_center_x = 0.5 * (float(reference_bbox[0]) + float(reference_bbox[2]))
    if answer_y > float(reference_bbox[1]) - MIN_ANSWER_SCREEN_ABOVE_REFERENCE_PX:
        raise ValueError("answer marker is not visibly above the named reference")
    if abs(float(answer_x - reference_center_x)) > max(80.0, 0.68 * (float(reference_bbox[2]) - float(reference_bbox[0]))):
        raise ValueError("answer marker is not visually aligned with the named reference")

    marker_radius = 24.0
    for marker in finalized_markers:
        marker_bbox = [
            float(marker["screen_xy"][0]) - marker_radius,
            float(marker["screen_xy"][1]) - marker_radius,
            float(marker["screen_xy"][0]) + marker_radius,
            float(marker["screen_xy"][1]) + marker_radius,
        ]
        for object_id, object_bbox in object_bboxes_by_id.items():
            if str(marker["marker_id"]) == str(answer_marker_id) and str(object_id) == reference_id:
                continue
            if _bbox_intersection_area(marker_bbox, object_bbox) > 1250.0:
                raise ValueError("marked point overlaps an unrelated object too much")
    return object_bboxes_by_id, float(reference_center_x)


def _build_vertical_relation_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    point_count: int,
    object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build a marked-point vertical relation scene with one target point directly above the referenced point."""
    if str(query_id) != "directly_above_reference":
        raise ValueError(f"unsupported query_id: {query_id}")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )

    for _attempt in range(620):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        object_specs = _sample_small_object_specs(rng=rng, object_count=int(object_count))
        prompt_name_counts = Counter(str(spec["prompt_name"]) for spec in object_specs)
        if any(int(count) != 1 for count in prompt_name_counts.values()):
            continue
        try:
            reference_spec = _choose_reference_spec(rng=rng, specs=object_specs)
            marker_records, answer_marker_id, min_distractor_xy_offset = _sample_vertical_marker_records(
                rng=rng,
                reference_spec=reference_spec,
                objects=object_specs,
                point_count=int(point_count),
                room_extent=float(render_params.room_extent),
                room_height=float(render_params.room_height),
            )
        except ValueError:
            continue

        reference_points = [
            *(point for spec in object_specs for point in _object_reference_points(spec)),
            *(tuple(float(value) for value in record["world_xyz"]) for record in marker_records),
        ]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_objects = _finalize_object_specs(object_specs, camera=camera, frame=frame)
        reference_spec = next(spec for spec in finalized_objects if str(spec["object_id"]) == str(reference_spec["object_id"]))
        try:
            finalized_markers = _finalize_marker_records(
                marker_records,
                camera=camera,
                frame=frame,
                render_params=render_params,
            )
            object_bboxes_by_id, reference_center_x = _validate_projection(
                finalized_objects=finalized_objects,
                finalized_markers=finalized_markers,
                reference_spec=reference_spec,
                answer_marker_id=str(answer_marker_id),
                camera=camera,
                frame=frame,
                render_params=render_params,
            )
        except ValueError:
            continue

        relabeled_markers = _assign_answer_label(
            records=finalized_markers,
            answer_marker_id=str(answer_marker_id),
            point_count=int(point_count),
            answer_label_index=int(instance_seed),
            rng=rng,
        )
        answer_marker = next(marker for marker in relabeled_markers if str(marker["marker_id"]) == str(answer_marker_id))
        reference_id = str(reference_spec["object_id"])
        ref_x, ref_y, _ref_z = (float(value) for value in reference_spec["base_xyz"])
        xy_offsets_by_label = {
            str(marker["point_label"]): round(
                math.hypot(float(marker["world_xyz"][0]) - ref_x, float(marker["world_xyz"][1]) - ref_y),
                4,
            )
            for marker in relabeled_markers
        }
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "object_count": int(object_count),
            "context_object_count": int(object_count),
            "point_specs": [],
            "context_object_specs": sorted(finalized_objects, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted(finalized_objects, key=lambda spec: str(spec["object_id"])),
            "marked_points": list(relabeled_markers),
            "answer_label": str(answer_marker["point_label"]),
            "answer_point_id": str(answer_marker["point_id"]),
            "answer_marker_id": str(answer_marker["marker_id"]),
            "answer_point_px": [round(float(value), 3) for value in answer_marker["screen_xy"]],
            "reference_object_id": str(reference_id),
            "reference_object_name": str(reference_spec["prompt_name"]),
            "reference_shape_type": str(reference_spec["shape_type"]),
            "reference_prompt_name_count": int(prompt_name_counts[str(reference_spec["prompt_name"])]),
            "reference_bbox_px": [round(float(value), 3) for value in object_bboxes_by_id[str(reference_id)]],
            "camera": {
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
            "projection_frame": {
                "scale": round(float(frame.scale), 5),
                "center_x": round(float(frame.center_x), 3),
                "center_y": round(float(frame.center_y), 3),
                "normalized_center_u": round(float(frame.normalized_center_u), 6),
                "normalized_center_v": round(float(frame.normalized_center_v), 6),
            },
            "solver_trace": {
                "relation": "directly_above_reference",
                "relation_frame": "world_vertical_z",
                "reference_object_id": str(reference_id),
                "reference_object_name": str(reference_spec["prompt_name"]),
                "reference_shape_type": str(reference_spec["shape_type"]),
                "reference_world_xy": [round(float(ref_x), 4), round(float(ref_y), 4)],
                "reference_screen_center_x": round(float(reference_center_x), 3),
                "answer_marker_id": str(answer_marker["marker_id"]),
                "answer_label": str(answer_marker["point_label"]),
                "xy_offsets_from_reference_by_label": dict(sorted(xy_offsets_by_label.items())),
                "minimum_distractor_reference_xy_offset": round(float(min_distractor_xy_offset), 4),
                "unique_directly_above_answer": True,
            },
        }
    raise ValueError("could not construct a valid 3D marked-point vertical relation scene")




_SCENE_DEFAULTS = get_scene_defaults("three_d", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


@register_task
class ThreeDSpatialMarkedPointVerticalRelationLabelTask:
    """Choose the marked point directly above a uniquely named small reference object."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    domain = "three_d"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        last_error: Exception | None = None
        camera_yaw_band = _camera_yaw_band_for_instance(int(instance_seed))
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt_index == 0
                else int(spawn_rng(int(instance_seed), f"{TASK_ID}.attempt_seed.{attempt_index}").randrange(1, 2**62))
            )
            try:
                return self._generate_once(int(attempt_seed), params=params, camera_yaw_band=camera_yaw_band)
            except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
                last_error = exc
        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}")

    def _generate_once(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        camera_yaw_band: Tuple[float, float] | None = None,
    ) -> TaskOutput:
        """Generate one vertical-relation instance from the accepted point layout and scalar answer annotation."""
        query_id, query_probabilities = _shared_resolve_axis_variant(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            supported_variants=SUPPORTED_QUERY_IDS,
            explicit_key="query_id",
            weights_key="query_id_weights",
            balance_flag_key="balanced_query_id_sampling",
            axis_namespace="query_id",
        )
        scene_variant, scene_probabilities = _shared_resolve_axis_variant(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            supported_variants=SUPPORTED_SCENE_VARIANTS,
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
            balance_flag_key="balanced_scene_variant_sampling",
            axis_namespace="scene_variant",
        )
        point_count, point_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="point_count",
            default_min=6,
            default_max=6,
            lower=6,
            upper=6,
        )
        object_count, object_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="object_count",
            default_min=8,
            default_max=8,
            lower=6,
            upper=11,
        )
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_vertical_relation_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            object_count=int(object_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
            camera_yaw_band=camera_yaw_band,
        )
        return _build_marked_point_object_scene_output(
            objective_name=TASK_ID,
            task_domain=self.domain,
            instance_seed=int(instance_seed),
            params=params,
            dataset=dataset,
            branch_key=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            render_params=render_params,
            prompt_defaults_config=_PROMPT_DEFAULTS,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            query_probabilities=query_probabilities,
            scene_probabilities=scene_probabilities,
            point_count_probabilities=point_count_probabilities,
            dynamic_slots={"reference_name": str(dataset["reference_object_name"])},
            scene_kind="three_d_object_scene_marked_point_vertical_relation",
            count_params={
                "object_count": int(object_count),
                "object_count_probabilities": dict(object_count_probabilities),
                "reference_shape_type": str(dataset["reference_shape_type"]),
                "reference_shape_support": list(REFERENCE_SHAPE_TYPES),
                "scene_object_shape_support": list(NAMED_SMALL_OBJECT_SHAPE_TYPES),
            },
            relation_fields={
                "object_count": int(object_count),
                "reference_object_id": str(dataset["reference_object_id"]),
                "reference_object_name": str(dataset["reference_object_name"]),
                "reference_shape_type": str(dataset["reference_shape_type"]),
                "relation_frame": "world_vertical_z",
            },
            execution_extra={
                "object_count": int(object_count),
                "reference_object_id": str(dataset["reference_object_id"]),
                "reference_object_name": str(dataset["reference_object_name"]),
                "reference_shape_type": str(dataset["reference_shape_type"]),
                "reference_prompt_name_count": int(dataset["reference_prompt_name_count"]),
            },
            witness_symbolic={
                "type": "marked_point_vertical_relation",
                "ids_by_role": {
                    "selected_point": str(dataset["answer_point_id"]),
                    "reference_object": str(dataset["reference_object_id"]),
                },
                "answer_label": str(dataset["answer_label"]),
            },
            draw_marked_points_fn=_render_marked_point_overlay,
            bbox_union_fn=_bbox_union,
            include_reference_render_map=True,
        )



__all__ = [
    "MIN_DISTRACTOR_REFERENCE_XY_OFFSET",
    "REFERENCE_SHAPE_TYPES",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDSpatialMarkedPointVerticalRelationLabelTask",
]
