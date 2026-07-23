"""Marked point inside a triangle formed by named objects in a synthetic 3D object scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_domain_defaults, get_scene_defaults
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ..shared.object_scene import (
    SCENE_ID,
    _RenderParams,
    _bbox_intersection_area,
    _build_projection_frame,
    _camera_yaw_band_for_instance,
    _object_reference_points,
    _sample_camera,
    _sample_scene_object_specs,
)
from ..shared.object_scene_marked_point_output import (
    generate_marked_point_object_scene_task_with_retries as _generate_marked_point_object_scene_task_with_retries,
)
from .shared.annotations import draw_marked_points as _render_marked_point_overlay
from .shared.labels import assign_answer_label as _assign_answer_label
from .shared.labels import bbox_union as _bbox_union
from .shared.spatial_primitives import (
    MARKER_WORLD_Z,
    MIN_MARKER_SCREEN_SEPARATION_PX,
    POINT_MARKER_COUNT,
    clear_of_existing_world_xy,
    clear_of_objects,
    demote_small_object_spec,
    finalize_object_specs,
    floor_sample_extent,
    make_marker_record,
    marker_object_overlap_ok,
    marker_screen_record,
    marker_screen_separation_ok,
    object_bboxes_by_id as _object_bboxes_by_id,
    prompt_names_are_unique,
)
from .shared.sampling import _projection_payload as _projection_payload


TASK_ID = "task_three_d__object_scene__reference_triangle_inside_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("reference_triangle_inside",)
POINT_COUNT = POINT_MARKER_COUNT
MIN_TRIANGLE_AREA_PX2 = 42000.0
MIN_TRIANGLE_SIDE_PX = 175.0
MIN_INSIDE_MARGIN_PX = 34.0
MIN_OUTSIDE_MARGIN_PX = 42.0


def _cross(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (
        float(c[0]) - float(a[0])
    )


def _triangle_area(points: Sequence[Sequence[float]]) -> float:
    return abs(_cross(points[0], points[1], points[2])) * 0.5


def _triangle_side_lengths(points: Sequence[Sequence[float]]) -> List[float]:
    return [
        math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
        for a, b in ((points[0], points[1]), (points[1], points[2]), (points[2], points[0]))
    ]


def _ordered_triangle_refs(refs: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    cx = sum(float(ref["screen_xy"][0]) for ref in refs) / 3.0
    cy = sum(float(ref["screen_xy"][1]) for ref in refs) / 3.0
    return sorted(refs, key=lambda ref: math.atan2(float(ref["screen_xy"][1]) - cy, float(ref["screen_xy"][0]) - cx))


def _edge_signed_distance(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return 0.0
    return (dx * (float(point[1]) - float(a[1])) - dy * (float(point[0]) - float(a[0]))) / denom


def _triangle_margin(point: Sequence[float], triangle_points: Sequence[Sequence[float]]) -> float:
    orient = 1.0 if _cross(triangle_points[0], triangle_points[1], triangle_points[2]) >= 0.0 else -1.0
    distances = [
        orient * _edge_signed_distance(point, triangle_points[0], triangle_points[1]),
        orient * _edge_signed_distance(point, triangle_points[1], triangle_points[2]),
        orient * _edge_signed_distance(point, triangle_points[2], triangle_points[0]),
    ]
    return min(float(value) for value in distances)


def _sample_inside_world_point(*, rng, references: Sequence[Mapping[str, Any]]) -> Tuple[float, float, float]:
    for _attempt in range(120):
        weights = [float(rng.uniform(0.18, 0.70)) for _ in range(3)]
        total = sum(weights)
        weights = [value / total for value in weights]
        if min(weights) < 0.18:
            continue
        x = sum(float(weight) * float(ref["base_xyz"][0]) for weight, ref in zip(weights, references))
        y = sum(float(weight) * float(ref["base_xyz"][1]) for weight, ref in zip(weights, references))
        return float(x), float(y), MARKER_WORLD_Z
    raise ValueError("could not sample interior triangle point")


def _sample_triangle_marker(
    *,
    rng,
    marker_id: str,
    surface_kind: str,
    want_inside: bool,
    references: Sequence[Mapping[str, Any]],
    triangle_points_px: Sequence[Sequence[float]],
    objects: Sequence[Mapping[str, Any]],
    existing_world_xy: Sequence[Sequence[float]],
    camera,
    frame,
    render_params: _RenderParams,
) -> Dict[str, Any]:
    """Place one marker with a signed screen-space margin relative to the reference triangle."""

    extent = floor_sample_extent(render_params)
    for _attempt in range(520):
        if bool(want_inside):
            x, y, z = _sample_inside_world_point(rng=rng, references=references)
        else:
            x = float(rng.uniform(-extent, extent))
            y = float(rng.uniform(-extent, extent))
            z = MARKER_WORLD_Z
        if not clear_of_existing_world_xy(x=x, y=y, existing_world_xy=existing_world_xy):
            continue
        if not clear_of_objects(x=x, y=y, objects=objects, clearance=0.16):
            continue
        record = make_marker_record(
            marker_id=str(marker_id),
            world_xyz=(x, y, z),
            surface_kind=str(surface_kind),
        )
        try:
            finalized = marker_screen_record(record, camera=camera, frame=frame, render_params=render_params)
        except ValueError:
            continue
        margin = _triangle_margin(finalized["screen_xy"], triangle_points_px)
        if bool(want_inside):
            if margin < MIN_INSIDE_MARGIN_PX:
                continue
        elif margin > -MIN_OUTSIDE_MARGIN_PX:
            continue
        finalized["triangle_margin_px"] = round(float(margin), 3)
        finalized["inside_reference_triangle"] = bool(margin > 0.0)
        return finalized
    raise ValueError("could not place triangle marked point")


def _build_reference_triangle_scene_dataset(
    *,
    branch_key: str,
    scene_variant: str,
    point_count: int,
    object_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build a triangle-containment instance with exactly one inside marker and outside distractors."""

    if str(branch_key) != "reference_triangle_inside":
        raise ValueError(f"unsupported query_id: {branch_key}")
    if int(point_count) != POINT_COUNT:
        raise ValueError(f"{TASK_ID} expects exactly {POINT_COUNT} marked point options")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )

    for _attempt in range(760):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        small_specs, large_specs = _sample_scene_object_specs(
            rng=rng,
            candidate_count=int(object_count),
            context_object_count=int(context_object_count),
        )
        object_specs = [
            *[demote_small_object_spec(spec, index=index, prefix="triangle") for index, spec in enumerate(small_specs)],
            *[dict(spec) for spec in large_specs],
        ]
        if not prompt_names_are_unique(object_specs):
            continue
        reference_points = [point for spec in object_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_objects = finalize_object_specs(object_specs, camera=camera, frame=frame)
        finalized_small = [spec for spec in finalized_objects if str(spec.get("object_role")) == "small_context"]
        if len(finalized_small) < 3:
            continue
        object_bbox_map = _object_bboxes_by_id(specs=finalized_objects, camera=camera, frame=frame)
        object_bboxes = list(object_bbox_map.values())
        if any(
            _bbox_intersection_area(a, b) > 3800.0
            for index, a in enumerate(object_bboxes)
            for b in object_bboxes[index + 1 :]
        ):
            continue
        triples = [
            (a, b, c)
            for first, a in enumerate(finalized_small)
            for second, b in enumerate(finalized_small[first + 1 :], start=first + 1)
            for c in finalized_small[second + 1 :]
        ]
        rng.shuffle(triples)
        for triple in triples[:12]:
            ordered_refs = _ordered_triangle_refs(triple)
            triangle_points_px = [list(ref["screen_xy"]) for ref in ordered_refs]
            if _triangle_area(triangle_points_px) < MIN_TRIANGLE_AREA_PX2:
                continue
            if min(_triangle_side_lengths(triangle_points_px)) < MIN_TRIANGLE_SIDE_PX:
                continue
            finalized_markers: List[Dict[str, Any]] = []
            existing_world_xy: List[Tuple[float, float]] = []
            try:
                answer_marker = _sample_triangle_marker(
                    rng=rng,
                    marker_id="raw_triangle_inside_answer",
                    surface_kind="floor_inside_reference_triangle",
                    want_inside=True,
                    references=ordered_refs,
                    triangle_points_px=triangle_points_px,
                    objects=finalized_objects,
                    existing_world_xy=existing_world_xy,
                    camera=camera,
                    frame=frame,
                    render_params=render_params,
                )
                finalized_markers.append(answer_marker)
                existing_world_xy.append(tuple(float(value) for value in answer_marker["world_xyz"][:2]))
                while len(finalized_markers) < int(point_count):
                    distractor = _sample_triangle_marker(
                        rng=rng,
                        marker_id=f"raw_triangle_outside_distractor_{len(finalized_markers)}",
                        surface_kind="floor_outside_reference_triangle",
                        want_inside=False,
                        references=ordered_refs,
                        triangle_points_px=triangle_points_px,
                        objects=finalized_objects,
                        existing_world_xy=existing_world_xy,
                        camera=camera,
                        frame=frame,
                        render_params=render_params,
                    )
                    finalized_markers.append(distractor)
                    existing_world_xy.append(tuple(float(value) for value in distractor["world_xyz"][:2]))
            except ValueError:
                continue
            if not marker_screen_separation_ok(finalized_markers, min_px=MIN_MARKER_SCREEN_SEPARATION_PX):
                continue
            if not marker_object_overlap_ok(
                markers=finalized_markers,
                object_bboxes=object_bboxes,
                intersection_area_fn=_bbox_intersection_area,
                marker_radius=24.0,
                max_intersection_area=1500.0,
            ):
                continue
            relabeled_markers = _assign_answer_label(
                records=finalized_markers,
                answer_marker_id="raw_triangle_inside_answer",
                point_count=int(point_count),
                answer_label_index=int(instance_seed),
                rng=rng,
            )
            answer_marker = next(marker for marker in relabeled_markers if str(marker["marker_id"]) == "raw_triangle_inside_answer")
            margins_by_label = {
                str(marker["point_label"]): round(float(marker["triangle_margin_px"]), 3)
                for marker in sorted(relabeled_markers, key=lambda item: str(item["point_label"]))
            }
            camera_payload, projection_frame_payload = _projection_payload(camera, frame, selected_camera_yaw_band)
            return {
                "query_id": str(branch_key),
                "scene_variant": str(scene_variant),
                "point_count": int(point_count),
                "object_count": int(object_count),
                "context_object_count": int(len(finalized_objects)),
                "small_context_object_count": int(len(finalized_small)),
                "large_context_object_count": int(len(finalized_objects) - len(finalized_small)),
                "point_specs": [],
                "context_object_specs": sorted(finalized_objects, key=lambda spec: str(spec["object_id"])),
                "object_specs": sorted(finalized_objects, key=lambda spec: str(spec["object_id"])),
                "marked_points": list(relabeled_markers),
                "answer_label": str(answer_marker["point_label"]),
                "answer_point_id": str(answer_marker["point_id"]),
                "answer_marker_id": str(answer_marker["marker_id"]),
                "answer_point_px": [round(float(value), 3) for value in answer_marker["screen_xy"]],
                "reference_object_a_id": str(ordered_refs[0]["object_id"]),
                "reference_object_b_id": str(ordered_refs[1]["object_id"]),
                "reference_object_c_id": str(ordered_refs[2]["object_id"]),
                "reference_object_a_name": str(ordered_refs[0]["prompt_name"]),
                "reference_object_b_name": str(ordered_refs[1]["prompt_name"]),
                "reference_object_c_name": str(ordered_refs[2]["prompt_name"]),
                "reference_object_a_shape_type": str(ordered_refs[0]["shape_type"]),
                "reference_object_b_shape_type": str(ordered_refs[1]["shape_type"]),
                "reference_object_c_shape_type": str(ordered_refs[2]["shape_type"]),
                "reference_triangle_points_px": [list(point) for point in triangle_points_px],
                "camera": camera_payload,
                "projection_frame": projection_frame_payload,
                "solver_trace": {
                    "relation": "inside_reference_triangle",
                    "relation_frame": "image_plane_triangle_containment",
                    "reference_object_a_id": str(ordered_refs[0]["object_id"]),
                    "reference_object_b_id": str(ordered_refs[1]["object_id"]),
                    "reference_object_c_id": str(ordered_refs[2]["object_id"]),
                    "reference_object_a_name": str(ordered_refs[0]["prompt_name"]),
                    "reference_object_b_name": str(ordered_refs[1]["prompt_name"]),
                    "reference_object_c_name": str(ordered_refs[2]["prompt_name"]),
                    "reference_triangle_points_px": [list(point) for point in triangle_points_px],
                    "triangle_margins_px_by_label": dict(margins_by_label),
                    "answer_triangle_margin_px": round(float(answer_marker["triangle_margin_px"]), 3),
                    "minimum_outside_margin_px": round(
                        min(
                            abs(float(marker["triangle_margin_px"]))
                            for marker in relabeled_markers
                            if str(marker["marker_id"]) != "raw_triangle_inside_answer"
                        ),
                        3,
                    ),
                    "triangle_area_px2": round(float(_triangle_area(triangle_points_px)), 3),
                },
            }
    raise ValueError("could not construct a valid reference-triangle marked-point scene")


_SCENE_DEFAULTS = get_scene_defaults("three_d", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _triangle_dynamic_slots(dataset: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "reference_a_name": str(dataset["reference_object_a_name"]),
        "reference_b_name": str(dataset["reference_object_b_name"]),
        "reference_c_name": str(dataset["reference_object_c_name"]),
    }


def _triangle_relation_fields(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "object_count": int(dataset["context_object_count"]),
        "reference_object_a_id": str(dataset["reference_object_a_id"]),
        "reference_object_b_id": str(dataset["reference_object_b_id"]),
        "reference_object_c_id": str(dataset["reference_object_c_id"]),
        "reference_object_a_name": str(dataset["reference_object_a_name"]),
        "reference_object_b_name": str(dataset["reference_object_b_name"]),
        "reference_object_c_name": str(dataset["reference_object_c_name"]),
        "relation_frame": "image_plane_triangle_containment",
    }


def _triangle_execution_extra(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "object_count": int(dataset["context_object_count"]),
        "small_context_object_count": int(dataset["small_context_object_count"]),
        "large_context_object_count": int(dataset["large_context_object_count"]),
        "reference_object_a_id": str(dataset["reference_object_a_id"]),
        "reference_object_b_id": str(dataset["reference_object_b_id"]),
        "reference_object_c_id": str(dataset["reference_object_c_id"]),
        "reference_object_a_name": str(dataset["reference_object_a_name"]),
        "reference_object_b_name": str(dataset["reference_object_b_name"]),
        "reference_object_c_name": str(dataset["reference_object_c_name"]),
    }


def _triangle_witness_symbolic(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "inside_reference_triangle",
        "ids_by_role": {
            "selected_point": str(dataset["answer_point_id"]),
            "reference_object_a": str(dataset["reference_object_a_id"]),
            "reference_object_b": str(dataset["reference_object_b_id"]),
            "reference_object_c": str(dataset["reference_object_c_id"]),
        },
        "answer_label": str(dataset["answer_label"]),
    }


_GENERATOR_CONFIG = {
    "branch_options": SUPPORTED_QUERY_IDS,
    "generation_defaults": _GEN_DEFAULTS,
    "render_defaults": _RENDER_DEFAULTS,
    "prompt_defaults_config": _PROMPT_DEFAULTS,
    "background_defaults": _BACKGROUND_DEFAULTS,
    "noise_defaults": _NOISE_DEFAULTS,
    "exact_point_count": POINT_COUNT,
    "dataset_builder": _build_reference_triangle_scene_dataset,
    "dynamic_slots_fn": _triangle_dynamic_slots,
    "scene_kind": "three_d_object_scene_reference_triangle_inside",
    "relation_fields_fn": _triangle_relation_fields,
    "execution_extra_fn": _triangle_execution_extra,
    "witness_symbolic_fn": _triangle_witness_symbolic,
    "draw_marked_points_fn": _render_marked_point_overlay,
    "bbox_union_fn": _bbox_union,
}


@register_task
class ThreeDObjectSceneReferenceTriangleInsideLabelTask:
    """Choose the marked point inside the triangle formed by three named objects."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    domain = "three_d"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Resolve task-specific triangle trace fields while shared code handles retry and output assembly."""

        public_name = TASK_ID
        public_domain = str(self.domain)
        attempt_limit = int(max_attempts)
        generator_config = dict(_GENERATOR_CONFIG)
        return _generate_marked_point_object_scene_task_with_retries(
            public_name=public_name,
            public_domain=public_domain,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=attempt_limit,
            **generator_config,
        )


__all__ = [
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDObjectSceneReferenceTriangleInsideLabelTask",
]
