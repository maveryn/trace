"""Marked point side of a directed object-to-object line in a synthetic 3D object scene."""

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


TASK_ID = "task_three_d__object_scene__line_side_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("left_of_directed_line", "right_of_directed_line")
MIN_REFERENCE_SCREEN_DISTANCE_PX = 240.0
MIN_SIDE_DISTANCE_PX = 54.0
MIN_SIDE_MARGIN_PX = 42.0


def _line_fraction(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 1e-6:
        return 0.0
    return ((px - ax) * dx + (py - ay) * dy) / denom


def _visual_left_signed_distance_px(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    """Positive means visually left when moving from a to b in image coordinates."""

    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return 0.0
    raw_y_down_cross = dx * (py - ay) - dy * (px - ax)
    return -raw_y_down_cross / denom


def _branch_requested_side(branch_key: str) -> Tuple[str, int]:
    if str(branch_key) == "left_of_directed_line":
        return "left", 1
    if str(branch_key) == "right_of_directed_line":
        return "right", -1
    raise ValueError(f"unsupported query_id: {branch_key}")


def _sample_side_marker(
    *,
    rng,
    marker_id: str,
    surface_kind: str,
    side_sign: int,
    objects: Sequence[Mapping[str, Any]],
    existing_world_xy: Sequence[Sequence[float]],
    reference_line_a: Sequence[float],
    reference_line_b: Sequence[float],
    camera,
    frame,
    render_params: _RenderParams,
) -> Dict[str, Any]:
    """Place one marker on the requested visual side while preserving floor readability."""

    extent = floor_sample_extent(render_params)
    for _attempt in range(420):
        x = float(rng.uniform(-extent, extent))
        y = float(rng.uniform(-extent, extent))
        if not clear_of_existing_world_xy(x=x, y=y, existing_world_xy=existing_world_xy):
            continue
        if not clear_of_objects(x=x, y=y, objects=objects, clearance=0.16):
            continue
        record = make_marker_record(
            marker_id=str(marker_id),
            world_xyz=(x, y, MARKER_WORLD_Z),
            surface_kind=str(surface_kind),
        )
        try:
            finalized = marker_screen_record(record, camera=camera, frame=frame, render_params=render_params)
        except ValueError:
            continue
        screen_xy = finalized["screen_xy"]
        signed_distance = _visual_left_signed_distance_px(screen_xy, reference_line_a, reference_line_b)
        fraction = _line_fraction(screen_xy, reference_line_a, reference_line_b)
        if int(side_sign) * float(signed_distance) < MIN_SIDE_DISTANCE_PX:
            continue
        if not -0.12 <= float(fraction) <= 1.12:
            continue
        finalized["directed_line_side"] = "left" if float(signed_distance) > 0.0 else "right"
        finalized["signed_side_distance_px"] = round(float(signed_distance), 3)
        finalized["line_fraction"] = round(float(fraction), 4)
        return finalized
    raise ValueError("could not place readable directed-line side marker")


def _build_line_side_scene_dataset(
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
    """Build a line-side instance where the answer marker is separated from opposite-side distractors."""

    requested_side, requested_sign = _branch_requested_side(str(branch_key))
    if int(point_count) != POINT_MARKER_COUNT:
        raise ValueError(f"{TASK_ID} expects exactly {POINT_MARKER_COUNT} marked point options")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )

    for _attempt in range(720):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        small_specs, large_specs = _sample_scene_object_specs(
            rng=rng,
            candidate_count=int(object_count),
            context_object_count=int(context_object_count),
        )
        object_specs = [
            *[demote_small_object_spec(spec, index=index, prefix="line_side") for index, spec in enumerate(small_specs)],
            *[dict(spec) for spec in large_specs],
        ]
        if not prompt_names_are_unique(object_specs):
            continue
        reference_points = [point for spec in object_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_objects = finalize_object_specs(object_specs, camera=camera, frame=frame)
        finalized_small = [spec for spec in finalized_objects if str(spec.get("object_role")) == "small_context"]
        if len(finalized_small) < 2:
            continue
        object_bbox_map = _object_bboxes_by_id(specs=finalized_objects, camera=camera, frame=frame)
        object_bboxes = list(object_bbox_map.values())
        if any(
            _bbox_intersection_area(a, b) > 3800.0
            for index, a in enumerate(object_bboxes)
            for b in object_bboxes[index + 1 :]
        ):
            continue

        pairs = [
            (a, b)
            for a in finalized_small
            for b in finalized_small
            if str(a["object_id"]) != str(b["object_id"])
            and math.hypot(
                float(a["screen_xy"][0]) - float(b["screen_xy"][0]),
                float(a["screen_xy"][1]) - float(b["screen_xy"][1]),
            )
            >= MIN_REFERENCE_SCREEN_DISTANCE_PX
        ]
        if not pairs:
            continue
        rng.shuffle(pairs)

        for reference_a, reference_b in pairs[:10]:
            ref_line_a = reference_a["screen_xy"]
            ref_line_b = reference_b["screen_xy"]
            finalized_markers: List[Dict[str, Any]] = []
            existing_world_xy: List[Tuple[float, float]] = []
            try:
                answer_marker = _sample_side_marker(
                    rng=rng,
                    marker_id="raw_line_side_answer",
                    surface_kind=f"floor_{requested_side}_of_directed_line",
                    side_sign=int(requested_sign),
                    objects=finalized_objects,
                    existing_world_xy=existing_world_xy,
                    reference_line_a=ref_line_a,
                    reference_line_b=ref_line_b,
                    camera=camera,
                    frame=frame,
                    render_params=render_params,
                )
                finalized_markers.append(answer_marker)
                existing_world_xy.append(tuple(float(value) for value in answer_marker["world_xyz"][:2]))
                while len(finalized_markers) < int(point_count):
                    distractor = _sample_side_marker(
                        rng=rng,
                        marker_id=f"raw_line_side_distractor_{len(finalized_markers)}",
                        surface_kind=f"floor_not_{requested_side}_of_directed_line",
                        side_sign=-int(requested_sign),
                        objects=finalized_objects,
                        existing_world_xy=existing_world_xy,
                        reference_line_a=ref_line_a,
                        reference_line_b=ref_line_b,
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
            answer_distance = float(answer_marker["signed_side_distance_px"])
            distractor_scores = [
                int(requested_sign) * float(marker["signed_side_distance_px"])
                for marker in finalized_markers
                if str(marker["marker_id"]) != "raw_line_side_answer"
            ]
            if min(abs(float(score)) for score in distractor_scores) < MIN_SIDE_DISTANCE_PX:
                continue
            if int(requested_sign) * answer_distance - max(distractor_scores) < MIN_SIDE_MARGIN_PX:
                continue

            relabeled_markers = _assign_answer_label(
                records=finalized_markers,
                answer_marker_id="raw_line_side_answer",
                point_count=int(point_count),
                answer_label_index=int(instance_seed),
                rng=rng,
            )
            answer_marker = next(marker for marker in relabeled_markers if str(marker["marker_id"]) == "raw_line_side_answer")
            signed_distances_by_label = {
                str(marker["point_label"]): round(float(marker["signed_side_distance_px"]), 3)
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
                "requested_side": str(requested_side),
                "requested_side_sign": int(requested_sign),
                "reference_object_a_id": str(reference_a["object_id"]),
                "reference_object_b_id": str(reference_b["object_id"]),
                "reference_object_a_name": str(reference_a["prompt_name"]),
                "reference_object_b_name": str(reference_b["prompt_name"]),
                "reference_object_a_shape_type": str(reference_a["shape_type"]),
                "reference_object_b_shape_type": str(reference_b["shape_type"]),
                "reference_line_points_px": [list(ref_line_a), list(ref_line_b)],
                "camera": camera_payload,
                "projection_frame": projection_frame_payload,
                "solver_trace": {
                    "relation": "directed_line_side",
                    "requested_side": str(requested_side),
                    "relation_frame": "image_plane_directed_line",
                    "reference_object_a_id": str(reference_a["object_id"]),
                    "reference_object_b_id": str(reference_b["object_id"]),
                    "reference_object_a_name": str(reference_a["prompt_name"]),
                    "reference_object_b_name": str(reference_b["prompt_name"]),
                    "reference_line_points_px": [list(ref_line_a), list(ref_line_b)],
                    "signed_side_distances_px_by_label": dict(signed_distances_by_label),
                    "answer_signed_side_distance_px": round(float(answer_marker["signed_side_distance_px"]), 3),
                    "minimum_absolute_side_distance_px": round(
                        min(abs(float(marker["signed_side_distance_px"])) for marker in relabeled_markers),
                        3,
                    ),
                },
            }
    raise ValueError("could not construct a valid directed-line side scene")


_SCENE_DEFAULTS = get_scene_defaults("three_d", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _line_side_dynamic_slots(dataset: Mapping[str, Any]) -> Dict[str, str]:
    return {
        "reference_a_name": str(dataset["reference_object_a_name"]),
        "reference_b_name": str(dataset["reference_object_b_name"]),
    }


def _line_side_relation_fields(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "object_count": int(dataset["context_object_count"]),
        "requested_side": str(dataset["requested_side"]),
        "reference_object_a_id": str(dataset["reference_object_a_id"]),
        "reference_object_b_id": str(dataset["reference_object_b_id"]),
        "reference_object_a_name": str(dataset["reference_object_a_name"]),
        "reference_object_b_name": str(dataset["reference_object_b_name"]),
        "relation_frame": "image_plane_directed_line",
    }


def _line_side_execution_extra(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "object_count": int(dataset["context_object_count"]),
        "small_context_object_count": int(dataset["small_context_object_count"]),
        "large_context_object_count": int(dataset["large_context_object_count"]),
        "requested_side": str(dataset["requested_side"]),
        "reference_object_a_id": str(dataset["reference_object_a_id"]),
        "reference_object_b_id": str(dataset["reference_object_b_id"]),
        "reference_object_a_name": str(dataset["reference_object_a_name"]),
        "reference_object_b_name": str(dataset["reference_object_b_name"]),
    }


def _line_side_witness_symbolic(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "type": "directed_line_side",
        "ids_by_role": {
            "selected_point": str(dataset["answer_point_id"]),
            "reference_object_a": str(dataset["reference_object_a_id"]),
            "reference_object_b": str(dataset["reference_object_b_id"]),
        },
        "answer_label": str(dataset["answer_label"]),
        "requested_side": str(dataset["requested_side"]),
    }


_GENERATOR_CONFIG = {
    "branch_options": SUPPORTED_QUERY_IDS,
    "generation_defaults": _GEN_DEFAULTS,
    "render_defaults": _RENDER_DEFAULTS,
    "prompt_defaults_config": _PROMPT_DEFAULTS,
    "background_defaults": _BACKGROUND_DEFAULTS,
    "noise_defaults": _NOISE_DEFAULTS,
    "exact_point_count": POINT_MARKER_COUNT,
    "dataset_builder": _build_line_side_scene_dataset,
    "dynamic_slots_fn": _line_side_dynamic_slots,
    "scene_kind": "three_d_object_scene_line_side",
    "relation_fields_fn": _line_side_relation_fields,
    "execution_extra_fn": _line_side_execution_extra,
    "witness_symbolic_fn": _line_side_witness_symbolic,
    "draw_marked_points_fn": _render_marked_point_overlay,
    "bbox_union_fn": _bbox_union,
}


@register_task
class ThreeDObjectSceneLineSideLabelTask:
    """Choose the marked point on a requested side of a directed line between two named objects."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    domain = "three_d"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Resolve task-specific trace fields while shared code handles retry, render, and output assembly."""

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
    "ThreeDObjectSceneLineSideLabelTask",
]
