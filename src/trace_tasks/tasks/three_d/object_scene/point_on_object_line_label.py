"""Marked point on an object-to-object line in a synthetic 3D object scene."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_domain_defaults, get_scene_defaults
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ..shared.object_scene_marked_point_output import build_marked_point_object_scene_output as _build_marked_point_object_scene_output
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.object_scene import (
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


TASK_ID = "task_three_d__object_scene__point_on_object_line_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("point_on_object_line",)
POINT_COUNT = 6
MIN_REFERENCE_SCREEN_DISTANCE_PX = 230.0
MAX_CORRECT_LINE_DISTANCE_PX = 18.0
MIN_DISTRACTOR_LINE_DISTANCE_PX = 52.0
MIN_MARKER_SCREEN_SEPARATION_PX = 76.0


def _demote_small_object_spec(spec: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    updated = dict(spec)
    shape_type = str(updated["shape_type"])
    updated["object_id"] = f"line_object_{int(index):02d}_{shape_type}"
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


def _line_distance_px(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    dx = bx - ax
    dy = by - ay
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return float("inf")
    return abs(dy * px - dx * py + bx * ay - by * ax) / denom


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


def _make_marker_record(*, marker_id: str, world_xyz: Sequence[float], surface_kind: str) -> Dict[str, Any]:
    return {
        "marker_id": str(marker_id),
        "surface_kind": str(surface_kind),
        "attached_object_id": None,
        "world_xyz": [round(float(value), 4) for value in world_xyz],
    }


def _marker_screen_record(record: Mapping[str, Any], *, camera, frame, render_params: _RenderParams) -> Dict[str, Any]:
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
    return finalized


def _object_bboxes_by_id(*, specs: Sequence[Mapping[str, Any]], camera, frame) -> Dict[str, List[float]]:
    return {str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=10.0) for spec in specs}


def _sample_distractor_marker(
    *,
    rng,
    objects: Sequence[Mapping[str, Any]],
    existing_world_xy: Sequence[Sequence[float]],
    reference_line_a: Sequence[float],
    reference_line_b: Sequence[float],
    camera,
    frame,
    render_params: _RenderParams,
) -> Dict[str, Any]:
    """Sample one readable off-line point marker while preserving line-answer uniqueness."""

    extent = min(2.82, max(2.1, float(render_params.room_extent) - 0.34))
    for _attempt in range(320):
        x = float(rng.uniform(-extent, extent))
        y = float(rng.uniform(-extent, extent))
        if any(math.hypot(x - float(other[0]), y - float(other[1])) < 0.44 for other in existing_world_xy):
            continue
        if any(
            math.hypot(x - float(obj["base_xyz"][0]), y - float(obj["base_xyz"][1]))
            < float(obj.get("footprint_radius", 0.42)) + 0.16
            for obj in objects
        ):
            continue
        record = _make_marker_record(
            marker_id=f"raw_line_distractor_{len(existing_world_xy)}",
            world_xyz=(x, y, 0.055),
            surface_kind="floor_off_reference_line",
        )
        try:
            finalized = _marker_screen_record(record, camera=camera, frame=frame, render_params=render_params)
        except ValueError:
            continue
        screen_xy = finalized["screen_xy"]
        line_distance = _line_distance_px(screen_xy, reference_line_a, reference_line_b)
        line_fraction = _line_fraction(screen_xy, reference_line_a, reference_line_b)
        if line_distance < MIN_DISTRACTOR_LINE_DISTANCE_PX:
            continue
        if not -0.15 <= line_fraction <= 1.15:
            continue
        finalized["line_distance_px"] = round(float(line_distance), 3)
        finalized["line_fraction"] = round(float(line_fraction), 4)
        return finalized
    raise ValueError("could not place off-line distractor marked point")


def _build_point_on_line_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    point_count: int,
    object_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build a scene whose only near-collinear marked point is between two named object projections."""

    if str(query_id) != "point_on_object_line":
        raise ValueError(f"unsupported query_id: {query_id}")
    if int(point_count) != POINT_COUNT:
        raise ValueError(f"{TASK_ID} expects exactly {POINT_COUNT} marked point options")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )

    for _attempt in range(640):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        small_specs, large_specs = _sample_scene_object_specs(
            rng=rng,
            candidate_count=int(object_count),
            context_object_count=int(context_object_count),
        )
        object_specs = [
            *[_demote_small_object_spec(spec, index=index) for index, spec in enumerate(small_specs)],
            *[dict(spec) for spec in large_specs],
        ]
        prompt_name_counts = Counter(str(spec["prompt_name"]) for spec in object_specs)
        if any(int(count) != 1 for count in prompt_name_counts.values()):
            continue
        reference_points = [point for spec in object_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_objects = _finalize_object_specs(object_specs, camera=camera, frame=frame)
        finalized_small = [spec for spec in finalized_objects if str(spec.get("object_role")) == "small_context"]
        if len(finalized_small) < 2:
            continue
        object_bboxes_by_id = _object_bboxes_by_id(specs=finalized_objects, camera=camera, frame=frame)
        object_bboxes = list(object_bboxes_by_id.values())
        if any(
            _bbox_intersection_area(a, b) > 3800.0
            for index, a in enumerate(object_bboxes)
            for b in object_bboxes[index + 1 :]
        ):
            continue

        pairs = [
            (a, b)
            for index, a in enumerate(finalized_small)
            for b in finalized_small[index + 1 :]
            if math.hypot(
                float(a["screen_xy"][0]) - float(b["screen_xy"][0]),
                float(a["screen_xy"][1]) - float(b["screen_xy"][1]),
            )
            >= MIN_REFERENCE_SCREEN_DISTANCE_PX
        ]
        if not pairs:
            continue
        rng.shuffle(pairs)

        for reference_a, reference_b in pairs[:8]:
            ax, ay = (float(value) for value in reference_a["base_xyz"][:2])
            bx, by = (float(value) for value in reference_b["base_xyz"][:2])
            world_distance = math.hypot(bx - ax, by - ay)
            if world_distance < 1.20:
                continue
            t = float(rng.uniform(0.34, 0.66))
            answer_world = (ax + (bx - ax) * t, ay + (by - ay) * t, 0.055)
            if any(
                math.hypot(float(answer_world[0]) - float(obj["base_xyz"][0]), float(answer_world[1]) - float(obj["base_xyz"][1]))
                < float(obj.get("footprint_radius", 0.42)) + 0.12
                for obj in finalized_objects
            ):
                continue
            answer_record = _make_marker_record(
                marker_id="raw_line_answer",
                world_xyz=answer_world,
                surface_kind="floor_on_reference_line",
            )
            try:
                answer_marker = _marker_screen_record(
                    answer_record,
                    camera=camera,
                    frame=frame,
                    render_params=render_params,
                )
            except ValueError:
                continue
            ref_line_a = reference_a["screen_xy"]
            ref_line_b = reference_b["screen_xy"]
            answer_line_distance = _line_distance_px(answer_marker["screen_xy"], ref_line_a, ref_line_b)
            answer_line_fraction = _line_fraction(answer_marker["screen_xy"], ref_line_a, ref_line_b)
            if answer_line_distance > MAX_CORRECT_LINE_DISTANCE_PX or not 0.20 <= answer_line_fraction <= 0.80:
                continue
            answer_marker["line_distance_px"] = round(float(answer_line_distance), 3)
            answer_marker["line_fraction"] = round(float(answer_line_fraction), 4)

            finalized_markers = [answer_marker]
            existing_world_xy = [answer_world]
            try:
                while len(finalized_markers) < int(point_count):
                    distractor = _sample_distractor_marker(
                        rng=rng,
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
            screen_centers = [(float(item["screen_xy"][0]), float(item["screen_xy"][1])) for item in finalized_markers]
            if any(
                math.hypot(a[0] - b[0], a[1] - b[1]) < MIN_MARKER_SCREEN_SEPARATION_PX
                for index, a in enumerate(screen_centers)
                for b in screen_centers[index + 1 :]
            ):
                continue
            marker_radius = 24.0
            if any(
                _bbox_intersection_area(
                    [center[0] - marker_radius, center[1] - marker_radius, center[0] + marker_radius, center[1] + marker_radius],
                    object_bbox,
                )
                > 1500.0
                for center in screen_centers
                for object_bbox in object_bboxes
            ):
                continue
            distractor_distances = [
                float(marker["line_distance_px"])
                for marker in finalized_markers
                if str(marker["marker_id"]) != "raw_line_answer"
            ]
            if min(distractor_distances) - float(answer_line_distance) < 42.0:
                continue

            relabeled_markers = _assign_answer_label(
                records=finalized_markers,
                answer_marker_id="raw_line_answer",
                point_count=int(point_count),
                answer_label_index=int(instance_seed),
                rng=rng,
            )
            answer_marker = next(marker for marker in relabeled_markers if str(marker["marker_id"]) == "raw_line_answer")
            line_distances_by_label = {
                str(marker["point_label"]): round(float(marker["line_distance_px"]), 3)
                for marker in sorted(relabeled_markers, key=lambda item: str(item["point_label"]))
            }
            return {
                "query_id": str(query_id),
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
                "reference_object_a_id": str(reference_a["object_id"]),
                "reference_object_b_id": str(reference_b["object_id"]),
                "reference_object_a_name": str(reference_a["prompt_name"]),
                "reference_object_b_name": str(reference_b["prompt_name"]),
                "reference_object_a_shape_type": str(reference_a["shape_type"]),
                "reference_object_b_shape_type": str(reference_b["shape_type"]),
                "reference_line_points_px": [list(ref_line_a), list(ref_line_b)],
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
                    "relation": "point_on_object_line",
                    "relation_frame": "image_plane_collinearity",
                    "reference_object_a_id": str(reference_a["object_id"]),
                    "reference_object_b_id": str(reference_b["object_id"]),
                    "reference_object_a_name": str(reference_a["prompt_name"]),
                    "reference_object_b_name": str(reference_b["prompt_name"]),
                    "reference_line_points_px": [list(ref_line_a), list(ref_line_b)],
                    "line_distances_px_by_label": dict(line_distances_by_label),
                    "answer_line_distance_px": round(float(answer_line_distance), 3),
                    "minimum_distractor_line_distance_px": round(float(min(distractor_distances)), 3),
                    "line_distance_margin_px": round(float(min(distractor_distances) - float(answer_line_distance)), 3),
                },
            }
    raise ValueError("could not construct a valid point-on-object-line scene")


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
class ThreeDObjectScenePointOnObjectLineLabelTask:
    """Choose the marked point aligned with the line between two named objects."""

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
        """Generate one point-on-line instance with prompt, answer, and scalar point annotation from one accepted trace."""

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
            default_min=POINT_COUNT,
            default_max=POINT_COUNT,
            lower=POINT_COUNT,
            upper=POINT_COUNT,
        )
        object_count, object_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="object_count",
            default_min=6,
            default_max=6,
            lower=5,
            upper=7,
        )
        context_object_count, context_object_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="context_object_count",
            default_min=1,
            default_max=1,
            lower=1,
            upper=2,
        )
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_point_on_line_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            object_count=int(object_count),
            context_object_count=int(context_object_count),
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
            dynamic_slots={
                "reference_a_name": str(dataset["reference_object_a_name"]),
                "reference_b_name": str(dataset["reference_object_b_name"]),
            },
            scene_kind="three_d_object_scene_point_on_object_line",
            count_params={
                "object_count": int(object_count),
                "object_count_probabilities": dict(object_count_probabilities),
                "context_object_count": int(context_object_count),
                "context_object_count_probabilities": dict(context_object_count_probabilities),
            },
            relation_fields={
                "object_count": int(dataset["context_object_count"]),
                "reference_object_a_id": str(dataset["reference_object_a_id"]),
                "reference_object_b_id": str(dataset["reference_object_b_id"]),
                "reference_object_a_name": str(dataset["reference_object_a_name"]),
                "reference_object_b_name": str(dataset["reference_object_b_name"]),
                "relation_frame": "image_plane_collinearity",
            },
            execution_extra={
                "object_count": int(dataset["context_object_count"]),
                "small_context_object_count": int(dataset["small_context_object_count"]),
                "large_context_object_count": int(dataset["large_context_object_count"]),
                "reference_object_a_id": str(dataset["reference_object_a_id"]),
                "reference_object_b_id": str(dataset["reference_object_b_id"]),
                "reference_object_a_name": str(dataset["reference_object_a_name"]),
                "reference_object_b_name": str(dataset["reference_object_b_name"]),
            },
            witness_symbolic={
                "type": "point_on_object_line",
                "ids_by_role": {
                    "selected_point": str(dataset["answer_point_id"]),
                    "reference_object_a": str(dataset["reference_object_a_id"]),
                    "reference_object_b": str(dataset["reference_object_b_id"]),
                },
                "answer_label": str(dataset["answer_label"]),
            },
            draw_marked_points_fn=_render_marked_point_overlay,
            bbox_union_fn=_bbox_union,
        )


__all__ = [
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDObjectScenePointOnObjectLineLabelTask",
]
