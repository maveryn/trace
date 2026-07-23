"""Marked-point depth extremum task for a synthetic 3D object scene."""

from __future__ import annotations

import math
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
    POINT_LABELS,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    _RenderParams,
    _bbox_intersection_area,
    _build_projection_frame,
    _camera_yaw_band_for_instance,
    _min_pairwise,
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


TASK_ID = "task_three_d__object_scene__marked_point_depth_extremum_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("closest_marked_point", "farthest_marked_point")
MIN_SCREEN_DEPTH_MARGIN_PX = 32.0


def _demote_small_context_spec(spec: Mapping[str, Any], *, index: int) -> Dict[str, Any]:
    """Turn a sampled small answer-candidate object into an unlettered context object."""

    updated = dict(spec)
    shape_type = str(updated["shape_type"])
    updated["object_id"] = f"context_small_{int(index)}_{shape_type}"
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


def _sample_floor_marker_world(
    *,
    rng,
    objects: Sequence[Mapping[str, Any]],
    existing_points: Sequence[Sequence[float]],
    room_extent: float,
) -> Tuple[float, float, float]:
    extent = min(2.82, max(2.1, float(room_extent) - 0.34))
    for _attempt in range(160):
        x = float(rng.uniform(-extent, extent))
        y = float(rng.uniform(-extent, extent))
        point = (x, y, 0.045)
        if any(math.hypot(x - float(other[0]), y - float(other[1])) < 0.48 for other in existing_points):
            continue
        if any(
            math.hypot(x - float(obj["base_xyz"][0]), y - float(obj["base_xyz"][1]))
            < float(obj.get("footprint_radius", 0.42)) + 0.20
            for obj in objects
        ):
            continue
        return point
    raise ValueError("could not place a visible floor marker")


def _sample_marker_world_points(
    *,
    rng,
    point_count: int,
    objects: Sequence[Mapping[str, Any]],
    room_extent: float,
) -> List[Dict[str, Any]]:
    """Sample marked floor points with enough separation for a visible depth extremum."""
    records: List[Dict[str, Any]] = []
    existing: List[Tuple[float, float, float]] = []

    while len(records) < int(point_count):
        world = _sample_floor_marker_world(
            rng=rng,
            objects=objects,
            existing_points=existing,
            room_extent=float(room_extent),
        )
        marker_index = len(records)
        records.append(
            {
                "marker_id": f"raw_marked_point_{marker_index}",
                "surface_kind": "floor",
                "attached_object_id": None,
                "world_xyz": [round(float(value), 4) for value in world],
            }
        )
        existing.append(world)

    rng.shuffle(records)
    return records


def _build_marked_point_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    point_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build a marked-point depth scene where one labeled point is uniquely closest or farthest from the camera."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )
    large_context_count = max(1, min(3, int(context_object_count) // 3))
    small_context_count = max(2, int(context_object_count) - int(large_context_count))

    for _attempt in range(420):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        small_specs, large_specs = _sample_scene_object_specs(
            rng=rng,
            candidate_count=int(small_context_count),
            context_object_count=int(large_context_count),
        )
        context_specs = [
            *[_demote_small_context_spec(spec, index=index) for index, spec in enumerate(small_specs)],
            *[dict(spec) for spec in large_specs],
        ]
        marker_records = _sample_marker_world_points(
            rng=rng,
            point_count=int(point_count),
            objects=context_specs,
            room_extent=float(render_params.room_extent),
        )
        reference_points = [
            *(point for spec in context_specs for point in _object_reference_points(spec)),
            *(tuple(float(value) for value in record["world_xyz"]) for record in marker_records),
        ]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_context = _finalize_object_specs(context_specs, camera=camera, frame=frame)
        object_bboxes_by_id = {
            str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=10.0)
            for spec in finalized_context
        }
        object_bboxes = list(object_bboxes_by_id.values())
        finalized_markers: List[Dict[str, Any]] = []
        for record in marker_records:
            screen = _project_screen(record["world_xyz"], camera, frame)
            x, y = float(screen[0]), float(screen[1])
            if not (
                58.0 <= x <= float(render_params.canvas_width) - 58.0
                and 58.0 <= y <= float(render_params.canvas_height) - 58.0
            ):
                break
            finalized = dict(record)
            finalized.update(
                {
                    "screen_xy": [round(float(x), 3), round(float(y), 3)],
                    "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                    "camera_distance": round(float(screen[7]), 4),
                }
            )
            finalized_markers.append(finalized)
        if len(finalized_markers) != int(point_count):
            continue
        screen_centers = [(float(item["screen_xy"][0]), float(item["screen_xy"][1])) for item in finalized_markers]
        if any(
            math.hypot(a[0] - b[0], a[1] - b[1]) < 82.0
            for index, a in enumerate(screen_centers)
            for b in screen_centers[index + 1 :]
        ):
            continue
        marker_bboxes_by_id = {
            str(marker["marker_id"]): [center[0] - 24.0, center[1] - 24.0, center[0] + 24.0, center[1] + 24.0]
            for marker, center in zip(finalized_markers, screen_centers)
        }
        if any(
            _bbox_intersection_area(a, b) > 4200.0
            for index, a in enumerate(object_bboxes)
            for b in object_bboxes[index + 1 :]
        ):
            continue
        heavy_marker_overlap = False
        for marker in finalized_markers:
            marker_bbox = marker_bboxes_by_id[str(marker["marker_id"])]
            attached_object_id = marker.get("attached_object_id")
            for object_id, object_bbox in object_bboxes_by_id.items():
                if attached_object_id is not None and str(attached_object_id) == str(object_id):
                    continue
                if _bbox_intersection_area(marker_bbox, object_bbox) > 1800.0:
                    heavy_marker_overlap = True
                    break
            if heavy_marker_overlap:
                break
        if heavy_marker_overlap:
            continue
        camera_distances = [float(item["camera_distance"]) for item in finalized_markers]
        if _min_pairwise(camera_distances) < 0.32:
            continue
        sorted_by_depth = sorted(finalized_markers, key=lambda item: (float(item["camera_distance"]), str(item["marker_id"])))
        if str(query_id) == "closest_marked_point":
            answer_marker_id = str(sorted_by_depth[0]["marker_id"])
            depth_margin = float(sorted_by_depth[1]["camera_distance"]) - float(sorted_by_depth[0]["camera_distance"])
        else:
            answer_marker_id = str(sorted_by_depth[-1]["marker_id"])
            depth_margin = float(sorted_by_depth[-1]["camera_distance"]) - float(sorted_by_depth[-2]["camera_distance"])
        if float(depth_margin) < 0.38:
            continue
        screen_y_values = sorted(float(item["screen_xy"][1]) for item in finalized_markers)
        answer_marker = next(item for item in finalized_markers if str(item["marker_id"]) == str(answer_marker_id))
        answer_y = float(answer_marker["screen_xy"][1])
        if str(query_id) == "closest_marked_point":
            screen_depth_margin = float(answer_y) - float(screen_y_values[-2])
        else:
            screen_depth_margin = float(screen_y_values[1]) - float(answer_y)
        if float(screen_depth_margin) < MIN_SCREEN_DEPTH_MARGIN_PX:
            continue
        relabeled_markers = _assign_answer_label(
            records=finalized_markers,
            answer_marker_id=str(answer_marker_id),
            point_count=int(point_count),
            answer_label_index=int(instance_seed),
            rng=rng,
        )
        answer_marker = next(item for item in relabeled_markers if str(item["marker_id"]) == str(answer_marker_id))
        sorted_relabeled_by_depth = sorted(relabeled_markers, key=lambda item: (float(item["camera_distance"]), str(item["point_label"])))
        sorted_relabeled_by_screen_depth = sorted(
            relabeled_markers,
            key=lambda item: (float(item["screen_xy"][1]), str(item["point_label"])),
            reverse=True,
        )
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "context_object_count": int(context_object_count),
            "small_context_object_count": int(small_context_count),
            "large_context_object_count": int(large_context_count),
            "point_specs": [],
            "context_object_specs": sorted(finalized_context, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted(finalized_context, key=lambda spec: str(spec["object_id"])),
            "marked_points": list(relabeled_markers),
            "answer_label": str(answer_marker["point_label"]),
            "answer_point_id": str(answer_marker["point_id"]),
            "answer_marker_id": str(answer_marker["marker_id"]),
            "answer_point_px": [round(float(value), 3) for value in answer_marker["screen_xy"]],
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
                "sort_key": "camera_distance",
                "camera_distance_order_near_to_far": [str(item["point_label"]) for item in sorted_relabeled_by_depth],
                "marker_id_order_near_to_far": [str(item["marker_id"]) for item in sorted_relabeled_by_depth],
                "screen_y_order_front_to_back": [str(item["point_label"]) for item in sorted_relabeled_by_screen_depth],
                "camera_distances_by_label": {
                    str(item["point_label"]): round(float(item["camera_distance"]), 4)
                    for item in sorted(relabeled_markers, key=lambda marker: str(marker["point_label"]))
                },
                "unique_camera_distance_margin": round(float(_min_pairwise([float(item["camera_distance"]) for item in relabeled_markers])), 4),
                "answer_depth_margin": round(float(depth_margin), 4),
                "answer_screen_depth_margin_px": round(float(screen_depth_margin), 3),
            },
        }
    raise ValueError("could not construct a valid 3D marked-point depth scene")




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
class ThreeDSpatialMarkedPointDepthExtremumLabelTask:
    """Choose the marked point closest to or farthest from the camera."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
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
        """Generate one marked-point depth instance with scalar point annotation for the selected visual witness."""
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
            lower=5,
            upper=6,
        )
        context_object_count, context_object_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="context_object_count",
            default_min=6,
            default_max=6,
            lower=4,
            upper=8,
        )
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_marked_point_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
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
            dynamic_slots={},
            scene_kind="three_d_object_scene_marked_point_depth",
            count_params={
                "context_object_count": int(context_object_count),
                "context_object_count_probabilities": dict(context_object_count_probabilities),
            },
            relation_fields={
                "context_object_count": int(context_object_count),
                "small_context_object_count": int(dataset["small_context_object_count"]),
                "large_context_object_count": int(dataset["large_context_object_count"]),
                "object_count": int(len(dataset["object_specs"])),
            },
            execution_extra={
                "context_object_count": int(context_object_count),
                "small_context_object_count": int(dataset["small_context_object_count"]),
                "large_context_object_count": int(dataset["large_context_object_count"]),
                "object_count": int(len(dataset["object_specs"])),
            },
            witness_symbolic={
                "type": "marked_point",
                "ids_by_role": {"selected_point": str(dataset["answer_point_id"])},
                "answer_label": str(dataset["answer_label"]),
            },
            draw_marked_points_fn=_render_marked_point_overlay,
            bbox_union_fn=_bbox_union,
        )



__all__ = [
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDSpatialMarkedPointDepthExtremumLabelTask",
]
