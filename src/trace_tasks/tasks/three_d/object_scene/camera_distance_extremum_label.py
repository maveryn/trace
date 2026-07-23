"""Camera-distance extremum task for a synthetic 3D object scene."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
)
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)
from ..shared.task_support import (
    resolve_axis_variant as _shared_resolve_axis_variant,
    resolve_count as _shared_resolve_count,
)
from ..shared.color_variation import resolve_three_d_object_fill_rgb
from ..shared.camera_projection import (
    CameraSpec as _CameraSpec,
    ProjectionFrame as _ProjectionFrame,
    build_projection_frame as _build_projection_frame,
    canvas_floor_polygon_xy as _canvas_floor_polygon_xy,
    dedupe_line_points as _dedupe_line_points,
    distance as _distance,
    grid_values_for_range as _grid_values_for_range,
    min_pairwise as _min_pairwise,
    polygon_axis_line_segment as _polygon_axis_line_segment,
    project_normalized as _project_normalized,
    project_screen as _project_screen,
    project_xy as _project_xy,
    sample_camera as _sample_camera,
    screen_to_floor_xy as _screen_to_floor_xy,
    screen_to_normalized as _screen_to_normalized,
    stage_reference_points as _stage_reference_points,
    vec_cross as _vec_cross,
    vec_dot as _vec_dot,
    vec_norm as _vec_norm,
    vec_sub as _vec_sub,
)
from ..shared.object_resources import (
    OBJECT_SCENE_CONTEXT_DIMENSIONS,
    OBJECT_SCENE_CONTEXT_SHAPE_TYPES,
    OBJECT_SCENE_NAME_BY_SHAPE_TYPE,
    OBJECT_SCENE_SHAPE_TYPES,
    OBJECT_SCENE_SMALL_DIMENSIONS,
    OBJECT_SCENE_SMALL_SHAPE_TYPES,
)
from ..shared.object_scene import (
    CAMERA_YAW_BANDS_DEGREES,
    CONTEXT_OBJECT_COLORS,
    LARGE_CONTEXT_SHAPE_TYPES,
    NAMEABLE_CONTEXT_SHAPE_TYPES,
    NAMED_SMALL_OBJECT_SHAPE_TYPES,
    OBJECT_NAME_BY_SHAPE_TYPE,
    POINT_COLORS,
    POINT_LABELS,
    SCENE_ID,
    SHAPE_TYPES,
    SMALL_OBJECT_SHAPE_TYPES,
    SUPPORTED_SCENE_VARIANTS,
    _RenderedScene,
    _RenderParams,
    _base_shape_dimensions,
    _bbox_intersection_area,
    _bool_value,
    _camera_from_dataset,
    _camera_yaw_band_for_instance,
    _frame_from_dataset,
    _make_object_spec,
    _nameable_for_prompt,
    _object_name,
    _object_reference_points,
    _object_screen_bbox,
    _resolve_render_params,
    _sample_scene_object_specs,
    _sample_shape_dimensions,
)
from ..shared.object_scene_output import build_option_label_object_scene_output as _build_option_label_object_scene_output


TASK_ID = "task_three_d__object_scene__camera_distance_extremum_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("closest_to_camera", "farthest_from_camera")
UNRELIABLE_CAMERA_DISTANCE_ANSWER_SHAPES: Tuple[str, ...] = ("horseshoe",)
MIN_ANSWER_BBOX_SIDE_PX = 28.0
MIN_ANSWER_BBOX_AREA_PX = 1800.0
MAX_ANSWER_CONTEXT_OVERLAP_FRACTION = 0.18
MAX_CANDIDATE_CONTEXT_OVERLAP_FRACTION = 0.48


def _resolve_point_count(params: Mapping[str, Any], *, gen_defaults: Mapping[str, Any], instance_seed: int) -> Tuple[int, Dict[str, float]]:
    return _shared_resolve_count(
        params=params,
        task_id=TASK_ID,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="point_count",
        default_min=5,
        default_max=7,
        lower=3,
        upper=8,
    )


def _resolve_context_object_count(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[int, Dict[str, float]]:
    return _shared_resolve_count(
        params=params,
        task_id=TASK_ID,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        key="context_object_count",
        default_min=2,
        default_max=2,
        lower=0,
        upper=3,
    )


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_min_side(bbox: Sequence[float]) -> float:
    return min(max(0.0, float(bbox[2]) - float(bbox[0])), max(0.0, float(bbox[3]) - float(bbox[1])))


def _max_context_overlap_fraction(candidate_bbox: Sequence[float], context_bboxes: Sequence[Sequence[float]]) -> float:
    candidate_area = max(1.0, _bbox_area(candidate_bbox))
    if not context_bboxes:
        return 0.0
    return max(
        float(_bbox_intersection_area(candidate_bbox, context_bbox)) / float(candidate_area)
        for context_bbox in context_bboxes
    )




def _build_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    point_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build a camera-distance extremum scene with one readable nearest or farthest candidate in the rendered view."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )
    for _attempt in range(300):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        point_specs, context_object_specs = _sample_scene_object_specs(
            rng=rng,
            candidate_count=int(point_count),
            context_object_count=int(context_object_count),
        )
        all_specs = list(point_specs) + list(context_object_specs)
        reference_points = [point for spec in all_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        screens = [_project_screen(spec["world_xyz"], camera, frame) for spec in point_specs]
        context_screens = [_project_screen(spec["world_xyz"], camera, frame) for spec in context_object_specs]
        screen_centers = [(screen[0], screen[1]) for screen in screens]
        screen_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=16.0) for spec in point_specs]
        candidate_readability_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=4.0) for spec in point_specs]
        context_readability_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=4.0) for spec in context_object_specs]
        all_screen_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=16.0) for spec in all_specs]
        if any(
            math.hypot(a[0] - b[0], a[1] - b[1]) < 54.0
            for index, a in enumerate(screen_centers)
            for b in screen_centers[index + 1 :]
        ):
            continue
        if any(
            _bbox_intersection_area(a, b) > 3200.0
            for index, a in enumerate(screen_bboxes)
            for b in screen_bboxes[index + 1 :]
        ):
            continue
        if any(
            _bbox_intersection_area(a, b) > 9200.0
            for index, a in enumerate(all_screen_bboxes)
            for b in all_screen_bboxes[index + 1 :]
        ):
            continue
        camera_distances = [float(screen[7]) for screen in screens]
        if _min_pairwise(camera_distances) < 0.24:
            continue
        finalized_specs: List[Dict[str, Any]] = []
        for index, spec in enumerate(point_specs):
            screen = screens[index]
            finalized = dict(spec)
            finalized.update(
                {
                    "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                    "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                    "camera_distance": round(float(screen[7]), 4),
                }
            )
            finalized_specs.append(finalized)
        finalized_context_specs: List[Dict[str, Any]] = []
        for index, spec in enumerate(context_object_specs):
            screen = context_screens[index]
            finalized = dict(spec)
            finalized.update(
                {
                    "screen_xy": [round(float(screen[0]), 3), round(float(screen[1]), 3)],
                    "camera_xyz": [round(float(screen[5]), 4), round(float(screen[6]), 4), round(float(screen[4]), 4)],
                    "camera_distance": round(float(screen[7]), 4),
                }
            )
            finalized_context_specs.append(finalized)
        pre_label_sorted_by_distance = sorted(finalized_specs, key=lambda spec: (float(spec["camera_distance"]), str(spec["object_id"])))
        if str(query_id) == "closest_to_camera":
            answer_object_id = str(pre_label_sorted_by_distance[0]["object_id"])
        else:
            answer_object_id = str(pre_label_sorted_by_distance[-1]["object_id"])
        answer_pre_label_index = next(
            index for index, spec in enumerate(point_specs) if str(spec["object_id"]) == str(answer_object_id)
        )
        answer_bbox = candidate_readability_bboxes[int(answer_pre_label_index)]
        answer_overlap_fraction = _max_context_overlap_fraction(answer_bbox, context_readability_bboxes)
        if str(point_specs[int(answer_pre_label_index)]["shape_type"]) in set(UNRELIABLE_CAMERA_DISTANCE_ANSWER_SHAPES):
            continue
        if _bbox_min_side(answer_bbox) < MIN_ANSWER_BBOX_SIDE_PX or _bbox_area(answer_bbox) < MIN_ANSWER_BBOX_AREA_PX:
            continue
        if float(answer_overlap_fraction) > MAX_ANSWER_CONTEXT_OVERLAP_FRACTION:
            continue
        if any(
            _max_context_overlap_fraction(candidate_bbox, context_readability_bboxes) > MAX_CANDIDATE_CONTEXT_OVERLAP_FRACTION
            for candidate_bbox in candidate_readability_bboxes
        ):
            continue
        query_offset = 0 if str(query_id) == "closest_to_camera" else 3
        answer_label_index = abs(int(instance_seed) + int(query_offset)) % int(point_count)
        answer_label = str(POINT_LABELS[answer_label_index])
        remaining_labels = [str(label) for label in POINT_LABELS[: int(point_count)] if str(label) != answer_label]
        rng.shuffle(remaining_labels)
        relabeled_specs: List[Dict[str, Any]] = []
        for spec in finalized_specs:
            updated = dict(spec)
            label = answer_label if str(updated["object_id"]) == answer_object_id else str(remaining_labels.pop())
            updated.update(
                {
                    "point_id": f"object_{label}",
                    "point_label": str(label),
                    "object_id": f"object_{label}",
                    "object_label": str(label),
                }
            )
            relabeled_specs.append(updated)
        finalized_specs = list(relabeled_specs)
        sorted_by_distance = sorted(finalized_specs, key=lambda spec: (float(spec["camera_distance"]), str(spec["point_label"])))
        answer_spec = next(spec for spec in finalized_specs if str(spec["point_label"]) == str(answer_label))
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "candidate_count": int(point_count),
            "context_object_count": int(context_object_count),
            "object_count": int(point_count) + int(context_object_count),
            "point_specs": sorted(finalized_specs, key=lambda spec: str(spec["point_label"])),
            "context_object_specs": sorted(finalized_context_specs, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted([*finalized_specs, *finalized_context_specs], key=lambda spec: str(spec["object_id"])),
            "answer_label": str(answer_spec["point_label"]),
            "answer_point_id": str(answer_spec["point_id"]),
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
                "candidate_only": True,
                "camera_distance_order_near_to_far": [str(spec["point_label"]) for spec in sorted_by_distance],
                "shape_order_near_to_far": [str(spec["shape_type"]) for spec in sorted_by_distance],
                "context_object_ids": [str(spec["object_id"]) for spec in sorted(finalized_context_specs, key=lambda spec: str(spec["object_id"]))],
                "context_shape_types": [str(spec["shape_type"]) for spec in sorted(finalized_context_specs, key=lambda spec: str(spec["object_id"]))],
                "unique_camera_distance_margin": round(float(_min_pairwise([float(spec["camera_distance"]) for spec in finalized_specs])), 4),
                "answer_context_overlap_fraction": round(float(answer_overlap_fraction), 4),
            },
        }
    raise ValueError("could not construct a valid 3D camera-distance scene")






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
class ThreeDSpatialCameraDistanceExtremumLabelTask:
    """Choose the lettered 3D object closest to or farthest from the camera."""

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
        """Generate one camera-distance extremum instance from a single accepted projection and scalar target bbox."""
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
        point_count, point_count_probabilities = _resolve_point_count(
            params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        context_object_count, context_object_count_probabilities = _resolve_context_object_count(
            params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            context_object_count=int(context_object_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
            camera_yaw_band=camera_yaw_band,
        )
        return _build_option_label_object_scene_output(
            objective_name=TASK_ID,
            task_domain=self.domain,
            instance_seed=int(instance_seed),
            params=params,
            dataset=dataset,
            branch_key=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            context_object_count=int(context_object_count),
            render_params=render_params,
            prompt_defaults_config=_PROMPT_DEFAULTS,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            query_probabilities=query_probabilities,
            scene_probabilities=scene_probabilities,
            point_count_probabilities=point_count_probabilities,
            context_object_count_probabilities=context_object_count_probabilities,
            dynamic_slots={},
            relation_fields={
                "shape_types": [str(spec["shape_type"]) for spec in dataset["object_specs"]],
                "nameable_candidate_object_names": [
                    str(spec["object_name"]) for spec in dataset["point_specs"] if bool(spec.get("nameable_for_prompt", False))
                ],
                "nameable_context_object_names": [
                    str(spec["object_name"]) for spec in dataset["context_object_specs"] if bool(spec.get("nameable_for_prompt", False))
                ],
            },
            query_params_extra={},
            execution_extra={},
        )



__all__ = ["ThreeDSpatialCameraDistanceExtremumLabelTask"]
