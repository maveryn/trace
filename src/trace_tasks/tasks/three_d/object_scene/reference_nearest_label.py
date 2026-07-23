"""Reference-nearest label task for a synthetic 3D object scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

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
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.object_scene import (
    NAMEABLE_CONTEXT_SHAPE_TYPES,
    NAMED_SMALL_OBJECT_SHAPE_TYPES,
    POINT_LABELS,
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
    _resolve_render_params,
    _sample_camera,
    _sample_shape_dimensions,
)
from ..shared.object_scene_output import build_option_label_object_scene_output as _build_option_label_object_scene_output
from .shared.relations import can_place as _can_place
from .shared.relations import finalize_specs as _finalize_specs
from .shared.relations import make_sampled_object as _make_sampled_object
from .shared.relations import prompt_name as _prompt_name


TASK_ID = "task_three_d__object_scene__reference_nearest_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("closest_to_reference", "farthest_from_reference")
REFERENCE_SHAPE_TYPES: Tuple[str, ...] = tuple(
    str(shape) for shape in NAMEABLE_CONTEXT_SHAPE_TYPES if str(shape) not in {"arch", "open_box"}
)
SMALL_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = tuple(NAMED_SMALL_OBJECT_SHAPE_TYPES)
EXCLUDED_REFERENCE_NEAREST_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = ("heart", "remote_control", "sword")


def _surface_gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax, ay, _az = (float(value) for value in a["world_xyz"])
    bx, by, _bz = (float(value) for value in b["world_xyz"])
    center_distance_xy = math.hypot(float(ax - bx), float(ay - by))
    return max(0.0, float(center_distance_xy) - float(a["footprint_radius"]) - float(b["footprint_radius"]))


def _screen_center_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    """Return projected center distance between two finalized object specs."""

    ax, ay = (float(value) for value in a["screen_xy"][:2])
    bx, by = (float(value) for value in b["screen_xy"][:2])
    return math.hypot(float(ax - bx), float(ay - by))


def _sample_candidate_shapes(*, rng, reference_name: str, candidate_count: int) -> List[str]:
    excluded = set(EXCLUDED_REFERENCE_NEAREST_CANDIDATE_SHAPE_TYPES)
    small_pool = [str(shape) for shape in SMALL_CANDIDATE_SHAPE_TYPES if str(shape) not in excluded]
    rng.shuffle(small_pool)
    selected: List[str] = []
    selected_names = {str(reference_name)}
    for shape in small_pool:
        probe = _make_sampled_object(
            rng=rng,
            object_id=f"probe_{shape}",
            shape_type=shape,
            object_role="candidate",
            xy=(0.0, 0.0),
        )
        if _prompt_name(probe) in selected_names:
            continue
        selected.append(str(shape))
        selected_names.add(_prompt_name(probe))
        if len(selected) >= int(candidate_count):
            break
    if len(selected) < int(candidate_count):
        raise ValueError("could not sample enough unique candidate object names")
    return list(selected[: int(candidate_count)])


def _build_reference_nearest_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    point_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build a reference-distance scene with one large prop and six small named candidates."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )
    for _attempt in range(420):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        reference_shape = str(rng.choice(REFERENCE_SHAPE_TYPES))
        reference_spec = _make_sampled_object(
            rng=rng,
            object_id=f"reference_{reference_shape}",
            shape_type=reference_shape,
            object_role="context",
            xy=(float(rng.uniform(-0.12, 0.12)), float(rng.uniform(-0.10, 0.16))),
        )
        if not bool(reference_spec.get("nameable_for_prompt", False)):
            continue

        labels = [str(label) for label in POINT_LABELS[: int(point_count)]]
        answer_label = labels[abs(int(instance_seed)) % int(point_count)]

        candidate_shapes = _sample_candidate_shapes(
            rng=rng,
            reference_name=_prompt_name(reference_spec),
            candidate_count=int(point_count),
        )
        placed: List[Dict[str, Any]] = [dict(reference_spec)]
        ref_x, ref_y, _ref_z = (float(value) for value in reference_spec["world_xyz"])
        ref_radius = float(reference_spec["footprint_radius"])

        unit_x = float(camera.camera_position[0] - ref_x)
        unit_y = float(camera.camera_position[1] - ref_y)
        unit_len = max(1e-6, math.hypot(unit_x, unit_y))
        unit_x /= unit_len
        unit_y /= unit_len
        side_x, side_y = -unit_y, unit_x

        label_shape_pairs = list(zip(labels, candidate_shapes, strict=True))
        rng.shuffle(label_shape_pairs)
        base_gaps = (
            [0.20, 0.46, 0.72, 0.98, 1.24, 1.50]
            if str(query_id) == "closest_to_reference"
            else [0.14, 0.32, 0.50, 0.68, 0.86, 1.14]
        )
        gap_values = [float(value + rng.uniform(-0.035, 0.035)) for value in base_gaps]
        extremal_gap = min(gap_values) if str(query_id) == "closest_to_reference" else max(gap_values)
        remaining_gaps = [gap for gap in gap_values if float(gap) != float(extremal_gap)]
        rng.shuffle(remaining_gaps)
        gap_by_label: Dict[str, float] = {}
        for label, _shape in label_shape_pairs:
            if str(label) == str(answer_label):
                gap_by_label[str(label)] = float(extremal_gap)
            else:
                gap_by_label[str(label)] = float(remaining_gaps.pop())

        if str(query_id) == "closest_to_reference":
            answer_angle_offset = float(rng.uniform(-0.12, 0.12))
            distractor_angle_offsets = [-1.12, -0.74, -0.38, 0.38, 0.74, 1.12]
        else:
            answer_angle_offset = float(rng.uniform(-0.12, 0.12))
            distractor_angle_offsets = [-1.12, -0.74, -0.38, 0.38, 0.74, 1.12]
        rng.shuffle(distractor_angle_offsets)
        angle_by_label: Dict[str, float] = {}
        for label, _shape in label_shape_pairs:
            if str(label) == str(answer_label):
                angle_by_label[str(label)] = float(answer_angle_offset)
            else:
                angle_by_label[str(label)] = float(distractor_angle_offsets.pop())
        candidate_specs: List[Dict[str, Any]] = []
        for index, (label, shape) in enumerate(label_shape_pairs):
            dimensions, scale = _sample_shape_dimensions(shape, object_role="candidate", rng=rng)
            probe = _make_object_spec(
                object_id=f"object_{label}",
                shape_type=str(shape),
                object_role="candidate",
                xy=(0.0, 0.0),
                dimensions_xyz=dimensions,
                dimension_scale=float(scale),
                label=str(label),
            )
            radius = float(probe["footprint_radius"])
            gap = float(gap_by_label[str(label)])
            center_distance = float(ref_radius + radius + gap)
            angle_offset = float(angle_by_label[str(label)] + rng.uniform(-0.04, 0.04))
            forward = math.cos(angle_offset)
            lateral = math.sin(angle_offset)
            placed_spec: Dict[str, Any] | None = None
            for _jitter_attempt in range(10):
                jittered_distance = center_distance + float(rng.uniform(-0.025, 0.025))
                jittered_lateral = lateral + float(rng.uniform(-0.025, 0.025))
                xy = (
                    float(ref_x + unit_x * forward * jittered_distance + side_x * jittered_lateral * jittered_distance),
                    float(ref_y + unit_y * forward * jittered_distance + side_y * jittered_lateral * jittered_distance),
                )
                if max(abs(xy[0]), abs(xy[1])) > float(render_params.room_extent) - 0.28:
                    continue
                spec = _make_object_spec(
                    object_id=f"object_{label}",
                    shape_type=str(shape),
                    object_role="candidate",
                    xy=xy,
                    dimensions_xyz=dimensions,
                    dimension_scale=float(scale),
                    label=str(label),
                )
                to_ref_x = float(spec["world_xyz"][0]) - float(ref_x)
                to_ref_y = float(spec["world_xyz"][1]) - float(ref_y)
                camera_axis = to_ref_x * unit_x + to_ref_y * unit_y
                if camera_axis < float(ref_radius * 0.10):
                    continue
                if not _can_place(spec, placed, clearance=0.12):
                    continue
                placed_spec = spec
                break
            if placed_spec is None:
                break
            candidate_specs.append(placed_spec)
            placed.append(placed_spec)
        if len(candidate_specs) != int(point_count):
            continue

        all_specs = [*candidate_specs, reference_spec]
        reference_points = [point for spec in all_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_candidates = _finalize_specs(candidate_specs, camera=camera, frame=frame)
        finalized_context = _finalize_specs([reference_spec], camera=camera, frame=frame)
        reference_spec = finalized_context[0]
        all_finalized = [*finalized_candidates, reference_spec]

        candidate_screen_centers = [
            (float(spec["screen_xy"][0]), float(spec["screen_xy"][1]))
            for spec in finalized_candidates
        ]
        candidate_screen_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=16.0) for spec in finalized_candidates]
        all_screen_bboxes = [
            (str(spec["object_id"]), _object_screen_bbox(spec, camera, frame, pad_px=16.0))
            for spec in all_finalized
        ]
        reference_bbox = _object_screen_bbox(reference_spec, camera, frame, pad_px=8.0)
        if any(
            math.hypot(a[0] - b[0], a[1] - b[1]) < 34.0
            for index, a in enumerate(candidate_screen_centers)
            for b in candidate_screen_centers[index + 1 :]
        ):
            continue
        if any(
            _bbox_intersection_area(a, b) > 15000.0
            for index, a in enumerate(candidate_screen_bboxes)
            for b in candidate_screen_bboxes[index + 1 :]
        ):
            continue
        if any(
            _bbox_intersection_area(a, b) > 42000.0
            for index, (_a_id, a) in enumerate(all_screen_bboxes)
            for _b_id, b in all_screen_bboxes[index + 1 :]
        ):
            continue
        hidden_by_reference = False
        for spec in finalized_candidates:
            candidate_bbox = _object_screen_bbox(spec, camera, frame, pad_px=8.0)
            overlap = _bbox_intersection_area(candidate_bbox, reference_bbox)
            candidate_area = max(
                1.0,
                (float(candidate_bbox[2]) - float(candidate_bbox[0]))
                * (float(candidate_bbox[3]) - float(candidate_bbox[1])),
            )
            if overlap > 0.24 * candidate_area:
                hidden_by_reference = True
                break
            if float(spec["camera_distance"]) > float(reference_spec["camera_distance"]) + 0.04:
                hidden_by_reference = True
                break
        if hidden_by_reference:
            continue

        gaps_by_label = {
            str(spec["point_label"]): round(float(_surface_gap(reference_spec, spec)), 4)
            for spec in finalized_candidates
        }
        screen_gaps_by_label = {
            str(spec["point_label"]): round(
                float(_screen_center_distance(reference_spec, spec)),
                4,
            )
            for spec in finalized_candidates
        }
        sorted_by_gap = sorted(finalized_candidates, key=lambda spec: (float(gaps_by_label[str(spec["point_label"])]), str(spec["point_label"])))
        sorted_by_screen_gap = sorted(
            finalized_candidates,
            key=lambda spec: (float(screen_gaps_by_label[str(spec["point_label"])]), str(spec["point_label"])),
        )
        screen_target_spec = sorted_by_screen_gap[0] if str(query_id) == "closest_to_reference" else sorted_by_screen_gap[-1]
        answer_label = str(screen_target_spec["point_label"])
        if _min_pairwise([float(value) for value in screen_gaps_by_label.values()]) < 10.0:
            continue
        if str(query_id) == "closest_to_reference" and len(sorted_by_gap) > 1:
            surface_margin = float(gaps_by_label[str(sorted_by_gap[1]["point_label"])]) - float(gaps_by_label[str(sorted_by_gap[0]["point_label"])])
            screen_margin = float(screen_gaps_by_label[str(sorted_by_screen_gap[1]["point_label"])]) - float(
                screen_gaps_by_label[str(sorted_by_screen_gap[0]["point_label"])]
            )
            if screen_margin < 16.0:
                continue
        elif str(query_id) == "farthest_from_reference" and len(sorted_by_gap) > 1:
            surface_margin = float(gaps_by_label[str(sorted_by_gap[-1]["point_label"])]) - float(gaps_by_label[str(sorted_by_gap[-2]["point_label"])])
            screen_margin = float(screen_gaps_by_label[str(sorted_by_screen_gap[-1]["point_label"])]) - float(
                screen_gaps_by_label[str(sorted_by_screen_gap[-2]["point_label"])]
            )
            if screen_margin < 16.0:
                continue
        else:
            surface_margin = 999.0
            screen_margin = 999.0

        sorted_candidates = sorted(finalized_candidates, key=lambda spec: str(spec["point_label"]))
        distance_order = [str(spec["point_label"]) for spec in sorted_by_gap]
        screen_distance_order = [str(spec["point_label"]) for spec in sorted_by_screen_gap]
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "candidate_count": int(point_count),
            "large_candidate_count": 0,
            "context_object_count": 1,
            "object_count": int(point_count) + 1,
            "point_specs": list(sorted_candidates),
            "context_object_specs": [dict(reference_spec)],
            "object_specs": sorted([*sorted_candidates, reference_spec], key=lambda spec: str(spec["object_id"])),
            "answer_label": str(answer_label),
            "answer_point_id": f"object_{answer_label}",
            "reference_object_id": str(reference_spec["object_id"]),
            "reference_object_name": _prompt_name(reference_spec),
            "reference_shape_type": str(reference_spec["shape_type"]),
            "candidate_reference_gaps_by_label": dict(sorted(gaps_by_label.items())),
            "candidate_reference_screen_gaps_by_label": dict(sorted(screen_gaps_by_label.items())),
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
                "sort_key": "projected_screen_center_distance_to_reference",
                "candidate_only": True,
                "reference_excluded_from_options": True,
                "reference_object_id": str(reference_spec["object_id"]),
                "reference_object_name": _prompt_name(reference_spec),
                "reference_shape_type": str(reference_spec["shape_type"]),
                "candidate_reference_gaps_by_label": dict(sorted(gaps_by_label.items())),
                "candidate_reference_screen_gaps_by_label": dict(sorted(screen_gaps_by_label.items())),
                "reference_distance_order": list(distance_order),
                "reference_screen_distance_order": list(screen_distance_order),
                "reference_nearest_order": list(screen_distance_order),
                "screen_distance_agrees_with_surface_gap": (
                    str(distance_order[0 if str(query_id) == "closest_to_reference" else -1])
                    == str(screen_distance_order[0 if str(query_id) == "closest_to_reference" else -1])
                ),
                "unique_reference_nearest_answer": True,
                "reference_nearest_margin": round(float(screen_margin), 4),
                "reference_nearest_surface_margin": round(float(surface_margin), 4),
                "reference_nearest_screen_margin_px": round(float(screen_margin), 4),
            },
        }
    raise ValueError("could not construct a valid 3D reference-nearest scene")






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
class ThreeDSpatialReferenceNearestLabelTask:
    """Choose the lettered 3D object closest to a named reference object."""

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
        """Generate one reference-nearest instance while preserving the accepted distance ranking in trace and annotation."""
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
            prefix="point_count",
            minimum_default=6,
            maximum_default=6,
            lower=4,
            upper=8,
        )
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_reference_nearest_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
            camera_yaw_band=camera_yaw_band,
        )
        relation_fields = {
            "large_candidate_count": int(dataset["large_candidate_count"]),
            "context_object_count": int(dataset["context_object_count"]),
            "reference_object_id": str(dataset["reference_object_id"]),
            "reference_object_name": str(dataset["reference_object_name"]),
            "reference_shape_type": str(dataset["reference_shape_type"]),
            "candidate_reference_gaps_by_label": dict(dataset["candidate_reference_gaps_by_label"]),
            "candidate_reference_screen_gaps_by_label": dict(dataset["candidate_reference_screen_gaps_by_label"]),
            "answer_point_id": str(dataset["answer_point_id"]),
        }
        return _build_option_label_object_scene_output(
            objective_name=TASK_ID,
            task_domain=self.domain,
            instance_seed=int(instance_seed),
            params=params,
            dataset=dataset,
            branch_key=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            context_object_count=int(dataset["context_object_count"]),
            render_params=render_params,
            prompt_defaults_config=_PROMPT_DEFAULTS,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            query_probabilities=query_probabilities,
            scene_probabilities=scene_probabilities,
            point_count_probabilities=point_count_probabilities,
            context_object_count_probabilities={str(dataset["context_object_count"]): 1.0},
            dynamic_slots={
                "reference_name": str(dataset["reference_object_name"]),
            },
            relation_fields=relation_fields,
            query_params_extra={
                "reference_object_id": str(dataset["reference_object_id"]),
                "reference_object_name": str(dataset["reference_object_name"]),
            },
            execution_extra={
                "large_candidate_count": int(dataset["large_candidate_count"]),
                "context_object_count": int(dataset["context_object_count"]),
                "reference_object_id": str(dataset["reference_object_id"]),
                "reference_object_name": str(dataset["reference_object_name"]),
                "reference_shape_type": str(dataset["reference_shape_type"]),
                "candidate_reference_gaps_by_label": dict(dataset["candidate_reference_gaps_by_label"]),
                "candidate_reference_screen_gaps_by_label": dict(dataset["candidate_reference_screen_gaps_by_label"]),
            },
        )



__all__ = ["ThreeDSpatialReferenceNearestLabelTask"]
