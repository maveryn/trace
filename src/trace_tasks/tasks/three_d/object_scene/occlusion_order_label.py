"""Occlusion-order label task for a synthetic 3D object scene."""

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
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.task_support import shuffled_repeated_support
from ..shared.object_scene import (
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
)
from ..shared.object_scene_output import build_option_label_object_scene_output as _build_option_label_object_scene_output
from .shared.relations import can_place as _can_place
from .shared.relations import finalize_specs as _finalize_specs
from .shared.relations import make_sampled_object as _make_sampled_object
from .shared.relations import prompt_name as _prompt_name
from .shared.relations import set_xy as _set_xy


TASK_ID = "task_three_d__object_scene__occlusion_order_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("occludes_reference",)
REFERENCE_SHAPE_TYPES: Tuple[str, ...] = ("platform",)
REFERENCE_DIMENSIONS_BY_SHAPE: Dict[str, Tuple[float, float, float]] = {
    "platform": (1.86, 1.12, 1.18),
}
OCCLUSION_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = (
    "bell",
    "open_book",
    "bottle",
    "cactus",
    "candle",
    "cone",
    "cube",
    "cup",
    "cylinder",
    "diamond",
    "dice",
    "glove",
    "half_cylinder",
    "lantern",
    "mushroom",
    "pyramid",
    "shield",
    "trophy",
    "umbrella",
)


def _make_occlusion_reference_spec(*, rng, shape_type: str, xy: Tuple[float, float]) -> Dict[str, Any]:
    """Create a solid reference prop with a broad face for visible occlusion."""
    base_dimensions = REFERENCE_DIMENSIONS_BY_SHAPE[str(shape_type)]
    scale = float(rng.uniform(0.94, 1.10))
    spec = _make_object_spec(
        object_id=f"reference_{shape_type}",
        shape_type=str(shape_type),
        object_role="context",
        xy=xy,
        dimensions_xyz=tuple(round(float(value) * scale, 4) for value in base_dimensions),
        dimension_scale=round(float(scale), 4),
    )
    spec.update(
        {
            "object_name": "platform",
            "prompt_name": "rectangular platform",
            "nameable_for_prompt": True,
            "occlusion_reference_role": "solid_platform",
        }
    )
    return spec


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
def _unit_towards_camera(reference_spec: Mapping[str, Any], camera) -> Tuple[float, float]:
    ref_x, ref_y, _ref_z = (float(value) for value in reference_spec["world_xyz"])
    dx = float(camera.camera_position[0]) - float(ref_x)
    dy = float(camera.camera_position[1]) - float(ref_y)
    length = max(1e-6, math.hypot(dx, dy))
    return (float(dx / length), float(dy / length))


def _axis_extent(spec: Mapping[str, Any], axis_xy: Tuple[float, float]) -> float:
    """Return the half footprint extent of an axis-aligned object along a floor axis."""
    width, depth, _height = (float(value) for value in spec["dimensions_xyz"])
    axis_x, axis_y = (float(value) for value in axis_xy)
    return 0.5 * (abs(axis_x) * float(width) + abs(axis_y) * float(depth))


def _front_gap_to_reference(
    candidate_spec: Mapping[str, Any],
    reference_spec: Mapping[str, Any],
    *,
    axis_xy: Tuple[float, float],
) -> float:
    """Measure candidate clearance beyond the camera-facing reference face."""
    cand_x, cand_y, _cand_z = (float(value) for value in candidate_spec["world_xyz"])
    ref_x, ref_y, _ref_z = (float(value) for value in reference_spec["world_xyz"])
    axis_x, axis_y = (float(value) for value in axis_xy)
    center_delta = (float(cand_x) - float(ref_x)) * float(axis_x) + (float(cand_y) - float(ref_y)) * float(axis_y)
    required_separation = _axis_extent(reference_spec, axis_xy) + _axis_extent(candidate_spec, axis_xy)
    return float(center_delta - required_separation)


def _occluding_answer_xy(
    answer_spec: Mapping[str, Any],
    reference_spec: Mapping[str, Any],
    *,
    camera,
    rng,
) -> Tuple[float, float]:
    """Place the answer just in front of the camera-facing platform face."""
    unit_x, unit_y = _unit_towards_camera(reference_spec, camera)
    side_x, side_y = -unit_y, unit_x
    ref_x, ref_y, _ref_z = (float(value) for value in reference_spec["world_xyz"])
    forward_offset = (
        _axis_extent(reference_spec, (unit_x, unit_y))
        + _axis_extent(answer_spec, (unit_x, unit_y))
        + float(rng.uniform(0.06, 0.14))
    )
    lateral_offset = float(rng.uniform(-0.15, 0.15))
    return (
        float(ref_x + unit_x * forward_offset + side_x * lateral_offset),
        float(ref_y + unit_y * forward_offset + side_y * lateral_offset),
    )


def _sample_distractor_specs(
    *,
    rng,
    labels: Sequence[str],
    shape_pool: Sequence[str],
    placed: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Sample occlusion distractors without changing the unique front or back ordering of the target pair."""
    ring_slots = [
        (-2.54, -2.18),
        (-1.16, -2.58),
        (1.18, -2.52),
        (2.52, -1.42),
        (2.44, 1.42),
        (1.02, 2.52),
        (-1.34, 2.46),
        (-2.52, 1.18),
    ]
    rng.shuffle(ring_slots)
    distractors: List[Dict[str, Any]] = []
    shape_order = shuffled_repeated_support(rng, shape_pool, len(labels))
    for index, (label, shape) in enumerate(zip(labels, shape_order)):
        placed_spec: Dict[str, Any] | None = None
        for slot_x, slot_y in ring_slots[index:] + ring_slots[:index]:
            for _jitter_attempt in range(6):
                spec = _make_sampled_object(
                    rng=rng,
                    object_id=f"object_{label}",
                    shape_type=shape,
                    object_role="candidate",
                    xy=(float(slot_x + rng.uniform(-0.13, 0.13)), float(slot_y + rng.uniform(-0.13, 0.13))),
                    label=str(label),
                )
                if _can_place(spec, placed, clearance=0.12):
                    placed_spec = spec
                    break
            if placed_spec is not None:
                break
        if placed_spec is None:
            raise ValueError("could not place 3D occlusion distractor")
        distractors.append(placed_spec)
        placed.append(placed_spec)
    return list(distractors)


def _build_occlusion_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    point_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
    answer_label_index: int | None = None,
    camera_yaw_band: Tuple[float, float] | None = None,
) -> Dict[str, Any]:
    """Build an occlusion scene where projected overlap yields one unique visible blocker."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    label_selection_index = int(answer_label_index) if answer_label_index is not None else int(instance_seed)
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )
    for _attempt in range(360):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        labels = [str(label) for label in POINT_LABELS[: int(point_count)]]
        answer_label = labels[abs(int(label_selection_index)) % int(point_count)]
        remaining_labels = [str(label) for label in labels if str(label) != str(answer_label)]
        rng.shuffle(remaining_labels)

        reference_shape = str(rng.choice(REFERENCE_SHAPE_TYPES))
        reference_spec = _make_occlusion_reference_spec(
            rng=rng,
            shape_type=reference_shape,
            xy=(float(rng.uniform(-0.16, 0.16)), float(rng.uniform(-0.12, 0.18))),
        )
        if not bool(reference_spec.get("nameable_for_prompt", False)):
            continue

        shape_pool = list(OCCLUSION_CANDIDATE_SHAPE_TYPES)
        rng.shuffle(shape_pool)
        answer_shape = str(shape_pool[0])
        answer_spec = _make_sampled_object(
            rng=rng,
            object_id=f"object_{answer_label}",
            shape_type=answer_shape,
            object_role="candidate",
            xy=(0.0, 0.0),
            label=answer_label,
        )

        unit_x, unit_y = _unit_towards_camera(reference_spec, camera)
        answer_xy = _occluding_answer_xy(answer_spec, reference_spec, camera=camera, rng=rng)
        if max(abs(answer_xy[0]), abs(answer_xy[1])) > float(render_params.room_extent) - 0.35:
            continue
        answer_spec = _set_xy(answer_spec, answer_xy)
        answer_front_gap = _front_gap_to_reference(answer_spec, reference_spec, axis_xy=(unit_x, unit_y))
        if float(answer_front_gap) < 0.045:
            continue

        placed = [dict(reference_spec), dict(answer_spec)]
        distractor_shapes = [str(shape) for shape in shape_pool[1:]] + [str(shape) for shape in shape_pool[:1]]
        try:
            distractor_specs = _sample_distractor_specs(
                rng=rng,
                labels=remaining_labels,
                shape_pool=distractor_shapes,
                placed=placed,
            )
        except ValueError:
            continue

        candidate_specs = [answer_spec, *distractor_specs]
        if len(candidate_specs) != int(point_count):
            continue
        if int(context_object_count) != 1:
            raise ValueError("occlusion-order scenes use exactly one named reference object")

        all_specs = [*candidate_specs, reference_spec]
        reference_points = [point for spec in all_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_candidates = _finalize_specs(candidate_specs, camera=camera, frame=frame)
        reference_spec = _finalize_specs([reference_spec], camera=camera, frame=frame)[0]
        all_finalized = [*finalized_candidates, reference_spec]

        reference_bbox = _object_screen_bbox(reference_spec, camera, frame, pad_px=0.0)
        candidate_bboxes = {
            str(spec["point_label"]): _object_screen_bbox(spec, camera, frame, pad_px=0.0)
            for spec in finalized_candidates
        }
        padded_candidate_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=16.0) for spec in finalized_candidates]
        all_padded_bboxes = [
            (str(spec["object_id"]), _object_screen_bbox(spec, camera, frame, pad_px=16.0))
            for spec in all_finalized
        ]
        answer_bbox = candidate_bboxes[str(answer_label)]
        answer_overlap = float(_bbox_intersection_area(answer_bbox, reference_bbox))
        answer_overlap_fraction = answer_overlap / max(1.0, min(_bbox_area(answer_bbox), _bbox_area(reference_bbox)))
        answer_depth_margin = float(reference_spec["camera_distance"]) - float(
            next(spec for spec in finalized_candidates if str(spec["point_label"]) == str(answer_label))["camera_distance"]
        )
        if answer_overlap < 900.0 or answer_overlap_fraction < 0.12 or answer_depth_margin < 0.22:
            continue

        overlap_area_by_label: Dict[str, float] = {}
        depth_margin_by_label: Dict[str, float] = {}
        occlusion_status_by_label: Dict[str, bool] = {}
        for spec in finalized_candidates:
            label = str(spec["point_label"])
            overlap_area = float(_bbox_intersection_area(candidate_bboxes[label], reference_bbox))
            depth_margin = float(reference_spec["camera_distance"]) - float(spec["camera_distance"])
            overlap_area_by_label[label] = round(float(overlap_area), 4)
            depth_margin_by_label[label] = round(float(depth_margin), 4)
            occlusion_status_by_label[label] = bool(overlap_area >= 650.0 and depth_margin >= 0.18)

        occluding_labels = [str(label) for label, is_occluding in sorted(occlusion_status_by_label.items()) if bool(is_occluding)]
        if occluding_labels != [str(answer_label)]:
            continue
        if any(
            str(label) != str(answer_label) and float(overlap_area) > 300.0
            for label, overlap_area in overlap_area_by_label.items()
        ):
            continue

        candidate_screen_centers = [
            (float(spec["screen_xy"][0]), float(spec["screen_xy"][1]))
            for spec in finalized_candidates
        ]
        if any(
            math.hypot(a[0] - b[0], a[1] - b[1]) < 34.0
            for index, a in enumerate(candidate_screen_centers)
            for b in candidate_screen_centers[index + 1 :]
        ):
            continue
        if any(
            _bbox_intersection_area(a, b) > 12000.0
            for index, a in enumerate(padded_candidate_bboxes)
            for b in padded_candidate_bboxes[index + 1 :]
        ):
            continue
        intended_reference_answer_pair = {str(reference_spec["object_id"]), f"object_{answer_label}"}
        if any(
            _bbox_intersection_area(a, b) > 22000.0
            for index, (a_id, a) in enumerate(all_padded_bboxes)
            for b_id, b in all_padded_bboxes[index + 1 :]
            if {str(a_id), str(b_id)} != intended_reference_answer_pair
        ):
            continue

        sorted_candidates = sorted(finalized_candidates, key=lambda spec: str(spec["point_label"]))
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "candidate_count": int(point_count),
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
            "reference_occlusion_role": str(reference_spec.get("occlusion_reference_role", "")),
            "answer_front_gap_to_reference": round(float(answer_front_gap), 4),
            "candidate_reference_overlap_area_by_label": dict(sorted(overlap_area_by_label.items())),
            "candidate_depth_margin_to_reference_by_label": dict(sorted(depth_margin_by_label.items())),
            "occlusion_status_by_label": dict(sorted(occlusion_status_by_label.items())),
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
                "sort_key": "projected_occlusion_with_depth_order",
                "candidate_only": True,
                "reference_excluded_from_options": True,
                "reference_object_id": str(reference_spec["object_id"]),
                "reference_object_name": _prompt_name(reference_spec),
                "reference_shape_type": str(reference_spec["shape_type"]),
                "reference_occlusion_role": str(reference_spec.get("occlusion_reference_role", "")),
                "answer_front_gap_to_reference": round(float(answer_front_gap), 4),
                "candidate_reference_overlap_area_by_label": dict(sorted(overlap_area_by_label.items())),
                "candidate_depth_margin_to_reference_by_label": dict(sorted(depth_margin_by_label.items())),
                "occlusion_status_by_label": dict(sorted(occlusion_status_by_label.items())),
                "occluding_reference_labels": list(occluding_labels),
                "unique_occlusion_answer": True,
                "occluding_depth_margin": round(float(answer_depth_margin), 4),
                "occluding_overlap_area": round(float(answer_overlap), 4),
            },
        }
    raise ValueError("could not construct a valid 3D occlusion-order scene")






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
class ThreeDSpatialOcclusionOrderLabelTask:
    """Choose the lettered 3D object that visibly occludes a named reference object."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
    supported_query_ids = SUPPORTED_QUERY_IDS
    domain = "three_d"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        last_error: Exception | None = None
        camera_yaw_band = _camera_yaw_band_for_instance(int(instance_seed))
        answer_seed = int(instance_seed)
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt_index == 0
                else int(spawn_rng(int(instance_seed), f"{TASK_ID}.attempt_seed.{attempt_index}").randrange(1, 2**62))
            )
            try:
                return self._generate_once(
                    int(attempt_seed),
                    params=params,
                    camera_yaw_band=camera_yaw_band,
                    answer_seed=answer_seed,
                )
            except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
                last_error = exc
        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}")

    def _generate_once(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        camera_yaw_band: Tuple[float, float] | None = None,
        answer_seed: int | None = None,
    ) -> TaskOutput:
        """Generate one occlusion-order instance with prompt, answer, and annotation tied to the accepted overlap trace."""
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
        context_object_count, context_object_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            prefix="context_object_count",
            minimum_default=1,
            maximum_default=1,
            lower=1,
            upper=1,
        )
        answer_rng = spawn_rng(
            int(answer_seed) if answer_seed is not None else int(instance_seed),
            f"{TASK_ID}.answer_label",
        )
        answer_label_index = int(answer_rng.randrange(int(point_count)))
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_occlusion_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            context_object_count=int(context_object_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
            answer_label_index=int(answer_label_index),
            camera_yaw_band=camera_yaw_band,
        )
        relation_fields = {
            "reference_object_id": str(dataset["reference_object_id"]),
            "reference_object_name": str(dataset["reference_object_name"]),
            "reference_shape_type": str(dataset["reference_shape_type"]),
            "occlusion_status_by_label": dict(dataset["occlusion_status_by_label"]),
            "candidate_reference_overlap_area_by_label": dict(dataset["candidate_reference_overlap_area_by_label"]),
            "candidate_depth_margin_to_reference_by_label": dict(dataset["candidate_depth_margin_to_reference_by_label"]),
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
            context_object_count=int(context_object_count),
            render_params=render_params,
            prompt_defaults_config=_PROMPT_DEFAULTS,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            query_probabilities=query_probabilities,
            scene_probabilities=scene_probabilities,
            point_count_probabilities=point_count_probabilities,
            context_object_count_probabilities=context_object_count_probabilities,
            dynamic_slots={
                "reference_name": str(dataset["reference_object_name"]),
            },
            relation_fields=relation_fields,
            query_params_extra={
                "reference_object_id": str(dataset["reference_object_id"]),
                "reference_object_name": str(dataset["reference_object_name"]),
            },
            execution_extra={
                "reference_object_id": str(dataset["reference_object_id"]),
                "reference_object_name": str(dataset["reference_object_name"]),
                "reference_shape_type": str(dataset["reference_shape_type"]),
                "reference_occlusion_role": str(dataset["reference_occlusion_role"]),
                "occlusion_status_by_label": dict(dataset["occlusion_status_by_label"]),
                "candidate_reference_overlap_area_by_label": dict(dataset["candidate_reference_overlap_area_by_label"]),
                "candidate_depth_margin_to_reference_by_label": dict(dataset["candidate_depth_margin_to_reference_by_label"]),
            },
        )



__all__ = ["ThreeDSpatialOcclusionOrderLabelTask"]
