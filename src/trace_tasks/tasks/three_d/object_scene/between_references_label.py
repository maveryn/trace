"""Between-two-references label task for a synthetic 3D object scene."""

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
from ..shared.object_resources import SPATIAL_BETWEEN_REFERENCE_SHAPE_TYPES
from ..shared.object_scene import (
    NAMED_SMALL_OBJECT_SHAPE_TYPES,
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
    _resolve_render_params,
    _sample_camera,
)
from ..shared.object_scene_output import build_option_label_object_scene_output as _build_option_label_object_scene_output
from .shared.relations import can_place as _can_place
from .shared.relations import finalize_specs as _finalize_specs
from .shared.relations import make_sampled_object as _make_sampled_object
from .shared.relations import prompt_name as _prompt_name
from .shared.relations import set_xy as _set_xy


TASK_ID = "task_three_d__object_scene__between_references_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("between_references",)
REFERENCE_SHAPE_TYPES: Tuple[str, ...] = SPATIAL_BETWEEN_REFERENCE_SHAPE_TYPES
SMALL_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = tuple(NAMED_SMALL_OBJECT_SHAPE_TYPES)
def _surface_gap(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    ax, ay, _az = (float(value) for value in a["world_xyz"])
    bx, by, _bz = (float(value) for value in b["world_xyz"])
    center_distance_xy = math.hypot(float(ax - bx), float(ay - by))
    return max(0.0, float(center_distance_xy) - float(a["footprint_radius"]) - float(b["footprint_radius"]))
def _sample_reference_pair(*, rng) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    shape_pool = [str(shape) for shape in REFERENCE_SHAPE_TYPES]
    rng.shuffle(shape_pool)
    for first_shape in shape_pool:
        first = _make_sampled_object(
            rng=rng,
            object_id=f"reference_{first_shape}_a",
            shape_type=first_shape,
            object_role="context",
            xy=(0.0, 0.0),
        )
        if not bool(first.get("nameable_for_prompt", False)):
            continue
        for second_shape in shape_pool:
            second = _make_sampled_object(
                rng=rng,
                object_id=f"reference_{second_shape}_b",
                shape_type=second_shape,
                object_role="context",
                xy=(0.0, 0.0),
            )
            if not bool(second.get("nameable_for_prompt", False)):
                continue
            if _prompt_name(first) == _prompt_name(second):
                continue
            return dict(first), dict(second)
    raise ValueError("could not sample two distinct 3D reference props")


def _between_metrics(
    candidate: Mapping[str, Any],
    reference_a: Mapping[str, Any],
    reference_b: Mapping[str, Any],
) -> Dict[str, float]:
    cx, cy, _cz = (float(value) for value in candidate["world_xyz"])
    ax, ay, _az = (float(value) for value in reference_a["world_xyz"])
    bx, by, _bz = (float(value) for value in reference_b["world_xyz"])
    vx = float(bx - ax)
    vy = float(by - ay)
    segment_length = max(1e-6, math.hypot(vx, vy))
    ux = float(vx / segment_length)
    uy = float(vy / segment_length)
    acx = float(cx - ax)
    acy = float(cy - ay)
    along = float(acx * ux + acy * uy)
    t = float(along / segment_length)
    lateral = abs(float(acx * (-uy) + acy * ux))
    end_margin = min(float(along), float(segment_length - along))
    return {
        "t": round(float(t), 6),
        "lateral_distance": round(float(lateral), 6),
        "end_margin": round(float(end_margin), 6),
        "segment_length": round(float(segment_length), 6),
    }


def _is_between_candidate(metrics: Mapping[str, float], *, lateral_threshold: float) -> bool:
    return bool(
        0.30 <= float(metrics["t"]) <= 0.70
        and float(metrics["lateral_distance"]) <= float(lateral_threshold)
        and float(metrics["end_margin"]) >= 0.42
    )


def _build_between_references_scene_dataset(
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
    """Build a between-references scene with one uniquely valid target; spatial relation metadata stays the verifier source of truth."""
    if int(context_object_count) != 2:
        raise ValueError("between-references scenes use exactly two named reference objects")
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    label_selection_index = int(answer_label_index) if answer_label_index is not None else int(instance_seed)
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )
    for _attempt in range(520):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        labels = [str(label) for label in POINT_LABELS[: int(point_count)]]
        answer_label = labels[abs(int(label_selection_index)) % int(point_count)]
        remaining_labels = [str(label) for label in labels if str(label) != str(answer_label)]
        rng.shuffle(remaining_labels)

        reference_a, reference_b = _sample_reference_pair(rng=rng)
        segment_angle = float(rng.choice([0.0, 0.55, 1.05, 1.57, 2.12, 2.62]) + rng.uniform(-0.16, 0.16))
        ux = math.cos(segment_angle)
        uy = math.sin(segment_angle)
        half_distance = float(rng.uniform(1.86, 2.18))
        center_x = float(rng.uniform(-0.12, 0.12))
        center_y = float(rng.uniform(-0.12, 0.12))
        ref_a_xy = (float(center_x - ux * half_distance), float(center_y - uy * half_distance))
        ref_b_xy = (float(center_x + ux * half_distance), float(center_y + uy * half_distance))
        if max(abs(ref_a_xy[0]), abs(ref_a_xy[1]), abs(ref_b_xy[0]), abs(ref_b_xy[1])) > float(render_params.room_extent) - 0.42:
            continue
        reference_a = _set_xy(reference_a, ref_a_xy)
        reference_b = _set_xy(reference_b, ref_b_xy)
        if not _can_place(reference_b, [reference_a], clearance=0.46):
            continue

        shape_pool = [str(shape) for shape in SMALL_CANDIDATE_SHAPE_TYPES]
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
        side_x, side_y = -uy, ux
        answer_t = float(rng.uniform(0.42, 0.58))
        answer_lateral = float(rng.uniform(-0.17, 0.17))
        answer_xy = (
            float(ref_a_xy[0] + (ref_b_xy[0] - ref_a_xy[0]) * answer_t + side_x * answer_lateral),
            float(ref_a_xy[1] + (ref_b_xy[1] - ref_a_xy[1]) * answer_t + side_y * answer_lateral),
        )
        answer_spec = _set_xy(answer_spec, answer_xy)
        if not _can_place(answer_spec, [reference_a, reference_b], clearance=0.10):
            continue

        placed: List[Dict[str, Any]] = [dict(reference_a), dict(reference_b), dict(answer_spec)]
        distractor_shapes = [str(shape) for shape in shape_pool[1:]] + [str(shape) for shape in shape_pool[:1]]
        ring_slots = [
            (-2.54, -2.18),
            (-1.18, -2.58),
            (1.20, -2.52),
            (2.54, -1.40),
            (2.42, 1.42),
            (1.02, 2.54),
            (-1.34, 2.46),
            (-2.54, 1.18),
        ]
        rng.shuffle(ring_slots)
        distractor_specs: List[Dict[str, Any]] = []
        distractor_shape_order = shuffled_repeated_support(rng, distractor_shapes, len(remaining_labels))
        for index, (label, shape) in enumerate(zip(remaining_labels, distractor_shape_order)):
            placed_spec: Dict[str, Any] | None = None
            for slot_x, slot_y in ring_slots[index:] + ring_slots[:index]:
                for _jitter_attempt in range(8):
                    candidate = _make_sampled_object(
                        rng=rng,
                        object_id=f"object_{label}",
                        shape_type=shape,
                        object_role="candidate",
                        xy=(float(slot_x + rng.uniform(-0.15, 0.15)), float(slot_y + rng.uniform(-0.15, 0.15))),
                        label=str(label),
                    )
                    metrics = _between_metrics(candidate, reference_a, reference_b)
                    lateral_threshold = max(0.36, float(candidate["footprint_radius"]) * 0.72)
                    if _is_between_candidate(metrics, lateral_threshold=float(lateral_threshold)):
                        continue
                    if not _can_place(candidate, placed, clearance=0.12):
                        continue
                    placed_spec = candidate
                    break
                if placed_spec is not None:
                    break
            if placed_spec is None:
                break
            distractor_specs.append(placed_spec)
            placed.append(placed_spec)
        if len(distractor_specs) != int(point_count) - 1:
            continue

        candidate_specs = [answer_spec, *distractor_specs]
        all_specs = [*candidate_specs, reference_a, reference_b]
        reference_points = [point for spec in all_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_candidates = _finalize_specs(candidate_specs, camera=camera, frame=frame)
        reference_a, reference_b = _finalize_specs([reference_a, reference_b], camera=camera, frame=frame)
        all_finalized = [*finalized_candidates, reference_a, reference_b]

        metrics_by_label: Dict[str, Dict[str, float]] = {}
        status_by_label: Dict[str, bool] = {}
        lateral_threshold_by_label: Dict[str, float] = {}
        for spec in finalized_candidates:
            label = str(spec["point_label"])
            lateral_threshold = max(0.36, float(spec["footprint_radius"]) * 0.72)
            metrics = _between_metrics(spec, reference_a, reference_b)
            metrics_by_label[label] = dict(metrics)
            lateral_threshold_by_label[label] = round(float(lateral_threshold), 4)
            status_by_label[label] = _is_between_candidate(metrics, lateral_threshold=float(lateral_threshold))

        between_labels = [str(label) for label, is_between in sorted(status_by_label.items()) if bool(is_between)]
        if between_labels != [str(answer_label)]:
            continue
        answer_metrics = metrics_by_label[str(answer_label)]
        if float(answer_metrics["lateral_distance"]) > 0.24 or not (0.38 <= float(answer_metrics["t"]) <= 0.62):
            continue
        gap_a = _surface_gap(answer_spec, reference_a)
        gap_b = _surface_gap(answer_spec, reference_b)
        if min(float(gap_a), float(gap_b)) < 0.06:
            continue

        candidate_centers = [
            (float(spec["screen_xy"][0]), float(spec["screen_xy"][1]))
            for spec in finalized_candidates
        ]
        if any(
            math.hypot(a[0] - b[0], a[1] - b[1]) < 34.0
            for index, a in enumerate(candidate_centers)
            for b in candidate_centers[index + 1 :]
        ):
            continue

        candidate_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=16.0) for spec in finalized_candidates]
        compact_candidate_bboxes = [
            (str(spec["point_label"]), _object_screen_bbox(spec, camera, frame, pad_px=4.0))
            for spec in finalized_candidates
        ]
        compact_reference_bboxes = [
            (str(spec["object_id"]), _object_screen_bbox(spec, camera, frame, pad_px=4.0))
            for spec in (reference_a, reference_b)
        ]
        all_bboxes = [
            (str(spec["object_id"]), _object_screen_bbox(spec, camera, frame, pad_px=16.0))
            for spec in all_finalized
        ]
        if any(
            _bbox_intersection_area(a, b) > 13000.0
            for index, a in enumerate(candidate_bboxes)
            for b in candidate_bboxes[index + 1 :]
        ):
            continue
        if any(
            _bbox_intersection_area(candidate_bbox, reference_bbox) > 3600.0
            for _label, candidate_bbox in compact_candidate_bboxes
            for _reference_id, reference_bbox in compact_reference_bboxes
        ):
            continue
        if any(
            _bbox_intersection_area(a, b) > 24000.0
            for index, (_a_id, a) in enumerate(all_bboxes)
            for _b_id, b in all_bboxes[index + 1 :]
        ):
            continue

        sorted_candidates = sorted(finalized_candidates, key=lambda spec: str(spec["point_label"]))
        reference_specs = sorted([reference_a, reference_b], key=lambda spec: str(spec["object_id"]))
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "candidate_count": int(point_count),
            "context_object_count": 2,
            "object_count": int(point_count) + 2,
            "point_specs": list(sorted_candidates),
            "context_object_specs": list(reference_specs),
            "object_specs": sorted([*sorted_candidates, *reference_specs], key=lambda spec: str(spec["object_id"])),
            "answer_label": str(answer_label),
            "answer_point_id": f"object_{answer_label}",
            "reference_object_ids": [str(reference_a["object_id"]), str(reference_b["object_id"])],
            "reference_object_names": [_prompt_name(reference_a), _prompt_name(reference_b)],
            "reference_shape_types": [str(reference_a["shape_type"]), str(reference_b["shape_type"])],
            "candidate_between_status_by_label": dict(sorted(status_by_label.items())),
            "candidate_between_metrics_by_label": dict(sorted(metrics_by_label.items())),
            "candidate_between_lateral_threshold_by_label": dict(sorted(lateral_threshold_by_label.items())),
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
                "sort_key": "world_floor_segment_between_two_references",
                "candidate_only": True,
                "references_excluded_from_options": True,
                "reference_object_ids": [str(reference_a["object_id"]), str(reference_b["object_id"])],
                "reference_object_names": [_prompt_name(reference_a), _prompt_name(reference_b)],
                "reference_shape_types": [str(reference_a["shape_type"]), str(reference_b["shape_type"])],
                "candidate_between_status_by_label": dict(sorted(status_by_label.items())),
                "candidate_between_metrics_by_label": dict(sorted(metrics_by_label.items())),
                "candidate_between_lateral_threshold_by_label": dict(sorted(lateral_threshold_by_label.items())),
                "between_reference_labels": list(between_labels),
                "unique_between_answer": True,
                "answer_between_t": round(float(answer_metrics["t"]), 4),
                "answer_lateral_distance": round(float(answer_metrics["lateral_distance"]), 4),
                "answer_gap_to_reference_a": round(float(gap_a), 4),
                "answer_gap_to_reference_b": round(float(gap_b), 4),
            },
        }
    raise ValueError("could not construct a valid 3D between-references scene")






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
class ThreeDSpatialBetweenReferencesLabelTask:
    """Choose the lettered 3D object between two named reference objects."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
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
        """Generate one between-references instance, keeping prompt, answer, annotation, and trace derived from the accepted dataset."""
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
            minimum_default=2,
            maximum_default=2,
            lower=2,
            upper=2,
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
        dataset = _build_between_references_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            context_object_count=int(context_object_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
            answer_label_index=int(answer_label_index),
            camera_yaw_band=camera_yaw_band,
        )
        reference_names = list(dataset["reference_object_names"])
        relation_fields = {
            "reference_object_ids": list(dataset["reference_object_ids"]),
            "reference_object_names": list(dataset["reference_object_names"]),
            "reference_shape_types": list(dataset["reference_shape_types"]),
            "candidate_between_status_by_label": dict(dataset["candidate_between_status_by_label"]),
            "candidate_between_metrics_by_label": dict(dataset["candidate_between_metrics_by_label"]),
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
                "reference_a_name": str(reference_names[0]),
                "reference_b_name": str(reference_names[1]),
            },
            relation_fields=relation_fields,
            query_params_extra={
                "reference_object_ids": list(dataset["reference_object_ids"]),
                "reference_object_names": list(dataset["reference_object_names"]),
            },
            execution_extra={
                "reference_object_ids": list(dataset["reference_object_ids"]),
                "reference_object_names": list(dataset["reference_object_names"]),
                "reference_shape_types": list(dataset["reference_shape_types"]),
                "candidate_between_status_by_label": dict(dataset["candidate_between_status_by_label"]),
                "candidate_between_metrics_by_label": dict(dataset["candidate_between_metrics_by_label"]),
                "candidate_between_lateral_threshold_by_label": dict(dataset["candidate_between_lateral_threshold_by_label"]),
            },
        )



__all__ = ["ThreeDSpatialBetweenReferencesLabelTask"]
