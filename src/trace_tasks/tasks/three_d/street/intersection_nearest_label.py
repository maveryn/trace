"""Nearest-to-intersection task for a synthetic 3D street scene."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
)
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    split_generation_rendering_prompt_defaults,
)
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.task_support import resolve_support_choice_for_namespace
from ..shared.object_resources import (
    STREET_OBJECT_TYPES,
)
from ..shared.camera_projection import (
    build_projection_frame as _build_projection_frame,
    sample_camera as _sample_camera,
)
from ..shared.object_scene import POINT_LABELS
from ._lifecycle import build_street_option_label_task_output
from .shared.state import (
    SCENE_ID,
    STREET_CAMERA_YAW_BANDS_DEGREES,
    SUPPORTED_INTERSECTION_LAYOUTS,
    SUPPORTED_SCENE_VARIANTS,
    _StreetRenderParams,
    _candidate_context_visibility_ok,
    _candidate_screen_separation_ok,
    _canvas_floor_polygon_available,
    _dimensions_for_orientation,
    _finalize_specs,
    _make_street_object_spec,
    _min_pairwise,
    _missing_arm_for_layout,
    _object_reference_points,
    _object_screen_bbox,
    _orientation_axis_for_xy,
    _resolve_render_params,
    _sample_context_specs,
    _sample_intersection_center,
    _slot_allowed_for_layout,
    _translate_scene_xy,
)


TASK_ID = "task_three_d__street__intersection_nearest_label"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "closest_to_intersection"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
ANSWER_SLOTS: Tuple[Tuple[float, float], ...] = (
    (-0.84, -0.22),
    (0.84, 0.22),
    (-0.22, 0.84),
    (0.22, -0.84),
)
DISTRACTOR_SLOTS: Tuple[Tuple[float, float], ...] = (
    (-2.12, 0.32),
    (2.14, -0.32),
    (0.35, 2.24),
    (-0.35, -2.24),
    (-2.28, -1.54),
    (2.30, 1.54),
    (-1.48, 2.70),
    (1.48, -2.70),
    (-3.04, 0.42),
    (3.02, -0.42),
)
MIN_NEAREST_DISTANCE_MARGIN = 0.52




def _resolve_camera_yaw_band(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> Tuple[Tuple[float, float], Dict[str, float], int]:
    explicit = params.get("camera_yaw_band_index")
    support = tuple(range(len(STREET_CAMERA_YAW_BANDS_DEGREES)))
    if explicit is not None:
        selected_index = int(explicit)
        if selected_index not in set(support):
            raise ValueError(f"unsupported camera_yaw_band_index: {selected_index}")
        probabilities = {str(value): (1.0 if int(value) == int(selected_index) else 0.0) for value in support}
    else:
        selected_index, probabilities = resolve_support_choice_for_namespace(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.camera_yaw_band_index",
            support_values=support,
        )
    return (
        tuple(float(value) for value in STREET_CAMERA_YAW_BANDS_DEGREES[int(selected_index)]),
        {str(key): float(value) for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))},
        int(selected_index),
    )




def _sample_candidate_specs(
    *,
    rng,
    candidate_count: int,
    intersection_center_xy: Tuple[float, float],
    intersection_layout: str,
    road_half_width: float,
    street_extent: float,
) -> List[Dict[str, Any]]:
    """Place answer and distractor candidates with a unique nearest-object gap."""

    object_types = list(STREET_OBJECT_TYPES)
    rng.shuffle(object_types)
    selected_types = object_types[: int(candidate_count)]
    answer_slots = [
        slot for slot in ANSWER_SLOTS
        if _slot_allowed_for_layout(slot, intersection_layout=str(intersection_layout), road_half_width=float(road_half_width))
    ] or list(ANSWER_SLOTS)
    distractor_slots = [
        slot for slot in DISTRACTOR_SLOTS
        if _slot_allowed_for_layout(slot, intersection_layout=str(intersection_layout), road_half_width=float(road_half_width))
    ] or list(DISTRACTOR_SLOTS)
    rng.shuffle(answer_slots)
    rng.shuffle(distractor_slots)
    if int(candidate_count) - 1 > len(distractor_slots):
        raise ValueError("not enough street-object distractor slots")
    raw_slots: List[Tuple[Tuple[float, float], bool]] = [(answer_slots[0], True)]
    raw_slots.extend((slot, False) for slot in distractor_slots[: int(candidate_count) - 1])
    rng.shuffle(raw_slots)
    specs: List[Dict[str, Any]] = []
    for index, (slot_xy, is_answer_slot) in enumerate(raw_slots):
        object_type = str(selected_types[index])
        jitter = 0.055 if bool(is_answer_slot) else 0.115
        xy = (
            float(slot_xy[0] + rng.uniform(-jitter, jitter)),
            float(slot_xy[1] + rng.uniform(-jitter, jitter)),
        )
        xy = _translate_scene_xy(
            xy,
            center_xy=intersection_center_xy,
            extent=float(street_extent),
            margin=0.44,
        )
        orientation_axis = _orientation_axis_for_xy(xy)
        scale = float(rng.uniform(0.92, 1.12))
        dimensions = _dimensions_for_orientation(
            object_type,
            orientation_axis=orientation_axis,
            scale=float(scale),
        )
        specs.append(
            _make_street_object_spec(
                object_id=f"candidate_{index}_{object_type}",
                object_type=str(object_type),
                object_role="street_candidate",
                xy=xy,
                intersection_center_xy=tuple(float(value) for value in intersection_center_xy),
                orientation_axis=str(orientation_axis),
                dimensions_xyz=dimensions,
                label="?",
                dimension_scale=float(scale),
            )
        )
    return list(specs)




def _build_street_dataset(
    *,
    params: Mapping[str, Any],
    query_id: str,
    scene_variant: str,
    intersection_layout: str,
    candidate_count: int,
    context_object_count: int,
    camera_yaw_band: Tuple[float, float],
    camera_yaw_band_index: int,
    render_params: _StreetRenderParams,
    instance_seed: int,
) -> Dict[str, Any]:
    """Build one finalized street sample and bind the nearest-intersection answer."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    for _attempt in range(360):
        intersection_center_xy = _sample_intersection_center(
            rng,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            render_params=render_params,
        )
        camera = _sample_camera(rng, yaw_band_degrees=tuple(float(value) for value in camera_yaw_band))
        candidate_specs = _sample_candidate_specs(
            rng=rng,
            candidate_count=int(candidate_count),
            intersection_center_xy=tuple(intersection_center_xy),
            intersection_layout=str(intersection_layout),
            road_half_width=float(render_params.road_half_width),
            street_extent=float(render_params.street_extent),
        )
        context_specs = _sample_context_specs(
            rng=rng,
            scene_variant=str(scene_variant),
            context_object_count=int(context_object_count),
            intersection_center_xy=tuple(intersection_center_xy),
            intersection_layout=str(intersection_layout),
            road_half_width=float(render_params.road_half_width),
            street_extent=float(render_params.street_extent),
        )
        all_specs = [*candidate_specs, *context_specs]
        reference_points: List[Tuple[float, float, float]] = [
            (-render_params.street_extent, -render_params.street_extent, 0.0),
            (render_params.street_extent, -render_params.street_extent, 0.0),
            (render_params.street_extent, render_params.street_extent, 0.0),
            (-render_params.street_extent, render_params.street_extent, 0.0),
            (-render_params.street_extent, -render_params.street_extent, 1.7),
            (render_params.street_extent, render_params.street_extent, 1.7),
        ]
        for spec in all_specs:
            reference_points.extend(_object_reference_points(spec))
        frame = _build_projection_frame(
            camera=camera,
            render_params=render_params,
            point_worlds=reference_points,
        )
        if not _canvas_floor_polygon_available(camera=camera, frame=frame, render_params=render_params):
            continue
        finalized_candidates = _finalize_specs(candidate_specs, camera=camera, frame=frame)
        finalized_context = _finalize_specs(context_specs, camera=camera, frame=frame)
        if not _candidate_screen_separation_ok(
            finalized_candidates,
            camera=camera,
            frame=frame,
            render_params=render_params,
        ):
            continue
        if not _candidate_context_visibility_ok(
            finalized_candidates,
            finalized_context,
            camera=camera,
            frame=frame,
        ):
            continue
        sorted_by_distance = sorted(
            finalized_candidates,
            key=lambda spec: (float(spec["ground_distance_to_intersection"]), str(spec["object_id"])),
        )
        distance_values = [float(spec["ground_distance_to_intersection"]) for spec in finalized_candidates]
        if len(sorted_by_distance) < 2:
            continue
        nearest_margin = float(sorted_by_distance[1]["ground_distance_to_intersection"]) - float(sorted_by_distance[0]["ground_distance_to_intersection"])
        if nearest_margin < MIN_NEAREST_DISTANCE_MARGIN:
            continue

        answer_object_id = str(sorted_by_distance[0]["object_id"])
        label_support = tuple(POINT_LABELS[: int(candidate_count)])
        answer_label = str(spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label").choice(label_support))
        remaining_labels = [
            str(label)
            for label in POINT_LABELS[: int(candidate_count)]
            if str(label) != str(answer_label)
        ]
        rng.shuffle(remaining_labels)
        relabeled_candidates: List[Dict[str, Any]] = []
        for spec in finalized_candidates:
            updated = dict(spec)
            label = str(answer_label) if str(updated["object_id"]) == answer_object_id else str(remaining_labels.pop())
            updated.update(
                {
                    "object_id": f"street_object_{label}",
                    "point_id": f"street_object_{label}",
                    "point_label": str(label),
                    "object_label": str(label),
                    "is_answer_candidate": True,
                }
            )
            relabeled_candidates.append(updated)
        sorted_by_distance = sorted(
            relabeled_candidates,
            key=lambda spec: (float(spec["ground_distance_to_intersection"]), str(spec["point_label"])),
        )
        answer_spec = next(spec for spec in relabeled_candidates if str(spec["point_label"]) == str(answer_label))
        all_finalized = [*relabeled_candidates, *finalized_context]
        candidate_ground_distances = {
            str(spec["point_label"]): round(float(spec["ground_distance_to_intersection"]), 4)
            for spec in relabeled_candidates
        }
        candidate_ground_xy = {
            str(spec["point_label"]): [round(float(spec["base_xyz"][0]), 4), round(float(spec["base_xyz"][1]), 4)]
            for spec in relabeled_candidates
        }
        candidate_object_types = {
            str(spec["point_label"]): str(spec["object_type"]) for spec in relabeled_candidates
        }
        candidate_projected_bboxes = {
            str(spec["point_label"]): [
                round(float(value), 3)
                for value in _object_screen_bbox(spec, camera, frame, pad_px=0.0)
            ]
            for spec in relabeled_candidates
        }
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "intersection_layout": str(intersection_layout),
            "missing_road_arm": _missing_arm_for_layout(str(intersection_layout)),
            "candidate_count": int(candidate_count),
            "context_object_count": int(context_object_count),
            "intersection_center_xy": [round(float(intersection_center_xy[0]), 4), round(float(intersection_center_xy[1]), 4)],
            "candidate_object_specs": sorted(relabeled_candidates, key=lambda spec: str(spec["point_label"])),
            "context_object_specs": sorted(finalized_context, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted(all_finalized, key=lambda spec: str(spec["object_id"])),
            "target_object_ids": [str(answer_spec["object_id"])],
            "answer_label": str(answer_label),
            "answer_object_id": str(answer_spec["object_id"]),
            "answer_object_type": str(answer_spec["object_type"]),
            "candidate_ground_xy_by_label": dict(sorted(candidate_ground_xy.items())),
            "ground_distance_to_intersection_by_label": dict(sorted(candidate_ground_distances.items())),
            "candidate_object_types_by_label": dict(sorted(candidate_object_types.items())),
            "candidate_projected_bboxes_by_label": dict(sorted(candidate_projected_bboxes.items())),
            "distance_order_near_to_far": [str(spec["point_label"]) for spec in sorted_by_distance],
            "nearest_distance_margin": round(float(nearest_margin), 4),
            "min_pairwise_ground_distance_gap": round(float(_min_pairwise(distance_values)), 4),
            "object_count": int(len(all_finalized)),
            "camera": {
                "camera_position": [round(float(value), 4) for value in camera.camera_position],
                "target": [round(float(value), 4) for value in camera.target],
                "yaw_degrees": round(float(camera.yaw_degrees), 4),
                "yaw_band_index": int(camera_yaw_band_index),
                "yaw_band_degrees": [round(float(value), 4) for value in camera_yaw_band],
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
                "sort_key": "ground_distance_to_intersection",
                "candidate_only": True,
                "intersection_center_xy": [round(float(intersection_center_xy[0]), 4), round(float(intersection_center_xy[1]), 4)],
                "intersection_layout": str(intersection_layout),
                "missing_road_arm": _missing_arm_for_layout(str(intersection_layout)),
                "ground_distance_to_intersection_by_label": dict(sorted(candidate_ground_distances.items())),
                "distance_order_near_to_far": [str(spec["point_label"]) for spec in sorted_by_distance],
                "candidate_object_types_by_label": dict(sorted(candidate_object_types.items())),
                "context_building_styles_by_id": {
                    str(spec["object_id"]): str(spec.get("building_style", ""))
                    for spec in sorted(finalized_context, key=lambda item: str(item["object_id"]))
                    if str(spec.get("object_type")) == "building"
                },
                "nearest_distance_margin": round(float(nearest_margin), 4),
                "answer_label": str(answer_label),
                "answer_object_id": str(answer_spec["object_id"]),
                "unique_answer": True,
            },
        }
    raise ValueError("could not construct a visible street-intersection nearest-object scene")






_TASK_GROUP_DEFAULTS = get_scene_defaults("three_d", "street")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)
_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


@register_task
class ThreeDStreetIntersectionNearestLabelTask:
    """Choose the option-panel street object nearest to an intersection center."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt_index == 0
                else int(
                    spawn_rng(
                        int(instance_seed),
                        f"{TASK_ID}.attempt_seed.{attempt_index}",
                    ).randrange(1, 2**62)
                )
            )
            try:
                return self._generate_once(int(attempt_seed), params=params)
            except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
                last_error = exc
        raise RuntimeError(
            f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}"
        )

    def _generate_once(self, instance_seed: int, *, params: Dict[str, Any]) -> TaskOutput:
        """Render one finalized sample and bind scalar bbox annotation from the trace."""

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
        intersection_layout, intersection_layout_probabilities = _shared_resolve_axis_variant(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            supported_variants=SUPPORTED_INTERSECTION_LAYOUTS,
            explicit_key="intersection_layout",
            weights_key="intersection_layout_weights",
            balance_flag_key="balanced_intersection_layout_sampling",
            axis_namespace="intersection_layout",
        )
        candidate_count, candidate_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="candidate_count",
            default_min=4,
            default_max=4,
            lower=4,
            upper=4,
        )
        context_object_count, context_object_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            key="context_object_count",
            default_min=10,
            default_max=10,
            lower=6,
            upper=12,
        )
        camera_yaw_band, camera_yaw_probabilities, camera_yaw_band_index = _resolve_camera_yaw_band(
            params,
            instance_seed=int(instance_seed),
        )
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_street_dataset(
            params=params,
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            intersection_layout=str(intersection_layout),
            candidate_count=int(candidate_count),
            context_object_count=int(context_object_count),
            camera_yaw_band=tuple(camera_yaw_band),
            camera_yaw_band_index=int(camera_yaw_band_index),
            render_params=render_params,
            instance_seed=int(instance_seed),
        )
        return build_street_option_label_task_output(
            task_id=TASK_ID,
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_query_key=PROMPT_QUERY_KEY,
            background_defaults=_BACKGROUND_DEFAULTS,
            noise_defaults=_NOISE_DEFAULTS,
            params=params,
            instance_seed=int(instance_seed),
            query_id=str(query_id),
            query_probabilities=query_probabilities,
            scene_variant=str(scene_variant),
            scene_probabilities=scene_probabilities,
            intersection_layout=str(intersection_layout),
            intersection_layout_probabilities=intersection_layout_probabilities,
            candidate_count=int(candidate_count),
            candidate_count_probabilities=candidate_count_probabilities,
            context_object_count=int(context_object_count),
            context_object_count_probabilities=context_object_count_probabilities,
            camera_yaw_band_index=int(camera_yaw_band_index),
            camera_yaw_probabilities=camera_yaw_probabilities,
            render_params=render_params,
            dataset=dataset,
            scene_relation_fields={
                "ground_distance_to_intersection_by_label": dict(dataset["ground_distance_to_intersection_by_label"]),
                "candidate_ground_xy_by_label": dict(dataset["candidate_ground_xy_by_label"]),
                "candidate_object_types_by_label": dict(dataset["candidate_object_types_by_label"]),
                "distance_order_near_to_far": list(dataset["distance_order_near_to_far"]),
            },
            execution_trace_fields={
                "candidate_ground_xy_by_label": dict(dataset["candidate_ground_xy_by_label"]),
                "ground_distance_to_intersection_by_label": dict(dataset["ground_distance_to_intersection_by_label"]),
                "candidate_object_types_by_label": dict(dataset["candidate_object_types_by_label"]),
                "candidate_projected_bboxes_by_label": dict(dataset["candidate_projected_bboxes_by_label"]),
                "distance_order_near_to_far": list(dataset["distance_order_near_to_far"]),
                "nearest_distance_margin": float(dataset["nearest_distance_margin"]),
                "min_pairwise_ground_distance_gap": float(dataset["min_pairwise_ground_distance_gap"]),
            },
        )


__all__ = ["ThreeDStreetIntersectionNearestLabelTask"]
