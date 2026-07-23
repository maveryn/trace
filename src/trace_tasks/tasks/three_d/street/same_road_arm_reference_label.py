"""Same-road-arm-as-reference task for a synthetic 3D street scene."""

from __future__ import annotations

from collections import Counter
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
from ..shared.object_resources import STREET_OBJECT_TYPES
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.task_support import resolve_support_choice_for_namespace
from ..shared.object_scene import (
    POINT_LABELS,
    _build_projection_frame,
    _object_reference_points,
    _sample_camera,
)
from ._lifecycle import build_street_option_label_task_output
from .shared.state import (
    SCENE_ID,
    STREET_CAMERA_YAW_BANDS_DEGREES,
    SUPPORTED_INTERSECTION_LAYOUTS,
    SUPPORTED_SCENE_VARIANTS,
    _StreetRenderParams,
    _arm_is_present,
    _canvas_floor_polygon_available,
    _candidate_context_visibility_ok,
    _candidate_screen_separation_ok,
    _dimensions_for_orientation,
    _finalize_specs,
    _make_street_object_spec,
    _min_pairwise,
    _missing_arm_for_layout,
    _object_screen_bbox as _street_object_screen_bbox,
    _orientation_axis_for_xy,
    _reference_visibility_ok,
    _resolve_render_params,
    _sample_context_specs,
    _sample_intersection_center,
    _translate_scene_xy,
)


TASK_ID = "task_three_d__street__same_road_arm_reference_label"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "same_road_arm_as_reference"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
ROAD_ARMS: Tuple[str, ...] = ("north", "south", "east", "west")
ROAD_ARM_SLOTS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "north": ((-0.42, 1.78), (0.36, 2.30), (-0.20, 2.88), (0.48, 3.34)),
    "south": ((0.42, -1.78), (-0.36, -2.30), (0.20, -2.88), (-0.48, -3.34)),
    "east": ((1.78, 0.42), (2.30, -0.36), (2.88, 0.20), (3.34, -0.48)),
    "west": ((-1.78, -0.42), (-2.30, 0.36), (-2.88, -0.20), (-3.34, 0.48)),
}
MIN_REFERENCE_CENTER_SEPARATION_PX = 46.0
MAX_REFERENCE_CANDIDATE_BBOX_INTERSECTION_PX = 5200.0






def _resolve_choice(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    key: str,
    support: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    choices = tuple(str(item) for item in support)
    explicit = params.get(str(key))
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(choices):
            raise ValueError(f"unsupported {key}: {selected}")
        return selected, {selected: 1.0}
    selected, probabilities = resolve_support_choice_for_namespace(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.{key}",
        support_values=choices,
        explicit_key=str(key),
    )
    return str(selected), {str(choice): float(probabilities[str(choice)]) for choice in choices}


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


def _present_road_arms(intersection_layout: str) -> Tuple[str, ...]:
    return tuple(
        str(arm)
        for arm in ROAD_ARMS
        if _arm_is_present(str(intersection_layout), str(arm))
    )


def _slot_arm(slot_xy: Sequence[float]) -> str:
    x, y = float(slot_xy[0]), float(slot_xy[1])
    if abs(y) >= abs(x):
        return "north" if y >= 0.0 else "south"
    return "east" if x >= 0.0 else "west"


def _make_arm_object_spec(
    *,
    rng,
    object_id: str,
    object_type: str,
    object_role: str,
    road_arm: str,
    relative_xy: Sequence[float],
    intersection_center_xy: Tuple[float, float],
    street_extent: float,
    label: str | None,
    jitter: float,
) -> Dict[str, Any]:
    """Place one arm-bound street object while preserving road-arm metadata."""

    jittered_xy = (
        float(relative_xy[0]) + float(rng.uniform(-float(jitter), float(jitter))),
        float(relative_xy[1]) + float(rng.uniform(-float(jitter), float(jitter))),
    )
    xy = _translate_scene_xy(
        jittered_xy,
        center_xy=intersection_center_xy,
        extent=float(street_extent),
        margin=0.44,
    )
    orientation_axis = _orientation_axis_for_xy(relative_xy)
    scale = float(rng.uniform(0.94, 1.14))
    dimensions = _dimensions_for_orientation(
        str(object_type),
        orientation_axis=str(orientation_axis),
        scale=float(scale),
    )
    spec = _make_street_object_spec(
        object_id=str(object_id),
        object_type=str(object_type),
        object_role=str(object_role),
        xy=tuple(float(value) for value in xy),
        intersection_center_xy=tuple(float(value) for value in intersection_center_xy),
        orientation_axis=str(orientation_axis),
        dimensions_xyz=dimensions,
        label=label,
        dimension_scale=float(scale),
    )
    spec["road_arm"] = str(road_arm)
    spec["relative_road_xy"] = [round(float(jittered_xy[0]), 4), round(float(jittered_xy[1]), 4)]
    return spec


def _sample_reference_and_candidate_specs(
    *,
    rng,
    candidate_count: int,
    intersection_center_xy: Tuple[float, float],
    intersection_layout: str,
    reference_road_arm: str,
    reference_object_type: str,
    street_extent: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Sample one unique reference arm answer plus off-arm distractor candidates."""

    present_arms = _present_road_arms(str(intersection_layout))
    if str(reference_road_arm) not in set(present_arms):
        raise ValueError("reference road arm is not present in this layout")
    same_arm_slots = list(ROAD_ARM_SLOTS[str(reference_road_arm)])
    rng.shuffle(same_arm_slots)
    reference_slot = same_arm_slots[0]
    answer_slot = same_arm_slots[1]
    distractor_slots: List[Tuple[str, Tuple[float, float]]] = []
    other_arms = [str(arm) for arm in present_arms if str(arm) != str(reference_road_arm)]
    rng.shuffle(other_arms)
    for arm in other_arms:
        slots = list(ROAD_ARM_SLOTS[str(arm)])
        rng.shuffle(slots)
        for slot in slots:
            distractor_slots.append((str(arm), tuple(float(value) for value in slot)))
    if int(candidate_count) - 1 > len(distractor_slots):
        raise ValueError("not enough same-road-arm distractor slots")

    candidate_types = [
        str(item)
        for item in STREET_OBJECT_TYPES
        if str(item) != str(reference_object_type)
    ]
    if int(candidate_count) > len(candidate_types):
        raise ValueError("not enough unique street candidate types")
    rng.shuffle(candidate_types)

    reference_spec = _make_arm_object_spec(
        rng=rng,
        object_id=f"reference_street_object_{reference_object_type}",
        object_type=str(reference_object_type),
        object_role="street_reference",
        road_arm=str(reference_road_arm),
        relative_xy=tuple(float(value) for value in reference_slot),
        intersection_center_xy=tuple(float(value) for value in intersection_center_xy),
        street_extent=float(street_extent),
        label=None,
        jitter=0.05,
    )
    reference_spec["is_named_reference"] = True

    raw_candidate_slots: List[Tuple[str, Tuple[float, float], bool]] = [
        (str(reference_road_arm), tuple(float(value) for value in answer_slot), True)
    ]
    raw_candidate_slots.extend(
        (str(arm), tuple(float(value) for value in slot), False)
        for arm, slot in distractor_slots[: int(candidate_count) - 1]
    )
    rng.shuffle(raw_candidate_slots)
    candidate_specs: List[Dict[str, Any]] = []
    for index, (road_arm, slot, is_answer_slot) in enumerate(raw_candidate_slots):
        object_type = str(candidate_types[index])
        spec = _make_arm_object_spec(
            rng=rng,
            object_id=f"candidate_{index}_{object_type}",
            object_type=str(object_type),
            object_role="street_candidate",
            road_arm=str(road_arm),
            relative_xy=tuple(float(value) for value in slot),
            intersection_center_xy=tuple(float(value) for value in intersection_center_xy),
            street_extent=float(street_extent),
            label="?",
            jitter=0.045 if bool(is_answer_slot) else 0.085,
        )
        spec["same_road_arm_as_reference"] = bool(is_answer_slot)
        candidate_specs.append(spec)
    return dict(reference_spec), list(candidate_specs)


def _build_street_same_road_arm_dataset(
    *,
    params: Mapping[str, Any],
    query_id: str,
    scene_variant: str,
    intersection_layout: str,
    candidate_count: int,
    context_object_count: int,
    reference_road_arm: str,
    reference_object_type: str,
    camera_yaw_band: Tuple[float, float],
    camera_yaw_band_index: int,
    render_params: _StreetRenderParams,
    instance_seed: int,
) -> Dict[str, Any]:
    """Build one finalized same-road-arm dataset with a unique option label."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    for _attempt in range(440):
        intersection_center_xy = _sample_intersection_center(
            rng,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            render_params=render_params,
        )
        present_arms = _present_road_arms(str(intersection_layout))
        actual_reference_arm = str(reference_road_arm)
        if actual_reference_arm not in set(present_arms):
            actual_reference_arm = str(rng.choice(present_arms))
        camera = _sample_camera(rng, yaw_band_degrees=tuple(float(value) for value in camera_yaw_band))
        reference_spec, candidate_specs = _sample_reference_and_candidate_specs(
            rng=rng,
            candidate_count=int(candidate_count),
            intersection_center_xy=tuple(float(value) for value in intersection_center_xy),
            intersection_layout=str(intersection_layout),
            reference_road_arm=str(actual_reference_arm),
            reference_object_type=str(reference_object_type),
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
        all_specs = [reference_spec, *candidate_specs, *context_specs]
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
        finalized_reference = _finalize_specs([reference_spec], camera=camera, frame=frame)[0]
        finalized_candidates = _finalize_specs(candidate_specs, camera=camera, frame=frame)
        finalized_context = _finalize_specs(context_specs, camera=camera, frame=frame)
        if not _candidate_screen_separation_ok(
            finalized_candidates,
            camera=camera,
            frame=frame,
            render_params=render_params,
        ):
            continue
        if not _reference_visibility_ok(
            finalized_reference,
            finalized_candidates,
            finalized_context,
            camera=camera,
            frame=frame,
            render_params=render_params,
            min_center_separation_px=MIN_REFERENCE_CENTER_SEPARATION_PX,
            max_reference_candidate_bbox_intersection_px=MAX_REFERENCE_CANDIDATE_BBOX_INTERSECTION_PX,
        ):
            continue
        if not _candidate_context_visibility_ok(
            finalized_candidates,
            [finalized_reference, *finalized_context],
            camera=camera,
            frame=frame,
        ):
            continue

        satisfying = [
            spec
            for spec in finalized_candidates
            if str(spec.get("road_arm")) == str(finalized_reference.get("road_arm"))
        ]
        if len(satisfying) != 1:
            continue

        answer_object_id = str(satisfying[0]["object_id"])
        answer_label_index = int(spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label").randrange(int(candidate_count)))
        answer_label = str(POINT_LABELS[int(answer_label_index)])
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
                    "same_road_arm_as_reference": str(updated.get("road_arm")) == str(finalized_reference.get("road_arm")),
                }
            )
            relabeled_candidates.append(updated)

        answer_spec = next(spec for spec in relabeled_candidates if str(spec["point_label"]) == str(answer_label))
        all_finalized = [finalized_reference, *relabeled_candidates, *finalized_context]
        reference_prompt_name = str(finalized_reference["prompt_name"])
        reference_prompt_name_count = sum(
            1
            for spec in all_finalized
            if str(spec.get("prompt_name")) == reference_prompt_name
        )
        if int(reference_prompt_name_count) != 1:
            continue

        candidate_road_arms = {
            str(spec["point_label"]): str(spec["road_arm"])
            for spec in relabeled_candidates
        }
        same_arm_flags = {
            str(spec["point_label"]): str(spec["road_arm"]) == str(finalized_reference["road_arm"])
            for spec in relabeled_candidates
        }
        candidate_ground_xy = {
            str(spec["point_label"]): [round(float(spec["base_xyz"][0]), 4), round(float(spec["base_xyz"][1]), 4)]
            for spec in relabeled_candidates
        }
        candidate_projected_bboxes = {
            str(spec["point_label"]): [
                round(float(value), 3)
                for value in _street_object_screen_bbox(spec, camera, frame, pad_px=0.0)
            ]
            for spec in relabeled_candidates
        }
        candidate_object_types = {
            str(spec["point_label"]): str(spec["object_type"]) for spec in relabeled_candidates
        }
        reference_bbox = [
            round(float(value), 3)
            for value in _street_object_screen_bbox(finalized_reference, camera, frame, pad_px=0.0)
        ]
        object_type_counts = Counter(str(spec["object_type"]) for spec in all_finalized)
        road_arm_counts = Counter(str(spec.get("road_arm")) for spec in relabeled_candidates)

        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "intersection_layout": str(intersection_layout),
            "missing_road_arm": _missing_arm_for_layout(str(intersection_layout)),
            "present_road_arms": list(present_arms),
            "candidate_count": int(candidate_count),
            "context_object_count": int(context_object_count),
            "intersection_center_xy": [round(float(intersection_center_xy[0]), 4), round(float(intersection_center_xy[1]), 4)],
            "reference_object": {
                "object_id": str(finalized_reference["object_id"]),
                "object_type": str(finalized_reference["object_type"]),
                "prompt_name": reference_prompt_name,
                "road_arm": str(finalized_reference["road_arm"]),
                "world_xyz": list(finalized_reference["world_xyz"]),
                "base_xyz": list(finalized_reference["base_xyz"]),
                "screen_xy": list(finalized_reference["screen_xy"]),
                "bbox_px": list(reference_bbox),
                "prompt_name_count": int(reference_prompt_name_count),
            },
            "candidate_object_specs": sorted(relabeled_candidates, key=lambda spec: str(spec["point_label"])),
            "reference_object_specs": [dict(finalized_reference)],
            "context_object_specs": sorted(finalized_context, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted(all_finalized, key=lambda spec: str(spec["object_id"])),
            "target_object_ids": [str(answer_spec["object_id"])],
            "answer_label": str(answer_label),
            "answer_object_id": str(answer_spec["object_id"]),
            "answer_object_type": str(answer_spec["object_type"]),
            "answer_road_arm": str(answer_spec["road_arm"]),
            "same_road_arm_as_reference_by_label": dict(sorted(same_arm_flags.items())),
            "same_road_arm_candidate_labels": sorted(
                [str(label) for label, flag in same_arm_flags.items() if bool(flag)]
            ),
            "candidate_road_arm_by_label": dict(sorted(candidate_road_arms.items())),
            "candidate_ground_xy_by_label": dict(sorted(candidate_ground_xy.items())),
            "candidate_object_types_by_label": dict(sorted(candidate_object_types.items())),
            "candidate_projected_bboxes_by_label": dict(sorted(candidate_projected_bboxes.items())),
            "road_arm_candidate_counts": dict(sorted(road_arm_counts.items())),
            "object_type_counts": dict(sorted(object_type_counts.items())),
            "object_count": int(len(all_finalized)),
            "min_pairwise_candidate_ground_gap": round(
                float(
                    _min_pairwise(
                        [
                            float(spec["ground_distance_to_intersection"])
                            for spec in relabeled_candidates
                        ]
                    )
                ),
                4,
            ),
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
                "predicate": "option-panel candidate whose road_arm equals the red-boxed reference object road_arm",
                "reference_object": {
                    "object_id": str(finalized_reference["object_id"]),
                    "object_type": str(finalized_reference["object_type"]),
                    "prompt_name": reference_prompt_name,
                    "road_arm": str(finalized_reference["road_arm"]),
                    "prompt_name_count": int(reference_prompt_name_count),
                },
                "present_road_arms": list(present_arms),
                "intersection_layout": str(intersection_layout),
                "missing_road_arm": _missing_arm_for_layout(str(intersection_layout)),
                "candidate_road_arm_by_label": dict(sorted(candidate_road_arms.items())),
                "same_road_arm_as_reference_by_label": dict(sorted(same_arm_flags.items())),
                "answer_label": str(answer_label),
                "answer_object_id": str(answer_spec["object_id"]),
                "answer_road_arm": str(answer_spec["road_arm"]),
                "unique_answer": True,
            },
        }
    raise ValueError(
        "could not construct a visible street same-road-arm-reference scene"
    )




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
class ThreeDStreetSameRoadArmReferenceLabelTask:
    """Choose the option-panel street object on the same road arm as a reference."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
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
        """Generate one same-road-arm sample with scalar bbox annotation."""

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
        present_arms = _present_road_arms(str(intersection_layout))
        reference_road_arm, reference_road_arm_probabilities = _resolve_choice(
            params,
            instance_seed=int(instance_seed),
            key="reference_road_arm",
            support=present_arms,
        )
        reference_object_type, reference_object_type_probabilities = _resolve_choice(
            params,
            instance_seed=int(instance_seed),
            key="reference_object_type",
            support=STREET_OBJECT_TYPES,
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
        dataset = _build_street_same_road_arm_dataset(
            params=params,
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            intersection_layout=str(intersection_layout),
            candidate_count=int(candidate_count),
            context_object_count=int(context_object_count),
            reference_road_arm=str(reference_road_arm),
            reference_object_type=str(reference_object_type),
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
                "present_road_arms": list(dataset["present_road_arms"]),
                "reference_object": dict(dataset["reference_object"]),
                "same_road_arm_as_reference_by_label": dict(dataset["same_road_arm_as_reference_by_label"]),
                "candidate_road_arm_by_label": dict(dataset["candidate_road_arm_by_label"]),
                "candidate_ground_xy_by_label": dict(dataset["candidate_ground_xy_by_label"]),
                "candidate_object_types_by_label": dict(dataset["candidate_object_types_by_label"]),
            },
            query_param_fields={
                "reference_road_arm": str(dataset["reference_object"]["road_arm"]),
                "reference_road_arm_probabilities": dict(reference_road_arm_probabilities),
                "reference_object_type": str(dataset["reference_object"]["object_type"]),
                "reference_object_type_probabilities": dict(reference_object_type_probabilities),
                "reference_object_name": str(dataset["reference_object"]["prompt_name"]),
            },
            render_spec_fields={
                "present_road_arms": list(dataset["present_road_arms"]),
            },
            execution_trace_fields={
                "present_road_arms": list(dataset["present_road_arms"]),
                "answer_road_arm": str(dataset["answer_road_arm"]),
                "reference_object": dict(dataset["reference_object"]),
                "reference_object_specs": [
                    dict(spec) for spec in dataset["reference_object_specs"]
                ],
                "same_road_arm_as_reference_by_label": dict(dataset["same_road_arm_as_reference_by_label"]),
                "same_road_arm_candidate_labels": list(dataset["same_road_arm_candidate_labels"]),
                "candidate_road_arm_by_label": dict(dataset["candidate_road_arm_by_label"]),
                "candidate_ground_xy_by_label": dict(dataset["candidate_ground_xy_by_label"]),
                "candidate_object_types_by_label": dict(dataset["candidate_object_types_by_label"]),
                "candidate_projected_bboxes_by_label": dict(dataset["candidate_projected_bboxes_by_label"]),
                "road_arm_candidate_counts": dict(dataset["road_arm_candidate_counts"]),
                "object_type_counts": dict(dataset["object_type_counts"]),
                "min_pairwise_candidate_ground_gap": float(dataset["min_pairwise_candidate_ground_gap"]),
            },
        )


__all__ = ["ThreeDStreetSameRoadArmReferenceLabelTask"]
