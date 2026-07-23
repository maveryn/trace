"""Vertical-height extremum task for a synthetic 3D object scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
    resolve_scene_section_defaults,
)
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ..shared.canvas import render_params_canvas_metadata
from ..shared.task_support import normalize_unit as _normalize_unit
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.task_support import resolve_count as _shared_resolve_count
from ..shared.object_resources import (
    SPATIAL_HEIGHT_SAFE_CANDIDATE_SHAPE_TYPES,
    SPATIAL_HEIGHT_SUPPORT_PLACEMENTS,
)
from ..shared.option_panel import build_text_option_choices
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
    _project_screen,
    _resolve_render_params,
    _sample_camera,
    _sample_shape_dimensions,
    render_object_scene_3d,
)


TASK_ID = "task_three_d__object_scene__height_extremum_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("highest_above_floor", "lowest_above_floor")
SUPPORT_PLACEMENTS: Tuple[Tuple[str, str | None, Tuple[float, float]], ...] = SPATIAL_HEIGHT_SUPPORT_PLACEMENTS
FLOOR_PLACEMENTS: Tuple[Tuple[str, str | None, Tuple[float, float]], ...] = tuple(
    placement for placement in SUPPORT_PLACEMENTS if placement[1] is None
)
ELEVATED_SUPPORT_PLACEMENTS: Tuple[Tuple[str, str | None, Tuple[float, float]], ...] = tuple(
    placement for placement in SUPPORT_PLACEMENTS if placement[1] is not None
)
HEIGHT_OPTION_COUNT = 4
HEIGHT_SUPPORT_COUNT = 5
HEIGHT_SLOT_COUNT = HEIGHT_SUPPORT_COUNT + 1
HEIGHT_PLACED_OBJECT_COUNT = HEIGHT_OPTION_COUNT
HEIGHT_EMPTY_SLOT_COUNT = HEIGHT_SLOT_COUNT - HEIGHT_PLACED_OBJECT_COUNT
HEIGHT_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = SPATIAL_HEIGHT_SAFE_CANDIDATE_SHAPE_TYPES
PLATFORM_DIMENSIONS_BY_PLACEMENT: Dict[str, Tuple[float, float, float]] = {
    "lowest_platform": (1.26, 0.86, 0.34),
    "low_platform": (1.30, 0.90, 0.56),
    "mid_platform": (1.34, 0.92, 0.84),
    "upper_platform": (1.34, 0.92, 1.16),
    "high_platform": (1.34, 0.92, 1.52),
}






def _set_object_base_z(spec: Mapping[str, Any], base_z: float) -> Dict[str, Any]:
    updated = dict(spec)
    height = float(updated["dimensions_xyz"][2])
    updated["base_xyz"] = [
        round(float(updated["base_xyz"][0]), 4),
        round(float(updated["base_xyz"][1]), 4),
        round(float(base_z), 4),
    ]
    updated["world_xyz"] = [
        round(float(updated["world_xyz"][0]), 4),
        round(float(updated["world_xyz"][1]), 4),
        round(float(base_z) + height * 0.5, 4),
    ]
    return updated


def _make_sampled_object(
    *,
    rng,
    object_id: str,
    shape_type: str,
    object_role: str,
    xy: Tuple[float, float],
    label: str | None = None,
    base_z: float = 0.0,
    dimensions_xyz: Tuple[float, float, float] | None = None,
) -> Dict[str, Any]:
    if dimensions_xyz is None:
        sampled_dimensions_xyz, dimension_scale = _sample_shape_dimensions(str(shape_type), object_role=str(object_role), rng=rng)
    else:
        sampled_dimensions_xyz = tuple(float(value) for value in dimensions_xyz)
        dimension_scale = 1.0
    spec = _make_object_spec(
        object_id=str(object_id),
        shape_type=str(shape_type),
        object_role=str(object_role),
        xy=xy,
        dimensions_xyz=sampled_dimensions_xyz,
        dimension_scale=float(dimension_scale),
        label=label,
    )
    if float(base_z) != 0.0:
        spec = _set_object_base_z(spec, float(base_z))
    return spec


def _support_base_height(support_spec: Mapping[str, Any] | None) -> float:
    if support_spec is None:
        return 0.0
    support_height = float(support_spec["dimensions_xyz"][2])
    return round(float(support_height), 4)


def _support_visibility_offset(support_spec: Mapping[str, Any] | None) -> float:
    if support_spec is None:
        return 0.0
    shape_type = str(support_spec["shape_type"])
    if shape_type == "platform":
        return 0.0
    return 0.0


def _finalize_specs(
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
    return list(finalized_specs)


def _xy_towards_camera(xy: Sequence[float], camera, *, distance: float) -> Tuple[float, float]:
    dx = float(camera.camera_position[0]) - float(xy[0])
    dy = float(camera.camera_position[1]) - float(xy[1])
    length = max(1e-6, math.hypot(dx, dy))
    return (
        float(xy[0]) + float(dx / length) * float(distance),
        float(xy[1]) + float(dy / length) * float(distance),
    )


def _height_value(spec: Mapping[str, Any]) -> float:
    return float(spec["base_xyz"][2])


def _build_height_scene_dataset(
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
    """Build a height-extremum scene where rendered object heights yield exactly one highest or lowest labeled candidate."""
    if int(point_count) != HEIGHT_OPTION_COUNT:
        raise ValueError(f"{TASK_ID} expects {HEIGHT_OPTION_COUNT} option-panel candidates")
    if len(FLOOR_PLACEMENTS) != 1:
        raise ValueError(f"{TASK_ID} expects exactly one floor placement")
    if int(context_object_count) != HEIGHT_SUPPORT_COUNT:
        raise ValueError(f"{TASK_ID} expects {HEIGHT_SUPPORT_COUNT} support props")
    if len(ELEVATED_SUPPORT_PLACEMENTS) < HEIGHT_SUPPORT_COUNT:
        raise ValueError(f"{TASK_ID} needs at least {HEIGHT_SUPPORT_COUNT} platform placements")

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    label_selection_index = int(answer_label_index) if answer_label_index is not None else int(instance_seed)
    selected_camera_yaw_band = (
        tuple(float(value) for value in camera_yaw_band)
        if camera_yaw_band is not None
        else _camera_yaw_band_for_instance(int(instance_seed))
    )
    for _attempt in range(360):
        camera = _sample_camera(rng, yaw_band_degrees=selected_camera_yaw_band)
        support_placements = list(ELEVATED_SUPPORT_PLACEMENTS)
        rng.shuffle(support_placements)
        selected_placements = [FLOOR_PLACEMENTS[0], *support_placements[: int(context_object_count)]]
        rng.shuffle(selected_placements)

        support_specs_by_name: Dict[str, Dict[str, Any]] = {}
        context_specs: List[Dict[str, Any]] = []
        for placement_name, support_shape, xy in selected_placements:
            if support_shape is None:
                continue
            support_spec = _make_sampled_object(
                rng=rng,
                object_id=f"support_{placement_name}_{support_shape}",
                shape_type=str(support_shape),
                object_role="context",
                xy=tuple(float(value) for value in xy),
                dimensions_xyz=PLATFORM_DIMENSIONS_BY_PLACEMENT.get(str(placement_name)),
            )
            if str(support_shape) == "platform":
                support_spec["object_name"] = "platform"
                support_spec["prompt_name"] = "platform"
            support_specs_by_name[str(placement_name)] = support_spec
            context_specs.append(support_spec)

        placement_records: List[Dict[str, Any]] = []
        for placement_name, _support_shape, xy in selected_placements:
            support_spec = support_specs_by_name.get(str(placement_name))
            base_z = _support_base_height(support_spec)
            placement_records.append(
                {
                    "placement_id": str(placement_name),
                    "xy": tuple(float(value) for value in xy),
                    "candidate_xy": _xy_towards_camera(
                        xy,
                        camera,
                        distance=_support_visibility_offset(support_spec),
                    ),
                    "base_z": round(float(base_z), 4),
                    "support_object_id": str(support_spec["object_id"]) if support_spec is not None else None,
                    "support_shape_type": str(support_spec["shape_type"]) if support_spec is not None else None,
                    "support_name": str(support_spec["prompt_name"]) if support_spec is not None else None,
                }
            )
        heights = [float(record["base_z"]) for record in placement_records]
        if len(placement_records) != HEIGHT_SLOT_COUNT:
            raise ValueError(f"{TASK_ID} expects {HEIGHT_SLOT_COUNT} possible height slots")

        option_placements = list(placement_records)
        rng.shuffle(option_placements)
        option_placement_ids = {str(record["placement_id"]) for record in option_placements[: int(point_count)]}
        empty_slot_placement_ids = sorted(
            str(record["placement_id"])
            for record in placement_records
            if str(record["placement_id"]) not in option_placement_ids
        )
        sorted_option_placements = sorted(
            [record for record in placement_records if str(record["placement_id"]) in option_placement_ids],
            key=lambda item: (float(item["base_z"]), str(item["placement_id"])),
        )
        answer_placement = sorted_option_placements[-1] if str(query_id) == "highest_above_floor" else sorted_option_placements[0]
        answer_height_margin = (
            float(sorted_option_placements[-1]["base_z"]) - float(sorted_option_placements[-2]["base_z"])
            if str(query_id) == "highest_above_floor"
            else float(sorted_option_placements[1]["base_z"]) - float(sorted_option_placements[0]["base_z"])
        )
        if float(answer_height_margin) < 0.20:
            continue

        labels = [str(label) for label in POINT_LABELS[: int(point_count)]]
        answer_label = labels[abs(int(label_selection_index)) % int(point_count)]
        remaining_labels = [str(label) for label in labels if str(label) != str(answer_label)]
        rng.shuffle(remaining_labels)

        shape_pool = list(HEIGHT_CANDIDATE_SHAPE_TYPES)
        rng.shuffle(shape_pool)
        if len(shape_pool) < int(point_count):
            raise ValueError(f"{TASK_ID} needs at least {point_count} height-safe candidate shapes")
        candidate_specs: List[Dict[str, Any]] = []
        shape_index = 0
        for placement in placement_records:
            if str(placement["placement_id"]) not in option_placement_ids:
                continue
            label = (
                str(answer_label)
                if str(placement["placement_id"]) == str(answer_placement["placement_id"])
                else str(remaining_labels.pop())
            )
            shape_type = str(shape_pool[int(shape_index)])
            shape_index += 1
            jitter = 0.02 if placement.get("support_object_id") else 0.10
            base_xy = tuple(float(value) for value in placement["candidate_xy"])
            spec = _make_sampled_object(
                rng=rng,
                object_id=f"object_{label}",
                shape_type=shape_type,
                object_role="candidate",
                xy=(float(base_xy[0] + rng.uniform(-jitter, jitter)), float(base_xy[1] + rng.uniform(-jitter, jitter))),
                label=str(label),
                base_z=float(placement["base_z"]),
            )
            spec.update(
                {
                    "height_placement_id": str(placement["placement_id"]),
                    "vertical_base_height": round(float(placement["base_z"]), 4),
                    "support_object_id": placement.get("support_object_id"),
                    "support_shape_type": placement.get("support_shape_type"),
                    "support_name": placement.get("support_name"),
                    "height_option_role": "option_candidate",
                }
            )
            if placement.get("support_object_id"):
                spec["render_order_bias"] = -10.0
            candidate_specs.append(spec)

        all_specs = [*candidate_specs, *context_specs]
        reference_points = [point for spec in all_specs for point in _object_reference_points(spec)]
        frame = _build_projection_frame(camera=camera, render_params=render_params, point_worlds=reference_points)
        finalized_candidates = _finalize_specs(candidate_specs, camera=camera, frame=frame)
        finalized_support_context = _finalize_specs(context_specs, camera=camera, frame=frame)
        finalized_context = list(finalized_support_context)
        all_finalized = [*finalized_candidates, *finalized_context]
        small_object_screen_centers = [
            (float(spec["screen_xy"][0]), float(spec["screen_xy"][1]))
            for spec in finalized_candidates
        ]
        if any(
            math.hypot(a[0] - b[0], a[1] - b[1]) < 34.0
            for index, a in enumerate(small_object_screen_centers)
            for b in small_object_screen_centers[index + 1 :]
        ):
            continue

        small_object_bboxes = [_object_screen_bbox(spec, camera, frame, pad_px=16.0) for spec in finalized_candidates]
        if any(
            _bbox_intersection_area(a, b) > 13000.0
            for index, a in enumerate(small_object_bboxes)
            for b in small_object_bboxes[index + 1 :]
        ):
            continue

        support_pair_ids = {
            frozenset({str(spec["object_id"]), str(spec["support_object_id"])})
            for spec in finalized_candidates
            if spec.get("support_object_id")
        }
        all_bboxes = [
            (str(spec["object_id"]), _object_screen_bbox(spec, camera, frame, pad_px=16.0))
            for spec in all_finalized
        ]
        if any(
            _bbox_intersection_area(a, b) > 30000.0
            for index, (a_id, a) in enumerate(all_bboxes)
            for b_id, b in all_bboxes[index + 1 :]
            if frozenset({str(a_id), str(b_id)}) not in support_pair_ids
        ):
            continue

        height_by_label = {
            str(spec["point_label"]): round(float(_height_value(spec)), 4)
            for spec in finalized_candidates
        }
        sorted_by_height = sorted(finalized_candidates, key=lambda spec: (float(_height_value(spec)), str(spec["point_label"])))
        expected_label = str(sorted_by_height[-1]["point_label"] if str(query_id) == "highest_above_floor" else sorted_by_height[0]["point_label"])
        if expected_label != str(answer_label):
            continue

        sorted_candidates = sorted(finalized_candidates, key=lambda spec: str(spec["point_label"]))
        sorted_context = sorted(finalized_context, key=lambda spec: str(spec["object_id"]))
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "candidate_count": int(point_count),
            "support_object_count": int(context_object_count),
            "empty_slot_count": len(empty_slot_placement_ids),
            "non_option_object_count": 0,
            "context_object_count": len(sorted_context),
            "visible_placed_object_count": HEIGHT_PLACED_OBJECT_COUNT,
            "height_slot_count": HEIGHT_SLOT_COUNT,
            "object_count": len(sorted_candidates) + len(sorted_context),
            "point_specs": list(sorted_candidates),
            "context_object_specs": list(sorted_context),
            "non_option_object_specs": [],
            "object_specs": sorted([*sorted_candidates, *sorted_context], key=lambda spec: str(spec["object_id"])),
            "answer_label": str(answer_label),
            "answer_point_id": f"object_{answer_label}",
            "height_by_label": dict(sorted(height_by_label.items())),
            "height_order_low_to_high": [str(spec["point_label"]) for spec in sorted_by_height],
            "option_placement_ids": sorted(str(item) for item in option_placement_ids),
            "empty_slot_placement_ids": list(empty_slot_placement_ids),
            "non_option_object_ids": [],
            "floor_object_is_option": "floor_spot" in option_placement_ids,
            "support_pair_ids": [sorted(list(pair)) for pair in sorted(support_pair_ids, key=lambda pair: tuple(sorted(pair)))],
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
                "sort_key": "vertical_base_height",
                "candidate_only": True,
                "height_by_label": dict(sorted(height_by_label.items())),
                "height_order_low_to_high": [str(spec["point_label"]) for spec in sorted_by_height],
                "highest_label": str(sorted_by_height[-1]["point_label"]),
                "lowest_label": str(sorted_by_height[0]["point_label"]),
                "unique_height_extremum_answer": True,
                "height_margin": round(float(answer_height_margin), 4),
                "min_pairwise_height_gap": round(float(_min_pairwise(list(height_by_label.values()))), 4),
                "option_placement_ids": sorted(str(item) for item in option_placement_ids),
                "empty_slot_placement_ids": list(empty_slot_placement_ids),
                "floor_object_is_option": "floor_spot" in option_placement_ids,
                "support_pair_ids": [sorted(list(pair)) for pair in sorted(support_pair_ids, key=lambda pair: tuple(sorted(pair)))],
            },
        }
    raise ValueError("could not construct a valid 3D height-extremum scene")






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
class ThreeDSpatialHeightExtremumLabelTask:
    """Choose the lettered 3D object highest or lowest above the floor."""

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
        """Generate one height-extremum instance, preserving the accepted object-height ordering in answer and annotation."""
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
            minimum_default=HEIGHT_OPTION_COUNT,
            maximum_default=HEIGHT_OPTION_COUNT,
            lower=HEIGHT_OPTION_COUNT,
            upper=HEIGHT_OPTION_COUNT,
        )
        context_object_count, context_object_count_probabilities = _shared_resolve_count(
            params,
            task_id=TASK_ID,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            prefix="context_object_count",
            minimum_default=HEIGHT_SUPPORT_COUNT,
            maximum_default=HEIGHT_SUPPORT_COUNT,
            lower=HEIGHT_SUPPORT_COUNT,
            upper=HEIGHT_SUPPORT_COUNT,
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
        dataset = _build_height_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            context_object_count=int(context_object_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
            answer_label_index=int(answer_label_index),
            camera_yaw_band=camera_yaw_band,
        )
        background, background_meta = make_background_canvas(
            canvas_width=int(render_params.canvas_width),
            canvas_height=int(render_params.canvas_height),
            instance_seed=int(instance_seed),
            params=params,
            default_config=_BACKGROUND_DEFAULTS,
        )
        option_choices = build_text_option_choices(dataset["point_specs"])
        rendered_scene = render_object_scene_3d(
            background,
            dataset=dataset,
            render_params=render_params,
            draw_candidate_labels=False,
            option_choices=option_choices,
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered_scene.image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=_NOISE_DEFAULTS,
        )

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            query_key=str(query_id),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        answer_label = str(dataset["answer_label"])
        answer_gt = TypedValue(type="option_letter", value=str(answer_label))
        annotation_bboxes = [[round(float(value), 3) for value in bbox] for bbox in rendered_scene.annotation_bboxes]
        if len(annotation_bboxes) != 1:
            raise RuntimeError(f"{TASK_ID} expected exactly one annotation bbox")
        annotation_payload = bbox_annotation_artifacts(annotation_bboxes[0])
        annotation_gt = annotation_payload.annotation_gt
        solver_trace = dict(dataset["solver_trace"])

        trace_payload = {
            "scene_ir": {
                "scene_kind": "three_d_object_scene",
                "entities": [dict(entity) for entity in rendered_scene.entities],
                "relations": {
                    "scene_variant": str(scene_variant),
                    "point_count": int(point_count),
                    "candidate_count": int(point_count),
                    "support_object_count": int(dataset["support_object_count"]),
                    "height_slot_count": int(dataset["height_slot_count"]),
                    "empty_slot_count": int(dataset["empty_slot_count"]),
                    "non_option_object_count": int(dataset["non_option_object_count"]),
                    "context_object_count": int(dataset["context_object_count"]),
                    "visible_placed_object_count": int(dataset["visible_placed_object_count"]),
                    "object_count": int(dataset["object_count"]),
                    "candidate_shape_types": [str(spec["shape_type"]) for spec in dataset["point_specs"]],
                    "context_shape_types": [str(spec["shape_type"]) for spec in dataset["context_object_specs"]],
                    "non_option_shape_types": [str(spec["shape_type"]) for spec in dataset["non_option_object_specs"]],
                    "candidate_object_names": [str(spec["object_name"]) for spec in dataset["point_specs"]],
                    "context_object_names": [str(spec["object_name"]) for spec in dataset["context_object_specs"]],
                    "non_option_object_names": [str(spec["object_name"]) for spec in dataset["non_option_object_specs"]],
                    "view_family": "synthetic_perspective_3d_scene",
                    "height_by_label": dict(dataset["height_by_label"]),
                    "height_order_low_to_high": list(dataset["height_order_low_to_high"]),
                    "option_placement_ids": list(dataset["option_placement_ids"]),
                    "empty_slot_placement_ids": list(dataset["empty_slot_placement_ids"]),
                    "non_option_object_ids": list(dataset["non_option_object_ids"]),
                    "floor_object_is_option": bool(dataset["floor_object_is_option"]),
                    "answer_point_id": str(dataset["answer_point_id"]),
                    "answer_label": str(answer_label),
                },
            },
            "query_spec": {
                "query_id": str(query_id),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(query_id),
                    "query_id_probabilities": dict(query_probabilities),
                    "scene_variant": str(scene_variant),
                    "scene_variant_probabilities": dict(scene_probabilities),
                    "point_count": int(point_count),
                    "candidate_count": int(point_count),
                    "support_object_count": int(dataset["support_object_count"]),
                    "height_slot_count": int(dataset["height_slot_count"]),
                    "empty_slot_count": int(dataset["empty_slot_count"]),
                    "context_object_count": int(context_object_count),
                    "rendered_context_object_count": int(dataset["context_object_count"]),
                    "non_option_object_count": int(dataset["non_option_object_count"]),
                    "visible_placed_object_count": int(dataset["visible_placed_object_count"]),
                    "context_object_count_probabilities": dict(context_object_count_probabilities),
                    "point_count_probabilities": dict(point_count_probabilities),
                    "object_count": int(dataset["object_count"]),
                },
            },
            "render_spec": {
                "canvas_width": int(render_params.canvas_width),
                "canvas_height": int(image.height),
                "scene_canvas_preset": str(render_params.canvas_preset),
                "scene_canvas_width": int(render_params.canvas_width),
                "scene_canvas_height": int(render_params.canvas_height),
                "scene_canvas_policy": str(render_params.canvas_policy),
                **render_params_canvas_metadata(render_params),
                "final_canvas_width": int(image.width),
                "final_canvas_height": int(image.height),
                "final_canvas_pixels": int(image.width) * int(image.height),
                "option_panel_height_px": int(rendered_scene.option_panel_height_px),
                "coord_space": "pixel",
                "scene_variant": str(scene_variant),
                "background_style": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
                "camera": dict(dataset["camera"]),
                "projection_frame": dict(dataset["projection_frame"]),
                "label_font_size_px": int(render_params.label_font_size_px),
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": list(rendered_scene.scene_bbox_px),
                "room_bbox_px": list(rendered_scene.room_bbox_px),
                "point_bboxes_px": {str(key): list(value) for key, value in rendered_scene.point_bboxes_px.items()},
                "point_centers_px": {str(key): list(value) for key, value in rendered_scene.point_centers_px.items()},
                "option_panel_bbox_px": list(rendered_scene.option_panel_bbox_px),
                "option_panel_height_px": int(rendered_scene.option_panel_height_px),
                "option_choice_bboxes_px": {
                    str(key): list(value) for key, value in rendered_scene.option_choice_bboxes_px.items()
                },
                "option_choices": [dict(choice) for choice in rendered_scene.option_choices],
                "object_bboxes_px": {str(key): list(value) for key, value in rendered_scene.object_bboxes_px.items()},
                "object_centers_px": {str(key): list(value) for key, value in rendered_scene.object_centers_px.items()},
                "context_object_bboxes_px": {str(key): list(value) for key, value in rendered_scene.context_object_bboxes_px.items()},
                "context_object_centers_px": {str(key): list(value) for key, value in rendered_scene.context_object_centers_px.items()},
            },
            "execution_trace": {
                "query_id": str(query_id),
                "scene_variant": str(scene_variant),
                "point_count": int(point_count),
                "candidate_count": int(point_count),
                "support_object_count": int(dataset["support_object_count"]),
                "height_slot_count": int(dataset["height_slot_count"]),
                "empty_slot_count": int(dataset["empty_slot_count"]),
                "context_object_count": int(dataset["context_object_count"]),
                "non_option_object_count": int(dataset["non_option_object_count"]),
                "visible_placed_object_count": int(dataset["visible_placed_object_count"]),
                "object_count": int(dataset["object_count"]),
                "point_specs": [dict(spec) for spec in dataset["point_specs"]],
                "context_object_specs": [dict(spec) for spec in dataset["context_object_specs"]],
                "non_option_object_specs": [dict(spec) for spec in dataset["non_option_object_specs"]],
                "object_specs": [dict(spec) for spec in dataset["object_specs"]],
                "option_choices": [dict(choice) for choice in rendered_scene.option_choices],
                "option_descriptor_by_label": {
                    str(choice["label"]): str(choice["descriptor"])
                    for choice in rendered_scene.option_choices
                },
                "answer_label": str(answer_label),
                "answer_point_id": str(dataset["answer_point_id"]),
                "height_by_label": dict(dataset["height_by_label"]),
                "height_order_low_to_high": list(dataset["height_order_low_to_high"]),
                "option_placement_ids": list(dataset["option_placement_ids"]),
                "empty_slot_placement_ids": list(dataset["empty_slot_placement_ids"]),
                "non_option_object_ids": list(dataset["non_option_object_ids"]),
                "floor_object_is_option": bool(dataset["floor_object_is_option"]),
                "support_pair_ids": [list(pair) for pair in dataset["support_pair_ids"]],
                "camera": dict(dataset["camera"]),
                "projection_frame": dict(dataset["projection_frame"]),
                "question_format": str(query_id),
                "view_family": "synthetic_perspective_3d_scene",
                "solver_trace": dict(solver_trace),
            },
            "witness_symbolic": {
                "type": "object",
                "ids": [str(item) for item in rendered_scene.annotation_entity_ids],
            },
            "projected_annotation": dict(annotation_payload.projected_annotation),
            "background": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
        }

        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )


__all__ = ["ThreeDSpatialHeightExtremumLabelTask"]
