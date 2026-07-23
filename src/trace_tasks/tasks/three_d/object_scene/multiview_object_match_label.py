"""Multi-view object correspondence task for a synthetic 3D object scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import (
    get_domain_defaults,
    get_scene_defaults,
)
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.canvas import render_params_canvas_metadata
from ..shared.task_support import resolve_axis_variant as _shared_resolve_axis_variant
from ..shared.object_scene import (
    CAMERA_YAW_BANDS_DEGREES,
    POINT_LABELS,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    _RenderParams,
    _bbox_intersection_area,
    _build_projection_frame,
    _make_object_spec,
    _min_pairwise,
    _object_reference_points,
    _object_screen_bbox,
    _project_screen,
    _resolve_render_params,
    _sample_camera,
)
from .shared.layout import (
    CANDIDATE_VIEW_KEY,
    REFERENCE_VIEW_KEY,
    multiview_scaled_panel_layout as _multiview_scaled_panel_layout,
    multiview_source_render_params as _multiview_source_render_params,
    offset_bbox as _offset_bbox,
    offset_entities as _offset_entities,
    render_multiview_scene as _render_multiview_scene,
    shift_render_maps as _shift_render_maps,
)


TASK_ID = "task_three_d__object_scene__multiview_object_match_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("same_object_in_second_view",)
MULTIVIEW_CANDIDATE_COUNT = 4
MULTIVIEW_MIN_CANDIDATE_BBOX_AREA_PX = 980.0
MULTIVIEW_CANDIDATE_SHAPE_TYPES: Tuple[str, ...] = (
    "cube",
    "cylinder",
    "cone",
    "pyramid",
    "half_cylinder",
)
MULTIVIEW_CANDIDATE_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (218, 76, 68),
    (63, 123, 214),
    (52, 159, 96),
    (151, 88, 204),
    (222, 143, 45),
    (43, 164, 184),
)
MULTIVIEW_ANCHOR_PLATFORM_DIMS = (1.76, 1.10, 0.18)
MULTIVIEW_ANCHOR_MARKER_DIMS = (0.38, 0.30, 0.30)
MULTIVIEW_CANDIDATE_POSITIONS: Tuple[Tuple[float, float], ...] = (
    (-1.56, -1.06),
    (1.88, -1.38),
    (-2.04, 1.16),
    (1.32, 1.82),
)


def _bbox_area(bbox: Sequence[float]) -> float:
    return max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))


def _bbox_is_readable(bbox: Sequence[float], *, width: int, height: int, min_side_px: float = 22.0) -> bool:
    box_width = float(bbox[2]) - float(bbox[0])
    box_height = float(bbox[3]) - float(bbox[1])
    if box_width < float(min_side_px) or box_height < float(min_side_px):
        return False
    return float(bbox[2]) > 6.0 and float(bbox[3]) > 6.0 and float(bbox[0]) < float(width - 6) and float(bbox[1]) < float(height - 6)


def _camera_yaw_bands_for_instance(instance_seed: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.camera_yaw_pair")
    band_count = len(CAMERA_YAW_BANDS_DEGREES)
    first_index = int(rng.randrange(band_count))
    second_index = int(first_index) + max(1, band_count // 2)
    if second_index >= band_count:
        second_index -= band_count
    return (
        tuple(float(value) for value in CAMERA_YAW_BANDS_DEGREES[first_index]),
        tuple(float(value) for value in CAMERA_YAW_BANDS_DEGREES[second_index]),
    )


def _yaw_separation_degrees(yaw_a: float, yaw_b: float) -> float:
    diff = abs(float(yaw_a) - float(yaw_b)) % 360.0
    return min(float(diff), 360.0 - float(diff))


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


def _with_base_z(spec: Mapping[str, Any], *, base_z: float) -> Dict[str, Any]:
    updated = dict(spec)
    width, depth, height = (float(value) for value in updated["dimensions_xyz"])
    base_x, base_y, _old_z = (float(value) for value in updated["base_xyz"])
    updated["base_xyz"] = [round(float(base_x), 4), round(float(base_y), 4), round(float(base_z), 4)]
    updated["world_xyz"] = [round(float(base_x), 4), round(float(base_y), 4), round(float(base_z) + height * 0.5, 4)]
    updated["footprint_radius"] = round(0.5 * math.sqrt(float(width) * float(width) + float(depth) * float(depth)), 4)
    return updated


def _candidate_dimensions_for_shape(shape_type: str) -> Tuple[float, float, float]:
    if str(shape_type) in {"cone", "pyramid"}:
        return (0.50, 0.50, 0.56)
    if str(shape_type) == "half_cylinder":
        return (0.58, 0.42, 0.38)
    return (0.50, 0.50, 0.46)


def _sample_anchor_context_specs(rng) -> List[Dict[str, Any]]:
    """Return the two-part asymmetric landmark that makes cross-view correspondence spatial."""

    platform = _make_object_spec(
        object_id="anchor_platform",
        shape_type="cube",
        object_role="context",
        xy=(0.0, 0.0),
        dimensions_xyz=tuple(float(value) for value in MULTIVIEW_ANCHOR_PLATFORM_DIMS),
        dimension_scale=1.0,
        label=None,
    )
    platform.update(
        {
            "object_name": "low platform",
            "prompt_name": "low platform",
            "is_multiview_anchor": True,
            "anchor_part": "platform",
            "fill_rgb": [138, 151, 164],
            "render_order_bias": 2.0,
        }
    )
    platform_height = float(MULTIVIEW_ANCHOR_PLATFORM_DIMS[2])
    marker_x = float(MULTIVIEW_ANCHOR_PLATFORM_DIMS[0]) * 0.34 + float(rng.uniform(-0.03, 0.03))
    marker_y = float(MULTIVIEW_ANCHOR_PLATFORM_DIMS[1]) * 0.32 + float(rng.uniform(-0.03, 0.03))
    marker = _make_object_spec(
        object_id="anchor_corner_marker",
        shape_type="cube",
        object_role="context",
        xy=(marker_x, marker_y),
        dimensions_xyz=tuple(float(value) for value in MULTIVIEW_ANCHOR_MARKER_DIMS),
        dimension_scale=1.0,
        label=None,
    )
    marker = _with_base_z(marker, base_z=platform_height)
    marker.update(
        {
            "object_name": "corner block",
            "prompt_name": "corner block",
            "is_multiview_anchor": True,
            "anchor_part": "corner_marker",
            "fill_rgb": [87, 110, 178],
            "render_order_bias": -2.0,
        }
    )
    return [platform, marker]


def _sample_identical_candidate_specs(
    *,
    rng,
    answer_label: str,
    target_index: int,
) -> Tuple[List[Dict[str, Any]], str, str, List[int]]:
    """Return four same-type, same-color floor candidates so appearance cannot determine the match."""

    shape_type = str(MULTIVIEW_CANDIDATE_SHAPE_TYPES[int(rng.randrange(len(MULTIVIEW_CANDIDATE_SHAPE_TYPES)))])
    fill_rgb = tuple(int(value) for value in MULTIVIEW_CANDIDATE_COLORS[int(rng.randrange(len(MULTIVIEW_CANDIDATE_COLORS)))])
    labels = [str(label) for label in POINT_LABELS[:MULTIVIEW_CANDIDATE_COUNT]]
    remaining_labels = [str(label) for label in labels if str(label) != str(answer_label)]
    rng.shuffle(remaining_labels)
    position_order = list(range(MULTIVIEW_CANDIDATE_COUNT))
    rng.shuffle(position_order)
    specs: List[Dict[str, Any]] = []
    dimensions_xyz = _candidate_dimensions_for_shape(shape_type)
    for stable_index, position_index in enumerate(position_order):
        label = str(answer_label) if int(stable_index) == int(target_index) else str(remaining_labels.pop())
        base_x, base_y = MULTIVIEW_CANDIDATE_POSITIONS[int(position_index)]
        jittered_xy = (
            float(base_x) + float(rng.uniform(-0.055, 0.055)),
            float(base_y) + float(rng.uniform(-0.055, 0.055)),
        )
        object_id = f"object_{stable_index:02d}"
        spec = _make_object_spec(
            object_id=str(object_id),
            shape_type=str(shape_type),
            object_role="candidate",
            xy=jittered_xy,
            dimensions_xyz=tuple(float(value) for value in dimensions_xyz),
            dimension_scale=1.0,
            label=str(label),
        )
        spec.update(
            {
                "object_id": str(object_id),
                "point_id": str(object_id),
                "canonical_object_id": str(object_id),
                "stable_object_index": int(stable_index),
                "point_label": str(label),
                "object_label": str(label),
                "is_answer_candidate": True,
                "fill_rgb": [int(channel) for channel in fill_rgb],
                "multiview_candidate_shape_type": str(shape_type),
                "multiview_identical_candidate_group": "floor_candidates",
                "multiview_position_slot": int(position_index),
            }
        )
        specs.append(spec)
    return list(specs), f"object_{int(target_index):02d}", str(shape_type), [int(channel) for channel in fill_rgb]


def _view_is_valid(
    *,
    candidate_specs: Sequence[Mapping[str, Any]],
    context_specs: Sequence[Mapping[str, Any]],
    target_object_id: str,
    camera,
    frame,
    panel_params: _RenderParams,
) -> bool:
    """Validate one projected view against readability and uniqueness constraints before accepting a multiview sample."""
    candidate_bboxes_by_id = {
        str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=12.0)
        for spec in candidate_specs
    }
    all_bboxes_by_id = {
        str(spec["object_id"]): _object_screen_bbox(spec, camera, frame, pad_px=12.0)
        for spec in [*candidate_specs, *context_specs]
    }
    candidate_bboxes = list(candidate_bboxes_by_id.values())
    if any(not _bbox_is_readable(bbox, width=int(panel_params.canvas_width), height=int(panel_params.canvas_height)) for bbox in candidate_bboxes):
        return False
    if any(_bbox_area(bbox) < MULTIVIEW_MIN_CANDIDATE_BBOX_AREA_PX for bbox in candidate_bboxes):
        return False
    if any(
        _bbox_intersection_area(a, b) > 7200.0
        for index, a in enumerate(candidate_bboxes)
        for b in candidate_bboxes[index + 1 :]
    ):
        return False
    target_bbox = candidate_bboxes_by_id[str(target_object_id)]
    max_allowed_target_overlap = min(1250.0, 0.34 * _bbox_area(target_bbox))
    for other_id, other_bbox in all_bboxes_by_id.items():
        if str(other_id) == str(target_object_id):
            continue
        if _bbox_intersection_area(target_bbox, other_bbox) > max_allowed_target_overlap:
            return False
    screen_centers = [
        (float(_project_screen(spec["world_xyz"], camera, frame)[0]), float(_project_screen(spec["world_xyz"], camera, frame)[1]))
        for spec in candidate_specs
    ]
    return not any(
        math.hypot(a[0] - b[0], a[1] - b[1]) < 36.0
        for index, a in enumerate(screen_centers)
        for b in screen_centers[index + 1 :]
    )


def _build_multiview_scene_dataset(
    *,
    query_id: str,
    scene_variant: str,
    point_count: int,
    context_object_count: int,
    render_params: _RenderParams,
    instance_seed: int,
) -> Dict[str, Any]:
    """Build two camera views of the same 3D scene with identical floor candidates around an asymmetric anchor."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    view_a_yaw_band, view_b_yaw_band = _camera_yaw_bands_for_instance(int(instance_seed))
    point_count = int(MULTIVIEW_CANDIDATE_COUNT)
    answer_label = str(POINT_LABELS[abs(int(instance_seed)) % int(point_count)])
    panel_params = _multiview_source_render_params(render_params)

    for _attempt in range(420):
        camera_a = _sample_camera(rng, yaw_band_degrees=view_a_yaw_band)
        camera_b = _sample_camera(rng, yaw_band_degrees=view_b_yaw_band)
        if _yaw_separation_degrees(float(camera_a.yaw_degrees), float(camera_b.yaw_degrees)) < 72.0:
            continue
        target_index = int(rng.randrange(int(point_count)))
        candidate_specs, target_object_id, candidate_shape_type, candidate_fill_rgb = _sample_identical_candidate_specs(
            rng=rng,
            answer_label=str(answer_label),
            target_index=int(target_index),
        )
        context_specs = _sample_anchor_context_specs(rng)
        all_specs = [*candidate_specs, *context_specs]
        reference_points = [point for spec in all_specs for point in _object_reference_points(spec)]
        frame_a = _build_projection_frame(camera=camera_a, render_params=panel_params, point_worlds=reference_points)
        frame_b = _build_projection_frame(camera=camera_b, render_params=panel_params, point_worlds=reference_points)
        if not _view_is_valid(
            candidate_specs=candidate_specs,
            context_specs=context_specs,
            target_object_id=str(target_object_id),
            camera=camera_a,
            frame=frame_a,
            panel_params=panel_params,
        ):
            continue
        if not _view_is_valid(
            candidate_specs=candidate_specs,
            context_specs=context_specs,
            target_object_id=str(target_object_id),
            camera=camera_b,
            frame=frame_b,
            panel_params=panel_params,
        ):
            continue

        view_a_candidates = _finalize_specs(candidate_specs, camera=camera_a, frame=frame_a)
        view_a_context = _finalize_specs(context_specs, camera=camera_a, frame=frame_a)
        view_b_candidates = _finalize_specs(candidate_specs, camera=camera_b, frame=frame_b)
        view_b_context = _finalize_specs(context_specs, camera=camera_b, frame=frame_b)
        matched_spec = next(spec for spec in view_b_candidates if str(spec["object_id"]) == str(target_object_id))
        if str(matched_spec["point_label"]) != str(answer_label):
            continue
        camera_a_distances = [float(spec["camera_distance"]) for spec in view_a_candidates]
        camera_b_distances = [float(spec["camera_distance"]) for spec in view_b_candidates]
        context_object_count = len(context_specs)
        return {
            "query_id": str(query_id),
            "scene_variant": str(scene_variant),
            "point_count": int(point_count),
            "candidate_count": int(point_count),
            "context_object_count": int(context_object_count),
            "object_count": int(point_count) + int(context_object_count),
            "answer_label": str(answer_label),
            "answer_point_id": str(target_object_id),
            "target_object_id": str(target_object_id),
            "target_shape_type": str(matched_spec["shape_type"]),
            "target_object_name": str(matched_spec["object_name"]),
            "candidate_shape_type": str(candidate_shape_type),
            "candidate_fill_rgb": [int(channel) for channel in candidate_fill_rgb],
            "point_specs": sorted(view_b_candidates, key=lambda spec: str(spec["point_label"])),
            "context_object_specs": sorted(view_b_context, key=lambda spec: str(spec["object_id"])),
            "object_specs": sorted([*view_b_candidates, *view_b_context], key=lambda spec: str(spec["object_id"])),
            "canonical_point_specs": sorted(candidate_specs, key=lambda spec: str(spec["object_id"])),
            "canonical_context_object_specs": sorted(context_specs, key=lambda spec: str(spec["object_id"])),
            "views": {
                REFERENCE_VIEW_KEY: {
                    "view_role": "source_reference",
                    "point_specs": sorted(view_a_candidates, key=lambda spec: str(spec["point_label"])),
                    "context_object_specs": sorted(view_a_context, key=lambda spec: str(spec["object_id"])),
                    "camera": _camera_record(camera_a, yaw_band=view_a_yaw_band),
                    "projection_frame": _frame_record(frame_a),
                },
                CANDIDATE_VIEW_KEY: {
                    "view_role": "answer_candidates",
                    "point_specs": sorted(view_b_candidates, key=lambda spec: str(spec["point_label"])),
                    "context_object_specs": sorted(view_b_context, key=lambda spec: str(spec["object_id"])),
                    "camera": _camera_record(camera_b, yaw_band=view_b_yaw_band),
                    "projection_frame": _frame_record(frame_b),
                },
            },
            "solver_trace": {
                "match_key": "canonical_object_id",
                "target_object_id": str(target_object_id),
                "answer_label": str(answer_label),
                "candidate_labels_by_object_id": {
                    str(spec["object_id"]): str(spec["point_label"])
                    for spec in sorted(view_b_candidates, key=lambda spec: str(spec["object_id"]))
                },
                "same_object_unique_answer": True,
                "candidate_appearance_control": "same_type_same_color",
                "anchor_structure": "low_rectangular_platform_with_corner_block",
                "candidate_shape_type": str(candidate_shape_type),
                "candidate_fill_rgb": [int(channel) for channel in candidate_fill_rgb],
                "view_yaw_separation_degrees": round(
                    float(_yaw_separation_degrees(float(camera_a.yaw_degrees), float(camera_b.yaw_degrees))),
                    4,
                ),
                "view_a_unique_camera_distance_margin": round(float(_min_pairwise(camera_a_distances)), 4),
                "view_b_unique_camera_distance_margin": round(float(_min_pairwise(camera_b_distances)), 4),
            },
        }
    raise ValueError("could not construct a valid 3D multiview object-match scene")


def _camera_record(camera, *, yaw_band: Sequence[float]) -> Dict[str, Any]:
    return {
        "camera_position": [round(float(value), 4) for value in camera.camera_position],
        "target": [round(float(value), 4) for value in camera.target],
        "yaw_degrees": round(float(camera.yaw_degrees), 4),
        "yaw_band_degrees": [round(float(value), 4) for value in yaw_band],
        "pitch_degrees": round(float(camera.pitch_degrees), 4),
        "distance": round(float(camera.distance), 4),
        "right": [round(float(value), 5) for value in camera.right],
        "up": [round(float(value), 5) for value in camera.up],
        "forward": [round(float(value), 5) for value in camera.forward],
    }


def _frame_record(frame) -> Dict[str, Any]:
    return {
        "scale": round(float(frame.scale), 5),
        "center_x": round(float(frame.center_x), 3),
        "center_y": round(float(frame.center_y), 3),
        "normalized_center_u": round(float(frame.normalized_center_u), 6),
        "normalized_center_v": round(float(frame.normalized_center_v), 6),
    }




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
class ThreeDSpatialMultiviewObjectMatchLabelTask:
    """Match a red-boxed object across two camera views of the same 3D scene."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    supported_query_ids = SUPPORTED_QUERY_IDS
    domain = "three_d"
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        last_error: Exception | None = None
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_seed = (
                int(instance_seed)
                if attempt_index == 0
                else int(spawn_rng(int(instance_seed), f"{TASK_ID}.attempt_seed.{attempt_index}").randrange(1, 2**62))
            )
            try:
                return self._generate_once(int(attempt_seed), params=params)
            except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
                last_error = exc
        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts: {last_error}")

    def _generate_once(self, instance_seed: int, *, params: Dict[str, Any]) -> TaskOutput:
        """Generate one multiview object-match instance with identical candidates and a scalar answer bbox."""
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
        point_count = int(MULTIVIEW_CANDIDATE_COUNT)
        context_object_count = 2
        point_count_probabilities = {str(point_count): 1.0}
        context_object_count_probabilities = {str(context_object_count): 1.0}
        render_params = _resolve_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.canvas",
        )
        dataset = _build_multiview_scene_dataset(
            query_id=str(query_id),
            scene_variant=str(scene_variant),
            point_count=int(point_count),
            context_object_count=int(context_object_count),
            render_params=render_params,
            instance_seed=int(instance_seed),
        )
        point_count = int(dataset["point_count"])
        context_object_count = int(dataset["context_object_count"])
        rendered_image, rendered_by_view, background_meta = _render_multiview_scene(
            dataset=dataset,
            render_params=render_params,
            instance_seed=int(instance_seed),
            params=params,
            background_defaults=_BACKGROUND_DEFAULTS,
        )
        image, post_noise_meta = apply_post_image_noise(
            rendered_image,
            instance_seed=int(instance_seed),
            params=params,
            default_config=_NOISE_DEFAULTS,
        )
        panel_params = _multiview_source_render_params(render_params)
        panel_layout = _multiview_scaled_panel_layout(panel_params)
        reference_panel = panel_layout[REFERENCE_VIEW_KEY]
        candidate_panel = panel_layout[CANDIDATE_VIEW_KEY]
        rendered_reference = rendered_by_view[REFERENCE_VIEW_KEY]
        rendered_candidate = rendered_by_view[CANDIDATE_VIEW_KEY]
        answer_label = str(dataset["answer_label"])
        target_object_id = str(dataset["target_object_id"])
        reference_bbox = _offset_bbox(
            rendered_reference.object_bboxes_px[target_object_id],
            dx=float(reference_panel["x"]),
            dy=float(reference_panel["y"]),
        )
        candidate_bbox = _offset_bbox(
            rendered_candidate.point_bboxes_px[answer_label],
            dx=float(candidate_panel["x"]),
            dy=float(candidate_panel["y"]),
        )
        annotation_bbox = list(candidate_bbox)

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

        answer_gt = TypedValue(type="option_letter", value=str(answer_label))
        annotation_gt = TypedValue(type="bbox", value=list(annotation_bbox))
        solver_trace = dict(dataset["solver_trace"])
        reference_maps = _shift_render_maps(rendered_reference, panel=reference_panel)
        candidate_maps = _shift_render_maps(rendered_candidate, panel=candidate_panel)
        scene_entities = [
            *_offset_entities(rendered_reference.entities, dx=float(reference_panel["x"]), dy=float(reference_panel["y"]), view_key=REFERENCE_VIEW_KEY),
            *_offset_entities(rendered_candidate.entities, dx=float(candidate_panel["x"]), dy=float(candidate_panel["y"]), view_key=CANDIDATE_VIEW_KEY),
        ]

        trace_payload = {
            "scene_ir": {
                "scene_kind": "three_d_object_scene_multiview",
                "entities": [dict(entity) for entity in scene_entities],
                "relations": {
                    "scene_variant": str(scene_variant),
                    "point_count": int(point_count),
                    "candidate_count": int(point_count),
                    "context_object_count": int(context_object_count),
                    "object_count": int(dataset["object_count"]),
                    "view_count": 2,
                    "view_keys": [REFERENCE_VIEW_KEY, CANDIDATE_VIEW_KEY],
                    "target_object_id": str(target_object_id),
                    "target_shape_type": str(dataset["target_shape_type"]),
                    "target_object_name": str(dataset["target_object_name"]),
                    "answer_label": str(answer_label),
                    "view_family": "two_camera_synthetic_perspective_3d_scene",
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
                    "point_count_probabilities": dict(point_count_probabilities),
                    "context_object_count": int(context_object_count),
                    "context_object_count_probabilities": dict(context_object_count_probabilities),
                    "object_count": int(dataset["object_count"]),
                    "target_object_id": str(target_object_id),
                },
            },
            "render_spec": {
                "canvas_width": int(image.width),
                "canvas_height": int(image.height),
                "scene_canvas_preset": str(render_params.canvas_preset),
                "scene_canvas_width": int(render_params.canvas_width),
                "scene_canvas_height": int(render_params.canvas_height),
                "scene_canvas_policy": str(render_params.canvas_policy),
                **render_params_canvas_metadata(render_params),
                "final_canvas_width": int(image.width),
                "final_canvas_height": int(image.height),
                "final_canvas_pixels": int(image.width) * int(image.height),
                "coord_space": "pixel",
                "scene_variant": str(scene_variant),
                "background_style": dict(background_meta),
                "post_image_noise": dict(post_noise_meta),
                "panel_layout": dict(panel_layout),
                "panel_render": {
                    "source_canvas_width": int(panel_params.canvas_width),
                    "source_canvas_height": int(panel_params.canvas_height),
                    "panel_canvas_width": int(reference_panel["width"]),
                    "panel_canvas_height": int(reference_panel["height"]),
                    "scale_x": round(float(reference_panel["scale_x"]), 8),
                    "scale_y": round(float(reference_panel["scale_y"]), 8),
                    "label_font_size_px": int(panel_params.label_font_size_px),
                },
                "views": {
                    REFERENCE_VIEW_KEY: {
                        "camera": dict(dataset["views"][REFERENCE_VIEW_KEY]["camera"]),
                        "projection_frame": dict(dataset["views"][REFERENCE_VIEW_KEY]["projection_frame"]),
                    },
                    CANDIDATE_VIEW_KEY: {
                        "camera": dict(dataset["views"][CANDIDATE_VIEW_KEY]["camera"]),
                        "projection_frame": dict(dataset["views"][CANDIDATE_VIEW_KEY]["projection_frame"]),
                    },
                },
            },
            "render_map": {
                "image_id": "img0",
                "scene_bbox_px": [0.0, 0.0, float(image.width), float(image.height)],
                "views": {
                    REFERENCE_VIEW_KEY: dict(reference_maps),
                    CANDIDATE_VIEW_KEY: dict(candidate_maps),
                },
                "reference_view_object_bbox_px": list(reference_bbox),
                "second_view_match_bbox_px": list(candidate_bbox),
                "point_bboxes_px": dict(candidate_maps["point_bboxes_px"]),
                "point_centers_px": dict(candidate_maps["point_centers_px"]),
                "object_bboxes_px": dict(candidate_maps["object_bboxes_px"]),
                "object_centers_px": dict(candidate_maps["object_centers_px"]),
            },
            "execution_trace": {
                "query_id": str(query_id),
                "scene_variant": str(scene_variant),
                "point_count": int(point_count),
                "candidate_count": int(point_count),
                "context_object_count": int(context_object_count),
                "object_count": int(dataset["object_count"]),
                "answer_label": str(answer_label),
                "answer_point_id": str(dataset["answer_point_id"]),
                "target_object_id": str(target_object_id),
                "target_shape_type": str(dataset["target_shape_type"]),
                "target_object_name": str(dataset["target_object_name"]),
                "canonical_point_specs": [dict(spec) for spec in dataset["canonical_point_specs"]],
                "canonical_context_object_specs": [dict(spec) for spec in dataset["canonical_context_object_specs"]],
                "point_specs": [dict(spec) for spec in dataset["point_specs"]],
                "context_object_specs": [dict(spec) for spec in dataset["context_object_specs"]],
                "object_specs": [dict(spec) for spec in dataset["object_specs"]],
                "views": {
                    REFERENCE_VIEW_KEY: {
                        "point_specs": [dict(spec) for spec in dataset["views"][REFERENCE_VIEW_KEY]["point_specs"]],
                        "context_object_specs": [dict(spec) for spec in dataset["views"][REFERENCE_VIEW_KEY]["context_object_specs"]],
                        "camera": dict(dataset["views"][REFERENCE_VIEW_KEY]["camera"]),
                        "projection_frame": dict(dataset["views"][REFERENCE_VIEW_KEY]["projection_frame"]),
                    },
                    CANDIDATE_VIEW_KEY: {
                        "point_specs": [dict(spec) for spec in dataset["views"][CANDIDATE_VIEW_KEY]["point_specs"]],
                        "context_object_specs": [dict(spec) for spec in dataset["views"][CANDIDATE_VIEW_KEY]["context_object_specs"]],
                        "camera": dict(dataset["views"][CANDIDATE_VIEW_KEY]["camera"]),
                        "projection_frame": dict(dataset["views"][CANDIDATE_VIEW_KEY]["projection_frame"]),
                    },
                },
                "question_format": str(query_id),
                "view_family": "two_camera_synthetic_perspective_3d_scene",
                "solver_trace": dict(solver_trace),
            },
            "witness_symbolic": {
                "type": "object_match",
                "ids_by_role": {
                    "reference_view_object": str(target_object_id),
                    "second_view_match": str(target_object_id),
                },
                "answer_label": str(answer_label),
            },
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(annotation_bbox),
                "pixel_bbox": list(annotation_bbox),
            },
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


__all__ = ["ThreeDSpatialMultiviewObjectMatchLabelTask"]
