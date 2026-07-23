"""Count scene icons that are smaller or larger than a reference icon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Tuple

from ....core.scene_config import get_scene_defaults
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_generation_rendering_prompt_defaults,
)
from ...shared.counting_sampling import resolve_counting_target_and_distractor_triplet
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.icon_assets import resolve_icon_pool
from ..shared.icon_scene import (
    IconInstanceSpec,
    panel_geometry_to_trace,
    render_two_panel_icon_scene,
    sort_bboxes_reading_order,
)
from ..shared.defaults import ICON_SHARED_DEFAULTS
from ..shared.icon_style import sample_icon_tints
from ..shared.icon_task_rendering import (
    icon_render_style_trace,
    resolve_icon_render_params,
    sample_icon_instance_noise,
)
from .shared.styles import sample_reference_canvas_palette


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for reference-icon size-relation counting."""

    object_count_min: int = ICON_SHARED_DEFAULTS.object_count_min
    object_count_max: int = 16
    canvas_width: int = ICON_SHARED_DEFAULTS.canvas_width
    canvas_height: int = ICON_SHARED_DEFAULTS.canvas_height
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = ICON_SHARED_DEFAULTS.scene_icon_size_min_px
    scene_icon_size_max_px: int = 120
    reference_icon_size_px: int = 80
    reference_icon_size_min_px: int = 64
    reference_icon_size_max_px: int = 96
    distractor_count_min: int = ICON_SHARED_DEFAULTS.distractor_count_min
    distractor_count_max: int = 8
    scene_max_overlap_fraction: float = ICON_SHARED_DEFAULTS.scene_max_overlap_fraction
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    pool_manifest: str = "all_icons.txt"
    rotation_candidates_degrees: Tuple[int, ...] = (0, 90, 180, 270)
    size_relation_min_delta_px: int = 12
    palette_size_min: int = 8
    palette_size_max: int = 12
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb


@dataclass(frozen=True)
class _ScenePayload:
    """Trace-ready payload for one icon size-relation counting instance."""

    object_count: int
    target_count: int
    distractor_count: int
    reference_icon_id: str
    size_relation: str
    size_relation_min_delta_px: int
    reference_nominal_size_px: int
    scene_nominal_sizes_px: Tuple[int, ...]
    reference_rotation_degrees: int
    scene_rotations_degrees: Tuple[int, ...]
    match_indices: Tuple[int, ...]
    match_bboxes: Tuple[Tuple[int, int, int, int], ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    scene_instances: Tuple[Dict[str, Any], ...]
    reference_instance: Dict[str, Any]


_DEFAULTS = _TaskDefaults()
SCENE_ID = "reference_canvas"
_SCENE_DEFAULTS = get_scene_defaults("icons", SCENE_ID)
TASK_ID = "task_icons__reference_canvas__reference_metric_relation_count"
SIZE_SMALLER_QUERY_ID = "size_smaller"
SIZE_LARGER_QUERY_ID = "size_larger"
SUPPORTED_QUERY_IDS: Tuple[str, str] = (SIZE_SMALLER_QUERY_ID, SIZE_LARGER_QUERY_ID)
_QUERY_ALIASES: Dict[str, str] = {
    "smaller": SIZE_SMALLER_QUERY_ID,
    "larger": SIZE_LARGER_QUERY_ID,
}
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _variant_mapping(defaults: Mapping[str, Any], key: str, *, variant: str) -> Dict[str, Any]:
    """Return per-query defaults for one public branch."""

    raw = defaults.get(str(key), {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"{key} must be a mapping")
    selected = raw.get(str(variant), {})
    if selected is None:
        return {}
    if not isinstance(selected, Mapping):
        raise ValueError(f"{key}.{variant} must be a mapping")
    return {str(name): value for name, value in dict(selected).items()}


def _resolve_query_id(instance_seed: int, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    """Resolve the public size-relation query branch."""

    normalized_params = dict(params)
    explicit = normalized_params.get("query_id")
    if explicit is None and normalized_params.get("size_relation") is not None:
        explicit = normalized_params.get("size_relation")
    if explicit is not None:
        normalized = _QUERY_ALIASES.get(str(explicit), str(explicit))
        if normalized not in SUPPORTED_QUERY_IDS:
            raise ValueError(f"query_id must be one of {SUPPORTED_QUERY_IDS}")
        normalized_params["query_id"] = str(normalized)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.query_id")
    selected, probabilities = resolve_variant(
        rng,
        params=normalized_params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=SUPPORTED_QUERY_IDS,
        explicit_key="query_id",
        weights_key="query_id_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=normalized_params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_QUERY_IDS,
        balance_flag_key="balanced_variant_sampling",
        explicit_key="query_id",
        weights_key="query_id_weights",
        sampling_namespace=f"{TASK_ID}.query_id",
    )
    return str(selected), {str(key): float(value) for key, value in sorted(probabilities.items())}


def _relation_for_query(query_id: str) -> str:
    """Return the concrete size relation requested by one public query."""

    if str(query_id) == SIZE_SMALLER_QUERY_ID:
        return "smaller"
    if str(query_id) == SIZE_LARGER_QUERY_ID:
        return "larger"
    raise ValueError(f"unsupported query_id: {query_id}")


def _rotation_candidates(params: Mapping[str, Any], gen_defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    """Resolve the supported scene rotations."""

    raw = params.get(
        "rotation_candidates_degrees",
        group_default(gen_defaults, "rotation_candidates_degrees", list(_DEFAULTS.rotation_candidates_degrees)),
    )
    if not isinstance(raw, (list, tuple)):
        raise ValueError("rotation_candidates_degrees must be a sequence")
    rotations = tuple(int(value) % 360 for value in raw)
    if not rotations:
        raise ValueError("rotation_candidates_degrees must contain at least one rotation")
    return rotations


def _sample_reference_nominal_size(
    rng,
    *,
    render_params: Mapping[str, Any],
    min_size_delta_px: int,
) -> int:
    """Sample one reference nominal size that leaves room on both sides of the relation."""

    ref_min = int(render_params["reference_icon_size_min_px"])
    ref_max = int(render_params["reference_icon_size_max_px"])
    scene_min = int(render_params["scene_icon_size_min_px"])
    scene_max = int(render_params["scene_icon_size_max_px"])
    size_delta = max(1, int(min_size_delta_px))
    lo = max(int(ref_min), int(scene_min) + int(size_delta))
    hi = min(int(ref_max), int(scene_max) - int(size_delta))
    if int(lo) > int(hi):
        raise ValueError("reference icon size range is infeasible for the configured size-relation delta")
    return int(rng.randint(int(lo), int(hi)))


def _sample_scene_nominal_size(
    rng,
    *,
    relation: str,
    is_match: bool,
    reference_nominal_size_px: int,
    scene_size_min_px: int,
    scene_size_max_px: int,
    min_size_delta_px: int,
) -> int:
    """Sample one scene icon nominal size that is clearly smaller or larger than the reference."""

    ref_size = int(reference_nominal_size_px)
    scene_min = int(scene_size_min_px)
    scene_max = int(scene_size_max_px)
    size_delta = max(1, int(min_size_delta_px))
    if str(relation) == "smaller":
        smaller_bounds = (int(scene_min), int(ref_size) - int(size_delta))
        larger_bounds = (int(ref_size) + int(size_delta), int(scene_max))
        lo, hi = smaller_bounds if bool(is_match) else larger_bounds
    elif str(relation) == "larger":
        smaller_bounds = (int(scene_min), int(ref_size) - int(size_delta))
        larger_bounds = (int(ref_size) + int(size_delta), int(scene_max))
        lo, hi = larger_bounds if bool(is_match) else smaller_bounds
    else:
        raise ValueError(f"unsupported size relation: {relation}")
    if int(lo) > int(hi):
        raise ValueError("scene icon size support is empty for the selected size relation")
    return int(rng.randint(int(lo), int(hi)))


def _sample_scene(
    rng,
    *,
    instance_seed: int,
    object_count: int,
    target_count: int,
    pool_manifest: str,
    size_relation: str,
    rotation_candidates: Tuple[int, ...],
    min_size_delta_px: int,
    render_params: Mapping[str, Any],
) -> Tuple[_ScenePayload, Any]:
    """Sample and render one reference+scene size-relation counting scene."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if not pool:
        raise ValueError("size-relation pool resolved no icons")
    reference_icon_id = str(rng.choice(pool))
    reference_rotation = int(rng.choice(rotation_candidates))
    reference_nominal_size_px = _sample_reference_nominal_size(
        rng,
        render_params=render_params,
        min_size_delta_px=int(min_size_delta_px),
    )

    match_indices = set(rng.sample(list(range(int(object_count))), int(target_count)))
    palette = sample_reference_canvas_palette(rng, render_params)
    sampled_tints = list(sample_icon_tints(rng, palette=palette, count=int(object_count) + 1))
    reference_tint = tuple(int(v) for v in sampled_tints.pop(0))

    scene_specs: List[IconInstanceSpec] = []
    scene_nominal_sizes_px: List[int] = []
    scene_rotations: List[int] = []
    for index in range(int(object_count)):
        nominal_size_px = _sample_scene_nominal_size(
            rng,
            relation=str(size_relation),
            is_match=int(index) in match_indices,
            reference_nominal_size_px=int(reference_nominal_size_px),
            scene_size_min_px=int(render_params["scene_icon_size_min_px"]),
            scene_size_max_px=int(render_params["scene_icon_size_max_px"]),
            min_size_delta_px=int(min_size_delta_px),
        )
        rotation = int(rng.choice(rotation_candidates))
        scene_nominal_sizes_px.append(int(nominal_size_px))
        scene_rotations.append(int(rotation))
        scene_specs.append(
            IconInstanceSpec(
                icon_id=str(reference_icon_id),
                nominal_size_px=int(nominal_size_px),
                rotation_degrees=int(rotation),
                tint_rgb=tuple(int(v) for v in sampled_tints.pop(0)),
            )
        )

    reference_noise_edits, reference_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:reference_icon",
        render_params=render_params,
    )
    for index, spec in enumerate(list(scene_specs)):
        scene_noise_edits, scene_noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:scene_icon_{int(index)}",
            render_params=render_params,
        )
        scene_specs[index] = IconInstanceSpec(
            icon_id=str(spec.icon_id),
            nominal_size_px=int(spec.nominal_size_px) if spec.nominal_size_px is not None else None,
            rotation_degrees=int(spec.rotation_degrees),
            mirror_x=bool(spec.mirror_x),
            tint_rgb=tuple(int(v) for v in spec.tint_rgb),
            noise_edits=tuple(scene_noise_edits),
            noise_seed=int(scene_noise_seed),
        )

    rendered = render_two_panel_icon_scene(
        rng=rng,
        reference_icon=IconInstanceSpec(
            icon_id=str(reference_icon_id),
            nominal_size_px=int(reference_nominal_size_px),
            rotation_degrees=int(reference_rotation),
            tint_rgb=tuple(int(v) for v in reference_tint),
            noise_edits=tuple(reference_noise_edits),
            noise_seed=int(reference_noise_seed),
        ),
        scene_icons=scene_specs,
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        panel_corner_radius_px=int(render_params["panel_corner_radius_px"]),
        scene_icon_size_min_px=int(render_params["scene_icon_size_min_px"]),
        scene_icon_size_max_px=int(render_params["scene_icon_size_max_px"]),
        reference_icon_size_px=int(reference_nominal_size_px),
        scene_max_overlap_fraction=float(render_params["scene_max_overlap_fraction"]),
        scene_placement_max_attempts=int(render_params["scene_placement_max_attempts"]),
        scene_size_shrink_rounds=int(render_params["scene_size_shrink_rounds"]),
        scene_size_shrink_factor=float(render_params["scene_size_shrink_factor"]),
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    match_bboxes = tuple(
        tuple(int(value) for value in rendered.scene_instances[int(index)].bbox_xyxy)
        for index in sorted(int(value) for value in match_indices)
    )
    scene_instances = tuple(
        {
            "instance_id": str(instance.instance_id),
            "icon_id": str(instance.icon_id),
            "panel": str(instance.panel),
            "bbox_xyxy": list(instance.bbox_xyxy),
            "nominal_size_px": int(instance.nominal_size_px),
            "rotation_degrees": int(instance.rotation_degrees),
            "mirror_x": bool(instance.mirror_x),
            "tint_rgb": list(instance.tint_rgb),
            "noise_edits": [dict(edit) for edit in instance.noise_edits],
            "noise_seed": None if instance.noise_seed is None else int(instance.noise_seed),
            "is_match": bool(index in match_indices),
            "index": int(index),
        }
        for index, instance in enumerate(rendered.scene_instances)
    )
    reference_instance = {
        "instance_id": str(rendered.reference_instance.instance_id),
        "icon_id": str(rendered.reference_instance.icon_id),
        "panel": str(rendered.reference_instance.panel),
        "bbox_xyxy": list(rendered.reference_instance.bbox_xyxy),
        "nominal_size_px": int(rendered.reference_instance.nominal_size_px),
        "rotation_degrees": int(rendered.reference_instance.rotation_degrees),
        "mirror_x": bool(rendered.reference_instance.mirror_x),
        "tint_rgb": list(rendered.reference_instance.tint_rgb),
        "noise_edits": [dict(edit) for edit in rendered.reference_instance.noise_edits],
        "noise_seed": None if rendered.reference_instance.noise_seed is None else int(rendered.reference_instance.noise_seed),
    }
    return _ScenePayload(
        object_count=int(object_count),
        target_count=int(target_count),
        distractor_count=int(object_count) - int(target_count),
        reference_icon_id=str(reference_icon_id),
        size_relation=str(size_relation),
        size_relation_min_delta_px=int(min_size_delta_px),
        reference_nominal_size_px=int(reference_nominal_size_px),
        scene_nominal_sizes_px=tuple(int(value) for value in scene_nominal_sizes_px),
        reference_rotation_degrees=int(reference_rotation),
        scene_rotations_degrees=tuple(int(value) for value in scene_rotations),
        match_indices=tuple(sorted(int(value) for value in match_indices)),
        match_bboxes=match_bboxes,
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in palette),
        panel_geometry=panel_geometry_to_trace(rendered.layout),
        scene_instances=scene_instances,
        reference_instance=reference_instance,
    ), rendered.image


@register_task
class IconsReferenceCanvasReferenceMetricRelationCountTask:
    """Count scene icons that are smaller or larger than the reference icon."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = "icons"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic reference-icon size-relation counting instance."""

        query_id, query_id_probabilities = _resolve_query_id(int(instance_seed), params)
        active_gen_defaults = dict(_GEN_DEFAULTS)
        active_gen_defaults.update(_variant_mapping(_GEN_DEFAULTS, "variant_generation_params", variant=str(query_id)))
        active_render_defaults = dict(_RENDER_DEFAULTS)
        active_render_defaults.update(_variant_mapping(_RENDER_DEFAULTS, "variant_render_params", variant=str(query_id)))

        scene_rng = spawn_rng(int(instance_seed), "scene")
        (
            object_count,
            object_count_probabilities,
            target_count,
            target_count_probabilities,
            distractor_count,
            distractor_count_probabilities,
        ) = resolve_counting_target_and_distractor_triplet(
            scene_rng,
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=active_gen_defaults,
            fallback_total_min=_DEFAULTS.object_count_min,
            fallback_total_max=_DEFAULTS.object_count_max,
            fallback_target_min=0,
            fallback_target_max=8,
            fallback_distractor_min=_DEFAULTS.distractor_count_min,
            fallback_distractor_max=_DEFAULTS.distractor_count_max,
        )
        render_params = resolve_icon_render_params(
            params=params,
            render_defaults=active_render_defaults,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        pool_manifest = str(params.get("pool_manifest", group_default(active_gen_defaults, "pool_manifest", _DEFAULTS.pool_manifest)))
        rotation_candidates = _rotation_candidates(params, active_gen_defaults)
        size_relation = _relation_for_query(str(query_id))
        min_size_delta_px = int(
            params.get(
                "size_relation_min_delta_px",
                group_default(active_gen_defaults, "size_relation_min_delta_px", _DEFAULTS.size_relation_min_delta_px),
            )
        )
        if int(min_size_delta_px) < 1:
            raise ValueError("size_relation_min_delta_px must be positive")

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = _sample_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    object_count=int(object_count),
                    target_count=int(target_count),
                    pool_manifest=str(pool_manifest),
                    size_relation=str(size_relation),
                    rotation_candidates=rotation_candidates,
                    min_size_delta_px=int(min_size_delta_px),
                    render_params=render_params,
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError(f"failed to generate {self.task_id} instance") from last_error

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
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        annotation_bboxes = sort_bboxes_reading_order(scene_payload.match_bboxes)
        annotation_payload = icon_bbox_set_annotation(
            annotation_bboxes,
            clip_bbox=scene_payload.panel_geometry["scene_content_xyxy"],
        )
        answer_gt = TypedValue(type="integer", value=int(scene_payload.target_count))
        annotation_gt = TypedValue(
            type=str(annotation_payload["annotation_type"]),
            value=list(annotation_payload["annotation_value"]),
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_reference_counting_size_relation",
                "entities": [dict(scene_payload.reference_instance), *[dict(item) for item in scene_payload.scene_instances]],
                "relations": {
                    "query_id": str(query_id),
                    "counting_target": f"same_icon_type_and_{size_relation}_than_reference",
                    "reference_icon_id": str(scene_payload.reference_icon_id),
                    "reference_nominal_size_px": int(scene_payload.reference_nominal_size_px),
                    "size_relation": str(size_relation),
                    "size_relation_min_delta_px": int(scene_payload.size_relation_min_delta_px),
                    "matching_scene_indices": list(scene_payload.match_indices),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": {
                "query_id": str(query_id),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "object_count": int(object_count),
                    "object_count_probabilities": dict(object_count_probabilities),
                    "target_count": int(target_count),
                    "target_count_probabilities": dict(target_count_probabilities),
                    "distractor_count": int(distractor_count),
                    "distractor_count_probabilities": dict(distractor_count_probabilities),
                    "pool_manifest": str(pool_manifest),
                    "rotation_candidates_degrees": [int(value) for value in rotation_candidates],
                    "size_relation": str(size_relation),
                    "size_relation_probabilities": dict(query_id_probabilities),
                    "query_id_probabilities": dict(query_id_probabilities),
                    "size_relation_min_delta_px": int(min_size_delta_px),
                },
            },
            "render_spec": {
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": icon_render_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                ),
            },
            "render_map": {
                "image_id": "img0",
                "anchors": {
                    "reference_icon": dict(scene_payload.reference_instance),
                    "matching_scene_boxes": list(annotation_payload["annotation_value"]),
                },
            },
            "execution_trace": {
                "scene_variant": "reference_scene",
                "query_id": str(query_id),
                "object_count": int(object_count),
                "object_count_probabilities": dict(object_count_probabilities),
                "target_count": int(target_count),
                "target_count_probabilities": dict(target_count_probabilities),
                "distractor_count": int(distractor_count),
                "distractor_count_probabilities": dict(distractor_count_probabilities),
                "reference_icon_id": str(scene_payload.reference_icon_id),
                "reference_nominal_size_px": int(scene_payload.reference_nominal_size_px),
                "reference_rotation_degrees": int(scene_payload.reference_rotation_degrees),
                "scene_icon_ids": [str(scene_payload.reference_icon_id)] * int(object_count),
                "scene_nominal_sizes_px": list(scene_payload.scene_nominal_sizes_px),
                "scene_rotations_degrees": list(scene_payload.scene_rotations_degrees),
                "matching_scene_indices": list(scene_payload.match_indices),
                "size_relation": str(size_relation),
                "size_relation_probabilities": dict(query_id_probabilities),
                "query_id_probabilities": dict(query_id_probabilities),
                "size_relation_min_delta_px": int(scene_payload.size_relation_min_delta_px),
                "question_format": "count_matching_scene_icons_by_size_relation",
            },
            "witness_symbolic": {
                "reference_icon_id": str(scene_payload.reference_icon_id),
                "reference_nominal_size_px": int(scene_payload.reference_nominal_size_px),
                "size_relation": str(size_relation),
                "matching_scene_indices": list(scene_payload.match_indices),
            },
            "projected_annotation": dict(annotation_payload["projected_annotation"]),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["IconsReferenceCanvasReferenceMetricRelationCountTask", "SUPPORTED_QUERY_IDS"]
