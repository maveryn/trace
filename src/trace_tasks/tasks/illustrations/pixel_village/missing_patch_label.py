"""Select the patch option that restores a pixel-village source image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.scene_config import get_scene_defaults
from ....core.seed import spawn_rng
from ....core.sampling import support_probability_map, uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ..shared.cutouts import (
    DEFAULT_OPTION_LABELS,
    FRAMELESS_ILLUSTRATION_PATCH_STYLE,
    PATCH_MODE_PLAIN,
    compose_patch_options,
    downscale_patch_option_artifacts,
    style_trace,
)
from ..shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS
from ..shared.missing_patch_sizing import sample_missing_patch_size
from ..shared.option_rendering import sample_visual_label_font_trace
from .shared.output import pixel_village_scene_ir
from .shared.prompts import build_pixel_village_prompt_artifacts
from .shared.sampling import SCENE_ID
from .shared.source_images import (
    build_pixel_village_source_spec,
    candidate_crop_boxes_from_scene_entities,
    render_pixel_village_source_scene,
    source_panel_for_scene,
)


TASK_ID = "task_illustrations__pixel_village__missing_patch_label"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "missing_patch_label"


@dataclass(frozen=True)
class _Defaults:
    option_count_support: Tuple[int, ...] = (4, 6)
    patch_width_ratio_min: float = 0.15
    patch_width_ratio_max: float = 0.30
    patch_height_ratio_min: float = 0.15
    patch_height_ratio_max: float = 0.26
    patch_area_ratio_max: float = 0.065
    crop_margin_px: int = 36
    source_width: int = 820
    source_height: int = 615


@dataclass(frozen=True)
class _SampleSpec:
    option_count: int
    correct_index: int
    patch_size: Tuple[int, int]
    patch_size_trace: Dict[str, Any]
    crop_margin_px: int
    option_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _option_count_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("option_count_support", group_default(_GEN_DEFAULTS, "option_count_support", _DEFAULTS.option_count_support))
    raw_values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    support = tuple(dict.fromkeys(int(value) for value in raw_values if int(value) in set(_DEFAULTS.option_count_support)))
    if not support:
        raise ValueError("option_count_support must include 4 or 6")
    return support


def _sample_option_count(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[int, Dict[str, float]]:
    support = _option_count_support(params)
    explicit = params.get("option_count")
    if explicit is not None:
        option_count = int(explicit)
        if option_count not in set(support):
            raise ValueError(f"option_count must be one of {support}")
        return int(option_count), support_probability_map(support, selected=int(option_count), sort_keys=True)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:option_count")
    selected, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(selected), dict(probabilities)


def _sample_correct_index(*, params: Mapping[str, Any], instance_seed: int, option_count: int) -> Tuple[int, Dict[str, float]]:
    explicit = params.get("correct_index")
    if explicit is not None:
        value = int(explicit)
        if value < 0 or value >= int(option_count):
            raise ValueError("correct_index outside option support")
        return int(value), {str(value): 1.0}
    if params.get("answer_label") is not None:
        label = str(params["answer_label"])
        labels = DEFAULT_OPTION_LABELS[: int(option_count)]
        if label not in set(labels):
            raise ValueError("answer_label outside option support")
        return int(labels.index(label)), {str(labels.index(label)): 1.0}
    namespace = f"{TASK_ID}:answer"
    if params.get("_sample_cursor") is not None:
        namespace = f"{namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), namespace)
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        tuple(range(int(option_count))),
        sort_keys=True,
    )
    return int(selected), dict(probabilities)


def _sample_spec(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    attempt_index: int,
    source_size: Sequence[int] | None = None,
) -> _SampleSpec:
    """Sample patch-option operands for a pixel-village source image."""

    option_count, option_count_probabilities = _sample_option_count(params=params, instance_seed=int(instance_seed))
    correct_index, correct_index_probabilities = _sample_correct_index(
        params=params,
        instance_seed=int(instance_seed),
        option_count=int(option_count),
    )
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:patch_spec", int(attempt_index))
    resolved_source_size = (
        tuple(int(value) for value in source_size)
        if source_size is not None
        else (int(_DEFAULTS.source_width), int(_DEFAULTS.source_height))
    )
    patch_sample = sample_missing_patch_size(
        rng=rng,
        params=params,
        defaults=_GEN_DEFAULTS,
        source_size=resolved_source_size,
        fallback_width_ratio_min=_DEFAULTS.patch_width_ratio_min,
        fallback_width_ratio_max=_DEFAULTS.patch_width_ratio_max,
        fallback_height_ratio_min=_DEFAULTS.patch_height_ratio_min,
        fallback_height_ratio_max=_DEFAULTS.patch_height_ratio_max,
        fallback_area_ratio_max=_DEFAULTS.patch_area_ratio_max,
    )
    crop_margin = int(params.get("crop_margin_px", group_default(_GEN_DEFAULTS, "crop_margin_px", _DEFAULTS.crop_margin_px)))
    return _SampleSpec(
        option_count=int(option_count),
        correct_index=int(correct_index),
        patch_size=tuple(int(value) for value in patch_sample.patch_size),
        patch_size_trace=dict(patch_sample.trace()),
        crop_margin_px=int(crop_margin),
        option_count_probabilities=dict(option_count_probabilities),
        correct_index_probabilities=dict(correct_index_probabilities),
    )


def _bbox_map(value: Mapping[str, Sequence[float]]) -> Dict[str, list[float]]:
    return {str(key): [round(float(coord), 3) for coord in bbox[:4]] for key, bbox in value.items()}


@register_task
class IllustrationsPixelVillageMissingPatchLabelTask:
    """Select the patch option that restores a missing pixel-village region."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "illustrations"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render a pixel-village source panel, compose patch options, and bind keyed annotation."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        artifacts = None
        frame_style = None
        label_font_trace: Dict[str, Any] | None = None
        source_spec = build_pixel_village_source_spec(
            params=params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            fallback_source_width=_DEFAULTS.source_width,
            fallback_source_height=_DEFAULTS.source_height,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:source_profile",
        )
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(
                    instance_seed=int(instance_seed),
                    params=params,
                    attempt_index=int(attempt),
                    source_size=source_spec.source_size,
                )
                scene = render_pixel_village_source_scene(
                    seed_namespace=TASK_ID,
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    source_spec=source_spec,
                )
                option_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:patch_options", int(attempt))
                frame_style = {"style_id": "frameless_illustration", **dict(FRAMELESS_ILLUSTRATION_PATCH_STYLE)}
                label_font_trace = sample_visual_label_font_trace(
                    namespace_prefix=TASK_ID,
                    instance_seed=int(instance_seed),
                    params={**dict(_RENDER_DEFAULTS), **dict(params)},
                    namespace_suffix="patch_option_labels",
                    explicit_key="patch_label_font_family",
                    weights_key="patch_label_font_weights",
                )
                source_panel = source_panel_for_scene(scene, source_spec.source_size)
                candidate_boxes = candidate_crop_boxes_from_scene_entities(
                    scene=scene,
                    source_size=source_spec.source_size,
                    patch_size=sample.patch_size,
                    margin_px=int(sample.crop_margin_px),
                )
                artifacts = compose_patch_options(
                    source_image=source_panel,
                    rng=option_rng,
                    patch_mode=PATCH_MODE_PLAIN,
                    correct_index=int(sample.correct_index),
                    option_count=int(sample.option_count),
                    patch_size=sample.patch_size,
                    crop_margin_px=int(sample.crop_margin_px),
                    frame_style=frame_style,
                    label_font_family=str(label_font_trace["font_family"]),
                    render_margin=0,
                    option_gap=14,
                    source_option_gap=16,
                    show_source_label=False,
                    draw_source_outline=False,
                    draw_option_outlines=False,
                    candidate_crop_boxes=candidate_boxes or None,
                )
                artifacts = downscale_patch_option_artifacts(
                    artifacts,
                    max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS,
                )
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                sample = None
                scene = None
                artifacts = None
                frame_style = None
                label_font_trace = None
        if sample is None or scene is None or artifacts is None or frame_style is None or label_font_trace is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        answer_label = str(artifacts.selected_label)
        annotation_value = _bbox_map(
            {
                "missing_region": artifacts.missing_region_bbox,
                "selected_option": artifacts.selected_option_bbox,
            }
        )
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_missing_patch",
                "annotation_hint_missing_patch",
                "json_example_missing_patch",
                "json_example_answer_only_missing_patch",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        slots = {
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(prompt_defaults["answer_hint_missing_patch"]),
            "annotation_hint": str(prompt_defaults["annotation_hint_missing_patch"]),
            "json_example": str(prompt_defaults["json_example_missing_patch"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only_missing_patch"]),
        }
        prompt_artifacts = build_pixel_village_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots=slots,
            instance_seed=int(instance_seed),
        )
        option_labels = list(DEFAULT_OPTION_LABELS[: int(sample.option_count)])
        trace_payload = {
            "scene_ir": pixel_village_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={
                    "query_id": QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "patch_mode": PATCH_MODE_PLAIN,
                    "answer_label": answer_label,
                },
            ),
            "query_spec": {
                "task_id": self.task_id,
                "scene_id": SCENE_ID,
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "patch_mode": PATCH_MODE_PLAIN,
                    "option_count": int(sample.option_count),
                    "option_count_support": [int(value) for value in _option_count_support(params)],
                    "option_count_probabilities": dict(sample.option_count_probabilities),
                    "option_labels": option_labels,
                    "answer_label": answer_label,
                    "correct_index": int(sample.correct_index),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                    "patch_size": [int(sample.patch_size[0]), int(sample.patch_size[1])],
                    "patch_size_ratio": dict(sample.patch_size_trace),
                    "crop_margin_px": int(sample.crop_margin_px),
                    "source_size": [int(value) for value in source_spec.source_size],
                    "canvas_profile": str(source_spec.canvas_profile),
                    "canvas_profile_size": [int(value) for value in source_spec.source_size],
                    "canvas_profile_probabilities": dict(source_spec.canvas_profile_probabilities),
                    "source_render_modes": {
                        "theme_mode": str(source_spec.theme_mode),
                        "cemetery_mode": str(source_spec.cemetery_mode),
                        "orchard_mode": str(source_spec.orchard_mode),
                        "windmill_mode": str(source_spec.windmill_mode),
                        "river_mode": str(source_spec.river_mode),
                        "river_orientation": str(source_spec.river_orientation),
                        "river_placement": str(source_spec.river_placement),
                    },
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(scene.image.width), int(scene.image.height)],
                "source_profile": {
                    "canvas_profile": str(source_spec.canvas_profile),
                    "canvas_profile_size": [int(value) for value in source_spec.source_size],
                    "canvas_profile_probabilities": dict(source_spec.canvas_profile_probabilities),
                },
                "style": {
                    "source_renderer_id": str(scene.trace.get("renderer_id", "")),
                    "source_theme_id": str(scene.trace.get("theme_id", "")),
                    "source_tile_px": int(scene.trace.get("tile_px", 0)),
                    "patch_frame_style": style_trace(frame_style),
                    "patch_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "image_id": "img0",
                "missing_region_bbox_px": list(artifacts.missing_region_bbox),
                "option_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.option_bboxes.items()},
                "selected_option_bbox_px": list(artifacts.selected_option_bbox),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "option_source_crop_boxes_px": [[int(coord) for coord in box] for box in artifacts.option_source_crop_boxes],
                "selected_transform": str(artifacts.selected_transform),
                "option_grid_shape": [int(artifacts.option_grid_shape[0]), int(artifacts.option_grid_shape[1])],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "patch_mode": PATCH_MODE_PLAIN,
                "answer": answer_label,
                "answer_label": answer_label,
                "correct_index": int(sample.correct_index),
                "selected_transform": str(artifacts.selected_transform),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "option_source_crop_boxes_px": [[int(coord) for coord in box] for box in artifacts.option_source_crop_boxes],
                "option_labels": option_labels,
            },
            "witness_symbolic": {
                "missing_region_bbox": list(artifacts.missing_region_bbox),
                "selected_option_bbox": list(artifacts.selected_option_bbox),
                "answer_label": answer_label,
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": dict(annotation_value),
                "pixel_bbox_map": dict(annotation_value),
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="option_letter", value=answer_label),
            annotation_gt=TypedValue(type="bbox_map", value=dict(annotation_value)),
            image=artifacts.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
        )


__all__ = ["IllustrationsPixelVillageMissingPatchLabelTask", "TASK_ID", "_sample_spec"]
