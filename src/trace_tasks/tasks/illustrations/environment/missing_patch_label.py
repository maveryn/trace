"""Select the patch option that restores an environment illustration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.sampling import support_probability_map, uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ..shared.cutouts import (
    DEFAULT_OPTION_LABELS,
    PATCH_FRAME_STYLES,
    PATCH_MODE_PLAIN,
    compose_patch_options,
    downscale_patch_option_artifacts,
    sample_style,
    style_trace,
)
from ..shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS, resolve_reconstruction_source_profile
from ..shared.missing_patch_sizing import sample_missing_patch_size
from ..shared.option_rendering import sample_visual_label_font_trace
from ..shared.task_support import uniform_string_probability_map
from .shared.annotations import feature_bbox_map, feature_path_map
from .shared.defaults import CountContractDefaults, render_fallback
from .shared.output import serialize_environment_objects
from .shared.prompts import render_environment_prompt, required_environment_prompt_defaults
from .shared.rendering import ENVIRONMENT_THEME_IDS, environment_scene_entities, render_environment_object_scene, serialize_environment_scene
from .shared.sampling import environment_render_params, environment_setting_name, sample_count_support, style_weights, theme_support


TASK_ID = "task_illustrations__environment__missing_patch_label"
DOMAIN = "illustrations"
SCENE_ID = "environment"
QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)


@dataclass(frozen=True)
class _Defaults:
    source_object_count_min: int = 12
    source_object_count_max: int = 18
    option_count_support: Tuple[int, ...] = (4, 6)
    patch_width_ratio_min: float = 0.15
    patch_width_ratio_max: float = 0.30
    patch_height_ratio_min: float = 0.15
    patch_height_ratio_max: float = 0.26
    patch_area_ratio_max: float = 0.065
    crop_margin_px: int = 34
    source_width: int = 820
    source_height: int = 560


@dataclass(frozen=True)
class _SampleSpec:
    theme_id: str
    theme_probabilities: Dict[str, float]
    source_object_count: int
    source_object_count_probabilities: Dict[str, float]
    option_count: int
    correct_index: int
    patch_size: Tuple[int, int]
    patch_size_trace: Dict[str, Any]
    crop_margin_px: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    option_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_RENDER_FALLBACK = render_fallback(
    CountContractDefaults(
        object_count_min=_DEFAULTS.source_object_count_min,
        object_count_max=_DEFAULTS.source_object_count_max,
        target_count_min=0,
        target_count_max=0,
    )
)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _int_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer value from params, defaults, or fallback."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _sample_theme(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[str, Dict[str, float]]:
    """Resolve the source environment theme used in the patch panel."""

    themes = theme_support(params, _GEN_DEFAULTS)
    explicit = params.get("theme_id")
    if explicit is not None:
        theme_id = str(explicit)
        if theme_id not in set(themes):
            raise ValueError(f"theme_id must be one of {themes}")
        return theme_id, uniform_string_probability_map(themes, selected=theme_id)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:theme")
    theme_id, probabilities = uniform_choice_with_probabilities(rng, themes, sort_keys=False)
    return str(theme_id), dict(probabilities)


def _option_count_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("option_count_support", group_default(_GEN_DEFAULTS, "option_count_support", _DEFAULTS.option_count_support))
    raw_values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    values = tuple(dict.fromkeys(int(value) for value in raw_values if int(value) in set(_DEFAULTS.option_count_support)))
    if not values:
        raise ValueError("option_count_support must include 4 or 6")
    return values


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


def _sample_source_object_count(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[int, Dict[str, float]]:
    low = int(params.get("source_object_count_min", group_default(_GEN_DEFAULTS, "source_object_count_min", _DEFAULTS.source_object_count_min)))
    high = int(params.get("source_object_count_max", group_default(_GEN_DEFAULTS, "source_object_count_max", _DEFAULTS.source_object_count_max)))
    if low < 1 or high < low:
        raise ValueError("invalid source_object_count_min/source_object_count_max range")
    return sample_count_support(
        params=params,
        support=tuple(range(int(low), int(high) + 1)),
        explicit_key="source_object_count",
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_object_count",
    )


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample semantic source-scene operands and fixed plain-patch option layout."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample", int(attempt_index))
    theme_id, theme_probabilities = _sample_theme(params=params, instance_seed=int(instance_seed))
    source_object_count, source_object_count_probabilities = _sample_source_object_count(
        params=params,
        instance_seed=int(instance_seed),
    )
    option_count, option_count_probabilities = _sample_option_count(params=params, instance_seed=int(instance_seed))
    source_profile = resolve_reconstruction_source_profile(
        params=params,
        defaults=_GEN_DEFAULTS,
        fallback_source_width=_DEFAULTS.source_width,
        fallback_source_height=_DEFAULTS.source_height,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
    if params.get("correct_index") is not None:
        correct_index = int(params["correct_index"])
        if correct_index < 0 or correct_index >= int(option_count):
            raise ValueError("correct_index outside option support")
        correct_index_probabilities = {str(correct_index): 1.0}
    else:
        support = tuple(range(int(option_count)))
        answer_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:answer")
        correct_index, correct_index_probabilities = uniform_choice_with_probabilities(
            answer_rng,
            support,
            sort_keys=True,
        )
        correct_index = int(correct_index)
        correct_index_probabilities = dict(correct_index_probabilities)

    patch_sample = sample_missing_patch_size(
        rng=rng,
        params=params,
        defaults=_GEN_DEFAULTS,
        source_size=source_profile.size,
        fallback_width_ratio_min=_DEFAULTS.patch_width_ratio_min,
        fallback_width_ratio_max=_DEFAULTS.patch_width_ratio_max,
        fallback_height_ratio_min=_DEFAULTS.patch_height_ratio_min,
        fallback_height_ratio_max=_DEFAULTS.patch_height_ratio_max,
        fallback_area_ratio_max=_DEFAULTS.patch_area_ratio_max,
    )
    return _SampleSpec(
        theme_id=str(theme_id),
        theme_probabilities=dict(theme_probabilities),
        source_object_count=int(source_object_count),
        source_object_count_probabilities=dict(source_object_count_probabilities),
        option_count=int(option_count),
        correct_index=int(correct_index),
        patch_size=tuple(int(value) for value in patch_sample.patch_size),
        patch_size_trace=dict(patch_sample.trace()),
        crop_margin_px=_int_value(params, _GEN_DEFAULTS, "crop_margin_px", _DEFAULTS.crop_margin_px),
        source_size=tuple(int(value) for value in source_profile.size),
        source_profile_trace=dict(source_profile.trace()),
        option_count_probabilities=dict(option_count_probabilities),
        correct_index_probabilities=dict(correct_index_probabilities),
    )


def _bbox_map(value: Mapping[str, Sequence[float]]) -> Dict[str, list[float]]:
    """Return a normalized keyed pixel-bbox map."""

    return {
        str(key): [round(float(coord), 3) for coord in bbox[:4]]
        for key, bbox in value.items()
    }


@register_task
class IllustrationsEnvironmentMissingPatchLabelTask:
    """Select the exact patch option that matches a missing environment region."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render one source scene, compose patch options, and bind keyed annotation from that trace."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        source_scene = None
        artifacts = None
        frame_style = None
        label_font_trace: Dict[str, Any] | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                render_overrides = {
                    **dict(params),
                    "canvas_width": int(sample.source_size[0]),
                    "canvas_height": int(sample.source_size[1]),
                }
                render_params = environment_render_params(
                    render_overrides,
                    _RENDER_DEFAULTS,
                    fallback=_RENDER_FALLBACK,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}:source_profile",
                )
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:source_scene", int(attempt))
                source_scene = render_environment_object_scene(
                    rng=scene_rng,
                    canvas_width=int(render_params["canvas_width"]),
                    canvas_height=int(render_params["canvas_height"]),
                    object_count=int(sample.source_object_count),
                    render_scale=int(render_params["render_scale"]),
                    theme_weights={theme: (1.0 if theme == sample.theme_id else 0.0) for theme in ENVIRONMENT_THEME_IDS},
                    style_weights=style_weights(params, _RENDER_DEFAULTS),
                    object_size_min_px=int(render_params["object_size_min_px"]),
                    object_size_max_px=int(render_params["object_size_max_px"]),
                    min_gap_px=int(render_params["min_gap_px"]),
                    max_overlap_fraction=float(render_params["max_overlap_fraction"]),
                    placement_max_attempts=int(render_params["placement_max_attempts"]),
                    skyline_building_min=int(render_params["skyline_building_min"]),
                    skyline_building_max=int(render_params["skyline_building_max"]),
                )
                option_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:patch_options", int(attempt))
                frame_style = sample_style(option_rng, PATCH_FRAME_STYLES)
                label_font_trace = sample_visual_label_font_trace(
                    namespace_prefix=TASK_ID,
                    instance_seed=int(instance_seed),
                    params={**dict(_RENDER_DEFAULTS), **dict(params)},
                    namespace_suffix="patch_option_labels",
                    explicit_key="patch_label_font_family",
                    weights_key="patch_label_font_weights",
                )
                source_panel = source_scene.image.convert("RGB")
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
                )
                artifacts = downscale_patch_option_artifacts(
                    artifacts,
                    max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS,
                )
                break
            except Exception as exc:  # pragma: no cover - random crop/placement feasibility is retry based
                last_error = exc
                sample = None
                source_scene = None
                artifacts = None
        if sample is None or source_scene is None or artifacts is None or frame_style is None or label_font_trace is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        annotation_value = _bbox_map(
            {
                "missing_region": artifacts.missing_region_bbox,
                "selected_option": artifacts.selected_option_bbox,
            }
        )
        serialized_objects, object_bboxes, part_bboxes = serialize_environment_objects(source_scene)
        feature_bboxes = feature_bbox_map(source_scene)
        feature_paths = feature_path_map(source_scene)
        prompt_defaults = required_environment_prompt_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "question_text_missing_patch_label",
                "answer_hint_missing_patch",
                "annotation_hint_missing_patch",
                "json_example_missing_patch",
                "json_example_answer_only_missing_patch",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = render_environment_prompt(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            dynamic_slots={
                "environment_setting": environment_setting_name(str(sample.theme_id)),
                "question_text": str(prompt_defaults["question_text_missing_patch_label"]),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_missing_patch"]),
                "answer_hint": str(prompt_defaults["answer_hint_missing_patch"]),
                "json_example": str(prompt_defaults["json_example_missing_patch"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_missing_patch"]),
            },
            instance_seed=int(instance_seed),
            preferred_mode="answer_and_annotation",
        )
        answer_label = str(artifacts.selected_label)
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "scene_kind": "environment_missing_patch_label",
                "entities": environment_scene_entities(source_scene),
                "relations": {
                    "query_id": SINGLE_QUERY_ID,
                    "patch_mode": PATCH_MODE_PLAIN,
                    "answer_label": answer_label,
                },
            },
            "query_spec": {
                "task_id": self.task_id,
                "scene_id": SCENE_ID,
                "query_id": SINGLE_QUERY_ID,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "patch_mode": PATCH_MODE_PLAIN,
                    "theme": str(sample.theme_id),
                    "theme_id": str(sample.theme_id),
                    "theme_probabilities": dict(sample.theme_probabilities),
                    "source_object_count": int(len(source_scene.placements)),
                    "requested_source_object_count": int(sample.source_object_count),
                    "source_object_count_probabilities": dict(sample.source_object_count_probabilities),
                    "option_count": int(sample.option_count),
                    "option_count_support": [int(value) for value in _option_count_support(params)],
                    "option_count_probabilities": dict(sample.option_count_probabilities),
                    "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                    "answer_label": answer_label,
                    "correct_index": int(sample.correct_index),
                    "patch_size": [int(sample.patch_size[0]), int(sample.patch_size[1])],
                    "patch_size_ratio": dict(sample.patch_size_trace),
                    "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                    **dict(sample.source_profile_trace),
                    "crop_margin_px": int(sample.crop_margin_px),
                    "candidate_crop_count": int(artifacts.candidate_crop_count),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(source_scene.canvas_width), int(source_scene.canvas_height)],
                "source_profile": dict(sample.source_profile_trace),
                "style": {
                    "source_theme_id": str(source_scene.theme_id),
                    "source_style_id": str(source_scene.style_id),
                    "render_scale": int(source_scene.render_scale),
                    "source_layout": dict(source_scene.layout),
                    "patch_frame_style": style_trace(frame_style),
                    "patch_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "source_object_bboxes_px": object_bboxes,
                "source_part_bboxes_px": part_bboxes,
                "source_feature_bboxes_px": feature_bboxes,
                "source_feature_paths_px": feature_paths,
                "missing_region_bbox_px": list(artifacts.missing_region_bbox),
                "option_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.option_bboxes.items()},
                "selected_option_bbox_px": list(artifacts.selected_option_bbox),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "option_source_crop_boxes_px": [
                    [int(coord) for coord in box]
                    for box in artifacts.option_source_crop_boxes
                ],
                "candidate_crop_count": int(artifacts.candidate_crop_count),
                "selected_transform": str(artifacts.selected_transform),
                "option_grid_shape": [int(artifacts.option_grid_shape[0]), int(artifacts.option_grid_shape[1])],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": SINGLE_QUERY_ID,
                "scene_id": SCENE_ID,
                "patch_mode": PATCH_MODE_PLAIN,
                "answer": answer_label,
                "answer_label": answer_label,
                "correct_index": int(sample.correct_index),
                "selected_transform": str(artifacts.selected_transform),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "option_source_crop_boxes_px": [
                    [int(coord) for coord in box]
                    for box in artifacts.option_source_crop_boxes
                ],
                "candidate_crop_count": int(artifacts.candidate_crop_count),
                "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                "source_scene": serialize_environment_scene(source_scene),
                "objects": serialized_objects,
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
            query_id=SINGLE_QUERY_ID,
        )


__all__ = ["IllustrationsEnvironmentMissingPatchLabelTask"]
