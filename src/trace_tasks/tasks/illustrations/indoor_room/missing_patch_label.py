"""Select the patch option that restores an indoor-room illustration."""

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
    PATCH_FRAME_STYLES,
    PATCH_MODE_PLAIN,
    compose_patch_options,
    downscale_patch_option_artifacts,
    sample_style,
    style_trace,
)
from ..shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS
from ..shared.missing_patch_sizing import sample_missing_patch_size
from ..shared.option_rendering import sample_visual_label_font_trace
from .shared.annotations import serialize_indoor_scene
from .shared.output import object_type_map
from .shared.prompts import build_indoor_prompt_artifacts, indoor_setting_name
from .shared.rendering import indoor_scene_entities
from .shared.source_images import IndoorSourceSceneSpec, render_indoor_source_scene, sample_indoor_source_scene_spec


TASK_ID = "task_illustrations__indoor_room__missing_patch_label"
SCENE_ID = "indoor_room"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "missing_patch_label"


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
    canvas_width: int = 1280
    canvas_height: int = 840
    object_size_min_px: int = 52
    object_size_max_px: int = 86
    render_scale: int = 2


@dataclass(frozen=True)
class _SampleSpec:
    source: IndoorSourceSceneSpec
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


def _int_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


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


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample dense room contents and patch-option operands before rendering binds the answer crop."""

    source = sample_indoor_source_scene_spec(
        seed_namespace=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        attempt_index=int(attempt_index),
        generation_defaults=_GEN_DEFAULTS,
        source_object_count_min=_DEFAULTS.source_object_count_min,
        source_object_count_max=_DEFAULTS.source_object_count_max,
        source_width=_DEFAULTS.source_width,
        source_height=_DEFAULTS.source_height,
    )
    option_count, option_count_probabilities = _sample_option_count(params=params, instance_seed=int(instance_seed))
    if params.get("correct_index") is not None:
        correct_index = int(params["correct_index"])
        if correct_index < 0 or correct_index >= int(option_count):
            raise ValueError("correct_index outside option support")
        correct_index_probabilities = {str(correct_index): 1.0}
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}:answer")
        correct_index, correct_index_probabilities = uniform_choice_with_probabilities(
            rng,
            tuple(range(int(option_count))),
            sort_keys=True,
        )
        correct_index = int(correct_index)
        correct_index_probabilities = dict(correct_index_probabilities)

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:spec", int(attempt_index))
    patch_sample = sample_missing_patch_size(
        rng=rng,
        params=params,
        defaults=_GEN_DEFAULTS,
        source_size=source.source_size,
        fallback_width_ratio_min=_DEFAULTS.patch_width_ratio_min,
        fallback_width_ratio_max=_DEFAULTS.patch_width_ratio_max,
        fallback_height_ratio_min=_DEFAULTS.patch_height_ratio_min,
        fallback_height_ratio_max=_DEFAULTS.patch_height_ratio_max,
        fallback_area_ratio_max=_DEFAULTS.patch_area_ratio_max,
    )
    return _SampleSpec(
        source=source,
        option_count=int(option_count),
        correct_index=int(correct_index),
        patch_size=tuple(int(value) for value in patch_sample.patch_size),
        patch_size_trace=dict(patch_sample.trace()),
        crop_margin_px=_int_value(params, _GEN_DEFAULTS, "crop_margin_px", _DEFAULTS.crop_margin_px),
        option_count_probabilities=dict(option_count_probabilities),
        correct_index_probabilities=dict(correct_index_probabilities),
    )


def _bbox_map(value: Mapping[str, Sequence[float]]) -> Dict[str, list[float]]:
    return {str(key): [round(float(coord), 3) for coord in bbox[:4]] for key, bbox in value.items()}


@register_task
class IllustrationsIndoorRoomMissingPatchLabelTask:
    """Select the exact patch option that matches a missing indoor-room region."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "illustrations"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render one room source panel, compose exact patch options, and bind keyed annotation."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        artifacts = None
        frame_style = None
        label_font_trace: Dict[str, Any] | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene = render_indoor_source_scene(
                    render_namespace="missing_patch_label",
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    source=sample.source,
                    params=params,
                    render_defaults=_RENDER_DEFAULTS,
                    fallback_defaults=_DEFAULTS,
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
                source_panel = scene.image.convert("RGB")
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
            except Exception as exc:  # pragma: no cover
                last_error = exc
                sample = None
                scene = None
                artifacts = None
                frame_style = None
                label_font_trace = None
        if sample is None or scene is None or artifacts is None or frame_style is None or label_font_trace is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        serialized_objects, object_bboxes, part_bboxes = serialize_indoor_scene(scene)
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
        prompt_artifacts = build_indoor_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots={
                "room_setting": indoor_setting_name(str(scene.theme_id)),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "answer_hint": str(prompt_defaults["answer_hint_missing_patch"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_missing_patch"]),
                "json_example": str(prompt_defaults["json_example_missing_patch"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_missing_patch"]),
            },
            instance_seed=int(instance_seed),
        )
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "scene_kind": "indoor_room_missing_patch_label",
                "entities": indoor_scene_entities(scene),
                "relations": {
                    "query_id": QUERY_ID,
                    "patch_mode": PATCH_MODE_PLAIN,
                    "answer_label": answer_label,
                },
            },
            "query_spec": {
                "task_id": self.task_id,
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "patch_mode": PATCH_MODE_PLAIN,
                    "theme": str(sample.source.theme_id),
                    "theme_id": str(sample.source.theme_id),
                    "source_object_count": int(len(scene.placements)),
                    "requested_source_object_count": int(sample.source.source_object_count),
                    "source_object_count_probabilities": dict(sample.source.source_object_count_probabilities),
                    "option_count": int(sample.option_count),
                    "option_count_support": [int(value) for value in _option_count_support(params)],
                    "option_count_probabilities": dict(sample.option_count_probabilities),
                    "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                    "answer_label": answer_label,
                    "correct_index": int(sample.correct_index),
                    "patch_size": [int(sample.patch_size[0]), int(sample.patch_size[1])],
                    "patch_size_ratio": dict(sample.patch_size_trace),
                    "source_size": [int(sample.source.source_size[0]), int(sample.source.source_size[1])],
                    **dict(sample.source.source_profile_trace),
                    "crop_margin_px": int(sample.crop_margin_px),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                    "theme_probabilities": dict(sample.source.theme_probabilities),
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
                "source_profile": dict(sample.source.source_profile_trace),
                "style": {
                    "source_theme_id": str(scene.theme_id),
                    "source_style_id": str(scene.style_id),
                    "render_scale": int(scene.render_scale),
                    "patch_frame_style": style_trace(frame_style),
                    "patch_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "source_object_bboxes_px": object_bboxes,
                "source_part_bboxes_px": part_bboxes,
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
                "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                "object_types": object_type_map(serialized_objects),
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


__all__ = ["IllustrationsIndoorRoomMissingPatchLabelTask"]
