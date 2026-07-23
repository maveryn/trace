"""Select the patch option that restores a park/playground illustration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import support_probability_map, uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.fixed_query import select_task_query_id
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
from .shared.annotations import park_scene_entities, serialize_park_scene
from .shared.prompts import build_park_prompt_artifacts
from .shared.rendering import PARK_EQUIPMENT_TYPES, PARK_PERSON_ACTIVITIES, ParkEquipmentSpec, ParkPersonSpec, render_park_playground_scene
from .shared.sampling import (
    activity_support,
    bounds,
    equipment_support,
    render_params,
    sample_count,
    setting_weights,
    spawned_task_rng,
    style_weights,
)


TASK_ID = "task_illustrations__park_playground__missing_patch_label"
DOMAIN = "illustrations"
SCENE_ID = "park_playground"
PLAIN_QUERY_ID = "plain_patch_label"
QUERY_IDS: Tuple[str, ...] = (PLAIN_QUERY_ID,)
_QUERY_TO_PATCH_MODE: Dict[str, str] = {
    PLAIN_QUERY_ID: PATCH_MODE_PLAIN,
}


@dataclass(frozen=True)
class _Defaults:
    source_person_count_min: int = 8
    source_person_count_max: int = 13
    source_equipment_count_min: int = 4
    source_equipment_count_max: int = 7
    option_count_support: Tuple[int, ...] = (4, 6)
    patch_width_ratio_min: float = 0.15
    patch_width_ratio_max: float = 0.30
    patch_height_ratio_min: float = 0.15
    patch_height_ratio_max: float = 0.26
    patch_area_ratio_max: float = 0.065
    crop_margin_px: int = 36
    source_width: int = 820
    source_height: int = 560
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2


@dataclass(frozen=True)
class _SampleSpec:
    query_id: str
    patch_mode: str
    source_person_count: int
    source_equipment_count: int
    option_count: int
    correct_index: int
    patch_size: Tuple[int, int]
    patch_size_trace: Dict[str, Any]
    crop_margin_px: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    person_specs: Tuple[ParkPersonSpec, ...]
    equipment_specs: Tuple[ParkEquipmentSpec, ...]
    query_probabilities: Dict[str, float]
    source_person_count_probabilities: Dict[str, float]
    source_equipment_count_probabilities: Dict[str, float]
    option_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _int_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _sample_query(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=QUERY_IDS,
        default_query_id=PLAIN_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    return str(query_id), dict(query_probabilities), dict(task_params)


def _sample_range(
    *,
    rng: Any,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
) -> int:
    low, high = bounds(params, defaults, low_key, high_key, int(fallback_low), int(fallback_high))
    return int(rng.randint(int(low), int(high)))


def _option_count_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("option_count_support", group_default(defaults, "option_count_support", _DEFAULTS.option_count_support))
    raw_values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    values = tuple(dict.fromkeys(int(value) for value in raw_values if int(value) in set(_DEFAULTS.option_count_support)))
    if not values:
        raise ValueError("option_count_support must include 4 or 6")
    return values


def _sample_option_count(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[int, Dict[str, float]]:
    explicit = params.get("option_count")
    support = _option_count_support(params, _GEN_DEFAULTS)
    if explicit is not None:
        option_count = int(explicit)
        if option_count not in set(support):
            raise ValueError(f"option_count must be one of {support}")
        return int(option_count), support_probability_map(support, selected=int(option_count), sort_keys=True)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:option_count")
    selected, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(selected), dict(probabilities)


def _sample_correct_index(*, params: Mapping[str, Any], instance_seed: int, option_count: int) -> Tuple[int, Dict[str, float]]:
    if params.get("correct_index") is not None:
        value = int(params["correct_index"])
        if value < 0 or value >= int(option_count):
            raise ValueError("correct_index outside option support")
        return int(value), {str(value): 1.0}
    namespace = f"{TASK_ID}:answer"
    if params.get("_sample_cursor") is not None:
        namespace = f"{namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), namespace)
    value, probabilities = uniform_choice_with_probabilities(
        rng,
        tuple(range(int(option_count))),
        sort_keys=True,
    )
    return int(value), dict(probabilities)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Choose a dense source park scene and one unique patch option answer."""

    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    query_id, query_probabilities, task_params = _sample_query(params=params, instance_seed=int(instance_seed))
    activities = activity_support(task_params, _GEN_DEFAULTS, fallback=PARK_PERSON_ACTIVITIES)
    equipment_values = equipment_support(task_params, _GEN_DEFAULTS, fallback=PARK_EQUIPMENT_TYPES)
    person_min, person_max = bounds(
        task_params,
        _GEN_DEFAULTS,
        "source_person_count_min",
        "source_person_count_max",
        _DEFAULTS.source_person_count_min,
        _DEFAULTS.source_person_count_max,
    )
    source_person_count, person_count_probabilities = sample_count(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_person_count",
        low=int(person_min),
        high=int(person_max),
        explicit_key="source_person_count",
    )
    equipment_min, equipment_max = bounds(
        task_params,
        _GEN_DEFAULTS,
        "source_equipment_count_min",
        "source_equipment_count_max",
        _DEFAULTS.source_equipment_count_min,
        _DEFAULTS.source_equipment_count_max,
    )
    source_equipment_count, equipment_count_probabilities = sample_count(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_equipment_count",
        low=int(equipment_min),
        high=int(equipment_max),
        explicit_key="source_equipment_count",
    )
    option_count, option_count_probabilities = _sample_option_count(params=task_params, instance_seed=int(instance_seed))
    correct_index, correct_index_probabilities = _sample_correct_index(
        params=task_params,
        instance_seed=int(instance_seed),
        option_count=int(option_count),
    )
    source_profile = resolve_reconstruction_source_profile(
        params=task_params,
        defaults=_GEN_DEFAULTS,
        fallback_source_width=_DEFAULTS.source_width,
        fallback_source_height=_DEFAULTS.source_height,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
    patch_sample = sample_missing_patch_size(
        rng=rng,
        params=task_params,
        defaults=_GEN_DEFAULTS,
        source_size=source_profile.size,
        fallback_width_ratio_min=_DEFAULTS.patch_width_ratio_min,
        fallback_width_ratio_max=_DEFAULTS.patch_width_ratio_max,
        fallback_height_ratio_min=_DEFAULTS.patch_height_ratio_min,
        fallback_height_ratio_max=_DEFAULTS.patch_height_ratio_max,
        fallback_area_ratio_max=_DEFAULTS.patch_area_ratio_max,
    )
    person_specs = tuple(
        ParkPersonSpec(activity=str(rng.choice(activities)), role="source")
        for _ in range(int(source_person_count))
    )
    equipment_specs = tuple(
        ParkEquipmentSpec(equipment_type=str(rng.choice(equipment_values)), role="source")
        for _ in range(int(source_equipment_count))
    )
    return _SampleSpec(
        query_id=str(query_id),
        patch_mode=str(_QUERY_TO_PATCH_MODE[str(query_id)]),
        source_person_count=int(source_person_count),
        source_equipment_count=int(source_equipment_count),
        option_count=int(option_count),
        correct_index=int(correct_index),
        patch_size=tuple(int(value) for value in patch_sample.patch_size),
        patch_size_trace=dict(patch_sample.trace()),
        crop_margin_px=_int_value(task_params, _GEN_DEFAULTS, "crop_margin_px", _DEFAULTS.crop_margin_px),
        source_size=tuple(int(value) for value in source_profile.size),
        source_profile_trace=dict(source_profile.trace()),
        person_specs=person_specs,
        equipment_specs=equipment_specs,
        query_probabilities=dict(query_probabilities),
        source_person_count_probabilities=dict(person_count_probabilities),
        source_equipment_count_probabilities=dict(equipment_count_probabilities),
        option_count_probabilities=dict(option_count_probabilities),
        correct_index_probabilities=dict(correct_index_probabilities),
    )


def _bbox_map(value: Mapping[str, Sequence[float]]) -> Dict[str, list[float]]:
    return {
        str(key): [round(float(coord), 3) for coord in bbox[:4]]
        for key, bbox in value.items()
    }


@register_task
class IllustrationsParkPlaygroundMissingPatchLabelTask:
    """Select the patch option that matches a missing park/playground region."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render a park source panel and keyed missing-region/option annotation."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        source_scene = None
        artifacts = None
        frame_style = None
        label_font_trace: Dict[str, Any] | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:source_scene", int(attempt))
                rp = render_params(
                    {
                        **dict(params),
                        "canvas_width": int(sample.source_size[0]),
                        "canvas_height": int(sample.source_size[1]),
                    },
                    _RENDER_DEFAULTS,
                    fallback_width=_DEFAULTS.canvas_width,
                    fallback_height=_DEFAULTS.canvas_height,
                    fallback_scale=_DEFAULTS.render_scale,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}:source_profile",
                )
                source_scene = render_park_playground_scene(
                    rng=scene_rng,
                    person_specs=sample.person_specs,
                    equipment_specs=sample.equipment_specs,
                    canvas_width=int(rp["canvas_width"]),
                    canvas_height=int(rp["canvas_height"]),
                    render_scale=int(rp["render_scale"]),
                    setting_weights=setting_weights(params, _RENDER_DEFAULTS),
                    style_weights=style_weights(params, _RENDER_DEFAULTS),
                )
                option_rng = spawned_task_rng(int(instance_seed), f"{TASK_ID}:patch_options", int(attempt))
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
                    patch_mode=str(sample.patch_mode),
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
            except Exception as exc:  # pragma: no cover - retry surface is seed/layout dependent.
                last_error = exc
                sample = None
                source_scene = None
                artifacts = None
                frame_style = None
                label_font_trace = None
        if sample is None or source_scene is None or artifacts is None or frame_style is None or label_font_trace is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        annotation_value = _bbox_map(
            {
                "missing_region": artifacts.missing_region_bbox,
                "selected_option": artifacts.selected_option_bbox,
            }
        )
        serialized_scene, source_bbox_map = serialize_park_scene(source_scene)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_missing_patch",
                "annotation_hint_missing_patch",
                "json_example_missing_patch",
                "json_example_answer_only_missing_patch",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = build_park_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=str(sample.query_id),
            slots={
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_missing_patch"]),
                "answer_hint": str(prompt_defaults["answer_hint_missing_patch"]),
                "json_example": str(prompt_defaults["json_example_missing_patch"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_missing_patch"]),
            },
            instance_seed=int(instance_seed),
        )
        answer_label = str(artifacts.selected_label)
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "scene_kind": "park_playground_missing_patch_label",
                "entities": park_scene_entities(source_scene),
                "relations": {
                    "query_id": str(sample.query_id),
                    "patch_mode": str(sample.patch_mode),
                    "answer_label": answer_label,
                },
            },
            "query_spec": {
                "task_id": self.task_id,
                "scene_id": SCENE_ID,
                "query_id": str(sample.query_id),
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(sample.query_id),
                    "patch_mode": str(sample.patch_mode),
                    "source_person_count": int(sample.source_person_count),
                    "source_person_count_probabilities": dict(sample.source_person_count_probabilities),
                    "source_equipment_count": int(sample.source_equipment_count),
                    "source_equipment_count_probabilities": dict(sample.source_equipment_count_probabilities),
                    "option_count": int(sample.option_count),
                    "option_count_support": [int(value) for value in _option_count_support(params, _GEN_DEFAULTS)],
                    "option_count_probabilities": dict(sample.option_count_probabilities),
                    "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                    "answer_label": answer_label,
                    "correct_index": int(sample.correct_index),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                    "patch_size": [int(sample.patch_size[0]), int(sample.patch_size[1])],
                    "patch_size_ratio": dict(sample.patch_size_trace),
                    "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                    **dict(sample.source_profile_trace),
                    "crop_margin_px": int(sample.crop_margin_px),
                    "query_id_probabilities": dict(sample.query_probabilities),
                    "query_probabilities": dict(sample.query_probabilities),
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(source_scene.canvas_width), int(source_scene.canvas_height)],
                "source_profile": dict(sample.source_profile_trace),
                "style": {
                    "source_setting_id": str(source_scene.setting_id),
                    "source_style_id": str(source_scene.style_id),
                    "render_scale": int(source_scene.render_scale),
                    "source_layout": dict(source_scene.layout),
                    "patch_frame_style": style_trace(frame_style),
                    "patch_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "source_bboxes_px": source_bbox_map,
                "missing_region_bbox_px": list(artifacts.missing_region_bbox),
                "option_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.option_bboxes.items()},
                "selected_option_bbox_px": list(artifacts.selected_option_bbox),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "selected_transform": str(artifacts.selected_transform),
                "option_grid_shape": [int(artifacts.option_grid_shape[0]), int(artifacts.option_grid_shape[1])],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": str(sample.query_id),
                "scene_id": SCENE_ID,
                "patch_mode": str(sample.patch_mode),
                "answer": answer_label,
                "answer_label": answer_label,
                "correct_index": int(sample.correct_index),
                "selected_transform": str(artifacts.selected_transform),
                "source_crop_box_px": [int(value) for value in artifacts.source_crop_box],
                "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                "source_scene": serialized_scene[0],
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
            query_id=str(sample.query_id),
        )


__all__ = [
    "IllustrationsParkPlaygroundMissingPatchLabelTask",
    "PLAIN_QUERY_ID",
    "QUERY_IDS",
    "TASK_ID",
    "_sample_spec",
]
