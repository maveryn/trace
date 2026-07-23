"""Select the patch option that restores a construction-site illustration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import support_probability_map, uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants
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
from .shared.output import construction_scene_entities, serialize_construction_scene
from .shared.rendering import render_construction_site_scene
from .shared.state import (
    ConstructionEquipmentSpec,
    ConstructionMaterialSpec,
    ConstructionWorkerSpec,
)
from .shared.sampling import (
    bounds,
    color_support,
    equipment_support,
    material_support,
    render_params,
    setting_weights,
    spawned_task_rng,
    style_weights,
    tool_support,
    uniform_string_probability_map,
)


TASK_ID = "task_illustrations__construction_site__missing_patch_label"
DOMAIN = "illustrations"
SCENE_ID = "construction_site"
PLAIN_QUERY_ID = "plain_patch_label"
QUERY_IDS: Tuple[str, ...] = (PLAIN_QUERY_ID,)
_QUERY_TO_PATCH_MODE: Dict[str, str] = {
    PLAIN_QUERY_ID: PATCH_MODE_PLAIN,
}


@dataclass(frozen=True)
class _Defaults:
    source_worker_count_min: int = 10
    source_worker_count_max: int = 16
    source_material_count_min: int = 6
    source_material_count_max: int = 10
    source_equipment_count_min: int = 4
    source_equipment_count_max: int = 7
    option_count: int = 6
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
    source_worker_count: int
    source_material_count: int
    source_equipment_count: int
    option_count: int
    correct_index: int
    patch_size: Tuple[int, int]
    patch_size_trace: Dict[str, Any]
    crop_margin_px: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    worker_specs: Tuple[ConstructionWorkerSpec, ...]
    material_specs: Tuple[ConstructionMaterialSpec, ...]
    equipment_specs: Tuple[ConstructionEquipmentSpec, ...]
    query_probabilities: Dict[str, float]
    option_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _int_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve one integer value from params, defaults, or fallback."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _sample_query(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[str, Dict[str, float]]:
    explicit_query = params.get("query_id")
    if explicit_query is not None:
        selected = str(explicit_query)
        if selected not in set(QUERY_IDS):
            raise ValueError(f"query_id must be one of {QUERY_IDS}")
        return selected, uniform_string_probability_map(QUERY_IDS, selected=selected)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:query")
    selected, probabilities = uniform_choice_with_probabilities(rng, QUERY_IDS, sort_keys=False)
    return str(selected), dict(probabilities)


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


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Choose source-scene density and a unique visual patch answer."""

    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    query_id, query_probabilities = _sample_query(params=params, instance_seed=int(instance_seed))
    colors = color_support(params, _GEN_DEFAULTS)
    tools = tool_support(params, _GEN_DEFAULTS)
    materials = material_support(params, _GEN_DEFAULTS)
    equipment_values = equipment_support(params, _GEN_DEFAULTS)

    worker_count = _sample_range(
        rng=rng,
        params=params,
        defaults=_GEN_DEFAULTS,
        low_key="source_worker_count_min",
        high_key="source_worker_count_max",
        fallback_low=_DEFAULTS.source_worker_count_min,
        fallback_high=_DEFAULTS.source_worker_count_max,
    )
    material_count = _sample_range(
        rng=rng,
        params=params,
        defaults=_GEN_DEFAULTS,
        low_key="source_material_count_min",
        high_key="source_material_count_max",
        fallback_low=_DEFAULTS.source_material_count_min,
        fallback_high=_DEFAULTS.source_material_count_max,
    )
    equipment_count = _sample_range(
        rng=rng,
        params=params,
        defaults=_GEN_DEFAULTS,
        low_key="source_equipment_count_min",
        high_key="source_equipment_count_max",
        fallback_low=_DEFAULTS.source_equipment_count_min,
        fallback_high=_DEFAULTS.source_equipment_count_max,
    )
    option_count, option_count_probabilities = _sample_option_count(params=params, instance_seed=int(instance_seed))
    if option_count < 2 or option_count > len(DEFAULT_OPTION_LABELS):
        raise ValueError("option_count is outside option label support")
    if params.get("correct_index") is not None:
        correct_index = int(params["correct_index"])
        if correct_index < 0 or correct_index >= int(option_count):
            raise ValueError("correct_index outside option support")
        correct_index_probabilities = {str(correct_index): 1.0}
    else:
        answer_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:answer")
        correct_index, correct_index_probabilities = uniform_choice_with_probabilities(
            answer_rng,
            tuple(range(int(option_count))),
            sort_keys=True,
        )
        correct_index = int(correct_index)
        correct_index_probabilities = dict(correct_index_probabilities)

    source_profile = resolve_reconstruction_source_profile(
        params=params,
        defaults=_RENDER_DEFAULTS,
        fallback_source_width=_DEFAULTS.source_width,
        fallback_source_height=_DEFAULTS.source_height,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
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
    crop_margin = _int_value(params, _GEN_DEFAULTS, "crop_margin_px", _DEFAULTS.crop_margin_px)

    worker_specs = tuple(
        ConstructionWorkerSpec(
            hard_hat_color=str(rng.choice(colors)),
            vest_color=str(rng.choice(colors)),
            tool_type=str(rng.choice(tools)) if rng.random() < 0.48 else None,
            role="source",
        )
        for _ in range(int(worker_count))
    )
    material_specs = tuple(
        ConstructionMaterialSpec(material_type=str(rng.choice(materials)), role="source")
        for _ in range(int(material_count))
    )
    equipment_specs = tuple(
        ConstructionEquipmentSpec(equipment_type=str(rng.choice(equipment_values)), role="source")
        for _ in range(int(equipment_count))
    )
    return _SampleSpec(
        query_id=str(query_id),
        patch_mode=str(_QUERY_TO_PATCH_MODE[str(query_id)]),
        source_worker_count=int(worker_count),
        source_material_count=int(material_count),
        source_equipment_count=int(equipment_count),
        option_count=int(option_count),
        correct_index=int(correct_index),
        patch_size=tuple(int(value) for value in patch_sample.patch_size),
        patch_size_trace=dict(patch_sample.trace()),
        crop_margin_px=int(crop_margin),
        source_size=tuple(int(value) for value in source_profile.size),
        source_profile_trace=dict(source_profile.trace()),
        worker_specs=worker_specs,
        material_specs=material_specs,
        equipment_specs=equipment_specs,
        query_probabilities=dict(query_probabilities),
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
class IllustrationsConstructionSiteMissingPatchLabelTask:
    """Select the patch option that matches a missing construction-site region."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render a construction source panel and keyed patch-option annotation."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        source_scene = None
        artifacts = None
        frame_style = None
        label_font_trace: Dict[str, Any] | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                source_size = sample.source_size
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:source_scene", int(attempt))
                render_overrides = {
                    **dict(params),
                    "canvas_width": int(source_size[0]),
                    "canvas_height": int(source_size[1]),
                }
                rp = render_params(
                    render_overrides,
                    _RENDER_DEFAULTS,
                    fallback_width=_DEFAULTS.canvas_width,
                    fallback_height=_DEFAULTS.canvas_height,
                    fallback_scale=_DEFAULTS.render_scale,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}:canvas_profile",
                )
                source_scene = render_construction_site_scene(
                    rng=scene_rng,
                    worker_specs=sample.worker_specs,
                    material_specs=sample.material_specs,
                    equipment_specs=sample.equipment_specs,
                    canvas_width=int(rp["canvas_width"]),
                    canvas_height=int(rp["canvas_height"]),
                    render_scale=int(rp["render_scale"]),
                    setting_weights=setting_weights(render_overrides, _RENDER_DEFAULTS),
                    style_weights=style_weights(render_overrides, _RENDER_DEFAULTS),
                    instance_seed=int(instance_seed),
                    font_params={**dict(_RENDER_DEFAULTS), **dict(render_overrides)},
                    show_zone_labels=False,
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
            except Exception as exc:  # pragma: no cover
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
        serialized_scene, source_bbox_map = serialize_construction_scene(source_scene)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description_missing_patch",
                "question_text_plain_patch_label",
                "answer_hint_missing_patch",
                "annotation_hint_missing_patch",
                "json_example_missing_patch",
                "json_example_answer_only_missing_patch",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            dynamic_slots={
                "object_description": str(prompt_defaults["object_description_missing_patch"]),
                "question_text": str(prompt_defaults[f"question_text_{sample.query_id}"]),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_missing_patch"]),
                "answer_hint": str(prompt_defaults["answer_hint_missing_patch"]),
                "json_example": str(prompt_defaults["json_example_missing_patch"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_missing_patch"]),
            },
            instance_seed=int(instance_seed),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            preferred_mode="answer_and_annotation",
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
        answer_label = str(artifacts.selected_label)
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "scene_kind": "construction_site_missing_patch_label",
                "entities": construction_scene_entities(source_scene),
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
                    "patch_mode": str(sample.patch_mode),
                    "source_worker_count": int(sample.source_worker_count),
                    "source_material_count": int(sample.source_material_count),
                    "source_equipment_count": int(sample.source_equipment_count),
                    "option_count": int(sample.option_count),
                    "option_count_support": [int(value) for value in _option_count_support(params, _GEN_DEFAULTS)],
                    "option_count_probabilities": dict(sample.option_count_probabilities),
                    "option_labels": list(DEFAULT_OPTION_LABELS[: int(sample.option_count)]),
                    "answer_label": answer_label,
                    "correct_index": int(sample.correct_index),
                    "patch_size": [int(sample.patch_size[0]), int(sample.patch_size[1])],
                    "patch_size_ratio": dict(sample.patch_size_trace),
                    "source_size": [int(source_size[0]), int(source_size[1])],
                    **dict(sample.source_profile_trace),
                    "crop_margin_px": int(sample.crop_margin_px),
                    "query_id_probabilities": dict(sample.query_probabilities),
                    "query_probabilities": dict(sample.query_probabilities),
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
    "IllustrationsConstructionSiteMissingPatchLabelTask",
    "PLAIN_QUERY_ID",
    "QUERY_IDS",
    "TASK_ID",
]
