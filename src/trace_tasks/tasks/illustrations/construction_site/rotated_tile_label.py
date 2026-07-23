"""Identify the rotated tile in a construction-site illustration grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.sampling import support_probability_map, uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants
from ..shared.cutouts import (
    DEFAULT_OPTION_LABELS,
    FRAMELESS_ILLUSTRATION_ROTATED_GRID_STYLE,
    compose_rotated_tile_grid,
    downscale_rotated_tile_artifacts,
    piece_crops,
    style_trace,
    tile_is_usable,
)
from ..shared.canvas_profiles import (
    MAX_RECONSTRUCTION_OUTPUT_PIXELS,
    reconstruction_grid_for_profile,
    resolve_reconstruction_source_profile,
)
from ..shared.option_rendering import sample_visual_label_font_trace
from .shared.output import construction_scene_entities, serialize_construction_scene
from .shared.rendering import render_construction_site_scene
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
)
from .shared.state import ConstructionEquipmentSpec, ConstructionMaterialSpec, ConstructionWorkerSpec


TASK_ID = "task_illustrations__construction_site__rotated_tile_label"
DOMAIN = "illustrations"
SCENE_ID = "construction_site"
QUERY_ID = SINGLE_QUERY_ID
GRID_ROWS = 2
GRID_COLS = 3
TILE_LABELS: Tuple[str, ...] = DEFAULT_OPTION_LABELS[: GRID_ROWS * GRID_COLS]
ROTATION_SUPPORT: Tuple[int, ...] = (90, 270)


@dataclass(frozen=True)
class _Defaults:
    source_worker_count_min: int = 10
    source_worker_count_max: int = 16
    source_material_count_min: int = 6
    source_material_count_max: int = 10
    source_equipment_count_min: int = 4
    source_equipment_count_max: int = 7
    source_width: int = 960
    source_height: int = 640
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2
    min_tile_detail_score: float = 160.0
    min_rotation_delta: float = 7.0


@dataclass(frozen=True)
class _SampleSpec:
    source_worker_count: int
    source_material_count: int
    source_equipment_count: int
    source_size: Tuple[int, int]
    rotation_degrees: int
    worker_specs: Tuple[ConstructionWorkerSpec, ...]
    material_specs: Tuple[ConstructionMaterialSpec, ...]
    equipment_specs: Tuple[ConstructionEquipmentSpec, ...]
    worker_count_probabilities: Dict[str, float]
    material_count_probabilities: Dict[str, float]
    equipment_count_probabilities: Dict[str, float]
    rotation_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _int_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _float_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(defaults, str(key), float(fallback))))


def _sample_count(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
    explicit_key: str,
) -> Tuple[int, Dict[str, float]]:
    low, high = bounds(params, _GEN_DEFAULTS, low_key, high_key, int(fallback_low), int(fallback_high))
    support = tuple(range(int(low), int(high) + 1))
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"{explicit_key} must be in {support}")
        return int(value), support_probability_map(support, selected=int(value), sort_keys=True)
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(selected), dict(probabilities)


def _rotation_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("rotation_degrees_support", group_default(_GEN_DEFAULTS, "rotation_degrees_support", ROTATION_SUPPORT))
    values = (raw,) if isinstance(raw, int) else tuple(raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else ())
    support = tuple(dict.fromkeys(int(value) for value in values if int(value) in set(ROTATION_SUPPORT)))
    if not support:
        raise ValueError("rotation_degrees_support must include 90 or 270")
    return support


def _sample_rotation(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[int, Dict[str, float]]:
    support = _rotation_support(params)
    explicit = params.get("rotation_degrees")
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"rotation_degrees must be one of {support}")
        return int(value), support_probability_map(support, selected=int(value), sort_keys=True)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:rotation_degrees")
    selected, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(selected), dict(probabilities)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample dense source-scene content and a rotation angle without choosing the answer tile."""

    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    colors = color_support(params, _GEN_DEFAULTS)
    tools = tool_support(params, _GEN_DEFAULTS)
    materials = material_support(params, _GEN_DEFAULTS)
    equipment_values = equipment_support(params, _GEN_DEFAULTS)
    worker_count, worker_probs = _sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_worker_count",
        low_key="source_worker_count_min",
        high_key="source_worker_count_max",
        fallback_low=_DEFAULTS.source_worker_count_min,
        fallback_high=_DEFAULTS.source_worker_count_max,
        explicit_key="source_worker_count",
    )
    material_count, material_probs = _sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_material_count",
        low_key="source_material_count_min",
        high_key="source_material_count_max",
        fallback_low=_DEFAULTS.source_material_count_min,
        fallback_high=_DEFAULTS.source_material_count_max,
        explicit_key="source_material_count",
    )
    equipment_count, equipment_probs = _sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_equipment_count",
        low_key="source_equipment_count_min",
        high_key="source_equipment_count_max",
        fallback_low=_DEFAULTS.source_equipment_count_min,
        fallback_high=_DEFAULTS.source_equipment_count_max,
        explicit_key="source_equipment_count",
    )
    rotation_degrees, rotation_probs = _sample_rotation(params=params, instance_seed=int(instance_seed))
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
        source_worker_count=int(worker_count),
        source_material_count=int(material_count),
        source_equipment_count=int(equipment_count),
        source_size=(
            _int_value(params, _GEN_DEFAULTS, "source_width", _DEFAULTS.source_width),
            _int_value(params, _GEN_DEFAULTS, "source_height", _DEFAULTS.source_height),
        ),
        rotation_degrees=int(rotation_degrees),
        worker_specs=worker_specs,
        material_specs=material_specs,
        equipment_specs=equipment_specs,
        worker_count_probabilities=dict(worker_probs),
        material_count_probabilities=dict(material_probs),
        equipment_count_probabilities=dict(equipment_probs),
        rotation_probabilities=dict(rotation_probs),
    )


def _usable_tile_indices(
    *,
    source_image: Image.Image,
    rows: int,
    cols: int,
    rotation_degrees: int,
    min_detail_score: float,
    min_rotation_delta: float,
) -> Tuple[int, ...]:
    pieces = piece_crops(source_image.convert("RGB"), rows=int(rows), cols=int(cols))
    usable: list[int] = []
    for index, (piece, _source_box) in enumerate(pieces):
        rotated = piece.rotate(-int(rotation_degrees), expand=False, resample=Image.Resampling.BICUBIC)
        if tile_is_usable(piece, rotated, min_detail_score=float(min_detail_score), min_rotation_delta=float(min_rotation_delta)):
            usable.append(int(index))
    return tuple(usable)


def _select_correct_index(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    attempt_index: int,
    usable_indices: Sequence[int],
    tile_labels: Sequence[str],
) -> Tuple[int, Dict[str, float]]:
    usable = tuple(int(index) for index in usable_indices)
    if not usable:
        raise ValueError("no visually usable construction-site tile for rotation")
    explicit = params.get("correct_index")
    if explicit is not None:
        value = int(explicit)
        if value < 0 or value >= len(tile_labels):
            raise ValueError("correct_index outside tile label support")
        if value not in set(usable):
            raise ValueError("explicit correct_index is not visually usable for rotation")
        return int(value), {str(value): 1.0}
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:answer", int(attempt_index))
    selected, probabilities = uniform_choice_with_probabilities(rng, usable, sort_keys=True)
    return int(selected), dict(probabilities)


@register_task
class IllustrationsConstructionSiteRotatedTileLabelTask:
    """Select the lettered tile that has been rotated inside a construction-site grid."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = DOMAIN
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render one source construction scene, rotate one usable tile, and bind tile annotation."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        source_scene = None
        artifacts = None
        grid_style = None
        label_font_trace: Dict[str, Any] | None = None
        correct_index = None
        correct_index_probabilities: Dict[str, float] | None = None
        usable_indices: Tuple[int, ...] = tuple()
        source_profile = resolve_reconstruction_source_profile(
            params=params,
            defaults=_RENDER_DEFAULTS,
            fallback_source_width=_DEFAULTS.source_width,
            fallback_source_height=_DEFAULTS.source_height,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:source_profile",
        )
        source_size = source_profile.size
        grid_rows, grid_cols = reconstruction_grid_for_profile(source_profile)
        tile_labels = DEFAULT_OPTION_LABELS[: int(grid_rows) * int(grid_cols)]
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
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
                source_panel = source_scene.image.convert("RGB")
                usable_indices = _usable_tile_indices(
                    source_image=source_panel,
                    rows=int(grid_rows),
                    cols=int(grid_cols),
                    rotation_degrees=int(sample.rotation_degrees),
                    min_detail_score=_float_value(params, _GEN_DEFAULTS, "min_tile_detail_score", _DEFAULTS.min_tile_detail_score),
                    min_rotation_delta=_float_value(params, _GEN_DEFAULTS, "min_rotation_delta", _DEFAULTS.min_rotation_delta),
                )
                correct_index, correct_index_probabilities = _select_correct_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    usable_indices=usable_indices,
                    tile_labels=tile_labels,
                )
                grid_style = {"style_id": "frameless_illustration", **dict(FRAMELESS_ILLUSTRATION_ROTATED_GRID_STYLE)}
                label_font_trace = sample_visual_label_font_trace(
                    namespace_prefix=TASK_ID,
                    instance_seed=int(instance_seed),
                    params={**dict(_RENDER_DEFAULTS), **dict(params)},
                    namespace_suffix="tile_labels",
                    explicit_key="tile_label_font_family",
                    weights_key="tile_label_font_weights",
                )
                artifacts = compose_rotated_tile_grid(
                    source_image=source_panel,
                    correct_index=int(correct_index),
                    rotation_degrees=int(sample.rotation_degrees),
                    grid_style=grid_style,
                    label_font_family=str(label_font_trace["font_family"]),
                    rows=int(grid_rows),
                    cols=int(grid_cols),
                    labels=tile_labels,
                    render_margin=0,
                )
                artifacts = downscale_rotated_tile_artifacts(
                    artifacts,
                    max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS,
                )
                break
            except Exception as exc:  # pragma: no cover
                last_error = exc
                sample = None
                source_scene = None
                artifacts = None
                grid_style = None
                label_font_trace = None
                correct_index = None
                correct_index_probabilities = None
                usable_indices = tuple()
        if (
            sample is None
            or source_scene is None
            or artifacts is None
            or grid_style is None
            or label_font_trace is None
            or correct_index is None
            or correct_index_probabilities is None
        ):
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        serialized_scene, source_bbox_map = serialize_construction_scene(source_scene)
        answer_label = str(artifacts.selected_label)
        annotation_artifacts = bbox_annotation_artifacts(artifacts.selected_bbox)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description_rotated_tile",
                "question_text_rotated_tile_label",
                "answer_hint_rotated_tile",
                "annotation_hint_rotated_tile",
                "json_example_rotated_tile",
                "json_example_answer_only_rotated_tile",
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
                "object_description": str(prompt_defaults["object_description_rotated_tile"]),
                "question_text": str(prompt_defaults["question_text_rotated_tile_label"]),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_rotated_tile"]),
                "answer_hint": str(prompt_defaults["answer_hint_rotated_tile"]),
                "json_example": str(prompt_defaults["json_example_rotated_tile"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_rotated_tile"]),
            },
            instance_seed=int(instance_seed),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            preferred_mode="answer_and_annotation",
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "scene_kind": "construction_site_rotated_tile_label",
                "entities": construction_scene_entities(source_scene),
                "relations": {
                    "query_id": QUERY_ID,
                    "rotated_tile_label": answer_label,
                    "rotated_tile_index": int(correct_index),
                    "rotation_degrees": int(sample.rotation_degrees),
                },
            },
            "query_spec": {
                "task_id": self.task_id,
                "scene_id": SCENE_ID,
                "query_id": QUERY_ID,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "source_worker_count": int(sample.source_worker_count),
                    "source_material_count": int(sample.source_material_count),
                    "source_equipment_count": int(sample.source_equipment_count),
                    "source_worker_count_probabilities": dict(sample.worker_count_probabilities),
                    "source_material_count_probabilities": dict(sample.material_count_probabilities),
                    "source_equipment_count_probabilities": dict(sample.equipment_count_probabilities),
                    "rotation_degrees": int(sample.rotation_degrees),
                    "rotation_degrees_support": [int(value) for value in _rotation_support(params)],
                    "rotation_degrees_probabilities": dict(sample.rotation_probabilities),
                    "grid_shape": [int(grid_rows), int(grid_cols)],
                    "option_labels": list(tile_labels),
                    "usable_tile_indices": [int(index) for index in usable_indices],
                    "answer_label": answer_label,
                    "correct_index": int(correct_index),
                    "correct_index_probabilities": dict(correct_index_probabilities),
                    "source_size": [int(source_size[0]), int(source_size[1])],
                    **source_profile.trace(),
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(source_scene.canvas_width), int(source_scene.canvas_height)],
                "source_profile": source_profile.trace(),
                "style": {
                    "source_setting_id": str(source_scene.setting_id),
                    "source_style_id": str(source_scene.style_id),
                    "render_scale": int(source_scene.render_scale),
                    "source_layout": dict(source_scene.layout),
                    "grid_style": style_trace(grid_style),
                    "tile_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "image_id": "img0",
                "tile_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.tile_bboxes.items()},
                "rotated_tile_bbox_px": list(artifacts.selected_bbox),
                "selected_tile_bbox_px": list(artifacts.selected_bbox),
                "source_bboxes_px": source_bbox_map,
                "source_scene_canvas_size": [int(source_scene.canvas_width), int(source_scene.canvas_height)],
                "source_tile_index": int(correct_index),
                "source_size": [int(source_size[0]), int(source_size[1])],
                "grid_shape": [int(grid_rows), int(grid_cols)],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": QUERY_ID,
                "scene_id": SCENE_ID,
                "answer": answer_label,
                "answer_label": answer_label,
                "rotated_tile_label": answer_label,
                "rotated_tile_index": int(correct_index),
                "rotation_degrees": int(sample.rotation_degrees),
                "grid_shape": [int(grid_rows), int(grid_cols)],
                "tile_labels": list(tile_labels),
                "usable_tile_indices": [int(index) for index in usable_indices],
                "source_scene": serialized_scene[0],
            },
            "witness_symbolic": {
                "rotated_tile_label": answer_label,
                "rotated_tile_index": int(correct_index),
                "rotated_tile_bbox": list(artifacts.selected_bbox),
                "rotation_degrees": int(sample.rotation_degrees),
            },
            "projected_annotation": {
                **dict(annotation_artifacts.projected_annotation),
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="option_letter", value=answer_label),
            annotation_gt=annotation_artifacts.annotation_gt,
            image=artifacts.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
        )


__all__ = ["IllustrationsConstructionSiteRotatedTileLabelTask"]
