"""Identify the rotated tile in an environment illustration grid."""

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
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.output_metadata import default_task_versions
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
    reconstruction_grid_for_size,
    reconstruction_option_labels,
    resolve_reconstruction_source_profile,
)
from ..shared.option_rendering import sample_visual_label_font_trace
from ..shared.task_support import uniform_string_probability_map
from .shared.annotations import feature_bbox_map, feature_path_map
from .shared.defaults import CountContractDefaults, render_fallback
from .shared.output import serialize_environment_objects
from .shared.prompts import render_environment_prompt, required_environment_prompt_defaults
from .shared.rendering import ENVIRONMENT_THEME_IDS, environment_scene_entities, render_environment_object_scene, serialize_environment_scene
from .shared.sampling import environment_render_params, environment_setting_name, sample_count_support, style_weights, theme_support


TASK_ID = "task_illustrations__environment__rotated_tile_label"
DOMAIN = "illustrations"
SCENE_ID = "environment"
QUERY_ID = SINGLE_QUERY_ID
GRID_ROWS = 2
GRID_COLS = 3
TILE_LABELS: Tuple[str, ...] = DEFAULT_OPTION_LABELS[: GRID_ROWS * GRID_COLS]
ROTATION_SUPPORT: Tuple[int, ...] = (90, 270)


@dataclass(frozen=True)
class _Defaults:
    source_object_count_min: int = 12
    source_object_count_max: int = 18
    source_width: int = 960
    source_height: int = 640
    min_tile_detail_score: float = 160.0
    min_rotation_delta: float = 7.0


@dataclass(frozen=True)
class _SampleSpec:
    theme_id: str
    theme_probabilities: Dict[str, float]
    source_object_count: int
    source_object_count_probabilities: Dict[str, float]
    rotation_degrees: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    rotation_probabilities: Dict[str, float]


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
    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _float_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(defaults, str(key), float(fallback))))


def _sample_theme(*, params: Mapping[str, Any], instance_seed: int) -> Tuple[str, Dict[str, float]]:
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
    """Sample source environment density and rotation parameters before rendering selects a usable tile."""

    del attempt_index
    theme_id, theme_probabilities = _sample_theme(params=params, instance_seed=int(instance_seed))
    source_object_count, source_object_count_probabilities = _sample_source_object_count(
        params=params,
        instance_seed=int(instance_seed),
    )
    rotation_degrees, rotation_probabilities = _sample_rotation(params=params, instance_seed=int(instance_seed))
    source_profile = resolve_reconstruction_source_profile(
        params=params,
        defaults=_GEN_DEFAULTS,
        fallback_source_width=_DEFAULTS.source_width,
        fallback_source_height=_DEFAULTS.source_height,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
    return _SampleSpec(
        theme_id=str(theme_id),
        theme_probabilities=dict(theme_probabilities),
        source_object_count=int(source_object_count),
        source_object_count_probabilities=dict(source_object_count_probabilities),
        rotation_degrees=int(rotation_degrees),
        source_size=tuple(int(value) for value in source_profile.size),
        source_profile_trace=dict(source_profile.trace()),
        rotation_probabilities=dict(rotation_probabilities),
    )


def _usable_tile_indices(
    *,
    source_image: Image.Image,
    rotation_degrees: int,
    min_detail_score: float,
    min_rotation_delta: float,
    rows: int,
    cols: int,
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
        raise ValueError("no visually usable environment tile for rotation")
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
class IllustrationsEnvironmentRotatedTileLabelTask:
    """Select the lettered tile that has been rotated inside an environment grid."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = DOMAIN
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render one source environment scene, rotate one usable tile, and bind tile annotation."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        source_scene = None
        artifacts = None
        grid_style = None
        label_font_trace: Dict[str, Any] | None = None
        correct_index = None
        correct_index_probabilities: Dict[str, float] | None = None
        usable_indices: Tuple[int, ...] = tuple()
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                source_width, source_height = int(sample.source_size[0]), int(sample.source_size[1])
                grid_rows, grid_cols = reconstruction_grid_for_size(source_width, source_height)
                tile_labels = reconstruction_option_labels(grid_rows, grid_cols)
                render_overrides = {
                    **dict(params),
                    "canvas_width": source_width,
                    "canvas_height": source_height,
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
                source_panel = source_scene.image.convert("RGB")
                usable_indices = _usable_tile_indices(
                    source_image=source_panel,
                    rotation_degrees=int(sample.rotation_degrees),
                    min_detail_score=_float_value(params, _GEN_DEFAULTS, "min_tile_detail_score", _DEFAULTS.min_tile_detail_score),
                    min_rotation_delta=_float_value(params, _GEN_DEFAULTS, "min_rotation_delta", _DEFAULTS.min_rotation_delta),
                    rows=grid_rows,
                    cols=grid_cols,
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
                    rows=grid_rows,
                    cols=grid_cols,
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

        serialized_objects, object_bboxes, part_bboxes = serialize_environment_objects(source_scene)
        feature_bboxes = feature_bbox_map(source_scene)
        feature_paths = feature_path_map(source_scene)
        answer_label = str(artifacts.selected_label)
        grid_rows, grid_cols = int(artifacts.grid_shape[0]), int(artifacts.grid_shape[1])
        tile_labels = reconstruction_option_labels(grid_rows, grid_cols)
        annotation_artifacts = bbox_annotation_artifacts(artifacts.selected_bbox)
        prompt_defaults = required_environment_prompt_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "question_text_rotated_tile_label",
                "answer_hint_rotated_tile",
                "annotation_hint_rotated_tile",
                "json_example_rotated_tile",
                "json_example_answer_only_rotated_tile",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = render_environment_prompt(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            dynamic_slots={
                "environment_setting": environment_setting_name(str(sample.theme_id)),
                "question_text": str(prompt_defaults["question_text_rotated_tile_label"]),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_rotated_tile"]),
                "answer_hint": str(prompt_defaults["answer_hint_rotated_tile"]),
                "json_example": str(prompt_defaults["json_example_rotated_tile"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_rotated_tile"]),
            },
            instance_seed=int(instance_seed),
            preferred_mode="answer_and_annotation",
        )
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "scene_kind": "environment_rotated_tile_label",
                "entities": environment_scene_entities(source_scene),
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
                    "theme": str(sample.theme_id),
                    "theme_id": str(sample.theme_id),
                    "theme_probabilities": dict(sample.theme_probabilities),
                    "source_object_count": int(len(source_scene.placements)),
                    "requested_source_object_count": int(sample.source_object_count),
                    "source_object_count_probabilities": dict(sample.source_object_count_probabilities),
                    "rotation_degrees": int(sample.rotation_degrees),
                    "rotation_degrees_support": [int(value) for value in _rotation_support(params)],
                    "rotation_degrees_probabilities": dict(sample.rotation_probabilities),
                    "grid_shape": [grid_rows, grid_cols],
                    "option_labels": list(tile_labels),
                    "usable_tile_indices": [int(index) for index in usable_indices],
                    "answer_label": answer_label,
                    "correct_index": int(correct_index),
                    "correct_index_probabilities": dict(correct_index_probabilities),
                    "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                    **dict(sample.source_profile_trace),
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
                    "grid_style": style_trace(grid_style),
                    "tile_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "image_id": "img0",
                "tile_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.tile_bboxes.items()},
                "rotated_tile_bbox_px": list(artifacts.selected_bbox),
                "selected_tile_bbox_px": list(artifacts.selected_bbox),
                "source_object_bboxes_px": object_bboxes,
                "source_part_bboxes_px": part_bboxes,
                "source_feature_bboxes_px": feature_bboxes,
                "source_feature_paths_px": feature_paths,
                "source_scene_canvas_size": [int(source_scene.canvas_width), int(source_scene.canvas_height)],
                "source_tile_index": int(correct_index),
                "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                "grid_shape": [grid_rows, grid_cols],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": QUERY_ID,
                "scene_id": SCENE_ID,
                "theme_id": str(source_scene.theme_id),
                "theme": str(source_scene.theme_id),
                "answer": answer_label,
                "answer_label": answer_label,
                "rotated_tile_label": answer_label,
                "rotated_tile_index": int(correct_index),
                "rotation_degrees": int(sample.rotation_degrees),
                "grid_shape": [grid_rows, grid_cols],
                "tile_labels": list(tile_labels),
                "usable_tile_indices": [int(index) for index in usable_indices],
                "source_scene": serialize_environment_scene(source_scene),
                "objects": serialized_objects,
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


__all__ = ["IllustrationsEnvironmentRotatedTileLabelTask"]
