"""Identify the swapped tile pair in a numbered park/playground grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.output_metadata import default_task_versions
from ..shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS, resolve_reconstruction_source_profile
from ..shared.cutouts import (
    FRAMELESS_ILLUSTRATION_SWAPPED_GRID_STYLE,
    SWAPPED_TILE_PAIR_OPTION_LABELS,
    compose_swapped_tile_pair_mcq,
    downscale_swapped_tile_pair_artifacts,
    style_trace,
    swapped_tile_pair_candidates,
)
from ..shared.option_rendering import sample_visual_label_font_trace
from .shared.annotations import park_scene_entities, serialize_park_scene
from .shared.prompts import build_park_prompt_artifacts
from .shared.rendering import PARK_EQUIPMENT_TYPES, PARK_PERSON_ACTIVITIES, ParkEquipmentSpec, ParkPersonSpec, render_park_playground_scene
from .shared.sampling import activity_support, bounds, equipment_support, render_params, sample_count, sample_option_answer_index, setting_weights, spawned_task_rng, style_weights


TASK_ID = "task_illustrations__park_playground__swapped_tile_pair_label"
SCENE_ID = "park_playground"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "swapped_tile_pair_label"
GRID_ROWS = 3
GRID_COLS = 3
OPTION_LABELS: Tuple[str, ...] = SWAPPED_TILE_PAIR_OPTION_LABELS


@dataclass(frozen=True)
class _Defaults:
    source_person_count_min: int = 8
    source_person_count_max: int = 13
    source_equipment_count_min: int = 4
    source_equipment_count_max: int = 7
    source_width: int = 1200
    source_height: int = 798
    canvas_width: int = 1280
    canvas_height: int = 900
    render_scale: int = 2
    min_tile_detail_score: float = 180.0
    min_pair_difference: float = 8.0


@dataclass(frozen=True)
class _SampleSpec:
    source_person_count: int
    source_equipment_count: int
    correct_index: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    person_specs: Tuple[ParkPersonSpec, ...]
    equipment_specs: Tuple[ParkEquipmentSpec, ...]
    source_person_count_probabilities: Dict[str, float]
    source_equipment_count_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "illustrations",
    SCENE_ID,
    task_id=TASK_ID,
)


def _int_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def _float_value(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(defaults, str(key), float(fallback))))


def _grid_aligned_source_profile(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[Tuple[int, int], Dict[str, Any]]:
    profile = resolve_reconstruction_source_profile(
        params=params,
        defaults=_GEN_DEFAULTS,
        fallback_source_width=_DEFAULTS.source_width,
        fallback_source_height=_DEFAULTS.source_height,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
    raw_width, raw_height = int(profile.width), int(profile.height)
    width = raw_width - (raw_width % GRID_COLS)
    height = raw_height - (raw_height % GRID_ROWS)
    if width < GRID_COLS or height < GRID_ROWS:
        raise ValueError("source profile is too small for a 3x3 swapped tile grid")
    trace = dict(profile.trace())
    if width != raw_width or height != raw_height:
        trace["pre_grid_snap_canvas_profile_size"] = [raw_width, raw_height]
        trace["canvas_profile_size"] = [int(width), int(height)]
    trace["grid_alignment"] = {"rows": GRID_ROWS, "cols": GRID_COLS}
    return (int(width), int(height)), trace


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int) -> _SampleSpec:
    """Sample a dense park source scene and one correct MCQ option position."""

    rng = spawned_task_rng(int(instance_seed), TASK_ID, int(attempt_index))
    activities = activity_support(params, _GEN_DEFAULTS, fallback=PARK_PERSON_ACTIVITIES)
    equipment_values = equipment_support(params, _GEN_DEFAULTS, fallback=PARK_EQUIPMENT_TYPES)
    person_min, person_max = bounds(
        params,
        _GEN_DEFAULTS,
        "source_person_count_min",
        "source_person_count_max",
        _DEFAULTS.source_person_count_min,
        _DEFAULTS.source_person_count_max,
    )
    source_person_count, person_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_person_count",
        low=int(person_min),
        high=int(person_max),
        explicit_key="source_person_count",
    )
    equipment_min, equipment_max = bounds(
        params,
        _GEN_DEFAULTS,
        "source_equipment_count_min",
        "source_equipment_count_max",
        _DEFAULTS.source_equipment_count_min,
        _DEFAULTS.source_equipment_count_max,
    )
    source_equipment_count, equipment_count_probabilities = sample_count(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_equipment_count",
        low=int(equipment_min),
        high=int(equipment_max),
        explicit_key="source_equipment_count",
    )
    source_size, source_profile_trace = _grid_aligned_source_profile(params=params, instance_seed=int(instance_seed))
    correct_index, correct_index_probabilities = sample_option_answer_index(
        params=params,
        instance_seed=int(instance_seed),
        seed_scope=TASK_ID,
        option_labels=OPTION_LABELS,
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
        source_person_count=int(source_person_count),
        source_equipment_count=int(source_equipment_count),
        correct_index=int(correct_index),
        source_size=tuple(int(value) for value in source_size),
        source_profile_trace=dict(source_profile_trace),
        person_specs=person_specs,
        equipment_specs=equipment_specs,
        source_person_count_probabilities=dict(person_count_probabilities),
        source_equipment_count_probabilities=dict(equipment_count_probabilities),
        correct_index_probabilities=dict(correct_index_probabilities),
    )


def _pair_key(pair: Sequence[int]) -> str:
    left, right = sorted(int(value) for value in pair[:2])
    return f"{left + 1}-{right + 1}"


def _select_swapped_pair(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    attempt_index: int,
    candidate_pairs: Sequence[Sequence[int]],
) -> Tuple[Tuple[int, int], Dict[str, float]]:
    candidates = tuple(tuple(sorted(int(value) for value in pair[:2])) for pair in candidate_pairs)
    if len(candidates) < 4:
        raise ValueError("not enough visually usable swapped tile pair candidates")
    explicit = params.get("swapped_pair")
    if explicit is not None and isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        pair = tuple(sorted(int(value) for value in tuple(explicit)[:2]))
        if pair not in set(candidates):
            raise ValueError("swapped_pair is outside visually usable candidate support")
        return (int(pair[0]), int(pair[1])), {_pair_key(pair): 1.0}
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:swapped_pair", int(attempt_index))
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        candidates,
        sort_keys=True,
    )
    return (int(selected[0]), int(selected[1])), {
        _pair_key(pair): float(probabilities[str(pair)])
        for pair in candidates
    }


def _option_pairs_by_label(option_pairs: Sequence[Sequence[int]]) -> Dict[str, list[int]]:
    return {
        str(OPTION_LABELS[index]): [int(pair[0]) + 1, int(pair[1]) + 1]
        for index, pair in enumerate(option_pairs)
    }


def _option_pair_indices_by_label(option_pairs: Sequence[Sequence[int]]) -> Dict[str, list[int]]:
    return {
        str(OPTION_LABELS[index]): [int(pair[0]), int(pair[1])]
        for index, pair in enumerate(option_pairs)
    }


@register_task
class IllustrationsParkPlaygroundSwappedTilePairLabelTask:
    """Select the option naming the two swapped numbered cells."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = "illustrations"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render a numbered 3x3 park grid with one swapped tile pair."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        artifacts = None
        grid_style = None
        label_font_trace: Dict[str, Any] | None = None
        candidate_pairs: Tuple[Tuple[int, int], ...] = tuple()
        swapped_pair: Tuple[int, int] | None = None
        swapped_pair_probabilities: Dict[str, float] | None = None
        min_tile_detail_score = _float_value(params, _GEN_DEFAULTS, "min_tile_detail_score", _DEFAULTS.min_tile_detail_score)
        min_pair_difference = _float_value(params, _GEN_DEFAULTS, "min_pair_difference", _DEFAULTS.min_pair_difference)

        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params, attempt_index=int(attempt))
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:source_scene", int(attempt))
                source_width, source_height = int(sample.source_size[0]), int(sample.source_size[1])
                rp = render_params(
                    {
                        **dict(params),
                        "canvas_width": source_width,
                        "canvas_height": source_height,
                    },
                    _RENDER_DEFAULTS,
                    fallback_width=_DEFAULTS.canvas_width,
                    fallback_height=_DEFAULTS.canvas_height,
                    fallback_scale=_DEFAULTS.render_scale,
                    instance_seed=int(instance_seed),
                    namespace=f"{TASK_ID}:source_profile",
                )
                scene = render_park_playground_scene(
                    rng=scene_rng,
                    person_specs=sample.person_specs,
                    equipment_specs=sample.equipment_specs,
                    canvas_width=int(rp["canvas_width"]),
                    canvas_height=int(rp["canvas_height"]),
                    render_scale=int(rp["render_scale"]),
                    setting_weights=setting_weights(params, _RENDER_DEFAULTS),
                    style_weights=style_weights(params, _RENDER_DEFAULTS),
                )
                source_panel = scene.image.convert("RGB")
                candidate_pairs = swapped_tile_pair_candidates(
                    source_panel,
                    rows=GRID_ROWS,
                    cols=GRID_COLS,
                    min_tile_detail_score=float(min_tile_detail_score),
                    min_pair_difference=float(min_pair_difference),
                )
                swapped_pair, swapped_pair_probabilities = _select_swapped_pair(
                    params=params,
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    candidate_pairs=candidate_pairs,
                )
                grid_style = {"style_id": "frameless_illustration", **dict(FRAMELESS_ILLUSTRATION_SWAPPED_GRID_STYLE)}
                label_font_trace = sample_visual_label_font_trace(
                    namespace_prefix=TASK_ID,
                    instance_seed=int(instance_seed),
                    params={**dict(_RENDER_DEFAULTS), **dict(params)},
                    namespace_suffix="swapped_tile_labels",
                    explicit_key="swapped_tile_label_font_family",
                    weights_key="swapped_tile_label_font_weights",
                )
                artifacts = compose_swapped_tile_pair_mcq(
                    source_image=source_panel,
                    swapped_pair=swapped_pair,
                    correct_index=int(sample.correct_index),
                    rng=spawn_rng(int(instance_seed), f"{TASK_ID}:options", int(attempt)),
                    grid_style=grid_style,
                    label_font_family=str(label_font_trace["font_family"]),
                    candidate_pairs=candidate_pairs,
                    rows=GRID_ROWS,
                    cols=GRID_COLS,
                    option_labels=OPTION_LABELS,
                )
                artifacts = downscale_swapped_tile_pair_artifacts(
                    artifacts,
                    max_pixels=MAX_RECONSTRUCTION_OUTPUT_PIXELS,
                )
                break
            except Exception as exc:  # pragma: no cover - retry surface is seed/layout dependent.
                last_error = exc
                sample = None
                scene = None
                artifacts = None
                grid_style = None
                label_font_trace = None
                candidate_pairs = tuple()
                swapped_pair = None
                swapped_pair_probabilities = None

        if (
            sample is None
            or scene is None
            or artifacts is None
            or grid_style is None
            or label_font_trace is None
            or swapped_pair is None
            or swapped_pair_probabilities is None
        ):
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        serialized_scene, person_bboxes = serialize_park_scene(scene)
        answer_label = str(artifacts.selected_label)
        annotation_artifacts = bbox_set_annotation_artifacts(artifacts.swapped_cell_bboxes)
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_swapped_tile_pair",
                "annotation_hint_swapped_tile_pair",
                "json_example_swapped_tile_pair",
                "json_example_answer_only_swapped_tile_pair",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_park_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots={
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "answer_hint": str(prompt_defaults["answer_hint_swapped_tile_pair"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_swapped_tile_pair"]),
                "json_example": str(prompt_defaults["json_example_swapped_tile_pair"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_swapped_tile_pair"]),
            },
            instance_seed=int(instance_seed),
        )
        option_pairs_by_label = _option_pairs_by_label(artifacts.option_pairs)
        option_pair_indices_by_label = _option_pair_indices_by_label(artifacts.option_pairs)
        swapped_cell_numbers = [int(swapped_pair[0]) + 1, int(swapped_pair[1]) + 1]
        trace_payload = {
            "scene_ir": {
                "domain": self.domain,
                "scene_id": SCENE_ID,
                "entities": park_scene_entities(scene),
                "relations": {
                    "query_id": QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "answer_label": answer_label,
                    "swapped_cell_numbers": list(swapped_cell_numbers),
                    "grid_shape": [GRID_ROWS, GRID_COLS],
                },
            },
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
                    "source_person_count": int(sample.source_person_count),
                    "source_person_count_probabilities": dict(sample.source_person_count_probabilities),
                    "source_equipment_count": int(sample.source_equipment_count),
                    "source_equipment_count_probabilities": dict(sample.source_equipment_count_probabilities),
                    "correct_index": int(sample.correct_index),
                    "correct_index_probabilities": dict(sample.correct_index_probabilities),
                    "answer_label": answer_label,
                    "option_labels": list(OPTION_LABELS),
                    "option_pairs_by_label": dict(option_pairs_by_label),
                    "option_pair_indices_by_label": dict(option_pair_indices_by_label),
                    "swapped_pair_indices": [int(swapped_pair[0]), int(swapped_pair[1])],
                    "swapped_cell_numbers": list(swapped_cell_numbers),
                    "swapped_pair_probabilities": dict(swapped_pair_probabilities),
                    "candidate_pair_count": int(len(candidate_pairs)),
                    "grid_shape": [GRID_ROWS, GRID_COLS],
                    "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                    **dict(sample.source_profile_trace),
                    "min_tile_detail_score": float(min_tile_detail_score),
                    "min_pair_difference": float(min_pair_difference),
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
                "source_profile": dict(sample.source_profile_trace),
                "style": {
                    "source_setting_id": str(scene.setting_id),
                    "source_style_id": str(scene.style_id),
                    "source_render_scale": int(scene.render_scale),
                    "source_layout": dict(scene.layout),
                    "swapped_grid_style": style_trace(grid_style),
                    "swapped_tile_label_font": dict(label_font_trace),
                },
            },
            "render_map": {
                "image_id": "img0",
                "tile_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.tile_bboxes.items()},
                "tile_bboxes_px_by_number": {str(key): list(value) for key, value in artifacts.tile_bboxes.items()},
                "option_bboxes_px_by_label": {str(key): list(value) for key, value in artifacts.option_bboxes.items()},
                "swapped_cell_bboxes_px": [list(bbox) for bbox in artifacts.swapped_cell_bboxes],
                "source_person_bboxes_px": person_bboxes,
                "source_scene_canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
                "source_size": [int(sample.source_size[0]), int(sample.source_size[1])],
                "tile_source_boxes_px": [[int(coord) for coord in box] for box in artifacts.tile_source_boxes],
                "option_pairs_by_label": dict(option_pairs_by_label),
                "option_pair_indices_by_label": dict(option_pair_indices_by_label),
                "swapped_pair_indices": [int(swapped_pair[0]), int(swapped_pair[1])],
                "swapped_cell_numbers": list(swapped_cell_numbers),
                "grid_shape": [GRID_ROWS, GRID_COLS],
                "pre_downscale_canvas_size": [int(value) for value in artifacts.pre_downscale_canvas_size],
                "output_scale_xy": [float(value) for value in artifacts.output_scale_xy],
            },
            "execution_trace": {
                "query_id": QUERY_ID,
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "answer": answer_label,
                "answer_label": answer_label,
                "correct_index": int(sample.correct_index),
                "option_labels": list(OPTION_LABELS),
                "option_pairs_by_label": dict(option_pairs_by_label),
                "option_pair_indices_by_label": dict(option_pair_indices_by_label),
                "swapped_pair_indices": [int(swapped_pair[0]), int(swapped_pair[1])],
                "swapped_cell_numbers": list(swapped_cell_numbers),
                "candidate_pair_count": int(len(candidate_pairs)),
                "grid_shape": [GRID_ROWS, GRID_COLS],
                "source_person_count": int(sample.source_person_count),
                "source_equipment_count": int(sample.source_equipment_count),
                "source_scene": serialized_scene[0],
            },
            "witness_symbolic": {
                "answer_label": answer_label,
                "swapped_pair_indices": [int(swapped_pair[0]), int(swapped_pair[1])],
                "swapped_cell_numbers": list(swapped_cell_numbers),
                "swapped_cell_bboxes": [list(bbox) for bbox in artifacts.swapped_cell_bboxes],
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


__all__ = ["IllustrationsParkPlaygroundSwappedTilePairLabelTask", "TASK_ID", "_sample_spec"]
