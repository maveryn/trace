"""Identify the swapped tile pair in a numbered pixel-village grid."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import gcd
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.scene_config import get_scene_defaults
from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_set_annotation_artifacts
from ...shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ..shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS
from ..shared.cutouts import (
    FRAMELESS_ILLUSTRATION_SWAPPED_GRID_STYLE,
    SWAPPED_TILE_PAIR_OPTION_LABELS,
    compose_swapped_tile_pair_mcq,
    downscale_swapped_tile_pair_artifacts,
    style_trace,
    swapped_tile_pair_candidates,
)
from ..shared.option_rendering import sample_visual_label_font_trace
from .shared.output import pixel_village_scene_ir
from .shared.prompts import build_pixel_village_prompt_artifacts
from .shared.sampling import SCENE_ID, sample_option_answer_index
from .shared.source_images import PixelVillageSourceSpec, build_pixel_village_source_spec, render_pixel_village_source_scene, source_panel_for_scene


TASK_ID = "task_illustrations__pixel_village__swapped_tile_pair_label"
QUERY_ID = SINGLE_QUERY_ID
PROMPT_QUERY_KEY = "swapped_tile_pair_label"
GRID_ROWS = 3
GRID_COLS = 3
OPTION_LABELS: Tuple[str, ...] = SWAPPED_TILE_PAIR_OPTION_LABELS


@dataclass(frozen=True)
class _Defaults:
    source_width: int = 1200
    source_height: int = 840
    min_tile_detail_score: float = 150.0
    min_pair_difference: float = 8.0


@dataclass(frozen=True)
class _SampleSpec:
    correct_index: int
    correct_index_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _float_value(params: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(_GEN_DEFAULTS, str(key), float(fallback))))


def _tile_grid_multiple(tile_px: int) -> int:
    tile_size = max(1, int(tile_px))
    return abs(tile_size * GRID_COLS) // gcd(tile_size, GRID_COLS)


def _align_dimension(value: int, *, multiple: int) -> int:
    raw = max(int(multiple), int(value))
    rounded = int(round(float(raw) / float(multiple))) * int(multiple)
    return max(int(multiple), int(rounded))


def _build_grid_aligned_source_spec(*, params: Mapping[str, Any], instance_seed: int) -> tuple[PixelVillageSourceSpec, Dict[str, Any]]:
    source_spec = build_pixel_village_source_spec(
        params=params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        fallback_source_width=_DEFAULTS.source_width,
        fallback_source_height=_DEFAULTS.source_height,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:source_profile",
    )
    tile_px = int(source_spec.tile_px)
    multiple = _tile_grid_multiple(tile_px)
    raw_width, raw_height = int(source_spec.source_size[0]), int(source_spec.source_size[1])
    width = _align_dimension(raw_width, multiple=multiple)
    height = _align_dimension(raw_height, multiple=multiple)
    trace = {
        "canvas_profile": str(source_spec.canvas_profile),
        "canvas_profile_probabilities": dict(source_spec.canvas_profile_probabilities),
        "pre_grid_tile_aligned_source_size": [raw_width, raw_height],
        "source_size": [width, height],
        "grid_tile_alignment": {
            "rows": GRID_ROWS,
            "cols": GRID_COLS,
            "tile_px": int(tile_px),
            "size_multiple_px": int(multiple),
        },
    }
    if width == raw_width and height == raw_height:
        trace["canvas_profile_size"] = [width, height]
        return source_spec, trace
    aligned_spec = replace(
        source_spec,
        source_size=(int(width), int(height)),
        canvas_size=(int(width), int(height)),
        grid_cols=int(width // tile_px),
        grid_rows=int(height // tile_px),
    )
    trace["canvas_profile_size"] = [width, height]
    return aligned_spec, trace


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any], attempt_index: int, option_labels: Sequence[str] = OPTION_LABELS) -> _SampleSpec:
    """Sample one correct option position for the swapped tile pair task."""

    del attempt_index
    correct_index, correct_index_probabilities = sample_option_answer_index(
        params=params,
        instance_seed=int(instance_seed),
        seed_scope=TASK_ID,
        option_labels=option_labels,
    )
    return _SampleSpec(correct_index=int(correct_index), correct_index_probabilities=dict(correct_index_probabilities))


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
class IllustrationsPixelVillageSwappedTilePairLabelTask:
    """Select the option naming the two swapped numbered pixel-village cells."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = "illustrations"
    supported_query_ids = (QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render a numbered 3x3 pixel-village grid with one swapped tile pair."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene = None
        artifacts = None
        grid_style = None
        label_font_trace: Dict[str, Any] | None = None
        candidate_pairs: Tuple[Tuple[int, int], ...] = tuple()
        swapped_pair: Tuple[int, int] | None = None
        swapped_pair_probabilities: Dict[str, float] | None = None
        source_spec, source_profile_trace = _build_grid_aligned_source_spec(
            params=params,
            instance_seed=int(instance_seed),
        )
        min_tile_detail_score = _float_value(params, "min_tile_detail_score", _DEFAULTS.min_tile_detail_score)
        min_pair_difference = _float_value(params, "min_pair_difference", _DEFAULTS.min_pair_difference)

        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(
                    instance_seed=int(instance_seed),
                    params=params,
                    attempt_index=int(attempt),
                    option_labels=OPTION_LABELS,
                )
                scene = render_pixel_village_source_scene(
                    seed_namespace=TASK_ID,
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    source_spec=source_spec,
                )
                source_panel = source_panel_for_scene(scene, source_spec.source_size)
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
        prompt_artifacts = build_pixel_village_prompt_artifacts(
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
            "scene_ir": pixel_village_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={
                    "query_id": QUERY_ID,
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "answer_label": answer_label,
                    "swapped_cell_numbers": list(swapped_cell_numbers),
                    "grid_shape": [GRID_ROWS, GRID_COLS],
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
                    **dict(source_profile_trace),
                    "min_tile_detail_score": float(min_tile_detail_score),
                    "min_pair_difference": float(min_pair_difference),
                },
            },
            "render_spec": {
                "canvas_size": [int(artifacts.image.width), int(artifacts.image.height)],
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "source_scene_canvas_size": [int(scene.image.width), int(scene.image.height)],
                "source_profile": dict(source_profile_trace),
                "style": {
                    "source_renderer_id": str(scene.trace.get("renderer_id", "")),
                    "source_theme_id": str(scene.trace.get("theme_id", "")),
                    "source_tile_px": int(scene.trace.get("tile_px", 0)),
                    "source_grid_cols": int(scene.trace.get("grid_cols", 0)),
                    "source_grid_rows": int(scene.trace.get("grid_rows", 0)),
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
                "source_scene_canvas_size": [int(scene.image.width), int(scene.image.height)],
                "source_size": [int(value) for value in source_spec.source_size],
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


__all__ = ["IllustrationsPixelVillageSwappedTilePairLabelTask", "TASK_ID", "_sample_spec"]
