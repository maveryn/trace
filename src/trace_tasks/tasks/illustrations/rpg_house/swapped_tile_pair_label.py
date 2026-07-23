"""Identify the swapped tile pair in a numbered RPG house grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import uniform_choice_with_probabilities
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_set_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.canvas_profiles import MAX_RECONSTRUCTION_OUTPUT_PIXELS
from trace_tasks.tasks.illustrations.shared.cutouts import (
    FRAMELESS_ILLUSTRATION_SWAPPED_GRID_STYLE,
    SWAPPED_TILE_PAIR_OPTION_LABELS,
    compose_swapped_tile_pair_mcq,
    downscale_swapped_tile_pair_artifacts,
    style_trace,
    swapped_tile_pair_candidates,
)
from trace_tasks.tasks.illustrations.shared.option_rendering import sample_visual_label_font_trace

from .shared.output import rpg_house_scene_ir
from .shared.prompts import build_rpg_house_prompt_artifacts
from .shared.rendering import SCENE_ID
from .shared.source_images import (
    render_rpg_house_source_scene,
    rpg_house_source_style_trace,
    sample_rpg_house_source_scene_spec,
    sample_support_index,
)


TASK_ID = "task_illustrations__rpg_house__swapped_tile_pair_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "swapped_tile_pair_label"
GRID_ROWS = 3
GRID_COLS = 3
OPTION_LABELS: Tuple[str, ...] = SWAPPED_TILE_PAIR_OPTION_LABELS


@dataclass(frozen=True)
class _Defaults:
    source_room_count_min: int = 6
    source_room_count_max: int = 8
    source_width: int = 1200
    source_height: int = 798
    min_tile_detail_score: float = 120.0
    min_pair_difference: float = 7.0


@dataclass(frozen=True)
class _SampleSpec:
    query_id: str
    correct_index: int
    source_room_count: int
    source_size: Tuple[int, int]
    source_profile_trace: Dict[str, Any]
    query_probabilities: Dict[str, float]
    correct_index_probabilities: Dict[str, float]
    source_room_count_probabilities: Dict[str, float]


_DEFAULTS = _Defaults()
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
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
    if len(candidates) < len(OPTION_LABELS):
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
    return (
        (int(selected[0]), int(selected[1])),
        {_pair_key(pair): float(probabilities[str(pair)]) for pair in candidates},
    )


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


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SINGLE_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    correct_index, correct_probabilities = sample_support_index(
        seed_namespace=f"{TASK_ID}:correct_index",
        instance_seed=int(instance_seed),
        params=task_params,
        support=tuple(range(len(OPTION_LABELS))),
        explicit_key="correct_index",
    )
    source = sample_rpg_house_source_scene_spec(
        seed_namespace=TASK_ID,
        instance_seed=int(instance_seed),
        params=task_params,
        generation_defaults=_GEN_DEFAULTS,
        source_room_count_min=_DEFAULTS.source_room_count_min,
        source_room_count_max=_DEFAULTS.source_room_count_max,
        source_width=_DEFAULTS.source_width,
        source_height=_DEFAULTS.source_height,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
    )
    return _SampleSpec(
        query_id=str(query_id),
        correct_index=int(correct_index),
        source_room_count=int(source.source_room_count),
        source_size=tuple(source.source_size),
        source_profile_trace=dict(source.source_profile_trace),
        query_probabilities=dict(query_probabilities),
        correct_index_probabilities=dict(correct_probabilities),
        source_room_count_probabilities=dict(source.source_room_count_probabilities),
    )


@register_task
class IllustrationsRpgHouseSwappedTilePairLabelTask:
    """Select the option naming the two swapped numbered cells."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Render a numbered 3x3 RPG house grid with one swapped tile pair."""

        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        source_scene = None
        artifacts = None
        grid_style = {"style_id": "frameless_illustration", **dict(FRAMELESS_ILLUSTRATION_SWAPPED_GRID_STYLE)}
        label_font_trace: Dict[str, Any] | None = None
        candidate_pairs: Tuple[Tuple[int, int], ...] = tuple()
        swapped_pair: Tuple[int, int] | None = None
        swapped_pair_probabilities: Dict[str, float] | None = None
        min_tile_detail_score = float(params.get("min_tile_detail_score", group_default(_GEN_DEFAULTS, "min_tile_detail_score", _DEFAULTS.min_tile_detail_score)))
        min_pair_difference = float(params.get("min_pair_difference", group_default(_GEN_DEFAULTS, "min_pair_difference", _DEFAULTS.min_pair_difference)))

        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params)
                source_spec = sample_rpg_house_source_scene_spec(
                    seed_namespace=TASK_ID,
                    instance_seed=int(instance_seed),
                    params={**dict(params), "source_room_count": int(sample.source_room_count)},
                    generation_defaults=_GEN_DEFAULTS,
                    source_room_count_min=_DEFAULTS.source_room_count_min,
                    source_room_count_max=_DEFAULTS.source_room_count_max,
                    source_width=_DEFAULTS.source_width,
                    source_height=_DEFAULTS.source_height,
                    grid_rows=GRID_ROWS,
                    grid_cols=GRID_COLS,
                )
                source_scene = render_rpg_house_source_scene(
                    seed_namespace=TASK_ID,
                    instance_seed=int(instance_seed),
                    attempt_index=int(attempt),
                    source=source_spec,
                    params={**dict(params), "source_room_count": int(sample.source_room_count)},
                    render_defaults=_RENDER_DEFAULTS,
                )
                source_panel = source_scene.image.convert("RGB")
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
                source_scene = None
                artifacts = None
                label_font_trace = None
                candidate_pairs = tuple()
                swapped_pair = None
                swapped_pair_probabilities = None

        if (
            sample is None
            or source_scene is None
            or artifacts is None
            or label_font_trace is None
            or swapped_pair is None
            or swapped_pair_probabilities is None
        ):
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        answer_label = str(artifacts.selected_label)
        annotation_artifacts = bbox_set_annotation_artifacts(artifacts.swapped_cell_bboxes)
        option_pairs_by_label = _option_pairs_by_label(artifacts.option_pairs)
        option_pair_indices_by_label = _option_pair_indices_by_label(artifacts.option_pairs)
        swapped_cell_numbers = [int(swapped_pair[0]) + 1, int(swapped_pair[1]) + 1]
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_rpg_house_swapped_tile_pair_label",
                "annotation_hint_rpg_house_swapped_tile_pair_label",
                "json_example_rpg_house_swapped_tile_pair_label",
                "json_example_answer_only_rpg_house_swapped_tile_pair_label",
            ],
            context=f"prompt defaults for {TASK_ID}",
        )
        prompt_artifacts = build_rpg_house_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=prompt_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots={
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "answer_hint": str(prompt_defaults["answer_hint_rpg_house_swapped_tile_pair_label"]),
                "annotation_hint": str(prompt_defaults["annotation_hint_rpg_house_swapped_tile_pair_label"]),
                "json_example": str(prompt_defaults["json_example_rpg_house_swapped_tile_pair_label"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only_rpg_house_swapped_tile_pair_label"]),
            },
            instance_seed=int(instance_seed),
        )
        trace_payload = {
            "scene_ir": rpg_house_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=source_scene,
                relations={
                    "query_id": str(sample.query_id),
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "answer_label": answer_label,
                    "swapped_pair_indices": [int(swapped_pair[0]), int(swapped_pair[1])],
                    "swapped_cell_numbers": list(swapped_cell_numbers),
                },
            ),
            "query_spec": {
                "task_id": TASK_ID,
                "query_id": str(sample.query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(sample.query_id),
                    "query_id_probabilities": dict(sample.query_probabilities),
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "source_room_count": int(sample.source_room_count),
                    "source_room_count_probabilities": dict(sample.source_room_count_probabilities),
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
                "source_scene_canvas_size": [int(source_scene.image.width), int(source_scene.image.height)],
                "source_profile": dict(sample.source_profile_trace),
                "style": {
                    **rpg_house_source_style_trace(source_scene),
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
                "source_scene_canvas_size": [int(source_scene.image.width), int(source_scene.image.height)],
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
                "query_id": str(sample.query_id),
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
                "source_scene": dict(source_scene.trace),
            },
            "witness_symbolic": {
                "answer_label": answer_label,
                "swapped_pair_indices": [int(swapped_pair[0]), int(swapped_pair[1])],
                "swapped_cell_numbers": list(swapped_cell_numbers),
                "swapped_cell_bboxes": [list(bbox) for bbox in artifacts.swapped_cell_bboxes],
            },
            "projected_annotation": dict(annotation_artifacts.projected_annotation),
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
            query_id=str(sample.query_id),
        )


__all__ = [
    "IllustrationsRpgHouseSwappedTilePairLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
