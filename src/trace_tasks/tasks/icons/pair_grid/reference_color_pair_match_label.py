"""Select the Scene cell with the same left/right colors as Reference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.fixed_query import select_task_query_id
from ...shared.labeling import LABEL_POOL_A_L
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import bbox_annotation
from ..shared.icon_assets import resolve_icon_pool
from ..shared.icon_style import sample_icon_palette
from ..shared.icon_task_rendering import sample_icon_instance_noise
from ..shared.icon_transform import IDENTITY_TRANSFORM_ID

from .shared.defaults import PairGridTaskDefaults
from .shared.output import render_pair_grid_payload, selected_scene_cell_bbox
from .shared.state import IconPairSpec
from .shared.styles import pair_grid_style_trace, resolve_pair_grid_render_params


Color = Tuple[int, int, int]
ColorPair = Tuple[Color, Color]


@dataclass(frozen=True)
class _ScenePayload:
    """Trace-ready payload for one color-pair selection scene."""

    option_count: int
    reference_icon_id: str
    answer_label: str
    reference_color_pair: ColorPair
    cell_labels: Tuple[str, ...]
    cell_icon_ids: Tuple[str, ...]
    cell_color_pairs: Tuple[ColorPair, ...]
    sampled_palette_rgb: Tuple[Color, ...]
    panel_geometry: Dict[str, Any]
    reference_pair: Dict[str, Any]
    scene_cells: Tuple[Dict[str, Any], ...]


_DEFAULTS = PairGridTaskDefaults(
    pool_manifest="all_icons.txt",
    scene_icon_size_max_px=88,
    reference_icon_size_px=96,
    palette_size_min=5,
    palette_size_max=5,
    min_color_distance=42.0,
)
TASK_ID = "task_icons__pair_grid__reference_color_pair_match_label"
DOMAIN = "icons"
SCENE_ID = "pair_grid"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _select_query(instance_seed: int, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate the single public query contract."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _resolve_option_count(params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the fixed number of visible labeled options."""

    option_count = int(params.get("option_count", group_default(_GEN_DEFAULTS, "option_count", _DEFAULTS.option_count)))
    if not 2 <= int(option_count) <= len(LABEL_POOL_A_L):
        raise ValueError(f"option_count must be in [2, {len(LABEL_POOL_A_L)}]")
    return int(option_count), {str(option_count): 1.0}


def _resolve_answer_label(rng, *, params: Mapping[str, Any], labels: Sequence[str]) -> Tuple[str, Dict[str, float]]:
    """Resolve the unique correct option label."""

    label_set = {str(label) for label in labels}
    explicit = params.get("answer_label")
    if explicit is not None:
        selected = str(explicit).strip().upper()
        if selected not in label_set:
            raise ValueError(f"answer_label must be one of {sorted(label_set)}")
    else:
        selected = str(rng.choice(tuple(labels)))
    probability = 1.0 / float(len(labels))
    return selected, {str(label): probability for label in labels}


def _sample_palette(rng, *, render_params: Mapping[str, Any]) -> Tuple[Color, ...]:
    """Sample a high-contrast color palette for left/right icon colors."""

    palette_size_min = max(4, int(render_params["palette_size_min"]))
    palette_size_max = max(palette_size_min, int(render_params["palette_size_max"]))
    palette_size = int(rng.randint(palette_size_min, palette_size_max))
    return tuple(
        tuple(int(channel) for channel in color)
        for color in sample_icon_palette(
            rng,
            palette_size=int(palette_size),
            channel_min=int(render_params["color_channel_min"]),
            channel_max=int(render_params["color_channel_max"]),
            anchor_colors=(
                tuple(int(v) for v in render_params["background_color_rgb"]),
                tuple(int(v) for v in render_params["panel_fill_rgb"]),
                tuple(int(v) for v in render_params["panel_border_rgb"]),
                tuple(int(v) for v in render_params["header_text_rgb"]),
            ),
            min_color_distance=float(render_params["min_color_distance"]),
            distance_space=str(render_params["color_distance_space"]),
        )
    )


def _sample_reference_color_pair(rng, *, palette: Sequence[Color]) -> ColorPair:
    """Choose different left/right colors for the Reference pair."""

    left = tuple(int(channel) for channel in rng.choice(tuple(palette)))
    right_options = [tuple(int(channel) for channel in color) for color in palette if tuple(int(channel) for channel in color) != left]
    if not right_options:
        raise ValueError("color-pair task needs at least two palette colors")
    return left, tuple(int(channel) for channel in rng.choice(tuple(right_options)))


def _distractor_color_pairs(rng, *, reference_color_pair: ColorPair, palette: Sequence[Color], count: int) -> Tuple[ColorPair, ...]:
    """Return unique non-reference color pairs for distractor options."""

    left, right = reference_color_pair
    colors = [tuple(int(channel) for channel in color) for color in palette]
    same_left = [(left, color) for color in colors if color != right]
    same_right = [(color, right) for color in colors if color != left]
    reversed_pair = [(right, left)] if left != right else []
    both_different = [(a, b) for a in colors for b in colors if a != left and b != right]
    buckets: List[List[ColorPair]] = [same_left, same_right, reversed_pair, both_different]
    for bucket in buckets:
        rng.shuffle(bucket)

    selected: List[ColorPair] = []
    while len(selected) < int(count):
        added = False
        for bucket in buckets:
            while bucket:
                candidate = bucket.pop(0)
                if candidate != reference_color_pair and candidate not in selected:
                    selected.append(candidate)
                    added = True
                    break
            if len(selected) >= int(count):
                break
        if not added:
            break
    if len(selected) < int(count):
        raise ValueError("insufficient unique distractor color pairs")
    return tuple(selected[: int(count)])


def _make_pair_spec(
    *,
    icon_id: str,
    color_pair: ColorPair,
    instance_seed: int,
    namespace: str,
    render_params: Mapping[str, Any],
) -> IconPairSpec:
    """Build one color-pair spec with no size or transform cue."""

    left_noise_edits, left_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:left",
        render_params=render_params,
    )
    right_noise_edits, right_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:right",
        render_params=render_params,
    )
    left_color, right_color = color_pair
    return IconPairSpec(
        icon_id=str(icon_id),
        transform_id=IDENTITY_TRANSFORM_ID,
        tint_rgb=tuple(int(v) for v in left_color),
        left_tint_rgb=tuple(int(v) for v in left_color),
        right_tint_rgb=tuple(int(v) for v in right_color),
        left_size_scale=1.0,
        right_size_scale=1.0,
        left_noise_edits=tuple(left_noise_edits),
        left_noise_seed=int(left_noise_seed),
        right_noise_edits=tuple(right_noise_edits),
        right_noise_seed=int(right_noise_seed),
    )


def _sample_scene(
    rng,
    *,
    instance_seed: int,
    option_count: int,
    answer_label: str,
    pool_manifest: str,
    render_params: Mapping[str, Any],
) -> Tuple[_ScenePayload, Any]:
    """Sample and render one reference color-pair match label scene."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if len(pool) < int(option_count) + 1:
        raise ValueError("icon pool is too small for requested color-pair scene")
    rng.shuffle(pool)
    reference_icon_id = str(pool[0])
    scene_icon_ids = [str(icon_id) for icon_id in pool[1 : 1 + int(option_count)]]
    labels = tuple(str(value) for value in LABEL_POOL_A_L[: int(option_count)])
    answer_index = labels.index(str(answer_label))
    palette = _sample_palette(rng, render_params=render_params)
    reference_color_pair = _sample_reference_color_pair(rng, palette=palette)
    distractor_pairs = list(
        _distractor_color_pairs(
            rng,
            reference_color_pair=reference_color_pair,
            palette=palette,
            count=int(option_count) - 1,
        )
    )

    cell_color_pairs: List[ColorPair] = []
    for index in range(int(option_count)):
        if int(index) == int(answer_index):
            cell_color_pairs.append(reference_color_pair)
        else:
            cell_color_pairs.append(distractor_pairs.pop(0))

    reference_pair = _make_pair_spec(
        icon_id=str(reference_icon_id),
        color_pair=reference_color_pair,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:reference",
        render_params=render_params,
    )
    scene_pairs = [
        _make_pair_spec(
            icon_id=str(icon_id),
            color_pair=color_pair,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:scene_{str(label)}",
            render_params=render_params,
        )
        for label, icon_id, color_pair in zip(labels, scene_icon_ids, cell_color_pairs)
    ]

    rendered_payload = render_pair_grid_payload(
        reference_pair=reference_pair,
        scene_pairs=scene_pairs,
        scene_labels=labels,
        render_params=render_params,
        matching_labels=(str(answer_label),),
        reference_extra={
            "target_relation": "left_right_color_pair_match",
            "reference_color_pair_rgb": [list(reference_color_pair[0]), list(reference_color_pair[1])],
        },
        cell_extra_by_label={
            str(label): {
                "target_relation": "left_right_color_pair_match",
                "color_pair_rgb": [list(color_pair[0]), list(color_pair[1])],
                "is_correct_option": bool(str(label) == str(answer_label)),
            }
            for label, color_pair in zip(labels, cell_color_pairs)
        },
    )
    return _ScenePayload(
        option_count=int(option_count),
        reference_icon_id=str(reference_icon_id),
        answer_label=str(answer_label),
        reference_color_pair=reference_color_pair,
        cell_labels=tuple(str(value) for value in labels),
        cell_icon_ids=tuple(str(icon_id) for icon_id in scene_icon_ids),
        cell_color_pairs=tuple(cell_color_pairs),
        sampled_palette_rgb=tuple(palette),
        panel_geometry=dict(rendered_payload.panel_geometry),
        reference_pair=dict(rendered_payload.reference_pair),
        scene_cells=tuple(dict(item) for item in rendered_payload.scene_cells),
    ), rendered_payload.image


@register_task
class IconsPairGridReferenceColorPairMatchLabelTask:
    """Select the labeled cell that matches the Reference pair colors."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic color-pair match label instance."""

        query_id, query_probabilities, task_params = _select_query(int(instance_seed), params)
        scene_rng = spawn_rng(int(instance_seed), "scene")
        option_count, option_count_probabilities = _resolve_option_count(task_params)
        labels = tuple(str(value) for value in LABEL_POOL_A_L[: int(option_count)])
        answer_label, answer_label_probabilities = _resolve_answer_label(scene_rng, params=task_params, labels=labels)
        render_params = resolve_pair_grid_render_params(
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        pool_manifest = str(task_params.get("pool_manifest", group_default(_GEN_DEFAULTS, "pool_manifest", _DEFAULTS.pool_manifest)))

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = _sample_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    option_count=int(option_count),
                    answer_label=str(answer_label),
                    pool_manifest=str(pool_manifest),
                    render_params=render_params,
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError(f"failed to generate {self.task_id} instance") from last_error

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                "question_text",
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "object_description": str(prompt_defaults["object_description"]),
                "question_text": str(prompt_defaults["question_text"]),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint"]),
                "answer_hint": str(prompt_defaults["answer_hint"]),
                "json_example": str(prompt_defaults["json_example"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        selected_bbox = selected_scene_cell_bbox(
            scene_cells=scene_payload.scene_cells,
            selected_label=str(scene_payload.answer_label),
        )
        annotation_artifacts = bbox_annotation(selected_bbox)
        answer_gt = TypedValue(type="option_letter", value=str(scene_payload.answer_label))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=list(annotation_artifacts["annotation_value"]),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(query_probabilities),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "pool_manifest": str(pool_manifest),
            },
        )

        reference_color_pair = [list(scene_payload.reference_color_pair[0]), list(scene_payload.reference_color_pair[1])]
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_reference_pair_color_pair_match_label",
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "entities": [dict(scene_payload.reference_pair), *[dict(item) for item in scene_payload.scene_cells]],
                "relations": {
                    "selection_target": "same_left_right_color_pair_as_reference",
                    "reference_color_pair_rgb": reference_color_pair,
                    "answer_label": str(scene_payload.answer_label),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": dict(query_spec),
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": pair_grid_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                ),
            },
            "render_map": {
                "image_id": "img0",
                "anchors": {
                    "reference_pair": dict(scene_payload.reference_pair),
                    "answer_label": str(scene_payload.answer_label),
                    "answer_cell_bbox_xyxy": list(selected_bbox),
                    "scene_cells": [dict(item) for item in scene_payload.scene_cells],
                },
            },
            "execution_trace": {
                "scene_variant": "reference_pair_grid",
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "query_id_probabilities": dict(query_probabilities),
                "option_count": int(option_count),
                "option_count_probabilities": dict(option_count_probabilities),
                "answer_label": str(scene_payload.answer_label),
                "answer_label_probabilities": dict(answer_label_probabilities),
                "reference_icon_id": str(scene_payload.reference_icon_id),
                "reference_color_pair_rgb": reference_color_pair,
                "cell_labels": list(scene_payload.cell_labels),
                "cell_icon_ids": list(scene_payload.cell_icon_ids),
                "cell_color_pairs_rgb": [[list(pair[0]), list(pair[1])] for pair in scene_payload.cell_color_pairs],
                "question_format": "select_scene_cell_matching_reference_color_pair",
            },
            "witness_symbolic": {
                "answer_label": str(scene_payload.answer_label),
                "reference_color_pair_rgb": reference_color_pair,
            },
            "projected_annotation": dict(annotation_artifacts["projected_annotation"]),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            query_id=str(query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["IconsPairGridReferenceColorPairMatchLabelTask"]
