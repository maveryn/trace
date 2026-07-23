"""Missing-value task for hydraulic piston diagrams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.bbox_projection import bbox_union_many as _bbox_union
from ...shared.config_defaults import group_default, required_group_defaults, split_generation_rendering_prompt_defaults
from ...shared.drawing import draw_arrow, draw_centered_text, draw_rounded_rect
from ...shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.fixed_query import select_task_query_id
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants
from ...shared.render_variation import resolve_layout_jitter, resolve_render_int
from ...shared.text_rendering import load_font, resolve_text_stroke_fill
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ..shared.diagram_style import prepare_physics_diagram_style_and_background
from ..shared.style import SUPPORTED_PHYSICS_COLOR_NAMES, build_physics_hydraulic_theme
from ..shared.support_sampling import resolve_integer_support
from ..shared.visual_defaults import load_physics_noise_defaults
from .shared.state import (
    HYDRAULIC_SEMANTIC_COLORS,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    HydraulicTaskDefaults,
    RenderedHydraulicScene,
)


TASK_ID = "task_physics__hydraulic__hydraulic_missing_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "missing_output_force",
    "missing_input_force",
    "missing_piston_area",
    "missing_input_area",
)


@dataclass(frozen=True)
class _ResolvedAxes:
    """Resolved scene/query axes and answer support for one instance."""

    scene_variant: str
    query_id: str
    accent_color_name: str
    target_answer: int
    scene_variant_probabilities: Dict[str, float]
    query_id_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _SceneSpec:
    """Symbolic hydraulic scene satisfying Pascal's law exactly."""

    scene_variant: str
    query_id: str
    input_force_value: int
    middle_force_value: int
    output_force_value: int
    input_area_value: int
    middle_area_value: int
    output_area_value: int
    mechanical_advantage: int
    middle_mechanical_advantage: int
    shown_input_force_value: int | None
    shown_output_force_value: int | None
    shown_input_area_value: int | None
    shown_output_area_value: int | None
    target_answer: int
    annotation_entity_ids: Tuple[str, ...]


_DEFAULTS = HydraulicTaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("physics", "hydraulic")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
def _support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Resolve one configured integer support."""

    return resolve_integer_support(params, gen_defaults=_GEN_DEFAULTS, key=str(key), fallback=fallback)


def _target_support_key(query_id: str) -> str:
    """Return the target-answer support key for one query branch."""

    if str(query_id) == "missing_output_force":
        return "output_force_support"
    if str(query_id) == "missing_input_force":
        return "input_force_support"
    if str(query_id) == "missing_input_area":
        return "input_area_support"
    return "output_area_support"


def _target_fallback_support(query_id: str) -> Tuple[int, ...]:
    """Return fallback target-answer support for one query branch."""

    if str(query_id) == "missing_output_force":
        return _DEFAULTS.output_force_support
    if str(query_id) == "missing_input_force":
        return _DEFAULTS.input_force_support
    if str(query_id) == "missing_input_area":
        return _DEFAULTS.input_area_support
    return _DEFAULTS.output_area_support


def _compatible_ratios_for_target(
    *,
    query_id: str,
    target: int,
    input_force_support: Sequence[int],
    input_area_support: Sequence[int],
    output_force_support: Sequence[int],
    output_area_support: Sequence[int],
    ratio_support: Sequence[int],
) -> Tuple[int, ...]:
    """Return mechanical-advantage ratios that can construct one target answer."""

    input_force_set = {int(value) for value in input_force_support}
    input_area_set = {int(value) for value in input_area_support}
    output_force_set = {int(value) for value in output_force_support}
    output_area_set = {int(value) for value in output_area_support}
    target = int(target)
    compatible: list[int] = []
    for raw_ratio in ratio_support:
        ratio = int(raw_ratio)
        if ratio <= 0:
            continue
        if str(query_id) == "missing_output_force":
            force_ok = target % ratio == 0 and int(target // ratio) in input_force_set
            area_ok = any(int(area) * ratio in output_area_set for area in input_area_support)
        elif str(query_id) == "missing_input_force":
            force_ok = target * ratio in output_force_set
            area_ok = any(int(area) * ratio in output_area_set for area in input_area_support)
        elif str(query_id) == "missing_input_area":
            force_ok = any(int(force) * ratio in output_force_set for force in input_force_support)
            area_ok = target * ratio in output_area_set
        else:
            force_ok = any(int(force) * ratio in output_force_set for force in input_force_support)
            area_ok = target % ratio == 0 and int(target // ratio) in input_area_set
        if bool(force_ok) and bool(area_ok):
            compatible.append(int(ratio))
    return tuple(compatible)


def _feasible_target_support(params: Mapping[str, Any], query_id: str) -> Tuple[int, ...]:
    """Return configured target answers that can be sampled for one query branch."""

    target_support = _support(params, _target_support_key(str(query_id)), _target_fallback_support(str(query_id)))
    input_force_support = _support(params, "input_force_support", _DEFAULTS.input_force_support)
    input_area_support = _support(params, "input_area_support", _DEFAULTS.input_area_support)
    output_force_support = _support(params, "output_force_support", _DEFAULTS.output_force_support)
    output_area_support = _support(params, "output_area_support", _DEFAULTS.output_area_support)
    ratio_support = _support(params, "mechanical_advantage_support", _DEFAULTS.mechanical_advantage_support)
    feasible = [
        int(target)
        for target in target_support
        if _compatible_ratios_for_target(
            query_id=str(query_id),
            target=int(target),
            input_force_support=input_force_support,
            input_area_support=input_area_support,
            output_force_support=output_force_support,
            output_area_support=output_area_support,
            ratio_support=ratio_support,
        )
    ]
    if not feasible:
        raise ValueError(f"no feasible target_answer values for hydraulic query {query_id}")
    return tuple(feasible)


def _resolve_target_answer(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    query_id: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve a feasible target answer for one query branch."""

    configured_support = _support(params, _target_support_key(str(query_id)), _target_fallback_support(str(query_id)))
    feasible_support = _feasible_target_support(params, str(query_id))
    explicit = params.get("target_answer")
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(int(value) for value in feasible_support):
            raise ValueError(f"unsupported target_answer for hydraulic query {query_id}: {selected}")
        return int(selected), {
            str(value): (1.0 if int(value) == int(selected) else 0.0)
            for value in configured_support
        }
    if bool(params.get("balanced_target_answer_sampling", group_default(_GEN_DEFAULTS, "balanced_target_answer_sampling", True))):
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{str(query_id)}.target_answer")
        selected = _choose_from_sequence(rng, feasible_support)
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{str(query_id)}.target_answer")
        selected = _choose_from_sequence(rng, feasible_support)
    probability = 1.0 / float(len(feasible_support))
    return int(selected), {
        str(value): (float(probability) if int(value) in set(feasible_support) else 0.0)
        for value in configured_support
    }


def _resolve_axes(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
) -> _ResolvedAxes:
    """Resolve all independently sampled axes for one hydraulic instance."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.axes")
    query_id = str(selected_query_id)
    query_probs = {str(key): float(value) for key, value in query_probabilities.items()}

    scene_params: Mapping[str, Any] = params
    scene_variant, scene_probs = resolve_variant(
        rng,
        params=scene_params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    scene_variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=scene_params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(scene_variant),
        variant_probabilities=scene_probs,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{TASK_ID}.scene_variant",
    )

    accent_color_name, accent_probs = resolve_variant(
        rng,
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
    )
    accent_color_name = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(accent_color_name),
        variant_probabilities=accent_probs,
        supported_variants=SUPPORTED_PHYSICS_COLOR_NAMES,
        balance_flag_key="balanced_accent_color_name_sampling",
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        sampling_namespace=f"{TASK_ID}.accent_color_name",
    )

    target_answer, target_probs = _resolve_target_answer(
        instance_seed=int(instance_seed),
        params=params,
        query_id=str(query_id),
    )
    return _ResolvedAxes(
        scene_variant=str(scene_variant),
        query_id=str(query_id),
        accent_color_name=str(accent_color_name),
        target_answer=int(target_answer),
        scene_variant_probabilities={str(key): float(value) for key, value in sorted(scene_probs.items())},
        query_id_probabilities={str(key): float(value) for key, value in sorted(query_probs.items())},
        accent_color_name_probabilities={str(key): float(value) for key, value in sorted(accent_probs.items())},
        target_answer_probabilities={str(key): float(value) for key, value in sorted(target_probs.items())},
    )


def _choose_from_sequence(rng, values: Sequence[int]) -> int:
    if not values:
        raise ValueError("cannot choose from an empty hydraulic support")
    return int(values[int(rng.randrange(len(values)))])


def _choose_mechanical_advantage(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    compatible: Sequence[int],
) -> Tuple[int, Dict[str, float]]:
    """Choose a mechanical-advantage ratio from the compatible configured support."""

    full_support = _support(params, "mechanical_advantage_support", _DEFAULTS.mechanical_advantage_support)
    compatible_values = [int(value) for value in full_support if int(value) in {int(item) for item in compatible}]
    if not compatible_values:
        raise ValueError("no compatible hydraulic mechanical advantage")
    explicit = params.get("mechanical_advantage")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in compatible_values:
            raise ValueError(f"unsupported mechanical_advantage for hydraulic scene: {selected}")
        return int(selected), {str(value): (1.0 if int(value) == int(selected) else 0.0) for value in full_support}

    if bool(params.get("balanced_mechanical_advantage_sampling", group_default(_GEN_DEFAULTS, "balanced_mechanical_advantage_sampling", True))):
        selected = _choose_from_sequence(rng, compatible_values)
    else:
        selected = _choose_from_sequence(rng, compatible_values)
    probability = 1.0 / float(len(compatible_values))
    return int(selected), {
        str(value): (float(probability) if int(value) in compatible_values else 0.0)
        for value in full_support
    }


def _sample_scene_spec(
    rng,
    *,
    instance_seed: int,
    axes: _ResolvedAxes,
    params: Mapping[str, Any],
) -> _SceneSpec:
    """Sample one exact hydraulic piston relation for the resolved query."""

    input_force_support = _support(params, "input_force_support", _DEFAULTS.input_force_support)
    input_area_support = _support(params, "input_area_support", _DEFAULTS.input_area_support)
    output_force_support = _support(params, "output_force_support", _DEFAULTS.output_force_support)
    output_area_support = _support(params, "output_area_support", _DEFAULTS.output_area_support)
    ratio_support = _support(params, "mechanical_advantage_support", _DEFAULTS.mechanical_advantage_support)
    input_force_set = set(int(value) for value in input_force_support)
    input_area_set = set(int(value) for value in input_area_support)
    output_force_set = set(int(value) for value in output_force_support)
    output_area_set = set(int(value) for value in output_area_support)

    query_id = str(axes.query_id)
    target = int(axes.target_answer)
    if query_id == "missing_output_force":
        compatible_ratios = [
            int(ratio)
            for ratio in ratio_support
            if int(ratio) > 0 and target % int(ratio) == 0 and int(target // int(ratio)) in input_force_set
        ]
        ratio, _ratio_probs = _choose_mechanical_advantage(
            rng,
            instance_seed=int(instance_seed),
            params=params,
            compatible=compatible_ratios,
        )
        input_force = int(target // ratio)
        output_force = int(target)
        compatible_input_areas = [
            int(area)
            for area in input_area_support
            if int(area) * int(ratio) in output_area_set
        ]
        input_area = _choose_from_sequence(rng, compatible_input_areas)
        output_area = int(input_area * ratio)
        shown_input_force = int(input_force)
        shown_output_force = None
        shown_input_area = int(input_area)
        shown_output_area = int(output_area)
    elif query_id == "missing_input_force":
        compatible_ratios = [
            int(ratio)
            for ratio in ratio_support
            if int(target) * int(ratio) in output_force_set
        ]
        ratio, _ratio_probs = _choose_mechanical_advantage(
            rng,
            instance_seed=int(instance_seed),
            params=params,
            compatible=compatible_ratios,
        )
        input_force = int(target)
        output_force = int(target * ratio)
        compatible_input_areas = [
            int(area)
            for area in input_area_support
            if int(area) * int(ratio) in output_area_set
        ]
        input_area = _choose_from_sequence(rng, compatible_input_areas)
        output_area = int(input_area * ratio)
        shown_input_force = None
        shown_output_force = int(output_force)
        shown_input_area = int(input_area)
        shown_output_area = int(output_area)
    elif query_id == "missing_piston_area":
        compatible_ratios = [
            int(ratio)
            for ratio in ratio_support
            if int(ratio) > 0 and target % int(ratio) == 0 and int(target // int(ratio)) in input_area_set
        ]
        ratio, _ratio_probs = _choose_mechanical_advantage(
            rng,
            instance_seed=int(instance_seed),
            params=params,
            compatible=compatible_ratios,
        )
        output_area = int(target)
        input_area = int(target // ratio)
        compatible_input_forces = [
            int(force)
            for force in input_force_support
            if int(force) * int(ratio) in output_force_set
        ]
        input_force = _choose_from_sequence(rng, compatible_input_forces)
        output_force = int(input_force * ratio)
        shown_input_force = int(input_force)
        shown_output_force = int(output_force)
        shown_input_area = int(input_area)
        shown_output_area = None
    else:
        compatible_ratios = [
            int(ratio)
            for ratio in ratio_support
            if int(ratio) > 0 and int(target) * int(ratio) in output_area_set
        ]
        ratio, _ratio_probs = _choose_mechanical_advantage(
            rng,
            instance_seed=int(instance_seed),
            params=params,
            compatible=compatible_ratios,
        )
        input_area = int(target)
        output_area = int(target * ratio)
        compatible_input_forces = [
            int(force)
            for force in input_force_support
            if int(force) * int(ratio) in output_force_set
        ]
        input_force = _choose_from_sequence(rng, compatible_input_forces)
        output_force = int(input_force * ratio)
        shown_input_force = int(input_force)
        shown_output_force = int(output_force)
        shown_input_area = None
        shown_output_area = int(output_area)

    output_ratio = int(ratio)
    middle_ratio_candidates = [
        int(candidate_ratio)
        for candidate_ratio in ratio_support
        if int(input_area) * int(candidate_ratio) in output_area_set
        and int(input_force) * int(candidate_ratio) in output_force_set
    ]
    preferred_middle_ratios = [int(candidate_ratio) for candidate_ratio in middle_ratio_candidates if int(candidate_ratio) != output_ratio]
    if not preferred_middle_ratios:
        preferred_middle_ratios = list(middle_ratio_candidates)
    middle_ratio = _choose_from_sequence(rng, preferred_middle_ratios)
    middle_area = int(input_area * middle_ratio)
    middle_force = int(input_force * middle_ratio)

    return _SceneSpec(
        scene_variant=str(axes.scene_variant),
        query_id=str(query_id),
        input_force_value=int(input_force),
        middle_force_value=int(middle_force),
        output_force_value=int(output_force),
        input_area_value=int(input_area),
        middle_area_value=int(middle_area),
        output_area_value=int(output_area),
        mechanical_advantage=int(ratio),
        middle_mechanical_advantage=int(middle_ratio),
        shown_input_force_value=shown_input_force,
        shown_output_force_value=shown_output_force,
        shown_input_area_value=shown_input_area,
        shown_output_area_value=shown_output_area,
        target_answer=int(target),
        annotation_entity_ids=_annotation_entity_ids_for_query(str(query_id)),
    )


def _annotation_entity_ids_for_query(query_id: str) -> Tuple[str, ...]:
    """Return stable prompt-facing side witnesses for one hydraulic query."""

    _ = str(query_id)
    return ("input_side", "output_side")


def _annotation_entity_key_map_for_query(query_id: str) -> Dict[str, str]:
    """Return stable semantic annotation keys by rendered entity id."""

    _ = str(query_id)
    return {
        "input_side": "input_side",
        "output_side": "output_side",
    }


def _draw_text_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    padding_px: int,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    text_rgb: Sequence[int],
    stroke_width_px: int,
    radius_px: int = 10,
) -> List[float]:
    """Draw one labeled tag and return its bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width_px)))
    width = float(text_bbox[2] - text_bbox[0] + (2 * int(padding_px)))
    height = float(text_bbox[3] - text_bbox[1] + (2 * int(padding_px)))
    cx, cy = float(center[0]), float(center[1])
    tag_bbox = [
        round(float(cx - (width / 2.0)), 3),
        round(float(cy - (height / 2.0)), 3),
        round(float(cx + (width / 2.0)), 3),
        round(float(cy + (height / 2.0)), 3),
    ]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in tag_bbox),
        radius=int(radius_px),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=2,
    )
    text_drawn_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=max(1, int(stroke_width_px)),
    )
    return _bbox_union(tag_bbox, text_drawn_bbox)


def _draw_texture(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    line_rgb: Sequence[int],
    spacing_px: int,
    width_px: int,
) -> None:
    """Draw a subtle deterministic texture inside a rectangular area."""

    left, top, right, bottom = [float(value) for value in bbox]
    step = max(10, int(spacing_px))
    x = float(left - (bottom - top))
    while x < float(right):
        draw.line(
            [(float(x), float(bottom)), (float(x + (bottom - top)), float(top))],
            fill=tuple(int(value) for value in line_rgb),
            width=max(1, int(width_px)),
        )
        x += float(step)


def _draw_chamber(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    piston_y: float,
    theme,
    render_defaults: Mapping[str, int],
    textured: bool,
) -> Tuple[List[float], List[float]]:
    """Draw one hydraulic chamber and return chamber and piston bboxes."""

    left, top, right, bottom = [float(value) for value in bbox]
    radius = int(render_defaults["chamber_corner_radius_px"])
    outline_width = int(render_defaults["chamber_outline_width_px"])
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in bbox),
        radius=radius,
        fill=tuple(int(value) for value in theme.chamber_fill_rgb),
        outline=tuple(int(value) for value in theme.chamber_outline_rgb),
        width=outline_width,
    )
    fluid_bbox = [
        round(float(left + outline_width), 3),
        round(float(piston_y + int(render_defaults["piston_height_px"])), 3),
        round(float(right - outline_width), 3),
        round(float(bottom - outline_width), 3),
    ]
    draw.rectangle(
        tuple(float(value) for value in fluid_bbox),
        fill=tuple(int(value) for value in theme.fluid_fill_rgb),
    )
    if textured:
        _draw_texture(
            draw,
            bbox=fluid_bbox,
            line_rgb=tuple(int(value) for value in theme.texture_rgb),
            spacing_px=int(render_defaults["texture_spacing_px"]),
            width_px=int(render_defaults["texture_line_width_px"]),
        )
    piston_bbox = [
        round(float(left + 8.0), 3),
        round(float(piston_y), 3),
        round(float(right - 8.0), 3),
        round(float(piston_y + int(render_defaults["piston_height_px"])), 3),
    ]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in piston_bbox),
        radius=8,
        fill=tuple(int(value) for value in theme.piston_fill_rgb),
        outline=tuple(int(value) for value in theme.piston_outline_rgb),
        width=int(render_defaults["piston_outline_width_px"]),
    )
    return [round(float(value), 3) for value in bbox], list(piston_bbox)


def _hydraulic_base_geometry(
    *,
    render_defaults: Mapping[str, int],
    scene_variant: str,
) -> Tuple[Tuple[float, float, float], float, float]:
    """Return base chamber centers/top/height before whole-diagram placement."""

    chamber_top = float(render_defaults["chamber_top_px"])
    chamber_height = float(render_defaults["chamber_height_px"])
    if str(scene_variant) == "compact_frame":
        centers = (245.0, 560.0, 875.0)
        chamber_top += 12.0
        chamber_height -= 18.0
    elif str(scene_variant) == "tall_columns":
        centers = (230.0, 560.0, 890.0)
        chamber_top -= 34.0
        chamber_height += 42.0
    else:
        centers = (230.0, 560.0, 890.0)
    return centers, float(chamber_top), float(chamber_height)


def _hydraulic_content_bbox(
    *,
    render_defaults: Mapping[str, int],
    scene_spec: _SceneSpec,
) -> List[float]:
    """Return a conservative bbox for the whole hydraulic diagram before placement."""

    centers, chamber_top, chamber_height = _hydraulic_base_geometry(
        render_defaults=render_defaults,
        scene_variant=str(scene_spec.scene_variant),
    )
    left_center_x, _middle_center_x, right_center_x = [float(value) for value in centers]
    chamber_bottom = float(chamber_top + chamber_height)
    area_scale = int(render_defaults["chamber_area_scale_px"])
    chamber_min_width = int(render_defaults["chamber_min_width_px"])
    left_width = float(chamber_min_width + (int(scene_spec.input_area_value) * area_scale))
    right_width = float(chamber_min_width + (int(scene_spec.output_area_value) * area_scale))
    left = min(float(left_center_x - (left_width / 2.0) - 42.0), float(left_center_x - 100.0))
    right = max(float(right_center_x + (right_width / 2.0) + 28.0), float(right_center_x + 152.0))
    top = min(
        float(chamber_top - 152.0),
        float(chamber_top + int(render_defaults["fluid_top_gap_px"]) - int(render_defaults["piston_height_px"]) - int(render_defaults["force_arrow_length_px"]) - 32.0),
    )
    bottom = float(chamber_bottom + 78.0)
    return [round(float(left), 3), round(float(top), 3), round(float(right), 3), round(float(bottom), 3)]


def _resolve_hydraulic_layout_placement(
    *,
    render_defaults: Mapping[str, int],
    params: Mapping[str, Any],
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
    scene_spec: _SceneSpec,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Resolve whole-diagram placement before rendering and annotation projection."""

    content_bbox = _hydraulic_content_bbox(render_defaults=render_defaults, scene_spec=scene_spec)
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        _RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.hydraulic_layout",
    )
    min_margin = int(jitter.get("min_margin_px", 18))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if int(min_dx) > int(max_dx):
        min_dx = 0
        max_dx = 0
    if int(min_dy) > int(max_dy):
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))
    adjusted = dict(render_defaults)
    adjusted["layout_offset_x_px"] = int(dx)
    adjusted["layout_offset_y_px"] = int(dy)

    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_hydraulic_diagram_offset",
            "content_bbox_px": list(content_bbox),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "default_origin_px": [round(float(content_left), 3), round(float(content_top), 3)],
            "final_origin_px": [round(float(content_left) + float(dx), 3), round(float(content_top) + float(dy), 3)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement


def _draw_force_label(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    piston_top_y: float,
    label_text: str,
    missing: bool,
    theme,
    render_defaults: Mapping[str, int],
    font_family: str,
) -> List[float]:
    """Draw a downward force arrow and label, returning the prompt-facing bbox."""

    arrow_length = int(render_defaults["force_arrow_length_px"])
    arrow_width = int(render_defaults["force_arrow_width_px"])
    start = (float(center_x), float(piston_top_y - arrow_length - 10.0))
    end = (float(center_x), float(piston_top_y - 8.0))
    draw_arrow(
        draw,
        start=start,
        end=end,
        fill=tuple(int(value) for value in theme.force_rgb),
        width=max(3, int(arrow_width)),
        head_length_px=20.0,
        head_width_px=18.0,
    )
    arrow_bbox = [
        round(float(center_x - 12.0), 3),
        round(float(start[1]), 3),
        round(float(center_x + 12.0), 3),
        round(float(end[1]), 3),
    ]
    tag_bbox = _draw_text_tag(
        draw,
        text=str(label_text),
        center=(float(center_x + 74.0), float(start[1] + 32.0)),
        font=load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=str(font_family)),
        padding_px=int(render_defaults["label_padding_px"]),
        fill_rgb=tuple(int(value) for value in theme.missing_fill_rgb)
        if missing
        else tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.missing_outline_rgb)
        if missing
        else tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.missing_text_rgb)
        if missing
        else tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    return _bbox_union(arrow_bbox, tag_bbox)


def _render_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, int],
    accent_color_name: str,
    scene_spec: _SceneSpec,
    font_family: str,
    diagram_style: Any | None = None,
) -> RenderedHydraulicScene:
    """Render the full hydraulic piston diagram and preserve label boxes."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_physics_hydraulic_theme(str(accent_color_name), diagram_style=diagram_style)
    canvas_width, canvas_height = image.size
    centers, chamber_top, chamber_height = _hydraulic_base_geometry(
        render_defaults=render_defaults,
        scene_variant=str(scene_spec.scene_variant),
    )
    dx = int(render_defaults.get("layout_offset_x_px", 0))
    dy = int(render_defaults.get("layout_offset_y_px", 0))
    left_center_x, middle_center_x, right_center_x = [float(value) + float(dx) for value in centers]
    chamber_top = float(chamber_top) + float(dy)

    chamber_bottom = float(chamber_top + chamber_height)
    area_scale = int(render_defaults["chamber_area_scale_px"])
    chamber_min_width = int(render_defaults["chamber_min_width_px"])
    left_width = float(chamber_min_width + (int(scene_spec.input_area_value) * area_scale))
    middle_width = float(chamber_min_width + (int(scene_spec.middle_area_value) * area_scale))
    right_width = float(chamber_min_width + (int(scene_spec.output_area_value) * area_scale))
    left_bbox = [
        round(float(left_center_x - (left_width / 2.0)), 3),
        round(float(chamber_top), 3),
        round(float(left_center_x + (left_width / 2.0)), 3),
        round(float(chamber_bottom), 3),
    ]
    middle_bbox = [
        round(float(middle_center_x - (middle_width / 2.0)), 3),
        round(float(chamber_top), 3),
        round(float(middle_center_x + (middle_width / 2.0)), 3),
        round(float(chamber_bottom), 3),
    ]
    right_bbox = [
        round(float(right_center_x - (right_width / 2.0)), 3),
        round(float(chamber_top), 3),
        round(float(right_center_x + (right_width / 2.0)), 3),
        round(float(chamber_bottom), 3),
    ]
    piston_y = float(chamber_top + int(render_defaults["fluid_top_gap_px"]) - int(render_defaults["piston_height_px"]))
    pipe_height = int(render_defaults["pipe_height_px"])
    pipe_y = float(chamber_bottom - pipe_height - 18.0)
    pipe_bbox = [
        round(float(left_center_x), 3),
        round(float(pipe_y), 3),
        round(float(right_center_x), 3),
        round(float(pipe_y + pipe_height), 3),
    ]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in pipe_bbox),
        radius=int(pipe_height // 2),
        fill=tuple(int(value) for value in theme.pipe_fill_rgb),
        outline=tuple(int(value) for value in theme.pipe_outline_rgb),
        width=int(render_defaults["pipe_outline_width_px"]),
    )
    if str(scene_spec.scene_variant) == "compact_frame":
        frame_bbox = [
            round(float(left_bbox[0] - 44.0), 3),
            round(float(chamber_top - 22.0), 3),
            round(float(right_bbox[2] + 44.0), 3),
            round(float(chamber_bottom + 28.0), 3),
        ]
        draw.rounded_rectangle(
            tuple(float(value) for value in frame_bbox),
            radius=18,
            outline=tuple(int(value) for value in theme.texture_rgb),
            width=2,
        )

    left_chamber_bbox, left_piston_bbox = _draw_chamber(
        draw,
        bbox=left_bbox,
        piston_y=float(piston_y),
        theme=theme,
        render_defaults=render_defaults,
        textured=str(scene_spec.scene_variant) == "tall_columns",
    )
    middle_chamber_bbox, middle_piston_bbox = _draw_chamber(
        draw,
        bbox=middle_bbox,
        piston_y=float(piston_y),
        theme=theme,
        render_defaults=render_defaults,
        textured=str(scene_spec.scene_variant) == "tall_columns",
    )
    right_chamber_bbox, right_piston_bbox = _draw_chamber(
        draw,
        bbox=right_bbox,
        piston_y=float(piston_y),
        theme=theme,
        render_defaults=render_defaults,
        textured=str(scene_spec.scene_variant) == "tall_columns",
    )

    small_font = load_font(int(render_defaults["small_label_font_size_px"]), bold=True, font_family=str(font_family))
    for label, center_x in (("input", left_center_x), ("reference", middle_center_x), ("output", right_center_x)):
        draw_centered_text(
            draw,
            text=label,
            center=(float(center_x), float(chamber_top - 128.0)),
            font=small_font,
            fill=tuple(int(value) for value in theme.label_text_rgb),
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.label_text_rgb)),
            stroke_width=max(1, int(render_defaults["label_stroke_width_px"])),
        )

    left_force_missing = scene_spec.shown_input_force_value is None
    right_force_missing = scene_spec.shown_output_force_value is None
    left_force_label = "F = ? N" if left_force_missing else f"F = {int(scene_spec.shown_input_force_value)} N"
    middle_force_label = f"F = {int(scene_spec.middle_force_value)} N"
    right_force_label = "F = ? N" if right_force_missing else f"F = {int(scene_spec.shown_output_force_value)} N"
    left_force_bbox = _draw_force_label(
        draw,
        center_x=float(left_center_x),
        piston_top_y=float(left_piston_bbox[1]),
        label_text=left_force_label,
        missing=bool(left_force_missing),
        theme=theme,
        render_defaults=render_defaults,
        font_family=str(font_family),
    )
    middle_force_bbox = _draw_force_label(
        draw,
        center_x=float(middle_center_x),
        piston_top_y=float(middle_piston_bbox[1]),
        label_text=middle_force_label,
        missing=False,
        theme=theme,
        render_defaults=render_defaults,
        font_family=str(font_family),
    )
    right_force_bbox = _draw_force_label(
        draw,
        center_x=float(right_center_x),
        piston_top_y=float(right_piston_bbox[1]),
        label_text=right_force_label,
        missing=bool(right_force_missing),
        theme=theme,
        render_defaults=render_defaults,
        font_family=str(font_family),
    )

    area_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=str(font_family))
    left_area_missing = scene_spec.shown_input_area_value is None
    left_area_bbox = _draw_text_tag(
        draw,
        text="A = ? cm^2" if left_area_missing else f"A = {int(scene_spec.shown_input_area_value)} cm^2",
        center=(float(left_center_x), float(chamber_bottom + 44.0)),
        font=area_font,
        padding_px=int(render_defaults["label_padding_px"]),
        fill_rgb=tuple(int(value) for value in theme.missing_fill_rgb)
        if left_area_missing
        else tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.missing_outline_rgb)
        if left_area_missing
        else tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.missing_text_rgb)
        if left_area_missing
        else tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    middle_area_bbox = _draw_text_tag(
        draw,
        text=f"A = {int(scene_spec.middle_area_value)} cm^2",
        center=(float(middle_center_x), float(chamber_bottom + 44.0)),
        font=area_font,
        padding_px=int(render_defaults["label_padding_px"]),
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    right_area_missing = scene_spec.shown_output_area_value is None
    right_area_bbox = _draw_text_tag(
        draw,
        text="A = ? cm^2" if right_area_missing else f"A = {int(scene_spec.shown_output_area_value)} cm^2",
        center=(float(right_center_x), float(chamber_bottom + 44.0)),
        font=area_font,
        padding_px=int(render_defaults["label_padding_px"]),
        fill_rgb=tuple(int(value) for value in theme.missing_fill_rgb)
        if right_area_missing
        else tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.missing_outline_rgb)
        if right_area_missing
        else tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.missing_text_rgb)
        if right_area_missing
        else tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    input_side_bbox = _bbox_union(left_force_bbox, left_chamber_bbox, left_area_bbox, padding=8.0)
    output_side_bbox = _bbox_union(right_force_bbox, right_chamber_bbox, right_area_bbox, padding=8.0)

    entity_bboxes = {
        "input_side": list(input_side_bbox),
        "output_side": list(output_side_bbox),
        "left_force_label": list(left_force_bbox),
        "middle_force_label": list(middle_force_bbox),
        "right_force_label": list(right_force_bbox),
        "left_area_label": list(left_area_bbox),
        "middle_area_label": list(middle_area_bbox),
        "right_area_label": list(right_area_bbox),
        "left_chamber": list(left_chamber_bbox),
        "middle_chamber": list(middle_chamber_bbox),
        "right_chamber": list(right_chamber_bbox),
        "fluid_pipe": list(pipe_bbox),
    }
    scene_entities = [
        {
            "entity_id": str(entity_id),
            "entity_type": str(entity_id),
            "bbox_px": list(bbox),
            "meta": {},
        }
        for entity_id, bbox in entity_bboxes.items()
    ]
    for entity in scene_entities:
        if entity["entity_id"] == "left_force_label":
            entity["meta"] = {
                "shown_value": None
                if scene_spec.shown_input_force_value is None
                else int(scene_spec.shown_input_force_value),
                "true_value": int(scene_spec.input_force_value),
                "unit": "N",
            }
        elif entity["entity_id"] == "middle_force_label":
            entity["meta"] = {
                "shown_value": int(scene_spec.middle_force_value),
                "true_value": int(scene_spec.middle_force_value),
                "unit": "N",
            }
        elif entity["entity_id"] == "right_force_label":
            entity["meta"] = {
                "shown_value": None
                if scene_spec.shown_output_force_value is None
                else int(scene_spec.shown_output_force_value),
                "true_value": int(scene_spec.output_force_value),
                "unit": "N",
            }
        elif entity["entity_id"] == "left_area_label":
            entity["meta"] = {
                "shown_value": None
                if scene_spec.shown_input_area_value is None
                else int(scene_spec.shown_input_area_value),
                "true_value": int(scene_spec.input_area_value),
                "unit": "cm^2",
            }
        elif entity["entity_id"] == "middle_area_label":
            entity["meta"] = {
                "shown_value": int(scene_spec.middle_area_value),
                "true_value": int(scene_spec.middle_area_value),
                "unit": "cm^2",
            }
        elif entity["entity_id"] == "right_area_label":
            entity["meta"] = {
                "shown_value": None
                if scene_spec.shown_output_area_value is None
                else int(scene_spec.shown_output_area_value),
                "true_value": int(scene_spec.output_area_value),
                "unit": "cm^2",
            }

    annotation_key_by_entity_id = _annotation_entity_key_map_for_query(str(scene_spec.query_id))
    annotation_bboxes: List[List[float]] = []
    annotation_bbox_map: Dict[str, List[float]] = {}
    for entity_id in scene_spec.annotation_entity_ids:
        annotation_key = str(annotation_key_by_entity_id[str(entity_id)])
        bbox = list(entity_bboxes[str(entity_id)])
        annotation_bboxes.append(list(bbox))
        annotation_bbox_map[annotation_key] = list(bbox)
    render_map = {
        "accent_color_name": str(accent_color_name),
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "left_chamber_bbox_px": list(left_chamber_bbox),
        "middle_chamber_bbox_px": list(middle_chamber_bbox),
        "right_chamber_bbox_px": list(right_chamber_bbox),
        "fluid_pipe_bbox_px": list(pipe_bbox),
        "left_force_label_bbox_px": list(left_force_bbox),
        "middle_force_label_bbox_px": list(middle_force_bbox),
        "right_force_label_bbox_px": list(right_force_bbox),
        "left_area_label_bbox_px": list(left_area_bbox),
        "middle_area_label_bbox_px": list(middle_area_bbox),
        "right_area_label_bbox_px": list(right_area_bbox),
        "input_side_bbox_px": list(input_side_bbox),
        "output_side_bbox_px": list(output_side_bbox),
        "input_force_value": int(scene_spec.input_force_value),
        "middle_force_value": int(scene_spec.middle_force_value),
        "output_force_value": int(scene_spec.output_force_value),
        "input_area_value": int(scene_spec.input_area_value),
        "middle_area_value": int(scene_spec.middle_area_value),
        "output_area_value": int(scene_spec.output_area_value),
        "shown_input_area_value": None
        if scene_spec.shown_input_area_value is None
        else int(scene_spec.shown_input_area_value),
        "shown_output_area_value": None
        if scene_spec.shown_output_area_value is None
        else int(scene_spec.shown_output_area_value),
        "mechanical_advantage": int(scene_spec.mechanical_advantage),
        "middle_mechanical_advantage": int(scene_spec.middle_mechanical_advantage),
        "annotation_entity_ids": [str(entity_id) for entity_id in scene_spec.annotation_entity_ids],
        "annotation_key_by_entity_id": dict(annotation_key_by_entity_id),
        "annotation_keyed_bboxes_px": {str(key): list(bbox) for key, bbox in annotation_bbox_map.items()},
        "annotation_bboxes_px": [list(bbox) for bbox in annotation_bboxes],
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
    }
    return RenderedHydraulicScene(
        image=image,
        annotation_bboxes=[list(bbox) for bbox in annotation_bboxes],
        annotation_bbox_map={str(key): list(bbox) for key, bbox in annotation_bbox_map.items()},
        annotation_entity_ids=[str(entity_id) for entity_id in scene_spec.annotation_entity_ids],
        annotation_key_by_entity_id={str(key): str(value) for key, value in annotation_key_by_entity_id.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )


@register_task
class PhysicsHydraulicMissingValueTask:
    """Return one missing-value question over a hydraulic piston diagram."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one hydraulic diagram and bind the missing-value contract."""

        selected_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        params = task_params
        axes = _resolve_axes(
            int(instance_seed),
            params=params,
            selected_query_id=str(selected_query_id),
            query_probabilities=query_probabilities,
        )
        for attempt_index in range(max(1, int(max_attempts))):
            attempt_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.attempt.{int(attempt_index)}")
            scene_spec = _sample_scene_spec(
                attempt_rng,
                instance_seed=int(instance_seed),
                axes=axes,
                params=params,
            )
            canvas_width = int(params.get("canvas_width", group_default(_RENDER_DEFAULTS, "canvas_width", _DEFAULTS.canvas_width)))
            canvas_height = int(params.get("canvas_height", group_default(_RENDER_DEFAULTS, "canvas_height", _DEFAULTS.canvas_height)))
            background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
                scene_id=SCENE_ID,
                canvas_width=int(canvas_width),
                canvas_height=int(canvas_height),
                instance_seed=int(instance_seed),
                params=params,
                protected_colors=HYDRAULIC_SEMANTIC_COLORS,
            )
            font_family = sample_font_family(
                role="readout",
                instance_seed=int(instance_seed),
                namespace=f"{TASK_ID}.render.font",
                params=params,
            )
            font_record = get_font_family_record(str(font_family))
            render_defaults = {
                key: resolve_render_int(
                    params,
                    _RENDER_DEFAULTS,
                    key,
                    int(getattr(_DEFAULTS, key)),
                    instance_seed=int(instance_seed),
                    namespace=TASK_ID,
                )
                for key in (
                    "chamber_top_px",
                    "chamber_height_px",
                    "chamber_min_width_px",
                    "chamber_area_scale_px",
                    "chamber_corner_radius_px",
                    "chamber_outline_width_px",
                    "piston_height_px",
                    "piston_outline_width_px",
                    "fluid_top_gap_px",
                    "pipe_height_px",
                    "pipe_outline_width_px",
                    "force_arrow_length_px",
                    "force_arrow_width_px",
                    "label_font_size_px",
                    "small_label_font_size_px",
                    "label_padding_px",
                    "label_stroke_width_px",
                    "texture_line_width_px",
                    "texture_spacing_px",
                )
            }
            render_defaults, layout_placement_meta = _resolve_hydraulic_layout_placement(
                render_defaults=render_defaults,
                params=params,
                instance_seed=int(instance_seed),
                canvas_width=int(canvas_width),
                canvas_height=int(canvas_height),
                scene_spec=scene_spec,
            )
            rendered_scene = _render_scene(
                background=background,
                render_defaults=render_defaults,
                accent_color_name=str(axes.accent_color_name),
                scene_spec=scene_spec,
                font_family=str(font_family),
                diagram_style=diagram_style,
            )
            image, post_noise_meta = apply_post_image_noise(
                rendered_scene.image,
                instance_seed=int(instance_seed),
                params=params,
                default_config=POST_IMAGE_NOISE_DEFAULTS,
            )

            prompt_defaults = required_group_defaults(
                _PROMPT_DEFAULTS,
                (
                    "bundle_id",
                    "task_key",
                ),
                context=f"prompt defaults for {self.task_id}",
            )
            prompt_selection = render_scene_prompt_variants(
                domain=self.domain,
                scene_id=SCENE_ID,
                bundle_id=str(prompt_defaults["bundle_id"]),
                scene_key="hydraulic_piston_diagram",
                task_key=str(prompt_defaults["task_key"]),
                query_key=str(axes.query_id),
                answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
                dynamic_slots={},
                instance_seed=int(instance_seed),
            )
            prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

            answer_gt = TypedValue(type="integer", value=int(axes.target_answer))
            annotation_gt = TypedValue(
                type="bbox_map",
                value={str(key): list(bbox) for key, bbox in rendered_scene.annotation_bbox_map.items()},
            )
            trace_payload = {
                "scene_ir": {
                    "scene_kind": f"physics_hydraulic_piston_{str(axes.scene_variant)}",
                    "entities": [dict(entity) for entity in rendered_scene.scene_entities],
                    "relations": {
                        "scene_variant": str(axes.scene_variant),
                        "query_id": str(axes.query_id),
                        "accent_color_name": str(axes.accent_color_name),
                        "input_force_value": int(scene_spec.input_force_value),
                        "middle_force_value": int(scene_spec.middle_force_value),
                        "output_force_value": int(scene_spec.output_force_value),
                        "input_area_value": int(scene_spec.input_area_value),
                        "middle_area_value": int(scene_spec.middle_area_value),
                        "output_area_value": int(scene_spec.output_area_value),
                        "mechanical_advantage": int(scene_spec.mechanical_advantage),
                        "middle_mechanical_advantage": int(scene_spec.middle_mechanical_advantage),
                        "target_answer": int(axes.target_answer),
                        "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                        "annotation_key_by_entity_id": dict(rendered_scene.annotation_key_by_entity_id),
                    },
                },
                "query_spec": {
                    "query_id": str(axes.query_id),
                    "template_id": str(prompt_defaults["bundle_id"]),
                    "prompt_variant": dict(prompt_artifacts.prompt_variant),
                    "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                    "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                    "params": {
                        "scene_variant": str(axes.scene_variant),
                        "query_id": str(axes.query_id),
                        "accent_color_name": str(axes.accent_color_name),
                        "target_answer": int(axes.target_answer),
                        "scene_variant_probabilities": dict(axes.scene_variant_probabilities),
                        "query_id_probabilities": dict(axes.query_id_probabilities),
                        "accent_color_name_probabilities": dict(axes.accent_color_name_probabilities),
                        "target_answer_probabilities": dict(axes.target_answer_probabilities),
                    },
                },
                "render_spec": {
                    "scene_variant": str(axes.scene_variant),
                    "canvas_width": int(image.size[0]),
                    "canvas_height": int(image.size[1]),
                    "accent_color_name": str(axes.accent_color_name),
                    "font": {
                        "font_family": str(font_family),
                        "font_asset_version": font_asset_version(),
                        "font_asset": font_record.to_trace(),
                        "scope": "hydraulic_piston_diagram",
                        "selection_policy": {
                            "pool": "global_approved_font_pool",
                            "include_tags": [],
                            "exclude_tags": [],
                            "exclusion_reason": "",
                        },
                    },
                    "technical_diagram_style": dict(diagram_style_meta),
                    "background_style": background_meta,
                    "layout_placement": dict(layout_placement_meta),
                    "post_image_noise": post_noise_meta,
                },
                "render_map": dict(rendered_scene.render_map),
                "execution_trace": {
                    "scene_variant": str(axes.scene_variant),
                    "query_id": str(axes.query_id),
                    "accent_color_name": str(axes.accent_color_name),
                    "input_force_value": int(scene_spec.input_force_value),
                    "middle_force_value": int(scene_spec.middle_force_value),
                    "output_force_value": int(scene_spec.output_force_value),
                    "input_area_value": int(scene_spec.input_area_value),
                    "middle_area_value": int(scene_spec.middle_area_value),
                    "output_area_value": int(scene_spec.output_area_value),
                    "mechanical_advantage": int(scene_spec.mechanical_advantage),
                    "middle_mechanical_advantage": int(scene_spec.middle_mechanical_advantage),
                    "shown_input_force_value": None
                    if scene_spec.shown_input_force_value is None
                    else int(scene_spec.shown_input_force_value),
                    "shown_output_force_value": None
                    if scene_spec.shown_output_force_value is None
                    else int(scene_spec.shown_output_force_value),
                    "shown_input_area_value": None
                    if scene_spec.shown_input_area_value is None
                    else int(scene_spec.shown_input_area_value),
                    "shown_output_area_value": None
                    if scene_spec.shown_output_area_value is None
                    else int(scene_spec.shown_output_area_value),
                    "target_answer": int(axes.target_answer),
                    "target_answer_support": list(
                        _support(params, _target_support_key(str(axes.query_id)), _target_fallback_support(str(axes.query_id)))
                    ),
                    "input_force_support": list(_support(params, "input_force_support", _DEFAULTS.input_force_support)),
                    "input_area_support": list(_support(params, "input_area_support", _DEFAULTS.input_area_support)),
                    "mechanical_advantage_support": list(
                        _support(params, "mechanical_advantage_support", _DEFAULTS.mechanical_advantage_support)
                    ),
                    "output_force_support": list(_support(params, "output_force_support", _DEFAULTS.output_force_support)),
                    "output_area_support": list(_support(params, "output_area_support", _DEFAULTS.output_area_support)),
                    "annotation_entity_ids": list(rendered_scene.annotation_entity_ids),
                    "annotation_key_by_entity_id": dict(rendered_scene.annotation_key_by_entity_id),
                },
                "witness_symbolic": {
                    "type": "object_key_map",
                    "ids": [str(item) for item in rendered_scene.annotation_entity_ids],
                    "keys": dict(rendered_scene.annotation_key_by_entity_id),
                },
                "projected_annotation": {
                    "type": "bbox_map",
                    "bbox_map": {str(key): list(bbox) for key, bbox in rendered_scene.annotation_bbox_map.items()},
                    "pixel_bbox_map": {
                        str(key): list(bbox) for key, bbox in rendered_scene.annotation_bbox_map.items()
                    },
                },
                "background": background_meta,
                "post_image_noise": post_noise_meta,
            }
            return TaskOutput(
                prompt=str(prompt_artifacts.prompt),
                prompt_variants=dict(prompt_artifacts.prompt_variants),
                answer_gt=answer_gt,
                annotation_gt=annotation_gt,
                image=image,
                image_id="img0",
                trace_payload=trace_payload,
                task_versions=default_task_versions(),
                scene_id=SCENE_ID,
                query_id=str(axes.query_id),
            )

        raise RuntimeError(f"{self.task_id} failed to generate a valid scene after {max_attempts} attempts")


__all__ = ["PhysicsHydraulicMissingValueTask"]
