"""Sampling helpers for shadow-cause scene state."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant

from .state import (
    DIRECTION_VECTORS,
    OBJECT_SHAPES,
    OPPOSITE_DIRECTION,
    OPTION_LETTERS,
    SCENE_NAMESPACE,
    SHADOW_DIRECTIONS,
    LampSpec,
    ShadowCauseAxes,
    ShadowCauseTaskDefaults,
    ShadowSceneSpec,
)


OBJECT_PALETTES: Tuple[Tuple[int, int, int], ...] = (
    (68, 136, 201),
    (87, 156, 111),
    (191, 114, 73),
    (132, 112, 190),
    (195, 137, 64),
    (89, 150, 157),
)
RENDER_DEFAULT_KEYS: Tuple[str, ...] = (
    "floor_left_px",
    "floor_top_px",
    "floor_right_margin_px",
    "floor_bottom_margin_px",
    "object_center_x_px",
    "object_base_y_px",
    "lamp_radius_x_px",
    "lamp_radius_y_px",
    "lamp_bulb_radius_px",
    "lamp_label_font_size_px",
    "label_stroke_width_px",
    "title_font_size_px",
    "shadow_length_px",
    "shadow_base_width_px",
    "shadow_tip_width_px",
    "object_size_px",
)


def _resolve_variant_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one task-local sampling axis with optional balancing."""

    selected, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), namespace),
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=generation_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=supported,
        balance_flag_key=balance_flag_key,
        explicit_key=explicit_key,
        weights_key=weights_key,
        sampling_namespace=namespace,
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def resolve_shadow_cause_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> ShadowCauseAxes:
    """Resolve all sampled physical and answer axes for the task."""

    correct_option_letter, letter_probs = _resolve_variant_axis(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=OPTION_LETTERS,
        explicit_key="correct_option_letter",
        weights_key="correct_option_letter_weights",
        balance_flag_key="balanced_correct_option_letter_sampling",
        namespace=f"{namespace}.correct_option_letter",
    )
    shadow_direction, shadow_probs = _resolve_variant_axis(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=SHADOW_DIRECTIONS,
        explicit_key="shadow_direction",
        weights_key="shadow_direction_weights",
        balance_flag_key="balanced_shadow_direction_sampling",
        namespace=f"{namespace}.shadow_direction",
    )
    object_shape, shape_probs = _resolve_variant_axis(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        supported=OBJECT_SHAPES,
        explicit_key="object_shape",
        weights_key="object_shape_weights",
        balance_flag_key="balanced_object_shape_sampling",
        namespace=f"{namespace}.object_shape",
    )
    return ShadowCauseAxes(
        correct_option_letter=str(correct_option_letter),
        shadow_direction=str(shadow_direction),
        object_shape=str(object_shape),
        correct_option_letter_probabilities=dict(letter_probs),
        shadow_direction_probabilities=dict(shadow_probs),
        object_shape_probabilities=dict(shape_probs),
    )


def resolve_shadow_render_defaults(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    fallback_defaults: ShadowCauseTaskDefaults,
    instance_seed: int,
    namespace: str,
) -> Dict[str, int]:
    """Resolve all integer rendering defaults for one instance."""

    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(fallback_defaults, key)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in RENDER_DEFAULT_KEYS
    }


def make_shadow_scene_spec(
    *,
    instance_seed: int,
    axes: ShadowCauseAxes,
    render_defaults: Mapping[str, int],
    namespace: str = SCENE_NAMESPACE,
) -> ShadowSceneSpec:
    """Create candidate light-source layout from sampled axes."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene")
    source_direction = OPPOSITE_DIRECTION[str(axes.shadow_direction)]
    distractor_directions = [direction for direction in SHADOW_DIRECTIONS if str(direction) != str(source_direction)]
    if str(axes.shadow_direction) in distractor_directions:
        distractor_directions.remove(str(axes.shadow_direction))
        distractor_directions = [str(axes.shadow_direction)] + distractor_directions
    tail = list(distractor_directions[1:])
    rng.shuffle(tail)
    distractor_directions = [distractor_directions[0], *tail]

    object_center = (
        float(render_defaults["object_center_x_px"]),
        float(render_defaults["object_base_y_px"]),
    )
    radius_x = float(render_defaults["lamp_radius_x_px"])
    radius_y = float(render_defaults["lamp_radius_y_px"])
    lamps: list[LampSpec] = []
    distractor_index = 0
    for label in OPTION_LETTERS:
        if str(label) == str(axes.correct_option_letter):
            direction = str(source_direction)
        else:
            direction = str(distractor_directions[distractor_index])
            distractor_index += 1
        vx, vy = DIRECTION_VECTORS[str(direction)]
        lamps.append(
            LampSpec(
                label=str(label),
                direction=str(direction),
                center_px=(
                    float(object_center[0] + vx * radius_x),
                    float(object_center[1] + vy * radius_y),
                ),
            )
        )
    object_fill_rgb = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.object_palette").choice(OBJECT_PALETTES)
    return ShadowSceneSpec(
        correct_option_letter=str(axes.correct_option_letter),
        shadow_direction=str(axes.shadow_direction),
        source_direction=str(source_direction),
        object_shape=str(axes.object_shape),
        object_fill_rgb=tuple(int(value) for value in object_fill_rgb),
        lamps=tuple(lamps),
    )


def resolve_canvas_size(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    fallback_defaults: ShadowCauseTaskDefaults,
) -> Tuple[int, int]:
    """Resolve canvas dimensions from params, config, and fallbacks."""

    return (
        int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", fallback_defaults.canvas_width))),
        int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", fallback_defaults.canvas_height))),
    )
