"""Sampling helpers for arithmetic-constraint puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
)
from trace_tasks.tasks.shared.support_sampling import (
    resolve_integer_choice,
    resolve_integer_support,
)
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

from .defaults import FALLBACK_GENERATION_DEFAULTS, FALLBACK_RENDERING_DEFAULTS
from .state import ArithmeticRenderParams, SUPPORTED_SCENE_VARIANTS


@dataclass(frozen=True)
class ResolvedSceneAxes:
    """Resolved visual axes for one arithmetic puzzle instance."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedIntegerTarget:
    """Resolved integer answer support for one arithmetic objective."""

    answer_value: int
    answer_support: Tuple[int, ...]
    answer_probabilities: Dict[str, float]


def resolve_scene_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> ResolvedSceneAxes:
    """Resolve the scene chrome variant without using public task identity."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_axes")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}:scene_variant",
    )
    return ResolvedSceneAxes(
        scene_variant=str(selected),
        scene_variant_probabilities={
            str(key): float(value) for key, value in probabilities.items()
        },
    )


def resolve_integer_target(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    fallback_min: int,
    fallback_max: int,
    namespace: str,
) -> ResolvedIntegerTarget:
    """Resolve an answer value from explicit support or inclusive bounds."""

    explicit_support = params.get(str(support_key), gen_defaults.get(str(support_key)))
    if explicit_support is not None:
        support = resolve_integer_support(
            params,
            gen_defaults=gen_defaults,
            key=str(support_key),
            fallback=tuple(int(value) for value in explicit_support),
        )
    else:
        low, high = resolve_required_int_bounds(
            params,
            gen_defaults,
            min_key=f"{support_key}_min",
            max_key=f"{support_key}_max",
            fallback_min=int(fallback_min),
            fallback_max=int(fallback_max),
            context=f"{namespace} answer support",
        )
        support = tuple(range(int(low), int(high) + 1))
    selected, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key="target_answer",
        fallback_support=support,
        namespace=str(namespace),
        balanced_flag_key=f"balanced_{support_key}_sampling",
    )
    return ResolvedIntegerTarget(
        answer_value=int(selected),
        answer_support=tuple(int(value) for value in support),
        answer_probabilities={
            str(key): float(value) for key, value in probabilities.items()
        },
    )


def resolve_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    unit_size_jitter: Mapping[str, Any],
) -> ArithmeticRenderParams:
    """Resolve canvas and cell dimensions after unit-size jitter."""

    merged = {**FALLBACK_RENDERING_DEFAULTS, **dict(render_defaults), **dict(params)}
    scale = (
        float(unit_size_jitter.get("scale", 1.0))
        if isinstance(unit_size_jitter, Mapping)
        else 1.0
    )

    def scaled_int(key: str) -> int:
        return max(1, int(round(float(merged[str(key)]) * float(scale))))

    return ArithmeticRenderParams(
        canvas_width=int(merged["canvas_width"]),
        canvas_height=int(merged["canvas_height"]),
        panel_padding_px=scaled_int("panel_padding_px"),
        panel_corner_radius_px=scaled_int("panel_corner_radius_px"),
        panel_border_width_px=max(1, scaled_int("panel_border_width_px")),
        cell_width_px=max(28, scaled_int("cell_width_px")),
        cell_height_px=max(28, scaled_int("cell_height_px")),
        node_radius_px=max(16, scaled_int("node_radius_px")),
        line_width_px=max(2, scaled_int("line_width_px")),
        value_font_size_px=max(16, scaled_int("value_font_size_px")),
        note_font_size_px=max(14, scaled_int("note_font_size_px")),
        symbol_font_size_px=max(16, scaled_int("symbol_font_size_px")),
        unit_size_jitter=dict(unit_size_jitter),
    )


def visible_value_bounds(
    params: Mapping[str, Any], gen_defaults: Mapping[str, Any]
) -> tuple[int, int]:
    """Resolve ordinary visible-value bounds for arithmetic examples."""

    low = int(
        params.get(
            "visible_value_min",
            group_default(
                gen_defaults,
                "visible_value_min",
                FALLBACK_GENERATION_DEFAULTS["visible_value_min"],
            ),
        )
    )
    high = int(
        params.get(
            "visible_value_max",
            group_default(
                gen_defaults,
                "visible_value_max",
                FALLBACK_GENERATION_DEFAULTS["visible_value_max"],
            ),
        )
    )
    if low > high:
        raise ValueError("visible_value_min must be <= visible_value_max")
    return int(low), int(high)


def answer_range_from_support(support: Sequence[int]) -> tuple[int, int]:
    """Return the inclusive visible answer range for trace metadata."""

    values = tuple(int(value) for value in support)
    if not values:
        raise ValueError("answer support cannot be empty")
    return int(min(values)), int(max(values))


__all__ = [
    "ResolvedIntegerTarget",
    "ResolvedSceneAxes",
    "answer_range_from_support",
    "resolve_integer_target",
    "resolve_render_params",
    "resolve_scene_axes",
    "visible_value_bounds",
]
