"""Config, visual-axis, and font defaults for color-gradient puzzle scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeVar

from trace_tasks.core.sampling import support_probability_map, weighted_support_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.color_distance import coerce_rgb
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)

from .state import DEFAULTS, RenderParams

T = TypeVar("T")


def sample_weighted_variant(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    instance_seed: int,
    support: Sequence[T],
    explicit_key: str,
    weights_key: str,
    namespace: str,
) -> tuple[T, dict[str, float]]:
    """Sample one semantic/render axis from explicit support and weights."""

    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected_key = str(explicit)
        by_key = {str(value): value for value in support}
        if selected_key not in by_key:
            raise ValueError(f"unsupported {explicit_key}: {explicit!r}")
        return (
            by_key[selected_key],
            support_probability_map(support, selected=by_key[selected_key]),
        )

    raw_weights = group_default(generation_defaults, str(weights_key), None)
    weights = raw_weights if isinstance(raw_weights, Mapping) else None
    rng = spawn_rng(int(instance_seed), str(namespace))
    return weighted_support_choice(rng, support, weights=weights)


def resolve_render_params(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> RenderParams:
    """Resolve render params from scene config and unit-size jitter."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.color_gradient.unit_size",
    )
    return RenderParams(
        canvas_width=int(
            group_default(rendering_defaults, "canvas_width", DEFAULTS.canvas_width)
        ),
        canvas_height=int(
            group_default(rendering_defaults, "canvas_height", DEFAULTS.canvas_height)
        ),
        swatch_size_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "swatch_size_px",
                DEFAULTS.swatch_size_px,
            ),
            unit_scale,
            min_px=42,
        ),
        swatch_gap_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "swatch_gap_px",
                DEFAULTS.swatch_gap_px,
            ),
            unit_scale,
            min_px=6,
        ),
        panel_padding_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "panel_padding_px",
                DEFAULTS.panel_padding_px,
            ),
            unit_scale,
            min_px=14,
        ),
        panel_corner_radius_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "panel_corner_radius_px",
                DEFAULTS.panel_corner_radius_px,
            ),
            unit_scale,
            min_px=8,
        ),
        panel_border_width_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "panel_border_width_px",
                DEFAULTS.panel_border_width_px,
            ),
            unit_scale,
            min_px=1,
        ),
        swatch_corner_radius_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "swatch_corner_radius_px",
                DEFAULTS.swatch_corner_radius_px,
            ),
            unit_scale,
            min_px=5,
        ),
        swatch_border_width_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "swatch_border_width_px",
                DEFAULTS.swatch_border_width_px,
            ),
            unit_scale,
            min_px=1,
        ),
        label_chip_size_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "label_chip_size_px",
                DEFAULTS.label_chip_size_px,
            ),
            unit_scale,
            min_px=28,
        ),
        label_margin_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "label_margin_px",
                DEFAULTS.label_margin_px,
            ),
            unit_scale,
            min_px=4,
        ),
        label_font_size_px=scale_puzzle_px(
            group_default(
                rendering_defaults,
                "label_font_size_px",
                DEFAULTS.label_font_size_px,
            ),
            unit_scale,
            min_px=16,
        ),
        panel_fill_rgb=coerce_rgb(
            group_default(
                rendering_defaults,
                "panel_fill_rgb",
                DEFAULTS.panel_fill_rgb,
            ),
            DEFAULTS.panel_fill_rgb,
        ),
        panel_border_rgb=coerce_rgb(
            group_default(
                rendering_defaults,
                "panel_border_rgb",
                DEFAULTS.panel_border_rgb,
            ),
            DEFAULTS.panel_border_rgb,
        ),
        swatch_border_rgb=coerce_rgb(
            group_default(
                rendering_defaults,
                "swatch_border_rgb",
                DEFAULTS.swatch_border_rgb,
            ),
            DEFAULTS.swatch_border_rgb,
        ),
        notebook_line_rgb=coerce_rgb(
            group_default(
                rendering_defaults,
                "notebook_line_rgb",
                DEFAULTS.notebook_line_rgb,
            ),
            DEFAULTS.notebook_line_rgb,
        ),
        unit_size_jitter=dict(unit_meta),
    )


def sample_label_font(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> str:
    """Sample one global font family for visible swatch and option labels."""

    return sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.color_gradient_label_font",
        params={**dict(rendering_defaults), **dict(params)},
    )


def font_trace_record(font_family: str, *, scope: str) -> dict[str, Any]:
    """Build trace metadata for one sampled visible-label font."""

    return {
        **get_font_family_record(str(font_family)).to_trace(),
        "source": "global_font_pool",
        "font_asset_version": font_asset_version(),
        "scope": str(scope),
    }


def post_noise_policy_trace() -> dict[str, Any]:
    """Document why color-gradient scenes disable post-render noise."""

    return {
        "default_override": True,
        "reason": "color_semantics_preserve_rgb_separability",
    }


__all__ = [
    "font_trace_record",
    "post_noise_policy_trace",
    "resolve_render_params",
    "sample_label_font",
    "sample_weighted_variant",
]
