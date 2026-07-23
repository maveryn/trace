"""Style helpers for multiseries chart rendering."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from .....core.seed import spawn_rng
from ....shared.color_distance import sample_color_palette_with_distance_constraints
from ....shared.config_defaults import group_default
from ....shared.named_colors import darken_color
from .state import MultiseriesChartDefaults


def resolve_multiseries_chart_colors(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: MultiseriesChartDefaults,
    instance_seed: int,
    series_count: int,
) -> Dict[str, Any]:
    """Resolve one per-series color palette for multiseries charts."""

    color_rng = spawn_rng(int(instance_seed), "charts.multiseries.colors")
    channel_min = int(
        params.get(
            "mark_color_channel_min",
            group_default(render_defaults, "mark_color_channel_min", defaults.mark_color_channel_min),
        )
    )
    channel_max = int(
        params.get(
            "mark_color_channel_max",
            group_default(render_defaults, "mark_color_channel_max", defaults.mark_color_channel_max),
        )
    )
    min_distance = float(
        params.get(
            "mark_color_min_distance",
            group_default(render_defaults, "mark_color_min_distance", defaults.mark_color_min_distance),
        )
    )
    distance_space = str(
        params.get(
            "mark_color_distance_space",
            group_default(render_defaults, "mark_color_distance_space", defaults.mark_color_distance_space),
        )
    ).strip().lower()
    fill_palette = sample_color_palette_with_distance_constraints(
        color_rng,
        palette_size=int(series_count),
        channel_min=int(channel_min),
        channel_max=int(channel_max),
        anchor_colors=((255, 255, 255), (248, 248, 248)),
        min_distance=float(min_distance),
        distance_space=str(distance_space),
    )
    outline_palette = [darken_color(fill_rgb, factor=0.55) for fill_rgb in fill_palette]
    return {
        "sampling_policy": "random_rgb_palette",
        "series_fill_palette_rgb": [[int(channel) for channel in fill_rgb] for fill_rgb in fill_palette],
        "series_outline_palette_rgb": [[int(channel) for channel in outline_rgb] for outline_rgb in outline_palette],
        "mark_color_min_distance": float(min_distance),
        "mark_color_distance_space": str(distance_space),
    }


__all__ = ["resolve_multiseries_chart_colors"]
