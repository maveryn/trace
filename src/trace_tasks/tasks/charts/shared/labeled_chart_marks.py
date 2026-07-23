"""Mark color and spec helpers for labeled chart task families."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.color_distance import (
    sample_color_palette_with_distance_constraints,
    sample_color_with_distance_constraints,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.named_colors import darken_color

from .chart_scene_types import ChartMarkSpec
from .labeled_chart_defaults import LabeledChartDefaults
from .labeled_chart_variants import is_pie_like_scene_variant


def normalize_rgb(value: Sequence[int]) -> Tuple[int, int, int]:
    """Normalize one RGB-like sequence into a clamped color triple."""

    if len(value) < 3:
        raise ValueError("RGB value must contain three channels")
    return (
        max(0, min(255, int(value[0]))),
        max(0, min(255, int(value[1]))),
        max(0, min(255, int(value[2]))),
    )


def build_chart_mark_specs(
    *,
    labels: Sequence[str],
    values: Sequence[int],
    scene_variant: str,
    mark_style: Mapping[str, Any],
) -> List[ChartMarkSpec]:
    """Build chart-mark specs with per-mark colors resolved for the scene variant."""

    resolved_labels = [str(label) for label in labels]
    resolved_values = [int(value) for value in values]
    if len(resolved_labels) != len(resolved_values):
        raise ValueError("labels and values must have the same length")

    if is_pie_like_scene_variant(str(scene_variant)):
        fill_palette = [normalize_rgb(value) for value in mark_style.get("slice_fill_palette_rgb", [])]
        outline_palette = [normalize_rgb(value) for value in mark_style.get("slice_outline_palette_rgb", [])]
        if len(fill_palette) != len(resolved_labels) or len(outline_palette) != len(resolved_labels):
            raise ValueError("pie-style chart marks require one fill/outline color per slice")
        return [
            ChartMarkSpec(
                label=str(label),
                value=int(value),
                fill_rgb=tuple(int(channel) for channel in fill_palette[index]),
                outline_rgb=tuple(int(channel) for channel in outline_palette[index]),
            )
            for index, (label, value) in enumerate(zip(resolved_labels, resolved_values))
        ]

    fill_rgb = normalize_rgb(mark_style.get("mark_fill_rgb", (86, 138, 214)))
    outline_rgb = normalize_rgb(mark_style.get("mark_outline_rgb", darken_color(fill_rgb, factor=0.55)))
    return [
        ChartMarkSpec(
            label=str(label),
            value=int(value),
            fill_rgb=tuple(int(channel) for channel in fill_rgb),
            outline_rgb=tuple(int(channel) for channel in outline_rgb),
        )
        for label, value in zip(resolved_labels, resolved_values)
    ]


def resolve_chart_mark_colors(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: LabeledChartDefaults,
    instance_seed: int,
    scene_variant: str,
    mark_count: int,
) -> Dict[str, Any]:
    """Resolve one per-instance chart mark color style block."""

    explicit_fill = params.get("mark_fill_rgb")
    explicit_outline = params.get("mark_outline_rgb")

    if not is_pie_like_scene_variant(str(scene_variant)) and (explicit_fill is not None or explicit_outline is not None):
        fill_rgb = normalize_rgb(explicit_fill if explicit_fill is not None else (86, 138, 214))
        outline_rgb = normalize_rgb(
            explicit_outline if explicit_outline is not None else darken_color(fill_rgb, factor=0.55)
        )
        return {
            "sampling_policy": "explicit_override",
            "mark_fill_rgb": [int(channel) for channel in fill_rgb],
            "mark_outline_rgb": [int(channel) for channel in outline_rgb],
        }

    color_rng = spawn_rng(int(instance_seed), "charts.mark_color")
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
    if is_pie_like_scene_variant(str(scene_variant)):
        pie_channel_max = int(
            params.get(
                "pie_like_mark_color_channel_max",
                group_default(
                    render_defaults,
                    "pie_like_mark_color_channel_max",
                    defaults.pie_like_mark_color_channel_max,
                ),
            )
        )
        pie_min_distance = float(
            params.get(
                "pie_like_mark_color_min_distance",
                group_default(
                    render_defaults,
                    "pie_like_mark_color_min_distance",
                    defaults.pie_like_mark_color_min_distance,
                ),
            )
        )
        fill_palette = sample_color_palette_with_distance_constraints(
            color_rng,
            palette_size=int(mark_count),
            channel_min=int(channel_min),
            channel_max=int(min(int(channel_max), int(pie_channel_max))),
            anchor_colors=((255, 255, 255), (248, 248, 248), (236, 238, 242)),
            min_distance=float(max(float(min_distance), float(pie_min_distance))),
            distance_space=str(distance_space),
        )
        outline_palette = [darken_color(fill_rgb, factor=0.55) for fill_rgb in fill_palette]
        return {
            "sampling_policy": "random_rgb_palette",
            "mark_fill_rgb": [int(channel) for channel in fill_palette[0]],
            "mark_outline_rgb": [int(channel) for channel in outline_palette[0]],
            "slice_fill_palette_rgb": [[int(channel) for channel in fill_rgb] for fill_rgb in fill_palette],
            "slice_outline_palette_rgb": [[int(channel) for channel in outline_rgb] for outline_rgb in outline_palette],
            "mark_color_min_distance": float(max(float(min_distance), float(pie_min_distance))),
            "pie_like_mark_color_channel_max": int(min(int(channel_max), int(pie_channel_max))),
            "mark_color_distance_space": str(distance_space),
        }
    fill_rgb = sample_color_with_distance_constraints(
        color_rng,
        channel_min=int(channel_min),
        channel_max=int(channel_max),
        anchor_colors=((255, 255, 255), (248, 248, 248)),
        min_distance=float(min_distance),
        distance_space=str(distance_space),
    )
    outline_rgb = darken_color(fill_rgb, factor=0.55)
    return {
        "sampling_policy": "random_rgb",
        "mark_fill_rgb": [int(channel) for channel in fill_rgb],
        "mark_outline_rgb": [int(channel) for channel in outline_rgb],
        "mark_color_min_distance": float(min_distance),
        "mark_color_distance_space": str(distance_space),
    }


__all__ = [
    "build_chart_mark_specs",
    "normalize_rgb",
    "resolve_chart_mark_colors",
]
