"""Style helpers for reference-canvas icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from ...shared.icon_style import icon_palette_meets_distance_constraints, sample_icon_palette


def sample_reference_canvas_palette(
    rng,
    render_params: Mapping[str, Any],
) -> Tuple[Tuple[int, int, int], ...]:
    """Sample a palette separated from panel chrome colors."""

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    anchor_colors = (
        tuple(int(v) for v in render_params["background_color_rgb"]),
        tuple(int(v) for v in render_params["panel_fill_rgb"]),
        tuple(int(v) for v in render_params["panel_border_rgb"]),
        tuple(int(v) for v in render_params["header_text_rgb"]),
    )
    palette = tuple(
        tuple(int(channel) for channel in color)
        for color in sample_icon_palette(
            rng,
            palette_size=int(palette_size),
            channel_min=int(render_params["color_channel_min"]),
            channel_max=int(render_params["color_channel_max"]),
            anchor_colors=anchor_colors,
            min_color_distance=float(render_params["min_color_distance"]),
            distance_space=str(render_params["color_distance_space"]),
        )
    )
    if not icon_palette_meets_distance_constraints(
        palette=palette,
        anchor_colors=anchor_colors,
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    ):
        raise ValueError("sampled reference-canvas palette did not satisfy distance constraints")
    return palette


__all__ = ["sample_reference_canvas_palette"]
