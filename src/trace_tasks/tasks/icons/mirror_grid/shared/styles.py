"""Style and render-parameter helpers for mirror-grid icon scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ...shared.icon_style import icon_palette_meets_distance_constraints, sample_icon_palette
from ...shared.icon_task_rendering import icon_render_style_trace, resolve_icon_cell_render_params

from .defaults import MirrorGridDefaults


def resolve_mirror_grid_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: MirrorGridDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve render params, including grid-cell patch controls."""

    render_params = resolve_icon_cell_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=defaults,
        instance_seed=int(instance_seed),
    )
    raw_rotations = params.get(
        "rotation_candidates_degrees",
        group_default(render_defaults, "rotation_candidates_degrees", list(defaults.rotation_candidates_degrees)),
    )
    if not isinstance(raw_rotations, (list, tuple)):
        raise ValueError("rotation_candidates_degrees must be a sequence")
    rotations = tuple(int(value) % 360 for value in raw_rotations)
    if not rotations:
        raise ValueError("rotation_candidates_degrees resolved no values")
    render_params["rotation_candidates_degrees"] = rotations
    render_params["symmetric_icon_count_choices"] = tuple(
        int(value)
        for value in params.get(
            "symmetric_icon_count_choices",
            group_default(render_defaults, "symmetric_icon_count_choices", list(defaults.symmetric_icon_count_choices)),
        )
    )
    render_params["both_axes_icon_count_choices"] = tuple(
        int(value)
        for value in params.get(
            "both_axes_icon_count_choices",
            group_default(render_defaults, "both_axes_icon_count_choices", list(defaults.both_axes_icon_count_choices)),
        )
    )
    render_params["nonsymmetric_icon_count_choices"] = tuple(
        int(value)
        for value in params.get(
            "nonsymmetric_icon_count_choices",
            group_default(
                render_defaults,
                "nonsymmetric_icon_count_choices",
                list(defaults.nonsymmetric_icon_count_choices),
            ),
        )
    )
    render_params["patch_inner_margin_px"] = int(
        params.get(
            "patch_inner_margin_px",
            group_default(render_defaults, "patch_inner_margin_px", defaults.patch_inner_margin_px),
        )
    )
    render_params["patch_min_gap_px"] = int(
        params.get(
            "patch_min_gap_px",
            group_default(render_defaults, "patch_min_gap_px", defaults.patch_min_gap_px),
        )
    )
    render_params["patch_sampling_attempts"] = int(
        params.get(
            "patch_sampling_attempts",
            group_default(render_defaults, "patch_sampling_attempts", defaults.patch_sampling_attempts),
        )
    )
    return render_params


def sample_mirror_grid_palette(rng, render_params: Mapping[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    """Sample an icon palette separated from the canvas chrome colors."""

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
        raise ValueError("sampled mirror-grid palette did not satisfy distance constraints")
    return palette


def mirror_grid_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Tuple[int, int, int]],
) -> Dict[str, Any]:
    """Return trace metadata for rendered mirror-grid style choices."""

    style = icon_render_style_trace(
        render_params=render_params,
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in sampled_palette_rgb),
    )
    style.update(
        {
            "rotation_candidates_degrees": [int(value) for value in render_params["rotation_candidates_degrees"]],
            "symmetric_icon_count_choices": [int(value) for value in render_params["symmetric_icon_count_choices"]],
            "both_axes_icon_count_choices": [int(value) for value in render_params["both_axes_icon_count_choices"]],
            "nonsymmetric_icon_count_choices": [int(value) for value in render_params["nonsymmetric_icon_count_choices"]],
            "patch_inner_margin_px": int(render_params["patch_inner_margin_px"]),
            "patch_min_gap_px": int(render_params["patch_min_gap_px"]),
            "patch_sampling_attempts": int(render_params["patch_sampling_attempts"]),
            "cell_padding_px": int(render_params["cell_padding_px"]),
            "cell_border_rgb": list(render_params["cell_border_rgb"]),
            "cell_label_color_rgb": list(render_params["cell_label_color_rgb"]),
            "cell_label_stroke_rgb": list(render_params.get("cell_label_stroke_rgb", render_params["panel_fill_rgb"])),
            "cell_label_font_size_px": int(render_params["cell_label_font_size_px"]),
        }
    )
    return style


__all__ = [
    "mirror_grid_style_trace",
    "resolve_mirror_grid_render_params",
    "sample_mirror_grid_palette",
]

