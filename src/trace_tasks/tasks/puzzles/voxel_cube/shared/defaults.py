"""Config and visual defaults for voxel-cube puzzle scenes."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.core.sampling import integer_range_choice, uniform_choice
from trace_tasks.tasks.shared.config_defaults import group_default

from .state import VoxelPalette, VoxelRenderParams


_VOXEL_PALETTES: tuple[VoxelPalette, ...] = (
    VoxelPalette(
        palette_id="blue",
        cube_top_rgb=(116, 178, 232),
        cube_left_rgb=(76, 130, 188),
        cube_right_rgb=(92, 153, 216),
        cube_edge_rgb=(34, 64, 96),
        projection_fill_rgb=(82, 142, 205),
        projection_empty_rgb=(244, 248, 252),
    ),
    VoxelPalette(
        palette_id="teal",
        cube_top_rgb=(114, 205, 194),
        cube_left_rgb=(58, 143, 137),
        cube_right_rgb=(80, 174, 166),
        cube_edge_rgb=(27, 78, 78),
        projection_fill_rgb=(56, 154, 146),
        projection_empty_rgb=(244, 249, 248),
    ),
    VoxelPalette(
        palette_id="amber",
        cube_top_rgb=(230, 178, 92),
        cube_left_rgb=(166, 118, 47),
        cube_right_rgb=(203, 142, 60),
        cube_edge_rgb=(91, 62, 32),
        projection_fill_rgb=(198, 137, 55),
        projection_empty_rgb=(252, 248, 240),
    ),
    VoxelPalette(
        palette_id="plum",
        cube_top_rgb=(178, 143, 232),
        cube_left_rgb=(112, 83, 168),
        cube_right_rgb=(141, 101, 202),
        cube_edge_rgb=(60, 45, 102),
        projection_fill_rgb=(132, 89, 196),
        projection_empty_rgb=(248, 246, 252),
    ),
    VoxelPalette(
        palette_id="coral",
        cube_top_rgb=(229, 133, 122),
        cube_left_rgb=(163, 76, 72),
        cube_right_rgb=(196, 93, 87),
        cube_edge_rgb=(96, 44, 48),
        projection_fill_rgb=(199, 87, 82),
        projection_empty_rgb=(252, 246, 245),
    ),
    VoxelPalette(
        palette_id="moss",
        cube_top_rgb=(145, 199, 122),
        cube_left_rgb=(83, 137, 78),
        cube_right_rgb=(108, 166, 91),
        cube_edge_rgb=(49, 80, 45),
        projection_fill_rgb=(92, 151, 82),
        projection_empty_rgb=(246, 250, 244),
    ),
)


def default_voxel_palette() -> VoxelPalette:
    """Return the stable fallback voxel palette."""

    return _VOXEL_PALETTES[0]


def sample_voxel_palette(rng) -> VoxelPalette:
    """Sample one fixed non-semantic palette for a rendered voxel instance."""

    return uniform_choice(rng, _VOXEL_PALETTES)


def int_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    """Resolve an inclusive integer range from params/defaults/fallbacks."""

    minimum = int(params.get(min_key, group_default(defaults, min_key, fallback_min)))
    maximum = int(params.get(max_key, group_default(defaults, max_key, fallback_max)))
    if minimum > maximum:
        raise ValueError(f"{min_key} must be <= {max_key}")
    return minimum, maximum


def resolve_answer_value(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    rng,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, tuple[int, int]]:
    """Sample one target answer from the configured answer support."""

    support = int_bounds(
        params,
        defaults,
        min_key="answer_min",
        max_key="answer_max",
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    selected, _probabilities = integer_range_choice(
        rng,
        int(support[0]),
        int(support[1]),
    )
    return int(selected), tuple(int(value) for value in support)


def resolve_option_count(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    fallback: int,
) -> int:
    """Resolve the number of visible option panels."""

    count = int(
        params.get("option_count", group_default(defaults, "option_count", fallback))
    )
    if not 3 <= count <= 6:
        raise ValueError("voxel option_count must be in 3..6")
    return int(count)


def resolve_axis_choice(
    params: Mapping[str, Any],
    *,
    key: str,
    support: tuple[str, ...],
    rng,
) -> str:
    """Resolve or sample one semantic axis value from explicit support."""

    explicit = params.get(str(key))
    if explicit is not None:
        selected = str(explicit)
        if selected not in support:
            raise ValueError(f"unsupported {key}: {selected}")
        return selected
    return str(uniform_choice(rng, support))


def resolve_render_params(
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> VoxelRenderParams:
    """Resolve concrete render dimensions and font sizes."""

    return VoxelRenderParams(
        canvas_width=int(
            params.get(
                "canvas_width",
                group_default(rendering_defaults, "canvas_width", 900),
            )
        ),
        canvas_height=int(
            params.get(
                "canvas_height",
                group_default(rendering_defaults, "canvas_height", 720),
            )
        ),
        cube_size_px=int(
            params.get(
                "cube_size_px",
                group_default(rendering_defaults, "cube_size_px", 48),
            )
        ),
        projection_cell_size_px=int(
            params.get(
                "projection_cell_size_px",
                group_default(
                    rendering_defaults,
                    "projection_cell_size_px",
                    42,
                ),
            )
        ),
        panel_gap_px=int(
            params.get(
                "panel_gap_px",
                group_default(rendering_defaults, "panel_gap_px", 26),
            )
        ),
        label_font_size_px=int(
            params.get(
                "label_font_size_px",
                group_default(rendering_defaults, "label_font_size_px", 24),
            )
        ),
        palette=default_voxel_palette(),
    )
