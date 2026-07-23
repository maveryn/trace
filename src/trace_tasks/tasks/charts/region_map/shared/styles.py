"""Shared style registries and resolvers for region-map charts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.color_distance import coerce_rgb as _rgb
from ...shared.labeled_chart_variants import resolve_chart_axis_variant_for_namespace


SUPPORTED_LEGEND_POSITIONS: Tuple[str, ...] = ("right", "bottom", "top", "none")
SUPPORTED_WORLD_MAP_STYLES: Tuple[str, ...] = (
    "atlas_light",
    "report_blue",
    "warm_print",
    "muted_gray",
    "clean_minimal",
)


NUMERIC_PALETTES_RGB: Dict[str, Tuple[Tuple[int, int, int], ...]] = {
    "blue": (
        (239, 246, 255),
        (219, 234, 254),
        (191, 219, 254),
        (147, 197, 253),
        (96, 165, 250),
        (59, 130, 246),
        (37, 99, 235),
        (30, 64, 175),
    ),
    "green": (
        (240, 253, 244),
        (220, 252, 231),
        (187, 247, 208),
        (134, 239, 172),
        (74, 222, 128),
        (34, 197, 94),
        (22, 163, 74),
        (21, 128, 61),
    ),
    "orange": (
        (255, 247, 237),
        (255, 237, 213),
        (254, 215, 170),
        (253, 186, 116),
        (251, 146, 60),
        (249, 115, 22),
        (234, 88, 12),
        (194, 65, 12),
    ),
    "purple": (
        (250, 245, 255),
        (243, 232, 255),
        (233, 213, 255),
        (216, 180, 254),
        (192, 132, 252),
        (168, 85, 247),
        (147, 51, 234),
        (126, 34, 206),
    ),
    "rose": (
        (255, 241, 242),
        (255, 228, 230),
        (254, 205, 211),
        (253, 164, 175),
        (251, 113, 133),
        (244, 63, 94),
        (225, 29, 72),
        (190, 18, 60),
    ),
    "teal": (
        (240, 253, 250),
        (204, 251, 241),
        (153, 246, 228),
        (94, 234, 212),
        (45, 212, 191),
        (20, 184, 166),
        (13, 148, 136),
        (15, 118, 110),
    ),
    "indigo": (
        (238, 242, 255),
        (224, 231, 255),
        (199, 210, 254),
        (165, 180, 252),
        (129, 140, 248),
        (99, 102, 241),
        (79, 70, 229),
        (67, 56, 202),
    ),
    "magma": (
        (252, 244, 250),
        (246, 210, 238),
        (230, 159, 211),
        (204, 101, 178),
        (172, 54, 143),
        (130, 31, 113),
        (86, 24, 91),
        (42, 17, 61),
    ),
    "earth": (
        (248, 250, 229),
        (230, 242, 194),
        (199, 226, 166),
        (153, 204, 142),
        (105, 172, 123),
        (80, 132, 106),
        (72, 93, 88),
        (57, 64, 67),
    ),
    "sunset": (
        (255, 247, 214),
        (254, 226, 178),
        (253, 198, 138),
        (248, 155, 108),
        (230, 103, 99),
        (197, 63, 101),
        (147, 43, 104),
        (93, 31, 84),
    ),
}
CATEGORICAL_PALETTES_RGB: Dict[str, Tuple[Tuple[int, int, int], ...]] = {
    "tableau": (
        (78, 121, 167),
        (242, 142, 43),
        (225, 87, 89),
        (118, 183, 178),
        (89, 161, 79),
        (237, 201, 72),
        (176, 122, 161),
        (255, 157, 167),
    ),
    "bold": (
        (35, 92, 146),
        (214, 93, 14),
        (33, 145, 140),
        (128, 64, 173),
        (73, 157, 63),
        (205, 59, 97),
        (166, 118, 29),
        (72, 109, 109),
    ),
    "pastel": (
        (141, 211, 199),
        (255, 255, 179),
        (190, 186, 218),
        (251, 128, 114),
        (128, 177, 211),
        (253, 180, 98),
        (179, 222, 105),
        (252, 205, 229),
    ),
    "metro": (
        (45, 117, 182),
        (239, 142, 72),
        (78, 166, 117),
        (197, 88, 90),
        (122, 95, 172),
        (151, 111, 76),
        (218, 124, 174),
        (116, 116, 116),
    ),
    "harbor": (
        (31, 119, 140),
        (87, 166, 161),
        (167, 203, 161),
        (244, 203, 103),
        (224, 134, 69),
        (193, 84, 84),
        (117, 88, 142),
        (77, 93, 104),
    ),
    "orchid": (
        (107, 70, 193),
        (190, 85, 212),
        (236, 72, 153),
        (244, 114, 182),
        (99, 102, 241),
        (14, 165, 233),
        (20, 184, 166),
        (132, 204, 22),
    ),
    "field": (
        (79, 121, 66),
        (159, 177, 83),
        (226, 193, 95),
        (202, 132, 74),
        (139, 91, 69),
        (94, 124, 139),
        (82, 92, 120),
        (150, 101, 137),
    ),
    "civic": (
        (49, 90, 158),
        (96, 150, 197),
        (244, 162, 97),
        (231, 111, 81),
        (42, 157, 143),
        (138, 176, 125),
        (196, 154, 108),
        (108, 117, 125),
    ),
}


WORLD_MAP_STYLES: Dict[str, Dict[str, Any]] = {
    "atlas_light": {
        "ocean_rgb": (230, 241, 247),
        "land_fill_rgb": (218, 224, 219),
        "land_outline_rgb": (128, 143, 148),
        "selected_outline_rgb": (37, 47, 56),
        "graticule_rgb": (197, 214, 224),
        "graticule_width_px": 1,
        "selected_outline_width_px": 2,
        "show_graticule": True,
    },
    "report_blue": {
        "ocean_rgb": (224, 236, 245),
        "land_fill_rgb": (226, 229, 226),
        "land_outline_rgb": (107, 128, 141),
        "selected_outline_rgb": (24, 40, 58),
        "graticule_rgb": (187, 207, 221),
        "graticule_width_px": 1,
        "selected_outline_width_px": 2,
        "show_graticule": True,
    },
    "warm_print": {
        "ocean_rgb": (244, 239, 226),
        "land_fill_rgb": (224, 219, 205),
        "land_outline_rgb": (139, 127, 110),
        "selected_outline_rgb": (64, 48, 38),
        "graticule_rgb": (218, 207, 188),
        "graticule_width_px": 1,
        "selected_outline_width_px": 2,
        "show_graticule": True,
    },
    "muted_gray": {
        "ocean_rgb": (238, 241, 243),
        "land_fill_rgb": (219, 223, 224),
        "land_outline_rgb": (118, 127, 132),
        "selected_outline_rgb": (31, 35, 40),
        "graticule_rgb": (210, 216, 220),
        "graticule_width_px": 1,
        "selected_outline_width_px": 2,
        "show_graticule": False,
    },
    "clean_minimal": {
        "ocean_rgb": (250, 252, 253),
        "land_fill_rgb": (225, 230, 226),
        "land_outline_rgb": (144, 154, 154),
        "selected_outline_rgb": (39, 45, 52),
        "graticule_rgb": (224, 230, 233),
        "graticule_width_px": 1,
        "selected_outline_width_px": 2,
        "show_graticule": False,
    },
}

MARKER_STYLES: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "teal": {
        "fill": (20, 126, 136),
        "outline": (8, 70, 78),
        "label_fill": (255, 255, 255),
        "label_outline": (31, 55, 64),
    },
    "coral": {
        "fill": (218, 88, 64),
        "outline": (122, 45, 34),
        "label_fill": (255, 255, 255),
        "label_outline": (97, 48, 41),
    },
    "violet": {
        "fill": (118, 94, 188),
        "outline": (61, 48, 112),
        "label_fill": (255, 255, 255),
        "label_outline": (54, 47, 84),
    },
    "gold": {
        "fill": (224, 166, 55),
        "outline": (121, 82, 28),
        "label_fill": (255, 255, 255),
        "label_outline": (88, 66, 40),
    },
    "ink": {
        "fill": (52, 74, 94),
        "outline": (20, 31, 43),
        "label_fill": (255, 255, 255),
        "label_outline": (20, 31, 43),
    },
}

def select_choropleth_palette_colors(
    palette: Sequence[Sequence[int]],
    *,
    required_palette_count: int,
) -> Tuple[Tuple[int, int, int], ...]:
    colors = tuple(_rgb(item, (120, 140, 160)) for item in palette)
    if not colors:
        colors = tuple(NUMERIC_PALETTES_RGB["blue"])
    if int(required_palette_count) <= 1:
        return (colors[0],)
    if len(colors) >= int(required_palette_count):
        return tuple(
            colors[int(round(index * (len(colors) - 1) / max(1, int(required_palette_count) - 1)))]
            for index in range(int(required_palette_count))
        )
    out = list(colors)
    while len(out) < int(required_palette_count):
        out.extend(colors)
    return tuple(out[: int(required_palette_count)])


def resolve_choropleth_palette(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    style_seed: int,
    required_palette_count: int,
    categorical: bool,
) -> Tuple[str, Dict[str, float], Tuple[Tuple[int, int, int], ...]]:
    """Resolve a map palette with enough contrast for bins or categorical region fills."""

    explicit_palette_raw = params.get("map_palette_rgb")
    if explicit_palette_raw is not None:
        palette = tuple(_rgb(item, (150, 180, 210)) for item in explicit_palette_raw)
        return "custom", {"custom": 1.0}, select_choropleth_palette_colors(
            palette,
            required_palette_count=int(required_palette_count),
        )

    palettes = CATEGORICAL_PALETTES_RGB if bool(categorical) else NUMERIC_PALETTES_RGB
    supported = tuple(sorted(palettes.keys()))
    prefix = "categorical" if bool(categorical) else "numeric"
    explicit_key = f"{prefix}_palette_variant"
    resolved_params: Mapping[str, Any] = params
    if "map_palette_variant" in params and explicit_key not in params:
        resolved_params = {**dict(params), explicit_key: params.get("map_palette_variant")}
    variant, probabilities = resolve_chart_axis_variant_for_namespace(
        params=resolved_params,
        gen_defaults=render_defaults,
        instance_seed=int(style_seed),
        supported_variants=supported,
        namespace=f"{namespace}.{prefix}_palette_variant",
        explicit_key=explicit_key,
        weights_key=f"{prefix}_palette_variant_weights",
        balance_flag_key=f"balanced_{prefix}_palette_variant_sampling",
    )
    palette = palettes.get(str(variant), next(iter(palettes.values())))
    return (
        str(variant),
        {str(key): float(value) for key, value in sorted(probabilities.items())},
        select_choropleth_palette_colors(palette, required_palette_count=int(required_palette_count)),
    )


def resolve_choropleth_legend_position(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=render_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_LEGEND_POSITIONS,
        namespace=f"{namespace}.legend_position",
        explicit_key="legend_position",
        weights_key="legend_position_weights",
        balance_flag_key="balanced_legend_position_sampling",
    )


def resolve_choropleth_world_map_style(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    style_id, probabilities = resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=render_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_WORLD_MAP_STYLES,
        namespace=f"{namespace}.world_map_style",
        explicit_key="world_map_style",
        weights_key="world_map_style_weights",
        balance_flag_key="balanced_world_map_style_sampling",
    )
    return (
        str(style_id),
        {str(key): float(value) for key, value in sorted(probabilities.items())},
        dict(WORLD_MAP_STYLES[str(style_id)]),
    )


def resolve_choropleth_marker_style(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> Tuple[str, Dict[str, float], Dict[str, Tuple[int, int, int]]]:
    style_id, probabilities = resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=render_defaults,
        instance_seed=int(instance_seed),
        supported_variants=tuple(sorted(MARKER_STYLES.keys())),
        namespace=f"{namespace}.marker_style_variant",
        explicit_key="marker_style_variant",
        weights_key="marker_style_variant_weights",
        balance_flag_key="balanced_marker_style_variant_sampling",
    )
    return (
        str(style_id),
        {str(key): float(value) for key, value in sorted(probabilities.items())},
        dict(MARKER_STYLES[str(style_id)]),
    )


__all__ = [
    "CATEGORICAL_PALETTES_RGB",
    "MARKER_STYLES",
    "NUMERIC_PALETTES_RGB",
    "SUPPORTED_LEGEND_POSITIONS",
    "SUPPORTED_WORLD_MAP_STYLES",
    "WORLD_MAP_STYLES",
    "resolve_choropleth_legend_position",
    "resolve_choropleth_marker_style",
    "resolve_choropleth_palette",
    "resolve_choropleth_world_map_style",
    "select_choropleth_palette_colors",
]
