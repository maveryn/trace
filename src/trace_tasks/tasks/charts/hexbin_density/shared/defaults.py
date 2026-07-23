"""Scene constants and default resolution for hexbin-density charts."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_int_bounds,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.render_variation import (
    apply_layout_jitter_to_margins,
    resolve_render_int,
    resolve_render_rgb,
)
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant
from trace_tasks.tasks.charts.shared.visual_defaults import (
    load_chart_scene_background_defaults,
    load_chart_scene_noise_defaults,
)


DOMAIN = "charts"
SCENE_ID = "hexbin_density"
SCENE_NAMESPACE = "charts_hexbin_density"
PROMPT_BUNDLE_ID = "charts_hexbin_density_v1"
RGB = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]

SUPPORTED_DENSITY_PALETTE_SCHEMES: Tuple[str, ...] = (
    "blue",
    "teal",
    "green",
    "purple",
    "amber",
    "rose",
    "slate",
    "viridis",
    "cividis",
)
AXIS_LABELS: Tuple[Tuple[str, str], ...] = (
    ("x", "y"),
    ("Feature A", "Feature B"),
    ("Axis 1", "Axis 2"),
    ("Horizontal score", "Vertical score"),
)
FALLBACK_DENSITY_PALETTE_SCHEMES: Dict[str, Tuple[RGB, ...]] = {
    "blue": (
        (210, 234, 247),
        (151, 203, 226),
        (88, 161, 197),
        (37, 113, 168),
        (15, 74, 127),
    ),
    "teal": (
        (205, 236, 232),
        (140, 210, 202),
        (76, 174, 166),
        (25, 126, 137),
        (13, 82, 103),
    ),
    "green": (
        (214, 238, 203),
        (166, 218, 145),
        (105, 186, 103),
        (49, 139, 80),
        (19, 90, 58),
    ),
    "purple": (
        (225, 216, 243),
        (188, 171, 226),
        (148, 126, 205),
        (102, 79, 169),
        (67, 49, 122),
    ),
    "amber": (
        (245, 224, 171),
        (236, 190, 101),
        (219, 148, 54),
        (183, 101, 32),
        (125, 65, 26),
    ),
    "rose": (
        (245, 208, 216),
        (232, 150, 166),
        (206, 93, 124),
        (166, 49, 91),
        (109, 28, 62),
    ),
    "slate": (
        (214, 224, 235),
        (164, 181, 199),
        (111, 135, 158),
        (67, 88, 112),
        (36, 52, 76),
    ),
    "viridis": (
        (205, 225, 101),
        (124, 201, 88),
        (53, 164, 121),
        (41, 120, 142),
        (68, 45, 130),
    ),
    "cividis": (
        (232, 211, 90),
        (196, 180, 84),
        (142, 144, 92),
        (87, 106, 110),
        (44, 65, 96),
    ),
}

_SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_chart_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_chart_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def render_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(
        resolve_render_int(
            params,
            RENDER_DEFAULTS,
            str(key),
            int(fallback),
            instance_seed=render_style_seed(params),
            namespace=f"{SCENE_NAMESPACE}.render",
        )
    )


def render_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.render",
    )


def generation_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(str(key), group_default(GEN_DEFAULTS, str(key), int(fallback))))


def balanced_int(
    params: Mapping[str, Any],
    *,
    key: str,
    support: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> int:
    if str(key) in params:
        value = int(params[str(key)])
        if value not in set(int(item) for item in support):
            raise ValueError(f"{key}={value} is outside supported values {list(support)}")
        return int(value)
    values = tuple(int(value) for value in support)
    if not values:
        raise ValueError(f"empty integer support for {key}")
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{namespace}"),
            values,
            sort_keys=True,
        )
    )


def resolve_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=f"generation defaults for {SCENE_ID}",
    )


def rgb_distance(a: RGB, b: RGB) -> float:
    return math.sqrt(sum((float(a[index]) - float(b[index])) ** 2 for index in range(3)))


def ensure_level_one_contrast(palette: Sequence[RGB]) -> Tuple[RGB, ...]:
    resolved = [tuple(int(channel) for channel in color) for color in palette[:5]]
    if not resolved:
        return tuple()
    first = resolved[0]
    if rgb_distance(first, (255, 255, 255)) < 52.0:
        for delta in (24, 36, 48, 60):
            adjusted = tuple(max(0, int(channel) - int(delta)) for channel in first)
            if rgb_distance(adjusted, (255, 255, 255)) >= 52.0:
                resolved[0] = adjusted
                break
        else:
            resolved[0] = tuple(max(0, int(channel) - 60) for channel in first)
    return tuple(resolved)


def coerce_palette(raw: Any) -> Tuple[RGB, ...]:
    palette: list[RGB] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 3:
                palette.append(tuple(max(0, min(255, int(channel))) for channel in item[:3]))  # type: ignore[index]
    if len(palette) >= 5:
        return ensure_level_one_contrast(tuple(palette[:5]))
    return tuple()


def configured_palette_schemes(params: Mapping[str, Any]) -> Dict[str, Tuple[RGB, ...]]:
    schemes: Dict[str, Tuple[RGB, ...]] = dict(FALLBACK_DENSITY_PALETTE_SCHEMES)
    raw = params.get("density_palette_schemes", group_default(RENDER_DEFAULTS, "density_palette_schemes", {}))
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            palette = coerce_palette(value)
            if len(palette) >= 5:
                schemes[str(key)] = tuple(palette[:5])
    return schemes


def resolve_density_palette(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Tuple[RGB, ...], Dict[str, Any]]:
    """Resolve the five-level density palette while preserving level-one contrast."""

    explicit_palette = params.get("density_palette_rgb")
    if explicit_palette is not None:
        palette = coerce_palette(explicit_palette)
        if len(palette) < 5:
            raise ValueError("density_palette_rgb must contain at least five RGB colors")
        return (
            "custom",
            tuple(palette[:5]),
            {
                "density_palette_scheme": "custom",
                "density_palette_selection_policy": "explicit_density_palette_rgb",
                "density_palette_scheme_probabilities": {"custom": 1.0},
                "level_one_distance_from_white": round(rgb_distance(tuple(palette[0]), (255, 255, 255)), 3),
                "level_one_minimum_distance_from_white": 52.0,
            },
        )
    schemes = configured_palette_schemes(params)
    supported = tuple(str(name) for name in SUPPORTED_DENSITY_PALETTE_SCHEMES if str(name) in schemes)
    if not supported:
        supported = tuple(sorted(schemes))
    scheme, probabilities = resolve_chart_axis_variant(
        params=params,
        gen_defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=supported,
        task_id=f"{SCENE_NAMESPACE}.palette",
        explicit_key="density_palette_scheme",
        weights_key="density_palette_scheme_weights",
        balance_flag_key="balanced_density_palette_scheme_sampling",
        axis_namespace="density_palette_scheme",
    )
    palette = tuple(schemes[str(scheme)][:5])
    return (
        str(scheme),
        palette,
        {
            "density_palette_scheme": str(scheme),
            "density_palette_selection_policy": "weighted_density_palette_scheme",
            "density_palette_scheme_probabilities": dict(probabilities),
            "level_one_distance_from_white": round(rgb_distance(tuple(palette[0]), (255, 255, 255)), 3),
            "level_one_minimum_distance_from_white": 52.0,
        },
    )


def jittered_margins(params: Mapping[str, Any], *, instance_seed: int) -> tuple[int, int, int, int, Dict[str, Any]]:
    return apply_layout_jitter_to_margins(
        left_px=render_int(params, "plot_margin_left_px", 96),
        right_px=render_int(params, "plot_margin_right_px", 238),
        top_px=render_int(params, "plot_margin_top_px", 94),
        bottom_px=render_int(params, "plot_margin_bottom_px", 92),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )


__all__ = [
    "AXIS_LABELS",
    "BBox",
    "DOMAIN",
    "GEN_DEFAULTS",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE_ID",
    "PROMPT_DEFAULTS",
    "RGB",
    "RENDER_DEFAULTS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_DENSITY_PALETTE_SCHEMES",
    "balanced_int",
    "generation_int",
    "jittered_margins",
    "render_int",
    "render_rgb",
    "resolve_bounds",
    "resolve_density_palette",
]
