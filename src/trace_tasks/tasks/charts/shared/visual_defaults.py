"""Chart-domain visual-default loader helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from ...shared.font_assets import font_asset_version, sample_font_family
from ...shared.render_variation import resolve_render_float, resolve_render_int, resolve_render_rgb
from ...shared.visual_defaults import (
    default_noise_fallback,
    load_scene_background_defaults,
    load_scene_noise_defaults,
)


def solid_light_background_fallback() -> Dict[str, Any]:
    """Return the canonical low-structure chart background fallback."""

    return {
        "enabled": True,
        "styles": {
            "solid_light": {
                "kind": "solid",
                "color": [248, 248, 248],
            }
        },
        "weights": {"solid_light": 1.0},
    }


def load_chart_scene_background_defaults(*, scene_id: str) -> Dict[str, Any]:
    """Load chart-scene background config with the canonical fallback."""

    return load_scene_background_defaults(
        domain="charts",
        scene_id=str(scene_id),
        fallback=solid_light_background_fallback(),
        merge_with_fallback=True,
    )


def load_chart_scene_noise_defaults(*, scene_id: str, apply_prob: float) -> Dict[str, Any]:
    """Load chart-scene post-image noise config with the canonical fallback."""

    return load_scene_noise_defaults(
        domain="charts",
        scene_id=str(scene_id),
        fallback=default_noise_fallback(apply_prob=float(apply_prob)),
        merge_with_fallback=False,
    )


def render_style_seed(params: Mapping[str, Any]) -> int:
    """Resolve the deterministic render-style seed from instance params."""

    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def resolve_chart_render_int(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: int,
    *,
    namespace: str,
) -> int:
    """Resolve one integer render parameter with chart-standard style sampling."""

    return int(
        resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(fallback),
            instance_seed=render_style_seed(params),
            namespace=str(namespace),
        )
    )


def resolve_chart_render_float(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: float,
    *,
    namespace: str,
) -> float:
    """Resolve one float render parameter with chart-standard style sampling."""

    return float(
        resolve_render_float(
            params,
            render_defaults,
            str(key),
            float(fallback),
            instance_seed=render_style_seed(params),
            namespace=str(namespace),
        )
    )


def resolve_chart_render_rgb(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    *,
    namespace: str,
) -> tuple[int, int, int]:
    """Resolve one RGB render parameter with chart-standard style sampling."""

    return resolve_render_rgb(
        params,
        render_defaults,
        str(key),
        fallback,
        instance_seed=render_style_seed(params),
        namespace=str(namespace),
    )


def coerce_rgb(value: Any, fallback: Sequence[int]) -> tuple[int, int, int]:
    """Coerce a config value to a clamped RGB tuple."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return tuple(int(channel) for channel in fallback[:3])  # type: ignore[index]
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[index]


def relative_luminance(color: Sequence[int]) -> float:
    """Return WCAG-style relative luminance for an RGB color."""

    channels = []
    for channel in color[:3]:
        value = max(0.0, min(1.0, float(channel) / 255.0))
        channels.append(value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4)
    while len(channels) < 3:
        channels.append(0.0)
    return float(0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2])


def sample_chart_font_family(
    *,
    instance_seed: int,
    namespace: str,
    params: Mapping[str, Any],
    exclude_tags: Sequence[str] = ("display",),
) -> str:
    """Sample one chart text font family from the shared vendored font pool."""

    return str(
        sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            params=params,
            exclude_tags=tuple(str(tag) for tag in exclude_tags),
            explicit_key="chart_font_family",
            weights_key="chart_font_family_weights",
        )
    )


def chart_font_asset_metadata(chart_font_family: str) -> Dict[str, str]:
    """Return trace metadata for the sampled chart text font."""

    return {
        "font_asset_version": str(font_asset_version()),
        "chart_font_family": str(chart_font_family),
    }


__all__ = [
    "chart_font_asset_metadata",
    "coerce_rgb",
    "load_chart_scene_background_defaults",
    "load_chart_scene_noise_defaults",
    "relative_luminance",
    "render_style_seed",
    "sample_chart_font_family",
    "resolve_chart_render_float",
    "resolve_chart_render_int",
    "resolve_chart_render_rgb",
    "solid_light_background_fallback",
]
