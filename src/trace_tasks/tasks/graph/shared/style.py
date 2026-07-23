"""Shared graph-domain style helpers for named-color visual themes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence, Tuple

from ...shared.named_colors import available_named_colors, darken_color, named_color


Color = Tuple[int, int, int]
SUPPORTED_NODE_COLOR_NAMES: Tuple[str, ...] = tuple(str(name) for name, _ in available_named_colors())
SUPPORTED_GRAPH_THEME_TONES: Tuple[str, ...] = ("standard", "soft", "crisp")
SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS: Tuple[str, ...] = ("default", "cool", "warm", "mint", "paper")


@dataclass(frozen=True)
class GraphNamedColorTheme:
    """Resolved per-instance graph color theme derived from one named node color."""

    node_color_name: str
    theme_tone: str
    background_color_rgb: Color
    panel_fill_rgb: Color
    panel_border_rgb: Color
    title_color_rgb: Color
    edge_color_rgb: Color
    node_fill_rgb: Color
    node_border_rgb: Color
    label_text_rgb: Color
    label_stroke_rgb: Color


def _blend_with_white(color: Sequence[int], *, color_weight: float) -> Color:
    """Blend one RGB color toward white by the requested color weight."""

    weight = max(0.0, min(1.0, float(color_weight)))
    if len(color) < 3:
        raise ValueError("graph color blends require three RGB channels")
    return tuple(
        max(0, min(255, int(round((255.0 * (1.0 - weight)) + (float(int(channel)) * weight)))))
        for channel in color[:3]
    )


def _relative_luminance(color: Sequence[int]) -> float:
    """Return one simple perceived-luminance estimate in ``[0, 1]``."""

    if len(color) < 3:
        raise ValueError("graph luminance requires three RGB channels")
    red, green, blue = [float(int(channel)) / 255.0 for channel in color[:3]]
    return float((0.2126 * red) + (0.7152 * green) + (0.0722 * blue))


def build_graph_named_color_theme(node_color_name: str, *, theme_tone: str = "standard") -> GraphNamedColorTheme:
    """Resolve one readable graph theme from a canonical named node color."""

    tone = str(theme_tone).strip().lower()
    if tone not in SUPPORTED_GRAPH_THEME_TONES:
        tone = "standard"

    node_fill_rgb = tuple(int(channel) for channel in named_color(str(node_color_name)))
    tone_params = {
        "standard": {
            "node_border_factor": 0.58,
            "edge_factor": 0.72,
            "title_factor": 0.46,
            "panel_border_weight": 0.28,
            "panel_fill_weight": 0.06,
            "background_weight": 0.03,
        },
        "soft": {
            "node_border_factor": 0.66,
            "edge_factor": 0.82,
            "title_factor": 0.54,
            "panel_border_weight": 0.20,
            "panel_fill_weight": 0.045,
            "background_weight": 0.02,
        },
        "crisp": {
            "node_border_factor": 0.48,
            "edge_factor": 0.60,
            "title_factor": 0.38,
            "panel_border_weight": 0.34,
            "panel_fill_weight": 0.075,
            "background_weight": 0.04,
        },
    }[tone]
    node_border_rgb = darken_color(node_fill_rgb, factor=float(tone_params["node_border_factor"]))
    edge_color_rgb = darken_color(node_fill_rgb, factor=float(tone_params["edge_factor"]))
    title_color_rgb = darken_color(node_fill_rgb, factor=float(tone_params["title_factor"]))
    panel_border_rgb = _blend_with_white(node_fill_rgb, color_weight=float(tone_params["panel_border_weight"]))
    panel_fill_rgb = _blend_with_white(node_fill_rgb, color_weight=float(tone_params["panel_fill_weight"]))
    background_color_rgb = _blend_with_white(node_fill_rgb, color_weight=float(tone_params["background_weight"]))

    if _relative_luminance(node_fill_rgb) >= 0.55:
        label_text_rgb = tuple(int(channel) for channel in darken_color(node_fill_rgb, factor=0.23))
        label_stroke_rgb = (255, 255, 255)
    else:
        label_text_rgb = (255, 255, 255)
        label_stroke_rgb = tuple(int(channel) for channel in node_border_rgb)

    return GraphNamedColorTheme(
        node_color_name=str(node_color_name),
        theme_tone=str(tone),
        background_color_rgb=tuple(int(channel) for channel in background_color_rgb),
        panel_fill_rgb=tuple(int(channel) for channel in panel_fill_rgb),
        panel_border_rgb=tuple(int(channel) for channel in panel_border_rgb),
        title_color_rgb=tuple(int(channel) for channel in title_color_rgb),
        edge_color_rgb=tuple(int(channel) for channel in edge_color_rgb),
        node_fill_rgb=tuple(int(channel) for channel in node_fill_rgb),
        node_border_rgb=tuple(int(channel) for channel in node_border_rgb),
        label_text_rgb=tuple(int(channel) for channel in label_text_rgb),
        label_stroke_rgb=tuple(int(channel) for channel in label_stroke_rgb),
    )


def apply_graph_panel_style(
    color_theme: GraphNamedColorTheme,
    *,
    panel_style_variant: str = "default",
) -> GraphNamedColorTheme:
    """Apply a non-semantic background/panel surface style to a graph theme.

    This intentionally leaves node, edge, arrow, and label colors unchanged so
    graph readability and topology grounding stay controlled by the existing
    graph render parameters.
    """

    style = str(panel_style_variant).strip().lower()
    if style not in SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS:
        style = "default"
    if style == "cool":
        return replace(
            color_theme,
            background_color_rgb=(246, 248, 252),
            panel_fill_rgb=(253, 254, 255),
            panel_border_rgb=(212, 220, 232),
        )
    if style == "warm":
        return replace(
            color_theme,
            background_color_rgb=(250, 247, 242),
            panel_fill_rgb=(255, 253, 248),
            panel_border_rgb=(225, 216, 203),
        )
    if style == "mint":
        return replace(
            color_theme,
            background_color_rgb=(245, 250, 248),
            panel_fill_rgb=(253, 255, 254),
            panel_border_rgb=(207, 224, 218),
        )
    if style == "paper":
        return replace(
            color_theme,
            background_color_rgb=(250, 249, 245),
            panel_fill_rgb=(255, 254, 250),
            panel_border_rgb=(222, 218, 208),
        )
    return color_theme


__all__ = [
    "GraphNamedColorTheme",
    "SUPPORTED_GRAPH_PANEL_STYLE_VARIANTS",
    "SUPPORTED_GRAPH_THEME_TONES",
    "SUPPORTED_NODE_COLOR_NAMES",
    "apply_graph_panel_style",
    "build_graph_named_color_theme",
]
