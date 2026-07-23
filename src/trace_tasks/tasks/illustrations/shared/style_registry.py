"""Shared renderer-native art styles for illustration tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.sampling import normalize_positive_weights
from ...shared.config_defaults import group_default


RGB = Tuple[int, int, int]

ART_STYLE_REGISTRY_VERSION = "illustration_art_styles_v1"


@dataclass(frozen=True)
class IllustrationArtStyle:
    """One renderer-native non-semantic illustration art style."""

    style_id: str
    display_name: str
    outline_rgb: RGB | None
    outline_width_px: int
    shadow: bool
    color_mode: str


ART_STYLES: Dict[str, IllustrationArtStyle] = {
    "flat_vector": IllustrationArtStyle("flat_vector", "flat vector", (79, 84, 94), 1, False, "normal"),
    "outlined_cartoon": IllustrationArtStyle("outlined_cartoon", "outlined cartoon", (35, 39, 48), 3, False, "normal"),
    "paper_cutout": IllustrationArtStyle("paper_cutout", "paper cutout", (252, 252, 247), 5, True, "paper"),
    "soft_shadow": IllustrationArtStyle("soft_shadow", "soft shadow", (79, 84, 94), 1, True, "normal"),
    "ink_sketch": IllustrationArtStyle("ink_sketch", "ink sketch", (28, 32, 38), 2, False, "ink"),
    "watercolor_wash": IllustrationArtStyle("watercolor_wash", "watercolor wash", (86, 94, 108), 1, True, "watercolor"),
    "blueprint_line": IllustrationArtStyle("blueprint_line", "blueprint line", (36, 91, 143), 2, False, "blueprint"),
}

STYLE_IDS: Tuple[str, ...] = tuple(ART_STYLES)


def art_style_trace(style_id: str) -> Dict[str, str]:
    """Return compact trace metadata for one resolved illustration style."""

    style = ART_STYLES.get(str(style_id), ART_STYLES["flat_vector"])
    return {
        "art_style_id": str(style.style_id),
        "art_style_name": str(style.display_name),
        "art_style_registry_version": ART_STYLE_REGISTRY_VERSION,
    }


def style_outline_params(style_id: str) -> Tuple[RGB | None, int, bool]:
    """Return outline color, outline width, and shadow flag for a style."""

    style = ART_STYLES.get(str(style_id), ART_STYLES["outlined_cartoon"])
    return style.outline_rgb, int(style.outline_width_px), bool(style.shadow)


def _blend_rgb(a: RGB, b: RGB, t: float) -> RGB:
    return tuple(int(round(float(a[i]) * (1.0 - float(t)) + float(b[i]) * float(t))) for i in range(3))


def style_object_colors(style_id: str, primary: RGB, accent: RGB) -> Tuple[RGB, RGB]:
    """Adjust object colors for renderer-native styles."""

    mode = ART_STYLES.get(str(style_id), ART_STYLES["flat_vector"]).color_mode
    if mode == "ink":
        primary_l = int(round(sum(int(v) for v in primary) / 3.0))
        accent_l = int(round(sum(int(v) for v in accent) / 3.0))
        return (
            _blend_rgb((244, 244, 240), (172, 176, 184), max(0.12, min(0.48, primary_l / 510.0))),
            _blend_rgb((252, 252, 248), (118, 124, 136), max(0.08, min(0.42, accent_l / 560.0))),
        )
    if mode == "watercolor":
        return _blend_rgb(primary, (255, 255, 255), 0.34), _blend_rgb(accent, (255, 255, 255), 0.42)
    if mode == "blueprint":
        primary_l = int(round(sum(int(v) for v in primary) / 3.0))
        accent_l = int(round(sum(int(v) for v in accent) / 3.0))
        return (
            _blend_rgb((122, 185, 231), (224, 242, 255), max(0.0, min(1.0, primary_l / 255.0))),
            _blend_rgb((61, 130, 190), (183, 224, 252), max(0.0, min(1.0, accent_l / 255.0))),
        )
    if mode == "paper":
        return _blend_rgb(primary, (250, 246, 233), 0.12), _blend_rgb(accent, (250, 246, 233), 0.18)
    return tuple(int(v) for v in primary), tuple(int(v) for v in accent)


def resolve_art_style_weights(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    style_ids: Sequence[str] = STYLE_IDS,
) -> Dict[str, float]:
    """Resolve style weights with explicit art-style support and legacy fallback."""

    valid = tuple(str(value) for value in style_ids if str(value) in ART_STYLES)
    if not valid:
        raise ValueError("style_ids resolved no supported illustration art styles")

    support_raw = params.get("art_style_support", group_default(render_defaults, "art_style_support", valid))
    if support_raw is None:
        support = valid
    else:
        if not isinstance(support_raw, Sequence) or isinstance(support_raw, (str, bytes)):
            raise ValueError("art_style_support must be a sequence")
        support = tuple(str(value) for value in support_raw if str(value) in set(valid))
    support = tuple(dict.fromkeys(support))
    if not support:
        raise ValueError("art_style_support resolved no supported styles")

    explicit = params.get("art_style_id")
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(support):
            raise ValueError(f"art_style_id must be one of {support}")
        return {selected: 1.0}

    raw = params.get("art_style_weights")
    if raw is None:
        raw = params.get("style_weights", group_default(render_defaults, "art_style_weights", None))
    if raw is None:
        raw = group_default(render_defaults, "style_weights", {style: 1.0 for style in support})
    if not isinstance(raw, Mapping):
        raise ValueError("art_style_weights/style_weights must be a mapping")
    weights = {style: float(raw.get(style, 0.0)) for style in support}
    return normalize_positive_weights(weights, default_keys=support)


__all__ = [
    "ART_STYLE_REGISTRY_VERSION",
    "ART_STYLES",
    "IllustrationArtStyle",
    "STYLE_IDS",
    "art_style_trace",
    "resolve_art_style_weights",
    "style_object_colors",
    "style_outline_params",
]
