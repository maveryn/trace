"""Icons-domain adapter for shared panel/canvas render styles."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from ...shared.config_defaults import group_default
from ...shared.visual_style.panel import (
    DEFAULT_PANEL_SCENE_STYLE,
    PANEL_SCENE_TREATMENTS,
    PanelSceneStyle,
    draw_panel_plain_chrome,
    draw_panel_scene_chrome,
    make_panel_scene_background,
    panel_scene_style_metadata,
    resolve_panel_scene_style,
)


IconCanvasStyle = PanelSceneStyle

ICON_CANVAS_TREATMENTS: tuple[str, ...] = tuple(PANEL_SCENE_TREATMENTS)

DEFAULT_ICON_CANVAS_TREATMENT_WEIGHTS: dict[str, float] = {
    "bare_canvas": 0.12,
    "plain_sheet": 0.18,
    "matte_sheet": 0.12,
    "thin_frame": 0.10,
    "soft_panel": 0.12,
    "margin_sheet": 0.09,
    "dot_sheet": 0.09,
    "worksheet_panel": 0.08,
    "index_card": 0.05,
    "printout_panel": 0.05,
}

DEFAULT_ICON_CANVAS_PALETTE_WEIGHTS: dict[str, float] = {
    "plain_neutral": 0.28,
    "cool_blue": 0.18,
    "mint_green": 0.16,
    "cyan_lab": 0.14,
    "warm_paper": 0.14,
    "teal_card": 0.10,
}


def _resolve_bool(value: Any, fallback: bool) -> bool:
    """Resolve one permissive boolean config value."""

    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on", "always"}:
            return True
        if text in {"0", "false", "no", "n", "off", "never"}:
            return False
    return bool(fallback)


def _resolve_treatments(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = params.get("icon_canvas_treatment", None)
    if explicit is not None and str(explicit).strip():
        requested = (str(explicit).strip(),)
    else:
        raw = params.get(
            "icon_canvas_treatments",
            group_default(render_defaults, "icon_canvas_treatments", ICON_CANVAS_TREATMENTS),
        )
        requested = (
            tuple(str(value).strip() for value in raw if str(value).strip())
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes))
            else ()
        )
    allowed = set(ICON_CANVAS_TREATMENTS)
    treatments = tuple(value for value in requested if value in allowed)
    if not treatments:
        treatments = ICON_CANVAS_TREATMENTS
    return treatments


def _resolve_weight_map(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: Mapping[str, float],
) -> dict[str, float]:
    raw = params.get(str(key), group_default(render_defaults, str(key), fallback))
    if not isinstance(raw, Mapping):
        raw = fallback
    out: dict[str, float] = {}
    for item_key, item_value in raw.items():
        try:
            weight = float(item_value)
        except Exception:
            continue
        if weight > 0.0:
            out[str(item_key)] = float(weight)
    return out or dict(fallback)


def _normalized_weight_map(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0.0:
        return {}
    return {str(key): float(value) / total for key, value in sorted(weights.items()) if float(value) > 0.0}


def resolve_icon_canvas_style(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int | None,
    namespace: str = "icons.canvas_style",
) -> tuple[IconCanvasStyle | None, dict[str, Any]]:
    """Resolve one non-semantic icon canvas style and trace metadata."""

    enabled = _resolve_bool(
        params.get(
            "icon_canvas_style_enabled",
            group_default(render_defaults, "icon_canvas_style_enabled", True),
        ),
        True,
    )
    if not enabled:
        return None, {"enabled": False}
    treatments = _resolve_treatments(params, render_defaults)
    treatment_weights = _resolve_weight_map(
        params,
        render_defaults,
        "icon_canvas_treatment_weights",
        DEFAULT_ICON_CANVAS_TREATMENT_WEIGHTS,
    )
    palette_weights = _resolve_weight_map(
        params,
        render_defaults,
        "icon_canvas_palette_weights",
        DEFAULT_ICON_CANVAS_PALETTE_WEIGHTS,
    )
    style, metadata = resolve_panel_scene_style(
        instance_seed=int(instance_seed or 0),
        namespace=str(namespace),
        treatments=treatments,
        treatment_weights=treatment_weights,
        palette_weights=palette_weights,
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "enabled": True,
            "available_treatments": list(treatments),
            "treatment_probabilities": _normalized_weight_map(treatment_weights),
            "palette_probabilities": _normalized_weight_map(palette_weights),
        }
    )
    return style, metadata


def icon_canvas_style_with_chrome(
    style: IconCanvasStyle | None,
    *,
    background_rgb: Sequence[int],
    panel_fill_rgb: Sequence[int],
    panel_border_rgb: Sequence[int],
    header_text_rgb: Sequence[int],
    header_text_stroke_rgb: Sequence[int],
) -> IconCanvasStyle | None:
    """Return a style object whose chrome colors match resolved render params."""

    if style is None:
        return None
    return replace(
        style,
        background_rgb=tuple(int(value) for value in background_rgb),
        panel_fill_rgb=tuple(int(value) for value in panel_fill_rgb),
        panel_border_rgb=tuple(int(value) for value in panel_border_rgb),
        text_rgb=tuple(int(value) for value in header_text_rgb),
        text_stroke_rgb=tuple(int(value) for value in header_text_stroke_rgb),
    )


def icon_canvas_style_trace(style: IconCanvasStyle | None, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Serialize the icon canvas style without non-JSON runtime objects."""

    if style is None:
        base = {"enabled": False}
    else:
        base = panel_scene_style_metadata(style)
        base["enabled"] = True
    if metadata:
        for key, value in metadata.items():
            if str(key) in {"text_legibility", "text_color_policy"}:
                continue
            base[str(key)] = value
    return base


def make_icon_canvas_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: IconCanvasStyle | None,
    fallback_rgb: Sequence[int],
) -> Image.Image:
    """Create the background image for one icon scene."""

    if style is None:
        return Image.new("RGBA", (int(canvas_width), int(canvas_height)), tuple(int(value) for value in fallback_rgb) + (255,))
    image, _metadata = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
    )
    return image.convert("RGBA")


def draw_icon_panel_chrome(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[int],
    style: IconCanvasStyle | None,
    fallback_fill_rgb: Sequence[int],
    fallback_border_rgb: Sequence[int],
    radius: int,
    border_width: int = 2,
) -> None:
    """Draw one icon panel using the shared style when available."""

    if style is None:
        draw.rounded_rectangle(
            tuple(int(value) for value in bbox),
            radius=max(0, int(radius)),
            fill=tuple(int(value) for value in fallback_fill_rgb),
            outline=tuple(int(value) for value in fallback_border_rgb),
            width=max(1, int(border_width)),
        )
        return
    if str(style.treatment) == "bare_canvas":
        draw_panel_plain_chrome(
            draw,
            bbox=tuple(int(value) for value in bbox),
            style=style,
            radius=max(0, int(radius)),
            border_width=max(1, int(border_width)),
        )
        return
    draw_panel_scene_chrome(
        draw,
        bbox=tuple(int(value) for value in bbox),
        style=style,
        radius=max(0, int(radius)),
        border_width=max(1, int(border_width)),
    )


DEFAULT_ICON_CANVAS_STYLE = DEFAULT_PANEL_SCENE_STYLE


__all__ = [
    "DEFAULT_ICON_CANVAS_STYLE",
    "DEFAULT_ICON_CANVAS_PALETTE_WEIGHTS",
    "DEFAULT_ICON_CANVAS_TREATMENT_WEIGHTS",
    "ICON_CANVAS_TREATMENTS",
    "IconCanvasStyle",
    "draw_icon_panel_chrome",
    "icon_canvas_style_trace",
    "icon_canvas_style_with_chrome",
    "make_icon_canvas_background",
    "resolve_icon_canvas_style",
]
