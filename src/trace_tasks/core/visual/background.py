"""Deterministic scene background-style helpers for Trace tasks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..sampling import normalize_positive_weights, weighted_choice
from ..seed import spawn_rng
from .ranges import normalize_int_range


_DEFAULT_BASE_COLOR = (245, 245, 245)

_DEFAULT_BACKGROUND_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "styles": {
        "solid_default": {
            "kind": "solid",
            "color": list(_DEFAULT_BASE_COLOR),
        }
    },
    "weights": {"solid_default": 1.0},
}

_ALLOWED_STYLE_KINDS = {"solid", "grid"}
_GRID_STYLE_VARIANT_KEYS = {
    "base_color",
    "line_color",
    "major_line_color",
    "axis_color",
    "center_point_color",
    "origin_label_color",
    "color_variation_enabled",
    "base_color_jitter",
    "line_color_jitter",
    "major_line_darken_range",
    "axis_darken_range",
    "center_point_darken_extra_range",
    "origin_label_darken_extra_range",
}
_DEFAULT_GRID_STYLE: Dict[str, Any] = {
    "kind": "grid",
    "base_color": list(_DEFAULT_BASE_COLOR),
    "line_color": [220, 220, 220],
    "spacing": 24,
    "outer_margin_px": 0,
    "line_width": 1,
    "major_every": 0,
    "major_line_color": [220, 220, 220],
    "major_line_width": 1,
    "axis_enabled": False,
    "axis_color": [220, 220, 220],
    "axis_line_width": 2,
    "axis_arrows_enabled": False,
    "axis_arrow_size": 8,
    "center_point_enabled": False,
    "center_point_color": [220, 220, 220],
    "center_point_radius": 2,
    "color_variation_enabled": False,
    "base_color_jitter": [0, 0],
    "line_color_jitter": [0, 0],
    "major_line_darken_range": [0, 0],
    "axis_darken_range": [0, 0],
    "center_point_darken_extra_range": [0, 0],
    "origin_label_darken_extra_range": [0, 0],
    "axis_scale_labels_enabled": False,
    "axis_scale_label_max_abs": 0,
    "origin_label_enabled": False,
    "origin_label_text": "(0,0)",
    "origin_label_color": [220, 220, 220],
    "origin_fraction_x": 0.5,
    "origin_fraction_y": 0.5,
    "supersample_scale": 1,
}


def _to_int(value: Any, fallback: int) -> int:
    """Parse an integer with fallback for invalid values."""
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _to_float(value: Any, fallback: float) -> float:
    """Parse a float with fallback for invalid values."""
    try:
        return float(value)
    except Exception:
        return float(fallback)


def _to_bool(value: Any, fallback: bool) -> bool:
    """Parse a boolean-like value with fallback for invalid values."""
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(fallback)


def _clamp_fraction(value: Any, fallback: float) -> float:
    """Clamp one fractional placement control into a safe interior band."""

    return max(0.10, min(0.90, _to_float(value, fallback)))


def _normalize_rgb(value: Any, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Normalize RGB-like input into a clamped 3-channel integer tuple."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (
            max(0, min(255, _to_int(value[0], fallback[0]))),
            max(0, min(255, _to_int(value[1], fallback[1]))),
            max(0, min(255, _to_int(value[2], fallback[2]))),
        )
    return tuple(fallback)


def _darken_rgb(color: Tuple[int, int, int], *, factor: float) -> Tuple[int, int, int]:
    """Return one darker RGB color by multiplicative factor in [0, 1]."""
    scale = max(0.0, min(1.0, float(factor)))
    return (
        max(0, min(255, int(round(float(color[0]) * scale)))),
        max(0, min(255, int(round(float(color[1]) * scale)))),
        max(0, min(255, int(round(float(color[2]) * scale)))),
    )


def _shift_rgb_by_delta(color: Tuple[int, int, int], *, delta: int) -> Tuple[int, int, int]:
    """Shift one RGB color by additive integer delta with channel clamping."""
    offset = int(delta)
    return (
        max(0, min(255, int(color[0]) + int(offset))),
        max(0, min(255, int(color[1]) + int(offset))),
        max(0, min(255, int(color[2]) + int(offset))),
    )


def _darken_rgb_by_delta(color: Tuple[int, int, int], *, delta: int) -> Tuple[int, int, int]:
    """Darken one RGB color by subtractive integer delta with channel clamping."""
    amount = max(0, int(delta))
    return (
        max(0, min(255, int(color[0]) - int(amount))),
        max(0, min(255, int(color[1]) - int(amount))),
        max(0, min(255, int(color[2]) - int(amount))),
    )


def coerce_grid_style_spec(
    spec: Mapping[str, Any],
    *,
    fallback_style: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalize one grid-style spec into deterministic, valid values."""
    fallback = dict(_DEFAULT_GRID_STYLE)
    if isinstance(fallback_style, Mapping):
        fallback.update(dict(fallback_style))

    merged = dict(fallback)
    if isinstance(spec, Mapping):
        merged.update(dict(spec))

    line_width = max(1, _to_int(merged.get("line_width"), fallback.get("line_width", 1)))
    major_line_color = _normalize_rgb(
        merged.get("major_line_color"),
        _normalize_rgb(merged.get("line_color"), (220, 220, 220)),
    )
    axis_color = _normalize_rgb(merged.get("axis_color"), major_line_color)
    axis_line_width = max(line_width, _to_int(merged.get("axis_line_width"), max(2, line_width + 1)))
    base_color_jitter = normalize_int_range(
        merged.get("base_color_jitter"),
        fallback_min=0,
        fallback_max=0,
    )
    line_color_jitter = normalize_int_range(
        merged.get("line_color_jitter"),
        fallback_min=0,
        fallback_max=0,
    )
    major_line_darken_range = normalize_int_range(
        merged.get("major_line_darken_range"),
        fallback_min=0,
        fallback_max=0,
    )
    axis_darken_range = normalize_int_range(
        merged.get("axis_darken_range"),
        fallback_min=0,
        fallback_max=0,
    )
    center_point_darken_extra_range = normalize_int_range(
        merged.get("center_point_darken_extra_range"),
        fallback_min=0,
        fallback_max=0,
    )
    origin_label_darken_extra_range = normalize_int_range(
        merged.get("origin_label_darken_extra_range"),
        fallback_min=0,
        fallback_max=0,
    )

    out = {
        "kind": "grid",
        "base_color": list(_normalize_rgb(merged.get("base_color"), _DEFAULT_BASE_COLOR)),
        "line_color": list(_normalize_rgb(merged.get("line_color"), (220, 220, 220))),
        "spacing": max(4, _to_int(merged.get("spacing"), 24)),
        "outer_margin_px": max(0, _to_int(merged.get("outer_margin_px"), 0)),
        "line_width": int(line_width),
        "major_every": max(0, _to_int(merged.get("major_every"), 0)),
        "major_line_color": list(major_line_color),
        "major_line_width": max(line_width, _to_int(merged.get("major_line_width"), line_width)),
        "axis_enabled": _to_bool(merged.get("axis_enabled"), False),
        "axis_color": list(axis_color),
        "axis_line_width": int(axis_line_width),
        "axis_arrows_enabled": _to_bool(merged.get("axis_arrows_enabled"), False),
        "axis_arrow_size": max(4, _to_int(merged.get("axis_arrow_size"), max(8, axis_line_width * 4))),
        "center_point_enabled": _to_bool(merged.get("center_point_enabled"), False),
        "center_point_color": list(_normalize_rgb(merged.get("center_point_color"), axis_color)),
        "center_point_radius": max(1, _to_int(merged.get("center_point_radius"), max(2, axis_line_width))),
        "color_variation_enabled": _to_bool(merged.get("color_variation_enabled"), False),
        "base_color_jitter": [int(base_color_jitter[0]), int(base_color_jitter[1])],
        "line_color_jitter": [int(line_color_jitter[0]), int(line_color_jitter[1])],
        "major_line_darken_range": [int(major_line_darken_range[0]), int(major_line_darken_range[1])],
        "axis_darken_range": [int(axis_darken_range[0]), int(axis_darken_range[1])],
        "center_point_darken_extra_range": [
            int(center_point_darken_extra_range[0]),
            int(center_point_darken_extra_range[1]),
        ],
        "origin_label_darken_extra_range": [
            int(origin_label_darken_extra_range[0]),
            int(origin_label_darken_extra_range[1]),
        ],
        "axis_scale_labels_enabled": _to_bool(merged.get("axis_scale_labels_enabled"), False),
        "axis_scale_label_max_abs": max(0, _to_int(merged.get("axis_scale_label_max_abs"), 0)),
        "origin_label_enabled": _to_bool(merged.get("origin_label_enabled"), False),
        "origin_label_text": str(merged.get("origin_label_text", "(0,0)")).strip() or "(0,0)",
        "origin_label_color": list(_normalize_rgb(merged.get("origin_label_color"), axis_color)),
        "origin_fraction_x": _clamp_fraction(
            merged.get("origin_fraction_x"),
            _to_float(fallback.get("origin_fraction_x", 0.5), 0.5),
        ),
        "origin_fraction_y": _clamp_fraction(
            merged.get("origin_fraction_y"),
            _to_float(fallback.get("origin_fraction_y", 0.5), 0.5),
        ),
        "supersample_scale": max(1, min(4, _to_int(merged.get("supersample_scale"), 1))),
    }
    raw_origin = merged.get("origin_pixel")
    if isinstance(raw_origin, (list, tuple)) and len(raw_origin) >= 2:
        out["origin_pixel"] = [
            max(0, _to_int(raw_origin[0], 0)),
            max(0, _to_int(raw_origin[1], 0)),
        ]
    return out


def _normalize_styles(raw: Any, fallback: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize style definitions to a validated internal representation."""
    if not isinstance(raw, Mapping):
        raw = fallback
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in raw.items():
        if not isinstance(spec, Mapping):
            continue
        style_name = str(name).strip()
        if not style_name:
            continue
        kind = str(spec.get("kind", "solid")).strip().lower()
        if kind not in _ALLOWED_STYLE_KINDS:
            continue
        if kind == "solid":
            out[style_name] = {
                "kind": "solid",
                "color": list(_normalize_rgb(spec.get("color"), _DEFAULT_BASE_COLOR)),
            }
        else:
            grid_style = coerce_grid_style_spec(spec)
            variant_specs = _normalize_grid_style_variants(spec.get("style_variants"))
            if variant_specs:
                grid_style["style_variants"] = variant_specs
                grid_style["style_variant_weights"] = _normalize_style_variant_weights(
                    spec.get("style_variant_weights"),
                    variant_specs=variant_specs,
                )
            out[style_name] = grid_style
    return out


def _normalize_grid_style_variants(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Return color-only grid style variants that cannot alter graph geometry."""

    if not isinstance(raw, Mapping):
        return {}
    variants: Dict[str, Dict[str, Any]] = {}
    for name, spec in raw.items():
        variant_name = str(name).strip()
        if not variant_name or not isinstance(spec, Mapping):
            continue
        sanitized = {
            str(key): deepcopy(value)
            for key, value in spec.items()
            if str(key) in _GRID_STYLE_VARIANT_KEYS
        }
        if sanitized:
            variants[str(variant_name)] = sanitized
    return variants


def _normalize_style_variant_weights(raw: Any, *, variant_specs: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Normalize optional weights over sanitized style variants."""

    weights = {}
    if isinstance(raw, Mapping):
        weights = {str(name): _to_float(raw.get(str(name), 0.0), 0.0) for name in variant_specs.keys()}
    try:
        return normalize_positive_weights(weights, default_keys=variant_specs.keys())
    except ValueError:
        return {}


def _resolve_style_variant(
    style_spec: Mapping[str, Any],
    *,
    rng,
) -> Dict[str, Any]:
    """Merge one deterministic color-only style variant into a style spec."""

    variants = style_spec.get("style_variants")
    if not isinstance(variants, Mapping) or not variants:
        return dict(style_spec)
    probabilities = _normalize_style_variant_weights(
        style_spec.get("style_variant_weights"),
        variant_specs=variants,
    )
    if not probabilities:
        return {
            str(key): deepcopy(value)
            for key, value in style_spec.items()
            if str(key) not in {"style_variants", "style_variant_weights"}
        }
    selected_variant = weighted_choice(rng, probabilities, sort_keys=True)
    base = {
        str(key): deepcopy(value)
        for key, value in style_spec.items()
        if str(key) not in {"style_variants", "style_variant_weights"}
    }
    selected_spec = variants.get(str(selected_variant), {})
    if isinstance(selected_spec, Mapping):
        base.update({str(key): deepcopy(value) for key, value in selected_spec.items()})
    base["style_variant"] = str(selected_variant)
    base["style_variant_probabilities"] = dict(probabilities)
    return base


def _normalize_weights(raw: Any, styles: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Normalize style weights and return a probability map over style names."""
    if not isinstance(raw, Mapping):
        raw = {}
    weights = {name: _to_float(raw.get(name, 0.0), 0.0) for name in styles.keys()}
    try:
        return normalize_positive_weights(weights, default_keys=styles.keys())
    except ValueError:
        return {}


def _normalize_default_config(default_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Normalize scene background defaults against global fallbacks."""
    base = deepcopy(_DEFAULT_BACKGROUND_CONFIG)
    if not isinstance(default_config, Mapping):
        return base

    enabled = bool(default_config.get("enabled", base["enabled"]))
    styles = _normalize_styles(default_config.get("styles"), fallback=base["styles"])
    weights = _normalize_weights(default_config.get("weights"), styles)
    if not styles:
        styles = deepcopy(base["styles"])
        weights = deepcopy(base["weights"])
    return {
        "enabled": enabled,
        "styles": styles,
        "weights": weights,
    }


def _resolve_background_overrides(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Collect background overrides from nested task params."""
    merged: Dict[str, Any] = {}
    visual = params.get("visual")
    if isinstance(visual, Mapping):
        background_cfg = visual.get("background")
        if isinstance(background_cfg, Mapping):
            merged.update(dict(background_cfg))
    return merged


def _resolve_background_config(params: Mapping[str, Any], *, default_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Merge defaults and overrides into one background rendering config."""
    base = _normalize_default_config(default_config)
    overrides = _resolve_background_overrides(params)

    styles = _normalize_styles(overrides.get("styles", base.get("styles", {})), fallback=base.get("styles", {}))
    weights = _normalize_weights(overrides.get("weights", base.get("weights", {})), styles)
    if not styles:
        styles = deepcopy(base["styles"])
        weights = deepcopy(base["weights"])

    return {
        "enabled": bool(overrides.get("enabled", base.get("enabled", True))),
        "styles": styles,
        "weights": weights,
        "style_name": str(overrides.get("style_name", "")).strip(),
    }


def _draw_grid_lines(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    spacing: int,
    line_color: Tuple[int, int, int],
    line_width: int,
    major_every: int,
    major_line_color: Tuple[int, int, int],
    major_line_width: int,
    inset: int = 0,
    origin_x: int | None = None,
    origin_y: int | None = None,
) -> None:
    """Draw minor/major graph-paper lines with deterministic boundary clamping."""
    left = max(0, int(inset))
    top = max(0, int(inset))
    right = max(int(left), int(width) - 1 - int(inset))
    bottom = max(int(top), int(height) - 1 - int(inset))
    if right < left or bottom < top:
        return

    origin_x_px = (
        int(origin_x)
        if origin_x is not None
        else _axis_origin_for_bounds(lower=int(left), upper=int(right), spacing=int(spacing))
    )
    origin_y_px = (
        int(origin_y)
        if origin_y is not None
        else _axis_origin_for_bounds(lower=int(top), upper=int(bottom), spacing=int(spacing))
    )

    max_x_start_minor = max(int(left), int(right) - int(line_width) + 1)
    max_x_start_major = max(int(left), int(right) - int(major_line_width) + 1)
    x_positions = _lattice_positions_within_bounds(
        lower=int(left),
        upper=int(right),
        spacing=int(spacing),
        origin=int(origin_x_px),
    )
    for idx, x in enumerate(x_positions):
        use_major = bool(major_every) and int(idx) % int(major_every) == 0
        color = major_line_color if use_major else line_color
        width_px = int(major_line_width if use_major else line_width)
        max_x_start = max_x_start_major if use_major else max_x_start_minor
        x_start = max(int(left), min(int(x), int(max_x_start)))
        if x_start > int(right):
            continue
        draw.rectangle([x_start, int(top), min(int(right), x_start + width_px - 1), int(bottom)], fill=color)

    max_y_start_minor = max(int(top), int(bottom) - int(line_width) + 1)
    max_y_start_major = max(int(top), int(bottom) - int(major_line_width) + 1)
    y_positions = _lattice_positions_within_bounds(
        lower=int(top),
        upper=int(bottom),
        spacing=int(spacing),
        origin=int(origin_y_px),
    )
    for idx, y in enumerate(y_positions):
        use_major = bool(major_every) and int(idx) % int(major_every) == 0
        color = major_line_color if use_major else line_color
        width_px = int(major_line_width if use_major else line_width)
        max_y_start = max_y_start_major if use_major else max_y_start_minor
        y_start = max(int(top), min(int(y), int(max_y_start)))
        if y_start > int(bottom):
            continue
        draw.rectangle([int(left), y_start, int(right), min(int(bottom), y_start + width_px - 1)], fill=color)


def _axis_origin_for_bounds(*, lower: int, upper: int, spacing: int, fraction: float = 0.5) -> int:
    """Return one axis coordinate within one bounded drawable interval."""
    lo = int(lower)
    hi = int(upper)
    if hi < lo:
        return int(lo)
    placement = float(lo) + (max(0.0, min(1.0, float(fraction))) * float(hi - lo))
    return max(int(lo), min(int(round(placement)), int(hi)))


def _lattice_positions_within_bounds(*, lower: int, upper: int, spacing: int, origin: int) -> List[int]:
    """Return lattice line positions in `[lower, upper]` anchored at `origin`."""
    lo = int(lower)
    hi = int(upper)
    if hi < lo:
        return []
    step = max(1, int(spacing))
    start = int(origin)
    while start - step >= lo:
        start -= step
    return [int(value) for value in range(int(start), int(hi) + 1, int(step))]


def compute_grid_axis_origin_for_canvas(
    *,
    width: int,
    height: int,
    spacing: int,
    inset: int = 0,
    x_fraction: float = 0.5,
    y_fraction: float = 0.5,
) -> Tuple[int, int]:
    """Compute graph-paper center-origin pixel coordinates for one rectangular canvas."""
    canvas_width = max(1, int(width))
    canvas_height = max(1, int(height))
    left = max(0, int(inset))
    top = max(0, int(inset))
    right = max(int(left), int(canvas_width) - 1 - int(inset))
    bottom = max(int(top), int(canvas_height) - 1 - int(inset))
    x = _axis_origin_for_bounds(
        lower=int(left),
        upper=int(right),
        spacing=int(spacing),
        fraction=float(x_fraction),
    )
    y = _axis_origin_for_bounds(
        lower=int(top),
        upper=int(bottom),
        spacing=int(spacing),
        fraction=float(y_fraction),
    )
    return (x, y)


def compute_grid_axis_origin(
    *,
    canvas_size: int,
    spacing: int,
    inset: int = 0,
    x_fraction: float = 0.5,
    y_fraction: float = 0.5,
) -> Tuple[int, int]:
    """Compute graph-paper center-origin pixel coordinates for square canvases."""
    return compute_grid_axis_origin_for_canvas(
        width=int(canvas_size),
        height=int(canvas_size),
        spacing=int(spacing),
        inset=int(inset),
        x_fraction=float(x_fraction),
        y_fraction=float(y_fraction),
    )


def _draw_center_axes(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    spacing: int,
    axis_color: Tuple[int, int, int],
    axis_line_width: int,
    inset: int = 0,
    origin: Tuple[int, int] | None = None,
) -> Tuple[int, int]:
    """Draw center x/y axes aligned to nearest graph-paper intersections."""
    left = max(0, int(inset))
    top = max(0, int(inset))
    right = max(int(left), int(width) - 1 - int(inset))
    bottom = max(int(top), int(height) - 1 - int(inset))
    axis_width = max(1, int(axis_line_width))
    max_x_start = max(int(left), int(right) - int(axis_width) + 1)
    max_y_start = max(int(top), int(bottom) - int(axis_width) + 1)

    if origin is None:
        center_x = _axis_origin_for_bounds(lower=int(left), upper=int(right), spacing=int(spacing))
        center_y = _axis_origin_for_bounds(lower=int(top), upper=int(bottom), spacing=int(spacing))
    else:
        center_x, center_y = (int(origin[0]), int(origin[1]))
    x_start = max(int(left), min(int(center_x), int(max_x_start)))
    y_start = max(int(top), min(int(center_y), int(max_y_start)))

    draw.rectangle([x_start, int(top), min(int(right), x_start + axis_width - 1), int(bottom)], fill=axis_color)
    draw.rectangle([int(left), y_start, int(right), min(int(bottom), y_start + axis_width - 1)], fill=axis_color)
    return (x_start, y_start)


def _draw_center_origin_marker(
    draw: ImageDraw.ImageDraw,
    *,
    origin_x: int,
    origin_y: int,
    color: Tuple[int, int, int],
    radius: int,
) -> None:
    """Draw one filled origin marker where center axes intersect."""
    radius_px = max(1, int(radius))
    x = int(origin_x)
    y = int(origin_y)
    draw.ellipse(
        [x - radius_px, y - radius_px, x + radius_px, y + radius_px],
        fill=color,
        outline=(20, 20, 20),
        width=max(1, int(round(radius_px / 2))),
    )


def _draw_axis_arrows(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    origin_x: int,
    origin_y: int,
    color: Tuple[int, int, int],
    arrow_size: int,
    inset: int = 0,
) -> None:
    """Draw arrowheads for positive x (right) and positive y (up) axes."""
    left = max(0, int(inset))
    top = max(0, int(inset))
    right = max(int(left), int(width) - 1 - int(inset))
    bottom = max(int(top), int(height) - 1 - int(inset))
    if int(right) - int(left) < 6 or int(bottom) - int(top) < 6:
        return

    size_px = max(4, int(arrow_size))
    half = max(2, int(round(float(size_px) / 2.0)))
    x = max(int(left), min(int(origin_x), int(right)))
    y = max(int(top), min(int(origin_y), int(bottom)))
    outline_color = (248, 248, 248)

    x_tip = max(int(left), int(right) - 1)
    x_base = max(int(left), x_tip - int(size_px))
    x_points = [
        (x_tip, y),
        (x_base, max(int(top), y - half)),
        (x_base, min(int(bottom), y + half)),
    ]
    draw.polygon(x_points, fill=color, outline=outline_color)

    y_tip = min(int(bottom), int(top) + 1)
    y_base = min(int(bottom), int(y_tip) + int(size_px))
    if int(y_base) <= int(y_tip):
        return
    y_points = [
        (x, y_tip),
        (max(int(left), x - half), y_base),
        (min(int(right), x + half), y_base),
    ]
    draw.polygon(y_points, fill=color, outline=outline_color)


def _load_label_font(*, size: int) -> ImageFont.ImageFont:
    """Load a deterministic sans-serif font with fallback to Pillow default."""
    font_size = max(8, int(size))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=font_size)
    except Exception:
        pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size=font_size)
    except Exception:
        return ImageFont.load_default()


def _draw_origin_label(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    origin_x: int,
    origin_y: int,
    spacing: int,
    label: str,
    color: Tuple[int, int, int],
    inset: int = 0,
) -> None:
    """Draw the graph origin text label near the center axes intersection."""
    text = str(label).strip() or "(0,0)"
    spacing_px = max(4, int(spacing))
    font_size = max(10, min(64, int(round(float(spacing_px) * 0.62))))
    font = _load_label_font(size=font_size)
    anchor_x = int(origin_x) + max(3, int(round(float(spacing_px) * 0.18)))
    anchor_y = int(origin_y) + max(2, int(round(float(spacing_px) * 0.14)))
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = float(bbox[2] - bbox[0])
        text_height = float(bbox[3] - bbox[1])
    except Exception:
        text_width, text_height = draw.textsize(text, font=font)
        text_width = float(text_width)
        text_height = float(text_height)
    x = float(anchor_x) - (0.5 * float(text_width))
    y = float(anchor_y) - (0.5 * float(text_height))
    x = float(x) - float(max(1, int(round(float(spacing_px) * 0.08))))
    y = float(y) + float(max(1, int(round(float(spacing_px) * 0.07))))
    left_bound = float(max(0, int(inset)))
    top_bound = float(max(0, int(inset)))
    right_bound = float(max(int(left_bound) + 1, int(width) - int(inset)))
    bottom_bound = float(max(int(top_bound) + 1, int(height) - int(inset)))
    if float(text_width) >= float(right_bound - left_bound) or float(text_height) >= float(bottom_bound - top_bound):
        return
    x = min(max(float(x), float(left_bound)), float(right_bound) - float(text_width))
    y = min(max(float(y), float(top_bound)), float(bottom_bound) - float(text_height))
    fill_color = _darken_rgb(color, factor=0.72)
    draw.text(
        (x, y),
        text,
        fill=fill_color,
        font=font,
        stroke_width=max(1, int(round(float(font_size) * 0.06))),
        stroke_fill=(252, 252, 252),
    )


def _draw_axis_scale_labels(
    draw: ImageDraw.ImageDraw,
    *,
    width: int,
    height: int,
    origin_x: int,
    origin_y: int,
    spacing: int,
    max_abs_value: int,
    color: Tuple[int, int, int],
    inset: int = 0,
) -> None:
    """Draw signed integer axis scale labels around the center origin."""
    grid_left_bound = max(0, int(inset))
    grid_top_bound = max(0, int(inset))
    grid_right_bound = max(int(grid_left_bound), int(width) - 1 - int(inset))
    grid_bottom_bound = max(int(grid_top_bound), int(height) - 1 - int(inset))
    canvas_left_bound = 0
    canvas_top_bound = 0
    canvas_right_bound = max(0, int(width) - 1)
    canvas_bottom_bound = max(0, int(height) - 1)
    spacing_px = max(4, int(spacing))
    max_abs_cap = max(0, int(max_abs_value))
    x_pos_steps = max(0, int((int(grid_right_bound) - int(origin_x)) // int(spacing_px)))
    x_neg_steps = max(0, int((int(origin_x) - int(grid_left_bound)) // int(spacing_px)))
    y_pos_steps = max(0, int((int(origin_y) - int(grid_top_bound)) // int(spacing_px)))
    y_neg_steps = max(0, int((int(grid_bottom_bound) - int(origin_y)) // int(spacing_px)))
    if max_abs_cap > 0:
        x_pos_steps = min(int(x_pos_steps), int(max_abs_cap))
        x_neg_steps = min(int(x_neg_steps), int(max_abs_cap))
        y_pos_steps = min(int(y_pos_steps), int(max_abs_cap))
        y_neg_steps = min(int(y_neg_steps), int(max_abs_cap))

    font_size = max(9, min(64, int(round(float(spacing_px) * 0.50))))
    font = _load_label_font(size=font_size)
    stroke_width = max(1, int(round(float(font_size) * 0.05)))
    fill_color = _darken_rgb(color, factor=0.72)
    y_offset = max(4, int(round(float(spacing_px) * 0.34)))
    x_offset = max(5, int(round(float(spacing_px) * 0.40)))

    def _label_rect(text: str, *, x: float, y: float) -> Tuple[float, float, float, float] | None:
        label = str(text).strip()
        if not label:
            return None
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = float(bbox[2] - bbox[0])
            text_height = float(bbox[3] - bbox[1])
        except Exception:
            text_width, text_height = draw.textsize(label, font=font)
            text_width = float(text_width)
            text_height = float(text_height)
        left = float(x) - (0.5 * float(text_width))
        top = float(y) - (0.5 * float(text_height))
        if (
            float(left) < float(canvas_left_bound)
            or float(top) < float(canvas_top_bound)
            or (float(left) + float(text_width)) > float(canvas_right_bound + 1)
            or (float(top) + float(text_height)) > float(canvas_bottom_bound + 1)
        ):
            return None
        return (float(left), float(top), float(text_width), float(text_height))

    def _draw_centered(text: str, *, x: float, y: float) -> bool:
        rect = _label_rect(text, x=x, y=y)
        if rect is None:
            return False
        left, top, _text_width, _text_height = rect
        label = str(text).strip()
        draw.text(
            (left, top),
            label,
            fill=fill_color,
            font=font,
            stroke_width=int(stroke_width),
            stroke_fill=(252, 252, 252),
        )
        return True

    x_steps = min(int(x_pos_steps), int(x_neg_steps))
    for value in range(1, int(x_steps) + 1):
        delta = int(value) * int(spacing_px)
        x_pos = int(origin_x) + int(delta)
        x_neg = int(origin_x) - int(delta)
        y_pos = float(origin_y) + float(y_offset)
        label_pos = f"{int(value)}"
        label_neg = f"-{int(value)}"
        if _label_rect(label_pos, x=float(x_pos), y=y_pos) is None or _label_rect(
            label_neg, x=float(x_neg), y=y_pos
        ) is None:
            break
        _draw_centered(label_pos, x=float(x_pos), y=y_pos)
        _draw_centered(label_neg, x=float(x_neg), y=y_pos)

    y_steps = min(int(y_pos_steps), int(y_neg_steps))
    for value in range(1, int(y_steps) + 1):
        delta = int(value) * int(spacing_px)
        y_pos = int(origin_y) - int(delta)
        y_neg = int(origin_y) + int(delta)
        x_pos = float(origin_x) - float(x_offset)
        label_pos = f"{int(value)}"
        label_neg = f"-{int(value)}"
        if _label_rect(label_pos, x=x_pos, y=float(y_pos)) is None or _label_rect(
            label_neg, x=x_pos, y=float(y_neg)
        ) is None:
            break
        _draw_centered(label_pos, x=x_pos, y=float(y_pos))
        _draw_centered(label_neg, x=x_pos, y=float(y_neg))


def _render_style(canvas_width: int, canvas_height: int, style_spec: Mapping[str, Any], *, rng=None) -> tuple[Image.Image, Dict[str, Any]]:
    """Render one background style and return image plus resolved style metadata."""
    resolved_spec = dict(style_spec)
    kind = str(style_spec.get("kind", "solid"))
    width = max(1, int(canvas_width))
    height = max(1, int(canvas_height))
    if kind == "solid":
        color = _normalize_rgb(style_spec.get("color"), _DEFAULT_BASE_COLOR)
        resolved_spec["color"] = list(color)
        return Image.new("RGB", (width, height), color), resolved_spec

    if kind == "grid":
        base_color = _normalize_rgb(style_spec.get("base_color"), _DEFAULT_BASE_COLOR)
        line_color = _normalize_rgb(style_spec.get("line_color"), (220, 220, 220))
        spacing = max(4, _to_int(style_spec.get("spacing"), 24))
        outer_margin_px = max(0, _to_int(style_spec.get("outer_margin_px"), 0))
        line_width = max(1, _to_int(style_spec.get("line_width"), 1))
        major_every = max(0, _to_int(style_spec.get("major_every"), 0))
        major_line_color = _normalize_rgb(style_spec.get("major_line_color"), line_color)
        major_line_width = max(line_width, _to_int(style_spec.get("major_line_width"), line_width))
        axis_enabled = _to_bool(style_spec.get("axis_enabled"), False)
        axis_color = _normalize_rgb(style_spec.get("axis_color"), major_line_color)
        axis_line_width = max(line_width, _to_int(style_spec.get("axis_line_width"), max(2, line_width + 1)))
        axis_arrows_enabled = _to_bool(style_spec.get("axis_arrows_enabled"), False)
        axis_arrow_size = max(4, _to_int(style_spec.get("axis_arrow_size"), max(8, axis_line_width * 4)))
        center_point_enabled = _to_bool(style_spec.get("center_point_enabled"), False)
        center_point_color = _normalize_rgb(style_spec.get("center_point_color"), axis_color)
        center_point_radius = max(1, _to_int(style_spec.get("center_point_radius"), max(2, axis_line_width)))
        color_variation_enabled = _to_bool(style_spec.get("color_variation_enabled"), False)
        base_color_jitter = normalize_int_range(style_spec.get("base_color_jitter"), fallback_min=0, fallback_max=0)
        line_color_jitter = normalize_int_range(style_spec.get("line_color_jitter"), fallback_min=0, fallback_max=0)
        major_line_darken_range = normalize_int_range(
            style_spec.get("major_line_darken_range"),
            fallback_min=0,
            fallback_max=0,
        )
        axis_darken_range = normalize_int_range(style_spec.get("axis_darken_range"), fallback_min=0, fallback_max=0)
        center_point_darken_extra_range = normalize_int_range(
            style_spec.get("center_point_darken_extra_range"),
            fallback_min=0,
            fallback_max=0,
        )
        origin_label_darken_extra_range = normalize_int_range(
            style_spec.get("origin_label_darken_extra_range"),
            fallback_min=0,
            fallback_max=0,
        )
        axis_scale_labels_enabled = _to_bool(style_spec.get("axis_scale_labels_enabled"), False)
        axis_scale_label_max_abs = max(0, _to_int(style_spec.get("axis_scale_label_max_abs"), 0))
        origin_label_enabled = _to_bool(style_spec.get("origin_label_enabled"), False)
        origin_label_text = str(style_spec.get("origin_label_text", "(0,0)"))
        origin_label_color = _normalize_rgb(style_spec.get("origin_label_color"), axis_color)
        supersample_scale = max(1, min(4, _to_int(style_spec.get("supersample_scale"), 1)))

        variation_applied = False
        if color_variation_enabled and rng is not None:
            base_delta = int(rng.randint(int(base_color_jitter[0]), int(base_color_jitter[1])))
            base_color = _shift_rgb_by_delta(base_color, delta=int(base_delta))

            line_delta = int(rng.randint(int(line_color_jitter[0]), int(line_color_jitter[1])))
            line_color = _shift_rgb_by_delta(line_color, delta=int(line_delta))

            sampled_major_darken = int(rng.randint(int(major_line_darken_range[0]), int(major_line_darken_range[1])))
            major_line_color = _darken_rgb_by_delta(line_color, delta=int(sampled_major_darken))

            sampled_axis_darken = int(rng.randint(int(axis_darken_range[0]), int(axis_darken_range[1])))
            sampled_axis_darken = max(int(sampled_axis_darken), int(sampled_major_darken) + 2)
            axis_color = _darken_rgb_by_delta(line_color, delta=int(sampled_axis_darken))

            sampled_center_extra = int(
                rng.randint(int(center_point_darken_extra_range[0]), int(center_point_darken_extra_range[1]))
            )
            center_point_color = _darken_rgb_by_delta(axis_color, delta=int(sampled_center_extra))

            sampled_label_extra = int(
                rng.randint(int(origin_label_darken_extra_range[0]), int(origin_label_darken_extra_range[1]))
            )
            origin_label_color = _darken_rgb_by_delta(axis_color, delta=int(sampled_label_extra))
            variation_applied = True
            resolved_spec["color_variation_sampled"] = {
                "base_color_delta": int(base_delta),
                "line_color_delta": int(line_delta),
                "major_line_darken": int(sampled_major_darken),
                "axis_darken": int(sampled_axis_darken),
                "center_point_darken_extra": int(sampled_center_extra),
                "origin_label_darken_extra": int(sampled_label_extra),
            }

        resolved_spec["base_color"] = list(base_color)
        resolved_spec["line_color"] = list(line_color)
        resolved_spec["outer_margin_px"] = int(outer_margin_px)
        resolved_spec["major_line_color"] = list(major_line_color)
        resolved_spec["axis_color"] = list(axis_color)
        resolved_spec["center_point_color"] = list(center_point_color)
        resolved_spec["origin_label_color"] = list(origin_label_color)
        resolved_spec["color_variation_applied"] = bool(variation_applied)

        render_width = int(width) * int(supersample_scale)
        render_height = int(height) * int(supersample_scale)
        scaled_spacing = int(spacing) * int(supersample_scale)
        scaled_line_width = int(line_width) * int(supersample_scale)
        scaled_major_line_width = int(major_line_width) * int(supersample_scale)
        scaled_axis_line_width = int(axis_line_width) * int(supersample_scale)
        scaled_axis_arrow_size = int(axis_arrow_size) * int(supersample_scale)
        scaled_center_point_radius = int(center_point_radius) * int(supersample_scale)
        scaled_outer_margin_px = int(outer_margin_px) * int(supersample_scale)
        max_scaled_margin = max(0, (min(int(render_width), int(render_height)) - 8) // 2)
        scaled_outer_margin_px = min(int(scaled_outer_margin_px), int(max_scaled_margin))
        resolved_spec["outer_margin_px_scaled"] = int(scaled_outer_margin_px)

        image = Image.new("RGB", (render_width, render_height), base_color)
        draw = ImageDraw.Draw(image)
        centered_origin = compute_grid_axis_origin_for_canvas(
            width=int(render_width),
            height=int(render_height),
            spacing=int(scaled_spacing),
            inset=int(scaled_outer_margin_px),
            x_fraction=float(resolved_spec.get("origin_fraction_x", 0.5)),
            y_fraction=float(resolved_spec.get("origin_fraction_y", 0.5)),
        )
        explicit_origin = resolved_spec.get("origin_pixel")
        if isinstance(explicit_origin, (list, tuple)) and len(explicit_origin) >= 2:
            centered_origin = (
                max(
                    0,
                    min(
                        int(render_width) - 1,
                        _to_int(explicit_origin[0], int(centered_origin[0])) * int(supersample_scale),
                    ),
                ),
                max(
                    0,
                    min(
                        int(render_height) - 1,
                        _to_int(explicit_origin[1], int(centered_origin[1])) * int(supersample_scale),
                    ),
                ),
            )
        resolved_spec["origin_pixel"] = [int(centered_origin[0]), int(centered_origin[1])]
        _draw_grid_lines(
            draw,
            width=render_width,
            height=render_height,
            spacing=scaled_spacing,
            line_color=line_color,
            line_width=scaled_line_width,
            major_every=major_every,
            major_line_color=major_line_color,
            major_line_width=scaled_major_line_width,
            inset=int(scaled_outer_margin_px),
            origin_x=int(centered_origin[0]),
            origin_y=int(centered_origin[1]),
        )
        if axis_enabled:
            center_origin = _draw_center_axes(
                draw,
                width=render_width,
                height=render_height,
                spacing=scaled_spacing,
                axis_color=axis_color,
                axis_line_width=scaled_axis_line_width,
                inset=int(scaled_outer_margin_px),
                origin=(int(centered_origin[0]), int(centered_origin[1])),
            )
            if axis_arrows_enabled:
                _draw_axis_arrows(
                    draw,
                    width=render_width,
                    height=render_height,
                    origin_x=int(center_origin[0]),
                    origin_y=int(center_origin[1]),
                    color=axis_color,
                    arrow_size=scaled_axis_arrow_size,
                    inset=int(scaled_outer_margin_px),
                )
            if center_point_enabled:
                _draw_center_origin_marker(
                    draw,
                    origin_x=int(center_origin[0]),
                    origin_y=int(center_origin[1]),
                    color=center_point_color,
                    radius=scaled_center_point_radius,
                )
            if axis_scale_labels_enabled:
                _draw_axis_scale_labels(
                    draw,
                    width=render_width,
                    height=render_height,
                    origin_x=int(center_origin[0]),
                    origin_y=int(center_origin[1]),
                    spacing=int(scaled_spacing),
                    max_abs_value=int(axis_scale_label_max_abs),
                    color=axis_color,
                    inset=int(scaled_outer_margin_px),
                )
            if origin_label_enabled:
                _draw_origin_label(
                    draw,
                    width=render_width,
                    height=render_height,
                    origin_x=int(center_origin[0]),
                    origin_y=int(center_origin[1]),
                    spacing=int(scaled_spacing),
                    label=str(origin_label_text),
                    color=origin_label_color,
                    inset=int(scaled_outer_margin_px),
                )
        if supersample_scale > 1:
            image = image.resize((int(width), int(height)), resample=Image.Resampling.LANCZOS)
        return image, resolved_spec

    return Image.new("RGB", (width, height), _DEFAULT_BASE_COLOR), resolved_spec


def make_background_canvas(
    *,
    canvas_size: int | None = None,
    canvas_width: int | None = None,
    canvas_height: int | None = None,
    instance_seed: int,
    params: Mapping[str, Any],
    default_config: Mapping[str, Any] | None = None,
    fallback_color: Tuple[int, int, int] = _DEFAULT_BASE_COLOR,
) -> tuple[Image.Image, Dict[str, Any]]:
    """Create deterministic background canvas plus trace metadata.

    Callers may pass `canvas_size` for a square canvas or explicit
    `canvas_width`/`canvas_height` for rectangular canvases.
    """
    if canvas_size is not None:
        width = max(1, int(canvas_size))
        height = max(1, int(canvas_size))
    else:
        width = max(1, int(canvas_width) if canvas_width is not None else 1)
        height = max(1, int(canvas_height) if canvas_height is not None else width)
    cfg = _resolve_background_config(params, default_config=default_config)
    enabled = bool(cfg["enabled"])
    styles = cfg.get("styles", {})

    if not enabled or not styles:
        return (
            Image.new("RGB", (width, height), _normalize_rgb(fallback_color, _DEFAULT_BASE_COLOR)),
            {
                "enabled": False,
                "selected_style": None,
                "available_styles": sorted(styles.keys()),
            },
        )

    style_name = str(cfg.get("style_name", "")).strip()
    if style_name and style_name in styles:
        selected_style = style_name
    else:
        rng = spawn_rng(instance_seed, "visual.background_style")
        selected_style = weighted_choice(rng, cfg.get("weights", {}), sort_keys=True)
        if not selected_style:
            selected_style = sorted(styles.keys())[0]

    variant_rng = spawn_rng(instance_seed, f"visual.background_style_variant.{selected_style}")
    selected_spec = _resolve_style_variant(styles[selected_style], rng=variant_rng)
    render_rng = spawn_rng(instance_seed, f"visual.background_render.{selected_style}")
    image, resolved_style_spec = _render_style(width, height, selected_spec, rng=render_rng)
    metadata = {
        "enabled": True,
        "selected_style": selected_style,
        "available_styles": sorted(styles.keys()),
        "style_spec": resolved_style_spec,
    }
    return image, metadata
