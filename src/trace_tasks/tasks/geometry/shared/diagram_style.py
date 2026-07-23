"""Geometry-domain adapter for shared technical-diagram styling."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from ...shared.color_distance import color_distance
from ...shared.text_legibility import READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO, resolve_readable_text_style
from ...shared.visual_style.technical_diagram import (
    Color,
    TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL,
    TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER,
    TECHNICAL_DIAGRAM_PROFILE_THEME_IDS,
    TECHNICAL_DIAGRAM_THEMES,
    TechnicalDiagramStyle,
    make_technical_diagram_background,
    resolve_technical_diagram_style,
    technical_diagram_style_metadata,
)
from .coordinate_panel_grid import CoordinatePanelStyle
from .shape_style import GeometryShapeStyle


GeometryDiagramStyle = TechnicalDiagramStyle

GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM = "analytical_diagram"
GEOMETRY_STYLE_PROFILE_COORDINATE_GRID = "coordinate_grid"
GEOMETRY_STYLE_PROFILES = (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    GEOMETRY_STYLE_PROFILE_COORDINATE_GRID,
)
COORDINATE_GRID_SCENE_IDS: frozenset[str] = frozenset(
    {
        "coordinate_composite",
        "coordinate_panels",
        "coordinate_plane",
        "function_graph",
        "function_panels",
        "graph_paper",
        "graph_paper_panel",
        "shape_reference",
    }
)

ANALYTICAL_DIAGRAM_TREATMENTS: tuple[str, ...] = tuple(
    dict.fromkeys(
        TECHNICAL_DIAGRAM_THEMES[theme_id].treatment_id
        for theme_id in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL]
    )
)
COORDINATE_GRID_TREATMENTS: tuple[str, ...] = tuple(
    dict.fromkeys(
        TECHNICAL_DIAGRAM_THEMES[theme_id].treatment_id
        for theme_id in TECHNICAL_DIAGRAM_PROFILE_THEME_IDS[TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER]
    )
)

_GEOMETRY_LABEL_DARK_RGB: Color = (10, 14, 22)
_GEOMETRY_LABEL_LIGHT_RGB: Color = (250, 252, 255)


def _normalize_style_profile(style_profile: str | None) -> str | None:
    if style_profile is None:
        return None
    normalized = str(style_profile).strip().lower()
    if not normalized:
        return None
    if normalized not in set(GEOMETRY_STYLE_PROFILES):
        raise ValueError(f"unknown geometry style profile: {style_profile!r}")
    return normalized


def _default_style_profile_for_scene(scene_id: str) -> str:
    if str(scene_id) in COORDINATE_GRID_SCENE_IDS:
        return GEOMETRY_STYLE_PROFILE_COORDINATE_GRID
    return GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM


def _profile_treatments(style_profile: str | None) -> tuple[str, ...] | None:
    normalized = _normalize_style_profile(style_profile)
    if normalized == GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM:
        return ANALYTICAL_DIAGRAM_TREATMENTS
    if normalized == GEOMETRY_STYLE_PROFILE_COORDINATE_GRID:
        return COORDINATE_GRID_TREATMENTS
    return None


def _technical_profile_for_geometry_profile(style_profile: str | None) -> str | None:
    normalized = _normalize_style_profile(style_profile)
    if normalized == GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM:
        return TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL
    if normalized == GEOMETRY_STYLE_PROFILE_COORDINATE_GRID:
        return TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER
    return None


def _normalize_treatment_sequence(treatments: Sequence[str] | str | None) -> tuple[str, ...] | None:
    if treatments is None:
        return None
    if isinstance(treatments, str):
        text = treatments.strip()
        return (text,) if text else None
    return tuple(str(item) for item in treatments)


def _resolve_profile_treatments(
    *,
    style_profile: str | None,
    requested_treatments: Sequence[str] | str | None,
) -> tuple[str, ...] | None:
    profile_treatments = _profile_treatments(style_profile)
    requested = _normalize_treatment_sequence(requested_treatments)
    if profile_treatments is None:
        return requested
    if requested is None:
        return tuple(profile_treatments)
    filtered = tuple(item for item in requested if item in set(profile_treatments))
    return filtered or tuple(profile_treatments)


def _resolve_profile_require_grid(
    *,
    style_profile: str | None,
    require_grid: bool | None,
) -> bool | None:
    normalized = _normalize_style_profile(style_profile)
    if normalized == GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM:
        return False
    if normalized == GEOMETRY_STYLE_PROFILE_COORDINATE_GRID:
        return True
    return require_grid


def _relative_luminance(color: Color) -> float:
    """Return WCAG-style relative luminance for one sRGB color."""

    def channel(value: int) -> float:
        normalized = max(0.0, min(1.0, float(int(value)) / 255.0))
        if normalized <= 0.03928:
            return normalized / 12.92
        return ((normalized + 0.055) / 1.055) ** 2.4

    red = channel(int(color[0]))
    green = channel(int(color[1]))
    blue = channel(int(color[2]))
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _contrast_ratio(color_a: Color, color_b: Color) -> float:
    lum_a = _relative_luminance(color_a)
    lum_b = _relative_luminance(color_b)
    lighter = max(float(lum_a), float(lum_b))
    darker = min(float(lum_a), float(lum_b))
    return (lighter + 0.05) / (darker + 0.05)


def _resolve_geometry_label_contrast(style: GeometryDiagramStyle) -> tuple[GeometryDiagramStyle, dict[str, Any]]:
    """Strengthen geometry label ink against the sampled technical surface.

    The shared technical-diagram palettes are used by multiple domains. For
    geometry, small point labels and measurement text are often drawn directly
    over pale panels; using a light halo around dark text can make thin glyphs
    look washed out. Geometry therefore resolves one high-contrast label ink
    against the sampled surface anchors and uses the same ink for the text
    stroke so small labels render as thicker glyphs instead of low-contrast
    haloed text.
    """

    anchors = (
        tuple(int(v) for v in style.canvas_rgb),
        tuple(int(v) for v in style.paper_rgb),
        tuple(int(v) for v in style.panel_fill_rgb),
        tuple(int(v) for v in style.panel_alt_fill_rgb),
        tuple(int(v) for v in style.option_fill_rgb),
    )
    preferred = (
        tuple(int(v) for v in style.label_rgb),
        tuple(int(v) for v in style.stroke_rgb),
        tuple(int(v) for v in style.secondary_stroke_rgb),
        _GEOMETRY_LABEL_DARK_RGB,
        _GEOMETRY_LABEL_LIGHT_RGB,
    )
    label_style = resolve_readable_text_style(
        instance_seed=sum((index + 1) * int(channel) for index, color in enumerate(anchors) for channel in color),
        namespace=f"geometry.label_contrast.{style.style_pack}",
        role="read_required_geometry_label",
        surface_rgbs=anchors,
        preferred_rgbs=preferred,
        min_contrast_ratio=READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO,
        required=True,
    )
    label_rgb = tuple(int(v) for v in label_style.fill_rgb)
    adjusted = replace(
        style,
        label_rgb=tuple(int(v) for v in label_rgb),
        label_stroke_rgb=tuple(int(v) for v in label_rgb),
        label_stroke_width_px=min(1, max(0, int(style.label_stroke_width_px))),
    )
    guard_meta = {
        "enabled": True,
        "surface_anchor_rgb": [list(anchor) for anchor in anchors],
        "candidate_label_rgb": [list(candidate) for candidate in preferred],
        "resolved_label_rgb": list(adjusted.label_rgb),
        "resolved_label_stroke_rgb": list(adjusted.label_stroke_rgb),
        "min_surface_contrast_ratio": round(min(_contrast_ratio(adjusted.label_rgb, anchor) for anchor in anchors), 3),
        "min_surface_lab_distance": round(
            min(color_distance(adjusted.label_rgb, anchor, distance_space="lab") for anchor in anchors),
            3,
        ),
        "shared_text_legibility": label_style.metadata(),
    }
    return adjusted, guard_meta


def resolve_geometry_diagram_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None = None,
    scene_id: str,
    treatments: Sequence[str] | None = None,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool = False,
    require_grid: bool | None = None,
    style_profile: str | None = None,
) -> tuple[GeometryDiagramStyle, dict[str, Any]]:
    """Resolve the shared technical style for one geometry scene."""

    resolved_params = params or {}
    normalized_profile = _normalize_style_profile(style_profile) or _default_style_profile_for_scene(str(scene_id))
    technical_profile = _technical_profile_for_geometry_profile(normalized_profile)
    requested_treatments = treatments or resolved_params.get("technical_diagram_treatments")
    resolved_treatments = _resolve_profile_treatments(
        style_profile=normalized_profile,
        requested_treatments=requested_treatments,
    )
    resolved_require_grid = _resolve_profile_require_grid(
        style_profile=normalized_profile,
        require_grid=require_grid,
    )
    style, metadata = resolve_technical_diagram_style(
        instance_seed=int(instance_seed),
        namespace=f"geometry.{str(scene_id)}.technical_diagram_style",
        theme_profile=technical_profile,
        themes=resolved_params.get("technical_diagram_themes"),
        theme_weights=resolved_params.get("technical_diagram_theme_weights", {}),
        treatments=resolved_treatments,
        treatment_weights=resolved_params.get("technical_diagram_treatment_weights", {}),
        palettes=resolved_params.get("technical_diagram_palettes"),
        palette_weights=resolved_params.get("technical_diagram_palette_weights", {}),
        frame_modes=resolved_params.get("technical_diagram_frame_modes"),
        frame_mode_weights=resolved_params.get("technical_diagram_frame_mode_weights", {}),
        allow_dark=True if normalized_profile is not None else bool(allow_dark),
        require_grid=resolved_require_grid,
        protected_colors=protected_colors or (),
    )
    adjusted_style, guard_meta = _resolve_geometry_label_contrast(style)
    adjusted_metadata = technical_diagram_style_metadata(adjusted_style)
    if isinstance(metadata, Mapping) and isinstance(metadata.get("selection"), Mapping):
        adjusted_metadata["selection"] = dict(metadata["selection"])
    adjusted_metadata["geometry_style_profile"] = normalized_profile
    adjusted_metadata["technical_profile"] = technical_profile
    adjusted_metadata["profile_treatments"] = list(_profile_treatments(normalized_profile) or [])
    if normalized_profile is not None:
        adjusted_metadata["available_treatments"] = list(_profile_treatments(normalized_profile) or [])
        adjusted_metadata["available_theme_ids"] = list(
            TECHNICAL_DIAGRAM_PROFILE_THEME_IDS.get(str(technical_profile), ())
        )
    normalized_requested_treatments = _normalize_treatment_sequence(requested_treatments)
    adjusted_metadata["requested_treatments_before_profile"] = (
        list(normalized_requested_treatments) if normalized_requested_treatments is not None else None
    )
    adjusted_metadata["geometry_label_contrast_guard"] = dict(guard_meta)
    return adjusted_style, adjusted_metadata


def make_geometry_diagram_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: GeometryDiagramStyle,
    instance_seed: int,
    namespace: str = "geometry.technical_diagram_background",
) -> tuple[Any, dict[str, Any]]:
    """Create a geometry technical-diagram background."""

    return make_technical_diagram_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def prepare_geometry_diagram_style_and_background(
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    scene_id: str,
    canvas_width: int,
    canvas_height: int,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool = False,
    require_grid: bool | None = None,
    treatments: Sequence[str] | None = None,
    style_profile: str | None = None,
    namespace_suffix: str = "technical_diagram_background",
) -> tuple[Any, dict[str, Any], GeometryDiagramStyle, dict[str, Any]]:
    """Resolve one geometry technical style and create its background before rendering."""

    diagram_style, diagram_style_meta = resolve_geometry_diagram_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=str(scene_id),
        treatments=treatments,
        protected_colors=protected_colors or (),
        allow_dark=bool(allow_dark),
        require_grid=require_grid,
        style_profile=style_profile,
    )
    background, background_meta = make_geometry_diagram_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=diagram_style,
        instance_seed=int(instance_seed),
        namespace=f"geometry.{str(scene_id)}.{str(namespace_suffix)}",
    )
    normalized_profile = _normalize_style_profile(style_profile) or _default_style_profile_for_scene(str(scene_id))
    if normalized_profile is not None and isinstance(background_meta, dict):
        technical_profile = _technical_profile_for_geometry_profile(normalized_profile)
        background_meta["geometry_style_profile"] = normalized_profile
        background_meta["technical_profile"] = technical_profile
        style_spec = background_meta.get("style_spec")
        if isinstance(style_spec, dict):
            profile_treatments = list(_profile_treatments(normalized_profile) or [])
            style_spec["geometry_style_profile"] = normalized_profile
            style_spec["technical_profile"] = technical_profile
            style_spec["profile_treatments"] = profile_treatments
            style_spec["available_treatments"] = profile_treatments
            style_spec["available_theme_ids"] = list(
                TECHNICAL_DIAGRAM_PROFILE_THEME_IDS.get(str(technical_profile), ())
            )
    return background, background_meta, diagram_style, diagram_style_meta


def geometry_diagram_style_metadata(style: GeometryDiagramStyle) -> dict[str, Any]:
    """Serialize a geometry technical-diagram style."""

    return technical_diagram_style_metadata(style)


def geometry_shape_style_from_diagram_style(style: GeometryDiagramStyle) -> GeometryShapeStyle:
    """Map shared technical style roles to geometry ink roles."""

    return GeometryShapeStyle(
        line_color=tuple(int(value) for value in style.stroke_rgb),
        label_color=tuple(int(value) for value in style.label_rgb),
        label_stroke_color=tuple(int(value) for value in style.label_stroke_rgb),
    )


def geometry_coordinate_panel_style_from_diagram_style(style: GeometryDiagramStyle) -> CoordinatePanelStyle:
    """Map shared technical style roles to small coordinate-panel chrome."""

    return CoordinatePanelStyle(
        panel_fill=tuple(int(value) for value in style.panel_fill_rgb),
        panel_outline=tuple(int(value) for value in style.panel_border_rgb),
        plot_fill=tuple(int(value) for value in style.panel_alt_fill_rgb),
        plot_outline=tuple(int(value) for value in style.panel_border_rgb),
        grid_color=tuple(int(value) for value in style.grid_minor_rgb),
        axis_color=tuple(int(value) for value in style.axis_rgb),
        tick_color=tuple(int(value) for value in style.secondary_stroke_rgb),
        text_color=tuple(int(value) for value in style.label_rgb),
        text_stroke_color=tuple(int(value) for value in style.label_stroke_rgb),
    )


def geometry_graph_style_from_diagram_style(
    style: GeometryDiagramStyle,
    *,
    spacing_px: int = 24,
    outer_margin_px: int = 16,
    axis_enabled: bool = True,
) -> dict[str, Any]:
    """Map shared technical style roles to the geometry graph-paper spec shape."""

    return {
        "kind": "grid",
        "base_color": list(style.panel_fill_rgb),
        "line_color": list(style.grid_minor_rgb),
        "spacing": max(4, int(spacing_px)),
        "outer_margin_px": max(0, int(outer_margin_px)),
        "line_width": int(style.grid_minor_width_px),
        "major_every": int(style.major_every),
        "major_line_color": list(style.grid_major_rgb),
        "major_line_width": int(style.grid_major_width_px),
        "axis_enabled": bool(axis_enabled),
        "axis_color": list(style.axis_rgb),
        "axis_line_width": int(style.axis_stroke_width_px),
        "axis_arrows_enabled": bool(axis_enabled),
        "axis_arrow_size": max(8, int(style.axis_stroke_width_px) * 4),
        "center_point_enabled": bool(axis_enabled),
        "center_point_color": list(style.axis_rgb),
        "center_point_radius": max(2, int(style.axis_stroke_width_px) // 2),
        "color_variation_enabled": False,
        "base_color_jitter": [0, 0],
        "line_color_jitter": [0, 0],
        "major_line_darken_range": [0, 0],
        "axis_darken_range": [0, 0],
        "center_point_darken_extra_range": [0, 0],
        "origin_label_darken_extra_range": [0, 0],
        "axis_scale_labels_enabled": True,
        "axis_scale_label_max_abs": 0,
        "origin_label_enabled": False,
        "origin_label_text": "0",
        "origin_label_color": list(style.label_rgb),
        "supersample_scale": 1,
        "scene_supersample_scale": 3,
    }


__all__ = [
    "ANALYTICAL_DIAGRAM_TREATMENTS",
    "COORDINATE_GRID_TREATMENTS",
    "COORDINATE_GRID_SCENE_IDS",
    "GeometryDiagramStyle",
    "GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM",
    "GEOMETRY_STYLE_PROFILE_COORDINATE_GRID",
    "GEOMETRY_STYLE_PROFILES",
    "geometry_coordinate_panel_style_from_diagram_style",
    "geometry_diagram_style_metadata",
    "geometry_graph_style_from_diagram_style",
    "geometry_shape_style_from_diagram_style",
    "make_geometry_diagram_background",
    "prepare_geometry_diagram_style_and_background",
    "resolve_geometry_diagram_style",
]
