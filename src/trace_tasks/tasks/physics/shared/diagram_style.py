"""Physics-domain adapter for shared technical-diagram styling."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.visual_style.technical_diagram import (
    Color,
    TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL,
    TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER,
    TechnicalDiagramStyle,
    make_technical_diagram_background,
    resolve_technical_diagram_style,
    technical_diagram_style_metadata,
)
from .style import PhysicsElectrostaticsTheme


PhysicsDiagramStyle = TechnicalDiagramStyle

PHYSICS_ELECTROSTATICS_SEMANTIC_COLORS: tuple[Color, ...] = (
    (255, 226, 221),
    (187, 58, 46),
    (142, 40, 32),
    (222, 237, 255),
    (55, 91, 172),
    (38, 66, 132),
)


def _resolve_physics_technical_profile(
    *,
    params: Mapping[str, Any] | None,
    style_profile: str | None,
    require_grid: bool | None,
) -> str:
    requested = style_profile or (params or {}).get("technical_diagram_style_profile")
    if requested is not None and str(requested).strip():
        normalized = str(requested).strip().lower()
        if normalized in {"coordinate_grid", "graph_paper"}:
            return TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER
        if normalized in {"analytical", "analytical_diagram"}:
            return TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL
        raise ValueError(f"unknown physics technical diagram style profile: {requested!r}")
    return TECHNICAL_DIAGRAM_PROFILE_ANALYTICAL


def _resolve_physics_profile_require_grid(*, technical_profile: str, require_grid: bool | None) -> bool:
    """Translate physics scene style profile into shared resolver grid eligibility."""

    _ = require_grid
    if str(technical_profile) == TECHNICAL_DIAGRAM_PROFILE_GRAPH_PAPER:
        return True
    return False


def resolve_physics_diagram_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None = None,
    scene_id: str,
    treatments: Sequence[str] | None = None,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool = True,
    require_grid: bool | None = None,
    style_profile: str | None = None,
    min_protected_lab_distance: float | None = None,
) -> tuple[PhysicsDiagramStyle, dict[str, Any]]:
    """Resolve the shared technical style for one physics scene."""

    resolved_params = params or {}
    resolved_min_protected_lab_distance = (
        float(min_protected_lab_distance)
        if min_protected_lab_distance is not None
        else float(resolved_params.get("technical_diagram_min_protected_lab_distance", 18.0))
    )
    technical_profile = _resolve_physics_technical_profile(
        params=resolved_params,
        style_profile=style_profile,
        require_grid=require_grid,
    )
    resolved_require_grid = _resolve_physics_profile_require_grid(
        technical_profile=str(technical_profile),
        require_grid=require_grid,
    )
    return resolve_technical_diagram_style(
        instance_seed=int(instance_seed),
        namespace=f"physics.{str(scene_id)}.{str(scene_id)}.technical_diagram_style",
        theme_profile=technical_profile,
        themes=resolved_params.get("technical_diagram_themes"),
        theme_weights=resolved_params.get("technical_diagram_theme_weights", {}),
        treatments=treatments or resolved_params.get("technical_diagram_treatments"),
        treatment_weights=resolved_params.get("technical_diagram_treatment_weights", {}),
        palettes=resolved_params.get("technical_diagram_palettes"),
        palette_weights=resolved_params.get("technical_diagram_palette_weights", {}),
        frame_modes=resolved_params.get("technical_diagram_frame_modes"),
        frame_mode_weights=resolved_params.get("technical_diagram_frame_mode_weights", {}),
        allow_dark=bool(allow_dark),
        require_grid=resolved_require_grid,
        protected_colors=protected_colors or (),
        min_protected_lab_distance=float(resolved_min_protected_lab_distance),
    )


def make_physics_diagram_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: PhysicsDiagramStyle,
    instance_seed: int,
    namespace: str = "physics.technical_diagram_background",
) -> tuple[Any, dict[str, Any]]:
    """Create a physics technical-diagram background."""

    return make_technical_diagram_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def prepare_physics_diagram_style_and_background(
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    scene_id: str,
    canvas_width: int,
    canvas_height: int,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool = True,
    require_grid: bool | None = None,
    treatments: Sequence[str] | None = None,
    style_profile: str | None = None,
    namespace_suffix: str = "technical_diagram_background",
    min_protected_lab_distance: float | None = None,
) -> tuple[Any, dict[str, Any], PhysicsDiagramStyle, dict[str, Any]]:
    """Resolve one physics technical style and create its background before rendering."""

    diagram_style, diagram_style_meta = resolve_physics_diagram_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=str(scene_id),
        treatments=treatments,
        protected_colors=protected_colors or (),
        allow_dark=bool(allow_dark),
        require_grid=require_grid,
        style_profile=style_profile,
        min_protected_lab_distance=min_protected_lab_distance,
    )
    background, background_meta = make_physics_diagram_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=diagram_style,
        instance_seed=int(instance_seed),
        namespace=f"physics.{str(scene_id)}.{str(scene_id)}.{str(namespace_suffix)}",
    )
    return background, background_meta, diagram_style, diagram_style_meta


def physics_diagram_style_metadata(style: PhysicsDiagramStyle) -> dict[str, Any]:
    """Serialize a physics technical-diagram style."""

    return technical_diagram_style_metadata(style)


def physics_electrostatics_theme_from_diagram_style(
    style: PhysicsDiagramStyle,
    *,
    accent_color_name: str,
) -> PhysicsElectrostaticsTheme:
    """Map shared technical style roles to the electrostatics field-map theme."""

    return PhysicsElectrostaticsTheme(
        accent_color_name=str(accent_color_name),
        board_fill_rgb=tuple(int(value) for value in style.panel_fill_rgb),
        board_alt_fill_rgb=tuple(int(value) for value in style.panel_alt_fill_rgb),
        board_outline_rgb=tuple(int(value) for value in style.panel_border_rgb),
        grid_rgb=tuple(int(value) for value in style.grid_minor_rgb),
        axis_rgb=tuple(int(value) for value in style.axis_rgb),
        axis_text_rgb=tuple(int(value) for value in style.label_rgb),
        positive_fill_rgb=(255, 226, 221),
        positive_outline_rgb=(187, 58, 46),
        positive_text_rgb=(142, 40, 32),
        negative_fill_rgb=(222, 237, 255),
        negative_outline_rgb=(55, 91, 172),
        negative_text_rgb=(38, 66, 132),
        point_fill_rgb=tuple(int(value) for value in style.muted_fill_rgb),
        point_outline_rgb=tuple(int(value) for value in style.secondary_stroke_rgb),
        point_text_rgb=tuple(int(value) for value in style.label_rgb),
        target_fill_rgb=tuple(int(value) for value in style.accent_rgb),
        target_outline_rgb=tuple(int(value) for value in style.stroke_rgb),
        option_arrow_rgb=tuple(int(value) for value in style.stroke_rgb),
        option_outline_rgb=tuple(int(value) for value in style.panel_border_rgb),
        guide_rgb=tuple(int(value) for value in style.guide_rgb),
        label_fill_rgb=tuple(int(value) for value in style.label_fill_rgb),
        label_outline_rgb=tuple(int(value) for value in style.label_border_rgb),
        label_text_rgb=tuple(int(value) for value in style.label_rgb),
    )


__all__ = [
    "PHYSICS_ELECTROSTATICS_SEMANTIC_COLORS",
    "PhysicsDiagramStyle",
    "make_physics_diagram_background",
    "physics_diagram_style_metadata",
    "physics_electrostatics_theme_from_diagram_style",
    "prepare_physics_diagram_style_and_background",
    "resolve_physics_diagram_style",
]
