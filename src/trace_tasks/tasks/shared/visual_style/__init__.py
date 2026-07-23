"""Shared non-semantic visual style primitives for task renderers."""

from .palette import PANEL_SCENE_PALETTES, PanelScenePalette
from .surface_tones import (
    DARK_SURFACE_TONE_IDS,
    DEFAULT_SURFACE_TONES,
    LIGHT_SURFACE_TEXT_RGB,
    LIGHT_SURFACE_TEXT_STROKE_RGB,
)
from .metadata import color_separation_metadata
from .request import (
    VISUAL_STYLE_FAMILIES,
    VisualStyleRequest,
    build_visual_style_request,
    resolve_style_bool,
    visual_style_request_metadata,
)
from .panel import (
    DEFAULT_PANEL_SCENE_STYLE,
    PANEL_SCENE_TREATMENTS,
    PanelSceneStyle,
    draw_panel_grid_cell,
    draw_panel_option_card,
    draw_panel_scene_chrome,
    make_panel_scene_background,
    panel_scene_style_metadata,
    resolve_panel_scene_style,
)
from .technical_diagram import (
    DEFAULT_TECHNICAL_DIAGRAM_STYLE,
    DEFAULT_TECHNICAL_DIAGRAM_FRAME_WEIGHTS,
    TECHNICAL_DIAGRAM_FRAME_MODES,
    TECHNICAL_DIAGRAM_PALETTES,
    TECHNICAL_DIAGRAM_TREATMENTS,
    TECHNICAL_DIAGRAM_TREATMENT_IDS,
    TechnicalDiagramPalette,
    TechnicalDiagramStyle,
    TechnicalDiagramTreatment,
    make_technical_diagram_background,
    resolve_technical_diagram_style,
    technical_diagram_style_metadata,
)

__all__ = [
    "DEFAULT_PANEL_SCENE_STYLE",
    "DARK_SURFACE_TONE_IDS",
    "DEFAULT_SURFACE_TONES",
    "DEFAULT_TECHNICAL_DIAGRAM_STYLE",
    "DEFAULT_TECHNICAL_DIAGRAM_FRAME_WEIGHTS",
    "PANEL_SCENE_PALETTES",
    "PANEL_SCENE_TREATMENTS",
    "TECHNICAL_DIAGRAM_FRAME_MODES",
    "TECHNICAL_DIAGRAM_PALETTES",
    "TECHNICAL_DIAGRAM_TREATMENTS",
    "TECHNICAL_DIAGRAM_TREATMENT_IDS",
    "VISUAL_STYLE_FAMILIES",
    "PanelScenePalette",
    "PanelSceneStyle",
    "TechnicalDiagramPalette",
    "TechnicalDiagramStyle",
    "TechnicalDiagramTreatment",
    "VisualStyleRequest",
    "LIGHT_SURFACE_TEXT_RGB",
    "LIGHT_SURFACE_TEXT_STROKE_RGB",
    "build_visual_style_request",
    "color_separation_metadata",
    "draw_panel_grid_cell",
    "draw_panel_option_card",
    "draw_panel_scene_chrome",
    "make_panel_scene_background",
    "make_technical_diagram_background",
    "panel_scene_style_metadata",
    "resolve_panel_scene_style",
    "resolve_style_bool",
    "resolve_technical_diagram_style",
    "technical_diagram_style_metadata",
    "visual_style_request_metadata",
]
