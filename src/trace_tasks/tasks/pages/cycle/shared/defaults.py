"""Defaults and semantic resources for pages cycle diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.diagram.visual_defaults import (
    load_diagrams_scene_background_defaults,
    load_diagrams_scene_noise_defaults,
)
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import CycleDefaults


DOMAIN = "pages"
SCENE = "cycle"
PROMPT_BUNDLE = "pages_cycle_v1"
PROMPT_SCENE_KEY = "cycle_diagram"
PROMPT_TASK_KEY = "offset_stage_query"
NAMESPACE_ROOT = "pages.cycle"

SCENE_VARIANTS: Tuple[str, ...] = ("cycle_ring",)
CYCLE_DIRECTIONS: Tuple[str, ...] = ("clockwise", "counterclockwise")
TITLE_OPTIONS: Tuple[str, ...] = (
    "Cycle Diagram",
    "Stage Loop",
    "Process Ring",
    "Cycle Overview",
    "Loop Stages",
)

DEFAULTS = CycleDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS: Dict[str, Any] = {
    "canvas_width": 1200,
    "canvas_height": 900,
    "outer_margin_px": 52,
    "panel_padding_px": 28,
    "panel_corner_radius_px": 30,
    "title_font_size_px": 32,
    "title_band_height_px": 78,
    "node_width_px": 122,
    "node_height_px": 58,
    "node_corner_radius_px": 24,
    "node_border_width_px": 3,
    "ring_radius_x_px": 360,
    "ring_radius_y_px": 250,
    "edge_width_px": 5,
    "arrow_head_length_px": 17,
    "arrow_head_width_px": 14,
    "label_font_size_px": 22,
    "panel_fill_rgb": (252, 252, 255),
    "panel_border_rgb": (88, 98, 112),
    "title_color_rgb": (34, 40, 48),
    "node_fill_rgb": (243, 247, 255),
    "node_border_rgb": (77, 90, 109),
    "label_color_rgb": (29, 34, 41),
    "label_stroke_rgb": (255, 255, 255),
    "edge_color_rgb": (88, 99, 114),
}
POST_IMAGE_BACKGROUND_DEFAULTS = load_diagrams_scene_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_diagrams_scene_noise_defaults(scene_id=SCENE, apply_prob=0.0)
