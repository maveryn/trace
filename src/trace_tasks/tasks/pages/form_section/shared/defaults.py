"""Defaults for form-section page tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .forms import DocumentDefaults


DOMAIN = "pages"
SCENE = "form_section"
PROMPT_BUNDLE = "pages_form_section_v1"
PROMPT_SCENE_KEY = "structured_document_sections"
PROMPT_TASK_KEY = "section_expression_query"
NAMESPACE_ROOT = "pages.form_section"

DEFAULTS = DocumentDefaults()
SCENE_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
GENERATION_DEFAULTS, RENDERING_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, Mapping) else {},
)
RENDER_FALLBACKS: Dict[str, Any] = {
    "canvas_width": int(DEFAULTS.canvas_width),
    "canvas_height": int(DEFAULTS.canvas_height),
    "sheet_page_width_px": int(DEFAULTS.sheet_page_width_px),
    "sheet_page_height_px": int(DEFAULTS.sheet_page_height_px),
    "receipt_page_width_px": int(DEFAULTS.receipt_page_width_px),
    "receipt_page_height_px": int(DEFAULTS.receipt_page_height_px),
    "page_shadow_offset_px": int(DEFAULTS.page_shadow_offset_px),
    "page_corner_radius_px": int(DEFAULTS.page_corner_radius_px),
    "field_corner_radius_px": int(DEFAULTS.field_corner_radius_px),
    "page_outline_width_px": int(DEFAULTS.page_outline_width_px),
    "field_outline_width_px": int(DEFAULTS.field_outline_width_px),
    "title_font_size_px": int(DEFAULTS.title_font_size_px),
    "section_font_size_px": int(DEFAULTS.section_font_size_px),
    "label_font_size_px": int(DEFAULTS.label_font_size_px),
    "value_font_size_px": int(DEFAULTS.value_font_size_px),
    "page_fill_rgb": tuple(DEFAULTS.page_fill_rgb),
    "page_outline_rgb": tuple(DEFAULTS.page_outline_rgb),
    "page_shadow_rgb": tuple(DEFAULTS.page_shadow_rgb),
    "field_fill_rgb": tuple(DEFAULTS.field_fill_rgb),
    "field_outline_rgb": tuple(DEFAULTS.field_outline_rgb),
    "label_fill_rgb": tuple(DEFAULTS.label_fill_rgb),
    "label_stroke_rgb": tuple(DEFAULTS.label_stroke_rgb),
    "value_fill_rgb": tuple(DEFAULTS.value_fill_rgb),
    "divider_rgb": tuple(DEFAULTS.divider_rgb),
}
POST_IMAGE_BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)


__all__ = [
    "DEFAULTS",
    "DOMAIN",
    "GENERATION_DEFAULTS",
    "NAMESPACE_ROOT",
    "POST_IMAGE_BACKGROUND_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_BUNDLE",
    "PROMPT_SCENE_KEY",
    "PROMPT_TASK_KEY",
    "PROMPT_DEFAULTS",
    "RENDERING_DEFAULTS",
    "RENDER_FALLBACKS",
    "SCENE",
]
