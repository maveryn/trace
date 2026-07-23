"""Structured-document style adapter for form-section pages."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.pages.shared.information_style import PagesInformationStyle, resolve_pages_information_style
from trace_tasks.tasks.shared.visual_style.information_scene import Color, make_information_scene_background

from .forms import DocumentRenderParams


def _shadows_enabled(params: Mapping[str, Any]) -> bool:
    if "information_scene_shadows_enabled" in params:
        return bool(params.get("information_scene_shadows_enabled"))
    policy = str(params.get("information_scene_shadow_policy", "auto")).strip().lower()
    return policy not in {"none", "off", "disabled", "disable"}


def apply_document_information_style(
    render_params: DocumentRenderParams,
    style: PagesInformationStyle,
    *,
    suppress_shadows: bool = False,
) -> DocumentRenderParams:
    """Map shared information-scene style roles into structured-document chrome."""

    return replace(
        render_params,
        page_shadow_offset_px=0 if bool(suppress_shadows) else int(render_params.page_shadow_offset_px),
        page_fill_rgb=tuple(int(value) for value in style.surface_rgb),
        page_outline_rgb=tuple(int(value) for value in style.panel_border_rgb),
        page_shadow_rgb=tuple(int(value) for value in style.shadow_rgb),
        field_fill_rgb=tuple(int(value) for value in style.panel_fill_rgb),
        field_outline_rgb=tuple(int(value) for value in style.panel_border_rgb),
        label_fill_rgb=tuple(int(value) for value in style.muted_text_rgb),
        label_stroke_rgb=tuple(int(value) for value in style.text_stroke_rgb),
        value_fill_rgb=tuple(int(value) for value in style.text_rgb),
        divider_rgb=tuple(int(value) for value in style.guide_rgb),
    )


def prepare_document_information_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_id: str,
    render_params: DocumentRenderParams,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool | None = None,
) -> tuple[DocumentRenderParams, Any, dict[str, Any], dict[str, Any]]:
    """Resolve pages information style, apply it, and create the document background."""

    style, style_meta = resolve_pages_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=str(scene_id),
        protected_colors=protected_colors or (),
        allow_dark=allow_dark,
    )
    styled_render_params = apply_document_information_style(
        render_params,
        style,
        suppress_shadows=not _shadows_enabled(params),
    )
    background, background_meta = make_information_scene_background(
        canvas_width=int(styled_render_params.canvas_width),
        canvas_height=int(styled_render_params.canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"pages.{str(scene_id)}.information_scene_background",
    )
    return styled_render_params, background, background_meta, style_meta


__all__ = [
    "apply_document_information_style",
    "prepare_document_information_scene",
]
