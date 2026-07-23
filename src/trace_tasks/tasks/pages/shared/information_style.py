"""Pages-domain adapter for shared structured-information styling."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

from ....core.scene_config import get_scene_defaults
from ...shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from ...shared.visual_style.information_scene import (
    Color,
    INFORMATION_SCENE_TREATMENT_IDS,
    InformationSceneStyle,
    make_information_scene_background,
    resolve_information_scene_style_from_request,
)
from ...shared.visual_style.request import (
    build_visual_style_request,
    resolve_style_bool,
)


PagesInformationStyle = InformationSceneStyle


def _information_scene_shadows_enabled(params: Mapping[str, Any]) -> bool:
    """Return whether pages adapters should draw shadow offsets for this scene."""

    if "information_scene_shadows_enabled" in params:
        return bool(params.get("information_scene_shadows_enabled"))
    policy = str(params.get("information_scene_shadow_policy", "auto")).strip().lower()
    return policy not in {"none", "off", "disabled", "disable"}


def _suppress_information_scene_shadows(
    style: PagesInformationStyle,
    metadata: Mapping[str, Any],
) -> tuple[PagesInformationStyle, dict[str, Any]]:
    """Zero non-semantic shadow offsets and keep trace metadata in sync."""

    original_shadow_offset = int(style.shadow_offset_px)
    adjusted = replace(style, shadow_offset_px=0)
    adjusted_meta = dict(metadata)
    layout_style = dict(adjusted_meta.get("layout_style", {}))
    layout_style["shadow_offset_px"] = 0
    adjusted_meta["layout_style"] = layout_style
    adapter_meta = dict(adjusted_meta.get("pages_adapter", {}))
    adapter_meta.update(
        {
            "information_scene_shadow_policy": "none",
            "original_shadow_offset_px": int(original_shadow_offset),
        }
    )
    adjusted_meta["pages_adapter"] = adapter_meta
    return adjusted, adjusted_meta


@lru_cache(maxsize=64)
def _pages_scene_render_defaults(scene_id: str) -> dict[str, Any]:
    """Return shared Pages rendering defaults for scene-style resolution."""

    try:
        defaults = get_scene_defaults("pages", str(scene_id))
    except Exception:
        return {}
    if not isinstance(defaults, Mapping):
        return {}
    _gen, rendering, _prompt = split_scene_generation_rendering_prompt_defaults(
        defaults,
        task_id=f"pages_{str(scene_id)}_style_defaults",
    )
    return dict(rendering)


def resolve_pages_information_style(
    *,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    scene_id: str,
    protected_colors: Sequence[Color] | None = None,
    allow_dark: bool | None = None,
    allow_colored_surface: bool | None = None,
) -> tuple[PagesInformationStyle, dict[str, Any]]:
    """Resolve one pages presentation style without changing visible values."""

    route_id = str(scene_id)
    default_params = _pages_scene_render_defaults(str(scene_id))
    resolved_params = {**default_params, **dict(params or {})}
    resolved_allow_dark = (
        resolve_style_bool(resolved_params, "information_scene_allow_dark", False)
        if allow_dark is None
        else bool(allow_dark)
    )
    resolved_allow_colored_surface = (
        resolve_style_bool(resolved_params, "information_scene_allow_colored_surface", True)
        if allow_colored_surface is None
        else bool(allow_colored_surface)
    )
    request = build_visual_style_request(
        domain="pages",
        scene_id=str(scene_id),
        routing_key=str(route_id),
        instance_seed=int(instance_seed),
        params=resolved_params,
        style_family="information_scene",
        allow_dark=bool(resolved_allow_dark),
        allow_colored_surface=bool(resolved_allow_colored_surface),
        protected_colors=protected_colors or (),
        required_text_roles=("page_title", "page_label", "page_value"),
    )
    style, metadata = resolve_information_scene_style_from_request(
        request,
        treatments=resolved_params.get("information_scene_treatments"),
        treatment_weights=resolved_params.get("information_scene_treatment_weights", {}),
        palettes=resolved_params.get("information_scene_palettes"),
        palette_weights=resolved_params.get("information_scene_palette_weights", {}),
        chrome_modes=resolved_params.get("information_scene_chrome_modes"),
        chrome_mode_weights=resolved_params.get("information_scene_chrome_mode_weights", {}),
    )
    if not _information_scene_shadows_enabled(resolved_params):
        return _suppress_information_scene_shadows(style, metadata)

    adapter_meta = dict(metadata.get("pages_adapter", {}))
    adapter_meta.setdefault("information_scene_shadow_policy", "auto")
    metadata["pages_adapter"] = adapter_meta
    return style, metadata


def make_pages_information_background(
    *,
    canvas_width: int,
    canvas_height: int,
    style: PagesInformationStyle,
    instance_seed: int,
    namespace: str,
) -> tuple[Any, dict[str, Any]]:
    """Create a Pages-domain background from the shared information style."""

    image, metadata = make_information_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    out = dict(metadata)
    out.setdefault("available_styles", [f"information_scene_style:{item}" for item in INFORMATION_SCENE_TREATMENT_IDS])
    return image, out


__all__ = [
    "make_pages_information_background",
    "PagesInformationStyle",
    "resolve_pages_information_style",
]
