"""Shared request/metadata helpers for non-semantic visual styling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple


Color = Tuple[int, int, int]

VISUAL_STYLE_FAMILIES: Tuple[str, ...] = (
    "information_scene",
    "panel_scene",
    "technical_diagram",
)


def _normalize_color(value: Sequence[int]) -> Color:
    if len(value) < 3:
        raise ValueError("RGB colors require three channels")
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])


def resolve_style_bool(
    params: Mapping[str, Any] | None,
    key: str,
    fallback: bool,
) -> bool:
    """Resolve one permissive boolean visual-style config value."""

    value = (params or {}).get(str(key), fallback)
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


@dataclass(frozen=True)
class VisualStyleRequest:
    """Domain-neutral request for one non-semantic scene style."""

    domain: str
    scene_id: str
    routing_key: str
    instance_seed: int
    style_family: str
    params: Mapping[str, Any]
    allow_dark: bool = False
    allow_colored_surface: bool = True
    protected_colors: Tuple[Color, ...] = ()
    required_text_roles: Tuple[str, ...] = ()


def build_visual_style_request(
    *,
    domain: str,
    scene_id: str,
    routing_key: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    style_family: str,
    allow_dark: bool = False,
    allow_colored_surface: bool = True,
    protected_colors: Sequence[Sequence[int]] | None = None,
    required_text_roles: Sequence[str] | None = None,
) -> VisualStyleRequest:
    """Normalize a visual-style request shared by domain adapters."""

    family = str(style_family).strip()
    if family not in set(VISUAL_STYLE_FAMILIES):
        raise ValueError(f"unsupported visual style family: {family}")
    return VisualStyleRequest(
        domain=str(domain),
        scene_id=str(scene_id),
        routing_key=str(routing_key),
        instance_seed=int(instance_seed),
        style_family=str(family),
        params=dict(params or {}),
        allow_dark=bool(allow_dark),
        allow_colored_surface=bool(allow_colored_surface),
        protected_colors=tuple(_normalize_color(color) for color in (protected_colors or ())),
        required_text_roles=tuple(str(role) for role in (required_text_roles or ())),
    )


def visual_style_request_metadata(request: VisualStyleRequest) -> dict[str, Any]:
    """Serialize one style request into trace metadata."""

    return {
        "domain": str(request.domain),
        "scene_id": str(request.scene_id),
        "routing_key": str(request.routing_key),
        "style_family": str(request.style_family),
        "allow_dark": bool(request.allow_dark),
        "allow_colored_surface": bool(request.allow_colored_surface),
        "protected_colors_rgb": [list(color) for color in request.protected_colors],
        "required_text_roles": [str(role) for role in request.required_text_roles],
    }


__all__ = [
    "Color",
    "VISUAL_STYLE_FAMILIES",
    "VisualStyleRequest",
    "build_visual_style_request",
    "resolve_style_bool",
    "visual_style_request_metadata",
]
