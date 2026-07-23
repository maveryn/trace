"""Renderer-variant metadata for reusable three_d objects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple


RENDERER_STYLE_PROJECTED_3D = "projected_3d"


@dataclass(frozen=True)
class ThreeDRendererVariantProfile:
    """One renderer-facing variant profile for a three_d object."""

    renderer_id: str
    renderer_style: str = RENDERER_STYLE_PROJECTED_3D
    renderer_variant_id: str = ""
    public_name_suffix: str = ""
    visual_attributes: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ThreeDObjectVariantProfile:
    """Render-only variation metadata for a canonical object."""

    object_type: str
    variant_id: str
    public_name: str = ""
    renderer_variants: Tuple[ThreeDRendererVariantProfile, ...] = ()
    visual_attributes: Mapping[str, Any] | None = None


def object_variant_ids(object_type: str) -> Tuple[str, ...]:
    """Return registered variant ids for ``object_type``.

    Three_d currently keeps most variation in scene profiles and render attrs.
    This hook mirrors the illustration-domain API so future object variants can
    be promoted without changing scene call sites.
    """

    _ = str(object_type)
    return ()


def normalize_object_variant_id(object_type: str, variant_id: str | None) -> str:
    """Return a stable registered variant id, or an empty string."""

    requested = str(variant_id or "")
    if not requested:
        return ""
    return requested if requested in set(object_variant_ids(str(object_type))) else requested


def variant_visual_metadata(
    object_type: str,
    *,
    variant_id: str = "",
    renderer_id: str = "",
    renderer_style: str = RENDERER_STYLE_PROJECTED_3D,
    renderer_variant_id: str = "",
) -> dict[str, Any]:
    """Return trace metadata describing the selected object renderer variant."""

    return {
        "object_variant_id": normalize_object_variant_id(str(object_type), variant_id),
        "renderer_id": str(renderer_id),
        "renderer_style": str(renderer_style),
        "renderer_variant_id": str(renderer_variant_id),
    }


__all__ = [
    "RENDERER_STYLE_PROJECTED_3D",
    "ThreeDObjectVariantProfile",
    "ThreeDRendererVariantProfile",
    "normalize_object_variant_id",
    "object_variant_ids",
    "variant_visual_metadata",
]
