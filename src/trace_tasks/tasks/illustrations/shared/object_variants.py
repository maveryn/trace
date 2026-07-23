"""Shared illustration object-variant profiles across renderer styles."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Mapping, Sequence, Tuple


RENDERER_STYLE_VECTOR = "vector"
RENDERER_STYLE_TOP_DOWN_PIXEL_RPG = "top_down_pixel_rpg"
RENDERER_STYLE_ISOMETRIC_PIXEL_RPG = "isometric_pixel_rpg"
ILLUSTRATION_RENDERER_STYLES: Tuple[str, ...] = (
    RENDERER_STYLE_VECTOR,
    RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
    RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
)


@dataclass(frozen=True)
class RendererVariantProfile:
    """Renderer-native treatment for one object variant."""

    renderer_style: str
    renderer_variant_id: str
    visual_attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectVariantProfile:
    """Stable object variant identity plus per-renderer visual profiles."""

    object_type: str
    variant_id: str
    public_name: str
    family: str
    queryable: bool
    semantic_attributes: Mapping[str, Any] = field(default_factory=dict)
    visual_attributes: Mapping[str, Any] = field(default_factory=dict)
    renderer_profiles: Mapping[str, RendererVariantProfile] = field(default_factory=dict)


def _renderer_profile(
    renderer_style: str,
    renderer_variant_id: str,
    **visual_attributes: Any,
) -> RendererVariantProfile:
    return RendererVariantProfile(
        renderer_style=str(renderer_style),
        renderer_variant_id=str(renderer_variant_id),
        visual_attributes=dict(visual_attributes),
    )


def _tree_profile(variant_id: str, public_name: str, *, vector_shape: str, fruit_visible: bool) -> ObjectVariantProfile:
    return ObjectVariantProfile(
        object_type="tree",
        variant_id=str(variant_id),
        public_name=str(public_name),
        family="plant",
        queryable=False,
        visual_attributes={"tree_style": str(variant_id)},
        renderer_profiles={
            RENDERER_STYLE_VECTOR: _renderer_profile(
                RENDERER_STYLE_VECTOR,
                str(vector_shape),
                crown_shape=str(vector_shape),
                fruit_visible=bool(fruit_visible),
            ),
            RENDERER_STYLE_TOP_DOWN_PIXEL_RPG: _renderer_profile(
                RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
                str(variant_id),
                tree_style=str(variant_id),
            ),
            RENDERER_STYLE_ISOMETRIC_PIXEL_RPG: _renderer_profile(
                RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
                str(variant_id),
                tree_style=str(variant_id),
            ),
        },
    )


def _person_profile(
    variant_id: str,
    public_name: str,
    *,
    vector_accessory: str,
    pixel_accessory: str,
    scale: float = 1.0,
) -> ObjectVariantProfile:
    return ObjectVariantProfile(
        object_type="person",
        variant_id=str(variant_id),
        public_name=str(public_name),
        family="person",
        queryable=False,
        visual_attributes={"person_variant_id": str(variant_id)},
        renderer_profiles={
            RENDERER_STYLE_VECTOR: _renderer_profile(
                RENDERER_STYLE_VECTOR,
                str(variant_id),
                person_variant_id=str(variant_id),
                accessory=str(vector_accessory),
                scale=float(scale),
            ),
            RENDERER_STYLE_TOP_DOWN_PIXEL_RPG: _renderer_profile(
                RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
                str(variant_id),
                person_variant_id=str(variant_id),
                accessory=str(pixel_accessory),
                scale=float(scale),
            ),
            RENDERER_STYLE_ISOMETRIC_PIXEL_RPG: _renderer_profile(
                RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
                str(variant_id),
                person_variant_id=str(variant_id),
                accessory=str(pixel_accessory),
                scale=float(scale),
            ),
        },
    )


TREE_VARIANT_IDS: Tuple[str, ...] = ("oak", "pine", "maple", "fruit_tree")
PERSON_VARIANT_IDS: Tuple[str, ...] = ("adult", "farmer", "worker", "vendor", "soldier")


OBJECT_VARIANT_PROFILES: Mapping[tuple[str, str], ObjectVariantProfile] = {
    ("tree", "oak"): _tree_profile("oak", "oak tree", vector_shape="round_crown", fruit_visible=False),
    ("tree", "pine"): _tree_profile("pine", "pine tree", vector_shape="conifer", fruit_visible=False),
    ("tree", "maple"): _tree_profile("maple", "maple tree", vector_shape="lobed_crown", fruit_visible=False),
    ("tree", "fruit_tree"): _tree_profile("fruit_tree", "fruit tree", vector_shape="fruit_crown", fruit_visible=True),
    ("person", "adult"): _person_profile("adult", "adult", vector_accessory="none", pixel_accessory="none"),
    ("person", "farmer"): _person_profile("farmer", "farmer", vector_accessory="straw_hat", pixel_accessory="straw_hat"),
    ("person", "worker"): _person_profile("worker", "worker", vector_accessory="hard_hat", pixel_accessory="hard_hat"),
    ("person", "vendor"): _person_profile("vendor", "vendor", vector_accessory="cap_apron", pixel_accessory="cap_apron"),
    ("person", "soldier"): _person_profile("soldier", "soldier", vector_accessory="helmet", pixel_accessory="helmet"),
}


def object_variant_ids(object_type: str) -> Tuple[str, ...]:
    """Return supported stable variant ids for one illustration object type."""

    key = str(object_type)
    if key == "tree":
        return TREE_VARIANT_IDS
    if key == "person":
        return PERSON_VARIANT_IDS
    return tuple(
        variant_id
        for obj_type, variant_id in sorted(OBJECT_VARIANT_PROFILES)
        if obj_type == key
    )


def normalize_object_variant_id(object_type: str, variant_id: str | None) -> str:
    """Return a supported variant id, falling back to the first registered one."""

    support = object_variant_ids(str(object_type))
    if not support:
        return str(variant_id or "")
    value = str(variant_id or support[0])
    if value in set(support):
        return value
    return str(support[0])


def object_variant_profile(object_type: str, variant_id: str | None) -> ObjectVariantProfile:
    """Return the shared variant profile for one object type and variant."""

    normalized = normalize_object_variant_id(str(object_type), variant_id)
    try:
        return OBJECT_VARIANT_PROFILES[(str(object_type), normalized)]
    except KeyError as exc:
        raise ValueError(f"unsupported illustration object variant: {object_type}/{variant_id}") from exc


def renderer_variant_profile(
    object_type: str,
    variant_id: str | None,
    renderer_style: str,
) -> RendererVariantProfile:
    """Return the renderer-native variant treatment for one object variant."""

    profile = object_variant_profile(str(object_type), variant_id)
    style = str(renderer_style)
    try:
        return profile.renderer_profiles[style]
    except KeyError as exc:
        raise ValueError(f"renderer style {style!r} does not support {object_type}/{profile.variant_id}") from exc


def renderer_supported_variant_ids(object_type: str, renderer_style: str) -> Tuple[str, ...]:
    """Return variant ids that have a profile for the requested renderer style."""

    style = str(renderer_style)
    return tuple(
        variant_id
        for variant_id in object_variant_ids(str(object_type))
        if style in object_variant_profile(str(object_type), variant_id).renderer_profiles
    )


def variant_visual_metadata(
    object_type: str,
    variant_id: str | None,
    renderer_style: str,
) -> dict[str, Any]:
    """Return trace-ready visual metadata for one resolved object variant."""

    profile = object_variant_profile(str(object_type), variant_id)
    renderer_profile = renderer_variant_profile(str(object_type), profile.variant_id, str(renderer_style))
    metadata: dict[str, Any] = {
        "object_variant_id": str(profile.variant_id),
        "object_variant_public_name": str(profile.public_name),
        "object_variant_queryable": bool(profile.queryable),
        "renderer_style": str(renderer_profile.renderer_style),
        "renderer_variant_id": str(renderer_profile.renderer_variant_id),
    }
    metadata.update(dict(profile.visual_attributes))
    metadata.update(dict(renderer_profile.visual_attributes))
    return metadata


def sample_object_variant_id(
    rng: random.Random,
    object_type: str,
    support: Sequence[str] | None = None,
) -> str:
    """Sample a stable object variant id from explicit support."""

    values = tuple(str(value) for value in (support if support is not None else object_variant_ids(str(object_type))))
    if not values:
        raise ValueError(f"no illustration object variants registered for {object_type!r}")
    invalid = [value for value in values if value not in set(object_variant_ids(str(object_type)))]
    if invalid:
        raise ValueError(f"unsupported {object_type!r} variant ids: {invalid}")
    return str(values[int(rng.randrange(len(values)))])


__all__ = [
    "ILLUSTRATION_RENDERER_STYLES",
    "OBJECT_VARIANT_PROFILES",
    "PERSON_VARIANT_IDS",
    "RENDERER_STYLE_ISOMETRIC_PIXEL_RPG",
    "RENDERER_STYLE_TOP_DOWN_PIXEL_RPG",
    "RENDERER_STYLE_VECTOR",
    "RendererVariantProfile",
    "ObjectVariantProfile",
    "TREE_VARIANT_IDS",
    "normalize_object_variant_id",
    "object_variant_ids",
    "object_variant_profile",
    "renderer_supported_variant_ids",
    "renderer_variant_profile",
    "sample_object_variant_id",
    "variant_visual_metadata",
]
