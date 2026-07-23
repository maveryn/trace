"""Shared scene/placement schema for three_d renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

from .object_schema import json_safe
from .object_variants import RENDERER_STYLE_PROJECTED_3D


RGB = Tuple[int, int, int]
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class ThreeDPlacementSpec:
    """Renderer-neutral placement for one reusable 3D object in a finalized scene."""

    object_id: str
    object_type: str
    role: str = "distractor"
    shape_type: str = ""
    public_name: str = ""
    canonical_id: str = ""
    profile_id: str = ""
    family: str = "object"
    renderer_id: str = ""
    renderer_variant_id: str = ""
    variant_id: str = ""
    world_xyz: Point3D | None = None
    base_xyz: Point3D | None = None
    dimensions_xyz: Point3D | None = None
    screen_xy: Point2D | None = None
    semantic_attributes: Mapping[str, Any] = field(default_factory=dict)
    visual_attributes: Mapping[str, Any] = field(default_factory=dict)
    parts: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    source_entity_type: str = "three_d_object"
    render_spec: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        spec: Mapping[str, Any],
        *,
        object_type_key: str = "object_type",
        role: str | None = None,
        source_entity_type: str = "three_d_object",
    ) -> "ThreeDPlacementSpec":
        """Build a placement spec from an existing scene-local object mapping."""

        object_type = str(spec.get(object_type_key, spec.get("object_type", spec.get("shape_type", ""))))
        shape_type = str(spec.get("shape_type", object_type))
        public_name = str(spec.get("prompt_name", spec.get("object_name", object_type.replace("_", " "))))
        return cls(
            object_id=str(spec.get("object_id", "")),
            object_type=object_type,
            role=str(role if role is not None else spec.get("object_role", "distractor")),
            shape_type=shape_type,
            public_name=public_name,
            canonical_id=str(spec.get("canonical_id", "")),
            profile_id=str(spec.get("profile_id", "")),
            family=str(spec.get("family", spec.get("resource_kind", "object"))),
            renderer_id=str(spec.get("renderer_id", "")),
            renderer_variant_id=str(spec.get("renderer_variant_id", "")),
            variant_id=str(spec.get("variant_id", spec.get("object_variant_id", ""))),
            world_xyz=_point3(spec.get("world_xyz")),
            base_xyz=_point3(spec.get("base_xyz")),
            dimensions_xyz=_point3(spec.get("dimensions_xyz")),
            screen_xy=_point2(spec.get("screen_xy")),
            parts=tuple(dict(part) for part in spec.get("parts", ())),
            source_entity_type=str(source_entity_type),
            render_spec=dict(spec),
        )

    def as_render_mapping(self, *, profile: Any | None = None) -> dict[str, Any]:
        """Return a legacy-compatible object mapping with profile defaults applied."""

        mapping = dict(self.render_spec)
        profile_object_type = str(getattr(profile, "object_type", "")) if profile is not None else ""
        object_type = str(self.object_type or profile_object_type)
        shape_type = str(self.shape_type or mapping.get("shape_type", object_type))
        public_name = str(self.public_name or getattr(profile, "display_name", "") or object_type.replace("_", " "))
        renderer_id = str(self.renderer_id or getattr(profile, "renderer", ""))
        canonical_id = str(self.canonical_id or getattr(profile, "canonical_id", "") or object_type)
        profile_id = str(self.profile_id or getattr(profile, "profile_id", ""))
        family = str(self.family or getattr(profile, "resource_kind", "object") or "object")
        mapping.update(
            {
                "object_id": str(self.object_id),
                "object_type": object_type,
                "shape_type": shape_type,
                "object_name": str(mapping.get("object_name", public_name)),
                "prompt_name": str(mapping.get("prompt_name", public_name)),
                "object_role": str(self.role),
                "canonical_id": canonical_id,
                "family": family,
                "resource_kind": str(mapping.get("resource_kind", getattr(profile, "resource_kind", family))),
            }
        )
        if profile_id:
            mapping["profile_id"] = profile_id
        if renderer_id:
            mapping["renderer_id"] = renderer_id
        if self.renderer_variant_id:
            mapping["renderer_variant_id"] = str(self.renderer_variant_id)
        if self.variant_id:
            mapping["variant_id"] = str(self.variant_id)
        if self.world_xyz is not None:
            mapping["world_xyz"] = _rounded_point3(self.world_xyz)
        if self.base_xyz is not None:
            mapping["base_xyz"] = _rounded_point3(self.base_xyz)
        if self.dimensions_xyz is not None:
            mapping["dimensions_xyz"] = _rounded_point3(self.dimensions_xyz)
        if self.screen_xy is not None:
            mapping["screen_xy"] = _rounded_point2(self.screen_xy)
        for key, value in self.semantic_attributes.items():
            mapping.setdefault(str(key), json_safe(value))
        for key, value in self.visual_attributes.items():
            mapping.setdefault(str(key), json_safe(value))
        return mapping


@dataclass(frozen=True)
class ThreeDSceneStyleSpec:
    """Resolved non-semantic style metadata for one finalized 3D scene."""

    scene_id: str
    scene_variant: str
    renderer_style: str = RENDERER_STYLE_PROJECTED_3D
    camera_style_id: str = ""
    palette_id: str = ""
    surface_style_id: str = ""
    background_style_id: str = ""
    colors: Mapping[str, RGB] = field(default_factory=dict)
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene_id": str(self.scene_id),
            "scene_variant": str(self.scene_variant),
            "renderer_style": str(self.renderer_style),
            "camera_style_id": str(self.camera_style_id),
            "palette_id": str(self.palette_id),
            "surface_style_id": str(self.surface_style_id),
            "background_style_id": str(self.background_style_id),
            "colors": {str(key): [int(channel) for channel in value] for key, value in self.colors.items()},
            "params": json_safe(dict(self.params)),
        }


def _point2(value: Any) -> Point2D | None:
    if value is None:
        return None
    values = [float(item) for item in value]
    if len(values) != 2:
        raise ValueError("2D point must contain exactly two values")
    return (values[0], values[1])


def _point3(value: Any) -> Point3D | None:
    if value is None:
        return None
    values = [float(item) for item in value]
    if len(values) != 3:
        raise ValueError("3D point must contain exactly three values")
    return (values[0], values[1], values[2])


def _rounded_point2(value: Sequence[float]) -> list[float]:
    return [round(float(item), 3) for item in value]


def _rounded_point3(value: Sequence[float]) -> list[float]:
    return [round(float(item), 4) for item in value]


__all__ = [
    "Point2D",
    "Point3D",
    "RGB",
    "ThreeDPlacementSpec",
    "ThreeDSceneStyleSpec",
]
