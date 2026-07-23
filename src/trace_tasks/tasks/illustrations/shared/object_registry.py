"""Global illustration object taxonomy and normalized record helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .object_catalog import canonical_entry_for_object_type
from .object_library import OBJECT_TEMPLATES, display_name_for_object_type
from .object_schema import BBox, ObjectAttributeDef, ObjectRecord, ObjectTypeDef


def _attrs(*names: str, public: bool = True) -> Tuple[ObjectAttributeDef, ...]:
    return tuple(ObjectAttributeDef(str(name), public=bool(public)) for name in names)


def _base_object_defs() -> Dict[str, ObjectTypeDef]:
    return {
        str(object_type): ObjectTypeDef(
            object_type=str(object_type),
            public_name=display_name_for_object_type(str(object_type)),
            family=str(template.family),
            visual_attributes=_attrs(
                "primary_color_rgb",
                "accent_color_rgb",
                "style_id",
                "object_variant_id",
                "renderer_style",
                "renderer_variant_id",
            ),
        )
        for object_type, template in sorted(OBJECT_TEMPLATES.items())
    }


_SCENE_OBJECT_DEFS: Dict[str, ObjectTypeDef] = {
    "boarding_area": ObjectTypeDef(
        "boarding_area",
        "boarding area",
        "scene_region",
        semantic_attributes=_attrs("area_id", "display_name"),
    ),
    "book": ObjectTypeDef(
        "book",
        "book",
        "object",
        semantic_attributes=_attrs("section_id", "section_key", "section_name", "color_name", "color_label", "orientation"),
        visual_attributes=_attrs("primary_color_rgb", "accent_color_rgb", "style_id", "color_rgb"),
    ),
    "building": ObjectTypeDef(
        "building",
        "building",
        "structure",
        semantic_attributes=_attrs("roof_type", "window_count", "lit_window_count"),
    ),
    "construction_equipment": ObjectTypeDef(
        "construction_equipment",
        "construction equipment",
        "equipment",
        semantic_attributes=_attrs("equipment_type", "equipment_label", "zone_id"),
        visual_attributes=_attrs("style_id"),
    ),
    "construction_material": ObjectTypeDef(
        "construction_material",
        "construction material",
        "material",
        semantic_attributes=_attrs("material_type", "material_label"),
        visual_attributes=_attrs("style_id"),
    ),
    "container": ObjectTypeDef(
        "container",
        "container",
        "fixture",
        semantic_attributes=_attrs("container_type", "label"),
    ),
    "basket": ObjectTypeDef(
        "basket",
        "basket",
        "container",
        visual_attributes=_attrs("wicker_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "bed": ObjectTypeDef(
        "bed",
        "bed",
        "fixture",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("bed_size", "wood_rgb", "blanket_rgb", "pillow_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "chest": ObjectTypeDef(
        "chest",
        "chest",
        "container",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("wood_rgb", "metal_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "boulder": ObjectTypeDef(
        "boulder",
        "boulder",
        "obstacle",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "archway": ObjectTypeDef(
        "archway",
        "archway",
        "structure",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "shadow_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "brazier": ObjectTypeDef(
        "brazier",
        "brazier",
        "fixture",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("metal_rgb", "flame_rgb", "fire_state", "renderer_style", "renderer_variant_id"),
    ),
    "broken_wall": ObjectTypeDef(
        "broken_wall",
        "broken wall",
        "structure",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "break_style", "renderer_style", "renderer_variant_id"),
    ),
    "cave_entrance": ObjectTypeDef(
        "cave_entrance",
        "cave entrance",
        "terrain_feature",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "shadow_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "crystal_cluster": ObjectTypeDef(
        "crystal_cluster",
        "crystal cluster",
        "resource",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("crystal_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "counter": ObjectTypeDef(
        "counter",
        "counter",
        "fixture",
        semantic_attributes=_attrs("zone_id", "surface_id"),
        visual_attributes=_attrs("wood_rgb", "top_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "decor": ObjectTypeDef(
        "decor",
        "decor",
        "support",
        semantic_attributes=_attrs("decor_type"),
    ),
    "domestic_animal": ObjectTypeDef(
        "domestic_animal",
        "domestic animal",
        "animal",
        semantic_attributes=_attrs("animal_type", "region_id", "inside_pen"),
        visual_attributes=_attrs("facing", "body_rgb", "accent_rgb", "spot_rgb", "renderer_style", "animal_type"),
    ),
    "crop_row": ObjectTypeDef(
        "crop_row",
        "crop row",
        "plant",
        semantic_attributes=_attrs("territory_id", "region_id"),
        visual_attributes=_attrs("crop_style", "crop_rgb", "soil_rgb", "theme_id", "autumn_intensity", "leaf_coverage", "leaf_style", "leaf_overlay_rgb", "leaf_shadow_rgb", "leaf_accent_rgb", "renderer_style"),
    ),
    "vegetable_patch": ObjectTypeDef(
        "vegetable_patch",
        "vegetable patch",
        "vegetable",
        semantic_attributes=_attrs("territory_id", "region_id"),
        visual_attributes=_attrs("vegetable_style", "vegetable_rgb", "leaf_rgb", "soil_rgb", "renderer_style"),
    ),
    "environment_feature": ObjectTypeDef(
        "environment_feature",
        "environment feature",
        "scene_region",
        semantic_attributes=_attrs("feature_type"),
    ),
    "fireplace": ObjectTypeDef(
        "fireplace",
        "fireplace",
        "fixture",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("stone_rgb", "fire_state", "flame_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "floor_switch": ObjectTypeDef(
        "floor_switch",
        "floor switch",
        "fixture",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("plate_rgb", "switch_state", "renderer_style", "renderer_variant_id"),
    ),
    "furniture": ObjectTypeDef(
        "furniture",
        "furniture",
        "fixture",
        semantic_attributes=_attrs("furniture_type", "label"),
    ),
    "library_section": ObjectTypeDef(
        "library_section",
        "library section",
        "scene_region",
        semantic_attributes=_attrs("section_key", "section_name", "label"),
    ),
    "gazebo": ObjectTypeDef(
        "gazebo",
        "gazebo",
        "structure",
        visual_attributes=_attrs("roof_rgb", "wood_rgb", "theme_id", "snow_intensity", "snow_coverage", "snow_style", "autumn_intensity", "leaf_coverage", "leaf_style", "leaf_overlay_rgb", "leaf_shadow_rgb", "leaf_accent_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "jar": ObjectTypeDef(
        "jar",
        "jar",
        "container",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("glass_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "ladder": ObjectTypeDef(
        "ladder",
        "ladder",
        "route_feature",
        semantic_attributes=_attrs("region_id", "route_id", "level_id"),
        visual_attributes=_attrs("wood_rgb", "orientation", "renderer_style", "renderer_variant_id"),
    ),
    "lamp_post": ObjectTypeDef(
        "lamp_post",
        "lamp post",
        "object",
        visual_attributes=_attrs("glow_rgb", "metal_rgb", "theme_id", "snow_intensity", "snow_coverage", "snow_style", "autumn_intensity", "leaf_coverage", "leaf_style", "leaf_overlay_rgb", "leaf_shadow_rgb", "leaf_accent_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "luggage": ObjectTypeDef(
        "luggage",
        "luggage",
        "object",
        semantic_attributes=_attrs("luggage_type", "area_id"),
        visual_attributes=_attrs("primary_color_rgb"),
    ),
    "market_stall": ObjectTypeDef(
        "market_stall",
        "market stall",
        "fixture",
        visual_attributes=_attrs("canopy_rgb", "wood_rgb", "goods_type", "theme_id", "snow_intensity", "snow_coverage", "snow_style", "autumn_intensity", "leaf_coverage", "leaf_style", "leaf_overlay_rgb", "leaf_shadow_rgb", "leaf_accent_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "magic_circle": ObjectTypeDef(
        "magic_circle",
        "magic circle",
        "fixture",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("rune_rgb", "glow_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "mine_cart": ObjectTypeDef(
        "mine_cart",
        "mine cart",
        "vehicle",
        semantic_attributes=_attrs("region_id", "route_id"),
        visual_attributes=_attrs("body_rgb", "ore_rgb", "orientation", "renderer_style", "renderer_variant_id"),
    ),
    "person": ObjectTypeDef(
        "person",
        "person",
        "person",
        semantic_attributes=_attrs("activity", "activity_label", "area_id", "pose_id", "zone"),
        visual_attributes=_attrs("primary_color_rgb", "accent_color_rgb", "skin_color_rgb", "style_id", "gender_id", "object_variant_id", "person_variant_id", "renderer_style", "renderer_variant_id"),
    ),
    "pond": ObjectTypeDef(
        "pond",
        "pond",
        "scene_region",
        semantic_attributes=_attrs("feature_type"),
        visual_attributes=_attrs("pond_shape", "water_rgb", "rim_rgb", "theme_id", "snow_intensity", "snow_coverage", "snow_style", "autumn_intensity", "leaf_coverage", "leaf_style", "leaf_overlay_rgb", "leaf_shadow_rgb", "leaf_accent_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "pot": ObjectTypeDef(
        "pot",
        "pot",
        "container",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("clay_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "ore_vein": ObjectTypeDef(
        "ore_vein",
        "ore vein",
        "resource",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "ore_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "produce_bin": ObjectTypeDef(
        "produce_bin",
        "produce bin",
        "fixture",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("goods_type", "wood_rgb", "produce_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "pedestrian_with_bag": ObjectTypeDef(
        "pedestrian_with_bag",
        "pedestrian with bag",
        "person",
        semantic_attributes=_attrs("activity", "activity_label", "area_id", "pose_id", "zone"),
        visual_attributes=_attrs("primary_color_rgb", "accent_color_rgb", "skin_color_rgb", "style_id", "gender_id", "object_variant_id", "person_variant_id", "renderer_style", "renderer_variant_id"),
    ),
    "playground_equipment": ObjectTypeDef(
        "playground_equipment",
        "playground equipment",
        "equipment",
        semantic_attributes=_attrs("equipment_type", "equipment_label", "zone"),
    ),
    "service_point": ObjectTypeDef(
        "service_point",
        "service point",
        "fixture",
        semantic_attributes=_attrs("service_point_id", "display_name"),
    ),
    "rug": ObjectTypeDef(
        "rug",
        "rug",
        "fixture",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("cloth_rgb", "trim_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "rail_track": ObjectTypeDef(
        "rail_track",
        "rail track",
        "route_feature",
        semantic_attributes=_attrs("region_id", "route_id"),
        visual_attributes=_attrs("wood_rgb", "rail_rgb", "track_shape", "renderer_style", "renderer_variant_id"),
    ),
    "rubble": ObjectTypeDef(
        "rubble",
        "rubble",
        "obstacle",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "room_divider": ObjectTypeDef(
        "room_divider",
        "room divider",
        "fixture",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("divider_style", "wood_rgb", "cloth_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "sack": ObjectTypeDef(
        "sack",
        "sack",
        "container",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("cloth_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "shelf": ObjectTypeDef(
        "shelf",
        "shelf",
        "fixture",
        semantic_attributes=_attrs("zone_id", "surface_id"),
        visual_attributes=_attrs("goods_type", "wood_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "sealed_door": ObjectTypeDef(
        "sealed_door",
        "sealed door",
        "structure",
        semantic_attributes=_attrs("region_id", "route_id"),
        visual_attributes=_attrs("stone_rgb", "seal_rgb", "door_orientation", "renderer_style", "renderer_variant_id"),
    ),
    "stairs": ObjectTypeDef(
        "stairs",
        "stairs",
        "route_feature",
        semantic_attributes=_attrs("region_id", "route_id", "level_id"),
        visual_attributes=_attrs("stone_rgb", "stair_direction", "renderer_style", "renderer_variant_id"),
    ),
    "stool": ObjectTypeDef(
        "stool",
        "stool",
        "fixture",
        semantic_attributes=_attrs("zone_id"),
        visual_attributes=_attrs("wood_rgb", "cushion_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "stalagmite": ObjectTypeDef(
        "stalagmite",
        "stalagmite",
        "obstacle",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "stone_column": ObjectTypeDef(
        "stone_column",
        "stone column",
        "structure",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("stone_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "surface": ObjectTypeDef(
        "surface",
        "surface",
        "fixture",
        semantic_attributes=_attrs("surface_type", "label", "furniture_id"),
    ),
    "torch": ObjectTypeDef(
        "torch",
        "torch",
        "fixture",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("wood_rgb", "flame_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "worker": ObjectTypeDef(
        "worker",
        "worker",
        "person",
        semantic_attributes=_attrs("hard_hat_color", "vest_color", "tool_type"),
        visual_attributes=_attrs("style_id", "gender_id"),
    ),
    "wagon": ObjectTypeDef(
        "wagon",
        "wagon",
        "vehicle",
        visual_attributes=_attrs("facing", "body_rgb", "cover_rgb", "theme_id", "snow_intensity", "snow_coverage", "snow_style", "autumn_intensity", "leaf_coverage", "leaf_style", "leaf_overlay_rgb", "leaf_shadow_rgb", "leaf_accent_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "wood_support": ObjectTypeDef(
        "wood_support",
        "wood support",
        "structure",
        semantic_attributes=_attrs("region_id"),
        visual_attributes=_attrs("wood_rgb", "renderer_style", "renderer_variant_id"),
    ),
    "zone": ObjectTypeDef(
        "zone",
        "zone",
        "scene_region",
        semantic_attributes=_attrs("zone_id", "label", "zone_type"),
        visual_attributes=_attrs("fill_rgb", "outline_rgb"),
    ),
}

def _with_catalog_defaults(definition: ObjectTypeDef) -> ObjectTypeDef:
    entry = canonical_entry_for_object_type(str(definition.object_type))
    if entry is None:
        return definition
    return ObjectTypeDef(
        object_type=str(definition.object_type),
        public_name=str(definition.public_name),
        family=str(definition.family),
        render_layer=str(entry.render_layer),
        size_class=str(entry.size_class),
        placement_tags=tuple(entry.placement_tags),
        scene_tags=tuple(entry.scene_tags),
        semantic_attributes=tuple(definition.semantic_attributes),
        visual_attributes=tuple(definition.visual_attributes),
        aliases=tuple(definition.aliases),
    )


_OBJECT_DEFS: Dict[str, ObjectTypeDef] = {
    key: _with_catalog_defaults(value)
    for key, value in {**_base_object_defs(), **_SCENE_OBJECT_DEFS}.items()
}


def registered_object_types() -> Tuple[str, ...]:
    """Return all globally registered illustration object type ids."""

    return tuple(sorted(_OBJECT_DEFS))


def object_type_definition(object_type: str) -> ObjectTypeDef:
    """Return a registered object type, falling back to a generic support object."""

    key = str(object_type)
    if key in _OBJECT_DEFS:
        return _OBJECT_DEFS[key]
    return ObjectTypeDef(key, key.replace("_", " "), "support", semantic_attributes=_attrs("source_type"))


def public_name_for_object_type(object_type: str) -> str:
    """Return the public display name for an object type."""

    return str(object_type_definition(str(object_type)).public_name)


def family_for_object_type(object_type: str) -> str:
    """Return the public object family for an object type."""

    return str(object_type_definition(str(object_type)).family)


def make_object_record(
    *,
    object_id: str,
    object_type: str,
    bbox_xyxy: Sequence[float] | None,
    semantic_attributes: Mapping[str, Any] | None = None,
    visual_attributes: Mapping[str, Any] | None = None,
    public_name: str | None = None,
    family: str | None = None,
    role: str = "distractor",
    source_entity_type: str = "",
    parts: Sequence[Mapping[str, Any]] = (),
) -> ObjectRecord:
    """Build a normalized object record from scene-specific metadata."""

    definition = object_type_definition(str(object_type))
    bbox: BBox | None = None
    if bbox_xyxy is not None:
        values = tuple(float(value) for value in bbox_xyxy)
        if len(values) != 4:
            raise ValueError("bbox_xyxy must contain exactly four values")
        bbox = values  # type: ignore[assignment]
    return ObjectRecord(
        object_id=str(object_id),
        object_type=str(object_type),
        public_name=str(public_name if public_name is not None else definition.public_name),
        family=str(family if family is not None else definition.family),
        bbox_xyxy=bbox,
        semantic_attributes=dict(semantic_attributes or {}),
        visual_attributes=dict(visual_attributes or {}),
        role=str(role),
        source_entity_type=str(source_entity_type),
        parts=tuple(dict(part) for part in parts),
    )


__all__ = [
    "family_for_object_type",
    "make_object_record",
    "object_type_definition",
    "public_name_for_object_type",
    "registered_object_types",
]
