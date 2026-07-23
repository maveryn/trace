"""Tests for the centralized illustration object catalog."""

from __future__ import annotations

from trace_tasks.tasks.illustrations.construction_site.shared import labels as construction_labels
from trace_tasks.tasks.illustrations.construction_site.shared import state as construction_state
from trace_tasks.tasks.illustrations.environment.shared import rendering as environment
from trace_tasks.tasks.illustrations.library.shared import state as library
from trace_tasks.tasks.illustrations.shared import mixed_object_rendering as mixed
from trace_tasks.tasks.illustrations.park_playground.shared import state as park
from trace_tasks.tasks.illustrations.indoor_room.shared import rendering as indoor
from trace_tasks.tasks.illustrations.shared.object_catalog import (
    catalog_entries,
    entries_with_render_layer,
    entries_with_size_class,
    environment_theme_land_object_types,
    label_map_for_tag,
    plural_name_map_for_tag,
    public_name_map_for_tag,
    variant_ids_with_tag,
)
from trace_tasks.tasks.illustrations.shared.object_registry import object_type_definition


def test_catalog_has_layer_size_and_scene_coverage() -> None:
    entries = catalog_entries()
    assert entries
    assert any(entry.render_layer == "background" and entry.size_class == "scene_scale" for entry in entries)
    assert any(entry.render_layer == "fixture" and entry.size_class == "large" for entry in entries)
    assert any("surface_placeable" in entry.placement_tags for entry in entries)
    assert any(entry.catalog_id == "construction_worker.worker" and entry.object_type == "worker" for entry in entries)
    assert any(entry.catalog_id == "environment_feature.road" for entry in entries)
    assert {entry.render_layer for entry in entries_with_size_class("scene_scale")} >= {"background", "region"}
    assert {entry.size_class for entry in entries_with_render_layer("fixture")} >= {"medium", "large"}


def test_renderer_constants_are_catalog_derived() -> None:
    assert mixed.BACKGROUND_IDS == variant_ids_with_tag("mixed_background")
    assert mixed.SKY_OBJECT_TYPES == variant_ids_with_tag("mixed_sky")
    assert environment.ENVIRONMENT_THEME_IDS == variant_ids_with_tag("environment_theme")
    assert environment.ROAD_OBJECT_TYPES == variant_ids_with_tag("env_road")
    assert environment.RIVER_OBJECT_TYPES == variant_ids_with_tag("env_river")
    assert environment.SKY_OBJECT_TYPES == variant_ids_with_tag("env_sky")
    assert environment.LAND_OBJECT_TYPES == variant_ids_with_tag("env_land")
    assert environment.THEME_LAND_OBJECT_TYPES == {
        theme_id: environment_theme_land_object_types(theme_id) for theme_id in environment.ENVIRONMENT_THEME_IDS
    }
    assert indoor.INDOOR_OBJECT_TYPES == variant_ids_with_tag("indoor_object")
    assert indoor.INDOOR_SURFACE_TYPES == variant_ids_with_tag("indoor_surface")
    assert indoor.INDOOR_CONTAINER_TYPES == variant_ids_with_tag("indoor_container")
    assert indoor.INDOOR_FURNITURE_TYPES == variant_ids_with_tag("indoor_furniture")


def test_scene_variant_maps_are_catalog_derived() -> None:
    assert library.LIBRARY_SETTING_IDS == variant_ids_with_tag("library_setting")
    assert library.LIBRARY_SECTION_TYPES == variant_ids_with_tag("library_section")
    assert library.LIBRARY_SECTION_LABELS == label_map_for_tag("library_section")
    assert library.LIBRARY_SECTION_DISPLAY_NAMES == public_name_map_for_tag("library_section")
    assert park.PARK_EQUIPMENT_TYPES == variant_ids_with_tag("park_equipment")
    assert park.PARK_EQUIPMENT_LABELS == plural_name_map_for_tag("park_equipment")
    assert park.PARK_ZONE_LABELS == label_map_for_tag("park_zone")
    assert construction_state.CONSTRUCTION_MATERIAL_TYPES == variant_ids_with_tag("construction_material")
    assert construction_labels.CONSTRUCTION_MATERIAL_LABELS == plural_name_map_for_tag("construction_material")
    assert construction_labels.CONSTRUCTION_EQUIPMENT_LABELS == plural_name_map_for_tag("construction_equipment")


def test_registry_exposes_catalog_categories_for_object_types() -> None:
    assert object_type_definition("worker").render_layer == "foreground"
    assert object_type_definition("worker").size_class == "medium"
    assert "construction_worker" in object_type_definition("worker").placement_tags
    assert object_type_definition("building").render_layer == "background"
