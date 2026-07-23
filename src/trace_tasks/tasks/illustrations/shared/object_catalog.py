"""Single source for illustration object, fixture, and background vocabularies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

from .object_library import OBJECT_TEMPLATES, display_name_for_object_type


@dataclass(frozen=True)
class CatalogEntry:
    """One drawable or placeable illustration catalog entry."""

    catalog_id: str
    object_type: str
    public_name: str
    family: str
    render_layer: str
    size_class: str
    placement_tags: Tuple[str, ...] = ()
    scene_tags: Tuple[str, ...] = ()
    renderer_id: str = ""
    variant_id: str = ""
    plural_name: str = ""
    label_text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _entry(
    catalog_id: str,
    *,
    object_type: str | None = None,
    public_name: str | None = None,
    family: str = "object",
    render_layer: str = "foreground",
    size_class: str = "medium",
    placement_tags: Sequence[str] = (),
    scene_tags: Sequence[str] = (),
    renderer_id: str = "",
    variant_id: str | None = None,
    plural_name: str = "",
    label_text: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> CatalogEntry:
    return CatalogEntry(
        catalog_id=str(catalog_id),
        object_type=str(object_type if object_type is not None else catalog_id),
        public_name=str(public_name if public_name is not None else str(catalog_id).replace("_", " ")),
        family=str(family),
        render_layer=str(render_layer),
        size_class=str(size_class),
        placement_tags=tuple(str(tag) for tag in placement_tags),
        scene_tags=tuple(str(tag) for tag in scene_tags),
        renderer_id=str(renderer_id),
        variant_id=str(variant_id if variant_id is not None else catalog_id),
        plural_name=str(plural_name),
        label_text=str(label_text),
        metadata=dict(metadata or {}),
    )


def _with_tags(base: Sequence[str], *extra: str) -> Tuple[str, ...]:
    seen: list[str] = []
    for tag in [*base, *extra]:
        value = str(tag)
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


_MIXED_SKY = ("airplane", "bird", "butterfly", "cloud", "kite", "sun")
_MIXED_WATER = ("boat", "sailboat", "fish", "duck", "buoy", "lily_pad")
_MIXED_ROADLIKE = ("car", "bus", "truck", "taxi", "van", "train", "bicycle", "scooter")
_ENV_ROAD = ("car", "bus", "truck", "taxi", "van", "bicycle", "scooter")
_ENV_RIVER = ("boat", "sailboat", "fish", "duck", "buoy", "lily_pad")
_ENV_SKY = ("airplane", "bird", "butterfly", "kite")
_ENV_LAND = (
    "quadruped",
    "rabbit",
    "person",
    "pedestrian_with_bag",
    "tree",
    "flower",
    "cactus",
    "mushroom",
    "potted_plant",
    "umbrella",
    "backpack",
    "bucket",
    "camera",
    "chair",
    "table",
    "lamp",
    "streetlamp",
    "traffic_light",
    "road_sign",
    "mailbox",
    "bench",
    "soccer_ball",
    "rugby_ball",
    "shovel",
    "trash_bin",
    "mug",
    "bottle",
)
_INDOOR_OBJECTS = (
    "apple",
    "backpack",
    "egg",
    "soccer_ball",
    "rugby_ball",
    "spoon",
    "plate",
    "book",
    "camera",
    "remote",
    "pencil",
    "ruler",
    "clock",
    "gift",
    "vase",
    "bowl",
    "candle",
    "lightbulb",
    "mug",
    "teapot",
    "bottle",
    "guitar",
    "key",
    "scissors",
    "lamp",
    "potted_plant",
    "flower",
    "mushroom",
)
_LARGE_SHARED = {"airplane", "bus", "truck", "van", "train", "boat", "sailboat", "tree", "bench", "table"}
_SMALL_SHARED = {
    "apple",
    "egg",
    "soccer_ball",
    "rugby_ball",
    "spoon",
    "camera",
    "pencil",
    "ruler",
    "key",
    "gift",
    "candle",
    "lightbulb",
    "mug",
    "teapot",
    "bottle",
    "bucket",
    "shovel",
    "flower",
    "lily_pad",
    "buoy",
    "sun",
    "cloud",
}

_ENV_THEME_LAND: Dict[str, Tuple[str, ...]] = {
    "park_road": (
        "quadruped",
        "rabbit",
        "person",
        "pedestrian_with_bag",
        "tree",
        "flower",
        "mushroom",
        "potted_plant",
        "umbrella",
        "backpack",
        "camera",
        "streetlamp",
        "road_sign",
        "bench",
        "soccer_ball",
        "rugby_ball",
        "trash_bin",
        "bottle",
    ),
    "river_meadow": (
        "quadruped",
        "rabbit",
        "person",
        "pedestrian_with_bag",
        "tree",
        "flower",
        "cactus",
        "mushroom",
        "potted_plant",
        "umbrella",
        "backpack",
        "bucket",
        "camera",
        "bench",
        "soccer_ball",
        "rugby_ball",
        "shovel",
        "trash_bin",
        "bottle",
    ),
    "road_and_river": (
        "quadruped",
        "rabbit",
        "person",
        "pedestrian_with_bag",
        "tree",
        "flower",
        "mushroom",
        "potted_plant",
        "umbrella",
        "backpack",
        "bucket",
        "camera",
        "streetlamp",
        "road_sign",
        "bench",
        "soccer_ball",
        "rugby_ball",
        "shovel",
        "trash_bin",
        "bottle",
    ),
    "canal_city": (
        "person",
        "pedestrian_with_bag",
        "tree",
        "flower",
        "potted_plant",
        "umbrella",
        "backpack",
        "camera",
        "streetlamp",
        "traffic_light",
        "road_sign",
        "mailbox",
        "bench",
        "trash_bin",
        "bottle",
    ),
    "skyline_street": (
        "person",
        "pedestrian_with_bag",
        "tree",
        "flower",
        "potted_plant",
        "backpack",
        "camera",
        "streetlamp",
        "traffic_light",
        "road_sign",
        "mailbox",
        "bench",
        "trash_bin",
        "bottle",
    ),
}


def _shared_entries() -> Tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for object_type, template in sorted(OBJECT_TEMPLATES.items()):
        placement_tags: list[str] = ["shared_object"]
        if object_type in _MIXED_SKY:
            placement_tags.extend(["sky", "mixed_sky"])
        if object_type in _MIXED_WATER:
            placement_tags.extend(["water", "mixed_water"])
        if object_type in _MIXED_ROADLIKE:
            placement_tags.extend(["road", "mixed_roadlike"])
        if object_type in _ENV_ROAD:
            placement_tags.extend(["env_road"])
        if object_type in _ENV_RIVER:
            placement_tags.extend(["env_river"])
        if object_type in _ENV_SKY:
            placement_tags.extend(["env_sky"])
        if object_type in _ENV_LAND:
            placement_tags.extend(["land", "env_land"])
        if object_type in _INDOOR_OBJECTS:
            placement_tags.extend(["indoor_object", "surface_placeable", "container_placeable"])
        for theme_id, theme_objects in _ENV_THEME_LAND.items():
            if object_type in theme_objects:
                placement_tags.append(f"env_land:{theme_id}")
        size_class = "large" if object_type in _LARGE_SHARED else "small" if object_type in _SMALL_SHARED else "medium"
        entries.append(
            _entry(
                str(object_type),
                public_name=display_name_for_object_type(str(object_type)),
                family=str(template.family),
                render_layer="foreground",
                size_class=size_class,
                placement_tags=placement_tags,
                renderer_id="object_library",
            )
        )
    return tuple(entries)


_MIXED_BACKGROUNDS = ("studio", "meadow", "sky_ground", "tabletop", "paper", "shelf")
_ENVIRONMENT_THEMES = ("park_road", "river_meadow", "road_and_river", "canal_city", "skyline_street")
_INDOOR_THEMES = ("living_room", "kitchen", "study", "bedroom")
_LIBRARY_SETTINGS = ("reading_room", "archive_room", "childrens_corner")
_PARK_SETTINGS = ("playground_lawn", "picnic_park", "pond_playground", "flower_garden")
_CONSTRUCTION_SETTINGS = ("urban_build", "roadwork", "foundation_yard", "scaffold_site")


def _background_entries() -> Tuple[CatalogEntry, ...]:
    specs = (
        ("mixed_background", "mixed_object", _MIXED_BACKGROUNDS),
        ("environment_theme", "environment", _ENVIRONMENT_THEMES),
        ("indoor_theme", "indoor_room", _INDOOR_THEMES),
        ("library_setting", "library", _LIBRARY_SETTINGS),
        ("park_setting", "park_playground", _PARK_SETTINGS),
        ("construction_setting", "construction_site", _CONSTRUCTION_SETTINGS),
    )
    entries: list[CatalogEntry] = []
    for tag, scene_id, values in specs:
        for value in values:
            entries.append(
                _entry(
                    f"background.{scene_id}.{value}",
                    object_type="background",
                    public_name=str(value).replace("_", " "),
                    family="background",
                    render_layer="background",
                    size_class="scene_scale",
                    placement_tags=(tag,),
                    scene_tags=(scene_id,),
                    renderer_id=f"{scene_id}_background",
                    variant_id=str(value),
                )
            )
    return tuple(entries)


def _environment_feature_entries() -> Tuple[CatalogEntry, ...]:
    specs = (
        ("road", "environment_feature", "road", "scene_region", "region", "scene_scale"),
        ("river", "environment_feature", "river", "scene_region", "region", "scene_scale"),
        ("bridge", "environment_feature", "bridge", "fixture", "fixture", "large"),
        ("crosswalk", "environment_feature", "crosswalk", "fixture", "fixture", "large"),
        ("building", "building", "building", "structure", "background", "scene_scale"),
    )
    return tuple(
        _entry(
            f"environment_feature.{feature_type}",
            object_type=object_type,
            public_name=public_name,
            family=family,
            render_layer=render_layer,
            size_class=size_class,
            placement_tags=("environment_feature", f"environment_feature:{feature_type}"),
            scene_tags=("environment",),
            renderer_id="environment_feature",
            variant_id=feature_type,
        )
        for feature_type, object_type, public_name, family, render_layer, size_class in specs
    )


_LIBRARY_SECTIONS: Dict[str, Tuple[str, str]] = {
    "history": ("HISTORY", "History"),
    "science": ("SCIENCE", "Science"),
    "art": ("ART", "Art"),
    "travel": ("TRAVEL", "Travel"),
    "fiction": ("FICTION", "Fiction"),
    "nature": ("NATURE", "Nature"),
    "music": ("MUSIC", "Music"),
    "math": ("MATH", "Math"),
}


def _library_entries() -> Tuple[CatalogEntry, ...]:
    entries = [
        _entry(
            f"library_section.{section_key}",
            object_type="library_section",
            public_name=display,
            family="scene_region",
            render_layer="fixture",
            size_class="large",
            placement_tags=("library_section",),
            scene_tags=("library",),
            renderer_id="library_section",
            variant_id=section_key,
            label_text=label,
        )
        for section_key, (label, display) in _LIBRARY_SECTIONS.items()
    ]
    for orientation in ("upright", "horizontal"):
        entries.append(
            _entry(
                f"book_orientation.{orientation}",
                object_type="book",
                public_name=f"{orientation} book",
                family="object",
                render_layer="foreground",
                size_class="small",
                placement_tags=("library_book_orientation",),
                scene_tags=("library",),
                renderer_id="library_book",
                variant_id=orientation,
            )
        )
    return tuple(entries)


_PARK_EQUIPMENT: Dict[str, str] = {
    "slide": "slides",
    "swing_set": "swing sets",
    "seesaw": "seesaws",
    "climbing_frame": "climbing frames",
}
_PARK_ZONES: Dict[str, str] = {
    "playground": "playground area",
    "picnic": "picnic area",
    "garden": "garden area",
}
_PARK_PERSON_ACTIVITIES: Dict[str, str] = {
    "sitting": "sitting",
    "walking": "walking",
    "standing": "standing",
    "playing_ball": "playing with a ball",
}


def _park_entries() -> Tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for equipment_type, plural in _PARK_EQUIPMENT.items():
        entries.append(
            _entry(
                f"park_equipment.{equipment_type}",
                object_type="playground_equipment",
                public_name=plural[:-1] if plural.endswith("s") else plural,
                family="equipment",
                render_layer="fixture",
                size_class="large",
                placement_tags=("park_equipment",),
                scene_tags=("park_playground",),
                renderer_id="park_equipment",
                variant_id=equipment_type,
                plural_name=plural,
            )
        )
    for zone_type, label in _PARK_ZONES.items():
        entries.append(
            _entry(
                f"park_zone.{zone_type}",
                object_type="zone",
                public_name=label,
                family="scene_region",
                render_layer="region",
                size_class="scene_scale",
                placement_tags=("park_zone",),
                scene_tags=("park_playground",),
                renderer_id="park_zone",
                variant_id=zone_type,
                label_text=label,
            )
        )
    for activity, label in _PARK_PERSON_ACTIVITIES.items():
        entries.append(
            _entry(
                f"park_person_activity.{activity}",
                object_type="person",
                public_name=label,
                family="person",
                render_layer="foreground",
                size_class="medium",
                placement_tags=("park_person_activity",),
                scene_tags=("park_playground",),
                renderer_id="park_person",
                variant_id=activity,
                label_text=label,
            )
        )
    return tuple(entries)


_CONSTRUCTION_ZONES: Dict[str, str] = {
    "excavation_zone": "Excavation Zone",
    "loading_zone": "Loading Zone",
    "roadwork_zone": "Roadwork Zone",
}
_CONSTRUCTION_MATERIALS: Dict[str, str] = {
    "brick_stack": "brick stacks",
    "pipe_bundle": "pipe bundles",
    "lumber_stack": "lumber stacks",
    "cement_bag_stack": "cement bag stacks",
}
_CONSTRUCTION_EQUIPMENT: Dict[str, str] = {
    "excavator": "excavators",
    "dump_truck": "dump trucks",
    "cement_mixer": "cement mixers",
    "forklift": "forklifts",
}
_CONSTRUCTION_TOOLS = ("hammer", "shovel", "wrench")


def _construction_entries() -> Tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    entries.append(
        _entry(
            "construction_worker.worker",
            object_type="worker",
            public_name="worker",
            family="person",
            render_layer="foreground",
            size_class="medium",
            placement_tags=("construction_worker",),
            scene_tags=("construction_site",),
            renderer_id="construction_worker",
            variant_id="worker",
        )
    )
    for zone_id, label in _CONSTRUCTION_ZONES.items():
        entries.append(
            _entry(
                f"construction_zone.{zone_id}",
                object_type="zone",
                public_name=label,
                family="scene_region",
                render_layer="region",
                size_class="scene_scale",
                placement_tags=("construction_zone",),
                scene_tags=("construction_site",),
                renderer_id="construction_zone",
                variant_id=zone_id,
                label_text=label,
            )
        )
    for material_type, plural in _CONSTRUCTION_MATERIALS.items():
        entries.append(
            _entry(
                f"construction_material.{material_type}",
                object_type="construction_material",
                public_name=plural[:-1] if plural.endswith("s") else plural,
                family="material",
                render_layer="foreground",
                size_class="medium",
                placement_tags=("construction_material",),
                scene_tags=("construction_site",),
                renderer_id="construction_material",
                variant_id=material_type,
                plural_name=plural,
            )
        )
    for equipment_type, plural in _CONSTRUCTION_EQUIPMENT.items():
        entries.append(
            _entry(
                f"construction_equipment.{equipment_type}",
                object_type="construction_equipment",
                public_name=plural[:-1] if plural.endswith("s") else plural,
                family="equipment",
                render_layer="foreground",
                size_class="large",
                placement_tags=("construction_equipment",),
                scene_tags=("construction_site",),
                renderer_id="construction_equipment",
                variant_id=equipment_type,
                plural_name=plural,
            )
        )
    for tool_type in _CONSTRUCTION_TOOLS:
        entries.append(
            _entry(
                f"construction_tool.{tool_type}",
                object_type="tool",
                public_name=tool_type.replace("_", " "),
                family="object",
                render_layer="foreground",
                size_class="tiny",
                placement_tags=("construction_tool",),
                scene_tags=("construction_site",),
                renderer_id="construction_worker_tool",
                variant_id=tool_type,
            )
        )
    return tuple(entries)


_INDOOR_SURFACES = ("table", "shelf", "counter")
_INDOOR_CONTAINERS = ("basket", "box", "drawer")
_INDOOR_FURNITURE = ("table", "sofa", "cabinet")


def _indoor_fixture_entries() -> Tuple[CatalogEntry, ...]:
    entries: list[CatalogEntry] = []
    for surface_type in _INDOOR_SURFACES:
        entries.append(
            _entry(
                f"indoor_surface.{surface_type}",
                object_type="surface",
                public_name=surface_type.replace("_", " "),
                family="fixture",
                render_layer="fixture",
                size_class="large",
                placement_tags=("indoor_surface",),
                scene_tags=("indoor_room",),
                renderer_id="indoor_surface",
                variant_id=surface_type,
            )
        )
    for container_type in _INDOOR_CONTAINERS:
        entries.append(
            _entry(
                f"indoor_container.{container_type}",
                object_type="container",
                public_name=container_type.replace("_", " "),
                family="fixture",
                render_layer="fixture",
                size_class="medium",
                placement_tags=("indoor_container",),
                scene_tags=("indoor_room",),
                renderer_id="indoor_container",
                variant_id=container_type,
            )
        )
    for furniture_type in _INDOOR_FURNITURE:
        entries.append(
            _entry(
                f"indoor_furniture.{furniture_type}",
                object_type="furniture",
                public_name=furniture_type.replace("_", " "),
                family="fixture",
                render_layer="fixture",
                size_class="large",
                placement_tags=("indoor_furniture",),
                scene_tags=("indoor_room",),
                renderer_id="indoor_furniture",
                variant_id=furniture_type,
            )
        )
    return tuple(entries)


CATALOG_ENTRIES: Tuple[CatalogEntry, ...] = (
    *_shared_entries(),
    *_background_entries(),
    *_environment_feature_entries(),
    *_library_entries(),
    *_park_entries(),
    *_construction_entries(),
    *_indoor_fixture_entries(),
)

_CATALOG_BY_ID: Dict[str, CatalogEntry] = {entry.catalog_id: entry for entry in CATALOG_ENTRIES}
_TAG_VARIANT_ORDER: Dict[str, Tuple[str, ...]] = {
    "mixed_background": _MIXED_BACKGROUNDS,
    "mixed_sky": _MIXED_SKY,
    "mixed_water": _MIXED_WATER,
    "mixed_roadlike": _MIXED_ROADLIKE,
    "environment_theme": _ENVIRONMENT_THEMES,
    "env_road": _ENV_ROAD,
    "env_river": _ENV_RIVER,
    "env_sky": _ENV_SKY,
    "env_land": _ENV_LAND,
    "environment_feature": ("road", "river", "bridge", "crosswalk", "building"),
    "indoor_theme": _INDOOR_THEMES,
    "indoor_object": _INDOOR_OBJECTS,
    "indoor_surface": _INDOOR_SURFACES,
    "indoor_container": _INDOOR_CONTAINERS,
    "indoor_furniture": _INDOOR_FURNITURE,
    "library_setting": _LIBRARY_SETTINGS,
    "library_section": tuple(_LIBRARY_SECTIONS),
    "library_book_orientation": ("upright", "horizontal"),
    "park_setting": _PARK_SETTINGS,
    "park_equipment": tuple(_PARK_EQUIPMENT),
    "park_zone": tuple(_PARK_ZONES),
    "park_person_activity": tuple(_PARK_PERSON_ACTIVITIES),
    "construction_setting": _CONSTRUCTION_SETTINGS,
    "construction_worker": ("worker",),
    "construction_zone": tuple(_CONSTRUCTION_ZONES),
    "construction_material": tuple(_CONSTRUCTION_MATERIALS),
    "construction_equipment": tuple(_CONSTRUCTION_EQUIPMENT),
    "construction_tool": _CONSTRUCTION_TOOLS,
}


def catalog_entry(catalog_id: str) -> CatalogEntry:
    """Return one catalog entry by id."""

    return _CATALOG_BY_ID[str(catalog_id)]


def catalog_entries() -> Tuple[CatalogEntry, ...]:
    """Return all catalog entries."""

    return CATALOG_ENTRIES


def entries_with_tag(tag: str) -> Tuple[CatalogEntry, ...]:
    """Return entries whose placement tags include ``tag``."""

    key = str(tag)
    return tuple(entry for entry in CATALOG_ENTRIES if key in set(entry.placement_tags))


def entries_for_scene(scene_id: str) -> Tuple[CatalogEntry, ...]:
    """Return entries explicitly available to one illustration scene."""

    key = str(scene_id)
    return tuple(entry for entry in CATALOG_ENTRIES if key in set(entry.scene_tags))


def entries_with_render_layer(render_layer: str) -> Tuple[CatalogEntry, ...]:
    """Return entries in one render layer bucket."""

    key = str(render_layer)
    return tuple(entry for entry in CATALOG_ENTRIES if str(entry.render_layer) == key)


def entries_with_size_class(size_class: str) -> Tuple[CatalogEntry, ...]:
    """Return entries in one size-class bucket."""

    key = str(size_class)
    return tuple(entry for entry in CATALOG_ENTRIES if str(entry.size_class) == key)


def variant_ids_with_tag(tag: str) -> Tuple[str, ...]:
    """Return renderer-facing variant ids for entries matching ``tag``."""

    variants = tuple(entry.variant_id for entry in entries_with_tag(str(tag)))
    preferred = _TAG_VARIANT_ORDER.get(str(tag))
    if not preferred:
        return variants
    variant_set = set(variants)
    ordered = tuple(value for value in preferred if value in variant_set)
    extras = tuple(value for value in variants if value not in set(ordered))
    return (*ordered, *extras)


def object_types_with_tag(tag: str) -> Tuple[str, ...]:
    """Return normalized object type ids for entries matching ``tag``."""

    seen: list[str] = []
    for entry in entries_with_tag(str(tag)):
        if entry.object_type not in seen:
            seen.append(entry.object_type)
    return tuple(seen)


def label_map_for_tag(tag: str) -> Dict[str, str]:
    """Return label text by variant id for entries matching ``tag``."""

    return {entry.variant_id: str(entry.label_text or entry.public_name) for entry in entries_with_tag(str(tag))}


def public_name_map_for_tag(tag: str) -> Dict[str, str]:
    """Return public names by variant id for entries matching ``tag``."""

    return {entry.variant_id: str(entry.public_name) for entry in entries_with_tag(str(tag))}


def plural_name_map_for_tag(tag: str) -> Dict[str, str]:
    """Return plural names by variant id for entries matching ``tag``."""

    return {entry.variant_id: str(entry.plural_name or entry.public_name) for entry in entries_with_tag(str(tag))}


def environment_theme_land_object_types(theme_id: str) -> Tuple[str, ...]:
    """Return land object types suitable for one environment theme."""

    return tuple(_ENV_THEME_LAND.get(str(theme_id), _ENV_LAND))


def canonical_entry_for_object_type(object_type: str) -> CatalogEntry | None:
    """Return the first catalog entry for one normalized object type."""

    key = str(object_type)
    for entry in CATALOG_ENTRIES:
        if str(entry.object_type) == key:
            return entry
    return None


__all__ = [
    "CATALOG_ENTRIES",
    "CatalogEntry",
    "canonical_entry_for_object_type",
    "catalog_entries",
    "catalog_entry",
    "entries_for_scene",
    "entries_with_tag",
    "entries_with_render_layer",
    "entries_with_size_class",
    "environment_theme_land_object_types",
    "label_map_for_tag",
    "object_types_with_tag",
    "plural_name_map_for_tag",
    "public_name_map_for_tag",
    "variant_ids_with_tag",
]
