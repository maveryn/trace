"""Prompt-facing labels and colors for the construction-site scene."""

from __future__ import annotations

from typing import Dict

from ...shared.object_catalog import label_map_for_tag, plural_name_map_for_tag
from ...shared.object_library import RGB


CONSTRUCTION_ZONE_LABELS: Dict[str, str] = label_map_for_tag("construction_zone")
CONSTRUCTION_MATERIAL_LABELS: Dict[str, str] = plural_name_map_for_tag("construction_material")
CONSTRUCTION_EQUIPMENT_LABELS: Dict[str, str] = plural_name_map_for_tag("construction_equipment")
CONSTRUCTION_COLOR_RGB: Dict[str, RGB] = {
    "yellow": (238, 194, 64),
    "orange": (232, 126, 54),
    "red": (198, 73, 62),
    "blue": (63, 118, 183),
    "green": (75, 146, 92),
    "purple": (129, 92, 165),
}


def construction_color_hex(color_name: str) -> str:
    """Return the canonical hex string for a construction color."""

    rgb = CONSTRUCTION_COLOR_RGB.get(str(color_name))
    if rgb is None:
        return ""
    return f"#{int(rgb[0]):02X}{int(rgb[1]):02X}{int(rgb[2]):02X}"


def construction_color_display_name(color_name: str) -> str:
    """Return a prompt-facing color name with hex disambiguation."""

    name = str(color_name).replace("_", " ")
    hex_value = construction_color_hex(str(color_name))
    return f"{name} [{hex_value}]" if hex_value else name


def construction_material_display_name(material_type: str) -> str:
    """Return prompt-facing plural material text."""

    return CONSTRUCTION_MATERIAL_LABELS.get(str(material_type), str(material_type).replace("_", " ") + "s")


def construction_equipment_display_name(equipment_type: str) -> str:
    """Return prompt-facing plural equipment text."""

    return CONSTRUCTION_EQUIPMENT_LABELS.get(str(equipment_type), str(equipment_type).replace("_", " ") + "s")


def construction_zone_display_name(zone_id: str) -> str:
    """Return prompt-facing construction-zone text."""

    return CONSTRUCTION_ZONE_LABELS.get(str(zone_id), str(zone_id).replace("_", " "))


__all__ = [
    "CONSTRUCTION_COLOR_RGB",
    "CONSTRUCTION_EQUIPMENT_LABELS",
    "CONSTRUCTION_MATERIAL_LABELS",
    "CONSTRUCTION_ZONE_LABELS",
    "construction_color_display_name",
    "construction_color_hex",
    "construction_equipment_display_name",
    "construction_material_display_name",
    "construction_zone_display_name",
]
