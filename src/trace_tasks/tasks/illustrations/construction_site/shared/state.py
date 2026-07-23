"""State contracts for the construction-site illustration scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from PIL import Image

from ...shared.object_catalog import variant_ids_with_tag
from ...shared.object_library import BBox, RGB


CONSTRUCTION_SETTING_IDS: Tuple[str, ...] = variant_ids_with_tag("construction_setting")
CONSTRUCTION_ZONE_TYPES: Tuple[str, ...] = variant_ids_with_tag("construction_zone")
CONSTRUCTION_MATERIAL_TYPES: Tuple[str, ...] = variant_ids_with_tag("construction_material")
CONSTRUCTION_EQUIPMENT_TYPES: Tuple[str, ...] = variant_ids_with_tag("construction_equipment")
CONSTRUCTION_TOOL_TYPES: Tuple[str, ...] = variant_ids_with_tag("construction_tool")
CONSTRUCTION_COLOR_NAMES: Tuple[str, ...] = ("yellow", "orange", "red", "blue", "green", "purple")


@dataclass(frozen=True)
class ConstructionWorkerSpec:
    """Requested semantic attributes for one rendered worker."""

    hard_hat_color: str
    vest_color: str
    tool_type: str | None = None
    role: str = "distractor"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructionMaterialSpec:
    """Requested semantic material type for one rendered stack/bundle."""

    material_type: str
    role: str = "distractor"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructionEquipmentSpec:
    """Requested semantic equipment type and optional zone placement."""

    equipment_type: str
    zone_id: str | None = None
    role: str = "distractor"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstructionZone:
    """One labeled construction-site region."""

    zone_id: str
    label: str
    bbox_xyxy: BBox
    fill_rgb: RGB
    outline_rgb: RGB


@dataclass(frozen=True)
class ConstructionWorker:
    """One rendered construction worker."""

    worker_id: str
    hard_hat_color: str
    vest_color: str
    tool_type: str | None
    bbox_xyxy: BBox
    style_id: str
    gender_id: str
    role: str
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ConstructionMaterial:
    """One rendered material stack or bundle."""

    material_id: str
    material_type: str
    material_label: str
    bbox_xyxy: BBox
    style_id: str
    role: str
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ConstructionEquipment:
    """One rendered construction vehicle."""

    equipment_id: str
    equipment_type: str
    equipment_label: str
    zone_id: str
    bbox_xyxy: BBox
    style_id: str
    role: str
    attributes: Mapping[str, Any]
    object_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ConstructionDecor:
    """Non-query construction-site visual element."""

    decor_id: str
    decor_type: str
    bbox_xyxy: BBox
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedConstructionSiteScene:
    """Rendered construction site plus verifier-ready metadata."""

    image: Image.Image
    setting_id: str
    zones: Tuple[ConstructionZone, ...]
    workers: Tuple[ConstructionWorker, ...]
    materials: Tuple[ConstructionMaterial, ...]
    equipment: Tuple[ConstructionEquipment, ...]
    decor: Tuple[ConstructionDecor, ...]
    canvas_width: int
    canvas_height: int
    render_scale: int
    style_id: str
    layout: Mapping[str, Any]


def safe_json_value(value: Any) -> Any:
    """Return a JSON-safe representation while preserving deterministic rounding."""

    if isinstance(value, Mapping):
        return {str(key): safe_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    if isinstance(value, float):
        return round(float(value), 3)
    return value


__all__ = [
    "CONSTRUCTION_COLOR_NAMES",
    "CONSTRUCTION_EQUIPMENT_TYPES",
    "CONSTRUCTION_MATERIAL_TYPES",
    "CONSTRUCTION_SETTING_IDS",
    "CONSTRUCTION_TOOL_TYPES",
    "CONSTRUCTION_ZONE_TYPES",
    "ConstructionDecor",
    "ConstructionEquipment",
    "ConstructionEquipmentSpec",
    "ConstructionMaterial",
    "ConstructionMaterialSpec",
    "ConstructionWorker",
    "ConstructionWorkerSpec",
    "ConstructionZone",
    "RenderedConstructionSiteScene",
    "safe_json_value",
]
