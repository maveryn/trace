"""State records and labels for the park/playground scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from .rendering import (
    PARK_EQUIPMENT_LABELS,
    PARK_EQUIPMENT_TYPES,
    PARK_PERSON_ACTIVITIES,
    PARK_PERSON_ACTIVITY_LABELS,
    PARK_SETTING_IDS,
    PARK_ZONE_LABELS,
    PARK_ZONE_TYPES,
    ParkEquipmentSpec,
    ParkPersonSpec,
    RenderedParkPlaygroundScene,
    park_activity_display_name,
    park_equipment_display_name,
    park_zone_display_name,
)


@dataclass(frozen=True)
class PersonCountSampleSpec:
    branch_id: str
    person_count: int
    person_specs: Tuple[ParkPersonSpec, ...]
    query_probabilities: Dict[str, float]
    person_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class EquipmentSampleSpec:
    branch_id: str
    target_equipment_type: str
    equipment_name: str
    target_count: int
    equipment_count: int
    person_count: int
    equipment_specs: Tuple[ParkEquipmentSpec, ...]
    person_specs: Tuple[ParkPersonSpec, ...]
    query_probabilities: Dict[str, float]
    target_equipment_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    equipment_count_probabilities: Dict[str, float]
    person_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ParkCountBinding:
    """Bound answer, annotation, prompt slots, and trace fragments."""

    prompt_defaults: Mapping[str, Any]
    slots: Mapping[str, Any]
    answer: int
    annotation_value: Tuple[Tuple[float, ...], ...] | Tuple[list[float], ...] | list[list[float]]
    render_map: Mapping[str, Any]
    scene_relations: Mapping[str, Any]
    branch_params: Mapping[str, Any]
    execution_trace: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    scene_entities: Tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]


__all__ = [
    "EquipmentSampleSpec",
    "PARK_EQUIPMENT_LABELS",
    "PARK_EQUIPMENT_TYPES",
    "PARK_PERSON_ACTIVITIES",
    "PARK_PERSON_ACTIVITY_LABELS",
    "PARK_SETTING_IDS",
    "PARK_ZONE_LABELS",
    "PARK_ZONE_TYPES",
    "ParkCountBinding",
    "ParkEquipmentSpec",
    "ParkPersonSpec",
    "PersonCountSampleSpec",
    "RenderedParkPlaygroundScene",
    "park_activity_display_name",
    "park_equipment_display_name",
    "park_zone_display_name",
]
