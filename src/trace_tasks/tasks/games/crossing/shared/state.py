"""Typed state for lane-crossing games tasks."""

from __future__ import annotations

from dataclasses import dataclass

from .defaults import SCENE_ID, SUPPORTED_CROSSING_SCENE_VARIANTS, SUPPORTED_CROSSING_STYLE_VARIANTS


@dataclass(frozen=True)
class CrossingVehicle:
    """One moving object in a horizontal traffic row."""

    vehicle_id: str
    row: int
    start_col: int
    direction: int
    color_index: int
    vehicle_kind: str = "car"
    option_label: str | None = None


@dataclass(frozen=True)
class CrossingRouteOption:
    """One visible runner route, with one lane column per traffic row."""

    route_id: str
    label: str
    path_cols: tuple[int, ...]
    color_index: int


@dataclass(frozen=True)
class CrossingSample:
    """Generated lane-crossing scene and task contract."""

    lane_count: int
    row_count: int
    count_mode: str
    scene_variant: str
    style_variant: str
    answer: int | str
    row_directions: tuple[int, ...]
    vehicles: tuple[CrossingVehicle, ...]
    start_labels: tuple[str, ...]
    route_options: tuple[CrossingRouteOption, ...]
    marked_route_label: str | None
    target_start_label: str | None
    target_route_label: str | None
    target_object_label: str | None
    first_collision_tick: int | None
    intersecting_vehicle_ids: tuple[str, ...]
    annotation_entity_ids: tuple[str, ...]
    target_answer: int | None
    target_label_index: int | None
    construction_mode: str


@dataclass(frozen=True)
class CrossingSceneAxes:
    """Resolved scene-level axes for one lane-crossing instance."""

    scene_variant: str
    style_variant: str
    lane_count: int
    row_count: int
    scene_variant_probabilities: dict[str, float]
    style_variant_probabilities: dict[str, float]
    lane_count_probabilities: dict[str, float]
    row_count_probabilities: dict[str, float]


def vehicle_entity_id(index: int) -> str:
    """Return the stable entity id for one moving object."""

    return f"vehicle_{int(index)}"


def start_entity_id(index: int) -> str:
    """Return the stable entity id for one bottom start pad."""

    return f"start_{int(index)}"


def route_entity_id(label: str) -> str:
    """Return the stable entity id for one visible route option."""

    return f"route_{str(label)}"


def route_cell_entity_id(label: str, row: int) -> str:
    """Return the stable entity id for one route cell at a traffic row."""

    return f"route_{str(label)}_cell_{int(row)}"


__all__ = [
    "CrossingRouteOption",
    "CrossingSample",
    "CrossingSceneAxes",
    "CrossingVehicle",
    "SCENE_ID",
    "SUPPORTED_CROSSING_SCENE_VARIANTS",
    "SUPPORTED_CROSSING_STYLE_VARIANTS",
    "route_cell_entity_id",
    "route_entity_id",
    "start_entity_id",
    "vehicle_entity_id",
]
