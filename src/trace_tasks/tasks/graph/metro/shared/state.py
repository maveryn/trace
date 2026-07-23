"""State records and fixed route templates for the metro graph scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image

from ...shared.label_assets import SUPPORTED_GRAPH_LABEL_VARIANTS

SCENE_ID = "metro"
GridPoint = Tuple[int, int]
LabelEdge = Tuple[str, str]
SUPPORTED_METRO_LABEL_VARIANTS: Tuple[str, ...] = SUPPORTED_GRAPH_LABEL_VARIANTS


@dataclass(frozen=True)
class MetroRouteTemplate:
    """One fixed schematic route template before label assignment."""

    route_id: str
    route_name: str
    color_rgb: Tuple[int, int, int]
    grid_points: Tuple[GridPoint, ...]


@dataclass(frozen=True)
class MetroRouteNetworkSample:
    """Trace-ready sampled metro-route network."""

    station_labels: Tuple[str, ...]
    label_by_coord: Dict[GridPoint, str]
    coord_by_label: Dict[str, GridPoint]
    route_templates: Tuple[MetroRouteTemplate, ...]
    route_station_labels: Dict[str, Tuple[str, ...]]
    station_route_ids_by_label: Dict[str, Tuple[str, ...]]
    adjacency_by_label: Dict[str, Tuple[str, ...]]
    edge_labels: Tuple[LabelEdge, ...]
    transfer_labels: Tuple[str, ...]
    target_transfer_count: int
    route_count: int
    station_count: int
    label_variant: str
    terminal_labels: Tuple[str, ...] = ()
    query_label: str = ""
    source_label: str = ""
    goal_label: str = ""
    query_route_ids: Tuple[str, ...] = ()
    target_labels: Tuple[str, ...] = ()
    target_terminal_count: int = 0
    target_single_route_count: int = 0
    target_exact_distance_count: int = 0
    target_shortest_path_length: int = 0
    label_source_kind: str = ""
    label_bucket: str = ""
    label_manifest: str = ""
    label_filter: Mapping[str, Any] | None = None
    label_bucket_probabilities: Mapping[str, float] | None = None


@dataclass(frozen=True)
class RenderedMetroStation:
    """One rendered metro station."""

    label: str
    grid_point: GridPoint
    route_ids: Tuple[str, ...]
    is_transfer: bool
    center_xy: Tuple[int, int]
    bbox_xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class RenderedMetroRoute:
    """One rendered metro route polyline."""

    route_id: str
    route_name: str
    color_rgb: Tuple[int, int, int]
    station_labels: Tuple[str, ...]
    polyline_px: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class RenderedMetroRouteScene:
    """Full metro-route render output."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    stations: Tuple[RenderedMetroStation, ...]
    routes: Tuple[RenderedMetroRoute, ...]
    resolved_label_font_size_px: int
    route_line_width_px: int
    station_radius_px: int
    transfer_station_radius_px: int


METRO_ROUTE_TEMPLATES: Tuple[MetroRouteTemplate, ...] = (
    MetroRouteTemplate("R", "Red", (217, 72, 69), ((1, 3), (3, 3), (5, 3), (7, 3), (9, 3))),
    MetroRouteTemplate("B", "Blue", (56, 111, 197), ((5, 1), (5, 3), (5, 5), (5, 7), (5, 9))),
    MetroRouteTemplate("G", "Green", (56, 151, 95), ((1, 7), (3, 7), (5, 7), (7, 7), (9, 7))),
    MetroRouteTemplate("O", "Orange", (225, 137, 55), ((3, 1), (3, 3), (3, 5), (3, 7), (3, 9))),
    MetroRouteTemplate("P", "Purple", (134, 91, 193), ((1, 1), (3, 3), (5, 5), (7, 7), (9, 9))),
    MetroRouteTemplate("T", "Teal", (32, 151, 156), ((1, 9), (3, 7), (5, 5), (7, 3), (9, 1))),
)
METRO_ROUTE_TEMPLATE_BY_ID: Dict[str, MetroRouteTemplate] = {str(route.route_id): route for route in METRO_ROUTE_TEMPLATES}

__all__ = [
    "GridPoint", "LabelEdge", "METRO_ROUTE_TEMPLATE_BY_ID", "METRO_ROUTE_TEMPLATES", "MetroRouteNetworkSample",
    "MetroRouteTemplate", "RenderedMetroRoute", "RenderedMetroRouteScene",
    "RenderedMetroStation", "SCENE_ID", "SUPPORTED_METRO_LABEL_VARIANTS",
]
