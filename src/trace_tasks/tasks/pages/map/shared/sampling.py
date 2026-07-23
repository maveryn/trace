"""Sampling primitives for printed map scene packages."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index

from .defaults import DEFAULTS, GENERATION_DEFAULTS, NAMESPACE_ROOT, SCENE_VARIANTS
from .routes import build_adjacency, neighbors
from .state import Cell, MapSceneCase


GRID_COLS = 5
GRID_ROWS = 4

LANDMARK_NAMES: Tuple[str, ...] = (
    "Library",
    "Atrium",
    "Clinic",
    "Studio",
    "Workshop",
    "Gallery",
    "Foundry",
    "Theater",
    "Cafeteria",
    "Depot",
    "Dormitory",
    "Garden",
    "Plaza",
    "Archive",
    "Gym",
    "Observatory",
    "Bookstore",
    "Lab Annex",
    "North Hall",
    "South Hall",
    "East Lab",
    "West Lab",
    "Market",
    "Chapel",
    "Arcade",
    "Tower",
    "Pool",
    "Annex",
    "Dining",
    "Media Lab",
    "Makerspace",
    "Maple Lab",
    "Cedar Lab",
    "Elm Hall",
    "Pine Hall",
    "Delta Lab",
    "Sigma Hall",
    "Harbor Lab",
    "Valley Hall",
    "Union",
    "Gateway",
    "Courtyard",
    "Auditorium",
    "Commons",
    "Greenhouse",
    "Data Lab",
    "Print Shop",
    "Music Hall",
    "Health Hub",
    "Design Lab",
    "Field House",
    "Science Hall",
    "Transit Hub",
)
ZONE_LABELS: Tuple[str, ...] = (
    "North Green",
    "Research Row",
    "West Yard",
    "South Court",
    "East Quad",
    "South Lawn",
    "North Quad",
    "Maker Row",
    "Study Grove",
    "Transit Yard",
    "Arts Court",
    "Science Park",
    "Harbor Wing",
    "Valley Court",
    "Market Lane",
    "Civic Yard",
    "Garden Row",
    "Central Field",
    "Elm Court",
    "Cedar Yard",
    "Maple Quad",
    "River Walk",
    "Depot Row",
    "Clinic Zone",
)
TITLE_OPTIONS: Tuple[str, ...] = (
    "Campus Orientation Map",
    "Facility Walking Map",
    "Site Navigation Map",
    "Visitor Route Map",
    "Campus Guide Map",
)
ZONE_LAYOUTS: Tuple[Dict[str, object], ...] = (
    {"zone_id": "zone_north_west", "cell_bounds": (0, 0, 2, 1)},
    {"zone_id": "zone_north_east", "cell_bounds": (3, 0, 4, 1)},
    {"zone_id": "zone_south_west", "cell_bounds": (0, 2, 2, 3)},
    {"zone_id": "zone_south_east", "cell_bounds": (3, 2, 4, 3)},
)


def resolve_scene_variant(params: Mapping[str, Any]) -> tuple[str, Dict[str, float]]:
    """Resolve the fixed map scene variant with explicit override validation."""

    explicit = params.get("scene_variant")
    if explicit is not None and str(explicit) not in set(SCENE_VARIANTS):
        raise ValueError(f"unsupported map scene variant: {explicit}")
    return SCENE_VARIANTS[0], {SCENE_VARIANTS[0]: 1.0}


def resolve_direction_step_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve the bounded route length for direction-following prompts."""

    return resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="direction_step_count_min",
        max_key="direction_step_count_max",
        fallback_min=int(DEFAULTS.direction_step_count_min),
        fallback_max=int(DEFAULTS.direction_step_count_max),
        context="map direction step count",
    )


def resolve_highlighted_route_step_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve the requested step range for highlighted-route prompts."""

    return resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="highlighted_route_step_min",
        max_key="highlighted_route_step_max",
        fallback_min=int(DEFAULTS.highlighted_route_step_min),
        fallback_max=int(DEFAULTS.highlighted_route_step_max),
        context="map highlighted route step",
    )


def _sample_connected_cells(*, rng, count: int) -> list[Cell]:
    """Sample a connected subset of grid cells for visible map landmarks."""

    start = (int(rng.randrange(GRID_COLS)), int(rng.randrange(GRID_ROWS)))
    selected: list[Cell] = [start]
    selected_set = {start}
    frontier = [cell for cell in neighbors(start, grid_cols=GRID_COLS, grid_rows=GRID_ROWS)]
    while len(selected) < int(count):
        if not frontier:
            candidates = [
                cell
                for cell in ((x, y) for y in range(GRID_ROWS) for x in range(GRID_COLS))
                if cell not in selected_set
            ]
            frontier.extend(candidates)
        choice = frontier.pop(int(rng.randrange(len(frontier))))
        if choice in selected_set:
            continue
        if any(neighbor in selected_set for neighbor in neighbors(choice, grid_cols=GRID_COLS, grid_rows=GRID_ROWS)):
            selected.append(choice)
            selected_set.add(choice)
            for neighbor in neighbors(choice, grid_cols=GRID_COLS, grid_rows=GRID_ROWS):
                if neighbor not in selected_set and neighbor not in frontier:
                    frontier.append(neighbor)
    return sorted(selected, key=lambda item: (int(item[1]), int(item[0])))


def _sample_zone_specs(*, rng) -> tuple[Mapping[str, object], ...]:
    labels = [str(label) for label in rng.sample(list(ZONE_LABELS), len(ZONE_LAYOUTS))]
    return tuple(
        {
            "zone_id": str(layout["zone_id"]),
            "zone_label": str(labels[index]),
            "cell_bounds": [int(value) for value in layout["cell_bounds"]],
        }
        for index, layout in enumerate(ZONE_LAYOUTS)
    )


def _zone_for_cell(cell: Cell, *, zone_specs: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    x, y = int(cell[0]), int(cell[1])
    if y <= 1 and x <= 2:
        return dict(zone_specs[0])
    if y <= 1:
        return dict(zone_specs[1])
    if x <= 2:
        return dict(zone_specs[2])
    return dict(zone_specs[3])


def _sample_title(*, rng) -> str:
    return str(TITLE_OPTIONS[int(rng.randrange(len(TITLE_OPTIONS)))])


def build_map_scene_case(*, instance_seed: int, params: Mapping[str, Any]) -> MapSceneCase:
    """Build one printed map scene graph before task-specific route binding."""

    scene_variant, scene_variant_probabilities = resolve_scene_variant(params)
    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.scene")
    zone_specs = _sample_zone_specs(rng=rng)
    landmark_count_min, landmark_count_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="landmark_count_min",
        max_key="landmark_count_max",
        fallback_min=int(DEFAULTS.landmark_count_min),
        fallback_max=int(DEFAULTS.landmark_count_max),
        context="map landmark count",
    )
    landmark_count = int(
        int(landmark_count_min)
        + (
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{NAMESPACE_ROOT}.landmark_count",
            )
            % (int(landmark_count_max) - int(landmark_count_min) + 1)
        )
    )
    cells = tuple(_sample_connected_cells(rng=rng, count=int(landmark_count)))
    labels = [str(label) for label in rng.sample(list(LANDMARK_NAMES), int(landmark_count))]
    adjacency = build_adjacency(cells, grid_cols=GRID_COLS, grid_rows=GRID_ROWS)
    landmark_specs: list[Mapping[str, object]] = []
    cell_to_landmark_id: Dict[Cell, str] = {}
    for index, (cell, label) in enumerate(zip(cells, labels)):
        zone = _zone_for_cell(cell, zone_specs=zone_specs)
        landmark_id = f"landmark_{index}"
        cell_to_landmark_id[cell] = landmark_id
        landmark_specs.append(
            {
                "landmark_id": landmark_id,
                "landmark_bbox_id": f"landmark_bbox_{index}",
                "landmark_label_bbox_id": f"landmark_label_bbox_{index}",
                "landmark_label": str(label),
                "grid_col": int(cell[0]),
                "grid_row": int(cell[1]),
                "zone_id": str(zone["zone_id"]),
                "zone_label": str(zone["zone_label"]),
            }
        )

    path_specs: list[Mapping[str, object]] = []
    edge_index = 0
    for source in cells:
        for target in adjacency[source]:
            if (int(source[1]), int(source[0])) > (int(target[1]), int(target[0])):
                continue
            path_specs.append(
                {
                    "path_id": f"path_{edge_index}",
                    "path_bbox_id": f"path_bbox_{edge_index}",
                    "source_landmark_id": str(cell_to_landmark_id[source]),
                    "target_landmark_id": str(cell_to_landmark_id[target]),
                    "source_grid_col": int(source[0]),
                    "source_grid_row": int(source[1]),
                    "target_grid_col": int(target[0]),
                    "target_grid_row": int(target[1]),
                }
            )
            edge_index += 1

    return MapSceneCase(
        scene_title=_sample_title(rng=rng),
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        grid_cols=int(GRID_COLS),
        grid_rows=int(GRID_ROWS),
        landmark_count=int(landmark_count),
        zone_specs=tuple(dict(spec) for spec in zone_specs),
        landmark_specs=tuple(dict(spec) for spec in landmark_specs),
        path_specs=tuple(dict(spec) for spec in path_specs),
        cells=tuple(cells),
        adjacency={cell: tuple(values) for cell, values in adjacency.items()},
        cell_to_landmark_id=dict(cell_to_landmark_id),
    )


__all__ = [
    "GRID_COLS",
    "GRID_ROWS",
    "build_map_scene_case",
    "resolve_direction_step_bounds",
    "resolve_highlighted_route_step_bounds",
    "resolve_scene_variant",
]
