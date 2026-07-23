"""Passive state records for Minecraft-like block-world games scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass(frozen=True)
class MinecraftCell:
    """One visible terrain cell in the block-world scene."""

    cell_id: str
    x: int
    y: int
    kind: str


@dataclass(frozen=True)
class MinecraftBlock:
    """One visible cube block in the block-world scene."""

    block_id: str
    x: int
    y: int
    z: int
    kind: str


@dataclass(frozen=True)
class MinecraftRouteOverlay:
    """One visible route cue drawn over the block-world floor."""

    label: str
    cells: Tuple[Tuple[int, int], ...]
    rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class MinecraftSceneSample:
    """Generated block-world state before rendering and prompt assembly."""

    grid_width: int
    grid_depth: int
    sample_kind: str
    style_variant: str
    answer: int
    terrain_cells: Tuple[MinecraftCell, ...]
    blocks: Tuple[MinecraftBlock, ...]
    player_cell: Tuple[int, int] | None
    target_cell: Tuple[int, int] | None
    river_width: int
    scaffold_cost: int
    ladder_present: bool
    annotation_entity_ids: Tuple[str, ...]
    construction_mode: str
    route_overlays: Tuple[MinecraftRouteOverlay, ...] = ()
    target_resource_kind: str = ""
    counted_resource_kind: str = ""
    ladder_columns: Tuple[Tuple[int, int, int], ...] = ()
    target_stack_height: int = 0
    stack_height_condition: str = ""
    track_cells: Tuple[Tuple[int, int], ...] = ()


@dataclass(frozen=True)
class MinecraftRenderParams:
    """Resolved render controls for one block-world scene."""

    canvas_width: int
    canvas_height: int
    tile_width_px: int
    tile_height_px: int
    cube_height_px: int
    outline_width_px: int
    player_marker_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None
    max_stack_height: int = 1
    player_marker_label: str = "P"


@dataclass(frozen=True)
class MinecraftTheme:
    """Palette for one block-world scene style."""

    ground_rgb: Tuple[int, int, int]
    ground_alt_rgb: Tuple[int, int, int]
    water_rgb: Tuple[int, int, int]
    water_line_rgb: Tuple[int, int, int]
    outline_rgb: Tuple[int, int, int]
    support_rgb: Tuple[int, int, int]
    stone_rgb: Tuple[int, int, int]
    gold_rgb: Tuple[int, int, int]
    diamond_rgb: Tuple[int, int, int]
    ladder_rgb: Tuple[int, int, int]
    player_rgb: Tuple[int, int, int]
    arrow_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedMinecraftScene:
    """Rendered block-world image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


def terrain_cell_entity_id(x: int, y: int) -> str:
    """Return a stable terrain-cell entity id."""

    return f"cell_{int(x):02d}_{int(y):02d}"


def water_cell_entity_id(x: int, y: int) -> str:
    """Return a stable water-cell entity id."""

    return f"water_{int(x):02d}_{int(y):02d}"


def track_block_entity_id(index: int) -> str:
    """Return a stable raised track-block id."""

    return f"track_raised_block_{int(index):02d}"


def stack_entity_id(x: int, y: int) -> str:
    """Return a stable visible cube-stack entity id."""

    return f"stack_{int(x):02d}_{int(y):02d}"


def ladder_entity_id() -> str:
    """Return the stable ladder entity id."""

    return "ladder"


def player_entity_id() -> str:
    """Return the stable player marker entity id."""

    return "player_marker"


__all__ = [
    "MinecraftBlock",
    "MinecraftCell",
    "MinecraftRenderParams",
    "MinecraftRouteOverlay",
    "MinecraftSceneSample",
    "MinecraftTheme",
    "RenderedMinecraftScene",
    "ladder_entity_id",
    "player_entity_id",
    "stack_entity_id",
    "terrain_cell_entity_id",
    "track_block_entity_id",
    "water_cell_entity_id",
]
