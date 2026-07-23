"""Procedural pixel RPG farm renderer prototype.

This module is intentionally not a public task. It is a renderer prototype for
reviewing whether farm-map scenes can support future illustration tasks such as
counting domestic animals inside fenced regions.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.illustrations.shared.object_variants import RENDERER_STYLE_TOP_DOWN_PIXEL_RPG, variant_visual_metadata
from trace_tasks.tasks.illustrations.shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    object_record_for_spec,
    render_illustration_object,
)
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import (
    CANONICAL_TILE_PX,
    PIXEL_DOMESTIC_ANIMALS,
)


BBox = tuple[float, float, float, float]
TileBox = tuple[int, int, int, int]
RGB = tuple[int, int, int]

DEFAULT_GRID_COLS = 28
DEFAULT_GRID_ROWS = 20
DEFAULT_DISPLAY_TILE_PX = 32

# Backward-compatible constants for review scripts/tests that inspect prototype
# modules directly.
GRID_COLS = DEFAULT_GRID_COLS
GRID_ROWS = DEFAULT_GRID_ROWS
BASE_TILE_PX = CANONICAL_TILE_PX


@dataclass(frozen=True)
class PixelFarmLayout:
    """Grid and display geometry for one generated pixel-farm map."""

    cols: int
    rows: int
    tile_px: int

    @property
    def canonical_width_px(self) -> int:
        return int(self.cols) * CANONICAL_TILE_PX

    @property
    def canonical_height_px(self) -> int:
        return int(self.rows) * CANONICAL_TILE_PX

    @property
    def display_width_px(self) -> int:
        return int(self.cols) * int(self.tile_px)

    @property
    def display_height_px(self) -> int:
        return int(self.rows) * int(self.tile_px)


@dataclass(frozen=True)
class PixelFarmRegion:
    """One semantic region in the farm map."""

    region_id: str
    public_name: str
    region_type: str
    tile_xywh: TileBox
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": str(self.region_id),
            "public_name": str(self.public_name),
            "region_type": str(self.region_type),
            "tile_xywh": [int(v) for v in self.tile_xywh],
            "bbox": [round(v, 3) for v in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PixelFarmEntity:
    """One semantic entity in the generated pixel farm map."""

    entity_id: str
    public_name: str
    category: str
    tile_xywh: TileBox
    bbox_xyxy: BBox
    layer: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        metadata = {str(key): value for key, value in self.metadata.items() if str(key) != "object_record"}
        payload = {
            "entity_id": str(self.entity_id),
            "public_name": str(self.public_name),
            "category": str(self.category),
            "tile_xywh": [int(v) for v in self.tile_xywh],
            "bbox": [round(v, 3) for v in self.bbox_xyxy],
            "center_tile_xy": [
                round(self.tile_xywh[0] + self.tile_xywh[2] * 0.5, 3),
                round(self.tile_xywh[1] + self.tile_xywh[3] * 0.5, 3),
            ],
            "layer": str(self.layer),
            "metadata": metadata,
        }
        if "object_record" in self.metadata:
            payload["object_record"] = self.metadata["object_record"]
        return payload


@dataclass(frozen=True)
class PixelFarmScene:
    """Rendered pixel-farm scene plus trace metadata."""

    image: Image.Image
    entities: tuple[PixelFarmEntity, ...]
    regions: tuple[PixelFarmRegion, ...]
    trace: Mapping[str, Any]


def _choose(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[int(rng.randrange(len(values)))]


def _sample_building_door_state(rng: random.Random) -> str:
    return "open" if rng.random() < 0.35 else "closed"


def _shade(color: RGB, delta: int) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def _tile_bbox(tile_xywh: TileBox, *, layout: PixelFarmLayout, offset_xy: tuple[int, int]) -> BBox:
    x, y, w, h = tile_xywh
    ox, oy = offset_xy
    return (
        float(ox + x * layout.tile_px),
        float(oy + y * layout.tile_px),
        float(ox + (x + w) * layout.tile_px),
        float(oy + (y + h) * layout.tile_px),
    )


def _base_rect(tile_xywh: TileBox, *, inset: int = 0) -> tuple[int, int, int, int]:
    x, y, w, h = tile_xywh
    return (
        x * CANONICAL_TILE_PX + int(inset),
        y * CANONICAL_TILE_PX + int(inset),
        (x + w) * CANONICAL_TILE_PX - 1 - int(inset),
        (y + h) * CANONICAL_TILE_PX - 1 - int(inset),
    )


def _rect_tiles(tile_xywh: TileBox) -> list[tuple[int, int]]:
    x, y, w, h = tile_xywh
    return [(tx, ty) for ty in range(y, y + h) for tx in range(x, x + w)]


def _rects_intersect(a: TileBox, b: TileBox, *, pad: int = 0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax - pad < bx + bw and ax + aw + pad > bx and ay - pad < by + bh and ay + ah + pad > by


def _inside_grid(tile_xywh: TileBox, *, layout: PixelFarmLayout, inset: int = 0) -> bool:
    x, y, w, h = tile_xywh
    return x >= inset and y >= inset and x + w <= layout.cols - inset and y + h <= layout.rows - inset


def _inside_region(tile_xywh: TileBox, region: TileBox, *, pad: int = 0) -> bool:
    x, y, w, h = tile_xywh
    rx, ry, rw, rh = region
    return x >= rx + pad and y >= ry + pad and x + w <= rx + rw - pad and y + h <= ry + rh - pad


def _is_clear(
    tile_xywh: TileBox,
    blocked: set[tuple[int, int]],
    occupied: Sequence[TileBox],
    *,
    layout: PixelFarmLayout,
    pad: int = 0,
) -> bool:
    if not _inside_grid(tile_xywh, layout=layout, inset=1):
        return False
    if any(tile in blocked for tile in _rect_tiles(tile_xywh)):
        return False
    return not any(_rects_intersect(tile_xywh, other, pad=pad) for other in occupied)


def _draw_grass_tile(draw: ImageDraw.ImageDraw, x: int, y: int, *, rng: random.Random) -> None:
    px = x * CANONICAL_TILE_PX
    py = y * CANONICAL_TILE_PX
    colors = [(87, 167, 84), (78, 157, 78), (95, 176, 90), (76, 150, 87)]
    draw.rectangle((px, py, px + 15, py + 15), fill=colors[(x * 5 + y * 3) % len(colors)])
    if rng.random() < 0.22:
        speck = (53, 126, 58) if rng.random() < 0.55 else (121, 195, 101)
        sx = px + rng.randrange(2, 14)
        sy = py + rng.randrange(2, 14)
        draw.point((sx, sy), fill=speck)
        draw.point((sx + 1, sy), fill=speck)


def _draw_dirt_tile(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    rng: random.Random,
    connections: Mapping[str, bool],
) -> None:
    px = x * CANONICAL_TILE_PX
    py = y * CANONICAL_TILE_PX
    dirt = (179, 135, 78)
    edge_dark = (127, 88, 55)
    edge_light = (211, 166, 96)
    rut = (134, 94, 58)
    draw.rectangle((px, py, px + 15, py + 15), fill=dirt)

    north = bool(connections.get("north"))
    south = bool(connections.get("south"))
    west = bool(connections.get("west"))
    east = bool(connections.get("east"))
    if not north:
        draw.line((px, py, px + 15, py), fill=edge_dark)
    if not south:
        draw.line((px, py + 15, px + 15, py + 15), fill=edge_light)
    if not west:
        draw.line((px, py, px, py + 15), fill=edge_dark)
    if not east:
        draw.line((px + 15, py, px + 15, py + 15), fill=edge_light)

    horizontal_only = east and west and not north and not south
    vertical_only = north and south and not east and not west
    if horizontal_only:
        draw.point((px + 5, py + 8), fill=rut)
        draw.point((px + 11, py + 7), fill=rut)
    elif vertical_only:
        draw.point((px + 7, py + 5), fill=rut)
        draw.point((px + 8, py + 11), fill=rut)
    elif sum([north, south, west, east]) >= 2:
        draw.rectangle((px + 7, py + 7, px + 8, py + 8), fill=(151, 105, 62))
    for _ in range(2):
        draw.point((px + rng.randrange(2, 14), py + rng.randrange(3, 13)), fill=rut)


def _draw_crop_tile(draw: ImageDraw.ImageDraw, x: int, y: int, *, rng: random.Random) -> None:
    px = x * CANONICAL_TILE_PX
    py = y * CANONICAL_TILE_PX
    draw.rectangle((px, py, px + 15, py + 15), fill=(149, 103, 58))
    leaf = _choose(rng, [(54, 139, 61), (75, 157, 62), (97, 154, 56)])
    dark = _shade(leaf, -38)
    for sx in (4, 8, 12):
        draw.line((px + sx, py + 4, px + sx, py + 13), fill=dark)
        draw.point((px + sx - 1, py + 7), fill=leaf)
        draw.point((px + sx + 1, py + 9), fill=leaf)


def _draw_fence_tile(draw: ImageDraw.ImageDraw, tile_xywh: TileBox, *, orientation: str) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    wood = (141, 88, 48)
    dark = (92, 58, 35)
    light = (187, 126, 66)
    if orientation == "horizontal":
        draw.line((x0 + 1, y0 + 6, x1 - 1, y0 + 6), fill=wood, width=2)
        draw.line((x0 + 1, y0 + 11, x1 - 1, y0 + 11), fill=wood, width=2)
        for px in range(x0 + 3, x1, 8):
            draw.rectangle((px, y0 + 3, px + 2, y0 + 14), fill=dark)
            draw.point((px + 1, y0 + 4), fill=light)
    else:
        draw.line((x0 + 6, y0 + 1, x0 + 6, y1 - 1), fill=wood, width=2)
        draw.line((x0 + 11, y0 + 1, x0 + 11, y1 - 1), fill=wood, width=2)
        for py in range(y0 + 3, y1, 8):
            draw.rectangle((x0 + 3, py, x0 + 14, py + 2), fill=dark)
            draw.point((x0 + 4, py + 1), fill=light)


def _draw_gate_tile(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a distinct closed pen gate tile."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    post = (74, 50, 34)
    gate = (193, 132, 70)
    gate_light = (224, 169, 91)
    gate_dark = (96, 63, 39)
    hinge = (48, 47, 44)
    draw.rectangle((x0 + 1, y0 + 1, x0 + 3, y0 + 15), fill=post)
    draw.rectangle((x1 - 3, y0 + 1, x1 - 1, y0 + 15), fill=post)
    draw.rectangle((x0 + 4, y0 + 3, x1 - 4, y0 + 14), fill=gate, outline=gate_dark)
    draw.line((x0 + 5, y0 + 5, x1 - 5, y0 + 5), fill=gate_light, width=2)
    draw.line((x0 + 5, y0 + 12, x1 - 5, y0 + 12), fill=gate_light, width=2)
    draw.line((x0 + 5, y0 + 13, x1 - 5, y0 + 4), fill=gate_dark)
    draw.line((x0 + 5, y0 + 4, x1 - 5, y0 + 13), fill=gate_dark)
    draw.rectangle((x0 + 3, y0 + 5, x0 + 4, y0 + 8), fill=hinge)
    draw.rectangle((x1 - 5, y0 + 8, x1 - 4, y0 + 10), fill=(246, 204, 103))


def _draw_pen_fence(draw: ImageDraw.ImageDraw, pen: TileBox, *, gate_x: int | None = None) -> set[tuple[int, int]]:
    x, y, w, h = pen
    tiles: set[tuple[int, int]] = set()
    for tx in range(x, x + w):
        if gate_x is not None and tx == int(gate_x):
            _draw_gate_tile(draw, (tx, y, 1, 1))
        else:
            _draw_fence_tile(draw, (tx, y, 1, 1), orientation="horizontal")
        _draw_fence_tile(draw, (tx, y + h - 1, 1, 1), orientation="horizontal")
        tiles.add((tx, y))
        tiles.add((tx, y + h - 1))
    for ty in range(y + 1, y + h - 1):
        _draw_fence_tile(draw, (x, ty, 1, 1), orientation="vertical")
        _draw_fence_tile(draw, (x + w - 1, ty, 1, 1), orientation="vertical")
        tiles.add((x, ty))
        tiles.add((x + w - 1, ty))
    return tiles


def _draw_barn(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    body_rgb: RGB,
    roof_rgb: RGB,
    door_state: str = "closed",
) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    outline = (82, 45, 39)
    draw.rectangle((x0 + 5, y0 + 19, x1 - 5, y1 - 2), fill=body_rgb, outline=outline)
    draw.polygon([(x0 + 2, y0 + 20), ((x0 + x1) // 2, y0 + 5), (x1 - 2, y0 + 20)], fill=roof_rgb, outline=outline)
    door_rect = (x0 + 26, y1 - 23, x0 + 44, y1 - 2)
    if door_state == "open":
        draw.rectangle(door_rect, fill=(35, 30, 25), outline=(62, 39, 31))
        draw.rectangle((x0 + 13, y1 - 23, x0 + 25, y1 - 2), fill=(117, 67, 45), outline=(62, 39, 31))
        draw.line((x0 + 13, y1 - 23, x0 + 25, y1 - 2), fill=(206, 161, 111))
        draw.line((x0 + 25, y1 - 23, x0 + 13, y1 - 2), fill=(206, 161, 111))
        draw.line((x0 + 26, y1 - 23, x0 + 44, y1 - 23), fill=(206, 161, 111))
    else:
        draw.rectangle(door_rect, fill=(117, 67, 45), outline=(62, 39, 31))
        draw.line((x0 + 26, y1 - 23, x0 + 44, y1 - 2), fill=(206, 161, 111))
        draw.line((x0 + 44, y1 - 23, x0 + 26, y1 - 2), fill=(206, 161, 111))
    for wx in (x0 + 11, x1 - 22):
        draw.rectangle((wx, y0 + 26, wx + 10, y0 + 36), fill=(238, 205, 118), outline=(81, 50, 39))
        draw.line((wx + 5, y0 + 26, wx + 5, y0 + 36), fill=(81, 50, 39))
    for rx in range(x0 + 8, x1 - 9, 12):
        draw.line((rx, y0 + 18, rx + 7, y0 + 11), fill=_shade(roof_rgb, -40))


def _draw_coop(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    body_rgb: RGB,
    roof_rgb: RGB,
    door_state: str = "closed",
) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    draw.rectangle((x0 + 4, y0 + 16, x1 - 4, y1 - 2), fill=body_rgb, outline=(78, 55, 37))
    draw.polygon([(x0 + 2, y0 + 17), ((x0 + x1) // 2, y0 + 7), (x1 - 2, y0 + 17)], fill=roof_rgb, outline=(89, 48, 39))
    if door_state == "open":
        draw.rectangle((x0 + 11, y1 - 17, x0 + 20, y1 - 2), fill=(34, 29, 25), outline=(54, 38, 30))
        draw.polygon(
            [(x0 + 7, y1 - 16), (x0 + 10, y1 - 17), (x0 + 10, y1 - 2), (x0 + 7, y1 - 4)],
            fill=(89, 60, 42),
            outline=(54, 38, 30),
        )
        draw.point((x0 + 8, y1 - 10), fill=(237, 203, 96))
    else:
        draw.rectangle((x0 + 11, y1 - 17, x0 + 20, y1 - 2), fill=(89, 60, 42), outline=(54, 38, 30))
        draw.point((x0 + 18, y1 - 10), fill=(237, 203, 96))
    draw.rectangle((x1 - 17, y1 - 17, x1 - 8, y1 - 9), fill=(218, 187, 96), outline=(76, 57, 38))
    draw.line((x1 - 13, y1 - 17, x1 - 13, y1 - 9), fill=(76, 57, 38))


def _draw_hay_bale(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    fill = (217, 177, 72)
    outline = (123, 91, 42)
    draw.rectangle((x0, y0 + 3, x1, y1 - 1), fill=fill, outline=outline)
    draw.line((x0 + 2, y0 + 6, x1 - 2, y0 + 6), fill=(240, 204, 101))
    draw.line((x0 + 4, y0 + 10, x1 - 4, y0 + 10), fill=(164, 120, 51))
    draw.rectangle((x0 + 4, y0 + 3, x0 + 5, y1 - 1), fill=outline)


def _draw_trough(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    draw.polygon([(x0, y0 + 5), (x1, y0 + 5), (x1 - 3, y1 - 2), (x0 + 3, y1 - 2)], fill=(124, 83, 48), outline=(76, 51, 33))
    draw.rectangle((x0 + 3, y0 + 6, x1 - 3, y0 + 9), fill=(86, 150, 175), outline=(54, 93, 112))
    draw.point((x0 + 7, y0 + 7), fill=(168, 217, 226))


def _draw_tile_layer(
    draw: ImageDraw.ImageDraw,
    grid: list[list[str]],
    *,
    rng: random.Random,
) -> None:
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    for y, row in enumerate(grid):
        for x, terrain in enumerate(row):
            _draw_grass_tile(draw, x, y, rng=rng)
            if terrain == "dirt":
                _draw_dirt_tile(
                    draw,
                    x,
                    y,
                    rng=rng,
                    connections={
                        "north": y > 0 and grid[y - 1][x] == "dirt",
                        "south": y + 1 < rows and grid[y + 1][x] == "dirt",
                        "west": x > 0 and grid[y][x - 1] == "dirt",
                        "east": x + 1 < cols and grid[y][x + 1] == "dirt",
                    },
                )
            elif terrain == "crop":
                _draw_crop_tile(draw, x, y, rng=rng)


def _shared_object_spec(
    *,
    entity_id: str,
    public_name: str,
    category: str,
    tile_xywh: TileBox,
    bbox_xyxy: BBox | None,
    metadata: Mapping[str, Any],
) -> IllustrationObjectSpec | None:
    variant = str(metadata.get("variant", public_name))
    if variant == "tree":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="tree",
            public_name="tree",
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            variant_id=str(metadata.get("tree_style", "fruit_tree")),
            visual_attributes=dict(metadata),
            source_entity_type="pixel_farm_entity",
        )
    if variant == "flower":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="flower",
            public_name="flower",
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            visual_attributes=dict(metadata),
            source_entity_type="pixel_farm_entity",
        )
    if variant == "animal" or category == "animal":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="domestic_animal",
            public_name=str(public_name),
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            semantic_attributes={
                "animal_type": str(metadata.get("animal_type", public_name)),
                "region_id": str(metadata.get("region_id", "")),
                "inside_pen": bool(metadata.get("inside_pen", False)),
            },
            visual_attributes=dict(metadata),
            source_entity_type="pixel_farm_entity",
        )
    return None


def _with_shared_object_record(
    *,
    entity_id: str,
    public_name: str,
    category: str,
    tile_xywh: TileBox,
    bbox_xyxy: BBox,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(metadata)
    spec = _shared_object_spec(
        entity_id=entity_id,
        public_name=public_name,
        category=category,
        tile_xywh=tile_xywh,
        bbox_xyxy=bbox_xyxy,
        metadata=result,
    )
    if spec is not None:
        result["object_record"] = object_record_for_spec(
            spec,
            RenderContext(renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG),
        )
    return result


def _add_entity(
    entities: list[PixelFarmEntity],
    *,
    entity_id: str,
    public_name: str,
    category: str,
    tile_xywh: TileBox,
    layer: str,
    layout: PixelFarmLayout,
    offset_xy: tuple[int, int],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    bbox_xyxy = _tile_bbox(tile_xywh, layout=layout, offset_xy=offset_xy)
    resolved_metadata = _with_shared_object_record(
        entity_id=str(entity_id),
        public_name=str(public_name),
        category=str(category),
        tile_xywh=tile_xywh,
        bbox_xyxy=bbox_xyxy,
        metadata=dict(metadata or {}),
    )
    entities.append(
        PixelFarmEntity(
            entity_id=str(entity_id),
            public_name=str(public_name),
            category=str(category),
            tile_xywh=tile_xywh,
            bbox_xyxy=bbox_xyxy,
            layer=str(layer),
            metadata=resolved_metadata,
        )
    )


def _add_region(
    regions: list[PixelFarmRegion],
    *,
    region_id: str,
    public_name: str,
    region_type: str,
    tile_xywh: TileBox,
    layout: PixelFarmLayout,
    offset_xy: tuple[int, int],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    regions.append(
        PixelFarmRegion(
            region_id=str(region_id),
            public_name=str(public_name),
            region_type=str(region_type),
            tile_xywh=tile_xywh,
            bbox_xyxy=_tile_bbox(tile_xywh, layout=layout, offset_xy=offset_xy),
            metadata=dict(metadata or {}),
        )
    )


def _sample_layout(
    rng: random.Random,
    *,
    width: int,
    height: int,
    tile_px: int,
    grid_cols: int | None,
    grid_rows: int | None,
) -> PixelFarmLayout:
    tile_px = max(20, min(36, int(tile_px)))
    max_cols = max(23, min(30, (int(width) - 24) // tile_px))
    max_rows = max(17, min(22, (int(height) - 24) // tile_px))
    min_cols = min(max_cols, 24)
    min_rows = min(max_rows, 17)
    cols = int(grid_cols) if grid_cols is not None else rng.randint(min_cols, max_cols)
    rows = int(grid_rows) if grid_rows is not None else rng.randint(min_rows, max_rows)
    if cols < 23 or rows < 16:
        raise ValueError("pixel farm grid must be at least 23x16")
    layout = PixelFarmLayout(cols=cols, rows=rows, tile_px=tile_px)
    if layout.display_width_px > int(width) or layout.display_height_px > int(height):
        raise ValueError(
            f"pixel farm grid {cols}x{rows} at {tile_px}px tiles does not fit in {width}x{height}"
        )
    return layout


def _make_pen_regions(rng: random.Random, *, layout: PixelFarmLayout) -> tuple[TileBox, TileBox]:
    pen_h = rng.randint(5, 6)
    pen_a_w = rng.randint(7, 9)
    pen_b_w = rng.randint(7, 9)
    pen_y = layout.rows - pen_h - rng.randint(2, 3)
    pen_a = (rng.randint(2, 4), pen_y, pen_a_w, pen_h)
    pen_b = (layout.cols - pen_b_w - rng.randint(2, 4), pen_y + rng.choice([-1, 0, 1]), pen_b_w, pen_h)
    pen_b = (pen_b[0], max(2, min(layout.rows - pen_h - 2, pen_b[1])), pen_b[2], pen_b[3])
    return pen_a, pen_b


def _connect_path(path_tiles: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    sx, sy = start
    ex, ey = end
    step_x = 1 if ex >= sx else -1
    for x in range(sx, ex + step_x, step_x):
        path_tiles.add((x, sy))
    step_y = 1 if ey >= sy else -1
    for y in range(sy, ey + step_y, step_y):
        path_tiles.add((ex, y))


def _animal_size(animal_type: str) -> tuple[int, int]:
    return (2, 1) if str(animal_type) == "cow" else (1, 1)


def _sample_animal_metadata(rng: random.Random, animal_type: str, *, region_id: str, inside_pen: bool) -> dict[str, Any]:
    body_palettes: dict[str, list[RGB]] = {
        "chicken": [(242, 221, 160), (242, 242, 232), (173, 112, 62)],
        "pig": [(222, 132, 150), (235, 155, 170), (203, 118, 139)],
        "sheep": [(230, 226, 205), (242, 239, 223), (212, 207, 189)],
        "cow": [(238, 236, 221), (226, 218, 194), (228, 230, 222)],
    }
    accent_palettes: dict[str, list[RGB]] = {
        "chicken": [(200, 48, 48), (184, 58, 44)],
        "pig": [(184, 84, 112), (199, 105, 132)],
        "sheep": [(73, 63, 56), (92, 77, 63), (44, 45, 44)],
        "cow": [(212, 144, 145), (206, 130, 132)],
    }
    spot_palettes: list[RGB] = [(48, 47, 45), (94, 66, 48), (69, 61, 52)]
    return {
        "variant": "animal",
        "animal_type": str(animal_type),
        "facing": str(_choose(rng, ["left", "right"])),
        "region_id": str(region_id),
        "inside_pen": bool(inside_pen),
        "body_rgb": list(_choose(rng, body_palettes[str(animal_type)])),
        "accent_rgb": list(_choose(rng, accent_palettes[str(animal_type)])),
        "spot_rgb": list(_choose(rng, spot_palettes)),
    }


def _try_place_in_region(
    rng: random.Random,
    *,
    animal_type: str,
    region: TileBox,
    blocked: set[tuple[int, int]],
    occupied: Sequence[TileBox],
    layout: PixelFarmLayout,
) -> TileBox | None:
    aw, ah = _animal_size(animal_type)
    rx, ry, rw, rh = region
    for _attempt in range(180):
        x_min = rx + 1
        x_max = rx + rw - 1 - aw
        y_min = ry + 1
        y_max = ry + rh - 1 - ah
        if x_max < x_min or y_max < y_min:
            return None
        tile_box = (rng.randint(x_min, x_max), rng.randint(y_min, y_max), aw, ah)
        if _inside_region(tile_box, region, pad=1) and _is_clear(tile_box, blocked, occupied, layout=layout, pad=0):
            return tile_box
    return None


def _try_place_outside_pens(
    rng: random.Random,
    *,
    animal_type: str,
    pen_regions: Sequence[TileBox],
    blocked: set[tuple[int, int]],
    occupied: Sequence[TileBox],
    layout: PixelFarmLayout,
) -> TileBox | None:
    aw, ah = _animal_size(animal_type)
    for _attempt in range(260):
        tile_box = (
            rng.randint(1, layout.cols - aw - 1),
            rng.randint(1, layout.rows - ah - 1),
            aw,
            ah,
        )
        if any(_rects_intersect(tile_box, pen, pad=0) for pen in pen_regions):
            continue
        if _is_clear(tile_box, blocked, occupied, layout=layout, pad=0):
            return tile_box
    return None


def _place_animals(
    rng: random.Random,
    entities: list[PixelFarmEntity],
    *,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    pens: Mapping[str, TileBox],
    layout: PixelFarmLayout,
    offset_xy: tuple[int, int],
) -> None:
    required = list(PIXEL_DOMESTIC_ANIMALS)
    rng.shuffle(required)
    inside_plan: list[tuple[str, str]] = []
    for index, animal_type in enumerate(required):
        region_id = "pen_a" if index % 2 == 0 else "pen_b"
        inside_plan.append((str(animal_type), region_id))
    for _ in range(rng.randint(8, 13)):
        inside_plan.append((str(_choose(rng, PIXEL_DOMESTIC_ANIMALS)), str(_choose(rng, ["pen_a", "pen_b"]))))

    animal_index = 0
    for animal_type, region_id in inside_plan:
        tile_box = _try_place_in_region(
            rng,
            animal_type=animal_type,
            region=pens[region_id],
            blocked=blocked,
            occupied=occupied,
            layout=layout,
        )
        if tile_box is None:
            continue
        metadata = _sample_animal_metadata(rng, animal_type, region_id=region_id, inside_pen=True)
        _add_entity(
            entities,
            entity_id=f"animal_{animal_index:02d}",
            public_name=animal_type,
            category="animal",
            tile_xywh=tile_box,
            layer="actor",
            layout=layout,
            offset_xy=offset_xy,
            metadata=metadata,
        )
        occupied.append(tile_box)
        animal_index += 1

    outside_plan = [str(_choose(rng, PIXEL_DOMESTIC_ANIMALS)) for _ in range(rng.randint(5, 8))]
    if not outside_plan:
        outside_plan = ["chicken"]
    for animal_type in outside_plan:
        tile_box = _try_place_outside_pens(
            rng,
            animal_type=animal_type,
            pen_regions=list(pens.values()),
            blocked=blocked,
            occupied=occupied,
            layout=layout,
        )
        if tile_box is None:
            continue
        metadata = _sample_animal_metadata(rng, animal_type, region_id="outside_fence", inside_pen=False)
        _add_entity(
            entities,
            entity_id=f"animal_{animal_index:02d}",
            public_name=animal_type,
            category="animal",
            tile_xywh=tile_box,
            layer="actor",
            layout=layout,
            offset_xy=offset_xy,
            metadata=metadata,
        )
        occupied.append(tile_box)
        animal_index += 1


def _draw_entity(draw: ImageDraw.ImageDraw, entity: PixelFarmEntity) -> None:
    kind = str(entity.metadata.get("variant", entity.public_name))
    if kind == "barn":
        _draw_barn(
            draw,
            entity.tile_xywh,
            body_rgb=tuple(entity.metadata.get("body_rgb", (181, 71, 59))),
            roof_rgb=tuple(entity.metadata.get("roof_rgb", (126, 50, 50))),
            door_state=str(entity.metadata.get("building_door_state", "closed")),
        )
    elif kind == "coop":
        _draw_coop(
            draw,
            entity.tile_xywh,
            body_rgb=tuple(entity.metadata.get("body_rgb", (188, 126, 70))),
            roof_rgb=tuple(entity.metadata.get("roof_rgb", (137, 62, 50))),
            door_state=str(entity.metadata.get("building_door_state", "closed")),
        )
    elif kind == "hay_bale":
        _draw_hay_bale(draw, entity.tile_xywh)
    elif kind == "trough":
        _draw_trough(draw, entity.tile_xywh)
    else:
        spec = _shared_object_spec(
            entity_id=entity.entity_id,
            public_name=entity.public_name,
            category=entity.category,
            tile_xywh=entity.tile_xywh,
            bbox_xyxy=entity.bbox_xyxy,
            metadata=entity.metadata,
        )
        if spec is not None:
            render_illustration_object(
                spec,
                RenderContext(renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG, draw=draw),
            )


def render_pixel_farm_map(
    seed: int,
    *,
    width: int = 960,
    height: int = 720,
    tile_px: int = DEFAULT_DISPLAY_TILE_PX,
    grid_cols: int | None = None,
    grid_rows: int | None = None,
) -> PixelFarmScene:
    """Render one deterministic old-school pixel RPG farm map."""

    rng = random.Random(int(seed))
    layout = _sample_layout(
        rng,
        width=int(width),
        height=int(height),
        tile_px=int(tile_px),
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )
    offset_xy = ((int(width) - layout.display_width_px) // 2, (int(height) - layout.display_height_px) // 2)

    base = Image.new("RGB", (layout.canonical_width_px, layout.canonical_height_px), (84, 162, 81))
    draw = ImageDraw.Draw(base)
    grid = [["grass" for _ in range(layout.cols)] for _ in range(layout.rows)]

    pen_a, pen_b = _make_pen_regions(rng, layout=layout)
    pens = {"pen_a": pen_a, "pen_b": pen_b}
    gate_tiles = {
        "pen_a": (pen_a[0] + pen_a[2] // 2, pen_a[1]),
        "pen_b": (pen_b[0] + pen_b[2] // 2, pen_b[1]),
    }
    barn = (layout.cols // 2 - 2, 2, 5, 4)
    coop = (pen_a[0] + 1, max(2, pen_a[1] - 3), 3, 2)
    path_tiles: set[tuple[int, int]] = set()
    barn_door = (barn[0] + barn[2] // 2, barn[1] + barn[3])
    _connect_path(path_tiles, barn_door, (pen_a[0] + pen_a[2] // 2, pen_a[1]))
    _connect_path(path_tiles, barn_door, (pen_b[0] + pen_b[2] // 2, pen_b[1]))
    for x, y in sorted(path_tiles):
        if 0 <= x < layout.cols and 0 <= y < layout.rows:
            grid[y][x] = "dirt"

    crop_tiles: set[tuple[int, int]] = set()
    crop_x0 = max(2, layout.cols // 2 - 8)
    crop_y0 = max(7, layout.rows // 2 - 2)
    for row in range(rng.randint(3, 5)):
        for x in range(crop_x0, min(layout.cols - 3, crop_x0 + rng.randint(7, 10))):
            if row % 2 == 0 and (x, crop_y0 + row) not in path_tiles:
                crop_tiles.add((x, crop_y0 + row))
    for x, y in crop_tiles:
        grid[y][x] = "crop"

    _draw_tile_layer(draw, grid, rng=rng)

    fence_tiles: set[tuple[int, int]] = set()
    for region_id, pen in pens.items():
        fence_tiles.update(_draw_pen_fence(draw, pen, gate_x=gate_tiles[region_id][0]))

    blocked = set(path_tiles) | set(crop_tiles) | set(fence_tiles)
    occupied: list[TileBox] = []
    entities: list[PixelFarmEntity] = []
    regions: list[PixelFarmRegion] = []
    _add_region(
        regions,
        region_id="pen_a",
        public_name="left pen",
        region_type="fenced_pen",
        tile_xywh=pen_a,
        layout=layout,
        offset_xy=offset_xy,
        metadata={"contains_fence_border": True, "gate_tile": [int(v) for v in gate_tiles["pen_a"]]},
    )
    _add_region(
        regions,
        region_id="pen_b",
        public_name="right pen",
        region_type="fenced_pen",
        tile_xywh=pen_b,
        layout=layout,
        offset_xy=offset_xy,
        metadata={"contains_fence_border": True, "gate_tile": [int(v) for v in gate_tiles["pen_b"]]},
    )

    fixture_specs: list[tuple[str, str, str, TileBox, dict[str, Any]]] = [
        (
            "barn",
            "barn",
            "building",
            barn,
            {
                "variant": "barn",
                "building_door_state": _sample_building_door_state(rng),
                "body_rgb": list(_choose(rng, [(176, 65, 54), (190, 73, 60), (158, 69, 58)])),
                "roof_rgb": list(_choose(rng, [(119, 49, 47), (132, 55, 49), (95, 55, 62)])),
            },
        ),
        (
            "coop",
            "chicken coop",
            "building",
            coop,
            {
                "variant": "coop",
                "building_door_state": _sample_building_door_state(rng),
                "body_rgb": list(_choose(rng, [(188, 126, 70), (202, 142, 82)])),
                "roof_rgb": list(_choose(rng, [(137, 62, 50), (151, 74, 54)])),
            },
        ),
        ("hay_00", "hay bale", "farm_fixture", (pen_b[0] + 1, pen_b[1] - 2, 1, 1), {"variant": "hay_bale"}),
        ("hay_01", "hay bale", "farm_fixture", (pen_b[0] + 3, pen_b[1] - 2, 1, 1), {"variant": "hay_bale"}),
        ("trough_00", "trough", "farm_fixture", (pen_a[0] + 2, pen_a[1] + pen_a[3] - 2, 2, 1), {"variant": "trough"}),
        ("trough_01", "trough", "farm_fixture", (pen_b[0] + 2, pen_b[1] + pen_b[3] - 2, 2, 1), {"variant": "trough"}),
    ]
    for entity_id, public_name, category, tile_box, metadata in fixture_specs:
        if not _inside_grid(tile_box, layout=layout, inset=1):
            continue
        _add_entity(
            entities,
            entity_id=entity_id,
            public_name=public_name,
            category=category,
            tile_xywh=tile_box,
            layer="building" if category == "building" else "object",
            layout=layout,
            offset_xy=offset_xy,
            metadata=metadata,
        )
        occupied.append(tile_box)
        blocked.update(_rect_tiles(tile_box))

    for index in range(rng.randint(5, 8)):
        tile_box = (rng.randint(1, layout.cols - 2), rng.randint(1, layout.rows - 3), 1, 2)
        if any(_rects_intersect(tile_box, pen, pad=0) for pen in pens.values()):
            continue
        if not _is_clear(tile_box, blocked, occupied, layout=layout, pad=0):
            continue
        metadata = {
            "variant": "tree",
            **variant_visual_metadata("tree", "fruit_tree", RENDERER_STYLE_TOP_DOWN_PIXEL_RPG),
            "tree_style": "fruit_tree",
            "leaf_rgb": list(_choose(rng, [(42, 139, 76), (54, 147, 72), (65, 137, 69)])),
            "fruit_rgb": list(_choose(rng, [(218, 62, 58), (235, 181, 54), (190, 65, 126)])),
        }
        _add_entity(
            entities,
            entity_id=f"tree_{index:02d}",
            public_name="tree",
            category="plant",
            tile_xywh=tile_box,
            layer="plant",
            layout=layout,
            offset_xy=offset_xy,
            metadata=metadata,
        )
        occupied.append(tile_box)
        blocked.update(_rect_tiles(tile_box))

    for index in range(rng.randint(5, 9)):
        tile_box = (rng.randint(1, layout.cols - 2), rng.randint(1, layout.rows - 2), 1, 1)
        if not _is_clear(tile_box, blocked, occupied, layout=layout, pad=0):
            continue
        _add_entity(
            entities,
            entity_id=f"flower_{index:02d}",
            public_name="flower",
            category="plant",
            tile_xywh=tile_box,
            layer="plant",
            layout=layout,
            offset_xy=offset_xy,
            metadata={
                "variant": "flower",
                "flower_rgb": list(_choose(rng, [(238, 126, 62), (236, 82, 111), (239, 201, 72)])),
                "leaf_rgb": list(_choose(rng, [(36, 125, 62), (44, 144, 72), (55, 132, 72)])),
            },
        )
        occupied.append(tile_box)

    _place_animals(rng, entities, blocked=blocked, occupied=occupied, pens=pens, layout=layout, offset_xy=offset_xy)

    for entity in sorted(entities, key=lambda item: (item.tile_xywh[1], item.tile_xywh[0], item.layer)):
        _draw_entity(draw, entity)

    draw.rectangle((0, 0, layout.canonical_width_px - 1, layout.canonical_height_px - 1), outline=(46, 92, 46), width=2)
    image = base.resize((layout.display_width_px, layout.display_height_px), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (int(width), int(height)), (230, 232, 214))
    canvas.paste(image, offset_xy)

    entities_sorted = tuple(sorted(entities, key=lambda item: item.entity_id))
    regions_sorted = tuple(sorted(regions, key=lambda item: item.region_id))
    animal_entities = [entity for entity in entities_sorted if entity.category == "animal"]
    trace = {
        "renderer_id": "pixel_farm_map_v0",
        "seed": int(seed),
        "inspiration_sources": [
            {
                "name": "Kenney Animal Pack",
                "url": "https://kenney.nl/assets/animal-pack",
                "notes": "CC0 pixel animal pack used as visual reference only.",
            },
            {
                "name": "OpenGameArt Animal Pack Redux",
                "url": "https://opengameart.org/content/animal-pack-redux",
                "notes": "CC0 animal sprite pack used as visual reference only.",
            },
        ],
        "uses_external_sprites": False,
        "grid_cols": layout.cols,
        "grid_rows": layout.rows,
        "canonical_tile_px": CANONICAL_TILE_PX,
        "tile_px": layout.tile_px,
        "map_size_px": [layout.display_width_px, layout.display_height_px],
        "map_offset_xy": list(offset_xy),
        "path_tiles": [[int(x), int(y)] for x, y in sorted(path_tiles)],
        "crop_tiles": [[int(x), int(y)] for x, y in sorted(crop_tiles)],
        "fence_tiles": [[int(x), int(y)] for x, y in sorted(fence_tiles)],
        "gate_tiles": {region_id: [int(x), int(y)] for region_id, (x, y) in sorted(gate_tiles.items())},
        "regions": [region.as_dict() for region in regions_sorted],
        "entity_count": len(entities_sorted),
        "animal_count": len(animal_entities),
        "inside_pen_animal_count": sum(1 for entity in animal_entities if bool(entity.metadata.get("inside_pen"))),
        "outside_pen_animal_count": sum(1 for entity in animal_entities if not bool(entity.metadata.get("inside_pen"))),
        "entities": [entity.as_dict() for entity in entities_sorted],
        "category_counts": {
            category: sum(1 for entity in entities_sorted if entity.category == category)
            for category in sorted({entity.category for entity in entities_sorted})
        },
        "public_name_counts": {
            name: sum(1 for entity in entities_sorted if entity.public_name == name)
            for name in sorted({entity.public_name for entity in entities_sorted})
        },
    }
    return PixelFarmScene(image=canvas, entities=entities_sorted, regions=regions_sorted, trace=trace)


def draw_pixel_farm_debug_overlay(scene: PixelFarmScene) -> Image.Image:
    """Return a debug overlay with semantic bboxes, regions, and public names."""

    image = scene.image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    try:
        from trace_tasks.tasks.shared.text_rendering import load_font

        font = load_font(11, bold=True)
    except Exception:  # pragma: no cover - review helper fallback
        font = None

    for region in scene.regions:
        box = [round(v) for v in region.bbox_xyxy]
        draw.rectangle(box, outline=(40, 105, 210, 230), width=3)
        text_bbox = draw.textbbox((box[0], box[1]), region.public_name, font=font)
        draw.rectangle(
            (text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1),
            fill=(240, 248, 255, 220),
        )
        draw.text((box[0], box[1]), region.public_name, fill=(40, 105, 210, 255), font=font)

    palette = {
        "animal": (198, 60, 70, 235),
        "building": (100, 78, 190, 235),
        "farm_fixture": (211, 132, 45, 235),
        "plant": (34, 144, 73, 235),
    }
    for entity in scene.entities:
        color = palette.get(entity.category, (35, 35, 35, 235))
        box = [round(v) for v in entity.bbox_xyxy]
        draw.rectangle(box, outline=color, width=2)
        label = entity.public_name
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        bg = (text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1)
        draw.rectangle(bg, fill=(255, 255, 238, 220))
        draw.text((box[0], box[1]), label, fill=color, font=font)
    image.alpha_composite(overlay)
    return image.convert("RGB")


__all__ = [
    "BASE_TILE_PX",
    "CANONICAL_TILE_PX",
    "DEFAULT_DISPLAY_TILE_PX",
    "GRID_COLS",
    "GRID_ROWS",
    "PixelFarmEntity",
    "PixelFarmLayout",
    "PixelFarmRegion",
    "PixelFarmScene",
    "draw_pixel_farm_debug_overlay",
    "render_pixel_farm_map",
]
