"""Procedural pixel RPG village renderer prototype.

This module is intentionally not a public task. It is a renderer prototype for
reviewing whether old-school pixel village map scenes are viable for future
illustration tasks. The village is composed from optional semantic territories
such as residential blocks, landmarks, orchards, and cemetery/churchyard areas.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.illustrations.shared.object_variants import (
    RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
    sample_object_variant_id,
    variant_visual_metadata,
)
from trace_tasks.tasks.illustrations.shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    object_record_for_spec,
    render_illustration_object,
)
from trace_tasks.tasks.illustrations.shared.object_schema import json_safe
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import (
    PIXEL_GRAVE_MARKER_STYLES,
    PIXEL_TREE_STYLES,
    draw_pixel_barrel,
    draw_pixel_bench,
    draw_pixel_castle,
    draw_pixel_church,
    draw_pixel_cart,
    draw_pixel_dead_tree,
    draw_pixel_gazebo,
    draw_pixel_grave_marker,
    draw_pixel_lamp_post,
    draw_pixel_market_stall,
    draw_pixel_notice_board,
    draw_pixel_pond,
    draw_pixel_rock,
    draw_pixel_statue,
    draw_pixel_well,
    draw_pixel_windmill,
    draw_pixel_woodpile,
    draw_pixel_wagon,
    draw_pixel_winter_overlay,
)
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import DEFAULT_RPG_TILE_PX
from .regions import (
    PixelOrchardPlan,
    draw_pixel_orchard_boundary,
    draw_pixel_orchard_ground,
    sample_pixel_orchard_plan,
)


RGB = tuple[int, int, int]
BBox = tuple[float, float, float, float]
TileBox = tuple[int, int, int, int]

DEFAULT_GRID_COLS = 30
DEFAULT_GRID_ROWS = 22
CANONICAL_TILE_PX = 16
DEFAULT_DISPLAY_TILE_PX = DEFAULT_RPG_TILE_PX

# Backward-compatible constants for review scripts/tests that inspect the
# prototype module directly. Rendered scenes now sample per-instance dimensions.
GRID_COLS = DEFAULT_GRID_COLS
GRID_ROWS = DEFAULT_GRID_ROWS
BASE_TILE_PX = CANONICAL_TILE_PX


@dataclass(frozen=True)
class PixelVillageLayout:
    """Grid and display geometry for one generated pixel-village map."""

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
class PixelVillageTheme:
    """Render-only seasonal treatment for one generated pixel-village map."""

    theme_id: str
    snow_intensity: str
    snow_coverage_base: float
    snow_rgb: RGB
    snow_shadow_rgb: RGB
    autumn_intensity: str
    leaf_coverage_base: float
    leaf_overlay_rgb: RGB
    leaf_shadow_rgb: RGB
    leaf_accent_rgb: RGB
    canvas_rgb: RGB


@dataclass(frozen=True)
class PixelVillageEntity:
    """One semantic entity in the generated pixel village map."""

    entity_id: str
    public_name: str
    category: str
    tile_xywh: TileBox
    bbox_xyxy: BBox
    layer: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        metadata = {str(key): json_safe(value) for key, value in self.metadata.items() if str(key) != "object_record"}
        payload = {
            "entity_id": self.entity_id,
            "public_name": self.public_name,
            "category": self.category,
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
            payload["object_record"] = json_safe(self.metadata["object_record"])
        return payload


@dataclass(frozen=True)
class PixelVillageTerritory:
    """One semantic area in the generated pixel village map."""

    territory_id: str
    territory_type: str
    public_name: str
    tile_xywh: TileBox
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "territory_id": self.territory_id,
            "territory_type": self.territory_type,
            "public_name": self.public_name,
            "tile_xywh": [int(v) for v in self.tile_xywh],
            "bbox": [round(v, 3) for v in self.bbox_xyxy],
            "center_tile_xy": [
                round(self.tile_xywh[0] + self.tile_xywh[2] * 0.5, 3),
                round(self.tile_xywh[1] + self.tile_xywh[3] * 0.5, 3),
            ],
            "metadata": json_safe(self.metadata),
        }


@dataclass(frozen=True)
class PixelVillageScene:
    """Rendered pixel-village scene plus trace metadata."""

    image: Image.Image
    entities: tuple[PixelVillageEntity, ...]
    territories: tuple[PixelVillageTerritory, ...]
    trace: Mapping[str, Any]


def _tile_bbox(tile_xywh: TileBox, *, layout: PixelVillageLayout, offset_xy: tuple[int, int]) -> BBox:
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


def _choose(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[int(rng.randrange(len(values)))]


def _shade_rgb(color: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def _sample_village_theme(rng: random.Random, theme_mode: str) -> PixelVillageTheme:
    """Choose seasonal palette metadata while preserving map geometry."""

    mode = str(theme_mode)
    if mode not in {"temperate", "winter", "autumn", "auto"}:
        raise ValueError("theme_mode must be one of: temperate, winter, autumn, auto")
    if mode == "auto":
        roll = rng.random()
        theme_id = "winter" if roll < 0.25 else "autumn" if roll < 0.55 else "temperate"
    else:
        theme_id = mode
    if theme_id == "temperate":
        return PixelVillageTheme(
            theme_id="temperate",
            snow_intensity="none",
            snow_coverage_base=0.0,
            snow_rgb=(239, 246, 248),
            snow_shadow_rgb=(178, 202, 215),
            autumn_intensity="none",
            leaf_coverage_base=0.0,
            leaf_overlay_rgb=(177, 126, 55),
            leaf_shadow_rgb=(104, 82, 47),
            leaf_accent_rgb=(154, 78, 51),
            canvas_rgb=(232, 234, 216),
        )
    if theme_id == "winter":
        snow_intensity = str(_choose(rng, ["light", "medium", "heavy"]))
        coverage_base = {"light": 0.32, "medium": 0.54, "heavy": 0.74}[snow_intensity]
        return PixelVillageTheme(
            theme_id="winter",
            snow_intensity=snow_intensity,
            snow_coverage_base=float(coverage_base),
            snow_rgb=(240, 247, 249),
            snow_shadow_rgb=(176, 204, 218),
            autumn_intensity="none",
            leaf_coverage_base=0.0,
            leaf_overlay_rgb=(177, 126, 55),
            leaf_shadow_rgb=(104, 82, 47),
            leaf_accent_rgb=(154, 78, 51),
            canvas_rgb=(224, 231, 232),
        )
    autumn_intensity = str(_choose(rng, ["early", "peak", "late"]))
    coverage_base = {"early": 0.18, "peak": 0.28, "late": 0.23}[autumn_intensity]
    leaf_palette = {
        "early": ((151, 140, 65), (91, 90, 49), (177, 117, 55)),
        "peak": ((177, 126, 55), (104, 82, 47), (154, 78, 51)),
        "late": ((138, 108, 62), (84, 74, 50), (167, 130, 67)),
    }[autumn_intensity]
    return PixelVillageTheme(
        theme_id="autumn",
        snow_intensity="none",
        snow_coverage_base=0.0,
        snow_rgb=(239, 246, 248),
        snow_shadow_rgb=(178, 202, 215),
        autumn_intensity=autumn_intensity,
        leaf_coverage_base=float(coverage_base),
        leaf_overlay_rgb=leaf_palette[0],
        leaf_shadow_rgb=leaf_palette[1],
        leaf_accent_rgb=leaf_palette[2],
        canvas_rgb=(232, 228, 207),
    )


def _rects_intersect(a: TileBox, b: TileBox, *, pad: int = 0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax - pad < bx + bw and ax + aw + pad > bx and ay - pad < by + bh and ay + ah + pad > by


def _inside_grid(tile_xywh: TileBox, *, layout: PixelVillageLayout, inset: int = 0) -> bool:
    x, y, w, h = tile_xywh
    return x >= inset and y >= inset and x + w <= layout.cols - inset and y + h <= layout.rows - inset


def _mark_tiles(grid: list[list[str]], tiles: Sequence[tuple[int, int]], value: str) -> None:
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    for x, y in tiles:
        if 0 <= x < cols and 0 <= y < rows:
            grid[y][x] = value


def _rect_tiles(tile_xywh: TileBox) -> list[tuple[int, int]]:
    x, y, w, h = tile_xywh
    return [(tx, ty) for ty in range(y, y + h) for tx in range(x, x + w)]


def _expanded_tiles(
    tiles: set[tuple[int, int]],
    *,
    radius: int,
    layout: PixelVillageLayout,
) -> set[tuple[int, int]]:
    distance = max(0, int(radius))
    if distance <= 0:
        return set(tiles)
    expanded: set[tuple[int, int]] = set()
    for x, y in tiles:
        for dx in range(-distance, distance + 1):
            for dy in range(-distance, distance + 1):
                tx = int(x) + dx
                ty = int(y) + dy
                if 0 <= tx < int(layout.cols) and 0 <= ty < int(layout.rows):
                    expanded.add((tx, ty))
    return expanded


def _is_clear(
    tile_xywh: TileBox,
    blocked: set[tuple[int, int]],
    occupied: Sequence[TileBox],
    *,
    layout: PixelVillageLayout,
    pad: int = 0,
) -> bool:
    if not _inside_grid(tile_xywh, layout=layout, inset=1):
        return False
    if any(tile in blocked for tile in _rect_tiles(tile_xywh)):
        return False
    return not any(_rects_intersect(tile_xywh, other, pad=pad) for other in occupied)


def _draw_grass_tile(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    rng: random.Random,
    variant: int,
    theme: PixelVillageTheme,
) -> None:
    """Draw one terrain tile with theme-specific visual noise."""

    px = x * CANONICAL_TILE_PX
    py = y * CANONICAL_TILE_PX
    if theme.theme_id == "winter":
        colors = [(211, 225, 226), (221, 233, 234), (203, 218, 221), (229, 237, 236)]
        draw.rectangle((px, py, px + 15, py + 15), fill=colors[variant % len(colors)])
        grass_prob = {"light": 0.42, "medium": 0.22, "heavy": 0.08}.get(theme.snow_intensity, 0.18)
        if rng.random() < grass_prob:
            patch = (82, 143, 86)
            sx = px + rng.randrange(2, 12)
            sy = py + rng.randrange(3, 13)
            draw.line((sx, sy, sx + 3, sy), fill=patch)
            draw.point((sx + 1, sy - 1), fill=(67, 126, 78))
        elif rng.random() < 0.22:
            sx = px + rng.randrange(2, 13)
            sy = py + rng.randrange(2, 13)
            draw.point((sx, sy), fill=(185, 207, 217))
        return
    if theme.theme_id == "autumn":
        colors = [(103, 150, 78), (116, 151, 75), (128, 151, 78), (96, 139, 82)]
        draw.rectangle((px, py, px + 15, py + 15), fill=colors[variant % len(colors)])
        if rng.random() < {"early": 0.18, "peak": 0.28, "late": 0.22}.get(theme.autumn_intensity, 0.22):
            sx = px + rng.randrange(2, 13)
            sy = py + rng.randrange(2, 13)
            fill = theme.leaf_overlay_rgb if rng.random() < 0.62 else theme.leaf_accent_rgb
            draw.point((sx, sy), fill=fill)
            if rng.random() < 0.45:
                draw.point((sx + 1, sy), fill=theme.leaf_shadow_rgb)
        elif rng.random() < 0.18:
            sx = px + rng.randrange(2, 13)
            sy = py + rng.randrange(2, 13)
            draw.point((sx, sy), fill=(83, 126, 68))
        return
    colors = [(77, 166, 85), (69, 154, 78), (86, 176, 88), (74, 160, 91)]
    draw.rectangle((px, py, px + 15, py + 15), fill=colors[variant % len(colors)])
    if rng.random() < 0.28:
        speck = (52, 127, 61) if rng.random() < 0.5 else (112, 193, 99)
        sx = px + rng.randrange(2, 13)
        sy = py + rng.randrange(2, 13)
        draw.point((sx, sy), fill=speck)
        draw.point((sx + 1, sy), fill=speck)


def _draw_path_tile(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    *,
    rng: random.Random,
    connections: Mapping[str, bool],
    theme: PixelVillageTheme,
) -> None:
    """Draw one connected path tile with edge continuity."""

    px = x * CANONICAL_TILE_PX
    py = y * CANONICAL_TILE_PX
    if theme.theme_id == "winter":
        dirt = (174, 156, 127)
        edge_dark = (123, 112, 96)
        edge_light = theme.snow_rgb
        rut = (136, 123, 103)
    elif theme.theme_id == "autumn":
        dirt = (174, 135, 79)
        edge_dark = (121, 91, 55)
        edge_light = (207, 162, 95)
        rut = (136, 99, 57)
    else:
        dirt = (181, 144, 83)
        edge_dark = (129, 98, 60)
        edge_light = (215, 174, 104)
        rut = (145, 109, 65)
    draw.rectangle((px, py, px + 15, py + 15), fill=dirt)

    north = bool(connections.get("north"))
    south = bool(connections.get("south"))
    west = bool(connections.get("west"))
    east = bool(connections.get("east"))

    if not north:
        draw.line((px, py, px + 15, py), fill=edge_dark)
        if theme.theme_id == "winter":
            inset_light = theme.snow_shadow_rgb
        elif theme.theme_id == "autumn":
            inset_light = (158, 116, 67)
        else:
            inset_light = (165, 125, 73)
        draw.line((px + 1, py + 1, px + 14, py + 1), fill=inset_light)
    if not south:
        draw.line((px, py + 15, px + 15, py + 15), fill=edge_light)
    if not west:
        draw.line((px, py, px, py + 15), fill=edge_dark)
        if theme.theme_id == "winter":
            inset_light = theme.snow_shadow_rgb
        elif theme.theme_id == "autumn":
            inset_light = (158, 116, 67)
        else:
            inset_light = (165, 125, 73)
        draw.line((px + 1, py + 1, px + 1, py + 14), fill=inset_light)
    if not east:
        draw.line((px + 15, py, px + 15, py + 15), fill=edge_light)

    horizontal_only = east and west and not north and not south
    vertical_only = north and south and not east and not west
    junction = sum([north, south, west, east]) >= 3
    turn = sum([north, south, west, east]) == 2 and not horizontal_only and not vertical_only
    if horizontal_only:
        draw.point((px + 5, py + 8), fill=rut)
        draw.point((px + 10, py + 7), fill=rut)
    elif vertical_only:
        draw.point((px + 7, py + 5), fill=rut)
        draw.point((px + 8, py + 10), fill=rut)
    elif junction or turn:
        draw.rectangle((px + 7, py + 7, px + 8, py + 8), fill=(158, 118, 69))
    for _ in range(2):
        sx = px + rng.randrange(2, 14)
        sy = py + rng.randrange(2, 14)
        if theme.theme_id == "winter":
            speck = (118, 109, 96)
        elif theme.theme_id == "autumn":
            speck = (130, 88, 52)
        else:
            speck = (139, 104, 63)
        draw.point((sx, sy), fill=speck)


def _draw_water_tile(draw: ImageDraw.ImageDraw, x: int, y: int, *, rng: random.Random, theme: PixelVillageTheme) -> None:
    px = x * CANONICAL_TILE_PX
    py = y * CANONICAL_TILE_PX
    if theme.theme_id == "winter":
        draw.rectangle((px, py, px + 15, py + 15), fill=(73, 141, 178))
        draw.line((px + 2, py + 5, px + 12, py + 5), fill=(194, 230, 239))
        if rng.random() < 0.65:
            draw.line((px + 5, py + 11, px + 14, py + 11), fill=(143, 198, 218))
        return
    draw.rectangle((px, py, px + 15, py + 15), fill=(51, 132, 188))
    draw.line((px + 2, py + 5, px + 6, py + 5), fill=(109, 185, 223))
    if rng.random() < 0.55:
        draw.line((px + 8, py + 11, px + 13, py + 11), fill=(33, 105, 165))


def _draw_flower_tile(draw: ImageDraw.ImageDraw, x: int, y: int, *, rng: random.Random, theme: PixelVillageTheme) -> None:
    px = x * CANONICAL_TILE_PX
    py = y * CANONICAL_TILE_PX
    if theme.theme_id == "winter":
        sx = px + rng.randrange(4, 12)
        sy = py + rng.randrange(5, 13)
        draw.line((sx - 2, sy, sx + 2, sy), fill=(188, 208, 216))
        draw.point((sx, sy - 1), fill=theme.snow_rgb)
        return
    if theme.theme_id == "autumn":
        sx = px + rng.randrange(4, 12)
        sy = py + rng.randrange(6, 13)
        stem = (82, 109, 58)
        head = theme.leaf_overlay_rgb if rng.random() < 0.58 else theme.leaf_accent_rgb
        draw.line((sx, sy, sx, py + 14), fill=stem)
        draw.point((sx - 1, sy), fill=head)
        draw.point((sx, sy - 1), fill=_shade_rgb(head, 18))
        draw.point((sx + 1, sy), fill=head)
        return
    color = _choose(rng, [(238, 91, 99), (246, 205, 88), (217, 116, 196)])
    cx = px + rng.randrange(4, 12)
    cy = py + rng.randrange(4, 12)
    draw.point((cx, cy), fill=color)
    draw.point((cx + 1, cy), fill=color)
    draw.point((cx, cy + 1), fill=color)


def _draw_tile_layer(
    draw: ImageDraw.ImageDraw,
    grid: list[list[str]],
    *,
    rng: random.Random,
    theme: PixelVillageTheme,
) -> None:
    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    for y, row in enumerate(grid):
        for x, terrain in enumerate(row):
            _draw_grass_tile(draw, x, y, rng=rng, variant=(x * 7 + y * 5) % 4, theme=theme)
            if terrain == "path":
                _draw_path_tile(
                    draw,
                    x,
                    y,
                    rng=rng,
                    theme=theme,
                    connections={
                        "north": y > 0 and grid[y - 1][x] == "path",
                        "south": y + 1 < rows and grid[y + 1][x] == "path",
                        "west": x > 0 and grid[y][x - 1] == "path",
                        "east": x + 1 < cols and grid[y][x + 1] == "path",
                    },
                )
            elif terrain == "water":
                _draw_water_tile(draw, x, y, rng=rng, theme=theme)
            elif terrain == "flower":
                _draw_flower_tile(draw, x, y, rng=rng, theme=theme)


def _draw_bridge(draw: ImageDraw.ImageDraw, tile_xywh: TileBox, *, orientation: str) -> None:
    rect = _base_rect(tile_xywh)
    draw.rectangle(rect, fill=(151, 101, 54), outline=(90, 62, 36))
    x0, y0, x1, y1 = rect
    if orientation == "horizontal":
        for x in range(x0 + 4, x1, 7):
            draw.line((x, y0 + 2, x, y1 - 2), fill=(202, 151, 83))
        draw.line((x0, y0 + 3, x1, y0 + 3), fill=(80, 54, 34))
        draw.line((x0, y1 - 3, x1, y1 - 3), fill=(80, 54, 34))
    else:
        for y in range(y0 + 4, y1, 7):
            draw.line((x0 + 2, y, x1 - 2, y), fill=(202, 151, 83))
        draw.line((x0 + 3, y0, x0 + 3, y1), fill=(80, 54, 34))
        draw.line((x1 - 3, y0, x1 - 3, y1), fill=(80, 54, 34))


def _draw_roof_texture(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    roof_color: tuple[int, int, int],
    roof_style: str,
    direction: str,
) -> None:
    """Apply roof texture without changing the building footprint."""

    x0, y0, x1, y1 = rect
    roof_dark = _shade_rgb(roof_color, -40)
    roof_light = _shade_rgb(roof_color, 24)
    style = str(roof_style)
    if direction == "vertical":
        ridge_x = (x0 + x1) // 2
        draw.line((ridge_x, y0 + 2, ridge_x, y1 - 2), fill=roof_dark)
        for y in range(y0 + 7, y1 - 3, 8):
            draw.line((x0 + 3, y, x1 - 3, y), fill=roof_dark)
        if style == "wood_plank":
            for x in range(x0 + 5, x1 - 3, 7):
                draw.line((x, y0 + 4, x, y1 - 4), fill=_shade_rgb(roof_color, -25))
        elif style == "tile":
            for y in range(y0 + 5, y1 - 4, 10):
                for x in range(x0 + 4, x1 - 4, 8):
                    draw.arc((x - 3, y - 2, x + 4, y + 5), start=180, end=360, fill=roof_dark)
        else:
            for y in range(y0 + 8, y1 - 5, 12):
                draw.line((x0 + 4, y, ridge_x - 1, y + 4), fill=roof_light)
                draw.line((ridge_x + 1, y + 4, x1 - 4, y), fill=roof_light)
    else:
        for x in range(x0 + 3, x1 - 4, 9):
            draw.line((x, y0 + 4, x + 5, y1 - 3), fill=roof_dark)
        if style == "wood_plank":
            for y in range(y0 + 7, y1 - 3, 6):
                draw.line((x0 + 3, y, x1 - 3, y), fill=_shade_rgb(roof_color, -25))
        elif style == "tile":
            for y in range(y0 + 7, y1 - 4, 7):
                for x in range(x0 + 5, x1 - 5, 9):
                    draw.arc((x - 4, y - 3, x + 4, y + 5), start=180, end=360, fill=roof_dark)
        else:
            draw.line((x0 + 2, y0 + 2, x1 - 2, y0 + 2), fill=roof_light)


def _draw_wall_texture(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    body_color: tuple[int, int, int],
    wall_style: str,
) -> None:
    x0, y0, x1, y1 = rect
    dark = _shade_rgb(body_color, -42)
    light = _shade_rgb(body_color, 24)
    style = str(wall_style)
    if style == "wood":
        for x in range(x0 + 5, x1 - 3, 8):
            draw.line((x, y0 + 2, x, y1 - 2), fill=dark)
        for y in range(y0 + 8, y1 - 3, 14):
            draw.line((x0 + 3, y, x1 - 3, y), fill=_shade_rgb(body_color, -18))
    elif style == "stone":
        for y in range(y0 + 6, y1 - 4, 9):
            draw.line((x0 + 3, y, x1 - 3, y), fill=dark)
        for y in range(y0 + 9, y1 - 5, 9):
            offset = 5 if ((y - y0) // 9) % 2 else 0
            for x in range(x0 + 6 + offset, x1 - 4, 13):
                draw.line((x, y - 3, x, y + 3), fill=dark)
    else:
        draw.point((x0 + 5, y0 + 5), fill=light)
        draw.point((x1 - 7, y0 + 11), fill=dark)
        draw.point((x0 + 9, y1 - 8), fill=_shade_rgb(body_color, -20))


def _draw_house_window(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    draw.rectangle(rect, fill=(127, 203, 221), outline=(65, 91, 106))
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    draw.line((cx, y0, cx, y1), fill=(229, 244, 239))
    draw.line((x0, cy, x1, cy), fill=(82, 126, 143))


def _draw_house_door(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    knob_side: str,
    door_state: str,
) -> None:
    x0, y0, x1, y1 = rect
    if door_state == "open":
        draw.rectangle(rect, fill=(34, 29, 25), outline=(54, 38, 28))
        if knob_side == "left":
            panel = [(x1, y0 + 1), (x1 + 5, y0 + 3), (x1 + 5, y1 - 2), (x1, y1)]
            knob = (x1 + 4, (y0 + y1) // 2)
        else:
            panel = [(x0, y0 + 1), (x0 - 5, y0 + 3), (x0 - 5, y1 - 2), (x0, y1)]
            knob = (x0 - 4, (y0 + y1) // 2)
        draw.polygon(panel, fill=(91, 55, 38), outline=(54, 38, 28))
        draw.line((x0 + 2, y0 + 2, x1 - 2, y0 + 2), fill=(62, 48, 40))
        draw.point(knob, fill=(237, 203, 96))
    else:
        draw.rectangle(rect, fill=(91, 55, 38), outline=(54, 38, 28))
        knob_x = x0 + 2 if knob_side == "left" else x1 - 2
        draw.point((knob_x, (y0 + y1) // 2), fill=(237, 203, 96))


def _draw_house_sign(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], *, variant: str) -> None:
    draw.rectangle(rect, fill=(236, 205, 116), outline=(99, 76, 43))
    mark = (74, 76, 96) if variant == "inn" else (126, 75, 51)
    x0, y0, x1, y1 = rect
    draw.rectangle((x0 + 5, y0 + 3, x1 - 5, y1 - 3), fill=mark)


def _draw_house(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    roof_color: tuple[int, int, int],
    body_color: tuple[int, int, int],
    variant: str,
    roof_style: str = "shingle",
    wall_style: str = "stucco",
    door_state: str = "closed",
) -> None:
    """Draw a building variant inside its allocated tile box."""

    if variant == "castle":
        draw_pixel_castle(draw, tile_xywh, stone_rgb=body_color, door_state=door_state)
        return
    if variant == "church":
        draw_pixel_church(draw, tile_xywh, body_rgb=body_color, roof_rgb=roof_color, door_state=door_state)
        return
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    shadow = (x0 + 2, y1 - 4, x1 + 2, y1 + 2)
    draw.rectangle(shadow, fill=(44, 83, 50))
    roof_h = max(12, int((y1 - y0 + 1) * 0.38))
    draw.rectangle((x0 + 2, y0 + roof_h - 2, x1 - 2, y1 - 2), fill=body_color, outline=(92, 74, 58))
    _draw_wall_texture(draw, (x0 + 2, y0 + roof_h - 2, x1 - 2, y1 - 2), body_color=body_color, wall_style=wall_style)
    draw.rectangle((x0, y0 + 6, x1, y0 + roof_h + 4), fill=roof_color, outline=(92, 52, 41))
    _draw_roof_texture(draw, (x0, y0 + 6, x1, y0 + roof_h + 4), roof_color=roof_color, roof_style=roof_style, direction="horizontal")
    door_w = 8
    door_h = 13
    door_x = (x0 + x1) // 2 - door_w // 2
    door_y = y1 - door_h - 2
    _draw_house_door(
        draw,
        (door_x, door_y, door_x + door_w, door_y + door_h),
        knob_side="right",
        door_state=door_state,
    )
    for wx in (x0 + 8, x1 - 15):
        _draw_house_window(draw, (wx, y1 - 22, wx + 8, y1 - 14))
    if variant in {"shop", "inn"}:
        _draw_house_sign(draw, (x0 + 7, y0 + roof_h + 6, x0 + 26, y0 + roof_h + 16), variant=variant)
    if variant == "tower":
        cx = (x0 + x1) // 2
        draw.rectangle((cx - 5, y0, cx + 5, y0 + 10), fill=body_color, outline=(92, 74, 58))
        draw.rectangle((cx - 7, y0 - 3, cx + 7, y0 + 3), fill=roof_color, outline=(92, 52, 41))


def _draw_fountain(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    draw.rectangle((x0 + 3, y0 + 5, x1 - 3, y1 - 3), fill=(142, 146, 145), outline=(76, 84, 92))
    draw.rectangle((x0 + 6, y0 + 8, x1 - 6, y1 - 7), fill=(72, 158, 198), outline=(43, 101, 142))
    draw.rectangle((x0 + 12, y0 + 2, x0 + 20, y0 + 11), fill=(163, 166, 158), outline=(86, 91, 88))
    draw.point((x0 + 11, y0 + 7), fill=(189, 235, 246))
    draw.point((x0 + 21, y0 + 7), fill=(189, 235, 246))


def _draw_sign(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    draw.rectangle((x0 + 7, y0 + 12, x0 + 10, y1 - 2), fill=(90, 60, 35))
    draw.rectangle((x0 + 3, y0 + 5, x1 - 3, y0 + 13), fill=(205, 159, 87), outline=(92, 65, 39))
    draw.line((x0 + 6, y0 + 9, x1 - 6, y0 + 9), fill=(92, 65, 39))


def _draw_crate(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    draw.rectangle((x0, y0, x1, y1), fill=(174, 110, 52), outline=(95, 63, 36))
    draw.line((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=(118, 77, 43))
    draw.line((x0 + 2, y1 - 2, x1 - 2, y0 + 2), fill=(118, 77, 43))


def _draw_fence(draw: ImageDraw.ImageDraw, tile_xywh: TileBox, *, orientation: str) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    wood = (140, 91, 48)
    if orientation == "horizontal":
        draw.line((x0 + 1, y0 + 7, x1 - 1, y0 + 7), fill=wood, width=2)
        draw.line((x0 + 1, y0 + 11, x1 - 1, y0 + 11), fill=wood, width=2)
        for x in range(x0 + 3, x1, 8):
            draw.rectangle((x, y0 + 4, x + 2, y0 + 14), fill=(104, 68, 38))
    else:
        draw.line((x0 + 7, y0 + 1, x0 + 7, y1 - 1), fill=wood, width=2)
        draw.line((x0 + 11, y0 + 1, x0 + 11, y1 - 1), fill=wood, width=2)
        for y in range(y0 + 3, y1, 8):
            draw.rectangle((x0 + 4, y, x0 + 14, y + 2), fill=(104, 68, 38))


def _draw_iron_fence(draw: ImageDraw.ImageDraw, tile_xywh: TileBox, *, orientation: str) -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    iron = (78, 84, 91)
    iron_dark = (42, 45, 49)
    cap = (124, 130, 136)
    if orientation == "horizontal":
        draw.line((x0 + 1, y0 + 8, x1 - 1, y0 + 8), fill=iron_dark)
        draw.line((x0 + 1, y0 + 12, x1 - 1, y0 + 12), fill=iron)
        for x in range(x0 + 2, x1, 5):
            draw.line((x, y0 + 3, x, y0 + 14), fill=iron_dark)
            draw.point((x, y0 + 2), fill=cap)
    else:
        draw.line((x0 + 8, y0 + 1, x0 + 8, y1 - 1), fill=iron_dark)
        draw.line((x0 + 12, y0 + 1, x0 + 12, y1 - 1), fill=iron)
        for y in range(y0 + 2, y1, 5):
            draw.line((x0 + 3, y, x0 + 14, y), fill=iron_dark)
            draw.point((x0 + 2, y), fill=cap)


def _draw_cemetery_gate(draw: ImageDraw.ImageDraw, tile_xywh: TileBox, *, orientation: str = "horizontal") -> None:
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    stone = (124, 128, 124)
    stone_dark = (68, 72, 70)
    iron = (45, 48, 52)
    if orientation == "horizontal":
        draw.rectangle((x0 + 2, y0 + 4, x0 + 4, y1 - 1), fill=stone, outline=stone_dark)
        draw.rectangle((x1 - 4, y0 + 4, x1 - 2, y1 - 1), fill=stone, outline=stone_dark)
        draw.line((x0 + 4, y0 + 8, x1 - 4, y0 + 8), fill=iron)
        draw.line((x0 + 4, y0 + 12, x1 - 4, y0 + 12), fill=iron)
        draw.line((x0 + 8, y0 + 7, x0 + 8, y1 - 2), fill=iron)
        draw.line((x0 + 11, y0 + 7, x0 + 11, y1 - 2), fill=iron)
    else:
        draw.rectangle((x0 + 4, y0 + 2, x1 - 1, y0 + 4), fill=stone, outline=stone_dark)
        draw.rectangle((x0 + 4, y1 - 4, x1 - 1, y1 - 2), fill=stone, outline=stone_dark)
        draw.line((x0 + 8, y0 + 4, x0 + 8, y1 - 4), fill=iron)
        draw.line((x0 + 12, y0 + 4, x0 + 12, y1 - 4), fill=iron)
        draw.line((x0 + 7, y0 + 8, x1 - 2, y0 + 8), fill=iron)
        draw.line((x0 + 7, y0 + 11, x1 - 2, y0 + 11), fill=iron)


def _draw_cemetery_ground(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    x, y, w, h = tile_xywh
    for ty in range(y, y + h):
        for tx in range(x, x + w):
            px = tx * CANONICAL_TILE_PX
            py = ty * CANONICAL_TILE_PX
            shade = (80, 139, 79) if (tx + ty) % 2 else (75, 132, 76)
            draw.rectangle((px, py, px + 15, py + 15), fill=shade)
            if (tx * 3 + ty) % 5 == 0:
                draw.point((px + 5, py + 9), fill=(55, 111, 61))
                draw.point((px + 10, py + 4), fill=(101, 158, 91))


def _draw_entity(draw: ImageDraw.ImageDraw, entity: PixelVillageEntity, *, rng: random.Random) -> None:
    """Dispatch entity rendering from semantic metadata to glyph drawers."""

    kind = str(entity.metadata.get("variant", entity.public_name))
    spec = _shared_object_spec(
        entity_id=entity.entity_id,
        public_name=entity.public_name,
        category=entity.category,
        tile_xywh=entity.tile_xywh,
        bbox_xyxy=entity.bbox_xyxy,
        metadata=entity.metadata,
    )
    if spec is not None:
        render_illustration_object(spec, RenderContext(renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG, draw=draw))
        return
    if entity.category == "building":
        roof = tuple(entity.metadata.get("roof_rgb", (179, 72, 55)))
        body = tuple(entity.metadata.get("body_rgb", (211, 176, 119)))
        _draw_house(
            draw,
            entity.tile_xywh,
            roof_color=roof,
            body_color=body,
            variant=kind,
            roof_style=str(entity.metadata.get("building_roof_style", "shingle")),
            wall_style=str(entity.metadata.get("building_wall_style", "stucco")),
            door_state=str(entity.metadata.get("building_door_state", "closed")),
        )
        if str(entity.metadata.get("theme_id", "")) == "winter":
            draw_pixel_winter_overlay(
                draw,
                entity.tile_xywh,
                target="building",
                snow_rgb=tuple(entity.metadata.get("snow_rgb", (239, 246, 248))),
                shadow_rgb=tuple(entity.metadata.get("snow_shadow_rgb", (178, 202, 215))),
                coverage=float(entity.metadata.get("snow_coverage", 0.5)),
                style=str(entity.metadata.get("snow_style", "patchy")),
            )
    elif kind == "well":
        draw_pixel_well(draw, entity.tile_xywh)
    elif kind == "windmill":
        draw_pixel_windmill(
            draw,
            entity.tile_xywh,
            body_rgb=tuple(entity.metadata.get("body_rgb", (213, 184, 137))),
            roof_rgb=tuple(entity.metadata.get("roof_rgb", (68, 116, 171))),
            blade_pose=str(entity.metadata.get("blade_pose", "plus")),
        )
    elif kind == "fountain":
        _draw_fountain(draw, entity.tile_xywh)
    elif kind == "sign":
        _draw_sign(draw, entity.tile_xywh)
    elif kind == "crate":
        _draw_crate(draw, entity.tile_xywh)
    elif kind == "rock":
        draw_pixel_rock(draw, entity.tile_xywh)
    elif kind == "barrel":
        draw_pixel_barrel(
            draw,
            entity.tile_xywh,
            barrel_rgb=tuple(entity.metadata.get("barrel_rgb", (151, 86, 45))),
            band_rgb=tuple(entity.metadata.get("band_rgb", (82, 59, 42))),
        )
    elif kind == "bench":
        draw_pixel_bench(
            draw,
            entity.tile_xywh,
            orientation=str(entity.metadata.get("orientation", "horizontal")),
            wood_rgb=tuple(entity.metadata.get("wood_rgb", (138, 83, 47))),
        )
    elif kind == "lamp_post":
        draw_pixel_lamp_post(
            draw,
            entity.tile_xywh,
            glow_rgb=tuple(entity.metadata.get("glow_rgb", (248, 210, 104))),
            metal_rgb=tuple(entity.metadata.get("metal_rgb", (61, 70, 75))),
        )
    elif kind == "notice_board":
        draw_pixel_notice_board(
            draw,
            entity.tile_xywh,
            board_rgb=tuple(entity.metadata.get("board_rgb", (177, 119, 62))),
            paper_rgb=tuple(entity.metadata.get("paper_rgb", (236, 214, 154))),
        )
    elif kind == "cart":
        draw_pixel_cart(
            draw,
            entity.tile_xywh,
            facing=str(entity.metadata.get("facing", "right")),
            body_rgb=tuple(entity.metadata.get("body_rgb", (151, 88, 45))),
        )
    elif kind == "market_stall":
        draw_pixel_market_stall(
            draw,
            entity.tile_xywh,
            canopy_rgb=tuple(entity.metadata.get("canopy_rgb", (194, 72, 66))),
            wood_rgb=tuple(entity.metadata.get("wood_rgb", (129, 80, 45))),
            goods_type=str(entity.metadata.get("goods_type", "fruit")),
        )
    elif kind == "wagon":
        cover_rgb_value = entity.metadata.get("cover_rgb")
        draw_pixel_wagon(
            draw,
            entity.tile_xywh,
            facing=str(entity.metadata.get("facing", "right")),
            body_rgb=tuple(entity.metadata.get("body_rgb", (143, 84, 43))),
            cover_rgb=tuple(cover_rgb_value) if cover_rgb_value else None,
        )
    elif kind == "statue":
        draw_pixel_statue(
            draw,
            entity.tile_xywh,
            material_rgb=tuple(entity.metadata.get("material_rgb", (145, 151, 148))),
        )
    elif kind == "gazebo":
        draw_pixel_gazebo(
            draw,
            entity.tile_xywh,
            roof_rgb=tuple(entity.metadata.get("roof_rgb", (154, 82, 74))),
            wood_rgb=tuple(entity.metadata.get("wood_rgb", (123, 83, 48))),
        )
    elif kind == "woodpile":
        draw_pixel_woodpile(
            draw,
            entity.tile_xywh,
            log_rgb=tuple(entity.metadata.get("log_rgb", (135, 82, 45))),
            stack_variant=str(entity.metadata.get("stack_variant", "low")),
        )
    elif kind == "pond":
        draw_pixel_pond(
            draw,
            entity.tile_xywh,
            shape=str(entity.metadata.get("pond_shape", "round")),
            water_rgb=tuple(entity.metadata.get("water_rgb", (55, 133, 188))),
            rim_rgb=tuple(entity.metadata.get("rim_rgb", (74, 129, 74))),
        )
    elif kind == "bridge":
        _draw_bridge(draw, entity.tile_xywh, orientation=str(entity.metadata.get("orientation", "horizontal")))
    elif kind == "fence":
        _draw_fence(draw, entity.tile_xywh, orientation=str(entity.metadata.get("orientation", "horizontal")))
    elif kind == "iron_fence":
        _draw_iron_fence(draw, entity.tile_xywh, orientation=str(entity.metadata.get("orientation", "horizontal")))
    elif kind == "cemetery_gate":
        _draw_cemetery_gate(draw, entity.tile_xywh, orientation=str(entity.metadata.get("orientation", "horizontal")))
    elif kind == "grave_marker":
        draw_pixel_grave_marker(
            draw,
            entity.tile_xywh,
            style=str(entity.metadata.get("marker_style", "rounded")),
            stone_rgb=tuple(entity.metadata.get("stone_rgb", (158, 164, 158))),
            mound_rgb=tuple(entity.metadata.get("mound_rgb", (104, 86, 56))),
            flower_rgb=tuple(entity.metadata.get("flower_rgb")) if entity.metadata.get("flower_rgb") else None,
        )
    elif kind == "dead_tree":
        draw_pixel_dead_tree(
            draw,
            entity.tile_xywh,
            bark_rgb=tuple(entity.metadata.get("bark_rgb", (82, 63, 50))),
        )


def _sample_npc_metadata(rng: random.Random) -> dict[str, Any]:
    gender_id = str(_choose(rng, ["male", "female"]))
    person_variant_id = sample_object_variant_id(rng, "person", support=("adult", "farmer", "worker", "vendor", "soldier"))
    hair_palette = [(54, 38, 28), (84, 49, 31), (116, 75, 35), (49, 44, 44), (154, 112, 50)]
    shirt_palette = [(55, 116, 190), (199, 71, 80), (91, 152, 73), (154, 87, 177), (224, 166, 64)]
    pants_palette = [(47, 66, 102), (73, 67, 60), (86, 96, 65), (56, 83, 119)]
    skin_palette = [(225, 171, 109), (189, 127, 80), (237, 190, 133), (154, 95, 65)]
    return {
        "variant": "npc",
        **variant_visual_metadata("person", person_variant_id, RENDERER_STYLE_TOP_DOWN_PIXEL_RPG),
        "gender_id": gender_id,
        "facing": str(_choose(rng, ["down", "up", "left", "right"])),
        "skin_rgb": list(_choose(rng, skin_palette)),
        "shirt_rgb": list(_choose(rng, shirt_palette)),
        "pants_rgb": list(_choose(rng, pants_palette)),
        "hair_rgb": list(_choose(rng, hair_palette)),
    }


def _sample_tree_metadata(rng: random.Random) -> dict[str, Any]:
    tree_style = str(_choose(rng, PIXEL_TREE_STYLES))
    leaf_palettes = {
        "oak": [(31, 126, 70), (38, 144, 78), (45, 132, 91), (58, 151, 73)],
        "pine": [(22, 105, 86), (29, 118, 82), (34, 96, 78), (39, 126, 91)],
        "maple": [(181, 71, 49), (211, 108, 45), (218, 164, 55), (154, 78, 51)],
        "fruit_tree": [(42, 139, 76), (50, 153, 84), (63, 145, 72), (72, 158, 88)],
    }
    metadata: dict[str, Any] = {
        "variant": "tree",
        **variant_visual_metadata("tree", tree_style, RENDERER_STYLE_TOP_DOWN_PIXEL_RPG),
        "tree_style": tree_style,
        "leaf_rgb": list(_choose(rng, leaf_palettes[tree_style])),
    }
    if tree_style == "fruit_tree":
        metadata["fruit_rgb"] = list(_choose(rng, [(218, 62, 58), (235, 181, 54), (190, 65, 126)]))
    return metadata


def _sample_flower_metadata(rng: random.Random) -> dict[str, Any]:
    return {
        "variant": "flower",
        "flower_rgb": list(_choose(rng, [(236, 82, 111), (239, 201, 72), (202, 91, 190), (238, 126, 62)])),
        "leaf_rgb": list(_choose(rng, [(36, 125, 62), (44, 144, 72), (55, 132, 72)])),
    }


def _sample_building_render_style(rng: random.Random, variant: str) -> dict[str, str]:
    if variant not in {"house", "shop", "inn"}:
        return {"building_roof_style": "landmark", "building_wall_style": "landmark"}
    return {
        "building_roof_style": str(_choose(rng, ["shingle", "wood_plank", "tile"])),
        "building_wall_style": str(_choose(rng, ["stucco", "wood", "stone"])),
    }


def _sample_building_door_state(rng: random.Random) -> str:
    return "open" if rng.random() < 0.35 else "closed"


_WINTER_SHARED_VARIANTS = {"tree", "bench", "lamp_post", "market_stall", "wagon", "gazebo", "pond"}
_WINTER_SCENE_VARIANTS = {"house", "shop", "inn", "tower", "church", "castle"}
_AUTUMN_SHARED_VARIANTS = {"tree", "flower", "bench", "market_stall", "wagon", "gazebo", "pond"}


def _shared_object_spec(
    *,
    entity_id: str,
    public_name: str,
    category: str,
    tile_xywh: TileBox,
    bbox_xyxy: BBox | None,
    metadata: Mapping[str, Any],
) -> IllustrationObjectSpec | None:
    """Convert supported pixel entities into shared object records."""

    variant = str(metadata.get("variant", public_name))
    if variant == "tree":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="tree",
            public_name="tree",
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            variant_id=str(metadata.get("tree_style", "oak")),
            visual_attributes=dict(metadata),
            source_entity_type="pixel_village_entity",
        )
    if variant == "flower":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="flower",
            public_name="flower",
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            visual_attributes=dict(metadata),
            source_entity_type="pixel_village_entity",
        )
    if variant == "npc" or category == "person":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="person",
            public_name=str(public_name),
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            variant_id=str(metadata.get("person_variant_id", "adult")),
            semantic_attributes={"role": str(metadata.get("role", "villager"))},
            visual_attributes=dict(metadata),
            source_entity_type="pixel_village_entity",
        )
    if variant in _WINTER_SHARED_VARIANTS - {"tree"}:
        semantic_attributes: dict[str, Any] = {}
        if variant == "pond":
            semantic_attributes["feature_type"] = "pond"
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type=str(variant),
            public_name=str(public_name),
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            renderer_id="pixel_world_object",
            renderer_variant_id=str(variant),
            semantic_attributes=semantic_attributes,
            visual_attributes=dict(metadata),
            source_entity_type="pixel_village_entity",
        )
    return None


def _winter_metadata_for_entity(
    rng: random.Random,
    entity: PixelVillageEntity,
    *,
    theme: PixelVillageTheme,
) -> dict[str, Any]:
    if theme.theme_id != "winter":
        return {}
    variant = str(entity.metadata.get("variant", entity.public_name))
    if variant not in _WINTER_SHARED_VARIANTS and variant not in _WINTER_SCENE_VARIANTS:
        return {}
    base = float(theme.snow_coverage_base)
    jitter = rng.uniform(-0.14, 0.16)
    coverage = max(0.16, min(0.92, base + jitter))
    style_options = {
        "tree": ["patchy", "ridge", "full_cap"],
        "building": ["left_edge", "ridge", "patchy", "full_cap"],
        "market_stall": ["ridge", "patchy", "full_cap"],
        "gazebo": ["ridge", "full_cap", "patchy"],
        "wagon": ["patchy", "ridge"],
        "bench": ["patchy", "left_edge"],
        "lamp_post": ["patchy"],
        "pond": ["ice_lines", "full_cap", "patchy"],
    }
    target = "building" if entity.category == "building" else variant
    if variant in {"oak", "pine", "maple", "fruit_tree"}:
        target = "tree"
    return {
        "theme_id": "winter",
        "snow_intensity": theme.snow_intensity,
        "snow_coverage": round(float(coverage), 3),
        "snow_style": str(_choose(rng, style_options.get(target, ["patchy"]))),
        "snow_rgb": list(theme.snow_rgb),
        "snow_shadow_rgb": list(theme.snow_shadow_rgb),
    }


def _autumn_metadata_for_entity(
    rng: random.Random,
    entity: PixelVillageEntity,
    *,
    theme: PixelVillageTheme,
) -> dict[str, Any]:
    """Sample autumn overlays without changing semantic entity identity."""

    if theme.theme_id != "autumn":
        return {}
    variant = str(entity.metadata.get("variant", entity.public_name))
    if variant not in _AUTUMN_SHARED_VARIANTS:
        return {}
    base = float(theme.leaf_coverage_base)
    coverage = max(0.08, min(0.48, base + rng.uniform(-0.07, 0.08)))
    style_options = {
        "tree": ["scattered", "ground_edge"],
        "flower": ["dry_heads", "scattered"],
        "market_stall": ["scattered"],
        "gazebo": ["scattered", "ground_edge"],
        "wagon": ["scattered"],
        "bench": ["scattered"],
        "pond": ["floating_leaves", "scattered"],
    }
    target = "tree" if variant == "tree" else variant
    metadata: dict[str, Any] = {
        "theme_id": "autumn",
        "autumn_intensity": theme.autumn_intensity,
        "leaf_coverage": round(float(coverage), 3),
        "leaf_style": str(_choose(rng, style_options.get(target, ["scattered"]))),
        "leaf_overlay_rgb": list(theme.leaf_overlay_rgb),
        "leaf_shadow_rgb": list(theme.leaf_shadow_rgb),
        "leaf_accent_rgb": list(theme.leaf_accent_rgb),
    }
    if variant == "tree":
        tree_style = str(entity.metadata.get("tree_style", "oak"))
        tree_palettes = {
            "early": {
                "oak": [(90, 133, 70), (123, 138, 65), (145, 132, 62)],
                "maple": [(165, 103, 54), (182, 127, 58), (150, 91, 55)],
                "fruit_tree": [(82, 129, 70), (115, 136, 67), (139, 125, 62)],
            },
            "peak": {
                "oak": [(155, 126, 60), (176, 116, 52), (141, 117, 65)],
                "maple": [(183, 90, 54), (204, 127, 50), (168, 76, 57)],
                "fruit_tree": [(136, 125, 64), (159, 116, 55), (114, 130, 70)],
            },
            "late": {
                "oak": [(124, 103, 61), (140, 111, 58), (101, 112, 69)],
                "maple": [(147, 82, 57), (158, 102, 55), (124, 78, 56)],
                "fruit_tree": [(107, 115, 68), (129, 103, 61), (88, 119, 73)],
            },
        }
        if tree_style == "pine":
            metadata["leaf_rgb"] = list(_choose(rng, [(28, 103, 77), (34, 111, 76), (38, 94, 72)]))
            metadata["leaf_coverage"] = round(float(coverage * 0.45), 3)
        else:
            palette = tree_palettes[theme.autumn_intensity].get(tree_style, tree_palettes[theme.autumn_intensity]["oak"])
            metadata["leaf_rgb"] = list(_choose(rng, palette))
            if tree_style == "fruit_tree":
                metadata["fruit_rgb"] = list(_choose(rng, [(192, 75, 55), (205, 144, 54), (154, 71, 92)]))
    elif variant == "flower":
        metadata["flower_rgb"] = list(_choose(rng, [(174, 119, 63), (189, 143, 70), (144, 103, 69)]))
        metadata["leaf_rgb"] = list(_choose(rng, [(76, 105, 57), (91, 108, 58), (82, 96, 55)]))
    return metadata


def _apply_village_theme_to_entities(
    rng: random.Random,
    entities: Sequence[PixelVillageEntity],
    *,
    theme: PixelVillageTheme,
) -> list[PixelVillageEntity]:
    if theme.theme_id not in {"winter", "autumn"}:
        return list(entities)
    themed: list[PixelVillageEntity] = []
    for entity in entities:
        metadata = dict(entity.metadata)
        if theme.theme_id == "winter":
            metadata.update(_winter_metadata_for_entity(rng, entity, theme=theme))
        elif theme.theme_id == "autumn":
            metadata.update(_autumn_metadata_for_entity(rng, entity, theme=theme))
        metadata = _with_shared_object_record(
            entity_id=entity.entity_id,
            public_name=entity.public_name,
            category=entity.category,
            tile_xywh=entity.tile_xywh,
            bbox_xyxy=entity.bbox_xyxy,
            metadata=metadata,
        )
        themed.append(
            PixelVillageEntity(
                entity_id=entity.entity_id,
                public_name=entity.public_name,
                category=entity.category,
                tile_xywh=entity.tile_xywh,
                bbox_xyxy=entity.bbox_xyxy,
                layer=entity.layer,
                metadata=metadata,
            )
        )
    return themed


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
    entities: list[PixelVillageEntity],
    *,
    entity_id: str,
    public_name: str,
    category: str,
    tile_xywh: TileBox,
    layer: str,
    layout: PixelVillageLayout,
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
        PixelVillageEntity(
            entity_id=str(entity_id),
            public_name=str(public_name),
            category=str(category),
            tile_xywh=tile_xywh,
            bbox_xyxy=bbox_xyxy,
            layer=str(layer),
            metadata=resolved_metadata,
        )
    )


def _add_territory(
    territories: list[PixelVillageTerritory],
    *,
    territory_id: str,
    territory_type: str,
    public_name: str,
    tile_xywh: TileBox,
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    territories.append(
        PixelVillageTerritory(
            territory_id=str(territory_id),
            territory_type=str(territory_type),
            public_name=str(public_name),
            tile_xywh=tile_xywh,
            bbox_xyxy=_tile_bbox(tile_xywh, layout=layout, offset_xy=offset_xy),
            metadata=dict(metadata or {}),
        )
    )


def _make_paths(rng: random.Random, *, layout: PixelVillageLayout) -> tuple[set[tuple[int, int]], int, int]:
    street_y = rng.randint(max(5, layout.rows // 3), max(6, layout.rows - layout.rows // 3))
    street_x = rng.randint(max(6, layout.cols // 3), max(7, layout.cols - layout.cols // 3))
    path_tiles: set[tuple[int, int]] = set()
    for x in range(1, layout.cols - 1):
        path_tiles.add((x, street_y))
    for y in range(2, layout.rows - 2):
        path_tiles.add((street_x, y))
    for _ in range(rng.randint(2, 4)):
        start_x = rng.choice([street_x, rng.randint(4, layout.cols - 5)])
        start_y = rng.choice([street_y, rng.randint(4, layout.rows - 5)])
        if rng.random() < 0.5:
            x0, x1 = sorted((start_x, rng.randint(2, layout.cols - 3)))
            for x in range(x0, x1 + 1):
                path_tiles.add((x, start_y))
        else:
            y0, y1 = sorted((start_y, rng.randint(2, layout.rows - 3)))
            for y in range(y0, y1 + 1):
                path_tiles.add((start_x, y))
    return path_tiles, street_x, street_y


def _connect_path_tiles(path_tiles: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    """Return a simple Manhattan path connector between two tiles."""

    sx, sy = start
    ex, ey = end
    out: set[tuple[int, int]] = set()
    step_x = 1 if ex >= sx else -1
    for x in range(sx, ex + step_x, step_x):
        out.add((x, sy))
    step_y = 1 if ey >= sy else -1
    for y in range(sy, ey + step_y, step_y):
        out.add((ex, y))
    out.update(path_tiles.intersection(out))
    return out


def _make_water(
    rng: random.Random,
    path_tiles: set[tuple[int, int]],
    *,
    layout: PixelVillageLayout,
    river_mode: str = "auto",
    river_orientation: str = "auto",
    river_placement: str = "edge",
) -> tuple[set[tuple[int, int]], TileBox | None, str]:
    """Plan river water tiles while keeping paths connected."""

    mode = str(river_mode).strip().lower()
    if mode not in {"auto", "force", "none"}:
        raise ValueError("river_mode must be 'auto', 'force', or 'none'")
    orientation_pref = str(river_orientation).strip().lower()
    if orientation_pref not in {"auto", "vertical", "horizontal"}:
        raise ValueError("river_orientation must be 'auto', 'vertical', or 'horizontal'")
    placement = str(river_placement).strip().lower()
    if placement not in {"edge", "balanced"}:
        raise ValueError("river_placement must be 'edge' or 'balanced'")
    if mode == "none" or (mode == "auto" and rng.random() > 0.55):
        return set(), None, ""
    water_tiles: set[tuple[int, int]] = set()
    orientation = orientation_pref if orientation_pref != "auto" else rng.choice(["vertical", "horizontal"])
    if orientation == "vertical":
        if placement == "balanced":
            river_x = max(3, min(layout.cols - 4, layout.cols // 2 + rng.choice([-1, 0, 1])))
        else:
            river_x = rng.choice([4, 5, layout.cols - 6, layout.cols - 5])
        for y in range(layout.rows):
            water_tiles.add((river_x, y))
            if placement == "balanced" or rng.random() < 0.65:
                water_tiles.add((river_x + 1, y))
        crossing_ys = sorted(y for x, y in path_tiles if x in {river_x - 1, river_x, river_x + 1, river_x + 2})
        bridge_y = crossing_ys[len(crossing_ys) // 2] if crossing_ys else layout.rows // 2
        bridge = (river_x, max(1, bridge_y), 2, 1)
    else:
        if placement == "balanced":
            river_y = max(3, min(layout.rows - 4, layout.rows // 2 + rng.choice([-1, 0, 1])))
        else:
            river_y = rng.choice([4, 5, layout.rows - 6, layout.rows - 5])
        for x in range(layout.cols):
            water_tiles.add((x, river_y))
            if placement == "balanced" or rng.random() < 0.65:
                water_tiles.add((x, river_y + 1))
        crossing_xs = sorted(x for x, y in path_tiles if y in {river_y - 1, river_y, river_y + 1, river_y + 2})
        bridge_x = crossing_xs[len(crossing_xs) // 2] if crossing_xs else layout.cols // 2
        bridge = (max(1, bridge_x), river_y, 1, 2)
    for tile in _rect_tiles(bridge):
        water_tiles.discard(tile)
        path_tiles.add(tile)
    return water_tiles, bridge, orientation


def _water_bounds(water_tiles: set[tuple[int, int]]) -> dict[str, int] | None:
    if not water_tiles:
        return None
    xs = [int(x) for x, _ in water_tiles]
    ys = [int(y) for _, y in water_tiles]
    return {
        "min_x": min(xs),
        "max_x": max(xs),
        "min_y": min(ys),
        "max_y": max(ys),
    }


def _plan_cemetery_territory(
    rng: random.Random,
    *,
    layout: PixelVillageLayout,
    path_tiles: set[tuple[int, int]],
    water_tiles: set[tuple[int, int]],
    cemetery_mode: str,
) -> Mapping[str, Any] | None:
    """Choose a cemetery territory that avoids water and paths."""

    mode = str(cemetery_mode)
    if mode not in {"auto", "force", "none"}:
        raise ValueError("cemetery_mode must be one of: auto, force, none")
    if mode == "none" or (mode == "auto" and rng.random() > 0.32):
        return None

    sizes = [(8, 5), (7, 5), (8, 6)]
    candidates: list[TileBox] = []
    for w, h in sizes:
        candidates.extend(
            [
                (2, 2, w, h),
                (layout.cols - w - 2, 2, w, h),
                (2, layout.rows - h - 2, w, h),
                (layout.cols - w - 2, layout.rows - h - 2, w, h),
                (max(2, layout.cols // 2 - w // 2), 2, w, h),
                (max(2, layout.cols // 2 - w // 2), layout.rows - h - 2, w, h),
            ]
        )
    rng.shuffle(candidates)

    for allow_path_overlap in ([False, True] if mode == "force" else [False]):
        blocked_tiles = set(water_tiles) if allow_path_overlap else set(path_tiles) | set(water_tiles)
        for tile_box in candidates:
            if not _inside_grid(tile_box, layout=layout, inset=1):
                continue
            footprint = set(_rect_tiles(tile_box))
            padded = {
                (x + dx, y + dy)
                for x, y in footprint
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            }
            if footprint & blocked_tiles:
                continue
            if (padded - footprint) & set(water_tiles):
                continue

            x, y, w, h = tile_box
            gate_side = "top" if y > layout.rows // 2 else "bottom"
            gate_x = x + w // 2
            gate_y = y if gate_side == "top" else y + h - 1
            outside_y = gate_y - 1 if gate_side == "top" else gate_y + 1
            if outside_y <= 0 or outside_y >= layout.rows - 1:
                continue
            outside_gate = (gate_x, outside_y)
            nearest_path = min(path_tiles, key=lambda tile: abs(tile[0] - outside_gate[0]) + abs(tile[1] - outside_gate[1]))
            connector = {
                tile
                for tile in _connect_path_tiles(path_tiles, outside_gate, nearest_path)
                if 0 <= tile[0] < layout.cols and 0 <= tile[1] < layout.rows and tile not in water_tiles and tile not in footprint
            }
            return {
                "tile_xywh": tile_box,
                "gate_tile": (gate_x, gate_y, 1, 1),
                "gate_side": gate_side,
                "gate_outside_tile": outside_gate,
                "connector_tiles": sorted(connector),
                "allow_path_overlap": allow_path_overlap,
            }

    if mode == "force":
        raise ValueError("could not place forced cemetery territory")
    return None


def _outside_gate_for_side(tile_xywh: TileBox, side: str) -> tuple[int, int] | None:
    x, y, w, h = tile_xywh
    if side == "top":
        return (x + w // 2, y - 1)
    if side == "bottom":
        return (x + w // 2, y + h)
    if side == "left":
        return (x - 1, y + h // 2)
    if side == "right":
        return (x + w, y + h // 2)
    return None


def _plan_orchard_territory(
    rng: random.Random,
    *,
    layout: PixelVillageLayout,
    path_tiles: set[tuple[int, int]],
    water_tiles: set[tuple[int, int]],
    reserved_tiles: set[tuple[int, int]],
    orchard_mode: str,
) -> PixelOrchardPlan | None:
    """Choose an orchard territory with room for rows and a gate."""

    mode = str(orchard_mode)
    if mode not in {"auto", "force", "none"}:
        raise ValueError("orchard_mode must be one of: auto, force, none")
    if mode == "none" or (mode == "auto" and rng.random() > 0.34):
        return None

    max_w = min(10, layout.cols - 4)
    max_h = min(8, layout.rows - 4)
    sizes = [(w, h) for w in range(7, max_w + 1) for h in range(6, max_h + 1)]
    rng.shuffle(sizes)
    candidates: list[TileBox] = []
    for w, h in sizes[:18]:
        candidates.extend(
            [
                (2, 2, w, h),
                (layout.cols - w - 2, 2, w, h),
                (2, layout.rows - h - 2, w, h),
                (layout.cols - w - 2, layout.rows - h - 2, w, h),
                (max(2, layout.cols // 2 - w // 2), 2, w, h),
                (max(2, layout.cols // 2 - w // 2), layout.rows - h - 2, w, h),
                (rng.randint(2, max(2, layout.cols - w - 2)), rng.randint(2, max(2, layout.rows - h - 2)), w, h),
            ]
        )
    rng.shuffle(candidates)

    for allow_path_overlap in ([False, True] if mode == "force" else [False]):
        blocked_tiles = set(water_tiles) | set(reserved_tiles)
        if not allow_path_overlap:
            blocked_tiles |= set(path_tiles)
        for tile_box in candidates:
            if not _inside_grid(tile_box, layout=layout, inset=1):
                continue
            footprint = set(_rect_tiles(tile_box))
            padded = {
                (x + dx, y + dy)
                for x, y in footprint
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            }
            if footprint & blocked_tiles:
                continue
            if (padded - footprint) & set(water_tiles):
                continue

            valid_sides: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
            for side in ("top", "bottom", "left", "right"):
                outside_gate = _outside_gate_for_side(tile_box, side)
                if outside_gate is None:
                    continue
                ox, oy = outside_gate
                if ox <= 0 or oy <= 0 or ox >= layout.cols - 1 or oy >= layout.rows - 1:
                    continue
                if outside_gate in water_tiles or outside_gate in footprint:
                    continue
                nearest_path = min(path_tiles, key=lambda tile: abs(tile[0] - ox) + abs(tile[1] - oy))
                valid_sides.append((abs(nearest_path[0] - ox) + abs(nearest_path[1] - oy), side, outside_gate, nearest_path))
            if not valid_sides:
                continue
            valid_sides.sort(key=lambda item: (item[0], item[1]))
            _, gate_side, outside_gate, nearest_path = valid_sides[0]
            connector = {
                tile
                for tile in _connect_path_tiles(path_tiles, outside_gate, nearest_path)
                if 0 <= tile[0] < layout.cols
                and 0 <= tile[1] < layout.rows
                and tile not in water_tiles
                and tile not in footprint
                and tile not in reserved_tiles
            }
            return sample_pixel_orchard_plan(
                rng,
                tile_xywh=tile_box,
                territory_id="orchard_0",
                gate_side=gate_side,
                connector_tiles=sorted(connector),
            )

    if mode == "force":
        raise ValueError("could not place forced orchard territory")
    return None


def _building_candidate_positions(path_tiles: set[tuple[int, int]]) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for x, y in path_tiles:
        for dx, dy in ((0, -4), (0, 1), (-4, -1), (1, -1)):
            candidates.append((x + dx, y + dy))
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for pos in candidates:
        if pos not in seen:
            seen.add(pos)
            out.append(pos)
    return out


def _place_buildings(
    rng: random.Random,
    entities: list[PixelVillageEntity],
    *,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    path_tiles: set[tuple[int, int]],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
) -> None:
    """Place countable buildings near paths without overlap."""

    candidates = _building_candidate_positions(path_tiles)
    rng.shuffle(candidates)
    common_variants = ("house", "house", "house", "shop", "inn")
    landmark_variants = ("tower", "church", "castle")
    roof_palette = [(174, 70, 55), (51, 112, 172), (121, 74, 138), (196, 111, 52)]
    body_palette = [(211, 176, 119), (196, 168, 122), (217, 190, 139)]
    target_count = rng.randint(3, 8)
    landmark_budget = 1 if target_count >= 5 and rng.random() < 0.45 else 0
    landmark_slots = set(rng.sample(range(1, target_count), k=landmark_budget)) if landmark_budget else set()
    placed = 0
    for x, y in candidates:
        if placed >= target_count:
            break
        if placed == 0:
            variant = "house"
        elif placed in landmark_slots:
            variant = str(_choose(rng, landmark_variants))
        else:
            variant = str(_choose(rng, common_variants))
        if variant == "castle":
            size = (5, 4)
        elif variant in {"church", "tower"}:
            size = (4, 4) if variant == "church" else (3, 4)
        else:
            size = (4, 3)
        tile_box = (x, y, size[0], size[1])
        if not _is_clear(tile_box, blocked, occupied, layout=layout, pad=1):
            continue
        if variant == "castle":
            roof_rgb = (100, 87, 108)
            body_rgb = _choose(rng, [(142, 145, 140), (157, 154, 145), (130, 137, 142)])
        elif variant == "church":
            roof_rgb = _choose(rng, [(54, 105, 166), (142, 74, 92), (87, 117, 110)])
            body_rgb = _choose(rng, [(216, 191, 145), (205, 181, 138), (222, 202, 164)])
        else:
            roof_rgb = _choose(rng, roof_palette)
            body_rgb = _choose(rng, body_palette)
        building_style = _sample_building_render_style(rng, variant)
        _add_entity(
            entities,
            entity_id=f"building_{placed:02d}",
            public_name="house" if variant == "house" else variant,
            category="building",
            tile_xywh=tile_box,
            layer="building",
            layout=layout,
            offset_xy=offset_xy,
            metadata={
                "variant": variant,
                "building_facing": "front",
                **building_style,
                "building_door_state": _sample_building_door_state(rng),
                "roof_rgb": list(roof_rgb),
                "body_rgb": list(body_rgb),
            },
        )
        occupied.append(tile_box)
        blocked.update(_rect_tiles(tile_box))
        placed += 1


def _place_windmill_landmark(
    rng: random.Random,
    entities: list[PixelVillageEntity],
    *,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    path_tiles: set[tuple[int, int]],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
    windmill_mode: str,
) -> bool:
    """Optionally place one large windmill landmark off paths."""

    mode = str(windmill_mode)
    if mode not in {"auto", "force", "none"}:
        raise ValueError("windmill_mode must be one of: auto, force, none")
    if mode == "none" or (mode == "auto" and rng.random() > 0.30):
        return False

    candidates: list[TileBox] = []
    for x, y in path_tiles:
        candidates.extend(
            [
                (x - 1, y - 4, 3, 4),
                (x - 1, y + 1, 3, 4),
                (x - 4, y - 2, 3, 4),
                (x + 1, y - 2, 3, 4),
            ]
        )
    for x in (2, layout.cols - 5, max(2, layout.cols // 2 - 2)):
        for y in (2, layout.rows - 6, max(2, layout.rows // 2 - 2)):
            candidates.append((x, y, 3, 4))
    rng.shuffle(candidates)

    for tile_box in candidates:
        if _is_clear(tile_box, blocked, occupied, layout=layout, pad=1):
            _add_entity(
                entities,
                entity_id="windmill_00",
                public_name="windmill",
                category="landmark",
                tile_xywh=tile_box,
                layer="building",
                layout=layout,
                offset_xy=offset_xy,
                metadata={
                    "variant": "windmill",
                    "body_rgb": list(_choose(rng, [(213, 184, 137), (202, 174, 128), (219, 195, 150)])),
                    "roof_rgb": list(_choose(rng, [(68, 116, 171), (139, 78, 128), (165, 84, 62)])),
                    "blade_pose": str(_choose(rng, ["plus", "diagonal"])),
                },
            )
            occupied.append(tile_box)
            blocked.update(_rect_tiles(tile_box))
            return True

    if mode == "force":
        raise ValueError("could not place forced windmill landmark")
    return False


def _path_side_candidate_boxes(
    path_tiles: set[tuple[int, int]],
    *,
    size: tuple[int, int],
    layout: PixelVillageLayout,
) -> list[TileBox]:
    w, h = size
    candidates: list[TileBox] = []
    for px, py in path_tiles:
        candidates.extend(
            [
                (px - w // 2, py - h, w, h),
                (px - w // 2, py + 1, w, h),
                (px - w, py - h // 2, w, h),
                (px + 1, py - h // 2, w, h),
            ]
        )
    seen: set[TileBox] = set()
    out: list[TileBox] = []
    for tile_box in candidates:
        if tile_box in seen or not _inside_grid(tile_box, layout=layout, inset=1):
            continue
        seen.add(tile_box)
        out.append(tile_box)
    return out


def _place_large_village_props(
    rng: random.Random,
    entities: list[PixelVillageEntity],
    *,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    path_tiles: set[tuple[int, int]],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
) -> None:
    """Place large non-building props as visual context objects."""

    prop_index = 0
    wood_palette = [(128, 76, 43), (146, 91, 50), (111, 80, 51)]
    roof_palette = [(154, 82, 74), (68, 116, 171), (161, 96, 58)]
    canopy_palette = [(194, 72, 66), (61, 128, 172), (224, 178, 72), (117, 92, 164)]
    stone_palette = [(145, 151, 148), (160, 160, 150), (130, 139, 145)]
    wagon_palette = [(143, 84, 43), (166, 101, 52), (128, 78, 47)]
    cover_palette: list[tuple[int, int, int] | None] = [(214, 192, 138), (163, 111, 83), (92, 128, 157), None]

    def random_candidates(size: tuple[int, int], *, count: int = 220) -> list[TileBox]:
        w, h = size
        return [
            (
                rng.randint(1, layout.cols - w - 1),
                rng.randint(1, layout.rows - h - 1),
                w,
                h,
            )
            for _ in range(int(count))
        ]

    def place_prop(
        *,
        variant: str,
        public_name: str,
        category: str,
        size: tuple[int, int],
        metadata: Mapping[str, Any],
        layer: str = "object",
        prefer_path: bool = True,
    ) -> bool:
        nonlocal prop_index
        path_candidates = _path_side_candidate_boxes(path_tiles, size=size, layout=layout) if prefer_path else []
        candidates = list(path_candidates) + random_candidates(size)
        rng.shuffle(candidates)
        for tile_box in candidates:
            if not _is_clear(tile_box, blocked, occupied, layout=layout, pad=0):
                continue
            _add_entity(
                entities,
                entity_id=f"{variant}_{prop_index:02d}",
                public_name=public_name,
                category=category,
                tile_xywh=tile_box,
                layer=layer,
                layout=layout,
                offset_xy=offset_xy,
                metadata={"variant": variant, **metadata},
            )
            occupied.append(tile_box)
            blocked.update(_rect_tiles(tile_box))
            prop_index += 1
            return True
        return False

    if rng.random() < 0.65:
        place_prop(
            variant="market_stall",
            public_name="market stall",
            category="landmark",
            size=(3, 2),
            metadata={
                "canopy_rgb": list(_choose(rng, canopy_palette)),
                "wood_rgb": list(_choose(rng, wood_palette)),
                "goods_type": str(_choose(rng, ["fruit", "cloth", "crates"])),
                "reference_source_ids": ["kenney_tiny_town", "kenney_roguelike_rpg"],
            },
        )
    if rng.random() < 0.50:
        cover_rgb = _choose(rng, cover_palette)
        wagon_metadata: dict[str, Any] = {
            "facing": str(_choose(rng, ["left", "right"])),
            "body_rgb": list(_choose(rng, wagon_palette)),
            "cover_rgb": list(cover_rgb) if cover_rgb is not None else None,
            "reference_source_ids": ["kenney_rpg_urban_pack", "kenney_roguelike_rpg"],
        }
        place_prop(
            variant="wagon",
            public_name="wagon",
            category="object",
            size=(3, 2),
            metadata=wagon_metadata,
        )
    if rng.random() < 0.40:
        place_prop(
            variant="statue",
            public_name="statue",
            category="landmark",
            size=(2, 2),
            metadata={
                "material_rgb": list(_choose(rng, stone_palette)),
                "reference_source_ids": ["kenney_tiny_town", "kenney_roguelike_rpg"],
            },
        )
    if rng.random() < 0.30:
        place_prop(
            variant="gazebo",
            public_name="gazebo",
            category="landmark",
            size=(3, 3),
            metadata={
                "roof_rgb": list(_choose(rng, roof_palette)),
                "wood_rgb": list(_choose(rng, wood_palette)),
                "reference_source_ids": ["kenney_tiny_town", "kenney_roguelike_rpg"],
            },
            prefer_path=False,
        )
    if rng.random() < 0.45:
        woodpile_size = _choose(rng, [(2, 1), (2, 2)])
        place_prop(
            variant="woodpile",
            public_name="woodpile",
            category="object",
            size=woodpile_size,
            metadata={
                "log_rgb": list(_choose(rng, wood_palette)),
                "stack_variant": "tall" if woodpile_size[1] > 1 else "low",
                "reference_source_ids": ["kenney_roguelike_rpg", "kenney_tiny_town"],
            },
            prefer_path=False,
        )
    for _ in range(2 if rng.random() < 0.48 else 1):
        pond_size = _choose(rng, [(3, 2), (4, 2), (4, 3)])
        place_prop(
            variant="pond",
            public_name="pond",
            category="landmark",
            size=pond_size,
            metadata={
                "pond_shape": str(_choose(rng, ["round", "long", "kidney"])),
                "water_rgb": list(_choose(rng, [(55, 133, 188), (48, 124, 176), (70, 147, 190)])),
                "rim_rgb": list(_choose(rng, [(74, 129, 74), (88, 139, 78), (93, 126, 70)])),
                "reference_source_ids": ["kenney_tiny_town"],
            },
            layer="terrain",
            prefer_path=False,
        )


def _place_path_side_props(
    rng: random.Random,
    entities: list[PixelVillageEntity],
    *,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    path_tiles: set[tuple[int, int]],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
) -> None:
    """Place small props adjacent to paths without blocking roads."""

    prop_index = 0

    def place_prop(
        *,
        variant: str,
        public_name: str,
        category: str,
        size: tuple[int, int],
        count: int,
        metadata: Mapping[str, Any],
    ) -> None:
        nonlocal prop_index
        candidates = _path_side_candidate_boxes(path_tiles, size=size, layout=layout)
        rng.shuffle(candidates)
        placed = 0
        for tile_box in candidates:
            if placed >= count:
                break
            if not _is_clear(tile_box, blocked, occupied, layout=layout, pad=0):
                continue
            _add_entity(
                entities,
                entity_id=f"{variant}_{prop_index:02d}",
                public_name=public_name,
                category=category,
                tile_xywh=tile_box,
                layer="object",
                layout=layout,
                offset_xy=offset_xy,
                metadata={"variant": variant, **metadata},
            )
            occupied.append(tile_box)
            blocked.update(_rect_tiles(tile_box))
            prop_index += 1
            placed += 1

    wood_palette = [(128, 76, 43), (146, 91, 50), (111, 80, 51)]
    barrel_palette = [(151, 86, 45), (164, 100, 53), (126, 78, 49)]
    metal_palette = [(58, 67, 72), (72, 77, 78), (52, 62, 67)]
    paper_palette = [(236, 214, 154), (224, 202, 142), (239, 222, 174)]
    cart_palette = [(151, 88, 45), (128, 78, 47), (166, 101, 52)]

    place_prop(
        variant="barrel",
        public_name="barrel",
        category="object",
        size=(1, 1),
        count=rng.randint(1, 3),
        metadata={
            "barrel_rgb": list(_choose(rng, barrel_palette)),
            "band_rgb": list(_choose(rng, [(82, 59, 42), (70, 61, 51), (93, 66, 43)])),
        },
    )
    bench_orientation = str(_choose(rng, ["horizontal", "vertical"]))
    place_prop(
        variant="bench",
        public_name="bench",
        category="object",
        size=(2, 1) if bench_orientation == "horizontal" else (1, 2),
        count=rng.randint(1, 2),
        metadata={"orientation": bench_orientation, "wood_rgb": list(_choose(rng, wood_palette))},
    )
    place_prop(
        variant="lamp_post",
        public_name="lamp post",
        category="landmark",
        size=(1, 2),
        count=rng.randint(2, 3),
        metadata={
            "glow_rgb": list(_choose(rng, [(248, 210, 104), (238, 197, 92), (255, 224, 126)])),
            "metal_rgb": list(_choose(rng, metal_palette)),
        },
    )
    place_prop(
        variant="notice_board",
        public_name="notice board",
        category="landmark",
        size=(2, 1),
        count=rng.randint(0, 1),
        metadata={
            "board_rgb": list(_choose(rng, wood_palette)),
            "paper_rgb": list(_choose(rng, paper_palette)),
        },
    )
    place_prop(
        variant="cart",
        public_name="cart",
        category="object",
        size=(2, 1),
        count=rng.randint(0, 1),
        metadata={"facing": str(_choose(rng, ["left", "right"])), "body_rgb": list(_choose(rng, cart_palette))},
    )


def _place_small_entities(
    rng: random.Random,
    entities: list[PixelVillageEntity],
    *,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
    npc_avoid_tiles: set[tuple[int, int]] | None = None,
) -> None:
    """Place countable people and small props in free cells."""

    avoid_tiles = set(npc_avoid_tiles or set())
    specs = [
        ("well", "landmark", "well", (2, 2), 1),
        ("fountain", "landmark", "fountain", (2, 2), 1),
        ("sign", "landmark", "sign", (1, 1), 1),
        ("crate", "object", "crate", (1, 1), 3),
        ("tree", "plant", "tree", (1, 2), 7),
        ("flower", "plant", "flower", (1, 1), 4),
        ("rock", "object", "rock", (1, 1), 2),
        ("npc", "person", "villager", (1, 1), 4),
    ]
    index = 0
    for variant, category, public_name, size, count in specs:
        for _ in range(count + rng.randint(0, 1)):
            for _attempt in range(180):
                tile_box = (
                    rng.randint(1, layout.cols - size[0] - 1),
                    rng.randint(1, layout.rows - size[1] - 1),
                    size[0],
                    size[1],
                )
                if not _is_clear(tile_box, blocked, occupied, layout=layout, pad=0):
                    continue
                if variant == "npc" and set(_rect_tiles(tile_box)) & avoid_tiles:
                    continue
                _add_entity(
                    entities,
                    entity_id=f"{variant}_{index:02d}",
                    public_name=public_name,
                    category=category,
                    tile_xywh=tile_box,
                    layer="object" if category != "person" else "actor",
                    layout=layout,
                    offset_xy=offset_xy,
                    metadata=(
                        _sample_npc_metadata(rng)
                        if variant == "npc"
                        else _sample_tree_metadata(rng)
                        if variant == "tree"
                        else _sample_flower_metadata(rng)
                        if variant == "flower"
                        else {"variant": variant}
                    ),
                )
                occupied.append(tile_box)
                if category == "landmark" or variant == "tree":
                    blocked.update(_rect_tiles(tile_box))
                index += 1
                break


def _place_path_people(
    rng: random.Random,
    entities: list[PixelVillageEntity],
    *,
    path_tiles: set[tuple[int, int]],
    water_tiles: set[tuple[int, int]],
    occupied: list[TileBox],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
    target_count: int,
) -> int:
    """Place optional people directly on path tiles for path-scoped tasks."""

    target = max(0, int(target_count))
    if target <= 0:
        return 0
    occupied_tiles = set()
    for tile_box in occupied:
        occupied_tiles.update(_rect_tiles(tile_box))
    candidates = [
        (x, y, 1, 1)
        for x, y in sorted(path_tiles)
        if (x, y) not in water_tiles
        and (x, y) not in occupied_tiles
        and 0 <= int(x) < int(layout.cols)
        and 0 <= int(y) < int(layout.rows)
    ]
    rng.shuffle(candidates)
    placed = 0
    for tile_box in candidates:
        if placed >= target:
            break
        metadata = dict(_sample_npc_metadata(rng))
        metadata["path_role"] = "on_path"
        _add_entity(
            entities,
            entity_id=f"path_npc_{placed:02d}",
            public_name="villager",
            category="person",
            tile_xywh=tile_box,
            layer="actor",
            layout=layout,
            offset_xy=offset_xy,
            metadata=metadata,
        )
        occupied.append(tile_box)
        occupied_tiles.update(_rect_tiles(tile_box))
        placed += 1
    return int(placed)


def _place_fences(
    rng: random.Random,
    entities: list[PixelVillageEntity],
    *,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
) -> None:
    if rng.random() > 0.18:
        return
    x = rng.randint(3, layout.cols - 9)
    y = rng.randint(3, layout.rows - 7)
    w = rng.randint(4, 7)
    h = rng.randint(3, 5)
    fence_tiles: list[tuple[TileBox, str]] = []
    for fx in range(x, x + w):
        fence_tiles.append(((fx, y, 1, 1), "horizontal"))
        fence_tiles.append(((fx, y + h - 1, 1, 1), "horizontal"))
    for fy in range(y + 1, y + h - 1):
        fence_tiles.append(((x, fy, 1, 1), "vertical"))
        fence_tiles.append(((x + w - 1, fy, 1, 1), "vertical"))
    if any(not _is_clear(tile_box, blocked, occupied, layout=layout, pad=0) for tile_box, _ in fence_tiles):
        return
    for index, (tile_box, orientation) in enumerate(fence_tiles):
        _add_entity(
            entities,
            entity_id=f"fence_{index:02d}",
            public_name="fence",
            category="barrier",
            tile_xywh=tile_box,
            layer="barrier",
            layout=layout,
            offset_xy=offset_xy,
            metadata={"variant": "fence", "orientation": orientation},
        )
        occupied.append(tile_box)
        blocked.update(_rect_tiles(tile_box))


def _place_cemetery_territory(
    rng: random.Random,
    draw: ImageDraw.ImageDraw,
    entities: list[PixelVillageEntity],
    territories: list[PixelVillageTerritory],
    *,
    cemetery_plan: Mapping[str, Any] | None,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
) -> None:
    """Render cemetery territory and register its countable entities."""

    if cemetery_plan is None:
        return

    tile_box = tuple(cemetery_plan["tile_xywh"])  # type: ignore[arg-type]
    x, y, w, h = tile_box
    gate_tile = tuple(cemetery_plan["gate_tile"])  # type: ignore[arg-type]
    gate_side = str(cemetery_plan.get("gate_side", "bottom"))
    gate_orientation = "horizontal" if gate_side in {"top", "bottom"} else "vertical"

    _draw_cemetery_ground(draw, tile_box)

    fence_tiles: list[tuple[TileBox, str]] = []
    for fx in range(x, x + w):
        top = (fx, y, 1, 1)
        bottom = (fx, y + h - 1, 1, 1)
        if top != gate_tile:
            fence_tiles.append((top, "horizontal"))
        if bottom != gate_tile:
            fence_tiles.append((bottom, "horizontal"))
    for fy in range(y + 1, y + h - 1):
        fence_tiles.append(((x, fy, 1, 1), "vertical"))
        fence_tiles.append(((x + w - 1, fy, 1, 1), "vertical"))

    for fence_box, orientation in fence_tiles:
        _draw_iron_fence(draw, fence_box, orientation=orientation)
        occupied.append(fence_box)
        blocked.update(_rect_tiles(fence_box))

    _add_entity(
        entities,
        entity_id="cemetery_gate_00",
        public_name="cemetery gate",
        category="territory_feature",
        tile_xywh=gate_tile,
        layer="barrier",
        layout=layout,
        offset_xy=offset_xy,
        metadata={"variant": "cemetery_gate", "orientation": gate_orientation, "territory_id": "cemetery_0"},
    )
    occupied.append(gate_tile)
    blocked.update(_rect_tiles(gate_tile))

    inner_tiles = [(tx, ty) for ty in range(y + 1, y + h - 1) for tx in range(x + 1, x + w - 1)]
    rng.shuffle(inner_tiles)
    used_inner: set[tuple[int, int]] = set()
    grave_target = min(len(inner_tiles), rng.randint(4, 9))
    stone_palette = [(150, 157, 151), (166, 166, 157), (138, 145, 144), (174, 170, 158)]
    mound_palette = [(101, 80, 55), (91, 74, 54), (112, 92, 58)]
    flower_palette = [(220, 74, 90), (232, 190, 70), (185, 89, 180)]
    marker_count = 0
    for tx, ty in inner_tiles:
        if marker_count >= grave_target:
            break
        if (tx, ty) in used_inner:
            continue
        tile = (tx, ty, 1, 1)
        _add_entity(
            entities,
            entity_id=f"grave_marker_{marker_count:02d}",
            public_name="grave marker",
            category="grave_marker",
            tile_xywh=tile,
            layer="object",
            layout=layout,
            offset_xy=offset_xy,
            metadata={
                "variant": "grave_marker",
                "marker_style": str(_choose(rng, PIXEL_GRAVE_MARKER_STYLES)),
                "stone_rgb": list(_choose(rng, stone_palette)),
                "mound_rgb": list(_choose(rng, mound_palette)),
                "flower_rgb": list(_choose(rng, flower_palette)) if rng.random() < 0.3 else None,
                "territory_id": "cemetery_0",
            },
        )
        occupied.append(tile)
        used_inner.add((tx, ty))
        marker_count += 1

    dead_tree_count = 0
    if rng.random() < 0.65:
        for tx, ty in inner_tiles:
            tree_box = (tx, ty, 1, 2)
            tree_tiles = set(_rect_tiles(tree_box))
            if ty + 1 >= y + h - 1 or tree_tiles & used_inner:
                continue
            _add_entity(
                entities,
                entity_id="cemetery_dead_tree_00",
                public_name="dead tree",
                category="plant",
                tile_xywh=tree_box,
                layer="object",
                layout=layout,
                offset_xy=offset_xy,
                metadata={
                    "variant": "dead_tree",
                    "bark_rgb": list(_choose(rng, [(82, 63, 50), (96, 72, 52), (74, 60, 50)])),
                    "territory_id": "cemetery_0",
                },
            )
            occupied.append(tree_box)
            blocked.update(_rect_tiles(tree_box))
            used_inner.update(tree_tiles)
            dead_tree_count = 1
            break

    blocked.update(_rect_tiles(tile_box))
    _add_territory(
        territories,
        territory_id="cemetery_0",
        territory_type="cemetery",
        public_name="cemetery",
        tile_xywh=tile_box,
        layout=layout,
        offset_xy=offset_xy,
        metadata={
            "gate_tile": list(gate_tile),
            "gate_side": gate_side,
            "gate_outside_tile": list(cemetery_plan.get("gate_outside_tile", [])),
            "connector_tiles": [list(tile) for tile in cemetery_plan.get("connector_tiles", [])],
            "fence_tile_count": len(fence_tiles),
            "grave_marker_count": marker_count,
            "dead_tree_count": dead_tree_count,
        },
    )


def _place_orchard_territory(
    draw: ImageDraw.ImageDraw,
    entities: list[PixelVillageEntity],
    territories: list[PixelVillageTerritory],
    *,
    orchard_plan: PixelOrchardPlan | None,
    blocked: set[tuple[int, int]],
    occupied: list[TileBox],
    layout: PixelVillageLayout,
    offset_xy: tuple[int, int],
) -> None:
    """Render orchard territory and register its countable trees."""

    if orchard_plan is None:
        return

    draw_pixel_orchard_ground(draw, orchard_plan)
    draw_pixel_orchard_boundary(draw, orchard_plan)

    for tree in orchard_plan.trees:
        _add_entity(
            entities,
            entity_id=tree.tree_id,
            public_name="tree",
            category="plant",
            tile_xywh=tree.tile_xywh,
            layer="object",
            layout=layout,
            offset_xy=offset_xy,
            metadata=tree.metadata(territory_id=orchard_plan.territory_id),
        )
        occupied.append(tree.tile_xywh)
        blocked.update(_rect_tiles(tree.tile_xywh))

    blocked.update(_rect_tiles(orchard_plan.tile_xywh))
    _add_territory(
        territories,
        territory_id=orchard_plan.territory_id,
        territory_type="orchard",
        public_name="orchard",
        tile_xywh=orchard_plan.tile_xywh,
        layout=layout,
        offset_xy=offset_xy,
        metadata=orchard_plan.metadata(),
    )


def _sample_layout(
    rng: random.Random,
    *,
    width: int,
    height: int,
    tile_px: int,
    grid_cols: int | None,
    grid_rows: int | None,
) -> PixelVillageLayout:
    """Choose a full-bleed tile grid so the village itself fills the output canvas."""

    requested_tile_px = max(20, min(DEFAULT_DISPLAY_TILE_PX, int(tile_px)))
    image_w = int(width)
    image_h = int(height)

    if grid_cols is None and grid_rows is None:
        candidates: list[PixelVillageLayout] = []
        for candidate_tile_px in range(20, DEFAULT_DISPLAY_TILE_PX + 1):
            if image_w % candidate_tile_px != 0 or image_h % candidate_tile_px != 0:
                continue
            candidate = PixelVillageLayout(
                cols=image_w // candidate_tile_px,
                rows=image_h // candidate_tile_px,
                tile_px=candidate_tile_px,
            )
            if 18 <= candidate.cols <= 34 and 15 <= candidate.rows <= 30:
                candidates.append(candidate)
        if candidates:
            return min(
                candidates,
                key=lambda candidate: (abs(candidate.tile_px - requested_tile_px), candidate.cols, candidate.rows),
            )
        raise ValueError(f"no full-bleed pixel village grid fits {width}x{height}")

    if grid_cols is not None and grid_rows is not None:
        cols = int(grid_cols)
        rows = int(grid_rows)
        if cols < 18 or rows < 15:
            raise ValueError("pixel village grid must be at least 18x15")
        if image_w % cols != 0 or image_h % rows != 0 or image_w // cols != image_h // rows:
            raise ValueError(
                f"explicit pixel village grid {cols}x{rows} cannot fill {width}x{height} with square tiles"
            )
        layout = PixelVillageLayout(cols=cols, rows=rows, tile_px=image_w // cols)
        if not (20 <= layout.tile_px <= DEFAULT_DISPLAY_TILE_PX):
            raise ValueError("explicit pixel village grid implies unsupported tile size")
        return layout

    explicit_cols = int(grid_cols) if grid_cols is not None else None
    explicit_rows = int(grid_rows) if grid_rows is not None else None
    candidates = []
    for candidate_tile_px in range(20, DEFAULT_DISPLAY_TILE_PX + 1):
        if image_w % candidate_tile_px != 0 or image_h % candidate_tile_px != 0:
            continue
        candidate = PixelVillageLayout(
            cols=explicit_cols if explicit_cols is not None else image_w // candidate_tile_px,
            rows=explicit_rows if explicit_rows is not None else image_h // candidate_tile_px,
            tile_px=candidate_tile_px,
        )
        if (
            candidate.display_width_px == image_w
            and candidate.display_height_px == image_h
            and 18 <= candidate.cols <= 34
            and 15 <= candidate.rows <= 30
        ):
            candidates.append(candidate)
    if candidates:
        return min(candidates, key=lambda candidate: abs(candidate.tile_px - requested_tile_px))
    raise ValueError(f"explicit pixel village grid cannot fill {width}x{height}")


def render_pixel_village_map(
    seed: int,
    *,
    width: int = 960,
    height: int = 720,
    tile_px: int = DEFAULT_DISPLAY_TILE_PX,
    grid_cols: int | None = None,
    grid_rows: int | None = None,
    cemetery_mode: str = "auto",
    orchard_mode: str = "auto",
    windmill_mode: str = "auto",
    theme_mode: str = "temperate",
    river_mode: str = "auto",
    river_orientation: str = "auto",
    river_placement: str = "edge",
    path_person_count: int = 0,
    background_person_path_clearance: int = 0,
) -> PixelVillageScene:
    """Render one deterministic old-school pixel RPG village map."""

    rng = random.Random(int(seed))
    theme = _sample_village_theme(random.Random(int(seed) + 99173), str(theme_mode))
    layout = _sample_layout(
        rng,
        width=int(width),
        height=int(height),
        tile_px=int(tile_px),
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )
    offset_xy = ((int(width) - layout.display_width_px) // 2, (int(height) - layout.display_height_px) // 2)

    base = Image.new("RGB", (layout.canonical_width_px, layout.canonical_height_px), (76, 158, 82))
    draw = ImageDraw.Draw(base)
    grid = [["grass" for _ in range(layout.cols)] for _ in range(layout.rows)]
    path_tiles, street_x, street_y = _make_paths(rng, layout=layout)
    water_tiles, bridge_box, water_orientation = _make_water(
        rng,
        path_tiles,
        layout=layout,
        river_mode=str(river_mode),
        river_orientation=str(river_orientation),
        river_placement=str(river_placement),
    )
    cemetery_plan = _plan_cemetery_territory(
        rng,
        layout=layout,
        path_tiles=path_tiles,
        water_tiles=water_tiles,
        cemetery_mode=str(cemetery_mode),
    )
    planned_territory_tiles: set[tuple[int, int]] = set()
    if cemetery_plan is not None:
        planned_territory_tiles.update(_rect_tiles(tuple(cemetery_plan["tile_xywh"])))  # type: ignore[arg-type]
    orchard_plan = _plan_orchard_territory(
        rng,
        layout=layout,
        path_tiles=path_tiles,
        water_tiles=water_tiles,
        reserved_tiles=planned_territory_tiles,
        orchard_mode=str(orchard_mode),
    )
    if orchard_plan is not None:
        planned_territory_tiles.update(_rect_tiles(orchard_plan.tile_xywh))
    if cemetery_plan is not None:
        path_tiles.update(tuple(tile) for tile in cemetery_plan.get("connector_tiles", []))
    if orchard_plan is not None:
        path_tiles.update(orchard_plan.connector_tiles)
    for _ in range(rng.randint(8, 16)):
        x = rng.randrange(1, layout.cols - 1)
        y = rng.randrange(1, layout.rows - 1)
        if (x, y) not in path_tiles and (x, y) not in water_tiles and (x, y) not in planned_territory_tiles:
            grid[y][x] = "flower"
    _mark_tiles(grid, sorted(path_tiles), "path")
    _mark_tiles(grid, sorted(water_tiles), "water")
    tile_render_seed = int(seed) + 314159
    _draw_tile_layer(draw, grid, rng=random.Random(tile_render_seed), theme=theme)

    blocked = set(path_tiles) | set(water_tiles)
    occupied: list[TileBox] = []
    entities: list[PixelVillageEntity] = []
    territories: list[PixelVillageTerritory] = []

    if bridge_box is not None:
        orientation = "horizontal" if water_orientation == "vertical" else "vertical"
        _add_entity(
            entities,
            entity_id="bridge_00",
            public_name="bridge",
            category="path_feature",
            tile_xywh=bridge_box,
            layer="path",
            layout=layout,
            offset_xy=offset_xy,
            metadata={"variant": "bridge", "orientation": orientation},
        )
        occupied.append(bridge_box)

    _place_cemetery_territory(
        rng,
        draw,
        entities,
        territories,
        cemetery_plan=cemetery_plan,
        blocked=blocked,
        occupied=occupied,
        layout=layout,
        offset_xy=offset_xy,
    )
    _place_orchard_territory(
        draw,
        entities,
        territories,
        orchard_plan=orchard_plan,
        blocked=blocked,
        occupied=occupied,
        layout=layout,
        offset_xy=offset_xy,
    )
    windmill_present = _place_windmill_landmark(
        rng,
        entities,
        blocked=blocked,
        occupied=occupied,
        path_tiles=path_tiles,
        layout=layout,
        offset_xy=offset_xy,
        windmill_mode=str(windmill_mode),
    )
    _place_buildings(rng, entities, blocked=blocked, occupied=occupied, path_tiles=path_tiles, layout=layout, offset_xy=offset_xy)
    _place_fences(rng, entities, blocked=blocked, occupied=occupied, layout=layout, offset_xy=offset_xy)
    _place_large_village_props(
        rng,
        entities,
        blocked=blocked,
        occupied=occupied,
        path_tiles=path_tiles,
        layout=layout,
        offset_xy=offset_xy,
    )
    _place_path_side_props(
        rng,
        entities,
        blocked=blocked,
        occupied=occupied,
        path_tiles=path_tiles,
        layout=layout,
        offset_xy=offset_xy,
    )
    path_person_placed_count = _place_path_people(
        rng,
        entities,
        path_tiles=path_tiles,
        water_tiles=water_tiles,
        occupied=occupied,
        layout=layout,
        offset_xy=offset_xy,
        target_count=int(path_person_count),
    )
    npc_avoid_tiles = (
        _expanded_tiles(path_tiles, radius=int(background_person_path_clearance), layout=layout)
        if int(background_person_path_clearance) > 0
        else set()
    )
    _place_small_entities(
        rng,
        entities,
        blocked=blocked,
        occupied=occupied,
        layout=layout,
        offset_xy=offset_xy,
        npc_avoid_tiles=npc_avoid_tiles,
    )
    entities = _apply_village_theme_to_entities(rng, entities, theme=theme)

    for entity in sorted(entities, key=lambda item: (item.tile_xywh[1], item.tile_xywh[0], item.layer)):
        _draw_entity(draw, entity, rng=rng)

    image = base.resize((layout.display_width_px, layout.display_height_px), Image.Resampling.NEAREST)
    canvas = image

    entities_sorted = tuple(sorted(entities, key=lambda item: item.entity_id))
    territories_sorted = tuple(sorted(territories, key=lambda item: item.territory_id))
    cemetery_territory = next((territory for territory in territories_sorted if territory.territory_type == "cemetery"), None)
    orchard_territory = next((territory for territory in territories_sorted if territory.territory_type == "orchard"), None)
    river_bounds = _water_bounds(water_tiles)
    trace = {
        "renderer_id": "pixel_village_map_v0",
        "seed": int(seed),
        "inspiration_sources": [
            {
                "name": "Kenney Tiny Town",
                "url": "https://kenney.nl/assets/tiny-town",
                "notes": "16x16 CC0 RPG/town pixel tile pack used as visual reference only.",
            },
            {
                "name": "Kenney Roguelike/RPG pack",
                "url": "https://kenney.nl/assets/roguelike-rpg-pack",
                "notes": "16x16 CC0 RPG/town/furniture pixel pack used as visual reference only.",
            },
            {
                "name": "Kenney RPG Urban Pack",
                "url": "https://opengameart.org/content/rpg-urban-pack",
                "notes": "CC0 urban prop/vehicle pixel pack used as visual reference only.",
            },
            {
                "name": "Kenney support/license page",
                "url": "https://kenney.nl/support",
                "notes": "License reference for Kenney assets used as visual references only.",
            },
            {
                "name": "Diogo Vernier RPG House Tileset",
                "url": "https://diogo-vernier.itch.io/rpg-house-tileset",
                "notes": "16x16 RPG house tileset used as a visual reference for modular roof directions and wall styles only.",
            },
        ],
        "uses_external_sprites": False,
        "theme_mode": str(theme_mode),
        "theme_id": theme.theme_id,
        "snow_intensity": theme.snow_intensity,
        "snow_coverage_base": round(float(theme.snow_coverage_base), 3),
        "autumn_intensity": theme.autumn_intensity,
        "leaf_coverage_base": round(float(theme.leaf_coverage_base), 3),
        "theme_palette": {
            "snow_rgb": list(theme.snow_rgb),
            "snow_shadow_rgb": list(theme.snow_shadow_rgb),
            "leaf_overlay_rgb": list(theme.leaf_overlay_rgb),
            "leaf_shadow_rgb": list(theme.leaf_shadow_rgb),
            "leaf_accent_rgb": list(theme.leaf_accent_rgb),
            "canvas_rgb": list(theme.canvas_rgb),
        },
        "tile_render_seed": int(tile_render_seed),
        "grid_cols": layout.cols,
        "grid_rows": layout.rows,
        "canonical_tile_px": CANONICAL_TILE_PX,
        "tile_px": layout.tile_px,
        "map_size_px": [layout.display_width_px, layout.display_height_px],
        "map_offset_xy": list(offset_xy),
        "uses_outer_canvas_background": False,
        "uses_pixel_frame_border": False,
        "main_street_tile": {"x": int(street_x), "y": int(street_y)},
        "path_tiles": [[int(x), int(y)] for x, y in sorted(path_tiles)],
        "water_tiles": [[int(x), int(y)] for x, y in sorted(water_tiles)],
        "river_mode": str(river_mode),
        "river_orientation": str(water_orientation),
        "river_placement": str(river_placement),
        "river_present": bool(water_tiles),
        "river_bounds": dict(river_bounds) if river_bounds is not None else None,
        "bridge_box": [int(v) for v in bridge_box] if bridge_box is not None else None,
        "entity_count": len(entities_sorted),
        "entities": [entity.as_dict() for entity in entities_sorted],
        "territory_count": len(territories_sorted),
        "territories": [territory.as_dict() for territory in territories_sorted],
        "territory_type_counts": {
            territory_type: sum(1 for territory in territories_sorted if territory.territory_type == territory_type)
            for territory_type in sorted({territory.territory_type for territory in territories_sorted})
        },
        "cemetery_mode": str(cemetery_mode),
        "cemetery_present": cemetery_territory is not None,
        "cemetery_grave_marker_count": (
            int(cemetery_territory.metadata.get("grave_marker_count", 0)) if cemetery_territory is not None else 0
        ),
        "orchard_mode": str(orchard_mode),
        "orchard_present": orchard_territory is not None,
        "orchard_tree_count": int(orchard_territory.metadata.get("tree_count", 0)) if orchard_territory is not None else 0,
        "windmill_mode": str(windmill_mode),
        "windmill_present": bool(windmill_present),
        "path_person_requested_count": int(path_person_count),
        "path_person_placed_count": int(path_person_placed_count),
        "background_person_path_clearance": max(0, int(background_person_path_clearance)),
        "category_counts": {
            category: sum(1 for entity in entities_sorted if entity.category == category)
            for category in sorted({entity.category for entity in entities_sorted})
        },
        "public_name_counts": {
            name: sum(1 for entity in entities_sorted if entity.public_name == name)
            for name in sorted({entity.public_name for entity in entities_sorted})
        },
    }
    return PixelVillageScene(image=canvas, entities=entities_sorted, territories=territories_sorted, trace=trace)


def draw_pixel_village_debug_overlay(scene: PixelVillageScene) -> Image.Image:
    """Return a debug overlay with semantic bboxes and names."""

    image = scene.image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    palette = {
        "building": (209, 72, 54, 235),
        "landmark": (80, 102, 204, 235),
        "plant": (34, 144, 73, 235),
        "object": (211, 132, 45, 235),
        "person": (142, 78, 184, 235),
        "path_feature": (86, 101, 120, 235),
        "barrier": (111, 80, 44, 235),
        "grave_marker": (82, 86, 91, 235),
        "territory_feature": (32, 83, 154, 235),
    }
    try:
        from trace_tasks.tasks.shared.text_rendering import load_font

        font = load_font(11, bold=True)
    except Exception:  # pragma: no cover - review helper fallback
        font = None
    for territory in scene.territories:
        box = [round(v) for v in territory.bbox_xyxy]
        draw.rectangle(box, outline=(32, 83, 154, 245), width=3)
        label = territory.public_name
        text_bbox = draw.textbbox((box[0] + 3, box[1] + 3), label, font=font)
        bg = (text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1)
        draw.rectangle(bg, fill=(235, 241, 255, 220))
        draw.text((box[0] + 3, box[1] + 3), label, fill=(32, 83, 154, 245), font=font)
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
    "PixelVillageEntity",
    "PixelVillageLayout",
    "PixelVillageScene",
    "PixelVillageTerritory",
    "draw_pixel_village_debug_overlay",
    "render_pixel_village_map",
]
