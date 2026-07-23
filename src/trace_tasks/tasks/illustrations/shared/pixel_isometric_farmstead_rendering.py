"""Procedural isometric pixel RPG farmstead renderer prototype.

This module is intentionally not a public task. It is a renderer prototype for
reviewing whether explicitly isometric farm scenes can support future
illustration tasks that reason about levels, terraces, transitions, regions,
and farm objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.illustrations.shared.object_variants import (
    RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
    sample_object_variant_id,
    variant_visual_metadata,
)
from trace_tasks.tasks.illustrations.shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    object_record_for_spec,
    render_illustration_object,
)
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import PIXEL_TREE_STYLES


BBox = tuple[float, float, float, float]
CanonicalBBox = tuple[float, float, float, float]
IsoPoint = tuple[float, float]
IsoPolygon = tuple[IsoPoint, ...]
RGB = tuple[int, int, int]
TileBox = tuple[int, int, int, int]

DEFAULT_GRID_COLS = 14
DEFAULT_GRID_ROWS = 12
DEFAULT_SCALE = 2
CANONICAL_ISO_TILE_W_PX = 32
CANONICAL_ISO_TILE_H_PX = 16
CANONICAL_LEVEL_PX = 12
DEFAULT_DISPLAY_ISO_TILE_W_PX = CANONICAL_ISO_TILE_W_PX * DEFAULT_SCALE
DEFAULT_DISPLAY_ISO_TILE_H_PX = CANONICAL_ISO_TILE_H_PX * DEFAULT_SCALE
DEFAULT_DISPLAY_LEVEL_PX = CANONICAL_LEVEL_PX * DEFAULT_SCALE

# Backward-compatible constants for review scripts/tests that inspect prototype
# modules directly.
GRID_COLS = DEFAULT_GRID_COLS
GRID_ROWS = DEFAULT_GRID_ROWS


@dataclass(frozen=True)
class IsoFarmLayout:
    """Grid, projection, and display geometry for one isometric farmstead."""

    cols: int
    rows: int
    scale: int
    width_px: int
    height_px: int
    origin_xy: tuple[int, int]

    @property
    def canonical_width_px(self) -> int:
        return max(1, int(self.width_px) // max(1, int(self.scale)))

    @property
    def canonical_height_px(self) -> int:
        return max(1, int(self.height_px) // max(1, int(self.scale)))

    @property
    def display_scale_x(self) -> float:
        return float(self.width_px) / float(self.canonical_width_px)

    @property
    def display_scale_y(self) -> float:
        return float(self.height_px) / float(self.canonical_height_px)

    @property
    def display_iso_tile_w_px(self) -> int:
        return int(round(CANONICAL_ISO_TILE_W_PX * self.display_scale_x))

    @property
    def display_iso_tile_h_px(self) -> int:
        return int(round(CANONICAL_ISO_TILE_H_PX * self.display_scale_y))

    @property
    def display_level_px(self) -> int:
        return int(round(CANONICAL_LEVEL_PX * self.display_scale_y))


@dataclass(frozen=True)
class IsoFarmTile:
    """One projected terrain tile in the isometric farmstead."""

    col: int
    row: int
    level: int
    terrain: str
    polygon_xy: IsoPolygon
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "col": int(self.col),
            "row": int(self.row),
            "level": int(self.level),
            "terrain": str(self.terrain),
            "polygon": [[round(x, 3), round(y, 3)] for x, y in self.polygon_xy],
            "bbox": [round(v, 3) for v in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoFarmRegion:
    """One semantic region in the isometric farmstead."""

    region_id: str
    public_name: str
    region_type: str
    tile_xywh: TileBox
    level: int
    polygon_xy: IsoPolygon
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "region_id": str(self.region_id),
            "public_name": str(self.public_name),
            "region_type": str(self.region_type),
            "tile_xywh": [int(v) for v in self.tile_xywh],
            "level": int(self.level),
            "polygon": [[round(x, 3), round(y, 3)] for x, y in self.polygon_xy],
            "bbox": [round(v, 3) for v in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoFarmTransition:
    """One traversable transition between farmstead levels."""

    transition_id: str
    transition_type: str
    lower_tile_xy: tuple[int, int]
    upper_tile_xy: tuple[int, int]
    lower_level: int
    upper_level: int
    polygon_xy: IsoPolygon
    bbox_xyxy: BBox
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "transition_id": str(self.transition_id),
            "transition_type": str(self.transition_type),
            "lower_tile_xy": [int(v) for v in self.lower_tile_xy],
            "upper_tile_xy": [int(v) for v in self.upper_tile_xy],
            "lower_level": int(self.lower_level),
            "upper_level": int(self.upper_level),
            "polygon": [[round(x, 3), round(y, 3)] for x, y in self.polygon_xy],
            "bbox": [round(v, 3) for v in self.bbox_xyxy],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class IsoFarmEntity:
    """One semantic entity in the generated isometric farmstead."""

    entity_id: str
    public_name: str
    category: str
    tile_xywh: TileBox
    level: int
    bbox_xyxy: BBox
    footprint_polygon_xy: IsoPolygon
    anchor_screen_xy: IsoPoint
    layer: str
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        metadata = {str(key): value for key, value in self.metadata.items() if str(key) != "object_record"}
        payload = {
            "entity_id": str(self.entity_id),
            "public_name": str(self.public_name),
            "category": str(self.category),
            "tile_xywh": [int(v) for v in self.tile_xywh],
            "level": int(self.level),
            "bbox": [round(v, 3) for v in self.bbox_xyxy],
            "footprint_polygon": [[round(x, 3), round(y, 3)] for x, y in self.footprint_polygon_xy],
            "anchor_screen_xy": [round(v, 3) for v in self.anchor_screen_xy],
            "center_tile_xy": [
                round(self.tile_xywh[0] + (self.tile_xywh[2] - 1) * 0.5, 3),
                round(self.tile_xywh[1] + (self.tile_xywh[3] - 1) * 0.5, 3),
            ],
            "layer": str(self.layer),
            "metadata": metadata,
        }
        if "object_record" in self.metadata:
            payload["object_record"] = self.metadata["object_record"]
        return payload


@dataclass(frozen=True)
class IsoFarmScene:
    """Rendered isometric farmstead scene plus trace metadata."""

    image: Image.Image
    tiles: tuple[IsoFarmTile, ...]
    entities: tuple[IsoFarmEntity, ...]
    regions: tuple[IsoFarmRegion, ...]
    transitions: tuple[IsoFarmTransition, ...]
    trace: Mapping[str, Any]


def _choose(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[int(rng.randrange(len(values)))]


def _shade(color: RGB, delta: int) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def _sample_building_door_state(rng: random.Random) -> str:
    return "open" if rng.random() < 0.35 else "closed"


def _sample_layout(
    rng: random.Random,
    *,
    width: int,
    height: int,
    scale: int,
    grid_cols: int | None,
    grid_rows: int | None,
) -> IsoFarmLayout:
    resolved_scale = max(1, min(4, int(scale)))
    cols = int(grid_cols) if grid_cols is not None else rng.randint(13, 15)
    rows = int(grid_rows) if grid_rows is not None else rng.randint(11, 13)
    if cols < 12 or rows < 10:
        raise ValueError("isometric farmstead grid must be at least 12x10")
    if cols > 16 or rows > 14:
        raise ValueError("isometric farmstead prototype supports grids up to 16x14")

    canonical_w = max(1, int(width) // resolved_scale)
    canonical_h = max(1, int(height) // resolved_scale)
    x_min = -((rows - 1) * (CANONICAL_ISO_TILE_W_PX // 2)) - (CANONICAL_ISO_TILE_W_PX // 2)
    x_max = ((cols - 1) * (CANONICAL_ISO_TILE_W_PX // 2)) + (CANONICAL_ISO_TILE_W_PX // 2)
    origin_x = int(round((canonical_w - (x_min + x_max)) * 0.5))

    object_top_extra = 64
    max_elevation_depth = 2
    projected_y_min = -(CANONICAL_ISO_TILE_H_PX // 2) - max_elevation_depth * CANONICAL_LEVEL_PX - object_top_extra
    projected_y_max = ((cols + rows - 2) * (CANONICAL_ISO_TILE_H_PX // 2)) + (
        CANONICAL_ISO_TILE_H_PX // 2
    ) + max_elevation_depth * CANONICAL_LEVEL_PX
    content_h = projected_y_max - projected_y_min
    if content_h > canonical_h - 10:
        raise ValueError(f"isometric farmstead grid {cols}x{rows} does not fit in {width}x{height}")
    origin_y = int(round((canonical_h - content_h) * 0.5 - projected_y_min))

    return IsoFarmLayout(
        cols=cols,
        rows=rows,
        scale=resolved_scale,
        width_px=int(width),
        height_px=int(height),
        origin_xy=(origin_x, origin_y),
    )


def _project_c(layout: IsoFarmLayout, col: float, row: float, level: int) -> IsoPoint:
    ox, oy = layout.origin_xy
    return (
        float(ox + (float(col) - float(row)) * (CANONICAL_ISO_TILE_W_PX * 0.5)),
        float(oy + (float(col) + float(row)) * (CANONICAL_ISO_TILE_H_PX * 0.5) - int(level) * CANONICAL_LEVEL_PX),
    )


def _display_point(point: IsoPoint, *, layout: IsoFarmLayout) -> IsoPoint:
    return (float(point[0]) * layout.display_scale_x, float(point[1]) * layout.display_scale_y)


def _display_polygon(points: Sequence[IsoPoint], *, layout: IsoFarmLayout) -> IsoPolygon:
    return tuple(_display_point(point, layout=layout) for point in points)


def _bbox_from_points(points: Sequence[IsoPoint]) -> CanonicalBBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _display_bbox(bbox: CanonicalBBox, *, layout: IsoFarmLayout) -> BBox:
    x0, y0, x1, y1 = bbox
    p0 = _display_point((x0, y0), layout=layout)
    p1 = _display_point((x1, y1), layout=layout)
    return (p0[0], p0[1], p1[0], p1[1])


def _tile_vertices_c(layout: IsoFarmLayout, col: int, row: int, level: int) -> tuple[IsoPoint, IsoPoint, IsoPoint, IsoPoint]:
    cx, cy = _project_c(layout, col, row, level)
    half_w = CANONICAL_ISO_TILE_W_PX * 0.5
    half_h = CANONICAL_ISO_TILE_H_PX * 0.5
    return (
        (cx, cy - half_h),
        (cx + half_w, cy),
        (cx, cy + half_h),
        (cx - half_w, cy),
    )


def _footprint_polygon_c(layout: IsoFarmLayout, tile_xywh: TileBox, level: int) -> IsoPolygon:
    x, y, w, h = tile_xywh
    top = _tile_vertices_c(layout, x, y, level)[0]
    right = _tile_vertices_c(layout, x + w - 1, y, level)[1]
    bottom = _tile_vertices_c(layout, x + w - 1, y + h - 1, level)[2]
    left = _tile_vertices_c(layout, x, y + h - 1, level)[3]
    return (top, right, bottom, left)


def _footprint_polygon_display(layout: IsoFarmLayout, tile_xywh: TileBox, level: int) -> IsoPolygon:
    return _display_polygon(_footprint_polygon_c(layout, tile_xywh, level), layout=layout)


def _region_bbox_display(layout: IsoFarmLayout, tile_xywh: TileBox, level: int) -> BBox:
    return _display_bbox(_bbox_from_points(_footprint_polygon_c(layout, tile_xywh, level)), layout=layout)


def _polygon_to_ints(points: Sequence[IsoPoint]) -> list[tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in points]


def _lerp_point(a: IsoPoint, b: IsoPoint, t: float) -> IsoPoint:
    return (float(a[0]) + (float(b[0]) - float(a[0])) * float(t), float(a[1]) + (float(b[1]) - float(a[1])) * float(t))


def _tile_bounds_for_entity_c(layout: IsoFarmLayout, tile_xywh: TileBox, level: int, *, kind: str) -> CanonicalBBox:
    x, y, w, h = tile_xywh
    center = _project_c(layout, x + (w - 1) * 0.5, y + (h - 1) * 0.5, level)
    polygon = _footprint_polygon_c(layout, tile_xywh, level)
    base_bbox = _bbox_from_points(polygon)
    if kind == "barn":
        return (base_bbox[0] - 8, base_bbox[1] - 66, base_bbox[2] + 8, base_bbox[3] + 2)
    if kind == "coop":
        return (base_bbox[0] - 5, base_bbox[1] - 42, base_bbox[2] + 5, base_bbox[3] + 2)
    if kind == "tree":
        bottom_y = center[1] + 6
        return (center[0] - 8, bottom_y - 32, center[0] + 8, bottom_y)
    if kind == "flower":
        bottom_y = center[1] + 6
        return (center[0] - 8, bottom_y - 16, center[0] + 8, bottom_y)
    if kind == "person":
        bottom_y = center[1] + 6
        return (center[0] - 8, bottom_y - 16, center[0] + 8, bottom_y)
    if kind == "animal":
        sprite_w = 16
        bottom_y = center[1] + 6
        return (center[0] - sprite_w * 0.5, bottom_y - 16, center[0] + sprite_w * 0.5, bottom_y)
    if kind == "cow":
        bottom_y = center[1] + 6
        return (center[0] - 16, bottom_y - 16, center[0] + 16, bottom_y)
    if kind in {"hay_bale", "crate"}:
        return (center[0] - 10, center[1] - 12, center[0] + 10, center[1] + 6)
    if kind == "trough":
        return (center[0] - 14, center[1] - 9, center[0] + 14, center[1] + 5)
    return (base_bbox[0], base_bbox[1] - 12, base_bbox[2], base_bbox[3])


def _entity_canonical_bbox(layout: IsoFarmLayout, tile_xywh: TileBox, level: int, metadata: Mapping[str, Any]) -> CanonicalBBox:
    variant = str(metadata.get("variant", "object"))
    if variant == "animal":
        animal_type = str(metadata.get("animal_type", "sheep"))
        return _tile_bounds_for_entity_c(layout, tile_xywh, level, kind="cow" if animal_type == "cow" else "animal")
    return _tile_bounds_for_entity_c(layout, tile_xywh, level, kind=variant)


def _entity_anchor_display(layout: IsoFarmLayout, tile_xywh: TileBox, level: int) -> IsoPoint:
    x, y, w, h = tile_xywh
    return _display_point(_project_c(layout, x + (w - 1) * 0.5, y + (h - 1) * 0.5, level), layout=layout)


def _rect_tiles(tile_xywh: TileBox) -> list[tuple[int, int]]:
    x, y, w, h = tile_xywh
    return [(tx, ty) for ty in range(y, y + h) for tx in range(x, x + w)]


def _rects_intersect(a: TileBox, b: TileBox, *, pad: int = 0) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax - pad < bx + bw and ax + aw + pad > bx and ay - pad < by + bh and ay + ah + pad > by


def _connect_path(path_tiles: set[tuple[int, int]], start: tuple[int, int], end: tuple[int, int]) -> None:
    sx, sy = start
    ex, ey = end
    step_x = 1 if ex >= sx else -1
    for x in range(sx, ex + step_x, step_x):
        path_tiles.add((x, sy))
    step_y = 1 if ey >= sy else -1
    for y in range(sy, ey + step_y, step_y):
        path_tiles.add((ex, y))


def _upper_terrace_box(layout: IsoFarmLayout) -> TileBox:
    terrace_h = 4
    terrace_w = max(8, layout.cols - 4)
    terrace_w = min(terrace_w, layout.cols - 3)
    x0 = max(1, (layout.cols - terrace_w) // 2)
    return (x0, 1, terrace_w, terrace_h)


def _inside(tile: tuple[int, int], tile_xywh: TileBox) -> bool:
    x, y = tile
    tx, ty, tw, th = tile_xywh
    return tx <= x < tx + tw and ty <= y < ty + th


def _make_level_grid(
    layout: IsoFarmLayout,
    upper_terrace: TileBox,
    *,
    upper_level: int,
) -> dict[tuple[int, int], int]:
    return {
        (col, row): int(upper_level) if _inside((col, row), upper_terrace) else 0
        for row in range(layout.rows)
        for col in range(layout.cols)
    }


def _resolve_elevation_depth(rng: random.Random, elevation_depth: int | None) -> int:
    if elevation_depth is None:
        return int(_choose(rng, [1, 2]))
    resolved = int(elevation_depth)
    if resolved not in {1, 2}:
        raise ValueError("isometric farmstead elevation_depth must be 1 or 2")
    return resolved


def _make_scene_plan(
    rng: random.Random,
    *,
    layout: IsoFarmLayout,
    elevation_depth: int | None,
) -> dict[str, Any]:
    upper_level = _resolve_elevation_depth(rng, elevation_depth)
    upper_terrace = _upper_terrace_box(layout)
    ux, uy, uw, uh = upper_terrace
    lower_start_y = uy + uh
    ramp_col = ux + uw // 2
    stair_col = min(ux + uw - 2, ramp_col + 2)
    barn = (ux + uw - 4, uy, 3, 3)
    upper_pen = (ux + 1, uy + 1, 3, 2)
    lower_pen = (layout.cols - 5, min(layout.rows - 4, lower_start_y + 2), 4, 3)
    coop = (1, min(layout.rows - 3, lower_start_y + 1), 2, 2)
    lower_field = (1, lower_start_y, layout.cols - 2, layout.rows - lower_start_y - 1)
    ramp_tile = (ramp_col, lower_start_y)
    upper_ramp_tile = (ramp_col, lower_start_y - 1)
    stair_tile = (stair_col, lower_start_y)
    upper_stair_tile = (stair_col, lower_start_y - 1)

    path_tiles: set[tuple[int, int]] = set()
    barn_door = (barn[0] + 1, barn[1] + barn[3])
    _connect_path(path_tiles, barn_door, upper_ramp_tile)
    _connect_path(path_tiles, upper_ramp_tile, upper_stair_tile)
    _connect_path(path_tiles, ramp_tile, (lower_pen[0] + 1, lower_pen[1]))
    _connect_path(path_tiles, stair_tile, (coop[0] + 1, coop[1] + coop[3] - 1))

    crop_tiles: set[tuple[int, int]] = set()
    crop_x0 = 2
    crop_w = max(4, min(6, lower_pen[0] - crop_x0 - 1))
    crop_y0 = min(layout.rows - 4, lower_start_y + 2)
    for row_offset in range(4):
        row = crop_y0 + row_offset
        if row >= layout.rows - 1:
            continue
        for col in range(crop_x0, min(layout.cols - 2, crop_x0 + crop_w)):
            if row_offset % 2 == 0 and (col, row) not in path_tiles:
                crop_tiles.add((col, row))

    pen_ground_tiles = set(_rect_tiles(upper_pen) + _rect_tiles(lower_pen))
    terrain: dict[tuple[int, int], str] = {}
    for row in range(layout.rows):
        for col in range(layout.cols):
            tile = (col, row)
            value = "grass"
            if tile in pen_ground_tiles:
                value = "pen_ground"
            if tile in crop_tiles:
                value = "crop"
            if tile in path_tiles:
                value = "dirt"
            if tile == ramp_tile:
                value = "ramp"
            if tile == stair_tile:
                value = "stair"
            terrain[tile] = value

    return {
        "upper_terrace": upper_terrace,
        "upper_level": int(upper_level),
        "elevation_depth": int(upper_level),
        "lower_field": lower_field,
        "upper_pen": upper_pen,
        "lower_pen": lower_pen,
        "barn": barn,
        "coop": coop,
        "ramp_tile": ramp_tile,
        "upper_ramp_tile": upper_ramp_tile,
        "stair_tile": stair_tile,
        "upper_stair_tile": upper_stair_tile,
        "path_tiles": path_tiles,
        "crop_tiles": crop_tiles,
        "pen_ground_tiles": pen_ground_tiles,
        "terrain": terrain,
        "gate_edges": {
            "upper_pen": {"side": "bottom", "index": 1, "tile_xy": [upper_pen[0] + 1, upper_pen[1] + upper_pen[3] - 1]},
            "lower_pen": {"side": "top", "index": 1, "tile_xy": [lower_pen[0] + 1, lower_pen[1]]},
        },
        "accent_variant": int(rng.randrange(4)),
    }


def _terrain_colors(terrain: str, level: int) -> tuple[RGB, RGB, RGB]:
    grass = (88, 164, 82) if int(level) == 0 else (104, 178, 88)
    if terrain == "dirt":
        fill = (181, 139, 82)
    elif terrain == "crop":
        fill = (138, 98, 58)
    elif terrain == "pen_ground":
        fill = (155, 135, 82)
    elif terrain == "ramp":
        fill = (188, 145, 86)
    elif terrain == "stair":
        fill = (151, 144, 122)
    else:
        fill = grass
    return fill, _shade(fill, -42), _shade(fill, 34)


def _draw_iso_tile(
    draw: ImageDraw.ImageDraw,
    layout: IsoFarmLayout,
    col: int,
    row: int,
    *,
    level: int,
    terrain: str,
    rng: random.Random,
) -> None:
    vertices = _tile_vertices_c(layout, col, row, level)
    fill, dark, light = _terrain_colors(terrain, level)
    draw.polygon(_polygon_to_ints(vertices), fill=fill)
    top, right, bottom, left = vertices
    draw.line(_polygon_to_ints((top, right)), fill=light)
    draw.line(_polygon_to_ints((right, bottom)), fill=_shade(dark, 8))
    draw.line(_polygon_to_ints((bottom, left)), fill=dark)
    draw.line(_polygon_to_ints((left, top)), fill=_shade(dark, -4))

    cx, cy = _project_c(layout, col, row, level)
    if terrain == "crop":
        leaf = _choose(rng, [(55, 143, 62), (76, 160, 69), (96, 154, 61)])
        for dx in (-7, 0, 7):
            draw.line((int(cx + dx), int(cy - 2), int(cx + dx), int(cy + 5)), fill=_shade(leaf, -35))
            draw.point((int(cx + dx - 2), int(cy + 1)), fill=leaf)
            draw.point((int(cx + dx + 2), int(cy + 2)), fill=leaf)
    elif terrain == "dirt":
        draw.point((int(cx - 5), int(cy + 1)), fill=_shade(fill, -35))
        draw.point((int(cx + 6), int(cy - 1)), fill=_shade(fill, 25))
        draw.line((int(cx - 8), int(cy + 3), int(cx - 2), int(cy + 5)), fill=_shade(fill, -28))
    elif terrain == "pen_ground":
        for dx, dy in ((-6, -1), (2, 2), (7, -2)):
            draw.point((int(cx + dx), int(cy + dy)), fill=_shade(fill, 38))
    elif terrain == "grass" and rng.random() < 0.28:
        speck = (57, 127, 60) if rng.random() < 0.55 else (125, 197, 94)
        draw.point((int(cx + rng.randrange(-9, 10)), int(cy + rng.randrange(-3, 5))), fill=speck)


def _draw_level_faces(
    draw: ImageDraw.ImageDraw,
    layout: IsoFarmLayout,
    *,
    level_grid: Mapping[tuple[int, int], int],
    open_edges: set[tuple[int, int, str]] | None = None,
) -> list[dict[str, Any]]:
    face_records: list[dict[str, Any]] = []
    open_edge_set = set(open_edges or set())
    for row in range(layout.rows):
        for col in range(layout.cols):
            level = int(level_grid[(col, row)])
            if level <= 0:
                continue
            vertices = _tile_vertices_c(layout, col, row, level)
            _, right, bottom, left = vertices
            for side_name, edge, neighbor in (
                ("east", (right, bottom), (col + 1, row)),
                ("south", (bottom, left), (col, row + 1)),
            ):
                if (col, row, side_name) in open_edge_set:
                    continue
                neighbor_level = int(level_grid.get(neighbor, 0))
                drop = max(0, level - neighbor_level) * CANONICAL_LEVEL_PX
                if drop <= 0:
                    continue
                p0, p1 = edge
                face = (p0, p1, (p1[0], p1[1] + drop), (p0[0], p0[1] + drop))
                base = (111, 91, 62) if side_name == "east" else (132, 103, 64)
                draw.polygon(_polygon_to_ints(face), fill=base)
                draw.line(_polygon_to_ints((p0, p1)), fill=(190, 153, 91))
                draw.line(
                    (int(round(p0[0])), int(round(p0[1] + drop)), int(round(p1[0])), int(round(p1[1] + drop))),
                    fill=(67, 60, 46),
                )
                draw.line((int(round(p0[0])), int(round(p0[1])), int(round(p0[0])), int(round(p0[1] + drop))), fill=(79, 67, 48))
                draw.line((int(round(p1[0])), int(round(p1[1])), int(round(p1[0])), int(round(p1[1] + drop))), fill=(159, 126, 76))
                for offset in range(4, int(drop), 4):
                    draw.line(
                        (
                            int(round(p0[0])),
                            int(round(p0[1] + offset)),
                            int(round(p1[0])),
                            int(round(p1[1] + offset)),
                        ),
                        fill=_shade(base, -24 if offset % 8 == 0 else 14),
                    )
                for t in (0.33, 0.66):
                    seam_top = _lerp_point(p0, p1, t)
                    seam_height = max(4, int(drop) - (2 if t < 0.5 else 4))
                    draw.line(
                        (
                            int(round(seam_top[0])),
                            int(round(seam_top[1] + 2)),
                            int(round(seam_top[0])),
                            int(round(seam_top[1] + seam_height)),
                        ),
                        fill=_shade(base, -31),
                    )
                face_records.append(
                    {
                        "tile_xy": [int(col), int(row)],
                        "side": side_name,
                        "from_level": int(level),
                        "to_level": int(neighbor_level),
                        "drop_px": int(drop * layout.scale),
                        "screen_polygon": [
                            [round(x, 3), round(y, 3)]
                            for x, y in _display_polygon(face, layout=layout)
                        ],
                    }
                )
    return face_records


def _draw_transition_surface(
    draw: ImageDraw.ImageDraw,
    layout: IsoFarmLayout,
    *,
    lower_tile_xy: tuple[int, int],
    upper_tile_xy: tuple[int, int],
    lower_level: int,
    upper_level: int,
    transition_type: str,
) -> IsoPolygon:
    lower_vertices = _tile_vertices_c(layout, lower_tile_xy[0], lower_tile_xy[1], int(lower_level))
    upper_vertices = _tile_vertices_c(layout, upper_tile_xy[0], upper_tile_xy[1], int(upper_level))
    lower_top = lower_vertices[0]
    lower_right = lower_vertices[1]
    upper_left = upper_vertices[3]
    upper_bottom = upper_vertices[2]
    polygon: IsoPolygon = (upper_left, upper_bottom, lower_right, lower_top)
    if transition_type == "ramp":
        draw.polygon(_polygon_to_ints(polygon), fill=(191, 145, 86), outline=(102, 77, 51))
        draw.line(_polygon_to_ints((upper_left, upper_bottom)), fill=(224, 178, 106))
        draw.line(_polygon_to_ints((lower_top, lower_right)), fill=(112, 82, 52))
        draw.line(_polygon_to_ints((upper_left, lower_top)), fill=(101, 73, 48), width=2)
        draw.line(_polygon_to_ints((upper_bottom, lower_right)), fill=(146, 103, 59), width=2)
        for t in (0.35, 0.65):
            top = _lerp_point(upper_left, upper_bottom, t)
            bottom = _lerp_point(lower_top, lower_right, t)
            draw.line(
                (int(round(top[0])), int(round(top[1] + 1)), int(round(bottom[0])), int(round(bottom[1] - 1))),
                fill=(139, 99, 62),
            )
        surface_marks = (0.22, 0.44, 0.66, 0.84) if int(upper_level) - int(lower_level) > 1 else (0.28, 0.56, 0.82)
        for t in surface_marks:
            left = _lerp_point(upper_left, lower_top, t)
            right = _lerp_point(upper_bottom, lower_right, t)
            draw.line(_polygon_to_ints((left, right)), fill=(207, 157, 93))
    else:
        draw.polygon(_polygon_to_ints(polygon), fill=(118, 107, 89), outline=(82, 75, 64))
        step_count = 4 + max(0, int(upper_level) - int(lower_level) - 1) * 3
        for index in range(step_count):
            t0 = index / step_count
            t1 = (index + 1) / step_count
            left0 = _lerp_point(upper_left, lower_top, t0)
            right0 = _lerp_point(upper_bottom, lower_right, t0)
            left1 = _lerp_point(upper_left, lower_top, t1)
            right1 = _lerp_point(upper_bottom, lower_right, t1)
            tread = (left0, right0, right1, left1)
            fill = (178, 166, 139) if index % 2 == 0 else (166, 153, 128)
            draw.polygon(_polygon_to_ints(tread), fill=fill)
            draw.line(_polygon_to_ints((left0, right0)), fill=(228, 215, 181))
            draw.line(_polygon_to_ints((left1, right1)), fill=(86, 78, 65), width=1)
        draw.line(_polygon_to_ints((upper_left, lower_top)), fill=(77, 70, 58), width=2)
        draw.line(_polygon_to_ints((upper_bottom, lower_right)), fill=(127, 114, 89), width=2)
        draw.line(_polygon_to_ints((upper_left, upper_bottom)), fill=(229, 216, 180))
        draw.line(_polygon_to_ints((lower_top, lower_right)), fill=(72, 66, 57))
    return polygon


def _edge_points(layout: IsoFarmLayout, col: int, row: int, level: int, side: str) -> tuple[IsoPoint, IsoPoint]:
    top, right, bottom, left = _tile_vertices_c(layout, col, row, level)
    if side == "top":
        return top, right
    if side == "right":
        return right, bottom
    if side == "bottom":
        return left, bottom
    return top, left


def _draw_fence_segment(
    draw: ImageDraw.ImageDraw,
    p0: IsoPoint,
    p1: IsoPoint,
    *,
    is_gate: bool,
) -> None:
    wood = (135, 88, 50)
    light = (203, 143, 76)
    dark = (74, 51, 35)
    if is_gate:
        mid = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
        draw.line(
            (
                int(round(p0[0])),
                int(round(p0[1] - 2)),
                int(round(mid[0] - 3)),
                int(round(mid[1] - 4)),
            ),
            fill=light,
            width=1,
        )
        draw.line(
            (
                int(round(mid[0] + 3)),
                int(round(mid[1] - 4)),
                int(round(p1[0])),
                int(round(p1[1] - 2)),
            ),
            fill=light,
            width=1,
        )
        for px, py in (p0, p1):
            draw.line((int(px), int(py - 8), int(px), int(py + 1)), fill=dark, width=2)
        return
    draw.line((int(p0[0]), int(p0[1] - 5), int(p1[0]), int(p1[1] - 5)), fill=dark, width=2)
    draw.line((int(p0[0]), int(p0[1] - 8), int(p1[0]), int(p1[1] - 8)), fill=wood, width=2)
    draw.line((int(p0[0]), int(p0[1] - 9), int(p1[0]), int(p1[1] - 9)), fill=light, width=1)
    for px, py in (p0, p1):
        draw.line((int(px), int(py - 11), int(px), int(py + 1)), fill=dark, width=2)
        draw.point((int(px), int(py - 10)), fill=light)


def _draw_pen_fence(
    draw: ImageDraw.ImageDraw,
    layout: IsoFarmLayout,
    tile_xywh: TileBox,
    *,
    level: int,
    gate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    x, y, w, h = tile_xywh
    side_segments: list[tuple[str, int, int, int]] = []
    for offset in range(w):
        side_segments.append(("top", offset, x + offset, y))
        side_segments.append(("bottom", offset, x + offset, y + h - 1))
    for offset in range(h):
        side_segments.append(("left", offset, x, y + offset))
        side_segments.append(("right", offset, x + w - 1, y + offset))

    records: list[dict[str, Any]] = []
    gate_side = str(gate.get("side", ""))
    gate_index = int(gate.get("index", -1))
    for side, index, col, row in sorted(side_segments, key=lambda item: (item[2] + item[3], item[2], item[0])):
        is_gate = side == gate_side and index == gate_index
        p0, p1 = _edge_points(layout, col, row, level, side)
        _draw_fence_segment(draw, p0, p1, is_gate=is_gate)
        records.append(
            {
                "side": side,
                "index": int(index),
                "tile_xy": [int(col), int(row)],
                "is_gate": bool(is_gate),
                "screen_segment": [[round(x, 3), round(y, 3)] for x, y in _display_polygon((p0, p1), layout=layout)],
            }
        )
    return records


def _draw_iso_building(
    draw: ImageDraw.ImageDraw,
    layout: IsoFarmLayout,
    tile_xywh: TileBox,
    *,
    level: int,
    variant: str,
    body_rgb: RGB,
    roof_rgb: RGB,
    door_state: str,
) -> None:
    top, right, bottom, left = _footprint_polygon_c(layout, tile_xywh, level)
    wall_h = 34 if variant == "barn" else 22
    roof_h = 19 if variant == "barn" else 12
    eave = 4 if variant == "barn" else 3

    top_u = (top[0], top[1] - wall_h)
    right_u = (right[0], right[1] - wall_h)
    bottom_u = (bottom[0], bottom[1] - wall_h)
    left_u = (left[0], left[1] - wall_h)

    draw.polygon(_polygon_to_ints((top, right, right_u, top_u)), fill=_shade(body_rgb, -42))
    draw.polygon(_polygon_to_ints((right, bottom, bottom_u, right_u)), fill=_shade(body_rgb, -25))
    draw.polygon(_polygon_to_ints((left, bottom, bottom_u, left_u)), fill=body_rgb)
    draw.polygon(_polygon_to_ints((top, left, left_u, top_u)), fill=_shade(body_rgb, -12))

    roof_top = (top_u[0], top_u[1] - roof_h)
    roof_right = (right_u[0] + eave, right_u[1] - 4)
    roof_bottom = (bottom_u[0], bottom_u[1] + eave)
    roof_left = (left_u[0] - eave, left_u[1] - 4)
    draw.polygon(_polygon_to_ints((roof_top, roof_right, roof_bottom)), fill=_shade(roof_rgb, -12))
    draw.polygon(_polygon_to_ints((roof_top, roof_bottom, roof_left)), fill=roof_rgb)
    draw.line(_polygon_to_ints((roof_top, roof_bottom)), fill=_shade(roof_rgb, -45), width=1)
    draw.line(_polygon_to_ints((roof_left, roof_bottom, roof_right)), fill=_shade(roof_rgb, -55), width=1)
    for step in range(5, int(abs(roof_bottom[1] - roof_top[1])), 6):
        draw.line(
            (
                int(roof_top[0] - step * 0.4),
                int(roof_top[1] + step),
                int(roof_top[0] + step * 0.8),
                int(roof_top[1] + step + 4),
            ),
            fill=_shade(roof_rgb, -35),
        )

    door_w = 12 if variant == "barn" else 8
    door_h = 21 if variant == "barn" else 13
    door_x = int(round(bottom[0]))
    door_bottom = int(round(bottom[1] - 2))
    if door_state == "open":
        draw.rectangle(
            (door_x - door_w // 2, door_bottom - door_h, door_x + door_w // 2, door_bottom),
            fill=(35, 31, 27),
            outline=(68, 45, 34),
        )
        draw.polygon(
            (
                (door_x - door_w // 2 - 1, door_bottom - door_h + 1),
                (door_x - door_w - 7, door_bottom - door_h + 5),
                (door_x - door_w - 6, door_bottom - 2),
                (door_x - door_w // 2 - 1, door_bottom),
            ),
            fill=(111, 69, 45),
            outline=(68, 45, 34),
        )
    else:
        draw.rectangle(
            (door_x - door_w // 2, door_bottom - door_h, door_x + door_w // 2, door_bottom),
            fill=(111, 69, 45),
            outline=(68, 45, 34),
        )
        draw.line((door_x - door_w // 2, door_bottom - door_h, door_x + door_w // 2, door_bottom), fill=(203, 150, 92))
        draw.line((door_x + door_w // 2, door_bottom - door_h, door_x - door_w // 2, door_bottom), fill=(203, 150, 92))

    window = (237, 205, 117)
    outline = (71, 50, 39)
    draw.rectangle((int(left_u[0] + 7), int(left_u[1] + 12), int(left_u[0] + 15), int(left_u[1] + 19)), fill=window, outline=outline)
    if variant == "barn":
        draw.rectangle(
            (int(right_u[0] - 20), int(right_u[1] + 14), int(right_u[0] - 11), int(right_u[1] + 22)),
            fill=window,
            outline=outline,
        )


def _shared_object_spec(
    *,
    entity_id: str,
    public_name: str,
    category: str,
    tile_xywh: TileBox,
    level: int,
    bbox_xyxy: BBox | None,
    metadata: Mapping[str, Any],
) -> IllustrationObjectSpec | None:
    variant = str(metadata.get("variant", public_name))
    if variant == "animal" or category == "animal":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="domestic_animal",
            public_name=str(public_name),
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            level=int(level),
            semantic_attributes={
                "animal_type": str(metadata.get("animal_type", public_name)),
                "region_id": str(metadata.get("region_id", "")),
                "inside_pen": bool(metadata.get("inside_pen", False)),
            },
            visual_attributes=dict(metadata),
            source_entity_type="pixel_isometric_farmstead_entity",
        )
    if variant == "tree":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="tree",
            public_name="tree",
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            level=int(level),
            variant_id=str(metadata.get("tree_style", "fruit_tree")),
            semantic_attributes={"region_id": str(metadata.get("region_id", ""))},
            visual_attributes=dict(metadata),
            source_entity_type="pixel_isometric_farmstead_entity",
        )
    if variant == "flower":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="flower",
            public_name="flower",
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            level=int(level),
            semantic_attributes={"region_id": str(metadata.get("region_id", ""))},
            visual_attributes=dict(metadata),
            source_entity_type="pixel_isometric_farmstead_entity",
        )
    if variant == "person" or category == "person":
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type="person",
            public_name=str(public_name),
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            level=int(level),
            variant_id=str(metadata.get("person_variant_id", "farmer")),
            semantic_attributes={"region_id": str(metadata.get("region_id", ""))},
            visual_attributes=dict(metadata),
            source_entity_type="pixel_isometric_farmstead_entity",
        )
    if variant in {"crate", "hay_bale", "trough"}:
        return IllustrationObjectSpec(
            object_id=str(entity_id),
            object_type=variant,
            public_name=str(public_name),
            bbox_xyxy=bbox_xyxy,
            tile_xywh=tile_xywh,
            level=int(level),
            semantic_attributes={"region_id": str(metadata.get("region_id", ""))},
            visual_attributes=dict(metadata),
            source_entity_type="pixel_isometric_farmstead_entity",
        )
    return None


def _isometric_render_context(layout: IsoFarmLayout, image: Image.Image | None = None) -> RenderContext:
    def project_tile_center(tile_xywh: TileBox, level: int) -> IsoPoint:
        x, y, w, h = tile_xywh
        return _project_c(layout, x + (w - 1) * 0.5, y + (h - 1) * 0.5, level)

    return RenderContext(
        renderer_style=RENDERER_STYLE_ISOMETRIC_PIXEL_RPG,
        image=image,
        project_tile_center=project_tile_center,
    )


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


def _sample_person_metadata(rng: random.Random, *, region_id: str) -> dict[str, Any]:
    person_variant_id = sample_object_variant_id(rng, "person", support=("farmer", "worker", "adult"))
    hair_palette = [(54, 38, 28), (84, 49, 31), (116, 75, 35), (49, 44, 44), (154, 112, 50)]
    shirt_palette = [(63, 127, 74), (177, 84, 63), (219, 154, 65), (72, 118, 171)]
    pants_palette = [(47, 66, 102), (73, 67, 60), (86, 96, 65), (56, 83, 119)]
    skin_palette = [(225, 171, 109), (189, 127, 80), (237, 190, 133), (154, 95, 65)]
    return {
        "variant": "person",
        **variant_visual_metadata("person", person_variant_id, RENDERER_STYLE_ISOMETRIC_PIXEL_RPG),
        "gender_id": str(_choose(rng, ["male", "female"])),
        "facing": str(_choose(rng, ["down", "up", "left", "right"])),
        "region_id": str(region_id),
        "skin_rgb": list(_choose(rng, skin_palette)),
        "shirt_rgb": list(_choose(rng, shirt_palette)),
        "pants_rgb": list(_choose(rng, pants_palette)),
        "hair_rgb": list(_choose(rng, hair_palette)),
    }


def _add_entity(
    entities: list[IsoFarmEntity],
    *,
    entity_id: str,
    public_name: str,
    category: str,
    tile_xywh: TileBox,
    level: int,
    layer: str,
    layout: IsoFarmLayout,
    metadata: Mapping[str, Any],
) -> None:
    bbox_c = _entity_canonical_bbox(layout, tile_xywh, level, metadata)
    bbox_display = _display_bbox(bbox_c, layout=layout)
    footprint = _footprint_polygon_display(layout, tile_xywh, level)
    metadata_with_level = dict(metadata)
    metadata_with_level.setdefault("base_level", int(level))
    metadata_with_level.setdefault("variant", str(metadata.get("variant", public_name)))
    spec = _shared_object_spec(
        entity_id=str(entity_id),
        public_name=str(public_name),
        category=str(category),
        tile_xywh=tile_xywh,
        level=int(level),
        bbox_xyxy=bbox_display,
        metadata=metadata_with_level,
    )
    if spec is not None:
        metadata_with_level["object_record"] = object_record_for_spec(
            spec,
            _isometric_render_context(layout),
        )
    entities.append(
        IsoFarmEntity(
            entity_id=str(entity_id),
            public_name=str(public_name),
            category=str(category),
            tile_xywh=tile_xywh,
            level=int(level),
            bbox_xyxy=bbox_display,
            footprint_polygon_xy=footprint,
            anchor_screen_xy=_entity_anchor_display(layout, tile_xywh, level),
            layer=str(layer),
            metadata=metadata_with_level,
        )
    )


def _add_region(
    regions: list[IsoFarmRegion],
    *,
    region_id: str,
    public_name: str,
    region_type: str,
    tile_xywh: TileBox,
    level: int,
    layout: IsoFarmLayout,
    metadata: Mapping[str, Any],
) -> None:
    polygon = _footprint_polygon_display(layout, tile_xywh, level)
    regions.append(
        IsoFarmRegion(
            region_id=str(region_id),
            public_name=str(public_name),
            region_type=str(region_type),
            tile_xywh=tile_xywh,
            level=int(level),
            polygon_xy=polygon,
            bbox_xyxy=_region_bbox_display(layout, tile_xywh, level),
            metadata=dict(metadata),
        )
    )


def _build_tiles(
    *,
    layout: IsoFarmLayout,
    level_grid: Mapping[tuple[int, int], int],
    terrain: Mapping[tuple[int, int], str],
    upper_terrace: TileBox,
) -> tuple[IsoFarmTile, ...]:
    tiles: list[IsoFarmTile] = []
    for row in range(layout.rows):
        for col in range(layout.cols):
            level = int(level_grid[(col, row)])
            terrain_id = str(terrain[(col, row)])
            polygon_c = _tile_vertices_c(layout, col, row, level)
            metadata = {
                "is_upper_terrace": bool(_inside((col, row), upper_terrace)),
                "projection": "2:1_isometric",
            }
            tiles.append(
                IsoFarmTile(
                    col=col,
                    row=row,
                    level=level,
                    terrain=terrain_id,
                    polygon_xy=_display_polygon(polygon_c, layout=layout),
                    bbox_xyxy=_display_bbox(_bbox_from_points(polygon_c), layout=layout),
                    metadata=metadata,
                )
            )
    return tuple(sorted(tiles, key=lambda tile: (tile.row + tile.col, tile.col)))


def _place_scene_entities(
    rng: random.Random,
    *,
    layout: IsoFarmLayout,
    level_grid: Mapping[tuple[int, int], int],
    plan: Mapping[str, Any],
) -> list[IsoFarmEntity]:
    entities: list[IsoFarmEntity] = []
    barn: TileBox = plan["barn"]
    coop: TileBox = plan["coop"]
    upper_pen: TileBox = plan["upper_pen"]
    lower_pen: TileBox = plan["lower_pen"]
    ramp_tile = plan["ramp_tile"]
    lower_field: TileBox = plan["lower_field"]
    upper_level = int(plan["upper_level"])

    _add_entity(
        entities,
        entity_id="barn_00",
        public_name="barn",
        category="building",
        tile_xywh=barn,
        level=upper_level,
        layer="building",
        layout=layout,
        metadata={
            "variant": "barn",
            "building_door_state": _sample_building_door_state(rng),
            "building_facing": "front_isometric",
            "body_rgb": list(_choose(rng, [(176, 65, 54), (190, 73, 60), (158, 69, 58)])),
            "roof_rgb": list(_choose(rng, [(119, 49, 47), (132, 55, 49), (95, 55, 62)])),
            "region_id": "upper_terrace",
        },
    )
    _add_entity(
        entities,
        entity_id="coop_00",
        public_name="chicken coop",
        category="building",
        tile_xywh=coop,
        level=0,
        layer="building",
        layout=layout,
        metadata={
            "variant": "coop",
            "building_door_state": _sample_building_door_state(rng),
            "building_facing": "front_isometric",
            "body_rgb": list(_choose(rng, [(188, 126, 70), (202, 142, 82)])),
            "roof_rgb": list(_choose(rng, [(137, 62, 50), (151, 74, 54)])),
            "region_id": "lower_field",
        },
    )

    fixture_specs: list[tuple[str, str, str, TileBox, int, dict[str, Any]]] = [
        ("hay_00", "hay bale", "farm_fixture", (barn[0] - 1, barn[1] + 2, 1, 1), upper_level, {"variant": "hay_bale"}),
        ("crate_00", "crate", "farm_fixture", (barn[0] - 1, barn[1] + 1, 1, 1), upper_level, {"variant": "crate"}),
        (
            "trough_upper",
            "trough",
            "farm_fixture",
            (upper_pen[0] + 1, upper_pen[1] + upper_pen[3] - 1, 2, 1),
            upper_level,
            {"variant": "trough", "region_id": "upper_pen"},
        ),
        (
            "trough_lower",
            "trough",
            "farm_fixture",
            (lower_pen[0] + 1, lower_pen[1] + lower_pen[3] - 1, 2, 1),
            0,
            {"variant": "trough", "region_id": "lower_pen"},
        ),
    ]
    for entity_id, public_name, category, tile_box, level, metadata in fixture_specs:
        _add_entity(
            entities,
            entity_id=entity_id,
            public_name=public_name,
            category=category,
            tile_xywh=tile_box,
            level=level,
            layer="object",
            layout=layout,
            metadata=metadata,
        )

    animal_specs: list[tuple[str, tuple[int, int], str, bool]] = [
        ("sheep", (upper_pen[0] + 1, upper_pen[1]), "upper_pen", True),
        ("chicken", (upper_pen[0] + 2, upper_pen[1] + 1), "upper_pen", True),
        ("pig", (lower_pen[0] + 1, lower_pen[1] + 1), "lower_pen", True),
        ("cow", (lower_pen[0] + 2, lower_pen[1] + 1), "lower_pen", True),
        ("sheep", (lower_pen[0] + 1, lower_pen[1] + 2), "lower_pen", True),
        ("chicken", (coop[0] + 2, coop[1] + 1), "outside_fence", False),
        ("pig", (ramp_tile[0] - 2, ramp_tile[1] + 2), "outside_fence", False),
    ]
    if rng.random() < 0.55:
        animal_specs.append(("chicken", (coop[0] + 2, coop[1] + 2), "outside_fence", False))
    if rng.random() < 0.45:
        animal_specs.append(("sheep", (lower_pen[0] - 2, lower_pen[1] + 1), "outside_fence", False))

    for index, (animal_type, xy, region_id, inside_pen) in enumerate(animal_specs):
        aw, ah = (2, 1) if animal_type == "cow" else (1, 1)
        col = max(1, min(layout.cols - aw - 1, int(xy[0])))
        row = max(1, min(layout.rows - ah - 1, int(xy[1])))
        level = int(level_grid[(col, row)])
        metadata = _sample_animal_metadata(rng, animal_type, region_id=region_id, inside_pen=inside_pen)
        _add_entity(
            entities,
            entity_id=f"animal_{index:02d}",
            public_name=animal_type,
            category="animal",
            tile_xywh=(col, row, aw, ah),
            level=level,
            layer="actor",
            layout=layout,
            metadata=metadata,
        )

    person_specs: list[tuple[tuple[int, int], str]] = [
        ((max(1, ramp_tile[0] - 1), min(layout.rows - 2, ramp_tile[1] + 1)), "lower_field"),
        ((max(1, barn[0] - 2), min(layout.rows - 2, barn[1] + 2)), "upper_terrace"),
    ]
    for index, (xy, region_id) in enumerate(person_specs):
        col = max(1, min(layout.cols - 2, int(xy[0])))
        row = max(1, min(layout.rows - 2, int(xy[1])))
        level = int(level_grid[(col, row)])
        metadata = _sample_person_metadata(rng, region_id=region_id)
        _add_entity(
            entities,
            entity_id=f"person_{index:02d}",
            public_name="person",
            category="person",
            tile_xywh=(col, row, 1, 1),
            level=level,
            layer="actor",
            layout=layout,
            metadata=metadata,
        )

    tree_positions = [
        (1, 2),
        (layout.cols - 2, 2),
        (lower_field[0], lower_field[1] + lower_field[3] - 2),
        (layout.cols - 3, layout.rows - 3),
        (2, layout.rows - 3),
    ]
    rng.shuffle(tree_positions)
    for index, (col, row) in enumerate(tree_positions[: rng.randint(4, 5)]):
        col = max(1, min(layout.cols - 2, col))
        row = max(1, min(layout.rows - 2, row))
        level = int(level_grid[(col, row)])
        style = str(_choose(rng, PIXEL_TREE_STYLES))
        metadata = {
            "variant": "tree",
            **variant_visual_metadata("tree", style, RENDERER_STYLE_ISOMETRIC_PIXEL_RPG),
            "tree_style": style,
            "leaf_rgb": list(_choose(rng, [(42, 139, 76), (54, 147, 72), (65, 137, 69), (92, 128, 66)])),
            "fruit_rgb": list(_choose(rng, [(218, 62, 58), (235, 181, 54), (190, 65, 126)])),
            "region_id": "upper_terrace" if level > 0 else "lower_field",
        }
        _add_entity(
            entities,
            entity_id=f"tree_{index:02d}",
            public_name="tree",
            category="plant",
            tile_xywh=(col, row, 1, 1),
            level=level,
            layer="plant",
            layout=layout,
            metadata=metadata,
        )

    flower_candidates = [
        (2, 4),
        (3, 5),
        (5, layout.rows - 3),
        (layout.cols - 4, layout.rows - 2),
        (layout.cols - 6, 6),
        (2, layout.rows - 2),
    ]
    for index, (col, row) in enumerate(flower_candidates[: rng.randint(4, 6)]):
        col = max(1, min(layout.cols - 2, col))
        row = max(1, min(layout.rows - 2, row))
        level = int(level_grid[(col, row)])
        _add_entity(
            entities,
            entity_id=f"flower_{index:02d}",
            public_name="flower",
            category="plant",
            tile_xywh=(col, row, 1, 1),
            level=level,
            layer="plant",
            layout=layout,
            metadata={
                "variant": "flower",
                "flower_rgb": list(_choose(rng, [(238, 126, 62), (236, 82, 111), (239, 201, 72)])),
                "leaf_rgb": list(_choose(rng, [(36, 125, 62), (44, 144, 72), (55, 132, 72)])),
                "region_id": "upper_terrace" if level > 0 else "lower_field",
            },
        )

    return entities


def _draw_entity(image: Image.Image, draw: ImageDraw.ImageDraw, layout: IsoFarmLayout, entity: IsoFarmEntity) -> None:
    variant = str(entity.metadata.get("variant", entity.public_name))
    spec = _shared_object_spec(
        entity_id=entity.entity_id,
        public_name=entity.public_name,
        category=entity.category,
        tile_xywh=entity.tile_xywh,
        level=entity.level,
        bbox_xyxy=entity.bbox_xyxy,
        metadata=entity.metadata,
    )
    if spec is not None:
        render_illustration_object(spec, _isometric_render_context(layout, image=image))
    elif variant in {"barn", "coop"}:
        _draw_iso_building(
            draw,
            layout,
            entity.tile_xywh,
            level=entity.level,
            variant=variant,
            body_rgb=tuple(entity.metadata.get("body_rgb", (176, 65, 54))),
            roof_rgb=tuple(entity.metadata.get("roof_rgb", (119, 49, 47))),
            door_state=str(entity.metadata.get("building_door_state", "closed")),
        )


def _layer_priority(layer: str) -> int:
    return {"plant": 10, "object": 20, "actor": 30, "building": 40}.get(str(layer), 50)


def render_pixel_isometric_farmstead(
    seed: int,
    *,
    width: int = 960,
    height: int = 720,
    scale: int = DEFAULT_SCALE,
    grid_cols: int | None = None,
    grid_rows: int | None = None,
    elevation_depth: int | None = None,
) -> IsoFarmScene:
    """Render one deterministic old-school isometric pixel RPG farmstead."""

    rng = random.Random(int(seed))
    layout = _sample_layout(
        rng,
        width=int(width),
        height=int(height),
        scale=int(scale),
        grid_cols=grid_cols,
        grid_rows=grid_rows,
    )
    plan = _make_scene_plan(rng, layout=layout, elevation_depth=elevation_depth)
    upper_level = int(plan["upper_level"])
    level_grid = _make_level_grid(layout, plan["upper_terrace"], upper_level=upper_level)
    terrain = plan["terrain"]
    tiles = _build_tiles(layout=layout, level_grid=level_grid, terrain=terrain, upper_terrace=plan["upper_terrace"])

    base = Image.new("RGBA", (layout.canonical_width_px, layout.canonical_height_px), (229, 232, 215, 255))
    draw = ImageDraw.Draw(base, "RGBA")
    terrain_rng = random.Random(int(seed) + 4107)
    for tile in tiles:
        _draw_iso_tile(
            draw,
            layout,
            tile.col,
            tile.row,
            level=tile.level,
            terrain=tile.terrain,
            rng=terrain_rng,
        )
    transition_specs = [
        ("transition_ramp_00", "ramp", plan["ramp_tile"], plan["upper_ramp_tile"]),
        ("transition_stair_00", "stair", plan["stair_tile"], plan["upper_stair_tile"]),
    ]
    transition_open_edges = {
        (int(upper_tile_xy[0]), int(upper_tile_xy[1]), "south")
        for _, _, _, upper_tile_xy in transition_specs
    }
    retaining_wall_faces = _draw_level_faces(
        draw,
        layout,
        level_grid=level_grid,
        open_edges=transition_open_edges,
    )

    transitions: list[IsoFarmTransition] = []
    for transition_id, transition_type, lower_tile_xy, upper_tile_xy in transition_specs:
        polygon_c = _draw_transition_surface(
            draw,
            layout,
            lower_tile_xy=lower_tile_xy,
            upper_tile_xy=upper_tile_xy,
            lower_level=0,
            upper_level=upper_level,
            transition_type=transition_type,
        )
        polygon_display = _display_polygon(polygon_c, layout=layout)
        transitions.append(
            IsoFarmTransition(
                transition_id=transition_id,
                transition_type=transition_type,
                lower_tile_xy=lower_tile_xy,
                upper_tile_xy=upper_tile_xy,
                lower_level=0,
                upper_level=upper_level,
                polygon_xy=polygon_display,
                bbox_xyxy=_display_bbox(_bbox_from_points(polygon_c), layout=layout),
                metadata={"connects_region_ids": ["lower_field", "upper_terrace"], "elevation_depth": upper_level},
            )
        )

    regions: list[IsoFarmRegion] = []
    _add_region(
        regions,
        region_id="upper_terrace",
        public_name="raised barn terrace",
        region_type="raised_terrace",
        tile_xywh=plan["upper_terrace"],
        level=upper_level,
        layout=layout,
        metadata={"level": upper_level, "supports_building": True, "retaining_wall_visible": True},
    )
    _add_region(
        regions,
        region_id="lower_field",
        public_name="lower field",
        region_type="lower_field",
        tile_xywh=plan["lower_field"],
        level=0,
        layout=layout,
        metadata={"level": 0, "contains_crop_rows": True},
    )
    _add_region(
        regions,
        region_id="upper_pen",
        public_name="upper pen",
        region_type="fenced_pen",
        tile_xywh=plan["upper_pen"],
        level=upper_level,
        layout=layout,
        metadata={"level": upper_level, "contains_fence_border": True, "gate_edge": dict(plan["gate_edges"]["upper_pen"])},
    )
    _add_region(
        regions,
        region_id="lower_pen",
        public_name="lower pen",
        region_type="fenced_pen",
        tile_xywh=plan["lower_pen"],
        level=0,
        layout=layout,
        metadata={"level": 0, "contains_fence_border": True, "gate_edge": dict(plan["gate_edges"]["lower_pen"])},
    )

    entities = _place_scene_entities(rng, layout=layout, level_grid=level_grid, plan=plan)
    draw_order = sorted(
        entities,
        key=lambda item: (
            item.anchor_screen_xy[1],
            item.level,
            _layer_priority(item.layer),
            item.anchor_screen_xy[0],
            item.entity_id,
        ),
    )
    for entity in draw_order:
        _draw_entity(base, draw, layout, entity)

    fence_records = {
        "upper_pen": _draw_pen_fence(
            draw,
            layout,
            plan["upper_pen"],
            level=upper_level,
            gate=plan["gate_edges"]["upper_pen"],
        ),
        "lower_pen": _draw_pen_fence(
            draw,
            layout,
            plan["lower_pen"],
            level=0,
            gate=plan["gate_edges"]["lower_pen"],
        ),
    }

    image = base.resize((int(width), int(height)), Image.Resampling.NEAREST).convert("RGB")
    entities_sorted = tuple(sorted(entities, key=lambda item: item.entity_id))
    regions_sorted = tuple(sorted(regions, key=lambda item: item.region_id))
    transitions_sorted = tuple(sorted(transitions, key=lambda item: item.transition_id))
    animal_entities = [entity for entity in entities_sorted if entity.category == "animal"]
    person_entities = [entity for entity in entities_sorted if entity.category == "person"]
    terrain_counts = {
        terrain_id: sum(1 for value in terrain.values() if value == terrain_id)
        for terrain_id in sorted(set(terrain.values()))
    }
    level_counts = {
        str(level): sum(1 for value in level_grid.values() if int(value) == level)
        for level in sorted(set(int(value) for value in level_grid.values()))
    }
    trace = {
        "renderer_id": "pixel_isometric_farmstead_v0",
        "seed": int(seed),
        "inspiration_sources": [
            {
                "name": "Kenney Isometric Miniature Farm",
                "url": "https://kenney.nl/assets/isometric-miniature-farm",
                "notes": "CC0 isometric farm pack used as visual reference only.",
            },
            {
                "name": "Kenney Isometric Tiles Buildings",
                "url": "https://kenney.nl/assets/isometric-tiles-buildings",
                "notes": "CC0 isometric building pack used as visual reference only.",
            },
        ],
        "uses_external_sprites": False,
        "projection": {
            "type": "2:1_isometric",
            "canonical_tile_size_px": [CANONICAL_ISO_TILE_W_PX, CANONICAL_ISO_TILE_H_PX],
            "display_tile_size_px": [layout.display_iso_tile_w_px, layout.display_iso_tile_h_px],
            "canonical_level_px": CANONICAL_LEVEL_PX,
            "display_level_px": layout.display_level_px,
            "origin_xy": [round(v, 3) for v in _display_point(layout.origin_xy, layout=layout)],
        },
        "grid_cols": layout.cols,
        "grid_rows": layout.rows,
        "scale": layout.scale,
        "canonical_canvas_size_px": [layout.canonical_width_px, layout.canonical_height_px],
        "canvas_size_px": [int(width), int(height)],
        "levels": sorted(set(int(value) for value in level_grid.values())),
        "level_count": len(set(int(value) for value in level_grid.values())),
        "lower_level": 0,
        "upper_level": upper_level,
        "elevation_depth": upper_level,
        "supported_elevation_depths": [1, 2],
        "level_tile_counts": level_counts,
        "terrain_counts": terrain_counts,
        "path_tiles": [[int(x), int(y), int(level_grid[(x, y)])] for x, y in sorted(plan["path_tiles"])],
        "crop_tiles": [[int(x), int(y), int(level_grid[(x, y)])] for x, y in sorted(plan["crop_tiles"])],
        "tiles": [tile.as_dict() for tile in tiles],
        "tile_count": len(tiles),
        "regions": [region.as_dict() for region in regions_sorted],
        "transitions": [transition.as_dict() for transition in transitions_sorted],
        "transition_count": len(transitions_sorted),
        "retaining_wall_face_count": len(retaining_wall_faces),
        "retaining_wall_faces": retaining_wall_faces,
        "transition_open_edges": [
            {"tile_xy": [int(col), int(row)], "side": str(side)}
            for col, row, side in sorted(transition_open_edges)
        ],
        "fence_segments": fence_records,
        "gate_edges": {region_id: dict(value) for region_id, value in sorted(plan["gate_edges"].items())},
        "entity_draw_order": [entity.entity_id for entity in draw_order],
        "entity_count": len(entities_sorted),
        "animal_count": len(animal_entities),
        "person_count": len(person_entities),
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
    return IsoFarmScene(
        image=image,
        tiles=tiles,
        entities=entities_sorted,
        regions=regions_sorted,
        transitions=transitions_sorted,
        trace=trace,
    )


def draw_pixel_isometric_farmstead_debug_overlay(scene: IsoFarmScene, *, show_levels: bool = True) -> Image.Image:
    """Return a debug overlay with projected regions, bboxes, transitions, and labels."""

    image = scene.image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    try:
        from trace_tasks.tasks.shared.text_rendering import load_font

        font = load_font(11, bold=True)
        small_font = load_font(9, bold=False)
    except Exception:  # pragma: no cover - review helper fallback
        font = None
        small_font = None

    if show_levels:
        for tile in scene.tiles:
            if tile.level <= 0:
                continue
            x0, y0, x1, y1 = tile.bbox_xyxy
            draw.text(
                (round((x0 + x1) * 0.5 - 5), round((y0 + y1) * 0.5 - 5)),
                f"L{tile.level}",
                fill=(32, 90, 160, 185),
                font=small_font,
            )

    region_palette = {
        "raised_terrace": (45, 95, 190, 230),
        "lower_field": (42, 134, 68, 230),
        "fenced_pen": (204, 113, 39, 235),
    }
    for region in scene.regions:
        color = region_palette.get(region.region_type, (60, 60, 60, 220))
        polygon = _polygon_to_ints(region.polygon_xy)
        draw.line(polygon + [polygon[0]], fill=color, width=3)
        x0, y0, _, _ = region.bbox_xyxy
        text_bbox = draw.textbbox((x0, y0), region.public_name, font=font)
        draw.rectangle(
            (text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1),
            fill=(248, 250, 238, 225),
        )
        draw.text((x0, y0), region.public_name, fill=color, font=font)

    for transition in scene.transitions:
        polygon = _polygon_to_ints(transition.polygon_xy)
        draw.line(polygon + [polygon[0]], fill=(60, 75, 210, 230), width=3)
        x0, y0, _, _ = transition.bbox_xyxy
        label = transition.transition_type
        text_bbox = draw.textbbox((x0, y0), label, font=font)
        draw.rectangle(
            (text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1),
            fill=(240, 246, 255, 225),
        )
        draw.text((x0, y0), label, fill=(60, 75, 210, 255), font=font)

    entity_palette = {
        "animal": (198, 60, 70, 235),
        "building": (105, 72, 190, 235),
        "farm_fixture": (210, 132, 45, 235),
        "person": (142, 78, 184, 235),
        "plant": (34, 144, 73, 235),
    }
    for entity in scene.entities:
        color = entity_palette.get(entity.category, (35, 35, 35, 235))
        box = [round(v) for v in entity.bbox_xyxy]
        draw.rectangle(box, outline=color, width=2)
        label = f"{entity.public_name} L{entity.level}"
        text_bbox = draw.textbbox((box[0], box[1]), label, font=font)
        draw.rectangle(
            (text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1),
            fill=(255, 255, 238, 220),
        )
        draw.text((box[0], box[1]), label, fill=color, font=font)
    image.alpha_composite(overlay)
    return image.convert("RGB")


__all__ = [
    "CANONICAL_ISO_TILE_H_PX",
    "CANONICAL_ISO_TILE_W_PX",
    "CANONICAL_LEVEL_PX",
    "DEFAULT_DISPLAY_ISO_TILE_H_PX",
    "DEFAULT_DISPLAY_ISO_TILE_W_PX",
    "DEFAULT_DISPLAY_LEVEL_PX",
    "DEFAULT_GRID_COLS",
    "DEFAULT_GRID_ROWS",
    "DEFAULT_SCALE",
    "GRID_COLS",
    "GRID_ROWS",
    "IsoFarmEntity",
    "IsoFarmLayout",
    "IsoFarmRegion",
    "IsoFarmScene",
    "IsoFarmTile",
    "IsoFarmTransition",
    "draw_pixel_isometric_farmstead_debug_overlay",
    "render_pixel_isometric_farmstead",
]
