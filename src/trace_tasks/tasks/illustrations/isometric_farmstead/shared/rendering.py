"""Renderer for the isometric farmstead scene package."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Callable, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.illustrations.shared.isometric_visual_styles import (
    IsometricIllustrationTone,
    isometric_terrain_triplet,
    resolve_isometric_illustration_tone,
    tint_isometric_semantic_rgb,
)
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import draw_pixel_animal, draw_pixel_person, draw_pixel_tree
from trace_tasks.tasks.illustrations.shared.option_rendering import draw_label_badge, sample_visual_label_font_trace

from .sampling import LabelTaskSampleSpec
from .state import (
    BBox,
    IsoFarmsteadEntity,
    IsoFarmsteadScene,
    IsoFarmsteadTile,
    IsoPoint,
    IsoPolygon,
)


SCENE_ID = "isometric_farmstead"
RENDERER_ID = "isometric_farmstead_v0"
RENDERER_STYLE = "isometric_pixel_farmstead"
SUPPORTED_LEVELS: tuple[int, ...] = (0, 1, 2, 3)
ACTIVE_MAX_LEVEL_SUPPORT: tuple[int, ...] = (1, 2)
DEFAULT_CANDIDATE_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
LAYOUT_FAMILIES: tuple[str, ...] = (
    "concentric_terrace",
    "side_plateau",
    "corner_plateau",
    "split_field",
)
FARM_PATCH_TERRAINS: tuple[str, ...] = ("crop", "soil", "flower", "pasture")
TREE_STYLES: tuple[str, ...] = ("oak", "pine", "maple", "fruit_tree")
ANIMAL_TYPES: tuple[str, ...] = ("chicken", "pig", "sheep", "cow")

RGB = tuple[int, int, int]
TileRect = tuple[int, int, int, int]


@dataclass(frozen=True)
class IsoLayout:
    """Resolved isometric grid geometry."""

    cols: int
    rows: int
    width: int
    height: int
    tile_w: float
    tile_h: float
    level_px: float
    origin_xy: IsoPoint


def _shade(color: RGB, delta: int) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def _clamp(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _point_bbox(points: Sequence[IsoPoint]) -> BBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_union(boxes: Sequence[Sequence[float]]) -> BBox:
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _tile_id(col: int, row: int) -> str:
    return f"tile_{int(col):02d}_{int(row):02d}"


def _profile_grid(width: int, height: int) -> tuple[int, int]:
    if int(width) > int(height) * 1.15:
        return (16, 12)
    return (14, 14)


def _inside(col: int, row: int, rect: TileRect) -> bool:
    x, y, w, h = rect
    return int(x) <= int(col) < int(x + w) and int(y) <= int(row) < int(y + h)


def _inset_rect(rect: TileRect, *, dx: int, dy: int, min_w: int, min_h: int) -> TileRect:
    x, y, w, h = rect
    next_w = max(int(min_w), int(w) - 2 * int(dx))
    next_h = max(int(min_h), int(h) - 2 * int(dy))
    return (int(x) + max(0, (int(w) - next_w) // 2), int(y) + max(0, (int(h) - next_h) // 2), next_w, next_h)


def _rect_cells(rect: TileRect) -> list[tuple[int, int]]:
    x, y, w, h = rect
    return [(int(col), int(row)) for row in range(int(y), int(y + h)) for col in range(int(x), int(x + w))]


def _snap_one_tile_border_gap(rect: TileRect, *, cols: int, rows: int) -> TileRect:
    """Remove one-tile low strips between a raised terrace and the grid edge."""

    x, y, w, h = [int(value) for value in rect]
    if x == 1:
        x = 0
        w += 1
    if y == 1:
        y = 0
        h += 1
    if int(cols) - (x + w) == 1:
        w += 1
    if int(rows) - (y + h) == 1:
        h += 1
    w = min(int(w), int(cols) - int(x))
    h = min(int(h), int(rows) - int(y))
    return (int(x), int(y), int(w), int(h))


def _snap_rects_one_tile_border_gaps(
    rects: Mapping[int, Sequence[TileRect]],
    *,
    cols: int,
    rows: int,
) -> dict[int, list[TileRect]]:
    return {
        int(level): [
            _snap_one_tile_border_gap(tuple(rect), cols=int(cols), rows=int(rows))
            for rect in level_rects
        ]
        for level, level_rects in rects.items()
    }


def _blank_level_grid(*, cols: int, rows: int) -> dict[tuple[int, int], int]:
    return {(int(col), int(row)): 0 for row in range(int(rows)) for col in range(int(cols))}


def _apply_rect(grid: dict[tuple[int, int], int], rect: TileRect, *, level: int, cols: int, rows: int) -> None:
    for col, row in _rect_cells(rect):
        if 0 <= int(col) < int(cols) and 0 <= int(row) < int(rows):
            grid[(int(col), int(row))] = max(int(grid.get((int(col), int(row)), 0)), int(level))


def _sample_active_max_level(rng: random.Random) -> int:
    return int(rng.choice((1, 2, 2)))


def _sample_layout_family(rng: random.Random) -> str:
    return str(rng.choice(LAYOUT_FAMILIES))


def _make_concentric_rects(rng: random.Random, *, cols: int, rows: int, active_max_level: int) -> dict[int, list[TileRect]]:
    pad_x = max(1, int(cols) // 6)
    pad_y = max(1, int(rows) // 6)
    w1 = max(8, int(cols) - 2 * pad_x)
    h1 = max(8, int(rows) - 2 * pad_y)
    x1 = _clamp(pad_x + rng.choice((-1, 0, 1)), 1, int(cols) - w1 - 1)
    y1 = _clamp(pad_y + rng.choice((-1, 0, 1)), 1, int(rows) - h1 - 1)
    rects: dict[int, list[TileRect]] = {1: [(x1, y1, w1, h1)]}
    current = rects[1][0]
    for level in range(2, int(active_max_level) + 1):
        current = _inset_rect(current, dx=rng.choice((1, 2)), dy=rng.choice((1, 2)), min_w=3, min_h=3)
        rects[int(level)] = [current]
    return rects


def _make_side_rects(rng: random.Random, *, cols: int, rows: int, active_max_level: int) -> dict[int, list[TileRect]]:
    side = rng.choice(("north", "south", "west", "east"))
    if side in {"west", "east"}:
        w = rng.randint(max(5, cols // 3), max(6, cols // 2))
        h = rng.randint(max(7, rows // 2), max(8, rows - 2))
        x = 1 if side == "west" else max(1, cols - w - 1)
        y = rng.randint(1, max(1, rows - h - 1))
    else:
        w = rng.randint(max(8, cols // 2), max(9, cols - 2))
        h = rng.randint(max(5, rows // 3), max(6, rows // 2))
        x = rng.randint(1, max(1, cols - w - 1))
        y = 1 if side == "north" else max(1, rows - h - 1)
    rects: dict[int, list[TileRect]] = {1: [(x, y, w, h)]}
    current = rects[1][0]
    for level in range(2, int(active_max_level) + 1):
        current = _inset_rect(current, dx=1, dy=1, min_w=3, min_h=3)
        cx, cy, cw, ch = current
        if side == "west":
            current = (max(1, cx - 1), cy, cw, ch)
        elif side == "east":
            current = (min(cols - cw - 1, cx + 1), cy, cw, ch)
        elif side == "north":
            current = (cx, max(1, cy - 1), cw, ch)
        else:
            current = (cx, min(rows - ch - 1, cy + 1), cw, ch)
        rects[int(level)] = [current]
    return rects


def _make_corner_rects(rng: random.Random, *, cols: int, rows: int, active_max_level: int) -> dict[int, list[TileRect]]:
    corner = rng.choice(("nw", "ne", "sw", "se"))
    w = rng.randint(max(6, cols // 3), max(7, cols // 2))
    h = rng.randint(max(6, rows // 3), max(7, rows // 2))
    x = 1 if "w" in corner else max(1, cols - w - 1)
    y = 1 if "n" in corner else max(1, rows - h - 1)
    rects: dict[int, list[TileRect]] = {1: [(x, y, w, h)]}
    current = rects[1][0]
    for level in range(2, int(active_max_level) + 1):
        current = _inset_rect(current, dx=1, dy=1, min_w=3, min_h=3)
        cx, cy, cw, ch = current
        if "w" in corner:
            cx = max(1, cx - 1)
        else:
            cx = min(cols - cw - 1, cx + 1)
        if "n" in corner:
            cy = max(1, cy - 1)
        else:
            cy = min(rows - ch - 1, cy + 1)
        current = (cx, cy, cw, ch)
        rects[int(level)] = [current]
    return rects


def _make_split_rects(rng: random.Random, *, cols: int, rows: int, active_max_level: int) -> dict[int, list[TileRect]]:
    w = max(4, cols // 3)
    h = max(4, rows // 3)
    left = (rng.randint(1, 2), rng.randint(1, max(1, rows - h - 2)), w, h)
    right = (max(1, cols - w - rng.randint(2, 3)), rng.randint(1, max(1, rows - h - 2)), w, h)
    rects: dict[int, list[TileRect]] = {1: [left, right]}
    current = rng.choice((left, right))
    for level in range(2, int(active_max_level) + 1):
        current = _inset_rect(current, dx=1, dy=1, min_w=3, min_h=3)
        rects[int(level)] = [current]
    return rects


def _count_plateau_width(*, count: int, cols: int, rows: int) -> int:
    """Choose a compact row width for an exact-count connected plateau."""

    best_width = 3
    best_score: tuple[float, int, int] | None = None
    for width in range(3, min(7, int(cols) - 4) + 1):
        height = (int(count) + int(width) - 1) // int(width)
        if height > int(rows) - 4:
            continue
        remainder = int(count) % int(width)
        missing = 0 if remainder == 0 else int(width) - int(remainder)
        aspect_penalty = abs(float(width) / float(max(1, height)) - 1.35)
        single_tail_penalty = 3 if remainder == 1 else 0
        score = (float(missing) + float(single_tail_penalty) + aspect_penalty, int(height), int(width))
        if best_score is None or score < best_score:
            best_score = score
            best_width = int(width)
    return int(best_width)


def _make_count_plateau_rects(
    rng: random.Random,
    *,
    cols: int,
    rows: int,
    highest_tile_count: int,
) -> dict[int, list[TileRect]]:
    """Build a level-2 plateau with exactly the requested number of top tiles."""

    count = int(highest_tile_count)
    if count < 1:
        raise ValueError("highest_tile_count must be positive")
    width = _count_plateau_width(count=count, cols=int(cols), rows=int(rows))
    full_rows = int(count) // int(width)
    remainder = int(count) % int(width)
    height = int(full_rows) + (1 if int(remainder) else 0)
    if height <= 0 or width + 4 > int(cols) or height + 4 > int(rows):
        raise ValueError(f"highest_tile_count {count} does not fit farmstead grid {cols}x{rows}")
    x = rng.randint(2, max(2, int(cols) - int(width) - 2))
    y = rng.randint(2, max(2, int(rows) - int(height) - 2))
    top_rects: list[TileRect] = []
    if full_rows:
        top_rects.append((int(x), int(y), int(width), int(full_rows)))
    if remainder:
        top_rects.append((int(x), int(y) + int(full_rows), int(remainder), 1))
    level_one = (int(x) - 1, int(y) - 1, int(width) + 2, int(height) + 2)
    return {1: [level_one], 2: top_rects}


def _make_level_grid(
    rng: random.Random,
    *,
    cols: int,
    rows: int,
    active_max_level: int,
    layout_family: str,
    highest_level_tile_count: int | None = None,
) -> tuple[dict[tuple[int, int], int], dict[int, list[TileRect]]]:
    if highest_level_tile_count is not None:
        rects = _make_count_plateau_rects(
            rng,
            cols=int(cols),
            rows=int(rows),
            highest_tile_count=int(highest_level_tile_count),
        )
    elif str(layout_family) == "side_plateau":
        rects = _make_side_rects(rng, cols=int(cols), rows=int(rows), active_max_level=int(active_max_level))
    elif str(layout_family) == "corner_plateau":
        rects = _make_corner_rects(rng, cols=int(cols), rows=int(rows), active_max_level=int(active_max_level))
    elif str(layout_family) == "split_field":
        rects = _make_split_rects(rng, cols=int(cols), rows=int(rows), active_max_level=int(active_max_level))
    else:
        rects = _make_concentric_rects(rng, cols=int(cols), rows=int(rows), active_max_level=int(active_max_level))
    rects = _snap_rects_one_tile_border_gaps(rects, cols=int(cols), rows=int(rows))
    grid = _blank_level_grid(cols=int(cols), rows=int(rows))
    for level in range(1, int(active_max_level) + 1):
        for rect in rects.get(int(level), []):
            _apply_rect(grid, rect, level=int(level), cols=int(cols), rows=int(rows))
    return grid, rects


def _project(layout: IsoLayout, col: float, row: float, level: int) -> IsoPoint:
    ox, oy = layout.origin_xy
    return (
        float(ox + (float(col) - float(row)) * float(layout.tile_w) * 0.5),
        float(oy + (float(col) + float(row)) * float(layout.tile_h) * 0.5 - int(level) * float(layout.level_px)),
    )


def _tile_vertices(layout: IsoLayout, col: int, row: int, level: int) -> tuple[IsoPoint, IsoPoint, IsoPoint, IsoPoint]:
    cx, cy = _project(layout, int(col), int(row), int(level))
    return (
        (cx, cy - float(layout.tile_h) * 0.5),
        (cx + float(layout.tile_w) * 0.5, cy),
        (cx, cy + float(layout.tile_h) * 0.5),
        (cx - float(layout.tile_w) * 0.5, cy),
    )


def _layout_for_scene(*, width: int, height: int, cols: int, rows: int, level_grid: Mapping[tuple[int, int], int]) -> IsoLayout:
    tile_w_by_width = float(width) * 0.9 * 2.0 / float(max(1, int(cols) + int(rows)))
    tile_w_by_height = float(height) * 1.05 * 2.0 / float(max(1, int(cols) + int(rows)))
    tile_w = max(58.0, min(96.0, tile_w_by_width, tile_w_by_height))
    tile_h = tile_w * 0.5
    level_px = max(22.0, tile_h * 0.82)
    raw_layout = IsoLayout(cols=int(cols), rows=int(rows), width=int(width), height=int(height), tile_w=tile_w, tile_h=tile_h, level_px=level_px, origin_xy=(0.0, 0.0))
    boxes: list[BBox] = []
    for (col, row), level in level_grid.items():
        boxes.append(_point_bbox(_tile_vertices(raw_layout, int(col), int(row), int(level))))
    min_x, min_y, max_x, max_y = _bbox_union(boxes)
    min_y -= 120.0
    max_y += 24.0 + level_px
    origin_x = (float(width) - (max_x - min_x)) * 0.5 - min_x
    origin_y = (float(height) - (max_y - min_y)) * 0.5 - min_y
    return IsoLayout(cols=int(cols), rows=int(rows), width=int(width), height=int(height), tile_w=tile_w, tile_h=tile_h, level_px=level_px, origin_xy=(origin_x, origin_y))


def _sample_farm_patches(
    rng: random.Random,
    *,
    cols: int,
    rows: int,
    active_levels: Sequence[int],
    level_grid: Mapping[tuple[int, int], int],
    blocked_tile_ids: set[str],
    skip_levels: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], str], set[str]]:
    """Place at most one connected farm terrain patch per active level without consuming candidate support."""

    patches: list[dict[str, Any]] = []
    terrain_by_cell: dict[tuple[int, int], str] = {}
    occupied: set[str] = set()
    skipped = set(int(level) for level in (skip_levels or set()))
    patch_sizes = ((2, 2), (2, 3), (3, 2), (3, 3))
    for level in active_levels:
        if int(level) in skipped:
            continue
        if rng.random() > 0.68:
            continue
        level_tile_total = sum(1 for value in level_grid.values() if int(value) == int(level))
        candidates: list[tuple[TileRect, tuple[str, ...]]] = []
        sizes = list(patch_sizes)
        rng.shuffle(sizes)
        for w, h in sizes:
            for y in range(1, max(1, int(rows) - int(h))):
                for x in range(1, max(1, int(cols) - int(w))):
                    rect = (int(x), int(y), int(w), int(h))
                    cells = _rect_cells(rect)
                    tile_ids = tuple(_tile_id(col, row) for col, row in cells)
                    if any(tile_id in blocked_tile_ids or tile_id in occupied for tile_id in tile_ids):
                        continue
                    if int(level_tile_total) - len(tile_ids) < 5:
                        continue
                    if all(int(level_grid[(col, row)]) == int(level) for col, row in cells):
                        candidates.append((rect, tile_ids))
        if not candidates:
            continue
        rect, tile_ids = rng.choice(candidates)
        terrain = str(rng.choice(FARM_PATCH_TERRAINS))
        patch_id = f"farm_patch_{len(patches):02d}"
        patches.append(
            {
                "patch_id": patch_id,
                "level": int(level),
                "terrain": terrain,
                "rect": [int(value) for value in rect],
                "tile_ids": list(tile_ids),
            }
        )
        occupied.update(tile_ids)
        for col, row in _rect_cells(rect):
            terrain_by_cell[(int(col), int(row))] = terrain
    if not patches:
        forced_levels = list(active_levels)
        rng.shuffle(forced_levels)
        for level in forced_levels:
            if int(level) in skipped:
                continue
            candidates = []
            level_tile_total = sum(1 for value in level_grid.values() if int(value) == int(level))
            for y in range(1, max(1, int(rows) - 2)):
                for x in range(1, max(1, int(cols) - 2)):
                    rect = (int(x), int(y), 2, 2)
                    cells = _rect_cells(rect)
                    tile_ids = tuple(_tile_id(col, row) for col, row in cells)
                    if any(tile_id in blocked_tile_ids for tile_id in tile_ids):
                        continue
                    if int(level_tile_total) - len(tile_ids) < 5:
                        continue
                    if all(int(level_grid[(col, row)]) == int(level) for col, row in cells):
                        candidates.append((rect, tile_ids))
            if not candidates:
                continue
            rect, tile_ids = rng.choice(candidates)
            terrain = str(rng.choice(FARM_PATCH_TERRAINS))
            patches.append(
                {
                    "patch_id": "farm_patch_00",
                    "level": int(level),
                    "terrain": terrain,
                    "rect": [int(value) for value in rect],
                    "tile_ids": list(tile_ids),
                }
            )
            occupied.update(tile_ids)
            for col, row in _rect_cells(rect):
                terrain_by_cell[(int(col), int(row))] = terrain
            break
    return patches, terrain_by_cell, occupied


def _terrain_colors(terrain: str, level: int, tone: IsometricIllustrationTone) -> tuple[RGB, RGB, RGB]:
    base_by_level = {
        0: (58, 124, 66),
        1: (101, 169, 80),
        2: (150, 207, 100),
        3: (171, 222, 113),
    }
    fill = base_by_level.get(int(level), base_by_level[0])
    if terrain == "crop":
        fill = (133, 92, 54)
    elif terrain == "soil":
        fill = (164, 116, 64)
    elif terrain == "flower":
        fill = (111, 172, 88)
    elif terrain == "pasture":
        fill = (103, 171, 92)
    return isometric_terrain_triplet(fill, tone)


def _draw_tile_top(
    draw: ImageDraw.ImageDraw,
    layout: IsoLayout,
    tile: IsoFarmsteadTile,
    rng: random.Random,
    tone: IsometricIllustrationTone,
) -> None:
    points = [(int(round(x)), int(round(y))) for x, y in tile.polygon_xy]
    fill, dark, light = _terrain_colors(tile.terrain, tile.level, tone)
    draw.polygon(points, fill=fill)
    top, right, bottom, left = tile.polygon_xy
    draw.line((top, right), fill=light, width=2)
    draw.line((right, bottom), fill=_shade(dark, 8), width=2)
    draw.line((bottom, left), fill=dark, width=2)
    draw.line((left, top), fill=_shade(dark, -4), width=2)
    cx, cy = tile.center_xy
    if tile.terrain == "crop":
        leaf = tint_isometric_semantic_rgb((63, 149, 62), tone, strength=0.04)
        for dx in (-0.17, 0.0, 0.17):
            px = int(round(cx + dx * layout.tile_w))
            draw.line((px, int(cy - 5), px, int(cy + 8)), fill=_shade(leaf, -35), width=2)
            draw.ellipse((px - 4, int(cy - 3), px + 3, int(cy + 5)), fill=leaf)
    elif tile.terrain == "soil":
        draw.line((int(cx - 10), int(cy + 2), int(cx + 10), int(cy - 2)), fill=tint_isometric_semantic_rgb((111, 72, 43), tone, strength=0.04), width=2)
    elif tile.terrain == "flower":
        blossom = rng.choice(((238, 87, 107), (239, 200, 73), (185, 105, 222)))
        for dx, dy in ((-7, -1), (3, 2), (9, -2)):
            draw.ellipse((int(cx + dx - 2), int(cy + dy - 2), int(cx + dx + 2), int(cy + dy + 2)), fill=blossom)
    elif tile.terrain == "pasture":
        for dx in (-0.18, 0.08, 0.21):
            px = int(round(cx + dx * layout.tile_w))
            draw.line((px, int(cy + 1), px + 5, int(cy - 5)), fill=tint_isometric_semantic_rgb((50, 128, 58), tone, strength=0.04), width=2)


def _draw_level_faces(
    draw: ImageDraw.ImageDraw,
    layout: IsoLayout,
    level_grid: Mapping[tuple[int, int], int],
    *,
    open_edges: set[tuple[int, int, str]],
    tone: IsometricIllustrationTone,
) -> list[dict[str, Any]]:
    """Draw visible vertical faces wherever adjacent terrain levels drop."""

    records: list[dict[str, Any]] = []
    for row in range(layout.rows):
        for col in range(layout.cols):
            level = int(level_grid[(col, row)])
            if level <= 0:
                continue
            top, right, bottom, left = _tile_vertices(layout, col, row, level)
            for side, edge, neighbor in (
                ("east", (right, bottom), (col + 1, row)),
                ("south", (bottom, left), (col, row + 1)),
            ):
                if (col, row, side) in open_edges:
                    continue
                neighbor_level = int(level_grid.get(neighbor, 0))
                drop = max(0, level - neighbor_level) * float(layout.level_px)
                if drop <= 0:
                    continue
                p0, p1 = edge
                face = (p0, p1, (p1[0], p1[1] + drop), (p0[0], p0[1] + drop))
                fill = tint_isometric_semantic_rgb((118, 92, 60) if side == "east" else (138, 103, 63), tone, strength=0.06)
                draw.polygon([(int(round(x)), int(round(y))) for x, y in face], fill=fill)
                draw.line((p0, p1), fill=tint_isometric_semantic_rgb((202, 159, 91), tone, strength=0.08), width=2)
                for offset in range(8, int(drop), 8):
                    draw.line((int(p0[0]), int(p0[1] + offset), int(p1[0]), int(p1[1] + offset)), fill=_shade(fill, -18), width=1)
                records.append(
                    {
                        "tile_id": _tile_id(col, row),
                        "side": side,
                        "from_level": int(level),
                        "to_level": int(neighbor_level),
                        "drop_px": round(float(drop), 3),
                        "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in face],
                    }
                )
    return records


def _add_entity(
    entities: list[IsoFarmsteadEntity],
    *,
    entity_id: str,
    public_name: str,
    object_type: str,
    tile_ids: Sequence[str],
    level: int,
    bbox: BBox,
    role: str,
    metadata: Mapping[str, Any],
) -> None:
    entities.append(
        IsoFarmsteadEntity(
            entity_id=str(entity_id),
            public_name=str(public_name),
            object_type=str(object_type),
            tile_ids=tuple(str(value) for value in tile_ids),
            level=int(level),
            bbox_xyxy=tuple(float(value) for value in bbox),
            point_xy=((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5),
            role=str(role),
            metadata=dict(metadata),
        )
    )


def _bbox_inside_canvas(bbox: Sequence[float], *, width: int, height: int) -> bool:
    return 0 <= float(bbox[0]) < float(bbox[2]) <= float(width) and 0 <= float(bbox[1]) < float(bbox[3]) <= float(height)


def _object_sprite_bbox(layout: IsoLayout, tile: IsoFarmsteadTile, *, object_type: str, subtype: str) -> BBox:
    cx, cy = tile.center_xy
    if str(object_type) == "tree":
        width = float(layout.tile_w) * 0.62
        height = float(layout.tile_w) * 1.18
        return (cx - width * 0.5, cy - height + float(layout.tile_h) * 0.28, cx + width * 0.5, cy + float(layout.tile_h) * 0.28)
    if str(object_type) == "farmer":
        width = float(layout.tile_w) * 0.5
        height = float(layout.tile_w) * 0.7
        return (cx - width * 0.5, cy - height + float(layout.tile_h) * 0.3, cx + width * 0.5, cy + float(layout.tile_h) * 0.3)
    if str(subtype) == "cow":
        width = float(layout.tile_w) * 0.96
        height = float(layout.tile_w) * 0.52
    elif str(subtype) == "chicken":
        width = float(layout.tile_w) * 0.46
        height = float(layout.tile_w) * 0.42
    else:
        width = float(layout.tile_w) * 0.6
        height = float(layout.tile_w) * 0.46
    return (cx - width * 0.5, cy - height * 0.72, cx + width * 0.5, cy + height * 0.28)


def _paste_sprite(image: Image.Image, bbox: Sequence[float], sprite: Image.Image) -> None:
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    if x1 <= x0 or y1 <= y0:
        return
    scaled = sprite.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    image.paste(scaled, (x0, y0), scaled)


def _tree_sprite(style: str) -> Image.Image:
    sprite = Image.new("RGBA", (16, 32), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite, "RGBA")
    leaf_by_style = {
        "oak": (44, 138, 76),
        "pine": (42, 118, 83),
        "maple": (176, 96, 58),
        "fruit_tree": (57, 142, 78),
    }
    draw_pixel_tree(
        sprite_draw,
        (0, 0, 1, 2),
        style=str(style),
        leaf_rgb=leaf_by_style.get(str(style), (44, 138, 76)),
        fruit_rgb=(220, 68, 61),
    )
    return sprite


def _animal_sprite(animal_type: str, *, facing: str) -> Image.Image:
    tile_width = 2 if str(animal_type) == "cow" else 1
    sprite = Image.new("RGBA", (16 * tile_width, 16), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite, "RGBA")
    draw_pixel_animal(sprite_draw, (0, 0, tile_width, 1), animal_type=str(animal_type), facing=str(facing))
    return sprite


def _farmer_sprite(*, rng: random.Random) -> Image.Image:
    sprite = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite, "RGBA")
    shirt_rgb = rng.choice(((76, 141, 82), (56, 118, 177), (180, 82, 65)))
    pants_rgb = rng.choice(((59, 76, 96), (76, 70, 58), (64, 88, 65)))
    skin_rgb = rng.choice(((225, 171, 109), (189, 127, 80), (237, 190, 133)))
    hair_rgb = rng.choice(((82, 50, 33), (116, 75, 35), (54, 38, 28)))
    draw_pixel_person(
        sprite_draw,
        (0, 0, 1, 1),
        skin_rgb=skin_rgb,
        shirt_rgb=shirt_rgb,
        pants_rgb=pants_rgb,
        hair_rgb=hair_rgb,
        gender_id="male",
        facing="down",
        person_variant_id="farmer",
    )
    return sprite


def _patches_with_bboxes(patches: Sequence[Mapping[str, Any]], tiles_by_id: Mapping[str, IsoFarmsteadTile]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for patch in patches:
        tile_ids = [str(value) for value in patch.get("tile_ids", []) if str(value) in tiles_by_id]
        if not tile_ids:
            continue
        bbox = _bbox_union([tiles_by_id[tile_id].bbox_xyxy for tile_id in tile_ids])
        resolved.append({**dict(patch), "bbox": [round(float(value), 3) for value in bbox]})
    return resolved


def _lower_tiles_adjacent_to_higher(tiles: Sequence[IsoFarmsteadTile]) -> set[str]:
    """Return lower tiles where object bases would read as attached to a terrace."""

    tiles_by_cell = {(int(tile.col), int(tile.row)): tile for tile in tiles}
    unsafe: set[str] = set()
    for tile in tiles:
        for delta_col in (-1, 0, 1):
            for delta_row in (-1, 0, 1):
                if delta_col == 0 and delta_row == 0:
                    continue
                neighbor = tiles_by_cell.get((int(tile.col) + delta_col, int(tile.row) + delta_row))
                if neighbor is not None and int(neighbor.level) > int(tile.level):
                    unsafe.add(str(tile.tile_id))
                    break
            if str(tile.tile_id) in unsafe:
                break
    return unsafe


def _draw_context_entities(
    image: Image.Image,
    layout: IsoLayout,
    tiles: Sequence[IsoFarmsteadTile],
    rng: random.Random,
    *,
    transition_tile_ids: set[str],
    farm_patch_tile_ids: set[str],
    required_entity_counts_by_level_type: Mapping[str, Mapping[int | str, int]] | None = None,
    object_unsafe_tile_ids: set[str] | None = None,
    reference_farmer_tile_id: str | None = None,
    reserved_tile_ids: set[str] | None = None,
) -> tuple[list[IsoFarmsteadEntity], set[str]]:
    """Draw reusable farm context objects and return occupied terrain tile ids."""

    entities: list[IsoFarmsteadEntity] = []
    occupied: set[str] = set()
    blocked = set(transition_tile_ids) | set(reserved_tile_ids or set())
    width, height = image.size
    tiles_by_id = {str(tile.tile_id): tile for tile in tiles}
    object_unsafe_tile_ids = set(object_unsafe_tile_ids or _lower_tiles_adjacent_to_higher(tiles))
    edge_tiles = [
        tile
        for tile in tiles
        if str(tile.tile_id) not in blocked
        and str(tile.tile_id) not in object_unsafe_tile_ids
        and str(tile.tile_id) not in farm_patch_tile_ids
        and str(tile.terrain) == "grass"
        and (tile.col <= 2 or tile.row <= 2 or tile.col >= layout.cols - 3 or tile.row >= layout.rows - 3)
    ]
    fallback_tree_tiles = [
        tile
        for tile in tiles
        if str(tile.tile_id) not in blocked
        and str(tile.tile_id) not in object_unsafe_tile_ids
        and str(tile.tile_id) not in farm_patch_tile_ids
        and str(tile.terrain) == "grass"
    ]
    required_counts_raw = required_entity_counts_by_level_type or {}

    def required_counts(object_type: str) -> dict[int, int]:
        raw_counts = required_counts_raw.get(str(object_type), {})
        return {int(level): int(count) for level, count in dict(raw_counts).items()}

    def candidate_tiles_for_level(candidates: Sequence[IsoFarmsteadTile], level: int) -> list[IsoFarmsteadTile]:
        return [tile for tile in candidates if int(tile.level) == int(level) and str(tile.tile_id) not in occupied]

    tree_required = required_counts("tree")
    tree_candidates = edge_tiles or fallback_tree_tiles
    rng.shuffle(tree_candidates)
    drawable: list[tuple[float, str, str, str, IsoFarmsteadTile, BBox]] = []
    tree_index = 0
    for level, required_count in sorted(tree_required.items()):
        candidates = candidate_tiles_for_level(fallback_tree_tiles, int(level))
        rng.shuffle(candidates)
        placed = 0
        for tile in candidates:
            if placed >= int(required_count):
                break
            style = str(rng.choice(TREE_STYLES))
            bbox = _object_sprite_bbox(layout, tile, object_type="tree", subtype=style)
            if not _bbox_inside_canvas(bbox, width=width, height=height):
                continue
            occupied.add(str(tile.tile_id))
            drawable.append((float(bbox[3]), f"tree_{tree_index:02d}", "tree", style, tile, bbox))
            tree_index += 1
            placed += 1
        if placed < int(required_count):
            raise ValueError(f"could not place required tree count {required_count} on level {level}")

    random_tree_candidates = [
        tile
        for tile in tree_candidates
        if str(tile.tile_id) not in occupied and int(tile.level) not in set(tree_required)
    ]
    tree_count = min(len(random_tree_candidates), max(0, rng.randint(3, 8) - sum(tree_required.values())))
    for tile in random_tree_candidates[:tree_count]:
        style = str(rng.choice(TREE_STYLES))
        bbox = _object_sprite_bbox(layout, tile, object_type="tree", subtype=style)
        if not _bbox_inside_canvas(bbox, width=width, height=height):
            continue
        occupied.add(str(tile.tile_id))
        drawable.append((float(bbox[3]), f"tree_{tree_index:02d}", "tree", style, tile, bbox))
        tree_index += 1

    animal_candidates = [
        tile
        for tile in tiles
        if str(tile.tile_id) not in blocked
        and str(tile.tile_id) not in object_unsafe_tile_ids
        and str(tile.tile_id) not in occupied
        and str(tile.terrain) in {"grass", "pasture"}
    ]
    rng.shuffle(animal_candidates)
    animal_required = required_counts("domestic_animal")
    animal_index = 0
    for level, required_count in sorted(animal_required.items()):
        candidates = candidate_tiles_for_level(animal_candidates, int(level))
        rng.shuffle(candidates)
        placed = 0
        for tile in candidates:
            if placed >= int(required_count):
                break
            animal_type = str(rng.choice(ANIMAL_TYPES))
            bbox = _object_sprite_bbox(layout, tile, object_type="animal", subtype=animal_type)
            if not _bbox_inside_canvas(bbox, width=width, height=height):
                continue
            occupied.add(str(tile.tile_id))
            drawable.append((float(bbox[3]), f"animal_{animal_index:02d}", "domestic_animal", animal_type, tile, bbox))
            animal_index += 1
            placed += 1
        if placed < int(required_count):
            raise ValueError(f"could not place required domestic_animal count {required_count} on level {level}")

    random_animal_candidates = [
        tile
        for tile in animal_candidates
        if str(tile.tile_id) not in occupied and int(tile.level) not in set(animal_required)
    ]
    animal_count = min(len(random_animal_candidates), max(0, rng.randint(3, 7) - sum(animal_required.values())))
    for tile in random_animal_candidates:
        if animal_index >= animal_count + sum(animal_required.values()):
            break
        animal_type = str(rng.choice(ANIMAL_TYPES))
        bbox = _object_sprite_bbox(layout, tile, object_type="animal", subtype=animal_type)
        if not _bbox_inside_canvas(bbox, width=width, height=height):
            continue
        occupied.add(str(tile.tile_id))
        drawable.append((float(bbox[3]), f"animal_{animal_index:02d}", "domestic_animal", animal_type, tile, bbox))
        animal_index += 1

    if reference_farmer_tile_id:
        farmer_tile = tiles_by_id.get(str(reference_farmer_tile_id))
        if farmer_tile is None:
            raise ValueError(f"reference_farmer_tile_id does not name a rendered tile: {reference_farmer_tile_id!r}")
        if str(farmer_tile.tile_id) in occupied or str(farmer_tile.tile_id) in blocked or str(farmer_tile.tile_id) in farm_patch_tile_ids:
            raise ValueError(f"reference farmer tile is unavailable: {reference_farmer_tile_id!r}")
        if str(farmer_tile.tile_id) in object_unsafe_tile_ids or str(farmer_tile.terrain) != "grass":
            raise ValueError(f"reference farmer tile is not a safe grass tile: {reference_farmer_tile_id!r}")
        farmer_bbox = _object_sprite_bbox(layout, farmer_tile, object_type="farmer", subtype="farmer")
        if not _bbox_inside_canvas(farmer_bbox, width=width, height=height):
            raise ValueError(f"reference farmer bbox is outside canvas: {reference_farmer_tile_id!r}")
        occupied.add(str(farmer_tile.tile_id))
        drawable.append((float(farmer_bbox[3]), "farmer_00", "farmer", "farmer", farmer_tile, farmer_bbox))

    for _, entity_id, object_type, subtype, tile, bbox in sorted(drawable, key=lambda item: item[0]):
        if object_type == "tree":
            _paste_sprite(image, bbox, _tree_sprite(subtype))
            public_name = f"{subtype.replace('_', ' ')}"
            metadata = {"base_level": int(tile.level), "tree_style": subtype}
        elif object_type == "farmer":
            _paste_sprite(image, bbox, _farmer_sprite(rng=rng))
            public_name = "farmer"
            metadata = {"base_level": int(tile.level), "person_variant_id": "farmer", "role": "reference"}
        else:
            facing = str(rng.choice(("left", "right")))
            _paste_sprite(image, bbox, _animal_sprite(subtype, facing=facing))
            public_name = subtype
            metadata = {"base_level": int(tile.level), "animal_type": subtype, "facing": facing}
        entity_role = "reference" if object_type == "farmer" else "context"
        _add_entity(
            entities,
            entity_id=entity_id,
            public_name=public_name,
            object_type=object_type,
            tile_ids=[str(tile.tile_id)],
            level=int(tile.level),
            bbox=bbox,
            role=entity_role,
            metadata=metadata,
        )
    return entities, occupied


def _build_tiles(
    layout: IsoLayout,
    level_grid: Mapping[tuple[int, int], int],
    farm_terrain_by_cell: Mapping[tuple[int, int], str],
) -> tuple[IsoFarmsteadTile, ...]:
    tiles: list[IsoFarmsteadTile] = []
    for row in range(layout.rows):
        for col in range(layout.cols):
            level = int(level_grid[(col, row)])
            terrain = str(farm_terrain_by_cell.get((int(col), int(row)), "grass"))
            polygon = _tile_vertices(layout, col, row, level)
            center = _project(layout, col, row, level)
            metadata: dict[str, Any] = {"candidate_allowed": terrain == "grass"}
            if terrain != "grass":
                metadata["farm_patch_terrain"] = terrain
            tiles.append(
                IsoFarmsteadTile(
                    tile_id=_tile_id(col, row),
                    col=int(col),
                    row=int(row),
                    level=int(level),
                    terrain=str(terrain),
                    polygon_xy=polygon,
                    bbox_xyxy=_point_bbox(polygon),
                    center_xy=center,
                    metadata=metadata,
                )
            )
    return tuple(sorted(tiles, key=lambda item: (item.row + item.col, item.col)))


def render_isometric_farmstead_scene(
    seed: int,
    *,
    width: int,
    height: int,
    canvas_profile: str = "",
    canvas_profile_probabilities: Mapping[str, float] | None = None,
    candidate_labels_by_tile_id: Mapping[str, str] | None = None,
    label_font_family: str | None = None,
    required_entity_counts_by_level_type: Mapping[str, Mapping[int | str, int]] | None = None,
    reference_farmer_tile_id: str | None = None,
    highest_level_tile_count: int | None = None,
    reserve_highest_level_tiles: bool = False,
    render_style_params: Mapping[str, Any] | None = None,
    render_style_defaults: Mapping[str, Any] | None = None,
) -> IsoFarmsteadScene:
    """Render a deterministic isometric farmstead with variable terrain levels."""

    rng = random.Random(int(seed))
    cols, rows = _profile_grid(int(width), int(height))
    active_max_level = 2 if highest_level_tile_count is not None else _sample_active_max_level(rng)
    layout_family = "count_plateau" if highest_level_tile_count is not None else _sample_layout_family(rng)
    active_levels = tuple(range(0, int(active_max_level) + 1))
    level_grid, level_shapes = _make_level_grid(
        rng,
        cols=cols,
        rows=rows,
        active_max_level=int(active_max_level),
        layout_family=str(layout_family),
        highest_level_tile_count=highest_level_tile_count,
    )
    transition_tile_ids: set[str] = set()
    reserved_highest_tile_ids = {
        _tile_id(col, row)
        for (col, row), level in level_grid.items()
        if bool(reserve_highest_level_tiles) and int(level) == int(active_max_level)
    }
    farm_patches, farm_terrain_by_cell, farm_patch_tile_ids = _sample_farm_patches(
        rng,
        cols=cols,
        rows=rows,
        active_levels=active_levels,
        level_grid=level_grid,
        blocked_tile_ids=set(transition_tile_ids),
        skip_levels={int(active_max_level)} if bool(reserve_highest_level_tiles) else set(),
    )
    layout = _layout_for_scene(width=int(width), height=int(height), cols=cols, rows=rows, level_grid=level_grid)
    tile_rng = random.Random(int(seed) + 17011)
    tiles = _build_tiles(layout, level_grid, farm_terrain_by_cell)
    tiles_by_id = {str(tile.tile_id): tile for tile in tiles}
    object_unsafe_tile_ids = _lower_tiles_adjacent_to_higher(tiles)
    farm_patches_with_bboxes = _patches_with_bboxes(farm_patches, tiles_by_id)
    tone = resolve_isometric_illustration_tone(
        params=dict(render_style_params or {}),
        render_defaults=dict(render_style_defaults or {}),
        instance_seed=int(seed),
        namespace=f"{SCENE_ID}:background_tone",
    )
    image = Image.new("RGB", (int(width), int(height)), tone.canvas_rgb)
    draw = ImageDraw.Draw(image, "RGBA")

    for tile in tiles:
        _draw_tile_top(draw, layout, tile, tile_rng, tone)
    face_records = _draw_level_faces(draw, layout, level_grid, open_edges=set(), tone=tone)
    transitions = ()
    entities, occupied_tile_ids = _draw_context_entities(
        image,
        layout,
        tiles,
        rng,
        transition_tile_ids=set(transition_tile_ids),
        farm_patch_tile_ids=set(farm_patch_tile_ids),
        required_entity_counts_by_level_type=required_entity_counts_by_level_type,
        object_unsafe_tile_ids=set(object_unsafe_tile_ids),
        reference_farmer_tile_id=reference_farmer_tile_id,
        reserved_tile_ids=set(reserved_highest_tile_ids),
    )

    labels = dict(candidate_labels_by_tile_id or {})
    label_bboxes: dict[str, BBox] = {}
    for tile_id, label in sorted(labels.items()):
        tile = next((candidate for candidate in tiles if candidate.tile_id == str(tile_id)), None)
        if tile is None:
            continue
        cx, cy = tile.center_xy
        box = (cx - 18.0, cy - 15.0, cx + 18.0, cy + 13.0)
        draw_label_badge(
            draw,
            str(label),
            box,
            font_family=label_font_family,
            fill=tone.label_fill_rgb,
            outline=tone.label_outline_rgb,
            text_fill=tone.label_text_rgb,
            radius=5,
            width=2,
        )
        label_bboxes[str(tile_id)] = tuple(float(value) for value in box)

    excluded_tile_ids = set(transition_tile_ids) | set(occupied_tile_ids) | set(farm_patch_tile_ids)
    trace = {
        "renderer_id": RENDERER_ID,
        "renderer_style": RENDERER_STYLE,
        "theme_id": "isometric_farmstead_elevation",
        "seed": int(seed),
        **tone.trace_metadata(),
        "canvas_profile": str(canvas_profile),
        "canvas_profile_probabilities": dict(canvas_profile_probabilities or {}),
        "canvas_size_px": [int(width), int(height)],
        "grid_cols": int(cols),
        "grid_rows": int(rows),
        "supported_levels": list(SUPPORTED_LEVELS),
        "levels": list(active_levels),
        "active_max_level": int(active_max_level),
        "layout_family": str(layout_family),
        "level_tile_counts": {
            str(level): sum(1 for value in level_grid.values() if int(value) == int(level))
            for level in active_levels
        },
        "highest_level_tile_count_request": None if highest_level_tile_count is None else int(highest_level_tile_count),
        "reserved_highest_level_tile_ids": sorted(reserved_highest_tile_ids),
        "projection": {
            "type": "2:1_isometric",
            "tile_size_px": [round(float(layout.tile_w), 3), round(float(layout.tile_h), 3)],
            "level_step_px": round(float(layout.level_px), 3),
            "origin_xy": [round(float(value), 3) for value in layout.origin_xy],
        },
        "level_shapes": {
            str(level): [[int(value) for value in rect] for rect in rects]
            for level, rects in sorted(level_shapes.items())
        },
        "farm_patches": farm_patches_with_bboxes,
        "farm_patch_tile_ids": sorted(farm_patch_tile_ids),
        "object_unsafe_low_adjacent_higher_tile_ids": sorted(object_unsafe_tile_ids),
        "reference_farmer_tile_id": str(reference_farmer_tile_id or ""),
        "transition_tile_ids": sorted(transition_tile_ids),
        "occupied_tile_ids": sorted(occupied_tile_ids),
        "eligible_tile_ids": [
            str(tile.tile_id)
            for tile in tiles
            if str(tile.tile_id) not in excluded_tile_ids and bool(tile.metadata.get("candidate_allowed", False))
        ],
        "tile_count": len(tiles),
        "entity_count": len(entities),
        "context_object_counts": {
            "tree": sum(1 for entity in entities if entity.object_type == "tree"),
            "domestic_animal": sum(1 for entity in entities if entity.object_type == "domestic_animal"),
        },
        "retaining_wall_faces": face_records,
        "label_bboxes_by_tile_id": {key: [round(float(value), 3) for value in bbox] for key, bbox in label_bboxes.items()},
    }
    return IsoFarmsteadScene(
        image=image,
        tiles=tiles,
        entities=tuple(sorted(entities, key=lambda item: item.entity_id)),
        transitions=tuple(sorted(transitions, key=lambda item: item.transition_id)),
        label_bboxes_by_tile_id=label_bboxes,
        trace=trace,
    )


def render_isometric_farmstead_labeled_scene_with_retry(
    *,
    instance_seed: int,
    max_attempts: int,
    sample: LabelTaskSampleSpec,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    font_seed_label: str,
    label_namespace_suffix: str,
    selection_fn: Callable[[IsoFarmsteadScene, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Render a base scene, select lettered tiles, and retry failed constraints."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_seed = int(instance_seed) + int(attempt) * 1009
            base_scene = render_isometric_farmstead_scene(
                scene_seed,
                width=sample.canvas_width,
                height=sample.canvas_height,
                canvas_profile=sample.canvas_profile,
                canvas_profile_probabilities=sample.canvas_profile_probabilities,
                render_style_params=params,
                render_style_defaults=rendering_defaults,
            )
            selection = dict(selection_fn(base_scene, scene_seed))
            candidates_by_label = {
                str(label): str(tile_id)
                for label, tile_id in dict(selection["candidate_tile_ids_by_label"]).items()
            }
            labels_by_tile_id = {str(tile_id): str(label) for label, tile_id in candidates_by_label.items()}
            label_font_trace = sample_visual_label_font_trace(
                namespace_prefix=str(font_seed_label),
                instance_seed=scene_seed,
                params={**dict(rendering_defaults), **dict(params)},
                namespace_suffix=str(label_namespace_suffix),
                explicit_key="terrain_tile_label_font_family",
                weights_key="terrain_tile_label_font_weights",
            )
            scene = render_isometric_farmstead_scene(
                scene_seed,
                width=sample.canvas_width,
                height=sample.canvas_height,
                canvas_profile=sample.canvas_profile,
                canvas_profile_probabilities=sample.canvas_profile_probabilities,
                candidate_labels_by_tile_id=labels_by_tile_id,
                label_font_family=str(label_font_trace["font_family"]),
                render_style_params=params,
                render_style_defaults=rendering_defaults,
                **dict(selection.get("render_kwargs", {})),
            )
            return {
                **selection,
                "scene_seed": int(scene_seed),
                "base_scene": base_scene,
                "scene": scene,
                "candidate_tile_ids_by_label": candidates_by_label,
                "selected_label": str(selection["selected_label"]),
                "label_font_trace": dict(label_font_trace),
            }
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"could not render labeled farmstead scene for {font_seed_label}: {last_error}") from last_error


__all__ = [
    "DEFAULT_CANDIDATE_LABELS",
    "RENDERER_ID",
    "RENDERER_STYLE",
    "SCENE_ID",
    "SUPPORTED_LEVELS",
    "render_isometric_farmstead_labeled_scene_with_retry",
    "render_isometric_farmstead_scene",
]
