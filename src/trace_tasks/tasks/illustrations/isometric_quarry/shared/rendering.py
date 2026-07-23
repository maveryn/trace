"""Renderer for the isometric quarry scene package."""

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
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import (
    draw_pixel_barrel,
    draw_pixel_crate,
    draw_pixel_person,
    draw_pixel_sign,
)
from trace_tasks.tasks.illustrations.shared.option_rendering import draw_label_badge, sample_visual_label_font_trace

from .sampling import LabelTaskSampleSpec
from .state import (
    BBox,
    IsoQuarryEntity,
    IsoQuarryScene,
    IsoQuarryTile,
    IsoPoint,
    IsoPolygon,
)


SCENE_ID = "isometric_quarry"
RENDERER_ID = "isometric_quarry_v0"
RENDERER_STYLE = "isometric_pixel_quarry"
SUPPORTED_LEVELS: tuple[int, ...] = (0, 1, 2, 3)
ACTIVE_MAX_LEVEL_SUPPORT: tuple[int, ...] = (1, 2)
DEFAULT_CANDIDATE_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
LAYOUT_FAMILIES: tuple[str, ...] = (
    "concentric_terrace",
    "side_plateau",
    "corner_plateau",
    "split_field",
)
QUARRY_PATCH_TERRAINS: tuple[str, ...] = ("gravel",)
QUARRY_OBJECT_TYPES: tuple[str, ...] = (
    "ore_vein",
    "barrel",
    "crate",
    "mine_cart",
    "sign",
)
COUNTABLE_QUARRY_OBJECT_TYPES: tuple[str, ...] = ("ore_vein", "mine_cart")
OBJECT_COUNT_DISTRACTOR_TYPES: tuple[str, ...] = ("barrel", "crate", "sign")
OBJECT_COUNT_QUARRY_OBJECT_TYPES: tuple[str, ...] = COUNTABLE_QUARRY_OBJECT_TYPES + OBJECT_COUNT_DISTRACTOR_TYPES

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
        raise ValueError(f"highest_tile_count {count} does not fit quarry grid {cols}x{rows}")
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


def _sample_quarry_patches(
    rng: random.Random,
    *,
    cols: int,
    rows: int,
    active_levels: Sequence[int],
    level_grid: Mapping[tuple[int, int], int],
    blocked_tile_ids: set[str],
    skip_levels: set[int] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], str], set[str]]:
    """Place connected quarry surface patches without consuming candidate support."""

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
        terrain = str(rng.choice(QUARRY_PATCH_TERRAINS))
        patch_id = f"quarry_patch_{len(patches):02d}"
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
            terrain = str(rng.choice(QUARRY_PATCH_TERRAINS))
            patches.append(
                {
                    "patch_id": "quarry_patch_00",
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
        0: (116, 111, 101),
        1: (145, 139, 125),
        2: (177, 169, 148),
        3: (194, 185, 160),
    }
    fill = base_by_level.get(int(level), base_by_level[0])
    if terrain == "gravel":
        fill = (133, 130, 120)
    return isometric_terrain_triplet(fill, tone)


def _draw_tile_top(
    draw: ImageDraw.ImageDraw,
    layout: IsoLayout,
    tile: IsoQuarryTile,
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
    if tile.terrain == "rock":
        speck = tint_isometric_semantic_rgb(rng.choice(((89, 88, 84), (198, 190, 169), (128, 122, 111))), tone, strength=0.04)
        for dx, dy in ((-10, 1), (6, -2), (13, 3)):
            draw.point((int(cx + dx), int(cy + dy)), fill=speck)
    elif tile.terrain == "gravel":
        for dx, dy in ((-12, 3), (-4, -2), (7, 1), (13, -3)):
            draw.rectangle((int(cx + dx), int(cy + dy), int(cx + dx + 2), int(cy + dy + 1)), fill=tint_isometric_semantic_rgb((88, 86, 81), tone, strength=0.04))


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
                fill = tint_isometric_semantic_rgb((94, 86, 76) if side == "east" else (117, 106, 91), tone, strength=0.06)
                draw.polygon([(int(round(x)), int(round(y))) for x, y in face], fill=fill)
                draw.line((p0, p1), fill=tint_isometric_semantic_rgb((186, 174, 146), tone, strength=0.08), width=2)
                for offset in range(8, int(drop), 8):
                    draw.line(
                        (int(p0[0]), int(p0[1] + offset), int(p1[0]), int(p1[1] + offset)),
                        fill=_shade(fill, -18),
                        width=1,
                    )
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
    entities: list[IsoQuarryEntity],
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
        IsoQuarryEntity(
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


def _object_sprite_bbox(layout: IsoLayout, tile: IsoQuarryTile, *, object_type: str, subtype: str) -> BBox:
    cx, cy = tile.center_xy
    if str(object_type) == "worker":
        width = float(layout.tile_w) * 0.5
        height = float(layout.tile_w) * 0.7
        return (cx - width * 0.5, cy - height + float(layout.tile_h) * 0.15, cx + width * 0.5, cy + float(layout.tile_h) * 0.15)
    if str(subtype) == "mine_cart":
        width = float(layout.tile_w) * 0.96
        height = float(layout.tile_w) * 0.52
    else:
        width = float(layout.tile_w) * 0.52
        height = float(layout.tile_w) * 0.48
    return (cx - width * 0.5, cy - height * 0.68, cx + width * 0.5, cy + height * 0.3)


def _paste_sprite(image: Image.Image, bbox: Sequence[float], sprite: Image.Image) -> None:
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox]
    if x1 <= x0 or y1 <= y0:
        return
    scaled = sprite.resize((x1 - x0, y1 - y0), Image.Resampling.NEAREST)
    image.paste(scaled, (x0, y0), scaled)


def _draw_countable_ore_vein(sprite_draw: ImageDraw.ImageDraw, *, ore_rgb: RGB) -> None:
    """Draw a high-contrast ore cluster that does not read as a generic rock."""

    outline = _shade(ore_rgb, -90)
    dark = _shade(ore_rgb, -42)
    light = _shade(ore_rgb, 52)
    base = (64, 61, 58)
    sprite_draw.ellipse((2, 11, 14, 15), fill=base, outline=(38, 37, 35))
    crystals = (
        ((3, 12), (5, 4), (8, 12)),
        ((6, 13), (9, 2), (12, 13)),
        ((9, 12), (13, 6), (15, 12)),
    )
    for points in crystals:
        sprite_draw.polygon(points, fill=ore_rgb, outline=outline)
    sprite_draw.line((5, 7, 6, 11), fill=light)
    sprite_draw.line((9, 5, 10, 12), fill=light)
    sprite_draw.polygon(((9, 2), (12, 13), (10, 13)), fill=dark)
    sprite_draw.point((4, 5), fill=(255, 250, 180))
    sprite_draw.point((10, 4), fill=(255, 250, 180))


def _draw_countable_mine_cart(sprite_draw: ImageDraw.ImageDraw) -> None:
    """Draw a broad mine cart with clear wheels and ore payload."""

    body = (70, 82, 88)
    body_dark = (24, 30, 35)
    body_light = (130, 146, 151)
    ore = (214, 155, 56)
    ore_dark = (120, 77, 28)
    sprite_draw.polygon(((4, 6), (28, 6), (25, 13), (7, 13)), fill=body, outline=body_dark)
    sprite_draw.polygon(((6, 4), (26, 4), (28, 7), (4, 7)), fill=body_light, outline=body_dark)
    sprite_draw.line((7, 8, 25, 8), fill=(168, 181, 185))
    for x in (7, 20):
        sprite_draw.ellipse((x, 11, x + 6, 16), fill=body_dark, outline=(0, 0, 0))
        sprite_draw.point((x + 3, 13), fill=body_light)
    for px, py in ((10, 5), (14, 4), (18, 5), (22, 6)):
        sprite_draw.rectangle((px - 1, py - 1, px + 2, py + 1), fill=ore, outline=ore_dark)


def _quarry_object_sprite(object_type: str, *, rng: random.Random) -> Image.Image:
    tile_width = 2 if str(object_type) == "mine_cart" else 1
    sprite = Image.new("RGBA", (16 * tile_width, 16), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite, "RGBA")
    if str(object_type) == "ore_vein":
        _draw_countable_ore_vein(
            sprite_draw,
            ore_rgb=rng.choice(((229, 176, 56), (99, 194, 224), (178, 129, 230))),
        )
    elif str(object_type) == "barrel":
        draw_pixel_barrel(sprite_draw, (0, 0, 1, 1), barrel_rgb=rng.choice(((151, 86, 45), (116, 92, 70), (97, 103, 106))))
    elif str(object_type) == "crate":
        draw_pixel_crate(sprite_draw, (0, 0, 1, 1))
    elif str(object_type) == "mine_cart":
        _draw_countable_mine_cart(sprite_draw)
    else:
        draw_pixel_sign(sprite_draw, (0, 0, 1, 1))
    return sprite


def _worker_sprite(*, rng: random.Random) -> Image.Image:
    sprite = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    sprite_draw = ImageDraw.Draw(sprite, "RGBA")
    draw_pixel_person(
        sprite_draw,
        (0, 0, 1, 1),
        skin_rgb=rng.choice(((225, 171, 109), (189, 127, 80), (237, 190, 133))),
        shirt_rgb=rng.choice(((210, 118, 51), (58, 126, 176), (117, 132, 74))),
        pants_rgb=rng.choice(((54, 67, 82), (73, 70, 62), (46, 82, 93))),
        hair_rgb=rng.choice(((82, 50, 33), (116, 75, 35), (54, 38, 28))),
        gender_id=str(rng.choice(("male", "female"))),
        facing="down",
        person_variant_id="worker",
    )
    return sprite


def _patches_with_bboxes(patches: Sequence[Mapping[str, Any]], tiles_by_id: Mapping[str, IsoQuarryTile]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for patch in patches:
        tile_ids = [str(value) for value in patch.get("tile_ids", []) if str(value) in tiles_by_id]
        if not tile_ids:
            continue
        bbox = _bbox_union([tiles_by_id[tile_id].bbox_xyxy for tile_id in tile_ids])
        resolved.append({**dict(patch), "bbox": [round(float(value), 3) for value in bbox]})
    return resolved


def _lower_tiles_adjacent_to_higher(tiles: Sequence[IsoQuarryTile]) -> set[str]:
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
    tiles: Sequence[IsoQuarryTile],
    rng: random.Random,
    *,
    transition_tile_ids: set[str],
    quarry_patch_tile_ids: set[str],
    required_entity_counts_by_level_type: Mapping[str, Mapping[int | str, int]] | None = None,
    context_object_types: Sequence[str] | None = None,
    object_unsafe_tile_ids: set[str] | None = None,
    reference_worker_tile_id: str | None = None,
    reserved_tile_ids: set[str] | None = None,
) -> tuple[list[IsoQuarryEntity], set[str]]:
    """Draw reusable quarry context objects and return occupied terrain tile ids."""

    entities: list[IsoQuarryEntity] = []
    occupied: set[str] = set()
    blocked = set(transition_tile_ids) | set(reserved_tile_ids or set())
    if reference_worker_tile_id:
        blocked.add(str(reference_worker_tile_id))
    width, height = image.size
    tiles_by_id = {str(tile.tile_id): tile for tile in tiles}
    object_unsafe_tile_ids = set(object_unsafe_tile_ids or _lower_tiles_adjacent_to_higher(tiles))
    drawable: list[tuple[float, str, str, str, IsoQuarryTile, BBox]] = []
    base_candidates = [
        tile
        for tile in tiles
        if str(tile.tile_id) not in blocked
        and str(tile.tile_id) not in object_unsafe_tile_ids
        and str(tile.tile_id) not in quarry_patch_tile_ids
        and str(tile.terrain) == "rock"
    ]
    required_counts_raw = required_entity_counts_by_level_type or {}
    required_subtypes = tuple(str(subtype) for subtype in dict(required_counts_raw))
    object_types = tuple(
        dict.fromkeys(
            str(subtype)
            for subtype in tuple(context_object_types or QUARRY_OBJECT_TYPES) + required_subtypes
            if str(subtype) in set(QUARRY_OBJECT_TYPES)
        )
    )
    if not object_types:
        raise ValueError("context_object_types must include at least one quarry object subtype")

    def required_counts(subtype: str) -> dict[int, int]:
        raw_counts = required_counts_raw.get(str(subtype), {})
        return {int(level): int(count) for level, count in dict(raw_counts).items()}

    def candidate_tiles_for_level(candidates: Sequence[IsoQuarryTile], level: int) -> list[IsoQuarryTile]:
        return [tile for tile in candidates if int(tile.level) == int(level) and str(tile.tile_id) not in occupied]

    quarry_object_index = 0
    required_levels_by_subtype: dict[str, set[int]] = {}
    for subtype in object_types:
        subtype_counts = required_counts(str(subtype))
        if subtype_counts:
            required_levels_by_subtype[str(subtype)] = set(int(level) for level in subtype_counts)
        for level, required_count in sorted(subtype_counts.items()):
            candidates = candidate_tiles_for_level(base_candidates, int(level))
            rng.shuffle(candidates)
            placed = 0
            for tile in candidates:
                if placed >= int(required_count):
                    break
                bbox = _object_sprite_bbox(layout, tile, object_type="quarry_object", subtype=str(subtype))
                if not _bbox_inside_canvas(bbox, width=width, height=height):
                    continue
                occupied.add(str(tile.tile_id))
                drawable.append((float(bbox[3]), f"quarry_object_{quarry_object_index:02d}", "quarry_object", str(subtype), tile, bbox))
                quarry_object_index += 1
                placed += 1
            if placed < int(required_count):
                raise ValueError(f"could not place required quarry object count {required_count} for {subtype} on level {level}")

    random_candidates = [tile for tile in base_candidates if str(tile.tile_id) not in occupied]
    rng.shuffle(random_candidates)
    required_total = len(drawable)
    requested_context_count = rng.randint(5, 9)
    if required_counts_raw:
        requested_context_count = max(int(requested_context_count), int(required_total) + 2)
    context_count = min(len(random_candidates) + int(required_total), int(requested_context_count))
    object_cycle = list(object_types)
    rng.shuffle(object_cycle)
    neutral_distractor_count = 0
    required_subtype_set = {str(subtype) for subtype in required_subtypes}
    for tile in random_candidates:
        if len(drawable) >= context_count:
            break
        available_subtypes = [
            subtype
            for subtype in object_cycle
            if int(tile.level) not in required_levels_by_subtype.get(str(subtype), set())
        ]
        neutral_subtypes = [subtype for subtype in available_subtypes if str(subtype) not in required_subtype_set]
        if required_counts_raw and neutral_distractor_count < 2 and neutral_subtypes:
            available_subtypes = neutral_subtypes
        if not available_subtypes:
            continue
        subtype = str(available_subtypes[(quarry_object_index + int(tile.col) + int(tile.row)) % len(available_subtypes)])
        bbox = _object_sprite_bbox(layout, tile, object_type="quarry_object", subtype=subtype)
        if not _bbox_inside_canvas(bbox, width=width, height=height):
            continue
        if str(subtype) not in required_subtype_set:
            neutral_distractor_count += 1
        occupied.add(str(tile.tile_id))
        drawable.append((float(bbox[3]), f"quarry_object_{quarry_object_index:02d}", "quarry_object", subtype, tile, bbox))
        quarry_object_index += 1

    if reference_worker_tile_id:
        worker_tile = tiles_by_id.get(str(reference_worker_tile_id))
        if worker_tile is None:
            raise ValueError(f"reference_worker_tile_id does not name a rendered tile: {reference_worker_tile_id!r}")
        if str(worker_tile.tile_id) in occupied or str(worker_tile.tile_id) in quarry_patch_tile_ids:
            raise ValueError(f"reference worker tile is unavailable: {reference_worker_tile_id!r}")
        if str(worker_tile.tile_id) in object_unsafe_tile_ids or str(worker_tile.terrain) != "rock":
            raise ValueError(f"reference worker tile is not a safe rock tile: {reference_worker_tile_id!r}")
        worker_bbox = _object_sprite_bbox(layout, worker_tile, object_type="worker", subtype="worker")
        if not _bbox_inside_canvas(worker_bbox, width=width, height=height):
            raise ValueError(f"reference worker bbox is outside canvas: {reference_worker_tile_id!r}")
        occupied.add(str(worker_tile.tile_id))
        drawable.append((float(worker_bbox[3]), "worker_00", "worker", "worker", worker_tile, worker_bbox))

    for _, entity_id, object_type, subtype, tile, bbox in sorted(drawable, key=lambda item: item[0]):
        if object_type == "worker":
            _paste_sprite(image, bbox, _worker_sprite(rng=rng))
            public_name = "worker"
            role = "reference"
            metadata = {"base_level": int(tile.level), "person_variant_id": "worker", "role": "reference"}
        else:
            _paste_sprite(image, bbox, _quarry_object_sprite(subtype, rng=rng))
            public_name = subtype.replace("_", " ")
            role = "context"
            metadata = {"base_level": int(tile.level), "quarry_object_type": subtype}
        _add_entity(
            entities,
            entity_id=entity_id,
            public_name=public_name,
            object_type=object_type,
            tile_ids=[str(tile.tile_id)],
            level=int(tile.level),
            bbox=bbox,
            role=role,
            metadata=metadata,
        )
    return entities, occupied


def _build_tiles(
    layout: IsoLayout,
    level_grid: Mapping[tuple[int, int], int],
    quarry_terrain_by_cell: Mapping[tuple[int, int], str],
) -> tuple[IsoQuarryTile, ...]:
    tiles: list[IsoQuarryTile] = []
    for row in range(layout.rows):
        for col in range(layout.cols):
            level = int(level_grid[(col, row)])
            terrain = str(quarry_terrain_by_cell.get((int(col), int(row)), "rock"))
            polygon = _tile_vertices(layout, col, row, level)
            center = _project(layout, col, row, level)
            metadata: dict[str, Any] = {"candidate_allowed": terrain == "rock"}
            if terrain != "rock":
                metadata["quarry_patch_terrain"] = terrain
            tiles.append(
                IsoQuarryTile(
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


def render_isometric_quarry_scene(
    seed: int,
    *,
    width: int,
    height: int,
    canvas_profile: str = "",
    canvas_profile_probabilities: Mapping[str, float] | None = None,
    candidate_labels_by_tile_id: Mapping[str, str] | None = None,
    label_font_family: str | None = None,
    required_entity_counts_by_level_type: Mapping[str, Mapping[int | str, int]] | None = None,
    context_object_types: Sequence[str] | None = None,
    quarry_patch_mode: str = "standard",
    reference_worker_tile_id: str | None = None,
    highest_level_tile_count: int | None = None,
    reserve_highest_level_tiles: bool = False,
    render_style_params: Mapping[str, Any] | None = None,
    render_style_defaults: Mapping[str, Any] | None = None,
) -> IsoQuarryScene:
    """Render a deterministic isometric quarry with variable terrain levels."""

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
    if str(quarry_patch_mode) == "none":
        quarry_patches: list[dict[str, Any]] = []
        quarry_terrain_by_cell: dict[tuple[int, int], str] = {}
        quarry_patch_tile_ids: set[str] = set()
    else:
        quarry_patches, quarry_terrain_by_cell, quarry_patch_tile_ids = _sample_quarry_patches(
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
    tiles = _build_tiles(layout, level_grid, quarry_terrain_by_cell)
    tiles_by_id = {str(tile.tile_id): tile for tile in tiles}
    object_unsafe_tile_ids = _lower_tiles_adjacent_to_higher(tiles)
    quarry_patches_with_bboxes = _patches_with_bboxes(quarry_patches, tiles_by_id)
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
        quarry_patch_tile_ids=set(quarry_patch_tile_ids),
        required_entity_counts_by_level_type=required_entity_counts_by_level_type,
        context_object_types=context_object_types,
        object_unsafe_tile_ids=set(object_unsafe_tile_ids),
        reference_worker_tile_id=reference_worker_tile_id,
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

    excluded_tile_ids = set(transition_tile_ids) | set(occupied_tile_ids) | set(quarry_patch_tile_ids)
    trace = {
        "renderer_id": RENDERER_ID,
        "renderer_style": RENDERER_STYLE,
        "theme_id": "isometric_quarry_elevation",
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
        "quarry_patches": quarry_patches_with_bboxes,
        "quarry_patch_mode": str(quarry_patch_mode),
        "quarry_patch_tile_ids": sorted(quarry_patch_tile_ids),
        "quarry_object_type_pool": list(context_object_types or QUARRY_OBJECT_TYPES),
        "object_unsafe_low_adjacent_higher_tile_ids": sorted(object_unsafe_tile_ids),
        "reference_worker_tile_id": str(reference_worker_tile_id or ""),
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
            "quarry_object": sum(1 for entity in entities if entity.object_type == "quarry_object"),
            "worker": sum(1 for entity in entities if entity.object_type == "worker"),
            **{
                subtype: sum(
                    1
                    for entity in entities
                    if entity.object_type == "quarry_object"
                    and str(entity.metadata.get("quarry_object_type", "")) == str(subtype)
                )
                for subtype in QUARRY_OBJECT_TYPES
            },
        },
        "retaining_wall_faces": face_records,
        "label_bboxes_by_tile_id": {key: [round(float(value), 3) for value in bbox] for key, bbox in label_bboxes.items()},
    }
    return IsoQuarryScene(
        image=image,
        tiles=tiles,
        entities=tuple(sorted(entities, key=lambda item: item.entity_id)),
        transitions=tuple(sorted(transitions, key=lambda item: item.transition_id)),
        label_bboxes_by_tile_id=label_bboxes,
        trace=trace,
    )


def render_isometric_quarry_labeled_scene_with_retry(
    *,
    instance_seed: int,
    max_attempts: int,
    sample: LabelTaskSampleSpec,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    font_seed_label: str,
    label_namespace_suffix: str,
    selection_fn: Callable[[IsoQuarryScene, int], Mapping[str, Any]],
) -> dict[str, Any]:
    """Render a base scene, select lettered tiles, and retry failed constraints."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            scene_seed = int(instance_seed) + int(attempt) * 1009
            base_scene = render_isometric_quarry_scene(
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
            scene = render_isometric_quarry_scene(
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
    raise RuntimeError(f"could not render labeled quarry scene for {font_seed_label}: {last_error}") from last_error


__all__ = [
    "DEFAULT_CANDIDATE_LABELS",
    "OBJECT_COUNT_QUARRY_OBJECT_TYPES",
    "RENDERER_ID",
    "RENDERER_STYLE",
    "SCENE_ID",
    "SUPPORTED_LEVELS",
    "render_isometric_quarry_labeled_scene_with_retry",
    "render_isometric_quarry_scene",
]
