"""Renderer for the isometric harbor illustration scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.illustrations.shared.isometric_visual_styles import (
    IsometricIllustrationTone,
    isometric_terrain_triplet,
    resolve_isometric_illustration_tone,
    tint_isometric_semantic_rgb,
)
from trace_tasks.tasks.illustrations.shared.option_rendering import draw_label_badge

from .state import BBox, IsoHarborEntity, IsoHarborScene, IsoHarborTile


SCENE_ID = "isometric_harbor"
RENDERER_ID = "isometric_harbor_v4"
SUPPORTED_CANVAS_PROFILES: Mapping[str, tuple[int, int, int, int, float, float]] = {
    "landscape": (16, 12, 60, 30, 0.5, 0.17),
    "square": (14, 14, 58, 29, 0.5, 0.18),
}
BOAT_SIDE_VALUES: tuple[str, ...] = ("left", "right")
BOAT_MOORING_STATUS_VALUES: tuple[str, ...] = ("moored", "open_water")
BOAT_HEADING_STATUS_VALUES: tuple[str, ...] = ("toward_shoreline", "away_from_shoreline")
DEFAULT_BOAT_CANDIDATE_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
OPEN_WATER_BOAT_ORIENTATIONS: tuple[str, ...] = ("dock_parallel", "dock_cross", "screen_horizontal", "screen_vertical")
BOAT_HEADING_STATUS_ORIENTATION: Mapping[str, str] = {
    "toward_shoreline": "shore_facing",
    "away_from_shoreline": "shore_away",
}
BOAT_COLOR_PALETTES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    ((164, 67, 50), (244, 196, 84)),
    ((35, 91, 143), (236, 108, 70)),
    ((65, 126, 97), (238, 187, 86)),
    ((128, 78, 142), (229, 178, 91)),
)
MIN_BOAT_BBOX_SIDE_PX = 24.5


def _shade(color: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def _clamp_bbox(bbox: Sequence[float], *, width: int, height: int) -> BBox:
    return (
        max(0.0, min(float(width), float(bbox[0]))),
        max(0.0, min(float(height), float(bbox[1]))),
        max(0.0, min(float(width), float(bbox[2]))),
        max(0.0, min(float(height), float(bbox[3]))),
    )


def _bbox_for_points(points: Sequence[Sequence[float]]) -> BBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _expand_bbox_to_min_side(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    min_side: float,
) -> BBox:
    """Expand a projected object bbox to a minimum side while keeping it on canvas."""

    x0, y0, x1, y1 = _clamp_bbox(bbox, width=int(width), height=int(height))
    target_w = min(float(width), max(float(x1) - float(x0), float(min_side)))
    target_h = min(float(height), max(float(y1) - float(y0), float(min_side)))
    cx = (float(x0) + float(x1)) / 2.0
    cy = (float(y0) + float(y1)) / 2.0

    new_x0 = cx - target_w / 2.0
    new_x1 = cx + target_w / 2.0
    new_y0 = cy - target_h / 2.0
    new_y1 = cy + target_h / 2.0

    if new_x0 < 0.0:
        new_x1 -= new_x0
        new_x0 = 0.0
    if new_x1 > float(width):
        overflow = new_x1 - float(width)
        new_x0 = max(0.0, new_x0 - overflow)
        new_x1 = float(width)
    if new_y0 < 0.0:
        new_y1 -= new_y0
        new_y0 = 0.0
    if new_y1 > float(height):
        overflow = new_y1 - float(height)
        new_y0 = max(0.0, new_y0 - overflow)
        new_y1 = float(height)
    return (new_x0, new_y0, new_x1, new_y1)


def _iso_center(
    col: int,
    row: int,
    *,
    tile_w: float,
    tile_h: float,
    origin_x: float,
    origin_y: float,
) -> tuple[float, float]:
    return (
        float(origin_x) + (float(col) - float(row)) * float(tile_w) / 2.0,
        float(origin_y) + (float(col) + float(row)) * float(tile_h) / 2.0,
    )


def _tile_polygon(cx: float, cy: float, *, tile_w: float, tile_h: float) -> tuple[tuple[float, float], ...]:
    return (
        (float(cx), float(cy) - float(tile_h) / 2.0),
        (float(cx) + float(tile_w) / 2.0, float(cy)),
        (float(cx), float(cy) + float(tile_h) / 2.0),
        (float(cx) - float(tile_w) / 2.0, float(cy)),
    )


def _profile_geometry(width: int, height: int, canvas_profile: str) -> tuple[int, int, float, float, float, float]:
    profile = str(canvas_profile or "")
    if profile not in SUPPORTED_CANVAS_PROFILES:
        profile = "landscape" if int(width) >= int(height) else "square"
    cols, rows, tile_w, tile_h, origin_x_ratio, origin_y_ratio = SUPPORTED_CANVAS_PROFILES[profile]
    return (
        int(cols),
        int(rows),
        float(tile_w),
        float(tile_h),
        float(width) * float(origin_x_ratio),
        float(height) * float(origin_y_ratio),
    )


def _dock_cells(cols: int, rows: int, rng: Any) -> tuple[set[tuple[int, int]], dict[str, Any]]:
    dock_width = 2
    jitter = int(rng.choice((-1, 0, 1))) if int(cols) >= 16 else int(rng.choice((-1, 0)))
    left_col = max(3, min(int(cols) - 5, int(cols) // 2 - 1 + jitter))
    row_start = 1
    row_end = int(rows) - 1
    cells: set[tuple[int, int]] = {
        (col, row)
        for col in range(left_col, left_col + dock_width)
        for row in range(row_start, row_end)
    }
    cap_row = int(row_start)
    for col in range(left_col - 1, left_col + dock_width + 1):
        if 0 <= int(col) < int(cols):
            cells.add((int(col), cap_row))
    return cells, {
        "left_col": int(left_col),
        "right_col": int(left_col + dock_width - 1),
        "row_start": int(row_start),
        "row_end_exclusive": int(row_end),
        "dock_width_tiles": int(dock_width),
    }


def _land_cells(cols: int, rows: int, dock_meta: Mapping[str, Any]) -> set[tuple[int, int]]:
    """Return shoreline terrain cells that anchor the dock to land."""

    shore_rows = range(0, min(int(rows), int(dock_meta["row_start"]) + 1))
    return {(int(col), int(row)) for row in shore_rows for col in range(int(cols))}


def _make_tiles(
    *,
    cols: int,
    rows: int,
    tile_w: float,
    tile_h: float,
    origin_x: float,
    origin_y: float,
    dock_cells: set[tuple[int, int]],
    land_cells: set[tuple[int, int]],
) -> tuple[IsoHarborTile, ...]:
    tiles: list[IsoHarborTile] = []
    for row in range(int(rows)):
        for col in range(int(cols)):
            cx, cy = _iso_center(col, row, tile_w=tile_w, tile_h=tile_h, origin_x=origin_x, origin_y=origin_y)
            polygon = _tile_polygon(cx, cy, tile_w=tile_w, tile_h=tile_h)
            cell = (int(col), int(row))
            if cell in dock_cells:
                terrain = "dock"
            elif cell in land_cells:
                terrain = "land"
            else:
                terrain = "water"
            tiles.append(
                IsoHarborTile(
                    tile_id=f"tile_{int(col):02d}_{int(row):02d}",
                    col=int(col),
                    row=int(row),
                    terrain=terrain,
                    walkable=str(terrain) == "dock",
                    polygon_xy=polygon,
                    bbox_xyxy=_bbox_for_points(polygon),
                    center_xy=(float(cx), float(cy)),
                    metadata={"candidate_allowed": str(terrain) == "dock"},
                )
            )
    return tuple(tiles)


def _draw_land_tile(draw: ImageDraw.ImageDraw, tile: IsoHarborTile, *, rng: Any, tone: IsometricIllustrationTone) -> None:
    green_shift = int(rng.randrange(-7, 8))
    fill = (126 + green_shift, 176 + green_shift, 103 + green_shift)
    if int(tile.row) > 0:
        fill = (194 + green_shift, 170 + green_shift, 105 + green_shift)
    fill, dark, light = isometric_terrain_triplet(fill, tone, shadow_delta=-42, light_delta=32)
    points = [(int(round(x)), int(round(y))) for x, y in tile.polygon_xy]
    draw.polygon(points, fill=fill)
    top, right, bottom, left = tile.polygon_xy
    draw.line((top, right), fill=light, width=2)
    draw.line((right, bottom), fill=_shade(dark, 8), width=2)
    draw.line((bottom, left), fill=dark, width=2)
    draw.line((left, top), fill=_shade(dark, -4), width=2)
    cx, cy = tile.center_xy
    if int(tile.row) == 0:
        for dx in (-0.2, 0.08, 0.22):
            px = int(round(cx + dx * (tile.bbox_xyxy[2] - tile.bbox_xyxy[0])))
            draw.line((px, int(cy + 2), px + 5, int(cy - 5)), fill=tint_isometric_semantic_rgb((53, 126, 58), tone, strength=0.04), width=2)
    else:
        draw.arc((cx - 11, cy - 2, cx + 12, cy + 7), 190, 350, fill=tint_isometric_semantic_rgb((143, 118, 74), tone, strength=0.04), width=1)


def _draw_water_tile(draw: ImageDraw.ImageDraw, tile: IsoHarborTile, *, rng: Any, tone: IsometricIllustrationTone) -> None:
    blue_shift = int(rng.randrange(-8, 9))
    fill = tint_isometric_semantic_rgb((42, 135 + blue_shift, 174 + blue_shift), tone, strength=0.045)
    outline = tint_isometric_semantic_rgb((35, 112, 154), tone, strength=0.06)
    draw.polygon(tile.polygon_xy, fill=fill, outline=outline)
    cx, cy = tile.center_xy
    wave = float(tile.bbox_xyxy[2] - tile.bbox_xyxy[0]) * 0.16
    draw.arc((cx - wave, cy - 3, cx + wave, cy + 7), 190, 350, fill=tint_isometric_semantic_rgb((120, 205, 220), tone, strength=0.04), width=1)


def _draw_dock_tile(draw: ImageDraw.ImageDraw, tile: IsoHarborTile, *, tone: IsometricIllustrationTone) -> None:
    fill = tint_isometric_semantic_rgb((163, 112, 66), tone, strength=0.06)
    outline = tint_isometric_semantic_rgb((92, 61, 36), tone, strength=0.06)
    draw.polygon(tile.polygon_xy, fill=fill, outline=outline)
    left, top, right, bottom = tile.bbox_xyxy
    cx, cy = tile.center_xy
    plank = tint_isometric_semantic_rgb((124, 80, 43), tone, strength=0.06)
    draw.line([(left + 8, cy), (cx, bottom - 2), (right - 8, cy)], fill=plank, width=1)
    draw.line([(cx, top + 2), (cx, bottom - 2)], fill=plank, width=1)


def _draw_post(draw: ImageDraw.ImageDraw, cx: float, cy: float, scale: float) -> BBox:
    w = float(scale) * 0.16
    h = float(scale) * 0.42
    bbox = (float(cx) - w, float(cy) - h, float(cx) + w, float(cy) + h * 0.25)
    draw.rectangle(bbox, fill=(91, 58, 33), outline=(49, 32, 21))
    draw.ellipse((bbox[0], bbox[1] - w * 0.45, bbox[2], bbox[1] + w * 0.45), fill=(123, 84, 45), outline=(49, 32, 21))
    return bbox


def _draw_crate(draw: ImageDraw.ImageDraw, cx: float, cy: float, scale: float) -> BBox:
    w = float(scale) * 0.28
    h = float(scale) * 0.22
    bbox = (float(cx) - w, float(cy) - h, float(cx) + w, float(cy) + h)
    draw.rectangle(bbox, fill=(170, 114, 57), outline=(82, 51, 27), width=2)
    draw.line([(bbox[0], bbox[1]), (bbox[2], bbox[3])], fill=(95, 57, 30), width=1)
    draw.line([(bbox[0], bbox[3]), (bbox[2], bbox[1])], fill=(95, 57, 30), width=1)
    return bbox


def _draw_barrel(draw: ImageDraw.ImageDraw, cx: float, cy: float, scale: float) -> BBox:
    w = float(scale) * 0.2
    h = float(scale) * 0.28
    bbox = (float(cx) - w, float(cy) - h, float(cx) + w, float(cy) + h)
    draw.ellipse((bbox[0], bbox[1], bbox[2], bbox[1] + h * 0.55), fill=(138, 89, 45), outline=(65, 42, 27))
    draw.rectangle((bbox[0], bbox[1] + h * 0.25, bbox[2], bbox[3] - h * 0.25), fill=(128, 80, 39), outline=(65, 42, 27))
    draw.ellipse((bbox[0], bbox[3] - h * 0.55, bbox[2], bbox[3]), fill=(103, 67, 36), outline=(65, 42, 27))
    draw.line([(bbox[0] + 2, cy), (bbox[2] - 2, cy)], fill=(51, 49, 43), width=1)
    return bbox


def _draw_polygon_with_outline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int,
) -> None:
    polygon = [(float(x), float(y)) for x, y in points]
    draw.polygon(polygon, fill=fill)
    draw.line([*polygon, polygon[0]], fill=outline, width=int(width))


def _boat_axes(orientation: str) -> tuple[tuple[float, float], tuple[float, float]]:
    if str(orientation) == "dock_cross":
        return (0.9, 0.45), (-0.9, 0.45)
    if str(orientation) == "shore_facing":
        return (0.9, -0.45), (0.9, 0.45)
    if str(orientation) == "shore_away":
        return (-0.9, 0.45), (0.9, 0.45)
    if str(orientation) == "screen_horizontal":
        return (1.0, 0.0), (0.0, 0.52)
    if str(orientation) == "screen_vertical":
        return (0.0, 0.78), (0.7, 0.0)
    return (-0.9, 0.45), (0.9, 0.45)


def _boat_iso_point(cx: float, cy: float, x: float, y: float, *, orientation: str = "dock_parallel") -> tuple[float, float]:
    """Project local boat coordinates onto the requested boat orientation axes."""

    x_axis, y_axis = _boat_axes(str(orientation))
    return (
        float(cx) + float(x) * float(x_axis[0]) + float(y) * float(y_axis[0]),
        float(cy) + float(x) * float(x_axis[1]) + float(y) * float(y_axis[1]),
    )


def _boat_local_polygon(
    cx: float,
    cy: float,
    coords: Sequence[tuple[float, float]],
    *,
    orientation: str = "dock_parallel",
) -> tuple[tuple[float, float], ...]:
    return tuple(_boat_iso_point(cx, cy, float(x), float(y), orientation=str(orientation)) for x, y in coords)


def _draw_local_line(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int],
    width: int,
    orientation: str = "dock_parallel",
) -> None:
    draw.line(
        (
            _boat_iso_point(cx, cy, *start, orientation=str(orientation)),
            _boat_iso_point(cx, cy, *end, orientation=str(orientation)),
        ),
        fill=fill,
        width=int(width),
    )


def _draw_local_poly(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    coords: Sequence[tuple[float, float]],
    *,
    fill: tuple[int, int, int],
    outline: tuple[int, int, int],
    width: int,
    orientation: str = "dock_parallel",
) -> tuple[tuple[float, float], ...]:
    points = _boat_local_polygon(cx, cy, coords, orientation=str(orientation))
    _draw_polygon_with_outline(draw, points, fill=fill, outline=outline, width=width)
    return points


def _draw_boat(
    draw: ImageDraw.ImageDraw,
    cx: float,
    cy: float,
    *,
    scale: float,
    boat_type: str,
    side: str,
    hull_fill: tuple[int, int, int],
    trim: tuple[int, int, int],
    orientation: str = "dock_parallel",
    draw_rope: bool = True,
) -> BBox:
    """Draw one countable boat hull; returned bbox tracks the boat body, not wake or rope."""

    length = float(scale) * (1.34 if str(boat_type) == "cargo_boat" else 1.18)
    beam = float(scale) * (0.55 if str(boat_type) == "cargo_boat" else 0.48)
    half_l = length * 0.5
    half_b = beam * 0.5
    side_drop = max(5.0, float(scale) * 0.12)

    if str(boat_type) == "cargo_boat":
        hull_coords = (
            (-half_l * 0.9, -half_b * 0.72),
            (-half_l * 0.62, -half_b),
            (half_l * 0.48, -half_b * 0.94),
            (half_l, 0.0),
            (half_l * 0.48, half_b * 0.94),
            (-half_l * 0.62, half_b),
            (-half_l * 0.9, half_b * 0.72),
            (-half_l * 0.94, 0.0),
        )
    else:
        hull_coords = (
            (-half_l * 0.88, -half_b * 0.58),
            (-half_l * 0.58, -half_b),
            (half_l * 0.48, -half_b * 0.94),
            (half_l, 0.0),
            (half_l * 0.48, half_b * 0.94),
            (-half_l * 0.58, half_b),
            (-half_l * 0.88, half_b * 0.58),
        )
    hull_screen = _boat_local_polygon(cx, cy, hull_coords, orientation=str(orientation))
    side_screen = tuple((x, y + side_drop) for x, y in hull_screen)
    if not bool(draw_rope):
        draw.ellipse(
            (
                min(x for x, _ in side_screen) - 3,
                min(y for _, y in side_screen) + 2,
                max(x for x, _ in side_screen) + 3,
                max(y for _, y in side_screen) + 8,
            ),
            fill=(28, 95, 128),
        )
        draw.arc(
            (
                min(x for x, _ in side_screen) - 6,
                max(y for _, y in side_screen) - 9,
                max(x for x, _ in side_screen) + 7,
                max(y for _, y in side_screen) + 12,
            ),
            180,
            350,
            fill=(122, 207, 221),
            width=1,
        )
    draw.polygon(side_screen, fill=_shade(hull_fill, -42))
    draw.line([*side_screen, side_screen[0]], fill=(30, 37, 36), width=2)
    _draw_polygon_with_outline(draw, hull_screen, fill=hull_fill, outline=(28, 30, 29), width=3)
    deck_coords = (
        (-half_l * 0.62, -half_b * 0.52),
        (half_l * 0.3, -half_b * 0.5),
        (half_l * 0.72, 0.0),
        (half_l * 0.3, half_b * 0.5),
        (-half_l * 0.62, half_b * 0.52),
        (-half_l * 0.82, 0.0),
    )
    deck_fill = (236, 225, 195) if str(boat_type) == "cargo_boat" else (219, 174, 105)
    _draw_local_poly(draw, cx, cy, deck_coords, fill=deck_fill, outline=trim, width=2, orientation=str(orientation))
    _draw_local_line(
        draw,
        cx,
        cy,
        (-half_l * 0.68, 0.0),
        (half_l * 0.68, 0.0),
        fill=_shade(trim, -28),
        width=2,
        orientation=str(orientation),
    )
    _draw_local_line(
        draw,
        cx,
        cy,
        (half_l * 0.5, -half_b * 0.38),
        (half_l * 0.72, 0.0),
        fill=(245, 245, 232),
        width=2,
        orientation=str(orientation),
    )
    bow_marker = _boat_local_polygon(
        cx,
        cy - 0.5,
        (
            (half_l * 0.5, -half_b * 0.62),
            (half_l * 1.1, 0.0),
            (half_l * 0.5, half_b * 0.62),
        ),
        orientation=str(orientation),
    )
    draw.polygon(bow_marker, fill=(255, 255, 245), outline=(24, 29, 30))
    stern_line = (
        _boat_iso_point(cx, cy, -half_l * 0.82, -half_b * 0.46, orientation=str(orientation)),
        _boat_iso_point(cx, cy, -half_l * 0.82, half_b * 0.46, orientation=str(orientation)),
    )
    draw.line(stern_line, fill=(22, 28, 28), width=2)
    if str(boat_type) == "cargo_boat":
        cabin = _draw_local_poly(
            draw,
            cx,
            cy - side_drop * 0.7,
            (
                (-half_l * 0.55, -half_b * 0.36),
                (-half_l * 0.18, -half_b * 0.36),
                (-half_l * 0.08, -half_b * 0.03),
                (-half_l * 0.45, 0.0),
            ),
            fill=(241, 237, 220),
            outline=(58, 62, 61),
            width=2,
            orientation=str(orientation),
        )
        roof = tuple((x, y - side_drop * 0.75) for x, y in cabin)
        draw.polygon(roof, fill=_shade(trim, 16))
        draw.line([*roof, roof[0]], fill=(58, 62, 61), width=1)
        for wx in (-0.45, -0.31):
            window = _boat_local_polygon(
                cx,
                cy - side_drop * 0.4,
                (
                    (half_l * wx, -half_b * 0.28),
                    (half_l * (wx + 0.08), -half_b * 0.28),
                    (half_l * (wx + 0.08), -half_b * 0.08),
                    (half_l * wx, -half_b * 0.08),
                ),
                orientation=str(orientation),
            )
            draw.polygon(window, fill=(65, 140, 170), outline=(35, 72, 86))
        cargo_colors = ((222, 77, 57), (241, 179, 56), (75, 153, 87))
        for index, ox in enumerate((-0.02, 0.2, 0.42)):
            _draw_local_poly(
                draw,
                cx,
                cy - 1.0,
                (
                    (half_l * ox, half_b * 0.05),
                    (half_l * (ox + 0.16), half_b * 0.05),
                    (half_l * (ox + 0.16), half_b * 0.38),
                    (half_l * ox, half_b * 0.38),
                ),
                fill=cargo_colors[index % len(cargo_colors)],
                outline=(73, 58, 45),
                width=1,
                orientation=str(orientation),
            )
    else:
        for offset in (-0.32, 0.0, 0.32):
            _draw_local_line(
                draw,
                cx,
                cy,
                (half_l * offset, -half_b * 0.45),
                (half_l * offset, half_b * 0.45),
                fill=(92, 58, 34),
                width=3,
                orientation=str(orientation),
            )
        _draw_local_line(
            draw,
            cx,
            cy,
            (-half_l * 0.28, -half_b * 0.86),
            (half_l * 0.28, half_b * 0.86),
            fill=(96, 63, 38),
            width=2,
            orientation=str(orientation),
        )
        _draw_local_line(
            draw,
            cx,
            cy,
            (-half_l * 0.12, half_b * 0.86),
            (half_l * 0.42, -half_b * 0.86),
            fill=(96, 63, 38),
            width=2,
            orientation=str(orientation),
        )
    if bool(draw_rope):
        rope_target = _boat_iso_point(
            cx,
            cy,
            0.0,
            -half_b * 1.25 if str(side) == "left" else half_b * 1.25,
            orientation=str(orientation),
        )
        rope_start = _boat_iso_point(cx, cy, -half_l * 0.08, 0.0, orientation=str(orientation))
        draw.line([rope_start, rope_target], fill=(231, 218, 178), width=2)
    bbox = _bbox_for_points(hull_screen)
    pad = 5.0
    return (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)


def _boat_half_length(*, scale: float, boat_type: str) -> float:
    """Return local half-length used by the boat renderer."""

    return float(scale) * (1.34 if str(boat_type) == "cargo_boat" else 1.18) * 0.5


def _add_entity(
    entities: list[IsoHarborEntity],
    *,
    entity_id: str,
    public_name: str,
    object_type: str,
    tile_ids: Sequence[str],
    bbox: Sequence[float],
    point: Sequence[float],
    role: str,
    metadata: Mapping[str, Any],
    canvas_size: tuple[int, int],
) -> None:
    width, height = canvas_size
    entity_bbox = _clamp_bbox(bbox, width=int(width), height=int(height))
    if str(object_type) == "boat":
        entity_bbox = _expand_bbox_to_min_side(
            entity_bbox,
            width=int(width),
            height=int(height),
            min_side=MIN_BOAT_BBOX_SIDE_PX,
        )
    entities.append(
        IsoHarborEntity(
            entity_id=str(entity_id),
            public_name=str(public_name),
            object_type=str(object_type),
            tile_ids=tuple(str(value) for value in tile_ids),
            bbox_xyxy=entity_bbox,
            point_xy=(round(float(point[0]), 3), round(float(point[1]), 3)),
            role=str(role),
            metadata=dict(metadata),
        )
    )


def _select_boat_rows(
    *,
    rng: Any,
    dock_meta: Mapping[str, Any],
    side: str,
    count: int,
) -> list[int]:
    rows = list(range(int(dock_meta["row_start"]) + 1, int(dock_meta["row_end_exclusive"]) - 1))
    rows = [row for index, row in enumerate(rows) if index % 2 == (0 if str(side) == "left" else 1)]
    if len(rows) < int(count):
        rows = list(range(int(dock_meta["row_start"]) + 1, int(dock_meta["row_end_exclusive"]) - 1))
    rng.shuffle(rows)
    return sorted(rows[: int(count)])


def _split_moored_count_by_side(*, rng: Any, total: int) -> dict[str, int]:
    total_count = max(0, min(10, int(total)))
    min_left = max(0, int(total_count) - 5)
    max_left = min(5, int(total_count))
    left = int(rng.randrange(int(min_left), int(max_left) + 1)) if int(max_left) >= int(min_left) else int(min_left)
    return {"left": int(left), "right": int(total_count) - int(left)}


def _draw_context_objects(
    *,
    draw: ImageDraw.ImageDraw,
    rng: Any,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_cells: set[tuple[int, int]],
    dock_meta: Mapping[str, Any],
    entities: list[IsoHarborEntity],
    canvas_size: tuple[int, int],
    tile_w: float,
) -> None:
    """Place non-answer cargo props on dock tiles without changing boat-side counts."""

    usable = [
        cell
        for cell in sorted(dock_cells)
        if int(dock_meta["row_start"]) + 1 <= int(cell[1]) < int(dock_meta["row_end_exclusive"]) - 1
    ]
    rng.shuffle(usable)
    context_count = min(len(usable), int(rng.randrange(4, 8)))
    for index, cell in enumerate(usable[:context_count]):
        tile = tiles_by_cell[cell]
        cx, cy = tile.center_xy
        offset_x = float(rng.choice((-0.16, 0.16))) * float(tile_w)
        offset_y = float(rng.choice((-0.08, 0.06))) * float(tile_w)
        object_type = str(rng.choice(("crate", "barrel")))
        if object_type == "crate":
            bbox = _draw_crate(draw, cx + offset_x, cy + offset_y, tile_w)
            name = "crate"
        else:
            bbox = _draw_barrel(draw, cx + offset_x, cy + offset_y, tile_w)
            name = "barrel"
        _add_entity(
            entities,
            entity_id=f"{object_type}_{index:02d}",
            public_name=name,
            object_type=object_type,
            tile_ids=(tile.tile_id,),
            bbox=bbox,
            point=(cx + offset_x, cy + offset_y),
            role="context",
            metadata={"base_tile_id": str(tile.tile_id)},
            canvas_size=canvas_size,
        )


def _draw_dock_posts(
    *,
    draw: ImageDraw.ImageDraw,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    entities: list[IsoHarborEntity],
    canvas_size: tuple[int, int],
    tile_w: float,
) -> None:
    rows = range(int(dock_meta["row_start"]) + 1, int(dock_meta["row_end_exclusive"]), 3)
    for index, row in enumerate(rows):
        for side, col in (("left", int(dock_meta["left_col"])), ("right", int(dock_meta["right_col"]))):
            tile = tiles_by_cell.get((int(col), int(row)))
            if tile is None:
                continue
            cx, cy = tile.center_xy
            x_offset = -0.42 * float(tile_w) if side == "left" else 0.42 * float(tile_w)
            bbox = _draw_post(draw, cx + x_offset, cy, tile_w)
            _add_entity(
                entities,
                entity_id=f"dock_post_{side}_{index:02d}",
                public_name="dock post",
                object_type="dock_post",
                tile_ids=(tile.tile_id,),
                bbox=bbox,
                point=(cx + x_offset, cy),
                role="context",
                metadata={"dock_side": side, "base_tile_id": str(tile.tile_id)},
                canvas_size=canvas_size,
            )


def _draw_boats(
    *,
    draw: ImageDraw.ImageDraw,
    rng: Any,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    required_boat_counts_by_side: Mapping[str, int],
    entities: list[IsoHarborEntity],
    canvas_size: tuple[int, int],
    tile_w: float,
) -> dict[str, int]:
    """Draw exact side-bound boat counts and record each boat's dock-side binding."""

    counts = {side: int(required_boat_counts_by_side.get(side, rng.randrange(0, 4))) for side in BOAT_SIDE_VALUES}
    for side in BOAT_SIDE_VALUES:
        counts[side] = max(0, min(5, int(counts[side])))
    side_to_water_col = {
        "left": int(dock_meta["left_col"]) - 1,
        "right": int(dock_meta["right_col"]) + 1,
    }
    entity_index = 0
    for side in BOAT_SIDE_VALUES:
        rows = _select_boat_rows(rng=rng, dock_meta=dock_meta, side=side, count=int(counts[side]))
        for row in rows:
            water_tile = tiles_by_cell.get((int(side_to_water_col[side]), int(row)))
            dock_tile = tiles_by_cell.get((int(dock_meta["left_col" if side == "left" else "right_col"]), int(row)))
            if water_tile is None or dock_tile is None:
                continue
            cx, cy = water_tile.center_xy
            side_offset = 0.18 * float(tile_w) if side == "left" else -0.18 * float(tile_w)
            boat_type = str(rng.choice(("rowboat", "cargo_boat")))
            hull_fill, trim = rng.choice(BOAT_COLOR_PALETTES)
            bbox = _draw_boat(
                draw,
                cx + side_offset,
                cy,
                scale=tile_w,
                boat_type=boat_type,
                side=side,
                hull_fill=tuple(hull_fill),
                trim=tuple(trim),
            )
            _add_entity(
                entities,
                entity_id=f"boat_{entity_index:02d}",
                public_name="boat",
                object_type="boat",
                tile_ids=(water_tile.tile_id, dock_tile.tile_id),
                bbox=bbox,
                point=(cx + side_offset, cy),
                role="queryable",
                metadata={
                    "boat_type": boat_type,
                    "hull_rgb": [int(value) for value in hull_fill],
                    "trim_rgb": [int(value) for value in trim],
                    "mooring_status": "moored",
                    "orientation": "dock_parallel",
                    "dock_side": side,
                    "water_tile_id": str(water_tile.tile_id),
                    "dock_tile_id": str(dock_tile.tile_id),
                },
                canvas_size=canvas_size,
            )
            entity_index += 1
    return counts


def _open_water_candidate_cells(
    *,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    cols: int,
    rows: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    left_col = int(dock_meta["left_col"])
    right_col = int(dock_meta["right_col"])
    for row in range(max(2, int(dock_meta["row_start"]) + 2), max(2, int(rows) - 1)):
        for col in range(1, max(1, int(cols) - 1)):
            tile = tiles_by_cell.get((int(col), int(row)))
            if tile is None or str(tile.terrain) != "water":
                continue
            if int(left_col) - 2 <= int(col) <= int(right_col) + 2:
                continue
            candidates.append((int(col), int(row)))
    return candidates


def _heading_status_candidate_cells(
    *,
    rng: Any,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    cols: int,
    rows: int,
    count: int,
) -> list[tuple[int, int]]:
    """Return separated open-water cells for shoreline-heading boats."""

    candidates = _open_water_candidate_cells(
        tiles_by_cell=tiles_by_cell,
        dock_meta=dock_meta,
        cols=int(cols),
        rows=int(rows),
    )
    rng.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(int(cell[0]) - int(other[0])) + abs(int(cell[1]) - int(other[1])) >= 2 for other in selected):
            selected.append((int(cell[0]), int(cell[1])))
        if len(selected) >= int(count):
            break
    if len(selected) < int(count):
        for cell in candidates:
            if (int(cell[0]), int(cell[1])) not in selected:
                selected.append((int(cell[0]), int(cell[1])))
            if len(selected) >= int(count):
                break
    return selected


def _draw_open_water_boats(
    *,
    draw: ImageDraw.ImageDraw,
    rng: Any,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    count: int,
    entities: list[IsoHarborEntity],
    canvas_size: tuple[int, int],
    tile_w: float,
    cols: int,
    rows: int,
) -> int:
    """Draw free-floating boats away from the dock and shoreline."""

    target_count = max(0, min(6, int(count)))
    if int(target_count) <= 0:
        return 0
    candidates = _open_water_candidate_cells(tiles_by_cell=tiles_by_cell, dock_meta=dock_meta, cols=int(cols), rows=int(rows))
    rng.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    for cell in candidates:
        if all(abs(int(cell[0]) - int(other[0])) + abs(int(cell[1]) - int(other[1])) >= 2 for other in selected):
            selected.append((int(cell[0]), int(cell[1])))
        if len(selected) >= int(target_count):
            break
    entity_offset = sum(1 for entity in entities if str(entity.object_type) == "boat")
    for index, cell in enumerate(selected):
        tile = tiles_by_cell[cell]
        cx, cy = tile.center_xy
        offset_x = float(rng.choice((-0.1, 0.0, 0.1))) * float(tile_w)
        offset_y = float(rng.choice((-0.08, 0.0, 0.08))) * float(tile_w)
        boat_type = str(rng.choice(("rowboat", "cargo_boat")))
        orientation = str(rng.choice(OPEN_WATER_BOAT_ORIENTATIONS))
        hull_fill, trim = rng.choice(BOAT_COLOR_PALETTES)
        bbox = _draw_boat(
            draw,
            cx + offset_x,
            cy + offset_y,
            scale=float(tile_w) * 0.94,
            boat_type=boat_type,
            side="",
            hull_fill=tuple(hull_fill),
            trim=tuple(trim),
            orientation=orientation,
            draw_rope=False,
        )
        _add_entity(
            entities,
            entity_id=f"open_water_boat_{int(entity_offset) + int(index):02d}",
            public_name="boat",
            object_type="boat",
            tile_ids=(tile.tile_id,),
            bbox=bbox,
            point=(cx + offset_x, cy + offset_y),
            role="queryable",
            metadata={
                "boat_type": boat_type,
                "hull_rgb": [int(value) for value in hull_fill],
                "trim_rgb": [int(value) for value in trim],
                "mooring_status": "open_water",
                "orientation": orientation,
                "water_tile_id": str(tile.tile_id),
            },
            canvas_size=canvas_size,
        )
    return len(selected)


def _draw_heading_status_boats(
    *,
    draw: ImageDraw.ImageDraw,
    rng: Any,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    required_heading_status_counts: Mapping[str, int],
    entities: list[IsoHarborEntity],
    canvas_size: tuple[int, int],
    tile_w: float,
    cols: int,
    rows: int,
) -> dict[str, int]:
    """Draw open-water boats with exact shoreline-relative heading-status counts."""

    counts = {status: int(required_heading_status_counts.get(status, 0)) for status in BOAT_HEADING_STATUS_VALUES}
    if any(int(value) < 0 for value in counts.values()):
        raise ValueError("heading status counts must be nonnegative")
    total_count = sum(int(value) for value in counts.values())
    if int(total_count) <= 0 or int(total_count) > 6:
        raise ValueError("heading status counts must sum to 1..6")

    statuses: list[str] = []
    for status in BOAT_HEADING_STATUS_VALUES:
        statuses.extend([str(status)] * int(counts[status]))
    rng.shuffle(statuses)

    selected = _heading_status_candidate_cells(
        rng=rng,
        tiles_by_cell=tiles_by_cell,
        dock_meta=dock_meta,
        cols=int(cols),
        rows=int(rows),
        count=len(statuses),
    )
    if len(selected) < len(statuses):
        raise ValueError("not enough separated open-water cells for heading-status boats")

    for index, (status, cell) in enumerate(zip(statuses, selected)):
        tile = tiles_by_cell[cell]
        cx, cy = tile.center_xy
        offset_x = float(rng.choice((-0.04, 0.0, 0.04))) * float(tile_w)
        offset_y = float(rng.choice((-0.02, 0.0, 0.02))) * float(tile_w)
        boat_type = "rowboat"
        orientation = str(BOAT_HEADING_STATUS_ORIENTATION[str(status)])
        hull_fill, trim = BOAT_COLOR_PALETTES[int(index) % len(BOAT_COLOR_PALETTES)]
        if int(index) >= len(BOAT_COLOR_PALETTES):
            hull_fill = _shade(hull_fill, int(rng.randrange(-18, 19)))
        bbox = _draw_boat(
            draw,
            float(cx) + float(offset_x),
            float(cy) + float(offset_y),
            scale=float(tile_w) * 0.95,
            boat_type=boat_type,
            side="",
            hull_fill=tuple(hull_fill),
            trim=tuple(trim),
            orientation=orientation,
            draw_rope=False,
        )
        _add_entity(
            entities,
            entity_id=f"heading_boat_{int(index):02d}",
            public_name="boat",
            object_type="boat",
            tile_ids=(tile.tile_id,),
            bbox=bbox,
            point=(float(cx) + float(offset_x), float(cy) + float(offset_y)),
            role="queryable",
            metadata={
                "boat_type": boat_type,
                "hull_rgb": [int(value) for value in hull_fill],
                "trim_rgb": [int(value) for value in trim],
                "mooring_status": "open_water",
                "heading_status": str(status),
                "orientation": orientation,
                "water_tile_id": str(tile.tile_id),
            },
            canvas_size=canvas_size,
        )
    return {str(status): int(counts[str(status)]) for status in BOAT_HEADING_STATUS_VALUES}


def _shoreline_candidate_rows(*, rows: int, count: int) -> list[int]:
    """Return row indices from nearest-to-farthest visible shoreline distance."""

    usable = list(range(2, max(2, int(rows) - 1)))
    if len(usable) < int(count):
        raise ValueError("not enough open-water rows for shoreline candidates")
    if int(count) == 1:
        return [usable[0]]
    indices = [
        round(index * (len(usable) - 1) / max(1, int(count) - 1))
        for index in range(int(count))
    ]
    selected: list[int] = []
    for index in indices:
        row = usable[int(index)]
        if row not in selected:
            selected.append(row)
    for row in usable:
        if len(selected) >= int(count):
            break
        if row not in selected:
            selected.append(row)
    return sorted(selected[: int(count)])


def _shoreline_candidate_cell_for_row(
    *,
    rng: Any,
    row: int,
    side: str,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    cols: int,
    selected_cells: Sequence[tuple[int, int]],
) -> tuple[int, int]:
    """Pick one visible water cell on the requested side of the dock for a candidate boat."""

    left_limit = int(dock_meta["left_col"]) - 3
    right_limit = int(dock_meta["right_col"]) + 3
    if str(side) == "left":
        col_pool = list(range(1, max(1, int(left_limit))))
    else:
        col_pool = list(range(min(int(cols) - 1, int(right_limit) + 1), max(1, int(cols) - 1)))
    if not col_pool:
        col_pool = [col for col in range(1, max(1, int(cols) - 1)) if col < int(left_limit) or col > int(right_limit)]
    rng.shuffle(col_pool)
    for col in col_pool:
        cell = (int(col), int(row))
        tile = tiles_by_cell.get(cell)
        if tile is None or str(tile.terrain) != "water":
            continue
        if all(abs(int(col) - other_col) + abs(int(row) - other_row) >= 2 for other_col, other_row in selected_cells):
            return cell
    for col in col_pool:
        cell = (int(col), int(row))
        tile = tiles_by_cell.get(cell)
        if tile is not None and str(tile.terrain) == "water":
            return cell
    raise ValueError("could not place shoreline candidate boat")


def _draw_shoreline_candidate_boats(
    *,
    draw: ImageDraw.ImageDraw,
    rng: Any,
    tiles_by_cell: Mapping[tuple[int, int], IsoHarborTile],
    dock_meta: Mapping[str, Any],
    labels: Sequence[str],
    nearest_label: str,
    label_font_family: str | None,
    entities: list[IsoHarborEntity],
    canvas_size: tuple[int, int],
    tile_w: float,
    cols: int,
    rows: int,
    tone: IsometricIllustrationTone,
) -> dict[str, Any]:
    """Draw lettered open-water boats ordered by distance from the shoreline."""

    candidate_labels = tuple(str(label) for label in labels)
    if len(candidate_labels) != len(set(candidate_labels)):
        raise ValueError("shoreline candidate labels must be unique")
    if str(nearest_label) not in candidate_labels:
        raise ValueError("nearest_label must be one of the shoreline candidate labels")

    other_labels = [label for label in candidate_labels if str(label) != str(nearest_label)]
    rng.shuffle(other_labels)
    labels_by_rank = [str(nearest_label), *other_labels]
    rows_by_rank = _shoreline_candidate_rows(rows=int(rows), count=len(candidate_labels))
    selected_cells: list[tuple[int, int]] = []
    entity_ids_by_label: dict[str, str] = {}
    label_bboxes_by_label: dict[str, list[float]] = {}
    distances_by_label: dict[str, int] = {}
    bow_points_by_label: dict[str, list[float]] = {}
    scale = float(tile_w) * 0.95
    boat_type = "rowboat"
    half_l = _boat_half_length(scale=scale, boat_type=boat_type)
    first_side = str(rng.choice(("left", "right")))
    second_side = "right" if first_side == "left" else "left"
    for rank, (label, row) in enumerate(zip(labels_by_rank, rows_by_rank)):
        side = first_side if int(rank) % 2 == 0 else second_side
        cell = _shoreline_candidate_cell_for_row(
            rng=rng,
            row=int(row),
            side=side,
            tiles_by_cell=tiles_by_cell,
            dock_meta=dock_meta,
            cols=int(cols),
            selected_cells=selected_cells,
        )
        selected_cells.append(cell)
        tile = tiles_by_cell[cell]
        cx, cy = tile.center_xy
        offset_x = float(rng.choice((-0.08, 0.0, 0.08))) * float(tile_w)
        offset_y = float(rng.choice((-0.04, 0.0, 0.04))) * float(tile_w)
        hull_fill, trim = BOAT_COLOR_PALETTES[int(rank) % len(BOAT_COLOR_PALETTES)]
        if int(rank) >= len(BOAT_COLOR_PALETTES):
            hull_fill = _shade(hull_fill, int(rng.randrange(-18, 19)))
        boat_cx = float(cx) + float(offset_x)
        boat_cy = float(cy) + float(offset_y)
        bbox = _draw_boat(
            draw,
            boat_cx,
            boat_cy,
            scale=scale,
            boat_type=boat_type,
            side="",
            hull_fill=tuple(hull_fill),
            trim=tuple(trim),
            orientation="shore_facing",
            draw_rope=False,
        )
        bow_point = _boat_iso_point(boat_cx, boat_cy, half_l, 0.0, orientation="shore_facing")
        label_w = max(26.0, float(tile_w) * 0.42)
        label_h = max(22.0, float(tile_w) * 0.36)
        label_bbox = _clamp_bbox(
            (
                boat_cx - label_w * 0.5,
                boat_cy - label_h * 0.5,
                boat_cx + label_w * 0.5,
                boat_cy + label_h * 0.5,
            ),
            width=int(canvas_size[0]),
            height=int(canvas_size[1]),
        )
        draw_label_badge(
            draw,
            str(label),
            label_bbox,
            font_family=label_font_family,
            fill=tone.label_fill_rgb,
            outline=tone.label_outline_rgb,
            text_fill=tone.label_text_rgb,
            radius=5,
            width=2,
        )
        entity_id = f"shoreline_candidate_boat_{str(label).lower()}"
        entity_ids_by_label[str(label)] = entity_id
        label_bboxes_by_label[str(label)] = [round(float(value), 3) for value in label_bbox]
        distances_by_label[str(label)] = int(row) - int(dock_meta["row_start"])
        bow_points_by_label[str(label)] = [round(float(bow_point[0]), 3), round(float(bow_point[1]), 3)]
        _add_entity(
            entities,
            entity_id=entity_id,
            public_name="boat",
            object_type="boat",
            tile_ids=(tile.tile_id,),
            bbox=bbox,
            point=(boat_cx, boat_cy),
            role="queryable",
            metadata={
                "boat_type": boat_type,
                "hull_rgb": [int(value) for value in hull_fill],
                "trim_rgb": [int(value) for value in trim],
                "mooring_status": "open_water",
                "orientation": "shore_facing",
                "water_tile_id": str(tile.tile_id),
                "shoreline_candidate_label": str(label),
                "shoreline_distance_rank": int(rank),
                "shoreline_distance_tiles": int(distances_by_label[str(label)]),
                "bow_point_xy": list(bow_points_by_label[str(label)]),
                "label_bbox_xyxy": list(label_bboxes_by_label[str(label)]),
            },
            canvas_size=canvas_size,
        )
    return {
        "candidate_boat_ids_by_label": entity_ids_by_label,
        "candidate_label_bboxes_px_by_label": label_bboxes_by_label,
        "shoreline_distance_tiles_by_label": distances_by_label,
        "shoreline_bow_points_px_by_label": bow_points_by_label,
        "nearest_label": str(nearest_label),
    }


def render_isometric_harbor_scene(
    instance_seed: int,
    *,
    width: int = 1200,
    height: int = 800,
    canvas_profile: str = "landscape",
    canvas_profile_probabilities: Mapping[str, float] | None = None,
    required_boat_counts_by_side: Mapping[str, int] | None = None,
    required_moored_boat_count: int | None = None,
    required_open_water_boat_count: int | None = None,
    required_heading_status_counts: Mapping[str, int] | None = None,
    shoreline_candidate_labels: Sequence[str] | None = None,
    shoreline_nearest_label: str | None = None,
    shoreline_label_font_family: str | None = None,
    render_style_params: Mapping[str, Any] | None = None,
    render_style_defaults: Mapping[str, Any] | None = None,
) -> IsoHarborScene:
    """Render a deterministic full-bleed isometric harbor scene."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_ID}:render")
    cols, rows, tile_w, tile_h, origin_x, origin_y = _profile_geometry(int(width), int(height), str(canvas_profile))
    dock_cells, dock_meta = _dock_cells(cols, rows, rng)
    land_cells = _land_cells(cols, rows, dock_meta)
    tiles = _make_tiles(
        cols=cols,
        rows=rows,
        tile_w=tile_w,
        tile_h=tile_h,
        origin_x=origin_x,
        origin_y=origin_y,
        dock_cells=dock_cells,
        land_cells=land_cells,
    )
    tiles_by_cell = {(int(tile.col), int(tile.row)): tile for tile in tiles}
    tone = resolve_isometric_illustration_tone(
        params=dict(render_style_params or {}),
        render_defaults=dict(render_style_defaults or {}),
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_ID}:background_tone",
    )
    image = Image.new("RGB", (int(width), int(height)), tone.canvas_rgb)
    draw = ImageDraw.Draw(image)
    for tile in tiles:
        if str(tile.terrain) == "water":
            _draw_water_tile(draw, tile, rng=rng, tone=tone)
        elif str(tile.terrain) == "land":
            _draw_land_tile(draw, tile, rng=rng, tone=tone)
    for tile in tiles:
        if str(tile.terrain) == "dock":
            _draw_dock_tile(draw, tile, tone=tone)

    entities: list[IsoHarborEntity] = []
    shoreline_candidate_trace: dict[str, Any] = {}
    if shoreline_candidate_labels is not None:
        labels = tuple(str(label) for label in shoreline_candidate_labels)
        nearest_label = str(shoreline_nearest_label or labels[0])
        shoreline_candidate_trace = _draw_shoreline_candidate_boats(
            draw=draw,
            rng=rng,
            tiles_by_cell=tiles_by_cell,
            dock_meta=dock_meta,
            labels=labels,
            nearest_label=nearest_label,
            label_font_family=shoreline_label_font_family,
            entities=entities,
            canvas_size=(int(width), int(height)),
            tile_w=tile_w,
            cols=int(cols),
            rows=int(rows),
            tone=tone,
        )
        side_counts = {side: 0 for side in BOAT_SIDE_VALUES}
        open_water_count = len(labels)
    elif required_heading_status_counts is not None:
        heading_counts = _draw_heading_status_boats(
            draw=draw,
            rng=rng,
            tiles_by_cell=tiles_by_cell,
            dock_meta=dock_meta,
            required_heading_status_counts=required_heading_status_counts,
            entities=entities,
            canvas_size=(int(width), int(height)),
            tile_w=tile_w,
            cols=int(cols),
            rows=int(rows),
        )
        side_counts = {side: 0 for side in BOAT_SIDE_VALUES}
        open_water_count = sum(int(value) for value in heading_counts.values())
    else:
        _draw_dock_posts(
            draw=draw,
            tiles_by_cell=tiles_by_cell,
            dock_meta=dock_meta,
            entities=entities,
            canvas_size=(int(width), int(height)),
            tile_w=tile_w,
        )
        _draw_context_objects(
            draw=draw,
            rng=rng,
            tiles_by_cell=tiles_by_cell,
            dock_cells=dock_cells,
            dock_meta=dock_meta,
            entities=entities,
            canvas_size=(int(width), int(height)),
            tile_w=tile_w,
        )
        side_requirements = {str(key): int(value) for key, value in (required_boat_counts_by_side or {}).items()}
        if required_moored_boat_count is not None and not side_requirements:
            side_requirements = _split_moored_count_by_side(rng=rng, total=int(required_moored_boat_count))
        side_counts = _draw_boats(
            draw=draw,
            rng=rng,
            tiles_by_cell=tiles_by_cell,
            dock_meta=dock_meta,
            required_boat_counts_by_side=side_requirements,
            entities=entities,
            canvas_size=(int(width), int(height)),
            tile_w=tile_w,
        )
        open_water_count = _draw_open_water_boats(
            draw=draw,
            rng=rng,
            tiles_by_cell=tiles_by_cell,
            dock_meta=dock_meta,
            count=0 if required_open_water_boat_count is None else int(required_open_water_boat_count),
            entities=entities,
            canvas_size=(int(width), int(height)),
            tile_w=tile_w,
            cols=int(cols),
            rows=int(rows),
        )

    dock_tile_ids = [tile.tile_id for tile in tiles if str(tile.terrain) == "dock"]
    water_tile_ids = [tile.tile_id for tile in tiles if str(tile.terrain) == "water"]
    land_tile_ids = [tile.tile_id for tile in tiles if str(tile.terrain) == "land"]
    trace = {
        "renderer_id": RENDERER_ID,
        "renderer_style": "isometric_pixel_harbor",
        "theme_id": "isometric_harbor_shoreline_dock",
        "seed": int(instance_seed),
        **tone.trace_metadata(),
        "canvas_profile": str(canvas_profile),
        "canvas_profile_probabilities": dict(canvas_profile_probabilities or {}),
        "canvas_size_px": [int(width), int(height)],
        "grid_cols": int(cols),
        "grid_rows": int(rows),
        "tile_count": len(tiles),
        "dock_tile_ids": dock_tile_ids,
        "water_tile_ids": water_tile_ids,
        "land_tile_ids": land_tile_ids,
        "terrain_tile_counts": {
            "dock": len(dock_tile_ids),
            "water": len(water_tile_ids),
            "land": len(land_tile_ids),
        },
        "dock_meta": dict(dock_meta),
        "boat_counts_by_side": {str(side): int(side_counts.get(side, 0)) for side in BOAT_SIDE_VALUES},
        "boat_counts_by_mooring_status": {
            "moored": sum(1 for entity in entities if entity.object_type == "boat" and str(entity.metadata.get("mooring_status", "")) == "moored"),
            "open_water": int(open_water_count),
        },
        "boat_counts_by_heading_status": {
            str(status): sum(
                1
                for entity in entities
                if entity.object_type == "boat" and str(entity.metadata.get("heading_status", "")) == str(status)
            )
            for status in BOAT_HEADING_STATUS_VALUES
        },
        "entity_count": len(entities),
        "context_object_counts": {
            "boat": sum(1 for entity in entities if entity.object_type == "boat"),
            "crate": sum(1 for entity in entities if entity.object_type == "crate"),
            "barrel": sum(1 for entity in entities if entity.object_type == "barrel"),
            "dock_post": sum(1 for entity in entities if entity.object_type == "dock_post"),
        },
        "projection": {
            "type": "2:1_isometric",
            "origin_xy": [round(float(origin_x), 3), round(float(origin_y), 3)],
            "tile_size_px": [round(float(tile_w), 3), round(float(tile_h), 3)],
        },
    }
    if shoreline_candidate_trace:
        trace["shoreline_reference"] = "first water row next to the land shoreline"
        trace["shoreline_candidate_count"] = len(shoreline_candidate_trace["candidate_boat_ids_by_label"])
        trace.update(shoreline_candidate_trace)
    if required_heading_status_counts is not None:
        trace["shoreline_reference"] = "first water row next to the land shoreline"
    return IsoHarborScene(
        image=image,
        tiles=tuple(tiles),
        entities=tuple(sorted(entities, key=lambda entity: str(entity.entity_id))),
        trace=trace,
    )


__all__ = [
    "BOAT_HEADING_STATUS_ORIENTATION",
    "BOAT_HEADING_STATUS_VALUES",
    "BOAT_SIDE_VALUES",
    "BOAT_MOORING_STATUS_VALUES",
    "DEFAULT_BOAT_CANDIDATE_LABELS",
    "OPEN_WATER_BOAT_ORIENTATIONS",
    "RENDERER_ID",
    "SCENE_ID",
    "SUPPORTED_CANVAS_PROFILES",
    "render_isometric_harbor_scene",
]
