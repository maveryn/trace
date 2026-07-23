"""Scene-local renderer for top-down tactical RPG movement maps."""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import uniform_choice_with_probabilities
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.illustrations.shared.canvas_profiles import (
    CANVAS_PROFILE_LANDSCAPE,
    CANVAS_PROFILE_PORTRAIT,
    CANVAS_PROFILE_SQUARE,
)
from trace_tasks.tasks.illustrations.shared.object_variants import RENDERER_STYLE_TOP_DOWN_PIXEL_RPG
from trace_tasks.tasks.illustrations.shared.option_rendering import draw_label_badge

from .relations import (
    TERRAIN_BRIDGE,
    TERRAIN_FOREST,
    TERRAIN_GRASS,
    TERRAIN_MOUNTAIN,
    TERRAIN_MOVEMENT_COSTS,
    TERRAIN_ROAD,
    TERRAIN_WATER,
    movement_cost_for_terrain,
)
from .state import BBox, RpgTacticalMapScene, RpgTacticalTile, RpgTacticalUnit


SCENE_ID = "rpg_tactical_map"
RENDERER_ID = "rpg_tactical_map_top_down_v0"
DEFAULT_TILE_PX = 80
DEFAULT_CANDIDATE_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
WATER_FEATURE_STYLE_STRAIGHT = "straight"
WATER_FEATURE_STYLE_ZIGZAG = "zigzag"
WATER_FEATURE_STYLES: tuple[str, str] = (WATER_FEATURE_STYLE_STRAIGHT, WATER_FEATURE_STYLE_ZIGZAG)
TACTICAL_PROFILE_GRIDS: Mapping[str, tuple[int, int]] = {
    CANVAS_PROFILE_LANDSCAPE: (12, 8),
    CANVAS_PROFILE_SQUARE: (10, 10),
    CANVAS_PROFILE_PORTRAIT: (8, 12),
}
TACTICAL_PROFILE_SUPPORT: tuple[str, ...] = (
    CANVAS_PROFILE_LANDSCAPE,
    CANVAS_PROFILE_SQUARE,
    CANVAS_PROFILE_PORTRAIT,
)

RGB = tuple[int, int, int]

THEMES: Mapping[str, Mapping[str, RGB]] = {
    "emerald_field": {
        "grass": (96, 167, 91),
        "grass_alt": (108, 177, 99),
        "grid": (52, 91, 57),
        "road": (188, 151, 91),
        "road_dark": (137, 103, 63),
        "forest": (39, 113, 62),
        "forest_dark": (25, 77, 48),
        "water": (60, 135, 194),
        "water_dark": (36, 86, 142),
        "bridge": (143, 89, 47),
        "bridge_dark": (86, 54, 36),
        "mountain": (137, 132, 122),
        "mountain_dark": (76, 76, 77),
        "mountain_light": (191, 188, 172),
    },
    "autumn_plain": {
        "grass": (139, 164, 82),
        "grass_alt": (153, 176, 92),
        "grid": (84, 97, 50),
        "road": (194, 142, 83),
        "road_dark": (126, 86, 55),
        "forest": (71, 112, 57),
        "forest_dark": (42, 72, 40),
        "water": (67, 126, 176),
        "water_dark": (42, 79, 126),
        "bridge": (139, 82, 43),
        "bridge_dark": (82, 50, 34),
        "mountain": (145, 126, 112),
        "mountain_dark": (82, 71, 68),
        "mountain_light": (205, 189, 169),
    },
    "highland": {
        "grass": (89, 148, 106),
        "grass_alt": (101, 160, 116),
        "grid": (50, 84, 66),
        "road": (174, 157, 101),
        "road_dark": (112, 103, 73),
        "forest": (38, 105, 79),
        "forest_dark": (24, 70, 56),
        "water": (54, 123, 177),
        "water_dark": (34, 78, 126),
        "bridge": (126, 88, 50),
        "bridge_dark": (77, 54, 38),
        "mountain": (130, 137, 133),
        "mountain_dark": (70, 78, 78),
        "mountain_light": (184, 196, 185),
    },
}


def resolve_tactical_map_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int | None = None,
    namespace: str = "illustrations:rpg_tactical_map:canvas_profile",
) -> dict[str, Any]:
    """Resolve the small-grid tactical-map profile and tile size."""

    tile_px = int(params.get("tile_px", group_default(render_defaults, "rpg_tactical_map_tile_px", DEFAULT_TILE_PX)))
    if tile_px <= 0:
        raise ValueError("rpg_tactical_map tile size must be positive")
    if "grid_cols" in params or "grid_rows" in params:
        cols = int(params.get("grid_cols", TACTICAL_PROFILE_GRIDS[CANVAS_PROFILE_LANDSCAPE][0]))
        rows = int(params.get("grid_rows", TACTICAL_PROFILE_GRIDS[CANVAS_PROFILE_LANDSCAPE][1]))
        if cols <= 0 or rows <= 0:
            raise ValueError("custom tactical-map grid dimensions must be positive")
        return {
            "canvas_profile": "custom",
            "canvas_width": int(cols) * int(tile_px),
            "canvas_height": int(rows) * int(tile_px),
            "tile_px": int(tile_px),
            "grid_cols": int(cols),
            "grid_rows": int(rows),
            "canvas_profile_size": [int(cols) * int(tile_px), int(rows) * int(tile_px)],
            "canvas_profile_probabilities": {"custom": 1.0},
        }

    support = _profile_support(params, render_defaults)
    explicit = params.get("canvas_profile", group_default(render_defaults, "canvas_profile", None))
    if explicit is not None:
        profile_id = str(explicit)
        if profile_id not in set(support):
            raise ValueError(f"canvas_profile must be one of {support}")
        probabilities = {profile_id: 1.0}
    else:
        if instance_seed is None:
            profile_id = str(support[0])
            probabilities = _uniform_probability_map(support)
        else:
            sample_namespace = str(namespace)
            if params.get("_sample_cursor") is not None:
                sample_namespace = f"{sample_namespace}:{int(params['_sample_cursor'])}"
            rng = spawn_rng(int(instance_seed), sample_namespace)
            profile_id, probabilities = uniform_choice_with_probabilities(
                rng,
                support,
                sort_keys=False,
            )
            profile_id = str(profile_id)
    cols, rows = TACTICAL_PROFILE_GRIDS[profile_id]
    return {
        "canvas_profile": str(profile_id),
        "canvas_width": int(cols) * int(tile_px),
        "canvas_height": int(rows) * int(tile_px),
        "tile_px": int(tile_px),
        "grid_cols": int(cols),
        "grid_rows": int(rows),
        "canvas_profile_size": [int(cols) * int(tile_px), int(rows) * int(tile_px)],
        "canvas_profile_probabilities": dict(probabilities),
    }


def render_rpg_tactical_map_scene(
    seed: int,
    *,
    width: int,
    height: int,
    grid_cols: int,
    grid_rows: int,
    tile_px: int = DEFAULT_TILE_PX,
    player_tile_id: str | None = None,
    candidate_tile_ids_by_label: Mapping[str, str] | None = None,
    target_tile_ids: Sequence[str] | None = None,
    terrain_grid_override: Sequence[Sequence[str]] | None = None,
    label_font_family: str | None = None,
    label_font_trace: Mapping[str, Any] | None = None,
    render_metadata: Mapping[str, Any] | None = None,
) -> RpgTacticalMapScene:
    """Render one full-bleed tactical map with optional lettered candidate tiles."""

    rng = random.Random(int(seed))
    cols = int(grid_cols)
    rows = int(grid_rows)
    tile_size = int(tile_px)
    if int(width) != cols * tile_size or int(height) != rows * tile_size:
        raise ValueError("tactical map canvas must exactly match grid_cols/grid_rows * tile_px")
    theme_id = str(_choose(rng, tuple(THEMES)))
    theme = THEMES[theme_id]
    if terrain_grid_override is None:
        terrain_grid, terrain_generation_trace = _make_terrain_grid(cols=cols, rows=rows, rng=rng)
        terrain_grid_source = "generated"
    else:
        terrain_grid = _normalize_terrain_grid_override(
            terrain_grid_override,
            cols=cols,
            rows=rows,
        )
        terrain_generation_trace = {"water_feature": {"kind": "override"}}
        terrain_grid_source = "override"
    tiles = _make_tiles(terrain_grid=terrain_grid, tile_px=tile_size)
    tiles_by_id = {str(tile.tile_id): tile for tile in tiles}
    resolved_player_tile_id = str(player_tile_id or _select_player_tile_id(tiles, rng=rng))
    if resolved_player_tile_id not in tiles_by_id or not bool(tiles_by_id[resolved_player_tile_id].passable):
        raise ValueError("player_tile_id must name a passable tile")

    image = Image.new("RGB", (int(width), int(height)), theme["grass"])
    draw = ImageDraw.Draw(image)
    for tile in tiles:
        _draw_tile(draw, tile=tile, tile_px=tile_size, theme=theme, rng=rng)
    _draw_grid(draw, cols=cols, rows=rows, tile_px=tile_size, theme=theme)
    resolved_target_tile_ids = [str(tile_id) for tile_id in (target_tile_ids or ())]
    _draw_target_markers(
        draw,
        tiles_by_id=tiles_by_id,
        target_tile_ids=resolved_target_tile_ids,
        tile_px=tile_size,
    )

    player_unit = _draw_player_unit(
        draw,
        tile=tiles_by_id[resolved_player_tile_id],
        tile_px=tile_size,
    )
    label_bboxes = _draw_candidate_labels(
        draw,
        tiles_by_id=tiles_by_id,
        candidate_tile_ids_by_label=candidate_tile_ids_by_label or {},
        tile_px=tile_size,
        font_family=label_font_family,
    )
    trace = {
        "renderer_id": RENDERER_ID,
        "renderer_style": RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
        "theme_id": theme_id,
        "width": int(width),
        "height": int(height),
        "grid_cols": cols,
        "grid_rows": rows,
        "tile_px": tile_size,
        "terrain_movement_costs": {str(key): int(value) for key, value in TERRAIN_MOVEMENT_COSTS.items()},
        "blocked_terrain": [TERRAIN_WATER],
        "terrain_rows": [[str(value) for value in row] for row in terrain_grid],
        "terrain_grid_source": terrain_grid_source,
        "terrain_generation": dict(terrain_generation_trace),
        "player_tile_id": resolved_player_tile_id,
        "candidate_tile_ids_by_label": {str(label): str(tile_id) for label, tile_id in (candidate_tile_ids_by_label or {}).items()},
        "target_tile_ids": list(resolved_target_tile_ids),
        "label_font": dict(label_font_trace or {}),
        **dict(render_metadata or {}),
    }
    return RpgTacticalMapScene(
        image=image,
        tiles=tiles,
        units=(player_unit,),
        label_bboxes_by_tile_id=label_bboxes,
        trace=trace,
    )


def _profile_support(params: Mapping[str, Any], defaults: Mapping[str, Any]) -> tuple[str, ...]:
    raw = params.get("canvas_profile_support", group_default(defaults, "canvas_profile_support", TACTICAL_PROFILE_SUPPORT))
    if isinstance(raw, str):
        values = (raw,)
    elif isinstance(raw, Sequence):
        values = tuple(raw)
    else:
        values = tuple(TACTICAL_PROFILE_SUPPORT)
    support = tuple(dict.fromkeys(str(value) for value in values if str(value) in set(TACTICAL_PROFILE_SUPPORT)))
    if not support:
        raise ValueError("canvas_profile_support must include at least one tactical-map canvas profile")
    return support


def _uniform_probability_map(values: Sequence[str]) -> dict[str, float]:
    support = tuple(str(value) for value in values)
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def _normalize_terrain_grid_override(
    terrain_grid: Sequence[Sequence[str]],
    *,
    cols: int,
    rows: int,
) -> list[list[str]]:
    """Validate and normalize a caller-supplied terrain grid."""

    if len(terrain_grid) != int(rows):
        raise ValueError("terrain_grid_override row count must match grid_rows")
    valid_terrains = {
        TERRAIN_BRIDGE,
        TERRAIN_FOREST,
        TERRAIN_GRASS,
        TERRAIN_MOUNTAIN,
        TERRAIN_ROAD,
        TERRAIN_WATER,
    }
    normalized: list[list[str]] = []
    for row in terrain_grid:
        if len(row) != int(cols):
            raise ValueError("terrain_grid_override column count must match grid_cols")
        normalized_row = [str(value) for value in row]
        unknown = sorted(set(normalized_row) - valid_terrains)
        if unknown:
            raise ValueError(f"terrain_grid_override contains unknown terrain values: {unknown}")
        normalized.append(normalized_row)
    return normalized


def _choose(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[int(rng.randrange(len(values)))]


def _make_terrain_grid(*, cols: int, rows: int, rng: random.Random) -> tuple[list[list[str]], dict[str, Any]]:
    grid = [[TERRAIN_GRASS for _ in range(int(cols))] for _ in range(int(rows))]
    water_feature_trace = _paint_water_feature(grid, rng=rng)
    _paint_road_feature(grid, rng=rng)
    for _ in range(rng.randint(3, 5)):
        _paint_blob(grid, rng=rng, terrain=TERRAIN_FOREST, size=rng.randint(4, 12), avoid={TERRAIN_WATER, TERRAIN_BRIDGE, TERRAIN_ROAD})
    for _ in range(rng.randint(1, 3)):
        _paint_blob(grid, rng=rng, terrain=TERRAIN_MOUNTAIN, size=rng.randint(3, 6), avoid={TERRAIN_WATER, TERRAIN_BRIDGE, TERRAIN_ROAD})
    return grid, {"water_feature": water_feature_trace}


def _paint_water_feature(grid: list[list[str]], *, rng: random.Random) -> dict[str, Any]:
    """Paint the scene-level water feature and record its sampled visual grammar."""

    rows = len(grid)
    cols = len(grid[0])
    orientation = str(_choose(rng, ("horizontal", "vertical", "pond")))
    if orientation == "horizontal" and rows >= 6:
        thickness = rng.randint(1, max(1, min(2, rows - 3)))
        start_index = rng.randrange(1, rows - int(thickness))
        style = resolve_water_feature_style(None, rng=rng)
        cells = map_spanning_water_cells(
            orientation=orientation,
            style=style,
            cols=cols,
            rows=rows,
            start_index=start_index,
            thickness=thickness,
            rng=rng,
        )
        for row, col in cells:
            grid[int(row)][int(col)] = TERRAIN_WATER
        return {
            "kind": "river",
            "orientation": orientation,
            "style": style,
            "start_index": int(start_index),
            "thickness": int(thickness),
            "water_tile_count": len(cells),
        }
    elif orientation == "vertical" and cols >= 6:
        thickness = rng.randint(1, max(1, min(2, cols - 3)))
        start_index = rng.randrange(1, cols - int(thickness))
        style = resolve_water_feature_style(None, rng=rng)
        cells = map_spanning_water_cells(
            orientation=orientation,
            style=style,
            cols=cols,
            rows=rows,
            start_index=start_index,
            thickness=thickness,
            rng=rng,
        )
        for row, col in cells:
            grid[int(row)][int(col)] = TERRAIN_WATER
        return {
            "kind": "river",
            "orientation": orientation,
            "style": style,
            "start_index": int(start_index),
            "thickness": int(thickness),
            "water_tile_count": len(cells),
        }
    else:
        center_row = rng.randrange(max(1, rows // 4), max(2, rows - rows // 4))
        center_col = rng.randrange(max(1, cols // 4), max(2, cols - cols // 4))
        water_tile_count = 0
        for row in range(max(0, center_row - 1), min(rows, center_row + 2)):
            for col in range(max(0, center_col - 2), min(cols, center_col + 3)):
                if rng.random() < 0.78:
                    grid[row][col] = TERRAIN_WATER
                    water_tile_count += 1
        return {
            "kind": "pond",
            "orientation": "blob",
            "style": "blob",
            "center": [int(center_row), int(center_col)],
            "water_tile_count": int(water_tile_count),
        }


def resolve_water_feature_style(explicit_style: str | None, *, rng: random.Random) -> str:
    """Resolve the shared tactical-map river style."""

    style = str(explicit_style or _choose(rng, (WATER_FEATURE_STYLE_ZIGZAG, WATER_FEATURE_STYLE_ZIGZAG, WATER_FEATURE_STYLE_STRAIGHT)))
    if style not in set(WATER_FEATURE_STYLES):
        raise ValueError(f"water feature style must be one of {WATER_FEATURE_STYLES}")
    return style


def map_spanning_water_cells(
    *,
    orientation: str,
    style: str,
    cols: int,
    rows: int,
    start_index: int,
    thickness: int,
    rng: random.Random,
) -> set[tuple[int, int]]:
    """Return connected water cells that span the map in one direction."""

    resolved_orientation = str(orientation)
    resolved_style = resolve_water_feature_style(str(style), rng=rng)
    if resolved_orientation not in {"vertical", "horizontal"}:
        raise ValueError("water feature orientation must be vertical or horizontal")
    barrier_cells: set[tuple[int, int]] = set()
    if resolved_orientation == "vertical":
        current_col = int(start_index)
        previous_col = int(current_col)
        forced_step = max(1, int(rows) // 2)
        for row in range(int(rows)):
            if resolved_style == WATER_FEATURE_STYLE_ZIGZAG and row > 0:
                if int(row) == int(forced_step):
                    delta = rng.choice((-1, 1))
                elif rng.random() < 0.42:
                    delta = rng.choice((-1, 0, 1))
                else:
                    delta = 0
                current_col = _clamp_water_path_index(
                    current_col + int(delta),
                    min_index=1,
                    max_index=int(cols) - int(thickness) - 1,
                )
            first_col = min(int(previous_col), int(current_col))
            last_col_exclusive = max(int(previous_col), int(current_col)) + int(thickness)
            for col in range(first_col, last_col_exclusive):
                barrier_cells.add((int(row), int(col)))
            previous_col = int(current_col)
    else:
        current_row = int(start_index)
        previous_row = int(current_row)
        forced_step = max(1, int(cols) // 2)
        for col in range(int(cols)):
            if resolved_style == WATER_FEATURE_STYLE_ZIGZAG and col > 0:
                if int(col) == int(forced_step):
                    delta = rng.choice((-1, 1))
                elif rng.random() < 0.42:
                    delta = rng.choice((-1, 0, 1))
                else:
                    delta = 0
                current_row = _clamp_water_path_index(
                    current_row + int(delta),
                    min_index=1,
                    max_index=int(rows) - int(thickness) - 1,
                )
            first_row = min(int(previous_row), int(current_row))
            last_row_exclusive = max(int(previous_row), int(current_row)) + int(thickness)
            for row in range(first_row, last_row_exclusive):
                barrier_cells.add((int(row), int(col)))
            previous_row = int(current_row)
    return barrier_cells


def _clamp_water_path_index(value: int, *, min_index: int, max_index: int) -> int:
    return max(int(min_index), min(int(max_index), int(value)))


def _paint_road_feature(grid: list[list[str]], *, rng: random.Random) -> None:
    rows = len(grid)
    cols = len(grid[0])
    road_row = rng.randrange(rows)
    road_col = rng.randrange(cols)
    style = str(_choose(rng, ("horizontal", "vertical", "cross", "elbow")))
    if style in {"horizontal", "cross"}:
        for col in range(cols):
            grid[road_row][col] = TERRAIN_BRIDGE if grid[road_row][col] == TERRAIN_WATER else TERRAIN_ROAD
    if style in {"vertical", "cross"}:
        for row in range(rows):
            grid[row][road_col] = TERRAIN_BRIDGE if grid[row][road_col] == TERRAIN_WATER else TERRAIN_ROAD
    if style == "elbow":
        for col in range(road_col + 1):
            grid[road_row][col] = TERRAIN_BRIDGE if grid[road_row][col] == TERRAIN_WATER else TERRAIN_ROAD
        for row in range(road_row, rows):
            grid[row][road_col] = TERRAIN_BRIDGE if grid[row][road_col] == TERRAIN_WATER else TERRAIN_ROAD


def _paint_blob(
    grid: list[list[str]],
    *,
    rng: random.Random,
    terrain: str,
    size: int,
    avoid: set[str],
) -> None:
    rows = len(grid)
    cols = len(grid[0])
    row = rng.randrange(rows)
    col = rng.randrange(cols)
    for _ in range(int(size)):
        if grid[row][col] not in avoid:
            grid[row][col] = str(terrain)
        drow, dcol = _choose(rng, ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)))
        row = min(rows - 1, max(0, row + int(drow)))
        col = min(cols - 1, max(0, col + int(dcol)))


def _make_tiles(*, terrain_grid: Sequence[Sequence[str]], tile_px: int) -> tuple[RpgTacticalTile, ...]:
    tiles: list[RpgTacticalTile] = []
    for row, values in enumerate(terrain_grid):
        for col, terrain in enumerate(values):
            x0 = int(col) * int(tile_px)
            y0 = int(row) * int(tile_px)
            bbox = (float(x0), float(y0), float(x0 + int(tile_px)), float(y0 + int(tile_px)))
            cost = movement_cost_for_terrain(str(terrain))
            tile = RpgTacticalTile(
                tile_id=f"r{int(row):02d}_c{int(col):02d}",
                row=int(row),
                col=int(col),
                terrain=str(terrain),
                movement_cost=cost,
                passable=cost is not None,
                bbox_xyxy=bbox,
                point_xy=(float(x0 + int(tile_px) * 0.5), float(y0 + int(tile_px) * 0.5)),
                metadata={"tile_px": int(tile_px)},
            )
            tiles.append(tile)
    return tuple(tiles)


def _select_player_tile_id(tiles: Sequence[RpgTacticalTile], *, rng: random.Random) -> str:
    passable = [tile for tile in tiles if bool(tile.passable) and str(tile.terrain) != TERRAIN_BRIDGE]
    if not passable:
        raise ValueError("tactical map has no passable player tile")
    interior = [
        tile
        for tile in passable
        if int(tile.row) not in {0, max(int(other.row) for other in tiles)}
        and int(tile.col) not in {0, max(int(other.col) for other in tiles)}
    ]
    pool = interior or passable
    return str(_choose(rng, pool).tile_id)


def _draw_tile(
    draw: ImageDraw.ImageDraw,
    *,
    tile: RpgTacticalTile,
    tile_px: int,
    theme: Mapping[str, RGB],
    rng: random.Random,
) -> None:
    x0, y0, x1, y1 = [int(round(value)) for value in tile.bbox_xyxy]
    terrain = str(tile.terrain)
    if terrain == TERRAIN_GRASS:
        fill = theme["grass_alt"] if (int(tile.row) + int(tile.col)) % 2 == 0 else theme["grass"]
        draw.rectangle((x0, y0, x1, y1), fill=fill)
        _draw_grass_marks(draw, x0, y0, int(tile_px), theme)
    elif terrain == TERRAIN_ROAD:
        draw.rectangle((x0, y0, x1, y1), fill=theme["road"])
        for offset in (0.28, 0.56, 0.82):
            yy = y0 + int(tile_px * offset)
            draw.line((x0 + 8, yy, x1 - 8, yy + rng.randint(-2, 2)), fill=theme["road_dark"], width=2)
    elif terrain == TERRAIN_FOREST:
        draw.rectangle((x0, y0, x1, y1), fill=theme["grass"])
        _draw_forest_tile(draw, x0, y0, int(tile_px), theme)
    elif terrain == TERRAIN_WATER:
        draw.rectangle((x0, y0, x1, y1), fill=theme["water"])
        for offset in (0.28, 0.52, 0.76):
            yy = y0 + int(tile_px * offset)
            draw.arc((x0 + 10, yy - 8, x0 + int(tile_px * 0.48), yy + 8), 10, 170, fill=theme["water_dark"], width=2)
            draw.arc((x0 + int(tile_px * 0.52), yy - 8, x1 - 10, yy + 8), 10, 170, fill=theme["water_dark"], width=2)
    elif terrain == TERRAIN_BRIDGE:
        draw.rectangle((x0, y0, x1, y1), fill=theme["bridge"])
        draw.rectangle((x0 + 5, y0 + 8, x1 - 5, y1 - 8), outline=theme["bridge_dark"], width=3)
        for xx in range(x0 + 16, x1, max(12, int(tile_px // 4))):
            draw.line((xx, y0 + 8, xx, y1 - 8), fill=theme["bridge_dark"], width=2)
    elif terrain == TERRAIN_MOUNTAIN:
        draw.rectangle((x0, y0, x1, y1), fill=theme["grass"])
        _draw_mountain_tile(draw, x0, y0, int(tile_px), theme)
    else:
        draw.rectangle((x0, y0, x1, y1), fill=theme["grass"])


def _draw_grass_marks(draw: ImageDraw.ImageDraw, x0: int, y0: int, tile_px: int, theme: Mapping[str, RGB]) -> None:
    mark = (64, 129, 66)
    for dx, dy in ((18, 22), (46, 17), (58, 51), (26, 58)):
        if dx + 4 < tile_px and dy + 7 < tile_px:
            draw.line((x0 + dx, y0 + dy + 5, x0 + dx + 3, y0 + dy), fill=mark, width=2)
            draw.line((x0 + dx + 3, y0 + dy, x0 + dx + 7, y0 + dy + 5), fill=mark, width=2)


def _draw_forest_tile(draw: ImageDraw.ImageDraw, x0: int, y0: int, tile_px: int, theme: Mapping[str, RGB]) -> None:
    """Draw a readable slow-terrain forest while preserving visible tile bounds."""

    draw.rectangle((x0, y0, x0 + tile_px, y0 + tile_px), fill=_mix_rgb(theme["grass"], theme["forest"], 0.18))
    tree_specs = (
        (0.36, 0.40, 0.165),
        (0.58, 0.36, 0.175),
        (0.48, 0.62, 0.155),
    )
    trunk_rgb = (92, 56, 32)
    outline = theme["forest_dark"]
    for cx_ratio, cy_ratio, scale in tree_specs:
        cx = x0 + int(tile_px * cx_ratio)
        cy = y0 + int(tile_px * cy_ratio)
        canopy_r = max(10, int(tile_px * scale))
        trunk_w = max(3, int(tile_px * 0.060))
        trunk_h = max(8, int(tile_px * 0.18))
        draw.ellipse(
            (cx - canopy_r + 2, cy + canopy_r * 0.38, cx + canopy_r - 2, cy + canopy_r * 0.72),
            fill=(35, 72, 43),
        )
        draw.rectangle(
            (cx - trunk_w // 2, cy + canopy_r // 4, cx + trunk_w // 2, cy + canopy_r // 4 + trunk_h),
            fill=trunk_rgb,
            outline=(55, 35, 24),
        )
        draw.polygon(
            (
                (cx, cy - canopy_r),
                (cx - canopy_r, cy + canopy_r // 3),
                (cx + canopy_r, cy + canopy_r // 3),
            ),
            fill=outline,
        )
        draw.polygon(
            (
                (cx, cy - canopy_r + max(3, canopy_r // 4)),
                (cx - canopy_r + 5, cy + canopy_r // 3 - 2),
                (cx + canopy_r - 5, cy + canopy_r // 3 - 2),
            ),
            fill=theme["forest"],
        )
        draw.polygon(
            (
                (cx, cy - canopy_r // 2),
                (cx - canopy_r + 4, cy + canopy_r // 2),
                (cx + canopy_r - 4, cy + canopy_r // 2),
            ),
            fill=_mix_rgb(theme["forest"], (255, 255, 255), 0.10),
        )
        draw.line(
            (cx - canopy_r // 4, cy - canopy_r // 5, cx + canopy_r // 4, cy + canopy_r // 5),
            fill=_mix_rgb(theme["forest"], (255, 255, 255), 0.22),
            width=max(1, int(tile_px * 0.025)),
        )


def _draw_mountain_tile(draw: ImageDraw.ImageDraw, x0: int, y0: int, tile_px: int, theme: Mapping[str, RGB]) -> None:
    """Draw mountain as rough passable terrain, not as an impassable wall."""

    ground = _mix_rgb(theme["grass"], theme["mountain"], 0.48)
    draw.rectangle((x0, y0, x0 + tile_px, y0 + tile_px), fill=ground)
    shadow = _mix_rgb(theme["mountain_dark"], (0, 0, 0), 0.15)
    rock_groups = (
        ((0.19, 0.63), (0.34, 0.35), (0.48, 0.68)),
        ((0.47, 0.48), (0.61, 0.19), (0.80, 0.54)),
        ((0.38, 0.80), (0.55, 0.56), (0.72, 0.82)),
    )
    for left_ratio, peak_ratio, right_ratio in rock_groups:
        left = (x0 + int(tile_px * left_ratio[0]), y0 + int(tile_px * left_ratio[1]))
        peak = (x0 + int(tile_px * peak_ratio[0]), y0 + int(tile_px * peak_ratio[1]))
        right = (x0 + int(tile_px * right_ratio[0]), y0 + int(tile_px * right_ratio[1]))
        mid = (x0 + int(tile_px * ((peak_ratio[0] + right_ratio[0]) * 0.5)), y0 + int(tile_px * ((peak_ratio[1] + right_ratio[1]) * 0.5)))
        draw.polygon((left, peak, right), fill=theme["mountain"], outline=theme["mountain_dark"])
        draw.polygon((peak, left, mid), fill=theme["mountain_light"])
        draw.polygon((peak, mid, right), fill=_mix_rgb(theme["mountain"], shadow, 0.35))
    crack_width = max(2, int(tile_px * 0.025))
    cracks = (
        ((0.22, 0.30), (0.34, 0.42), (0.29, 0.52)),
        ((0.72, 0.25), (0.64, 0.38), (0.70, 0.48)),
        ((0.18, 0.78), (0.28, 0.74), (0.36, 0.82)),
    )
    for start, mid, end in cracks:
        draw.line(
            (
                x0 + int(tile_px * start[0]),
                y0 + int(tile_px * start[1]),
                x0 + int(tile_px * mid[0]),
                y0 + int(tile_px * mid[1]),
                x0 + int(tile_px * end[0]),
                y0 + int(tile_px * end[1]),
            ),
            fill=shadow,
            width=crack_width,
            joint="curve",
        )
    pebble_rgb = _mix_rgb(theme["mountain_dark"], theme["mountain"], 0.35)
    for px_ratio, py_ratio in ((0.16, 0.20), (0.88, 0.34), (0.12, 0.88), (0.84, 0.78)):
        px = x0 + int(tile_px * px_ratio)
        py = y0 + int(tile_px * py_ratio)
        r = max(2, int(tile_px * 0.035))
        draw.ellipse((px - r, py - r, px + r, py + r), fill=pebble_rgb)


def _draw_grid(draw: ImageDraw.ImageDraw, *, cols: int, rows: int, tile_px: int, theme: Mapping[str, RGB]) -> None:
    line = theme["grid"]
    for col in range(1, int(cols)):
        x = int(col) * int(tile_px)
        draw.line((x, 0, x, int(rows) * int(tile_px)), fill=line, width=2)
    for row in range(1, int(rows)):
        y = int(row) * int(tile_px)
        draw.line((0, y, int(cols) * int(tile_px), y), fill=line, width=2)


def _draw_player_unit(draw: ImageDraw.ImageDraw, *, tile: RpgTacticalTile, tile_px: int) -> RpgTacticalUnit:
    """Draw the blue reference unit as a compact player sprite centered in one tile."""

    cx, cy = float(tile.point_xy[0]), float(tile.point_xy[1])
    radius = float(tile_px) * 0.36
    outline = (13, 27, 76)
    blue = (28, 82, 202)
    blue_light = (83, 151, 246)
    armor = (214, 224, 231)
    skin = (226, 176, 130)
    boot = (42, 43, 54)
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(
        (cx - radius * 0.95, cy + radius * 0.58, cx + radius * 0.95, cy + radius * 0.86),
        fill=(29, 45, 52),
    )
    # Legs and boots.
    leg_w = radius * 0.24
    leg_top = cy + radius * 0.22
    leg_bottom = cy + radius * 0.78
    draw.rounded_rectangle((cx - radius * 0.42, leg_top, cx - radius * 0.10, leg_bottom), radius=3, fill=outline)
    draw.rounded_rectangle((cx + radius * 0.10, leg_top, cx + radius * 0.42, leg_bottom), radius=3, fill=outline)
    draw.rectangle((cx - radius * 0.48, leg_bottom - leg_w, cx - radius * 0.04, leg_bottom + leg_w * 0.2), fill=boot)
    draw.rectangle((cx + radius * 0.04, leg_bottom - leg_w, cx + radius * 0.48, leg_bottom + leg_w * 0.2), fill=boot)
    # Shoulders, arms, and shield/sword silhouettes.
    draw.ellipse((cx - radius * 0.78, cy - radius * 0.10, cx - radius * 0.34, cy + radius * 0.36), fill=outline)
    draw.ellipse((cx + radius * 0.34, cy - radius * 0.10, cx + radius * 0.78, cy + radius * 0.36), fill=outline)
    draw.polygon(
        (
            (cx - radius * 0.92, cy + radius * 0.02),
            (cx - radius * 0.62, cy - radius * 0.24),
            (cx - radius * 0.42, cy + radius * 0.20),
            (cx - radius * 0.70, cy + radius * 0.44),
        ),
        fill=(200, 214, 220),
        outline=outline,
    )
    draw.line((cx + radius * 0.70, cy - radius * 0.34, cx + radius * 0.96, cy - radius * 0.72), fill=(218, 224, 229), width=max(2, int(tile_px * 0.035)))
    draw.line((cx + radius * 0.68, cy - radius * 0.34, cx + radius * 0.88, cy - radius * 0.13), fill=outline, width=max(2, int(tile_px * 0.025)))
    # Torso armor and blue tunic.
    draw.rounded_rectangle(
        (cx - radius * 0.46, cy - radius * 0.18, cx + radius * 0.46, cy + radius * 0.50),
        radius=max(4, int(tile_px * 0.055)),
        fill=blue,
        outline=outline,
        width=max(2, int(tile_px * 0.035)),
    )
    draw.polygon(
        (
            (cx - radius * 0.38, cy - radius * 0.12),
            (cx + radius * 0.38, cy - radius * 0.12),
            (cx + radius * 0.24, cy + radius * 0.18),
            (cx - radius * 0.24, cy + radius * 0.18),
        ),
        fill=armor,
    )
    draw.rectangle((cx - radius * 0.14, cy + radius * 0.18, cx + radius * 0.14, cy + radius * 0.50), fill=(18, 56, 158))
    # Head and helmet.
    head_r = radius * 0.30
    draw.ellipse((cx - head_r, cy - radius * 0.72, cx + head_r, cy - radius * 0.12), fill=skin, outline=outline, width=2)
    draw.pieslice(
        (cx - head_r * 1.22, cy - radius * 0.82, cx + head_r * 1.22, cy - radius * 0.18),
        180,
        360,
        fill=blue_light,
        outline=outline,
        width=2,
    )
    draw.rectangle((cx - head_r * 0.95, cy - radius * 0.49, cx + head_r * 0.95, cy - radius * 0.37), fill=blue_light, outline=outline)
    draw.line((cx, cy - radius * 0.78, cx, cy - radius * 0.24), fill=(235, 243, 255), width=max(2, int(tile_px * 0.025)))
    return RpgTacticalUnit(
        unit_id="blue_unit",
        public_name="blue unit",
        team="blue",
        tile_id=str(tile.tile_id),
        bbox_xyxy=(round(bbox[0], 3), round(bbox[1], 3), round(bbox[2], 3), round(bbox[3], 3)),
        point_xy=(round(cx, 3), round(cy, 3)),
        metadata={"role": "reference", "tile_coord": [int(tile.row), int(tile.col)]},
    )


def _mix_rgb(left: RGB, right: RGB, amount: float) -> RGB:
    ratio = max(0.0, min(1.0, float(amount)))
    return tuple(
        int(round(float(a) * (1.0 - ratio) + float(b) * ratio))
        for a, b in zip(left, right, strict=True)
    )


def _draw_candidate_labels(
    draw: ImageDraw.ImageDraw,
    *,
    tiles_by_id: Mapping[str, RpgTacticalTile],
    candidate_tile_ids_by_label: Mapping[str, str],
    tile_px: int,
    font_family: str | None,
) -> dict[str, BBox]:
    label_bboxes: dict[str, BBox] = {}
    for label, tile_id in candidate_tile_ids_by_label.items():
        tile = tiles_by_id[str(tile_id)]
        size = max(28, int(tile_px * 0.42))
        cx, cy = float(tile.point_xy[0]), float(tile.point_xy[1])
        bbox = (
            round(cx - size * 0.5, 3),
            round(cy - size * 0.5, 3),
            round(cx + size * 0.5, 3),
            round(cy + size * 0.5, 3),
        )
        draw_label_badge(
            draw,
            str(label),
            bbox,
            font_family=font_family,
            fill=(255, 244, 180),
            outline=(54, 48, 34),
            text_fill=(24, 22, 18),
            radius=6,
            width=3,
        )
        label_bboxes[str(tile.tile_id)] = bbox
    return label_bboxes


def _draw_target_markers(
    draw: ImageDraw.ImageDraw,
    *,
    tiles_by_id: Mapping[str, RpgTacticalTile],
    target_tile_ids: Sequence[str],
    tile_px: int,
) -> None:
    """Draw visible destination markers without introducing option labels."""

    for tile_id in target_tile_ids:
        if str(tile_id) not in tiles_by_id:
            raise ValueError(f"unknown target tile id {tile_id!r}")
        tile = tiles_by_id[str(tile_id)]
        x0, y0, x1, y1 = [int(round(value)) for value in tile.bbox_xyxy]
        outline_width = max(4, int(tile_px * 0.06))
        marker_bbox = (x0, y0, x1 - 1, y1 - 1)
        draw.rectangle(marker_bbox, outline=(232, 24, 36), width=outline_width)


__all__ = [
    "DEFAULT_CANDIDATE_LABELS",
    "DEFAULT_TILE_PX",
    "RENDERER_ID",
    "SCENE_ID",
    "TACTICAL_PROFILE_GRIDS",
    "TACTICAL_PROFILE_SUPPORT",
    "THEMES",
    "WATER_FEATURE_STYLE_STRAIGHT",
    "WATER_FEATURE_STYLE_ZIGZAG",
    "WATER_FEATURE_STYLES",
    "map_spanning_water_cells",
    "render_rpg_tactical_map_scene",
    "resolve_tactical_map_render_params",
    "resolve_water_feature_style",
]
