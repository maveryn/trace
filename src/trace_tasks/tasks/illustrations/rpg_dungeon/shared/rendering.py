"""Scene-local top-down RPG dungeon renderer."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.illustrations.shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    render_illustration_object,
)
from trace_tasks.tasks.illustrations.shared.object_variants import RENDERER_STYLE_TOP_DOWN_PIXEL_RPG
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import CANONICAL_TILE_PX
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import DEFAULT_RPG_TILE_PX

from .relations import reachable_entity_ids, reachable_tiles
from .state import (
    BBox,
    RpgDungeonBlocker,
    RpgDungeonChamber,
    RpgDungeonEntity,
    RpgDungeonScene,
    Tile,
    TileBox,
)


SCENE_ID = "rpg_dungeon"
RENDERER_ID = "rpg_dungeon_top_down_v0"
DEFAULT_TILE_PX = DEFAULT_RPG_TILE_PX
DEFAULT_CANVAS_WIDTH = 27 * DEFAULT_TILE_PX
DEFAULT_CANVAS_HEIGHT = 18 * DEFAULT_TILE_PX
MIN_TOTAL_CHEST_COUNT = 4
MAX_TOTAL_CHEST_COUNT = 6
DEFAULT_TOTAL_CHEST_COUNT = 5
MIN_REACHABLE_CHEST_COUNT = 0
MAX_REACHABLE_CHEST_COUNT = MAX_TOTAL_CHEST_COUNT
MIN_MONSTER_CHAMBER_COUNT = 0
MAX_MONSTER_CHAMBER_COUNT = 4
MONSTER_OBJECT_TYPES = ("monster_slime", "monster_bat", "monster_spider")
MONSTER_PUBLIC_NAMES: Mapping[str, str] = {
    "monster_slime": "slime",
    "monster_bat": "bat",
    "monster_spider": "spider",
}

RGB = tuple[int, int, int]

THEMES: Mapping[str, Mapping[str, Any]] = {
    "blue_stone": {
        "background_rgb": (31, 34, 43),
        "floor_rgb": (79, 88, 102),
        "floor_alt_rgb": (88, 96, 111),
        "floor_line_rgb": (55, 62, 74),
        "wall_edge_rgb": (28, 30, 37),
        "wall_light_rgb": (118, 126, 139),
        "chest_wood_rgb": (142, 84, 42),
        "chest_metal_rgb": (210, 172, 72),
    },
    "green_ruin": {
        "background_rgb": (33, 40, 35),
        "floor_rgb": (75, 91, 78),
        "floor_alt_rgb": (86, 101, 84),
        "floor_line_rgb": (52, 65, 56),
        "wall_edge_rgb": (25, 32, 28),
        "wall_light_rgb": (112, 132, 111),
        "chest_wood_rgb": (130, 79, 43),
        "chest_metal_rgb": (196, 161, 68),
    },
    "red_crypt": {
        "background_rgb": (42, 32, 35),
        "floor_rgb": (94, 78, 80),
        "floor_alt_rgb": (106, 86, 86),
        "floor_line_rgb": (66, 49, 54),
        "wall_edge_rgb": (34, 25, 28),
        "wall_light_rgb": (138, 114, 112),
        "chest_wood_rgb": (116, 67, 42),
        "chest_metal_rgb": (216, 154, 79),
    },
}


@dataclass(frozen=True)
class RpgDungeonLayout:
    """Tile grid and display geometry."""

    cols: int
    rows: int
    tile_px: int
    width_px: int
    height_px: int
    display_offset_xy: tuple[int, int]

    @property
    def canonical_width_px(self) -> int:
        return int(self.cols) * CANONICAL_TILE_PX

    @property
    def canonical_height_px(self) -> int:
        return int(self.rows) * CANONICAL_TILE_PX

    @property
    def display_grid_width_px(self) -> int:
        return int(self.cols) * int(self.tile_px)

    @property
    def display_grid_height_px(self) -> int:
        return int(self.rows) * int(self.tile_px)


@dataclass(frozen=True)
class _ChamberSpec:
    chamber_id: str
    public_name: str
    tile_xywh: TileBox
    layout_role: str
    side_id: str | None
    slot_index: int | None
    layout_orientation: str


@dataclass(frozen=True)
class _EntitySpec:
    entity_id: str
    object_type: str
    public_name: str
    chamber_id: str | None
    tile_xywh: TileBox
    role: str
    visual: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _BlockerSpec:
    blocker_id: str
    edge_id: str
    blocker_type: str
    tile_xy: Tile


@dataclass(frozen=True)
class _EdgeSpec:
    edge_id: str
    source_chamber_id: str
    target_chamber_id: str
    path: tuple[Tile, ...]


@dataclass(frozen=True)
class _LayoutSpec:
    chambers: tuple[_ChamberSpec, ...]
    edge_specs: tuple[_EdgeSpec, ...]
    layout_orientation: str
    side_counts: Mapping[str, int]
    total_chest_count: int
    floor_tiles: tuple[Tile, ...]
    corridor_tiles: tuple[Tile, ...]
    blocked_tiles: tuple[Tile, ...]
    blocked_edge_ids: tuple[str, ...]
    blocker_specs: tuple[_BlockerSpec, ...]
    entity_specs: tuple[_EntitySpec, ...]
    player_tile: Tile
    chest_tile_map: Mapping[str, Tile]
    reachable_chest_ids: tuple[str, ...]
    monster_chamber_ids: tuple[str, ...]
    monster_entity_ids: tuple[str, ...]


def render_rpg_dungeon_scene(
    seed: int,
    *,
    width: int = DEFAULT_CANVAS_WIDTH,
    height: int = DEFAULT_CANVAS_HEIGHT,
    tile_px: int = DEFAULT_TILE_PX,
    reachable_chest_count: int | None = None,
    total_chest_count: int | None = None,
    monster_chamber_count: int | None = None,
    reachable_monster_chamber_count: int | None = None,
    render_metadata: Mapping[str, Any] | None = None,
) -> RpgDungeonScene:
    """Render one top-down RPG dungeon layout with reachable chest metadata."""

    rng = random.Random(int(seed))
    layout = _sample_layout(width=int(width), height=int(height), tile_px=int(tile_px))
    theme_id = str(_choose(rng, tuple(THEMES)))
    theme = THEMES[theme_id]
    resolved_total_count = _resolve_total_chest_count(rng, total_chest_count)
    target_count = _resolve_reachable_chest_count(
        rng,
        reachable_chest_count,
        total_chest_count=int(resolved_total_count),
    )
    resolved_monster_count = _resolve_monster_chamber_count(
        monster_chamber_count,
        total_chest_count=int(resolved_total_count),
    )
    layout_spec = _make_valid_layout_spec(
        layout,
        rng=rng,
        target_count=target_count,
        total_chest_count=int(resolved_total_count),
        monster_chamber_count=int(resolved_monster_count),
        reachable_monster_chamber_count=reachable_monster_chamber_count,
    )

    canonical, draw = _render_base(layout, layout_spec=layout_spec, theme=theme, rng=rng)
    entities = _render_entities(
        draw,
        entity_specs=layout_spec.entity_specs,
        layout=layout,
        theme=theme,
        theme_id=theme_id,
    )
    blockers = _render_blockers(
        draw,
        blocker_specs=layout_spec.blocker_specs,
        layout=layout,
        theme=theme,
        theme_id=theme_id,
    )
    scaled = canonical.resize((layout.display_grid_width_px, layout.display_grid_height_px), Image.Resampling.NEAREST)
    final = Image.new("RGB", (int(width), int(height)), _shade(theme["background_rgb"], -4))
    final.paste(scaled.convert("RGB"), layout.display_offset_xy)
    chambers = tuple(
        RpgDungeonChamber(
            chamber_id=spec.chamber_id,
            public_name=spec.public_name,
            tile_xywh=spec.tile_xywh,
            bbox_xyxy=_tile_bbox(layout, spec.tile_xywh),
            metadata={
                "kind": "carved_chamber",
                "layout_role": spec.layout_role,
                "side_id": spec.side_id,
                "slot_index": spec.slot_index,
                "layout_orientation": spec.layout_orientation,
            },
        )
        for spec in layout_spec.chambers
    )
    trace = {
        "renderer_id": RENDERER_ID,
        "renderer_style": RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
        "theme_id": theme_id,
        "width": int(width),
        "height": int(height),
        "grid_cols": int(layout.cols),
        "grid_rows": int(layout.rows),
        "tile_px": int(layout.tile_px),
        "layout_orientation": str(layout_spec.layout_orientation),
        "side_counts": {str(key): int(value) for key, value in sorted(layout_spec.side_counts.items())},
        "total_chest_count": int(layout_spec.total_chest_count),
        "reachable_chest_target": int(target_count),
        "reachable_chest_ids": [str(entity_id) for entity_id in layout_spec.reachable_chest_ids],
        "monster_count": int(len(layout_spec.monster_chamber_ids)),
        "monster_chamber_ids": [str(chamber_id) for chamber_id in layout_spec.monster_chamber_ids],
        "monster_entity_ids": [str(entity_id) for entity_id in layout_spec.monster_entity_ids],
        "monster_type_counts": _monster_type_counts(layout_spec.entity_specs),
        "edge_ids": [str(edge.edge_id) for edge in layout_spec.edge_specs],
        "blocked_edge_ids": [str(edge_id) for edge_id in layout_spec.blocked_edge_ids],
        "player_tile": [int(value) for value in layout_spec.player_tile],
        "chest_tile_map": {
            str(entity_id): [int(tile[0]), int(tile[1])]
            for entity_id, tile in sorted(layout_spec.chest_tile_map.items())
        },
        **dict(render_metadata or {}),
    }
    return RpgDungeonScene(
        image=final,
        chambers=chambers,
        floor_tiles=layout_spec.floor_tiles,
        blocked_tiles=layout_spec.blocked_tiles,
        corridor_tiles=layout_spec.corridor_tiles,
        blockers=tuple(sorted(blockers, key=lambda blocker: blocker.blocker_id)),
        entities=tuple(sorted(entities, key=lambda entity: (entity.role, entity.entity_id))),
        player_entity_id="player_00",
        chest_entity_ids=tuple(sorted(layout_spec.chest_tile_map)),
        reachable_chest_ids=tuple(str(entity_id) for entity_id in layout_spec.reachable_chest_ids),
        trace=trace,
    )


def rpg_dungeon_profile_metadata(render_params: Mapping[str, Any]) -> dict[str, Any]:
    """Return render trace metadata for the selected shared canvas profile."""

    return {
        "canvas_profile": str(render_params.get("canvas_profile", "")),
        "canvas_profile_size": list(render_params.get("canvas_profile_size", [])),
        "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
        "rpg_tile_profile": dict(render_params.get("rpg_tile_profile", {})),
    }


def render_rpg_dungeon_profile_scene(
    seed: int,
    *,
    render_params: Mapping[str, Any],
    tile_px: int,
    reachable_chest_count: int,
    total_chest_count: int | None = None,
    monster_chamber_count: int | None = None,
    reachable_monster_chamber_count: int | None = None,
    render_metadata: Mapping[str, Any] | None = None,
) -> RpgDungeonScene:
    """Render an RPG dungeon using resolved canvas-profile parameters."""

    metadata = rpg_dungeon_profile_metadata(render_params)
    if render_metadata:
        metadata.update({str(key): value for key, value in render_metadata.items()})
    return render_rpg_dungeon_scene(
        int(seed),
        width=int(render_params["canvas_width"]),
        height=int(render_params["canvas_height"]),
        tile_px=int(tile_px),
        reachable_chest_count=int(reachable_chest_count),
        total_chest_count=total_chest_count,
        monster_chamber_count=monster_chamber_count,
        reachable_monster_chamber_count=reachable_monster_chamber_count,
        render_metadata=metadata,
    )


def draw_rpg_dungeon_debug_overlay(scene: RpgDungeonScene) -> Image.Image:
    """Return an overlay image with chamber, blocker, and entity bboxes."""

    image = scene.image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for chamber in scene.chambers:
        draw.rectangle(chamber.bbox_xyxy, outline=(42, 111, 210, 210), width=3)
    for blocker in scene.blockers:
        draw.rectangle(blocker.bbox_xyxy, outline=(214, 67, 59, 230), width=3)
    for entity in scene.entities:
        color = (38, 152, 91, 235) if entity.entity_id in set(scene.reachable_chest_ids) else (239, 173, 42, 220)
        if entity.entity_id == scene.player_entity_id:
            color = (88, 142, 246, 240)
        draw.rectangle(entity.bbox_xyxy, outline=color, width=2)
    return Image.alpha_composite(image, overlay).convert("RGB")


def _sample_layout(*, width: int, height: int, tile_px: int) -> RpgDungeonLayout:
    resolved_tile_px = max(40, min(56, int(tile_px)))
    cols = max(18, int(round(float(width) / float(resolved_tile_px))))
    rows = max(18, int(round(float(height) / float(resolved_tile_px))))
    grid_w = int(cols) * resolved_tile_px
    grid_h = int(rows) * resolved_tile_px
    return RpgDungeonLayout(
        cols=int(cols),
        rows=int(rows),
        tile_px=int(resolved_tile_px),
        width_px=int(width),
        height_px=int(height),
        display_offset_xy=((int(width) - grid_w) // 2, (int(height) - grid_h) // 2),
    )


def _resolve_total_chest_count(rng: random.Random, total_chest_count: int | None) -> int:
    if total_chest_count is None:
        return int(rng.randint(MIN_TOTAL_CHEST_COUNT, MAX_TOTAL_CHEST_COUNT))
    value = int(total_chest_count)
    if not MIN_TOTAL_CHEST_COUNT <= value <= MAX_TOTAL_CHEST_COUNT:
        raise ValueError(
            f"total_chest_count must be in [{MIN_TOTAL_CHEST_COUNT}, {MAX_TOTAL_CHEST_COUNT}], got {value}"
        )
    return value


def _resolve_reachable_chest_count(
    rng: random.Random,
    reachable_chest_count: int | None,
    *,
    total_chest_count: int,
) -> int:
    if reachable_chest_count is None:
        return int(rng.randint(MIN_REACHABLE_CHEST_COUNT, int(total_chest_count)))
    value = int(reachable_chest_count)
    if not MIN_REACHABLE_CHEST_COUNT <= value <= int(total_chest_count):
        raise ValueError(
            f"reachable_chest_count must be in [{MIN_REACHABLE_CHEST_COUNT}, {total_chest_count}], got {value}"
        )
    return value


def _resolve_monster_chamber_count(
    monster_chamber_count: int | None,
    *,
    total_chest_count: int,
) -> int:
    if monster_chamber_count is None:
        return 0
    value = int(monster_chamber_count)
    max_count = min(MAX_MONSTER_CHAMBER_COUNT, int(total_chest_count))
    if not MIN_MONSTER_CHAMBER_COUNT <= value <= max_count:
        raise ValueError(
            f"monster count must be in [{MIN_MONSTER_CHAMBER_COUNT}, {max_count}], got {value}"
        )
    return value


def _make_valid_layout_spec(
    layout: RpgDungeonLayout,
    *,
    rng: random.Random,
    target_count: int,
    total_chest_count: int,
    monster_chamber_count: int,
    reachable_monster_chamber_count: int | None,
) -> _LayoutSpec:
    for attempt in range(80):
        spec = _make_layout_spec(
            layout,
            rng=random.Random(rng.randrange(1 << 62) + int(attempt)),
            target_count=target_count,
            total_chest_count=int(total_chest_count),
            monster_chamber_count=int(monster_chamber_count),
            reachable_monster_chamber_count=reachable_monster_chamber_count,
        )
        reachable_ids = _reachable_chests_for_spec(spec)
        if len(reachable_ids) == int(target_count):
            return _replace_reachable_ids(spec, reachable_ids)
    raise ValueError(f"could not construct RPG dungeon with {target_count} of {total_chest_count} reachable chests")


def _make_layout_spec(
    layout: RpgDungeonLayout,
    *,
    rng: random.Random,
    target_count: int,
    total_chest_count: int,
    monster_chamber_count: int,
    reachable_monster_chamber_count: int | None,
) -> _LayoutSpec:
    """Construct the carved graph while keeping target answer control explicit."""

    chamber_specs = _make_chamber_specs(layout, rng=rng, total_chest_count=int(total_chest_count))
    chamber_by_id = {spec.chamber_id: spec for spec in chamber_specs}
    start = chamber_by_id["start"]
    start_tile = _rect_center_tile(start.tile_xywh)
    chest_chambers = tuple(spec for spec in chamber_specs if spec.chamber_id != "start")
    chest_ids = tuple(f"chest_{index:02d}" for index in range(len(chest_chambers)))
    shuffled_chest_ids = list(chest_ids)
    rng.shuffle(shuffled_chest_ids)
    reachable_chest_id_set = frozenset(str(chest_id) for chest_id in shuffled_chest_ids[: int(target_count)])
    chest_id_by_chamber_id = {
        str(chamber.chamber_id): str(chest_ids[index])
        for index, chamber in enumerate(chest_chambers)
    }
    reachable_chamber_ids = frozenset(
        str(chamber_id)
        for chamber_id, chest_id in chest_id_by_chamber_id.items()
        if str(chest_id) in reachable_chest_id_set
    )
    monster_chamber_ids = _select_monster_chamber_ids(
        chest_chambers,
        monster_chamber_count=int(monster_chamber_count),
        reachable_chamber_ids=reachable_chamber_ids,
        reachable_monster_chamber_count=reachable_monster_chamber_count,
        rng=rng,
    )
    edge_specs = _make_edge_specs(
        chamber_specs,
        reachable_chamber_ids=reachable_chamber_ids,
    )

    floor: set[Tile] = set()
    for chamber in chamber_specs:
        floor.update(_rect_tiles(chamber.tile_xywh))
    corridor_tiles: set[Tile] = set()
    chamber_floor_tiles = set(floor)
    for edge in edge_specs:
        floor.update(edge.path)
        corridor_tiles.update(tile for tile in edge.path if tile not in chamber_floor_tiles)
    chest_tile_map: dict[str, Tile] = {}
    chest_box_by_chamber_id: dict[str, TileBox] = {}
    entity_specs: list[_EntitySpec] = [
        _player_spec(start_tile),
    ]
    for index, chamber in enumerate(chest_chambers):
        chest_id = chest_ids[index]
        chest_box = _centered_box(chamber.tile_xywh, width_tiles=2, height_tiles=1)
        chest_tile_map[chest_id] = _rect_center_tile(chest_box)
        chest_box_by_chamber_id[str(chamber.chamber_id)] = chest_box
        entity_specs.append(
            _EntitySpec(
                entity_id=chest_id,
                object_type="chest",
                public_name="treasure chest",
                chamber_id=chamber.chamber_id,
                tile_xywh=chest_box,
                role="queryable",
                visual={
                    "wood_rgb": _choose(rng, ((126, 71, 39), (149, 84, 41), (116, 76, 48))),
                    "metal_rgb": _choose(rng, ((218, 171, 67), (198, 181, 96), (216, 142, 76))),
                },
            )
        )
    monster_entity_ids: list[str] = []
    for index, chamber_id in enumerate(monster_chamber_ids):
        chamber = chamber_by_id[str(chamber_id)]
        monster_id = f"monster_{index:02d}"
        monster_entity_ids.append(monster_id)
        entity_specs.append(
            _monster_spec(
                entity_id=monster_id,
                chamber=chamber,
                chest_box=chest_box_by_chamber_id[str(chamber.chamber_id)],
                rng=rng,
            )
        )

    blockers = _make_start_edge_blockers(
        chamber_by_id=chamber_by_id,
        edge_specs=edge_specs,
        reachable_chamber_ids=reachable_chamber_ids,
    )
    blocked_edge_ids = tuple(sorted({str(blocker.edge_id) for blocker in blockers}))
    blocked_tiles = tuple(sorted({blocker.tile_xy for blocker in blockers}, key=lambda tile: (tile[1], tile[0])))
    reached = reachable_tiles(floor, blocked_tiles=blocked_tiles, start_tile=start_tile)
    reachable_chest_ids = reachable_entity_ids(entity_tile_map=chest_tile_map, reachable_tile_set=reached)
    return _LayoutSpec(
        chambers=tuple(chamber_specs),
        edge_specs=tuple(edge_specs),
        layout_orientation=chamber_specs[0].layout_orientation,
        side_counts=_side_counts(chamber_specs),
        total_chest_count=len(chest_tile_map),
        floor_tiles=tuple(sorted(floor, key=lambda tile: (tile[1], tile[0]))),
        corridor_tiles=tuple(sorted(corridor_tiles, key=lambda tile: (tile[1], tile[0]))),
        blocked_tiles=blocked_tiles,
        blocked_edge_ids=tuple(sorted(blocked_edge_ids)),
        blocker_specs=tuple(blockers),
        entity_specs=tuple(entity_specs),
        player_tile=start_tile,
        chest_tile_map=chest_tile_map,
        reachable_chest_ids=tuple(str(entity_id) for entity_id in sorted(reachable_chest_ids)),
        monster_chamber_ids=tuple(str(chamber_id) for chamber_id in monster_chamber_ids),
        monster_entity_ids=tuple(str(entity_id) for entity_id in monster_entity_ids),
    )


def _replace_reachable_ids(spec: _LayoutSpec, reachable_ids: Sequence[str]) -> _LayoutSpec:
    return _LayoutSpec(
        chambers=spec.chambers,
        edge_specs=spec.edge_specs,
        layout_orientation=spec.layout_orientation,
        side_counts=spec.side_counts,
        total_chest_count=spec.total_chest_count,
        floor_tiles=spec.floor_tiles,
        corridor_tiles=spec.corridor_tiles,
        blocked_tiles=spec.blocked_tiles,
        blocked_edge_ids=spec.blocked_edge_ids,
        blocker_specs=spec.blocker_specs,
        entity_specs=spec.entity_specs,
        player_tile=spec.player_tile,
        chest_tile_map=spec.chest_tile_map,
        reachable_chest_ids=tuple(sorted(str(entity_id) for entity_id in reachable_ids)),
        monster_chamber_ids=spec.monster_chamber_ids,
        monster_entity_ids=spec.monster_entity_ids,
    )


def _reachable_chests_for_spec(spec: _LayoutSpec) -> tuple[str, ...]:
    reached = reachable_tiles(spec.floor_tiles, blocked_tiles=spec.blocked_tiles, start_tile=spec.player_tile)
    return reachable_entity_ids(entity_tile_map=dict(spec.chest_tile_map), reachable_tile_set=reached)


def _make_chamber_specs(
    layout: RpgDungeonLayout,
    *,
    rng: random.Random,
    total_chest_count: int,
) -> tuple[_ChamberSpec, ...]:
    """Create a profile-aware two-sided dungeon room layout."""

    room_w = 4
    room_h = 4
    start_x = int(layout.cols // 2) - 2
    start_y = int(layout.rows // 2) - 2
    orientation = _choose_layout_orientation(layout, rng=rng)
    side_counts = _choose_side_counts(int(total_chest_count), rng=rng)
    if orientation == "top_bottom":
        side_ids = ("top", "bottom")
        side_origins = {
            "top": 1 + int(rng.choice((0, 1))),
            "bottom": int(layout.rows) - room_h - 1 - int(rng.choice((0, 1))),
        }
        axis_origins = _slot_origins(
            total_len=int(layout.cols),
            room_size=room_w,
            rng=rng,
        )
        make_box = lambda side_id, slot: (axis_origins[int(slot)], side_origins[str(side_id)], room_w, room_h)
    else:
        side_ids = ("left", "right")
        side_origins = {
            "left": 1 + int(rng.choice((0, 1))),
            "right": int(layout.cols) - room_w - 1 - int(rng.choice((0, 1))),
        }
        axis_origins = _slot_origins(
            total_len=int(layout.rows),
            room_size=room_h,
            rng=rng,
        )
        make_box = lambda side_id, slot: (side_origins[str(side_id)], axis_origins[int(slot)], room_w, room_h)

    specs: list[_ChamberSpec] = [
        _ChamberSpec(
            "start",
            "starting chamber",
            (start_x, start_y, 4, 4),
            "start",
            None,
            None,
            orientation,
        )
    ]
    room_index = 0
    for side_id, count in zip(side_ids, side_counts, strict=True):
        slots = _slot_indices_for_count(int(count))
        for slot_index in slots:
            specs.append(
                _ChamberSpec(
                    chamber_id=f"chest_room_{room_index:02d}",
                    public_name=f"{side_id} chest chamber {slot_index + 1}",
                    tile_xywh=make_box(side_id, slot_index),
                    layout_role="chest_room",
                    side_id=str(side_id),
                    slot_index=int(slot_index),
                    layout_orientation=orientation,
                )
            )
            room_index += 1
    return tuple(specs)


def _choose_layout_orientation(layout: RpgDungeonLayout, *, rng: random.Random) -> str:
    if int(layout.rows) > int(layout.cols):
        return str(_choose(rng, ("left_right", "left_right", "top_bottom")))
    if int(layout.cols) > int(layout.rows):
        return str(_choose(rng, ("top_bottom", "top_bottom", "left_right")))
    return str(_choose(rng, ("top_bottom", "left_right")))


def _choose_side_counts(total_chest_count: int, *, rng: random.Random) -> tuple[int, int]:
    if int(total_chest_count) == 4:
        return (2, 2)
    if int(total_chest_count) == 6:
        return (3, 3)
    if int(total_chest_count) == 5:
        return tuple(_choose(rng, ((3, 2), (2, 3))))  # type: ignore[return-value]
    raise ValueError(f"total_chest_count must be 4, 5, or 6; got {total_chest_count}")


def _slot_origins(*, total_len: int, room_size: int, rng: random.Random) -> tuple[int, int, int]:
    start = 1 + int(rng.choice((0, 1)))
    end = int(total_len) - int(room_size) - 1 - int(rng.choice((0, 1)))
    center = int(total_len // 2) - int(room_size // 2) + int(rng.choice((-1, 0, 1)))
    center = max(start + int(room_size) + 2, min(end - int(room_size) - 2, center))
    return (int(start), int(center), int(end))


def _slot_indices_for_count(count: int) -> tuple[int, ...]:
    if int(count) == 3:
        return (0, 1, 2)
    if int(count) != 2:
        raise ValueError(f"side room count must be 2 or 3; got {count}")
    return (0, 2)


def _side_counts(chamber_specs: Sequence[_ChamberSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chamber in chamber_specs:
        if chamber.side_id is None:
            continue
        counts[str(chamber.side_id)] = counts.get(str(chamber.side_id), 0) + 1
    return counts


def _branch_path(start_tile: Tile, target_tile: Tile, *, layout_orientation: str) -> tuple[Tile, ...]:
    sx, sy = start_tile
    tx, ty = target_tile
    path: list[Tile] = []
    if str(layout_orientation) == "top_bottom":
        _append_horizontal(path, sy, sx, tx)
        _append_vertical(path, tx, sy, ty)
    else:
        _append_vertical(path, sx, sy, ty)
        _append_horizontal(path, ty, sx, tx)
    return tuple(dict.fromkeys(path))


def _make_edge_specs(
    chamber_specs: Sequence[_ChamberSpec],
    *,
    reachable_chamber_ids: frozenset[str],
) -> tuple[_EdgeSpec, ...]:
    """Return room-to-room graph edges used by reachability and blocker placement."""

    chamber_by_id = {chamber.chamber_id: chamber for chamber in chamber_specs}
    start = chamber_by_id["start"]
    chest_rooms = tuple(chamber for chamber in chamber_specs if chamber.layout_role == "chest_room")
    edges: list[_EdgeSpec] = []
    source_tile = _rect_center_tile(start.tile_xywh)
    for target in chest_rooms:
        target_tile = _rect_center_tile(target.tile_xywh)
        path = _branch_path(source_tile, target_tile, layout_orientation=target.layout_orientation)
        edges.append(
            _EdgeSpec(
                edge_id=f"edge_start_{target.chamber_id}",
                source_chamber_id="start",
                target_chamber_id=target.chamber_id,
                path=path,
            )
        )
    rooms_by_side: dict[str, list[_ChamberSpec]] = {}
    for chamber in chest_rooms:
        rooms_by_side.setdefault(str(chamber.side_id), []).append(chamber)
    for side_id, side_rooms in sorted(rooms_by_side.items()):
        ordered = sorted(side_rooms, key=lambda chamber: int(chamber.slot_index or 0))
        for source, target in zip(ordered, ordered[1:]):
            source_reachable = str(source.chamber_id) in reachable_chamber_ids
            target_reachable = str(target.chamber_id) in reachable_chamber_ids
            if source_reachable != target_reachable:
                continue
            source_tile = _rect_center_tile(source.tile_xywh)
            target_tile = _rect_center_tile(target.tile_xywh)
            edges.append(
                _EdgeSpec(
                    edge_id=f"edge_{side_id}_{source.chamber_id}_{target.chamber_id}",
                    source_chamber_id=source.chamber_id,
                    target_chamber_id=target.chamber_id,
                    path=_straight_path(source_tile, target_tile),
                )
            )
    return tuple(edges)


def _straight_path(source_tile: Tile, target_tile: Tile) -> tuple[Tile, ...]:
    sx, sy = source_tile
    tx, ty = target_tile
    path: list[Tile] = []
    if int(sy) == int(ty):
        _append_horizontal(path, sy, sx, tx)
    elif int(sx) == int(tx):
        _append_vertical(path, sx, sy, ty)
    else:
        _append_horizontal(path, sy, sx, tx)
        _append_vertical(path, tx, sy, ty)
    return tuple(dict.fromkeys(path))


def _append_horizontal(path: list[Tile], y: int, x0: int, x1: int) -> None:
    step = 1 if int(x1) >= int(x0) else -1
    for x in range(int(x0), int(x1) + step, step):
        path.append((int(x), int(y)))


def _append_vertical(path: list[Tile], x: int, y0: int, y1: int) -> None:
    step = 1 if int(y1) >= int(y0) else -1
    for y in range(int(y0), int(y1) + step, step):
        path.append((int(x), int(y)))


def _make_start_edge_blockers(
    *,
    chamber_by_id: Mapping[str, _ChamberSpec],
    edge_specs: Sequence[_EdgeSpec],
    reachable_chamber_ids: frozenset[str],
) -> tuple[_BlockerSpec, ...]:
    """Place one boulder near each unreachable room's start corridor entrance."""

    blockers: list[_BlockerSpec] = []
    for edge in edge_specs:
        if edge.source_chamber_id != "start" or edge.target_chamber_id in reachable_chamber_ids:
            continue
        blocker_tile = _start_edge_blocker_tile(edge, chamber_by_id=chamber_by_id)
        if blocker_tile is None:
            continue
        blockers.append(
            _BlockerSpec(
                blocker_id=f"blocker_{len(blockers):02d}",
                edge_id=str(edge.edge_id),
                blocker_type="boulder",
                tile_xy=blocker_tile,
            )
        )
    return tuple(blockers)


def _start_edge_blocker_tile(edge: _EdgeSpec, *, chamber_by_id: Mapping[str, _ChamberSpec]) -> Tile | None:
    source_tiles = set(_rect_tiles(chamber_by_id[edge.source_chamber_id].tile_xywh))
    target_tiles = set(_rect_tiles(chamber_by_id[edge.target_chamber_id].tile_xywh))
    candidates = [
        (index, tile)
        for index, tile in enumerate(edge.path)
        if tile not in source_tiles and tile not in target_tiles
    ]
    if not candidates:
        return None
    return candidates[-1][1]


def _player_spec(start_tile: Tile) -> _EntitySpec:
    return _EntitySpec(
        entity_id="player_00",
        object_type="person",
        public_name="player",
        chamber_id="start",
        tile_xywh=(int(start_tile[0]), int(start_tile[1]), 1, 1),
        role="reference",
        visual={
            "role": "player_marker",
            "person_variant_id": "soldier",
            "gender_id": "male",
            "facing": "down",
            "shirt_rgb": (63, 113, 197),
            "pants_rgb": (43, 58, 91),
            "hair_rgb": (61, 44, 34),
        },
    )


def _select_monster_chamber_ids(
    chest_chambers: Sequence[_ChamberSpec],
    *,
    monster_chamber_count: int,
    reachable_chamber_ids: frozenset[str],
    reachable_monster_chamber_count: int | None,
    rng: random.Random,
) -> tuple[str, ...]:
    if int(monster_chamber_count) <= 0:
        return ()
    if reachable_monster_chamber_count is not None:
        reachable = [str(chamber.chamber_id) for chamber in chest_chambers if str(chamber.chamber_id) in reachable_chamber_ids]
        unreachable = [str(chamber.chamber_id) for chamber in chest_chambers if str(chamber.chamber_id) not in reachable_chamber_ids]
        reachable_needed = int(reachable_monster_chamber_count)
        unreachable_needed = int(monster_chamber_count) - reachable_needed
        if reachable_needed < 0 or unreachable_needed < 0:
            raise ValueError(
                f"reachable monster split is invalid: total={monster_chamber_count}, reachable={reachable_needed}"
            )
        if reachable_needed > len(reachable) or unreachable_needed > len(unreachable):
            raise ValueError(
                "reachable monster split exceeds chamber support: "
                f"reachable_needed={reachable_needed}, reachable={len(reachable)}, "
                f"unreachable_needed={unreachable_needed}, unreachable={len(unreachable)}"
            )
        rng.shuffle(reachable)
        rng.shuffle(unreachable)
        return tuple(sorted(reachable[:reachable_needed] + unreachable[:unreachable_needed]))
    chamber_ids = [str(chamber.chamber_id) for chamber in chest_chambers]
    rng.shuffle(chamber_ids)
    return tuple(sorted(chamber_ids[: int(monster_chamber_count)]))


def _monster_spec(
    *,
    entity_id: str,
    chamber: _ChamberSpec,
    chest_box: TileBox,
    rng: random.Random,
) -> _EntitySpec:
    object_type = str(_choose(rng, MONSTER_OBJECT_TYPES))
    return _EntitySpec(
        entity_id=str(entity_id),
        object_type=object_type,
        public_name=str(MONSTER_PUBLIC_NAMES[object_type]),
        chamber_id=str(chamber.chamber_id),
        tile_xywh=_monster_box_for_chamber(chamber.tile_xywh, chest_box=chest_box, rng=rng),
        role="queryable",
        visual={
            "monster_type": object_type,
            "style": "top_down_pixel_creature",
        },
    )


def _monster_box_for_chamber(chamber_box: TileBox, *, chest_box: TileBox, rng: random.Random) -> TileBox:
    chamber_tiles = list(_rect_tiles(chamber_box))
    chest_tiles = set(_rect_tiles(chest_box))
    x, y, w, h = chamber_box
    interior_tiles = {
        (tx, ty)
        for tx, ty in chamber_tiles
        if int(x) < int(tx) < int(x) + int(w) - 1 and int(y) < int(ty) < int(y) + int(h) - 1
    }
    candidates = sorted(interior_tiles - chest_tiles, key=lambda tile: (tile[1], tile[0]))
    if not candidates:
        candidates = sorted(set(chamber_tiles) - chest_tiles, key=lambda tile: (tile[1], tile[0]))
    if not candidates:
        raise ValueError(f"could not place monster in chamber {chamber_box}")
    tile = _choose(rng, candidates)
    return (int(tile[0]), int(tile[1]), 1, 1)


def _monster_type_counts(entity_specs: Sequence[_EntitySpec]) -> dict[str, int]:
    counts = {str(object_type): 0 for object_type in MONSTER_OBJECT_TYPES}
    for spec in entity_specs:
        if spec.object_type in counts:
            counts[str(spec.object_type)] += 1
    return {key: value for key, value in counts.items() if int(value) > 0}


def _render_base(
    layout: RpgDungeonLayout,
    *,
    layout_spec: _LayoutSpec,
    theme: Mapping[str, Any],
    rng: random.Random,
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (layout.canonical_width_px, layout.canonical_height_px), _rgba(theme["background_rgb"]))
    draw = ImageDraw.Draw(image)
    _draw_background_stone(draw, layout, theme=theme, rng=rng)
    floor = set(layout_spec.floor_tiles)
    for tile in sorted(floor, key=lambda item: (item[1], item[0])):
        _draw_floor_tile(draw, tile, theme=theme)
    _draw_wall_edges(draw, floor_tiles=floor, theme=theme)
    return image, draw


def _draw_background_stone(
    draw: ImageDraw.ImageDraw,
    layout: RpgDungeonLayout,
    *,
    theme: Mapping[str, Any],
    rng: random.Random,
) -> None:
    bg = theme["background_rgb"]
    for y in range(0, layout.canonical_height_px, CANONICAL_TILE_PX):
        for x in range(0, layout.canonical_width_px, CANONICAL_TILE_PX):
            delta = int(rng.choice((-6, -3, 0, 3)))
            draw.rectangle(
                (x, y, x + CANONICAL_TILE_PX - 1, y + CANONICAL_TILE_PX - 1),
                fill=_rgba(_shade(bg, delta)),
            )
            if rng.random() < 0.18:
                draw.line((x + 2, y + 3, x + 12, y + 3), fill=_rgba(_shade(bg, 18), 80))


def _draw_floor_tile(draw: ImageDraw.ImageDraw, tile: Tile, *, theme: Mapping[str, Any]) -> None:
    x, y = tile
    x0 = int(x) * CANONICAL_TILE_PX
    y0 = int(y) * CANONICAL_TILE_PX
    fill = theme["floor_rgb"] if (int(x) + int(y)) % 2 == 0 else theme["floor_alt_rgb"]
    draw.rectangle((x0, y0, x0 + 15, y0 + 15), fill=_rgba(fill))
    draw.rectangle((x0, y0, x0 + 15, y0 + 15), outline=_rgba(theme["floor_line_rgb"], 95))
    if (int(x) * 17 + int(y) * 31) % 7 == 0:
        draw.line((x0 + 3, y0 + 9, x0 + 12, y0 + 11), fill=_rgba(_shade(fill, -32), 120))


def _draw_wall_edges(draw: ImageDraw.ImageDraw, *, floor_tiles: set[Tile], theme: Mapping[str, Any]) -> None:
    edge = theme["wall_edge_rgb"]
    light = theme["wall_light_rgb"]
    for x, y in floor_tiles:
        x0 = int(x) * CANONICAL_TILE_PX
        y0 = int(y) * CANONICAL_TILE_PX
        if (x, y - 1) not in floor_tiles:
            draw.rectangle((x0, y0 - 2, x0 + 15, y0 + 2), fill=_rgba(edge))
            draw.line((x0 + 1, y0 + 2, x0 + 14, y0 + 2), fill=_rgba(light, 120))
        if (x, y + 1) not in floor_tiles:
            draw.rectangle((x0, y0 + 13, x0 + 15, y0 + 17), fill=_rgba(edge))
        if (x - 1, y) not in floor_tiles:
            draw.rectangle((x0 - 2, y0, x0 + 2, y0 + 15), fill=_rgba(edge))
            draw.line((x0 + 2, y0 + 1, x0 + 2, y0 + 14), fill=_rgba(light, 95))
        if (x + 1, y) not in floor_tiles:
            draw.rectangle((x0 + 13, y0, x0 + 17, y0 + 15), fill=_rgba(edge))


def _render_entities(
    draw: ImageDraw.ImageDraw,
    *,
    entity_specs: Sequence[_EntitySpec],
    layout: RpgDungeonLayout,
    theme: Mapping[str, Any],
    theme_id: str,
) -> list[RpgDungeonEntity]:
    """Draw entities on canonical tiles while storing final pixel bboxes."""

    entities: list[RpgDungeonEntity] = []
    for spec in entity_specs:
        bbox = _tile_bbox(layout, spec.tile_xywh)
        visual = {"theme_id": theme_id, "renderer_style": RENDERER_STYLE_TOP_DOWN_PIXEL_RPG}
        if spec.object_type == "chest":
            visual.update({"wood_rgb": theme["chest_wood_rgb"], "metal_rgb": theme["chest_metal_rgb"]})
        visual.update(dict(spec.visual or {}))
        rendered_record: Mapping[str, Any] | None = None
        if spec.object_type in MONSTER_OBJECT_TYPES:
            _draw_monster(draw, spec, theme=theme)
        else:
            rendered = render_illustration_object(
                IllustrationObjectSpec(
                    object_id=spec.entity_id,
                    object_type=spec.object_type,
                    public_name=spec.public_name,
                    bbox_xyxy=bbox,
                    tile_xywh=spec.tile_xywh,
                    renderer_id=RENDERER_ID,
                    renderer_variant_id=f"top_down:{theme_id}",
                    semantic_attributes={
                        "chamber_id": spec.chamber_id,
                        "role": spec.role,
                        "layout_context": str(spec.role) == "context",
                    },
                    visual_attributes=visual,
                    role=spec.role,
                    source_entity_type="rpg_dungeon_entity",
                ),
                RenderContext(renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG, draw=draw),
            )
            rendered_record = rendered.object_record
        point = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        metadata: dict[str, Any] = {"visual_attributes": visual}
        if rendered_record is not None:
            metadata["object_record"] = rendered_record
        entities.append(
            RpgDungeonEntity(
                entity_id=spec.entity_id,
                public_name=spec.public_name,
                object_type=spec.object_type,
                chamber_id=spec.chamber_id,
                tile_xywh=spec.tile_xywh,
                bbox_xyxy=bbox,
                point_xy=point,
                role=spec.role,
                metadata=metadata,
            )
        )
    return entities


def _draw_monster(draw: ImageDraw.ImageDraw, spec: _EntitySpec, *, theme: Mapping[str, Any]) -> None:
    x, y, w, h = spec.tile_xywh
    if int(w) != 1 or int(h) != 1:
        raise ValueError(f"monster entity must occupy one tile, got {spec.tile_xywh}")
    x0 = int(x) * CANONICAL_TILE_PX
    y0 = int(y) * CANONICAL_TILE_PX
    shadow = _rgba(_shade(theme["floor_rgb"], -38), 170)
    draw.ellipse((x0 + 2, y0 + 10, x0 + 14, y0 + 15), fill=shadow)
    if spec.object_type == "monster_slime":
        _draw_slime_monster(draw, x0, y0)
    elif spec.object_type == "monster_bat":
        _draw_bat_monster(draw, x0, y0)
    elif spec.object_type == "monster_spider":
        _draw_spider_monster(draw, x0, y0)
    else:
        raise ValueError(f"unsupported RPG dungeon monster type: {spec.object_type}")


def _draw_slime_monster(draw: ImageDraw.ImageDraw, x0: int, y0: int) -> None:
    outline = (16, 65, 37)
    body = (80, 220, 129)
    dark = (41, 144, 78)
    light = (165, 248, 187)
    draw.polygon(
        [
            (x0 + 2, y0 + 10),
            (x0 + 4, y0 + 5),
            (x0 + 7, y0 + 3),
            (x0 + 12, y0 + 5),
            (x0 + 14, y0 + 11),
            (x0 + 12, y0 + 14),
            (x0 + 4, y0 + 14),
        ],
        fill=_rgba(body),
        outline=_rgba(outline),
    )
    draw.rectangle((x0 + 5, y0 + 11, x0 + 12, y0 + 14), fill=_rgba(dark), outline=_rgba(outline))
    draw.rectangle((x0 + 5, y0 + 7, x0 + 6, y0 + 8), fill=_rgba((10, 36, 26)))
    draw.rectangle((x0 + 10, y0 + 7, x0 + 11, y0 + 8), fill=_rgba((10, 36, 26)))
    draw.point((x0 + 7, y0 + 4), fill=_rgba(light))
    draw.line((x0 + 5, y0 + 2, x0 + 6, y0 + 4), fill=_rgba(light))


def _draw_bat_monster(draw: ImageDraw.ImageDraw, x0: int, y0: int) -> None:
    outline = (34, 20, 58)
    wing = (102, 74, 178)
    wing_dark = (67, 48, 124)
    body = (58, 43, 94)
    draw.polygon(
        [(x0 + 1, y0 + 8), (x0 + 3, y0 + 3), (x0 + 6, y0 + 7), (x0 + 8, y0 + 11), (x0 + 4, y0 + 10)],
        fill=_rgba(wing),
        outline=_rgba(outline),
    )
    draw.polygon(
        [(x0 + 15, y0 + 8), (x0 + 13, y0 + 3), (x0 + 10, y0 + 7), (x0 + 8, y0 + 11), (x0 + 12, y0 + 10)],
        fill=_rgba(wing),
        outline=_rgba(outline),
    )
    draw.polygon([(x0 + 3, y0 + 8), (x0 + 5, y0 + 9), (x0 + 6, y0 + 11), (x0 + 4, y0 + 10)], fill=_rgba(wing_dark))
    draw.polygon([(x0 + 13, y0 + 8), (x0 + 11, y0 + 9), (x0 + 10, y0 + 11), (x0 + 12, y0 + 10)], fill=_rgba(wing_dark))
    draw.ellipse((x0 + 5, y0 + 5, x0 + 11, y0 + 13), fill=_rgba(body), outline=_rgba(outline))
    draw.polygon([(x0 + 5, y0 + 5), (x0 + 6, y0 + 2), (x0 + 7, y0 + 5)], fill=_rgba(body), outline=_rgba(outline))
    draw.polygon([(x0 + 9, y0 + 5), (x0 + 10, y0 + 2), (x0 + 11, y0 + 5)], fill=_rgba(body), outline=_rgba(outline))
    draw.point((x0 + 7, y0 + 8), fill=_rgba((238, 228, 119)))
    draw.point((x0 + 10, y0 + 8), fill=_rgba((238, 228, 119)))


def _draw_spider_monster(draw: ImageDraw.ImageDraw, x0: int, y0: int) -> None:
    outline = (33, 27, 32)
    leg = (214, 216, 204)
    body = (54, 55, 70)
    abdomen = (84, 71, 56)
    for dy in (6, 8, 10, 12):
        draw.line((x0 + 6, y0 + dy, x0 + 2, y0 + dy - 2), fill=_rgba(leg), width=1)
        draw.line((x0 + 10, y0 + dy, x0 + 14, y0 + dy - 2), fill=_rgba(leg), width=1)
    draw.ellipse((x0 + 3, y0 + 6, x0 + 12, y0 + 14), fill=_rgba(abdomen), outline=_rgba(outline))
    draw.ellipse((x0 + 5, y0 + 3, x0 + 11, y0 + 9), fill=_rgba(body), outline=_rgba(outline))
    draw.point((x0 + 7, y0 + 5), fill=_rgba((236, 73, 67)))
    draw.point((x0 + 9, y0 + 5), fill=_rgba((236, 73, 67)))


def _render_blockers(
    draw: ImageDraw.ImageDraw,
    *,
    blocker_specs: Sequence[_BlockerSpec],
    layout: RpgDungeonLayout,
    theme: Mapping[str, Any],
    theme_id: str,
) -> list[RpgDungeonBlocker]:
    """Draw impassable blockers and serialize their graph-blocking tiles."""

    blockers: list[RpgDungeonBlocker] = []
    for spec in blocker_specs:
        tile_xywh = (int(spec.tile_xy[0]), int(spec.tile_xy[1]), 1, 1)
        bbox = _tile_bbox(layout, tile_xywh)
        visual: dict[str, Any] = {
            "theme_id": theme_id,
            "renderer_style": RENDERER_STYLE_TOP_DOWN_PIXEL_RPG,
            "contrast": "high",
            "blocker_shape": "single_tile_boulder",
        }
        _draw_one_tile_boulder(draw, spec.tile_xy, floor_rgb=theme["floor_rgb"])
        point = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        blockers.append(
            RpgDungeonBlocker(
                blocker_id=spec.blocker_id,
                blocker_type=spec.blocker_type,
                tile_xy=spec.tile_xy,
                tile_xywh=tile_xywh,
                bbox_xyxy=bbox,
                point_xy=point,
                metadata={
                    "passable": False,
                    "edge_id": spec.edge_id,
                    "visual_attributes": visual,
                },
            )
        )
    return blockers


def _draw_one_tile_boulder(draw: ImageDraw.ImageDraw, tile: Tile, *, floor_rgb: Any) -> None:
    x, y = tile
    x0 = int(x) * CANONICAL_TILE_PX
    y0 = int(y) * CANONICAL_TILE_PX
    outline = (35, 39, 37)
    shadow = _shade(floor_rgb, -40)
    stone_light = (225, 229, 219)
    stone_mid = (180, 188, 178)
    stone_dark = (130, 140, 132)
    draw.ellipse((x0 + 2, y0 + 8, x0 + 14, y0 + 15), fill=_rgba(shadow, 160))
    draw.ellipse((x0 + 1, y0 + 6, x0 + 8, y0 + 13), fill=_rgba(stone_mid), outline=_rgba(outline))
    draw.ellipse((x0 + 6, y0 + 3, x0 + 14, y0 + 12), fill=_rgba(stone_light), outline=_rgba(outline))
    draw.ellipse((x0 + 5, y0 + 8, x0 + 13, y0 + 15), fill=_rgba(stone_dark), outline=_rgba(outline))
    draw.point((x0 + 8, y0 + 5), fill=_rgba((246, 248, 240)))
    draw.point((x0 + 3, y0 + 8), fill=_rgba((222, 228, 218)))


def _tile_bbox(layout: RpgDungeonLayout, tile_xywh: TileBox) -> BBox:
    x, y, w, h = tile_xywh
    ox, oy = layout.display_offset_xy
    return _clip_bbox(
        (
            float(ox + int(x) * int(layout.tile_px)),
            float(oy + int(y) * int(layout.tile_px)),
            float(ox + (int(x) + int(w)) * int(layout.tile_px)),
            float(oy + (int(y) + int(h)) * int(layout.tile_px)),
        ),
        width=int(layout.width_px),
        height=int(layout.height_px),
    )


def _clip_bbox(bbox: BBox, *, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    clipped = (
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError(f"RPG dungeon bbox clipped outside the canvas: {bbox}")
    return clipped


def _rect_tiles(tile_xywh: TileBox) -> tuple[Tile, ...]:
    x, y, w, h = tile_xywh
    return tuple(
        (int(xx), int(yy))
        for yy in range(int(y), int(y) + int(h))
        for xx in range(int(x), int(x) + int(w))
    )


def _rect_center_tile(tile_xywh: TileBox) -> Tile:
    x, y, w, h = tile_xywh
    return (int(x) + int(w) // 2, int(y) + int(h) // 2)


def _centered_box(container: TileBox, *, width_tiles: int, height_tiles: int) -> TileBox:
    x, y, w, h = container
    box_w = min(int(width_tiles), max(1, int(w) - 2))
    box_h = min(int(height_tiles), max(1, int(h) - 2))
    return (
        int(x) + max(1, (int(w) - box_w) // 2),
        int(y) + max(1, (int(h) - box_h) // 2),
        int(box_w),
        int(box_h),
    )


def _choose(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[int(rng.randrange(len(values)))]


def _rgba(color: Any, alpha: int = 255) -> tuple[int, int, int, int]:
    r, g, b = tuple(int(v) for v in color)
    return (r, g, b, int(alpha))


def _shade(color: Any, delta: int) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)  # type: ignore[return-value]


__all__ = [
    "DEFAULT_CANVAS_HEIGHT",
    "DEFAULT_CANVAS_WIDTH",
    "DEFAULT_TILE_PX",
    "DEFAULT_TOTAL_CHEST_COUNT",
    "MAX_REACHABLE_CHEST_COUNT",
    "MAX_MONSTER_CHAMBER_COUNT",
    "MAX_TOTAL_CHEST_COUNT",
    "MIN_REACHABLE_CHEST_COUNT",
    "MIN_MONSTER_CHAMBER_COUNT",
    "MIN_TOTAL_CHEST_COUNT",
    "MONSTER_OBJECT_TYPES",
    "MONSTER_PUBLIC_NAMES",
    "RENDERER_ID",
    "SCENE_ID",
    "THEMES",
    "draw_rpg_dungeon_debug_overlay",
    "render_rpg_dungeon_profile_scene",
    "render_rpg_dungeon_scene",
    "rpg_dungeon_profile_metadata",
]
