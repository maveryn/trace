"""Scene-local top-down RPG house renderer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.illustrations.shared.object_rendering import (
    IllustrationObjectSpec,
    RenderContext,
    render_illustration_object,
)
from trace_tasks.tasks.illustrations.shared.object_variants import RENDERER_STYLE_TOP_DOWN_PIXEL_RPG
from trace_tasks.tasks.illustrations.shared.option_rendering import draw_label_badge
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import CANONICAL_TILE_PX
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import DEFAULT_RPG_TILE_PX

from .relations import room_graph
from .state import BBox, RpgHouseDoor, RpgHouseEntity, RpgHouseRoom, RpgHouseScene, TileBox


SCENE_ID = "rpg_house"
RENDERER_ID = "rpg_house_top_down_v0"
DEFAULT_TILE_PX = DEFAULT_RPG_TILE_PX
DEFAULT_CANVAS_WIDTH = 27 * DEFAULT_TILE_PX
DEFAULT_CANVAS_HEIGHT = 18 * DEFAULT_TILE_PX
CANONICAL_WALL_THICKNESS = 12
CANONICAL_WALL_SHADOW_THICKNESS = 16
CANONICAL_DOOR_CLEARANCE = 2
MIN_ROOM_COUNT = 4
MAX_ROOM_COUNT = 8
ROOM_IDS: tuple[str, ...] = (
    "bedroom",
    "kitchen",
    "storage",
    "study",
    "parlor",
    "dining_room",
    "workshop",
    "pantry",
)
HALL_ROOM_ID = ""
PASSABLE_DOOR_STATES: frozenset[str] = frozenset({"open"})

ROOM_PUBLIC_NAMES: Mapping[str, str] = {
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "storage": "storage room",
    "study": "study",
    "parlor": "parlor",
    "dining_room": "dining room",
    "workshop": "workshop",
    "pantry": "pantry",
}

RGB = tuple[int, int, int]

THEMES: Mapping[str, Mapping[str, Any]] = {
    "warm_cottage": {
        "background_rgb": (57, 47, 39),
        "wall_rgb": (94, 88, 78),
        "wall_dark_rgb": (45, 42, 36),
        "door_rgb": (145, 78, 39),
        "door_outline_rgb": (50, 27, 18),
        "door_panel_light_rgb": (184, 117, 58),
        "door_panel_dark_rgb": (82, 40, 25),
        "floor_rgbs": {
            "bedroom": (171, 128, 82),
            "kitchen": (154, 121, 84),
            "storage": (135, 106, 76),
            "study": (146, 112, 73),
            "parlor": (174, 134, 86),
            "dining_room": (164, 119, 73),
            "workshop": (137, 112, 83),
            "pantry": (151, 115, 78),
        },
        "wood_rgb": (132, 78, 42),
        "rug_rgb": (150, 67, 82),
    },
    "blue_inn": {
        "background_rgb": (45, 51, 61),
        "wall_rgb": (96, 105, 123),
        "wall_dark_rgb": (46, 52, 64),
        "door_rgb": (111, 73, 48),
        "door_outline_rgb": (42, 28, 22),
        "door_panel_light_rgb": (154, 104, 68),
        "door_panel_dark_rgb": (67, 42, 30),
        "floor_rgbs": {
            "bedroom": (146, 115, 78),
            "kitchen": (126, 120, 108),
            "storage": (119, 101, 78),
            "study": (130, 101, 73),
            "parlor": (154, 118, 76),
            "dining_room": (138, 108, 74),
            "workshop": (112, 111, 105),
            "pantry": (124, 104, 79),
        },
        "wood_rgb": (118, 75, 45),
        "rug_rgb": (74, 126, 153),
    },
    "stone_house": {
        "background_rgb": (49, 48, 45),
        "wall_rgb": (117, 111, 101),
        "wall_dark_rgb": (55, 54, 50),
        "door_rgb": (115, 75, 45),
        "door_outline_rgb": (45, 29, 20),
        "door_panel_light_rgb": (157, 106, 67),
        "door_panel_dark_rgb": (70, 43, 28),
        "floor_rgbs": {
            "bedroom": (133, 119, 95),
            "kitchen": (119, 117, 105),
            "storage": (107, 103, 93),
            "study": (124, 106, 82),
            "parlor": (138, 120, 92),
            "dining_room": (132, 113, 86),
            "workshop": (112, 108, 99),
            "pantry": (118, 105, 88),
        },
        "wood_rgb": (122, 75, 44),
        "rug_rgb": (126, 76, 112),
    },
}


@dataclass(frozen=True)
class RpgHouseLayout:
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
class _RoomSpec:
    room_id: str
    public_name: str
    tile_xywh: TileBox
    floor_rgb: RGB


@dataclass(frozen=True)
class _DoorSpec:
    door_id: str
    room_a_id: str
    room_b_id: str
    state: str
    orientation: str
    tile_xy: tuple[int, int]
    span_tiles: int


@dataclass(frozen=True)
class _EntitySpec:
    entity_id: str
    object_type: str
    public_name: str
    room_id: str
    tile_xywh: TileBox
    layer: str
    visual: Mapping[str, Any] | None = None


def render_rpg_house_scene(
    seed: int,
    *,
    width: int = DEFAULT_CANVAS_WIDTH,
    height: int = DEFAULT_CANVAS_HEIGHT,
    tile_px: int = DEFAULT_TILE_PX,
    room_count: int | None = None,
    start_room_id: str | None = None,
    player_room_id: str | None = None,
    room_labels: Mapping[str, str] | None = None,
    door_states: Mapping[str, str] | None = None,
    sample_mixed_door_states: bool = False,
    label_font_family: str | None = None,
    label_font_trace: Mapping[str, Any] | None = None,
    render_metadata: Mapping[str, Any] | None = None,
) -> RpgHouseScene:
    """Render one top-down RPG house layout."""

    rng = random.Random(int(seed))
    layout = _sample_layout(width=int(width), height=int(height), tile_px=int(tile_px))
    theme_id = str(_choose(rng, tuple(THEMES)))
    theme = THEMES[theme_id]
    resolved_room_count = _resolve_room_count(rng, room_count)
    room_specs, door_specs, layout_pattern_id = _make_layout_specs(
        layout,
        rng=rng,
        theme=theme,
        room_count=resolved_room_count,
        door_states=door_states or {},
        sample_mixed_door_states=bool(sample_mixed_door_states),
    )
    entity_specs = _make_entity_specs(room_specs, theme_id=theme_id, rng=rng)
    if player_room_id is not None:
        entity_specs.append(_make_player_entity_spec(room_specs, player_room_id=str(player_room_id)))
    canvas, draw = _render_base(layout, room_specs=room_specs, door_specs=door_specs, theme=theme)
    entities = _render_entities(draw, entity_specs=entity_specs, layout=layout, theme=theme, theme_id=theme_id)

    scaled = canvas.resize((layout.display_grid_width_px, layout.display_grid_height_px), Image.Resampling.NEAREST)
    final = Image.new("RGB", (int(width), int(height)), _shade(theme["background_rgb"], -8))
    final.paste(scaled.convert("RGB"), layout.display_offset_xy)
    final_draw = ImageDraw.Draw(final)
    room_labels = {str(key): str(value) for key, value in dict(room_labels or {}).items()}
    rooms = tuple(
        RpgHouseRoom(
            room_id=spec.room_id,
            public_name=spec.public_name,
            label=room_labels.get(spec.room_id),
            tile_xywh=spec.tile_xywh,
            bbox_xyxy=_tile_bbox(layout, spec.tile_xywh),
            metadata={"floor_rgb": list(spec.floor_rgb)},
        )
        for spec in room_specs
    )
    doors = tuple(_make_door(spec, layout) for spec in door_specs)
    _draw_final_room_marks(
        final_draw,
        rooms=rooms,
        start_room_id=start_room_id,
        label_font_family=label_font_family,
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
        "room_total": len(rooms),
        "layout_pattern_id": str(layout_pattern_id),
        "room_ids": [room.room_id for room in rooms],
        "candidate_room_labels": dict(room_labels),
        "start_room_id": None if start_room_id is None else str(start_room_id),
        "player_room_id": None if player_room_id is None else str(player_room_id),
        "doors": [door.as_dict() for door in doors],
        "door_state_policy": "mixed" if sample_mixed_door_states else "explicit_or_closed",
        "room_graph": room_graph(doors),
        "label_font": dict(label_font_trace or {}),
        **dict(render_metadata or {}),
    }
    return RpgHouseScene(
        image=final,
        rooms=rooms,
        doors=doors,
        entities=tuple(sorted(entities, key=lambda entity: (entity.layer, entity.room_id, entity.entity_id))),
        trace=trace,
    )


def resolve_rpg_house_tile_px(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> int:
    """Resolve the scene tile size from task params or scene render defaults."""

    return int(params.get("tile_px", group_default(render_defaults, "rpg_house_tile_px", DEFAULT_TILE_PX)))


def rpg_house_profile_metadata(render_params: Mapping[str, Any]) -> dict[str, Any]:
    """Return render trace metadata for the selected shared canvas profile."""

    return {
        "canvas_profile": str(render_params.get("canvas_profile", "")),
        "canvas_profile_size": list(render_params.get("canvas_profile_size", [])),
        "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
        "rpg_tile_profile": dict(render_params.get("rpg_tile_profile", {})),
    }


def render_rpg_house_profile_scene(
    seed: int,
    *,
    render_params: Mapping[str, Any],
    tile_px: int,
    room_count: int,
    render_metadata: Mapping[str, Any] | None = None,
    **scene_kwargs: Any,
) -> RpgHouseScene:
    """Render an RPG house using resolved canvas-profile parameters."""

    metadata = rpg_house_profile_metadata(render_params)
    if render_metadata:
        metadata.update({str(key): value for key, value in render_metadata.items()})
    return render_rpg_house_scene(
        int(seed),
        width=int(render_params["canvas_width"]),
        height=int(render_params["canvas_height"]),
        tile_px=int(tile_px),
        room_count=int(room_count),
        render_metadata=metadata,
        **scene_kwargs,
    )


def draw_rpg_house_debug_overlay(scene: RpgHouseScene) -> Image.Image:
    """Return an overlay image with room, door, and fixture bboxes."""

    image = scene.image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for room in scene.rooms:
        draw.rectangle(room.bbox_xyxy, outline=(42, 111, 210, 220), width=3)
    for door in scene.doors:
        color = (38, 152, 91, 235) if door.state == "open" else (214, 67, 59, 235)
        draw.rectangle(door.bbox_xyxy, outline=color, width=3)
    for entity in scene.entities:
        draw.rectangle(entity.bbox_xyxy, outline=(239, 173, 42, 220), width=2)
    return Image.alpha_composite(image, overlay).convert("RGB")


def _sample_layout(*, width: int, height: int, tile_px: int) -> RpgHouseLayout:
    resolved_tile_px = max(40, min(56, int(tile_px)))
    cols = max(14, int(math.ceil(float(width) / float(resolved_tile_px))))
    rows = max(12, int(math.ceil(float(height) / float(resolved_tile_px))))
    grid_w = int(cols) * resolved_tile_px
    grid_h = int(rows) * resolved_tile_px
    return RpgHouseLayout(
        cols=int(cols),
        rows=int(rows),
        tile_px=int(resolved_tile_px),
        width_px=int(width),
        height_px=int(height),
        display_offset_xy=((int(width) - grid_w) // 2, (int(height) - grid_h) // 2),
    )


def _resolve_room_count(rng: random.Random, room_count: int | None) -> int:
    """Resolve the number of enclosed rooms to render."""

    if room_count is None:
        return int(rng.randint(MIN_ROOM_COUNT, MAX_ROOM_COUNT))
    value = int(room_count)
    if not MIN_ROOM_COUNT <= value <= MAX_ROOM_COUNT:
        raise ValueError(f"room_count must be in [{MIN_ROOM_COUNT}, {MAX_ROOM_COUNT}], got {value}")
    return value


def _make_layout_specs(
    layout: RpgHouseLayout,
    *,
    rng: random.Random,
    theme: Mapping[str, Any],
    room_count: int,
    door_states: Mapping[str, str],
    sample_mixed_door_states: bool,
) -> tuple[list[_RoomSpec], list[_DoorSpec], str]:
    """Build an irregular room partition and its connecting doors."""

    room_boxes = _split_house_rectangles(rng, (0, 0, int(layout.cols), int(layout.rows)), int(room_count))
    room_ids = list(ROOM_IDS)
    rng.shuffle(room_ids)
    selected_ids = tuple(room_ids[: int(room_count)])
    floors = theme["floor_rgbs"]
    sorted_boxes = sorted(room_boxes, key=lambda box: (box[1], box[0], box[3], box[2]))
    rooms = [
        _RoomSpec(
            room_id=room_id,
            public_name=ROOM_PUBLIC_NAMES[room_id],
            tile_xywh=box,
            floor_rgb=floors.get(room_id, (146, 112, 73)),
        )
        for room_id, box in zip(selected_ids, sorted_boxes)
    ]
    base_doors = _adjacent_door_specs(rng, rooms)
    sampled_open_door_ids = _sample_open_door_ids(
        rng,
        tuple(str(door_id) for door_id, *_rest in base_doors),
    ) if sample_mixed_door_states else frozenset()
    doors = [
        _DoorSpec(
            door_id=door_id,
            room_a_id=room_a,
            room_b_id=room_b,
            state=str(door_states.get(door_id, "open" if door_id in sampled_open_door_ids else "closed")),
            orientation=orientation,
            tile_xy=tile_xy,
            span_tiles=span_tiles,
        )
        for door_id, room_a, room_b, orientation, tile_xy, span_tiles in base_doors
    ]
    return rooms, doors, "bsp_partition"


def _split_house_rectangles(rng: random.Random, root: TileBox, room_count: int) -> list[TileBox]:
    """Split one house footprint into connected rectangular rooms."""

    min_w = 4
    min_h = 3
    boxes: list[TileBox] = [root]
    while len(boxes) < int(room_count):
        candidates: list[tuple[int, int, tuple[str, ...]]] = []
        for index, box in enumerate(boxes):
            _x, _y, w, h = box
            orientations: list[str] = []
            if int(w) >= min_w * 2:
                orientations.append("vertical")
            if int(h) >= min_h * 2:
                orientations.append("horizontal")
            if orientations:
                candidates.append((int(w) * int(h), index, tuple(orientations)))
        if not candidates:
            raise ValueError(f"could not split RPG house into {room_count} rooms")
        _area, index, orientations = sorted(candidates, reverse=True)[0]
        x, y, w, h = boxes.pop(index)
        if len(orientations) == 1:
            orientation = orientations[0]
        elif int(w) > int(h) + 2:
            orientation = "vertical"
        elif int(h) > int(w) + 2:
            orientation = "horizontal"
        else:
            orientation = str(_choose(rng, orientations))
        if orientation == "vertical":
            cut = int(rng.randint(min_w, int(w) - min_w))
            boxes.extend([(x, y, cut, h), (x + cut, y, int(w) - cut, h)])
        else:
            cut = int(rng.randint(min_h, int(h) - min_h))
            boxes.extend([(x, y, w, cut), (x, y + cut, w, int(h) - cut)])
    return boxes


def _adjacent_door_specs(
    rng: random.Random,
    rooms: Sequence[_RoomSpec],
) -> list[tuple[str, str, str, str, tuple[int, int], int]]:
    """Place doors away from shared-wall endpoints and T-junctions."""

    doors: list[tuple[str, str, str, str, tuple[int, int], int]] = []
    for index, room_a in enumerate(rooms):
        ax, ay, aw, ah = room_a.tile_xywh
        for room_b in rooms[index + 1 :]:
            bx, by, bw, bh = room_b.tile_xywh
            if ax + aw == bx or bx + bw == ax:
                boundary_x = ax + aw if ax + aw == bx else bx + bw
                overlap0 = max(ay, by)
                overlap1 = min(ay + ah, by + bh)
                span_tiles = _door_span_for_overlap(overlap1 - overlap0, rng=rng)
                if span_tiles:
                    door_y = _sample_door_start(rng, overlap0=overlap0, overlap1=overlap1, span_tiles=span_tiles)
                    room_0, room_1 = sorted((room_a.room_id, room_b.room_id))
                    doors.append((f"{room_0}_{room_1}", room_a.room_id, room_b.room_id, "vertical", (boundary_x, door_y), span_tiles))
            if ay + ah == by or by + bh == ay:
                boundary_y = ay + ah if ay + ah == by else by + bh
                overlap0 = max(ax, bx)
                overlap1 = min(ax + aw, bx + bw)
                span_tiles = _door_span_for_overlap(overlap1 - overlap0, rng=rng)
                if span_tiles:
                    door_x = _sample_door_start(rng, overlap0=overlap0, overlap1=overlap1, span_tiles=span_tiles)
                    room_0, room_1 = sorted((room_a.room_id, room_b.room_id))
                    doors.append((f"{room_0}_{room_1}", room_a.room_id, room_b.room_id, "horizontal", (door_x, boundary_y), span_tiles))
    return sorted(doors, key=lambda item: item[0])


def _door_span_for_overlap(overlap_len: int, *, rng: random.Random) -> int:
    """Return a door span that leaves wall material on both ends."""

    if int(overlap_len) < 4:
        return 0
    if int(overlap_len) <= 6:
        return 2
    return int(rng.choice((2, 3)))


def _sample_door_start(
    rng: random.Random,
    *,
    overlap0: int,
    overlap1: int,
    span_tiles: int,
) -> int:
    start_min = int(overlap0) + 1
    start_max = int(overlap1) - int(span_tiles) - 1
    if start_max < start_min:
        return start_min
    return int(rng.randint(start_min, start_max))


def _sample_open_door_ids(rng: random.Random, door_ids: Sequence[str]) -> frozenset[str]:
    """Sample visible open doors for non-reachability scenes."""

    unique_ids = tuple(str(door_id) for door_id in door_ids)
    if not unique_ids:
        return frozenset()
    if len(unique_ids) == 1:
        return frozenset(unique_ids)
    open_count = max(1, min(len(unique_ids) - 1, int(round(len(unique_ids) * 0.35))))
    return frozenset(rng.sample(list(unique_ids), open_count))


def _make_entity_specs(
    room_specs: Sequence[_RoomSpec],
    *,
    theme_id: str,
    rng: random.Random,
) -> list[_EntitySpec]:
    """Place large context fixtures while preserving room-count independence."""

    wood = THEMES[theme_id]["wood_rgb"]
    rug = THEMES[theme_id]["rug_rgb"]
    entities: list[_EntitySpec] = []

    def add(entity_id: str, object_type: str, public_name: str, room_id: str, box: TileBox, **visual: Any) -> None:
        entities.append(_EntitySpec(entity_id, object_type, public_name, room_id, box, "fixture", visual))

    for index, room in enumerate(room_specs):
        x, y, w, h = room.tile_xywh
        inner_x = int(x) + 1
        inner_y = int(y) + 1
        inner_w = max(1, int(w) - 2)
        inner_h = max(1, int(h) - 2)
        room_id = str(room.room_id)
        suffix = f"{index:02d}"
        if room_id == "bedroom":
            if inner_h >= 2:
                add(f"bed_{suffix}", "bed", "bed", room_id, (inner_x, inner_y, inner_w, 2), bed_size="single", wood_rgb=wood, blanket_rgb=(96, 132, 174))
            else:
                add(f"rug_{suffix}", "rug", "rug", room_id, (inner_x, inner_y, inner_w, inner_h), cloth_rgb=rug)
        elif room_id == "kitchen":
            add(f"counter_{suffix}", "counter", "counter", room_id, (inner_x, inner_y, inner_w, 1), wood_rgb=wood, top_rgb=_shade(wood, 46))
            if inner_h >= 3:
                add(f"fireplace_{suffix}", "fireplace", "hearth", room_id, (inner_x, inner_y + inner_h - 1, inner_w, 1), fire_state="lit")
        elif room_id in {"storage", "pantry"}:
            add(f"shelf_{suffix}", "shelf", "shelf", room_id, (inner_x, inner_y, inner_w, 1), wood_rgb=wood, goods_type="mixed")
            add(f"chest_{suffix}", "chest", "chest", room_id, (inner_x, inner_y + inner_h - 1, max(1, min(3, inner_w)), 1), wood_rgb=wood)
        elif room_id == "study":
            add(f"shelf_{suffix}", "shelf", "shelf", room_id, (inner_x, inner_y, inner_w, 1), wood_rgb=wood, goods_type="books")
            if inner_h >= 3:
                add(f"table_{suffix}", "table", "table", room_id, (inner_x, inner_y + inner_h - 2, inner_w, 2), table_shape="long", wood_rgb=wood)
        elif room_id == "workshop":
            add(f"counter_{suffix}", "counter", "workbench", room_id, (inner_x, inner_y, inner_w, 1), wood_rgb=_shade(wood, -10), top_rgb=_shade(wood, 32))
            if inner_w >= 3:
                add(f"chest_{suffix}", "chest", "tool chest", room_id, (inner_x, inner_y + inner_h - 1, min(3, inner_w), 1), wood_rgb=wood)
        else:
            add(f"rug_{suffix}", "rug", "rug", room_id, (inner_x, inner_y, inner_w, inner_h), cloth_rgb=rug)
            if inner_h >= 3:
                table_y = inner_y + int(rng.randrange(max(1, inner_h - 2)))
                add(f"table_{suffix}", "table", "table", room_id, (inner_x, table_y, inner_w, 2), table_shape="long", wood_rgb=wood)
    return entities


def _make_player_entity_spec(room_specs: Sequence[_RoomSpec], *, player_room_id: str) -> _EntitySpec:
    """Return the visible player marker for reachability count tasks."""

    rooms_by_id = {str(room.room_id): room for room in room_specs}
    if str(player_room_id) not in rooms_by_id:
        raise ValueError(f"player_room_id must name a rendered room, got {player_room_id!r}")
    room = rooms_by_id[str(player_room_id)]
    x, y, w, h = room.tile_xywh
    inner_x = int(x) + 1
    inner_y = int(y) + 1
    inner_w = max(1, int(w) - 2)
    inner_h = max(1, int(h) - 2)
    player_x = inner_x + max(0, (inner_w - 1) // 2)
    player_y = inner_y + max(0, (inner_h - 1) // 2)
    return _EntitySpec(
        entity_id="player_00",
        object_type="person",
        public_name="player",
        room_id=str(player_room_id),
        tile_xywh=(int(player_x), int(player_y), 1, 1),
        layer="marker",
        visual={
            "role": "player_marker",
            "person_variant_id": "adult",
            "gender_id": "female",
            "primary_color_rgb": (52, 94, 196),
            "accent_color_rgb": (247, 205, 72),
        },
    )


def _render_base(
    layout: RpgHouseLayout,
    *,
    room_specs: Sequence[_RoomSpec],
    door_specs: Sequence[_DoorSpec],
    theme: Mapping[str, Any],
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (layout.canonical_width_px, layout.canonical_height_px), _rgba(theme["background_rgb"]))
    draw = ImageDraw.Draw(image)
    for room in room_specs:
        _draw_room_floor(draw, room, theme=theme)
        _draw_floor_lines(draw, room.tile_xywh, room.floor_rgb)
    _draw_room_walls(draw, room_specs=room_specs, theme=theme)
    for door in door_specs:
        _draw_door(draw, door, theme=theme)
    return image, draw


def _draw_room_floor(draw: ImageDraw.ImageDraw, room: _RoomSpec, *, theme: Mapping[str, Any]) -> None:
    x0, y0, x1, y1 = _canonical_rect(room.tile_xywh)
    inset = CANONICAL_WALL_THICKNESS // 2 + 1
    draw.rectangle((x0 + inset, y0 + inset, x1 - inset, y1 - inset), fill=_rgba(room.floor_rgb))
    draw.rectangle(
        (x0 + inset, y0 + inset, x1 - inset, y1 - inset),
        outline=_rgba(_shade(theme["wall_rgb"], 30), 120),
        width=1,
    )


def _draw_floor_lines(draw: ImageDraw.ImageDraw, tile_xywh: TileBox, floor_rgb: RGB) -> None:
    x, y, w, h = tile_xywh
    x0, y0, x1, y1 = _canonical_rect(tile_xywh)
    inset = CANONICAL_WALL_THICKNESS // 2 + 4
    for yy in range(int(y), int(y) + int(h)):
        py = yy * CANONICAL_TILE_PX + CANONICAL_TILE_PX // 2
        if py <= y0 + inset or py >= y1 - inset:
            continue
        draw.line(
            (int(x0) + inset, py, int(x1) - inset, py),
            fill=_rgba(_shade(floor_rgb, -20), 105),
        )


def _draw_room_walls(
    draw: ImageDraw.ImageDraw,
    *,
    room_specs: Sequence[_RoomSpec],
    theme: Mapping[str, Any],
) -> None:
    for room in room_specs:
        x, y, w, h = room.tile_xywh
        left = int(x) * CANONICAL_TILE_PX
        top = int(y) * CANONICAL_TILE_PX
        right = (int(x) + int(w)) * CANONICAL_TILE_PX
        bottom = (int(y) + int(h)) * CANONICAL_TILE_PX
        _draw_wall_band(draw, "vertical", left, top, bottom, theme=theme)
        _draw_wall_band(draw, "vertical", right, top, bottom, theme=theme)
        _draw_wall_band(draw, "horizontal", top, left, right, theme=theme)
        _draw_wall_band(draw, "horizontal", bottom, left, right, theme=theme)


def _draw_wall_band(
    draw: ImageDraw.ImageDraw,
    orientation: str,
    center: int,
    start: int,
    end: int,
    *,
    theme: Mapping[str, Any],
) -> None:
    half = CANONICAL_WALL_THICKNESS // 2
    shadow_half = CANONICAL_WALL_SHADOW_THICKNESS // 2
    wall = theme["wall_rgb"]
    wall_dark = theme["wall_dark_rgb"]
    wall_light = _shade(wall, 34)
    if orientation == "vertical":
        draw.rectangle((center - shadow_half, start - shadow_half, center + shadow_half, end + shadow_half), fill=_rgba(wall_dark))
        draw.rectangle((center - half, start - half, center + half, end + half), fill=_rgba(wall))
        draw.line((center - half + 1, start - half + 2, center - half + 1, end + half - 2), fill=_rgba(wall_light, 150))
        return
    if orientation == "horizontal":
        draw.rectangle((start - shadow_half, center - shadow_half, end + shadow_half, center + shadow_half), fill=_rgba(wall_dark))
        draw.rectangle((start - half, center - half, end + half, center + half), fill=_rgba(wall))
        draw.line((start - half + 2, center - half + 1, end + half - 2, center - half + 1), fill=_rgba(wall_light, 150))
        return
    raise ValueError(f"unsupported RPG house wall orientation: {orientation}")


def _draw_door(draw: ImageDraw.ImageDraw, door: _DoorSpec, *, theme: Mapping[str, Any]) -> None:
    """Render an open wall gap or a closed slab with palette-independent door contrast."""
    x, y = door.tile_xy
    px = int(x) * CANONICAL_TILE_PX
    py = int(y) * CANONICAL_TILE_PX
    span_px = max(CANONICAL_TILE_PX, int(door.span_tiles) * CANONICAL_TILE_PX)
    shadow_half = CANONICAL_WALL_SHADOW_THICKNESS // 2
    slab_half = CANONICAL_WALL_THICKNESS // 2
    margin = CANONICAL_DOOR_CLEARANCE
    threshold_rgb = _shade(theme["wall_rgb"], 42)
    threshold_dark = _shade(theme["wall_dark_rgb"], 12)
    if door.orientation == "vertical":
        gap = (px - shadow_half, py + margin, px + shadow_half, py + span_px - margin - 1)
        closed = (px - slab_half, py + margin + 3, px + slab_half, py + span_px - margin - 4)
        knob = (px + slab_half - 2, py + span_px // 2)
    elif door.orientation == "horizontal":
        gap = (px + margin, py - shadow_half, px + span_px - margin - 1, py + shadow_half)
        closed = (px + margin + 3, py - slab_half, px + span_px - margin - 4, py + slab_half)
        knob = (px + span_px // 2, py + slab_half - 2)
    else:
        raise ValueError(f"unsupported RPG house door orientation: {door.orientation}")
    draw.rectangle(gap, fill=_rgba(threshold_rgb))
    draw.rectangle(gap, outline=_rgba(threshold_dark), width=1)
    if door.state == "open":
        if door.orientation == "vertical":
            draw.line((px, gap[1] + 2, px, gap[3] - 2), fill=_rgba(_shade(threshold_rgb, 24), 160))
        else:
            draw.line((gap[0] + 2, py, gap[2] - 2, py), fill=_rgba(_shade(threshold_rgb, 24), 160))
        return
    door_rgb = theme["door_rgb"]
    door_outline_rgb = theme.get("door_outline_rgb", _shade(door_rgb, -58))
    door_panel_light_rgb = theme.get("door_panel_light_rgb", _shade(door_rgb, 32))
    door_panel_dark_rgb = theme.get("door_panel_dark_rgb", _shade(door_rgb, -36))
    draw.rectangle(closed, fill=_rgba(door_rgb), outline=_rgba(door_outline_rgb))
    if door.orientation == "vertical":
        panel_x = px - slab_half + 3
        draw.line((panel_x, closed[1] + 3, panel_x, closed[3] - 3), fill=_rgba(door_panel_light_rgb, 170))
        draw.line((px + slab_half - 3, closed[1] + 3, px + slab_half - 3, closed[3] - 3), fill=_rgba(door_panel_dark_rgb, 170))
    else:
        panel_y = py - slab_half + 3
        draw.line((closed[0] + 3, panel_y, closed[2] - 3, panel_y), fill=_rgba(door_panel_light_rgb, 170))
        draw.line((closed[0] + 3, py + slab_half - 3, closed[2] - 3, py + slab_half - 3), fill=_rgba(door_panel_dark_rgb, 170))
    knob_x, knob_y = knob
    draw.rectangle((knob_x - 1, knob_y - 1, knob_x + 1, knob_y + 1), fill=_rgba((230, 188, 82)))


def _render_entities(
    draw: ImageDraw.ImageDraw,
    *,
    entity_specs: Sequence[_EntitySpec],
    layout: RpgHouseLayout,
    theme: Mapping[str, Any],
    theme_id: str,
) -> list[RpgHouseEntity]:
    """Draw large context fixtures and serialize their final bboxes."""

    entities: list[RpgHouseEntity] = []
    for spec in entity_specs:
        bbox = _tile_bbox(layout, spec.tile_xywh)
        visual = {"theme_id": theme_id, "renderer_style": RENDERER_STYLE_TOP_DOWN_PIXEL_RPG}
        visual.update(dict(spec.visual or {}))
        entity_role = "reference" if str(spec.layer) == "marker" else "context"
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
                    "room_id": spec.room_id,
                    "layout_context": str(spec.layer) != "marker",
                    "role": str(visual.get("role", entity_role)),
                },
                visual_attributes=visual,
                role=entity_role,
                source_entity_type="rpg_house_entity",
            ),
            RenderContext(renderer_style=RENDERER_STYLE_TOP_DOWN_PIXEL_RPG, draw=draw),
        )
        point = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        entities.append(
            RpgHouseEntity(
                entity_id=spec.entity_id,
                public_name=spec.public_name,
                object_type=spec.object_type,
                room_id=spec.room_id,
                tile_xywh=spec.tile_xywh,
                bbox_xyxy=bbox,
                point_xy=point,
                layer=spec.layer,
                metadata={"role": entity_role, "visual_attributes": visual, "object_record": rendered.object_record},
            )
        )
    return entities


def _draw_final_room_marks(
    draw: ImageDraw.ImageDraw,
    *,
    rooms: Sequence[RpgHouseRoom],
    start_room_id: str | None,
    label_font_family: str | None,
) -> None:
    for room in rooms:
        if str(room.room_id) == str(start_room_id):
            x0, y0, x1, y1 = [float(value) for value in room.bbox_xyxy]
            inset = 5
            draw.rectangle((x0 + inset, y0 + inset, x1 - inset, y1 - inset), outline=(218, 45, 48), width=6)
    for room in rooms:
        if not room.label:
            continue
        x0, y0, _x1, _y1 = [float(value) for value in room.bbox_xyxy]
        draw_label_badge(
            draw,
            str(room.label),
            (x0 + 12, y0 + 12, x0 + 48, y0 + 48),
            font_family=label_font_family,
            fill=(255, 255, 245),
            outline=(32, 40, 54),
            text_fill=(24, 31, 43),
        )


def _make_door(spec: _DoorSpec, layout: RpgHouseLayout) -> RpgHouseDoor:
    return RpgHouseDoor(
        door_id=spec.door_id,
        room_a_id=spec.room_a_id,
        room_b_id=spec.room_b_id,
        state=spec.state,
        orientation=spec.orientation,
        tile_xy=spec.tile_xy,
        bbox_xyxy=_door_bbox(layout, spec),
        metadata={"passable": spec.state in PASSABLE_DOOR_STATES, "span_tiles": int(spec.span_tiles)},
    )


def _tile_bbox(layout: RpgHouseLayout, tile_xywh: TileBox) -> BBox:
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


def _door_bbox(layout: RpgHouseLayout, spec: _DoorSpec) -> BBox:
    x, y = spec.tile_xy
    ox, oy = layout.display_offset_xy
    px = ox + int(x) * int(layout.tile_px)
    py = oy + int(y) * int(layout.tile_px)
    span_px = max(int(layout.tile_px), int(spec.span_tiles) * int(layout.tile_px))
    cross_half = max(6, int(round(layout.tile_px * CANONICAL_WALL_SHADOW_THICKNESS / (2 * CANONICAL_TILE_PX))))
    margin = max(2, int(round(layout.tile_px * CANONICAL_DOOR_CLEARANCE / CANONICAL_TILE_PX)))
    if spec.orientation == "vertical":
        return _clip_bbox(
            (
                float(px - cross_half),
                float(py + margin),
                float(px + cross_half),
                float(py + span_px - margin),
            ),
            width=int(layout.width_px),
            height=int(layout.height_px),
        )
    if spec.orientation == "horizontal":
        return _clip_bbox(
            (
                float(px + margin),
                float(py - cross_half),
                float(px + span_px - margin),
                float(py + cross_half),
            ),
            width=int(layout.width_px),
            height=int(layout.height_px),
        )
    raise ValueError(f"unsupported RPG house door orientation: {spec.orientation}")


def _clip_bbox(bbox: BBox, *, width: int, height: int) -> BBox:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    clipped = (
        max(0.0, min(float(width), x0)),
        max(0.0, min(float(height), y0)),
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        raise ValueError(f"RPG house bbox clipped outside the canvas: {bbox}")
    return clipped


def _canonical_rect(tile_xywh: TileBox) -> tuple[int, int, int, int]:
    x, y, w, h = tile_xywh
    return (
        int(x) * CANONICAL_TILE_PX,
        int(y) * CANONICAL_TILE_PX,
        (int(x) + int(w)) * CANONICAL_TILE_PX - 1,
        (int(y) + int(h)) * CANONICAL_TILE_PX - 1,
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
    "HALL_ROOM_ID",
    "MAX_ROOM_COUNT",
    "MIN_ROOM_COUNT",
    "PASSABLE_DOOR_STATES",
    "RENDERER_ID",
    "ROOM_IDS",
    "ROOM_PUBLIC_NAMES",
    "SCENE_ID",
    "THEMES",
    "draw_rpg_house_debug_overlay",
    "render_rpg_house_profile_scene",
    "render_rpg_house_scene",
    "resolve_rpg_house_tile_px",
    "rpg_house_profile_metadata",
]
