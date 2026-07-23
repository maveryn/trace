"""Reusable pixel-world territory drawing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.tasks.illustrations.shared.object_variants import RENDERER_STYLE_TOP_DOWN_PIXEL_RPG, variant_visual_metadata
from trace_tasks.tasks.illustrations.shared.pixel_world_objects import CANONICAL_TILE_PX


TileBox = tuple[int, int, int, int]
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class PixelOrchardTreeSpec:
    """One tree placement inside a reusable pixel orchard territory."""

    tree_id: str
    tile_xywh: TileBox
    row_index: int
    column_index: int
    leaf_rgb: RGB
    fruit_rgb: RGB

    def metadata(self, *, territory_id: str) -> dict[str, Any]:
        return {
            "variant": "tree",
            **variant_visual_metadata("tree", "fruit_tree", RENDERER_STYLE_TOP_DOWN_PIXEL_RPG),
            "tree_style": "fruit_tree",
            "leaf_rgb": list(self.leaf_rgb),
            "fruit_rgb": list(self.fruit_rgb),
            "territory_id": str(territory_id),
            "orchard_row_index": int(self.row_index),
            "orchard_column_index": int(self.column_index),
        }


@dataclass(frozen=True)
class PixelOrchardPlan:
    """Reusable semantic plan for a variable-size orchard territory."""

    territory_id: str
    tile_xywh: TileBox
    gate_tile: TileBox
    gate_side: str
    connector_tiles: tuple[tuple[int, int], ...]
    boundary_tiles: tuple[tuple[TileBox, str], ...]
    trees: tuple[PixelOrchardTreeSpec, ...]
    ground_style: str

    @property
    def tree_count(self) -> int:
        return len(self.trees)

    @property
    def row_count(self) -> int:
        return len({tree.row_index for tree in self.trees})

    @property
    def column_count(self) -> int:
        return len({tree.column_index for tree in self.trees})

    def metadata(self) -> dict[str, Any]:
        return {
            "gate_tile": list(self.gate_tile),
            "gate_side": self.gate_side,
            "connector_tiles": [list(tile) for tile in self.connector_tiles],
            "boundary_tile_count": len(self.boundary_tiles),
            "tree_count": self.tree_count,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "ground_style": self.ground_style,
            "fruit_rgb_values": sorted({tuple(tree.fruit_rgb) for tree in self.trees}),
        }


def _choose(rng: random.Random, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot choose from an empty sequence")
    return values[int(rng.randrange(len(values)))]


def _base_rect(tile_xywh: TileBox, *, inset: int = 0) -> tuple[int, int, int, int]:
    x, y, w, h = tile_xywh
    return (
        x * CANONICAL_TILE_PX + int(inset),
        y * CANONICAL_TILE_PX + int(inset),
        (x + w) * CANONICAL_TILE_PX - 1 - int(inset),
        (y + h) * CANONICAL_TILE_PX - 1 - int(inset),
    )


def _orchard_boundary(tile_xywh: TileBox, gate_tile: TileBox) -> tuple[tuple[TileBox, str], ...]:
    x, y, w, h = tile_xywh
    boundary: list[tuple[TileBox, str]] = []
    for tx in range(x, x + w):
        top = (tx, y, 1, 1)
        bottom = (tx, y + h - 1, 1, 1)
        if top != gate_tile:
            boundary.append((top, "horizontal"))
        if bottom != gate_tile:
            boundary.append((bottom, "horizontal"))
    for ty in range(y + 1, y + h - 1):
        boundary.append(((x, ty, 1, 1), "vertical"))
        boundary.append(((x + w - 1, ty, 1, 1), "vertical"))
    return tuple(boundary)


def _sample_orchard_trees(rng: random.Random, tile_xywh: TileBox) -> tuple[PixelOrchardTreeSpec, ...]:
    x, y, w, h = tile_xywh
    leaf_palette = [(42, 139, 76), (50, 153, 84), (63, 145, 72), (72, 158, 88)]
    fruit_palette = [(218, 62, 58), (235, 181, 54), (190, 65, 126), (229, 102, 58)]
    x_offset = rng.randrange(0, 2) if w >= 8 else 0
    y_offset = rng.randrange(0, 2) if h >= 7 else 0
    tree_positions: list[tuple[int, int, int, int]] = []
    row_index = 0
    for ty in range(y + 1 + y_offset, y + h - 2, 2):
        column_index = 0
        for tx in range(x + 1 + x_offset, x + w - 1, 2):
            tree_positions.append((tx, ty, row_index, column_index))
            column_index += 1
        row_index += 1
    # Keep smaller orchards legible while still making size matter.
    min_count = min(len(tree_positions), max(4, (w * h) // 12))
    max_count = min(len(tree_positions), max(min_count, (w * h) // 6))
    target_count = rng.randint(min_count, max_count) if tree_positions else 0
    rng.shuffle(tree_positions)
    selected = sorted(tree_positions[:target_count], key=lambda item: (item[2], item[3], item[1], item[0]))
    trees: list[PixelOrchardTreeSpec] = []
    for index, (tx, ty, row, col) in enumerate(selected):
        trees.append(
            PixelOrchardTreeSpec(
                tree_id=f"orchard_tree_{index:02d}",
                tile_xywh=(tx, ty, 1, 2),
                row_index=row,
                column_index=col,
                leaf_rgb=_choose(rng, leaf_palette),
                fruit_rgb=_choose(rng, fruit_palette),
            )
        )
    return tuple(trees)


def sample_pixel_orchard_plan(
    rng: random.Random,
    *,
    tile_xywh: TileBox,
    territory_id: str,
    gate_side: str,
    connector_tiles: Sequence[tuple[int, int]] = (),
) -> PixelOrchardPlan:
    """Sample a reusable variable-size orchard plan for a caller-owned map."""

    x, y, w, h = tile_xywh
    if w < 6 or h < 5:
        raise ValueError("pixel orchard territory must be at least 6x5 tiles")
    normalized_gate_side = str(gate_side)
    if normalized_gate_side not in {"top", "bottom", "left", "right"}:
        raise ValueError("orchard gate_side must be top, bottom, left, or right")
    if normalized_gate_side == "top":
        gate_tile = (x + w // 2, y, 1, 1)
    elif normalized_gate_side == "bottom":
        gate_tile = (x + w // 2, y + h - 1, 1, 1)
    elif normalized_gate_side == "left":
        gate_tile = (x, y + h // 2, 1, 1)
    else:
        gate_tile = (x + w - 1, y + h // 2, 1, 1)
    return PixelOrchardPlan(
        territory_id=str(territory_id),
        tile_xywh=tile_xywh,
        gate_tile=gate_tile,
        gate_side=normalized_gate_side,
        connector_tiles=tuple((int(tx), int(ty)) for tx, ty in connector_tiles),
        boundary_tiles=_orchard_boundary(tile_xywh, gate_tile),
        trees=_sample_orchard_trees(rng, tile_xywh),
        ground_style=str(_choose(rng, ["meadow_rows", "tilled_rows", "leaf_litter"])),
    )


def draw_pixel_orchard_ground(draw: ImageDraw.ImageDraw, plan: PixelOrchardPlan) -> None:
    """Draw orchard ground, row hints, and fallen-fruit decoration."""

    x, y, w, h = plan.tile_xywh
    for ty in range(y, y + h):
        for tx in range(x, x + w):
            px = tx * CANONICAL_TILE_PX
            py = ty * CANONICAL_TILE_PX
            if plan.ground_style == "tilled_rows":
                fill = (94, 145, 72) if ty % 2 else (105, 151, 74)
                row_line = (116, 91, 54)
            elif plan.ground_style == "leaf_litter":
                fill = (91, 139, 72) if (tx + ty) % 2 else (82, 132, 69)
                row_line = (159, 109, 55)
            else:
                fill = (85, 151, 77) if (tx + ty) % 2 else (92, 160, 82)
                row_line = (61, 125, 63)
            draw.rectangle((px, py, px + 15, py + 15), fill=fill)
            if y < ty < y + h - 1 and tx % 2 == 0:
                draw.line((px + 2, py + 13, px + 13, py + 13), fill=row_line)
            if (tx * 7 + ty * 5) % 11 == 0:
                fruit = (214, 67, 53) if (tx + ty) % 2 else (231, 177, 61)
                draw.point((px + 5, py + 11), fill=fruit)
                draw.point((px + 10, py + 6), fill=(52, 120, 58))


def draw_pixel_orchard_boundary(draw: ImageDraw.ImageDraw, plan: PixelOrchardPlan) -> None:
    """Draw a low hedge/post boundary and simple orchard gate."""

    hedge = (42, 116, 58)
    hedge_dark = (30, 82, 46)
    post = (115, 77, 43)
    gate = (143, 93, 51)
    for tile_box, orientation in plan.boundary_tiles:
        x0, y0, x1, y1 = _base_rect(tile_box)
        if orientation == "horizontal":
            draw.line((x0 + 1, y0 + 8, x1 - 1, y0 + 8), fill=hedge_dark, width=2)
            draw.line((x0 + 1, y0 + 11, x1 - 1, y0 + 11), fill=hedge, width=2)
            for x in range(x0 + 2, x1, 7):
                draw.rectangle((x, y0 + 7, x + 1, y0 + 13), fill=post)
        else:
            draw.line((x0 + 8, y0 + 1, x0 + 8, y1 - 1), fill=hedge_dark, width=2)
            draw.line((x0 + 11, y0 + 1, x0 + 11, y1 - 1), fill=hedge, width=2)
            for y in range(y0 + 2, y1, 7):
                draw.rectangle((x0 + 7, y, x0 + 13, y + 1), fill=post)

    x0, y0, x1, y1 = _base_rect(plan.gate_tile)
    if plan.gate_side in {"top", "bottom"}:
        draw.line((x0 + 3, y0 + 9, x1 - 3, y0 + 9), fill=gate, width=2)
        draw.line((x0 + 5, y0 + 12, x1 - 5, y0 + 12), fill=gate, width=2)
        draw.rectangle((x0 + 3, y0 + 6, x0 + 5, y0 + 14), fill=post)
        draw.rectangle((x1 - 5, y0 + 6, x1 - 3, y0 + 14), fill=post)
    else:
        draw.line((x0 + 9, y0 + 3, x0 + 9, y1 - 3), fill=gate, width=2)
        draw.line((x0 + 12, y0 + 5, x0 + 12, y1 - 5), fill=gate, width=2)
        draw.rectangle((x0 + 6, y0 + 3, x0 + 14, y0 + 5), fill=post)
        draw.rectangle((x0 + 6, y1 - 5, x0 + 14, y1 - 3), fill=post)


__all__ = [
    "PixelOrchardPlan",
    "PixelOrchardTreeSpec",
    "draw_pixel_orchard_boundary",
    "draw_pixel_orchard_ground",
    "sample_pixel_orchard_plan",
]
