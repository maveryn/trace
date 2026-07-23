"""Reusable procedural pixel-world object drawing templates."""

from __future__ import annotations

from typing import Literal

from PIL import ImageDraw

from trace_tasks.tasks.illustrations.shared.object_variants import (
    PERSON_VARIANT_IDS,
    TREE_VARIANT_IDS,
    normalize_object_variant_id,
)


TileBox = tuple[int, int, int, int]
RGB = tuple[int, int, int]
PixelTreeStyle = Literal["oak", "pine", "maple", "fruit_tree"]
PixelPersonVariant = Literal["adult", "farmer", "worker", "vendor", "soldier"]
PixelDomesticAnimal = Literal["chicken", "pig", "sheep", "cow"]
PixelGraveMarkerStyle = Literal["rounded", "tablet", "cross", "obelisk"]
PixelCropStyle = Literal["wheat", "leafy", "flowering"]
PixelVegetableStyle = Literal["carrot", "cabbage", "corn", "tomato", "pumpkin"]
PixelShelfGoods = Literal["jars", "produce", "books", "mixed"]
PixelProduceGoods = Literal["fruit", "vegetable", "grain"]
PixelTableShape = Literal["square", "round", "long"]
PixelBedSize = Literal["single", "double"]
PixelFireState = Literal["lit", "unlit"]
PixelDividerStyle = Literal["screen", "curtain"]

CANONICAL_TILE_PX = 16
PIXEL_TREE_STYLES: tuple[PixelTreeStyle, ...] = tuple(TREE_VARIANT_IDS)  # type: ignore[assignment]
PIXEL_PERSON_VARIANTS: tuple[PixelPersonVariant, ...] = tuple(PERSON_VARIANT_IDS)  # type: ignore[assignment]
PIXEL_DOMESTIC_ANIMALS: tuple[PixelDomesticAnimal, ...] = ("chicken", "pig", "sheep", "cow")
PIXEL_GRAVE_MARKER_STYLES: tuple[PixelGraveMarkerStyle, ...] = ("rounded", "tablet", "cross", "obelisk")
PIXEL_CROP_STYLES: tuple[PixelCropStyle, ...] = ("wheat", "leafy", "flowering")
PIXEL_VEGETABLE_STYLES: tuple[PixelVegetableStyle, ...] = ("carrot", "cabbage", "corn", "tomato", "pumpkin")
PIXEL_SHELF_GOODS: tuple[PixelShelfGoods, ...] = ("jars", "produce", "books", "mixed")
PIXEL_PRODUCE_GOODS: tuple[PixelProduceGoods, ...] = ("fruit", "vegetable", "grain")


def _base_rect(tile_xywh: TileBox, *, inset: int = 0) -> tuple[int, int, int, int]:
    x, y, w, h = tile_xywh
    return (
        x * CANONICAL_TILE_PX + int(inset),
        y * CANONICAL_TILE_PX + int(inset),
        (x + w) * CANONICAL_TILE_PX - 1 - int(inset),
        (y + h) * CANONICAL_TILE_PX - 1 - int(inset),
    )


def _shade(color: RGB, delta: int) -> RGB:
    return tuple(max(0, min(255, int(channel) + int(delta))) for channel in color)


def _draw_row_crown(
    draw: ImageDraw.ImageDraw,
    *,
    x0: int,
    y0: int,
    rows: tuple[tuple[int, int, int], ...],
    fill: RGB,
    outline: RGB,
) -> None:
    for rel_y, start_x, end_x in rows:
        draw.line((x0 + max(0, start_x - 1), y0 + rel_y, x0 + min(15, end_x + 1), y0 + rel_y), fill=outline)
    for rel_y, start_x, end_x in rows:
        draw.line((x0 + start_x, y0 + rel_y, x0 + end_x, y0 + rel_y), fill=fill)


def _draw_oak_tree(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], *, leaf_rgb: RGB) -> None:
    x0, y0, x1, y1 = rect
    outline = _shade(leaf_rgb, -64)
    dark_leaf = _shade(leaf_rgb, -34)
    light_leaf = _shade(leaf_rgb, 38)
    trunk = (109, 70, 40)
    trunk_dark = (72, 46, 30)
    draw.rectangle((x0 + 6, y0 + 20, x0 + 10, y1 - 1), fill=trunk, outline=trunk_dark)
    draw.rectangle((x0 + 5, y1 - 3, x0 + 11, y1 - 1), fill=trunk_dark)
    _draw_row_crown(
        draw,
        x0=x0,
        y0=y0,
        rows=(
            (1, 6, 9),
            (2, 5, 10),
            (3, 4, 11),
            (4, 3, 11),
            (5, 2, 12),
            (6, 2, 13),
            (7, 1, 13),
            (8, 2, 14),
            (9, 1, 14),
            (10, 1, 14),
            (11, 2, 13),
            (12, 1, 13),
            (13, 2, 14),
            (14, 2, 13),
            (15, 3, 12),
            (16, 3, 12),
            (17, 4, 11),
            (18, 4, 11),
            (19, 5, 10),
            (20, 6, 9),
            (21, 6, 9),
            (22, 7, 8),
        ),
        fill=leaf_rgb,
        outline=outline,
    )
    draw.line((x0 + 3, y0 + 15, x0 + 12, y0 + 15), fill=dark_leaf)
    draw.line((x0 + 4, y0 + 17, x0 + 11, y0 + 17), fill=dark_leaf)
    draw.line((x0 + 5, y0 + 19, x0 + 10, y0 + 19), fill=dark_leaf)
    draw.rectangle((x0 + 5, y0 + 3, x0 + 8, y0 + 5), fill=light_leaf)
    draw.point((x0 + 11, y0 + 6), fill=light_leaf)
    draw.point((x0 + 3, y0 + 11), fill=_shade(leaf_rgb, 22))
    draw.point((x0 + 12, y0 + 18), fill=outline)
    draw.point((x0 + 2, y0 + 15), fill=outline)


def _draw_pine_tree(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], *, leaf_rgb: RGB) -> None:
    x0, y0, x1, y1 = rect
    outline = (18, 75, 57)
    trunk = (103, 65, 36)
    shadow_leaf = _shade(leaf_rgb, -26)
    draw.rectangle((x0 + 7, y0 + 22, x0 + 9, y1 - 1), fill=trunk, outline=(74, 47, 30))
    for points in (
        ((x0 + 8, y0), (x0 + 3, y0 + 10), (x0 + 13, y0 + 10)),
        ((x0 + 8, y0 + 5), (x0 + 1, y0 + 18), (x0 + 15, y0 + 18)),
        ((x0 + 8, y0 + 11), (x0 + 2, y0 + 25), (x0 + 14, y0 + 25)),
    ):
        draw.polygon(points, fill=leaf_rgb, outline=outline)
    draw.line((x0 + 6, y0 + 11, x0 + 10, y0 + 11), fill=shadow_leaf)
    draw.line((x0 + 5, y0 + 19, x0 + 11, y0 + 19), fill=shadow_leaf)
    draw.point((x0 + 8, y0 + 3), fill=_shade(leaf_rgb, 38))


def _draw_maple_tree(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], *, leaf_rgb: RGB) -> None:
    x0, y0, x1, y1 = rect
    outline = _shade(leaf_rgb, -68)
    dark_leaf = _shade(leaf_rgb, -38)
    light_leaf = _shade(leaf_rgb, 42)
    trunk = (114, 67, 39)
    trunk_dark = (78, 46, 30)
    draw.rectangle((x0 + 6, y0 + 20, x0 + 10, y1 - 1), fill=trunk, outline=trunk_dark)
    draw.rectangle((x0 + 5, y1 - 3, x0 + 11, y1 - 1), fill=trunk_dark)
    _draw_row_crown(
        draw,
        x0=x0,
        y0=y0,
        rows=(
            (0, 6, 9),
            (1, 5, 10),
            (2, 5, 11),
            (3, 3, 11),
            (4, 2, 12),
            (5, 4, 13),
            (6, 1, 13),
            (7, 2, 14),
            (8, 1, 14),
            (9, 0, 13),
            (10, 2, 15),
            (11, 1, 14),
            (12, 2, 13),
            (13, 3, 14),
            (14, 2, 12),
            (15, 3, 13),
            (16, 4, 11),
            (17, 3, 10),
            (18, 5, 11),
            (19, 6, 9),
            (20, 7, 8),
        ),
        fill=leaf_rgb,
        outline=outline,
    )
    draw.line((x0 + 3, y0 + 14, x0 + 12, y0 + 14), fill=dark_leaf)
    draw.line((x0 + 4, y0 + 16, x0 + 11, y0 + 16), fill=dark_leaf)
    draw.line((x0 + 5, y0 + 18, x0 + 10, y0 + 18), fill=dark_leaf)
    draw.rectangle((x0 + 4, y0 + 4, x0 + 7, y0 + 6), fill=light_leaf)
    draw.point((x0 + 10, y0 + 5), fill=light_leaf)
    draw.point((x0 + 2, y0 + 11), fill=_shade(leaf_rgb, 24))
    draw.point((x0 + 12, y0 + 12), fill=_shade(leaf_rgb, 18))
    draw.point((x0 + 13, y0 + 17), fill=outline)


def _draw_fruit_tree(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    leaf_rgb: RGB,
    fruit_rgb: RGB,
) -> None:
    x0, y0, x1, y1 = rect
    outline = _shade(leaf_rgb, -62)
    dark_leaf = _shade(leaf_rgb, -32)
    light_leaf = _shade(leaf_rgb, 38)
    trunk = (108, 70, 40)
    trunk_dark = (76, 48, 30)
    draw.rectangle((x0 + 6, y0 + 20, x0 + 10, y1 - 1), fill=trunk, outline=trunk_dark)
    draw.rectangle((x0 + 5, y1 - 3, x0 + 11, y1 - 1), fill=trunk_dark)
    _draw_row_crown(
        draw,
        x0=x0,
        y0=y0,
        rows=(
            (1, 6, 9),
            (2, 5, 10),
            (3, 4, 11),
            (4, 3, 11),
            (5, 2, 12),
            (6, 2, 13),
            (7, 1, 13),
            (8, 2, 14),
            (9, 1, 14),
            (10, 1, 14),
            (11, 2, 13),
            (12, 1, 13),
            (13, 2, 14),
            (14, 2, 13),
            (15, 3, 12),
            (16, 3, 12),
            (17, 4, 11),
            (18, 4, 11),
            (19, 5, 10),
            (20, 6, 9),
            (21, 6, 9),
            (22, 7, 8),
        ),
        fill=leaf_rgb,
        outline=outline,
    )
    draw.line((x0 + 3, y0 + 15, x0 + 12, y0 + 15), fill=dark_leaf)
    draw.line((x0 + 4, y0 + 17, x0 + 11, y0 + 17), fill=dark_leaf)
    draw.line((x0 + 5, y0 + 19, x0 + 10, y0 + 19), fill=dark_leaf)
    draw.rectangle((x0 + 5, y0 + 4, x0 + 8, y0 + 6), fill=light_leaf)
    for px, py in ((x0 + 5, y0 + 9), (x0 + 11, y0 + 11), (x0 + 8, y0 + 16), (x0 + 4, y0 + 15), (x0 + 12, y0 + 18)):
        draw.point((px, py), fill=fruit_rgb)
        draw.point((px + 1, py), fill=fruit_rgb)
        draw.point((px, py + 1), fill=_shade(fruit_rgb, -28))


def draw_pixel_tree(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    style: str = "oak",
    leaf_rgb: RGB = (38, 144, 78),
    fruit_rgb: RGB = (220, 65, 61),
) -> None:
    """Draw a reusable 1x2-tile pixel tree."""

    rect = _base_rect(tile_xywh)
    normalized_style = str(style)
    if normalized_style == "pine":
        _draw_pine_tree(draw, rect, leaf_rgb=leaf_rgb)
    elif normalized_style == "maple":
        _draw_maple_tree(draw, rect, leaf_rgb=leaf_rgb)
    elif normalized_style == "fruit_tree":
        _draw_fruit_tree(draw, rect, leaf_rgb=leaf_rgb, fruit_rgb=fruit_rgb)
    else:
        _draw_oak_tree(draw, rect, leaf_rgb=leaf_rgb)


def draw_pixel_flower_patch(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    flower_rgb: RGB = (236, 82, 111),
    leaf_rgb: RGB = (40, 128, 62),
) -> None:
    """Draw a reusable 1-tile flower/low-plant patch."""

    x0, y0, _, _ = _base_rect(tile_xywh)
    leaf_dark = _shade(leaf_rgb, -34)
    flower_dark = _shade(flower_rgb, -34)
    for sx, sy in ((4, 10), (8, 8), (11, 11)):
        draw.line((x0 + sx, y0 + sy, x0 + sx, y0 + 14), fill=leaf_dark)
        draw.point((x0 + sx - 1, y0 + sy + 2), fill=leaf_rgb)
        draw.point((x0 + sx + 1, y0 + sy + 1), fill=leaf_rgb)
        draw.point((x0 + sx, y0 + sy), fill=flower_rgb)
        draw.point((x0 + sx - 1, y0 + sy), fill=flower_rgb)
        draw.point((x0 + sx + 1, y0 + sy), fill=flower_rgb)
        draw.point((x0 + sx, y0 + sy - 1), fill=flower_rgb)
        draw.point((x0 + sx, y0 + sy + 1), fill=flower_dark)
    draw.line((x0 + 3, y0 + 14, x0 + 12, y0 + 14), fill=leaf_dark)


def draw_pixel_crop_row(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    style: str = "wheat",
    crop_rgb: RGB = (210, 171, 58),
    soil_rgb: RGB = (130, 87, 48),
) -> None:
    """Draw a reusable horizontal crop row for small farm territories."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    soil_dark = _shade(soil_rgb, -34)
    soil_light = _shade(soil_rgb, 22)
    crop_dark = _shade(crop_rgb, -38)
    crop_light = _shade(crop_rgb, 34)
    for sx in range(x0 + 2, x1 - 2, 8):
        draw.line((sx, y0 + 14, min(x1 - 2, sx + 4), y0 + 14), fill=soil_dark)
        draw.point((min(x1 - 2, sx + 2), y0 + 12), fill=soil_light)
    normalized_style = str(style)
    for px in range(x0 + 4, x1 - 1, 6):
        stem_bottom = y0 + 13
        if normalized_style == "leafy":
            draw.line((px, y0 + 8, px, stem_bottom), fill=crop_dark)
            draw.point((px - 2, y0 + 9), fill=crop_rgb)
            draw.point((px - 1, y0 + 8), fill=crop_light)
            draw.point((px + 1, y0 + 8), fill=crop_rgb)
            draw.point((px + 2, y0 + 10), fill=crop_dark)
            draw.point((px - 1, y0 + 11), fill=_shade(crop_rgb, 8))
        elif normalized_style == "flowering":
            leaf = (45, 128, 62)
            draw.line((px, y0 + 7, px, stem_bottom), fill=_shade(leaf, -28))
            draw.point((px - 1, y0 + 9), fill=leaf)
            draw.point((px + 1, y0 + 10), fill=leaf)
            draw.point((px - 1, y0 + 6), fill=crop_rgb)
            draw.point((px, y0 + 5), fill=crop_light)
            draw.point((px + 1, y0 + 6), fill=crop_rgb)
            draw.point((px, y0 + 7), fill=crop_dark)
        else:
            draw.line((px, y0 + 6, px, stem_bottom), fill=crop_dark)
            draw.point((px - 1, y0 + 6), fill=crop_rgb)
            draw.point((px, y0 + 5), fill=crop_light)
            draw.point((px + 1, y0 + 6), fill=crop_rgb)
            draw.point((px - 1, y0 + 8), fill=_shade(crop_rgb, 10))
            draw.point((px + 1, y0 + 9), fill=crop_dark)


def draw_pixel_vegetable_patch(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    style: str = "carrot",
    vegetable_rgb: RGB | None = None,
    leaf_rgb: RGB = (45, 139, 67),
    soil_rgb: RGB = (130, 87, 48),
) -> None:
    """Draw one reusable 1-tile vegetable/crop object."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    soil_dark = _shade(soil_rgb, -36)
    soil_light = _shade(soil_rgb, 26)
    leaf_dark = _shade(leaf_rgb, -38)
    leaf_light = _shade(leaf_rgb, 34)
    draw.ellipse((x0 + 1, y0 + 8, x1 - 1, y1), fill=soil_rgb, outline=soil_dark)
    draw.line((x0 + 3, y0 + 12, x1 - 3, y0 + 12), fill=soil_light)

    normalized_style = str(style)
    if normalized_style == "cabbage":
        cabbage = vegetable_rgb or (102, 180, 84)
        cabbage_dark = (39, 96, 54)
        cabbage_shadow = _shade(cabbage, -28)
        cabbage_light = _shade(cabbage, 58)
        draw.ellipse((x0 + 3, y0 + 4, x0 + 12, y0 + 13), fill=cabbage, outline=cabbage_dark)
        draw.pieslice((x0 + 1, y0 + 6, x0 + 8, y0 + 13), 110, 300, fill=cabbage_shadow, outline=cabbage_dark)
        draw.pieslice((x0 + 7, y0 + 5, x0 + 14, y0 + 12), 240, 70, fill=_shade(cabbage, 8), outline=cabbage_dark)
        draw.arc((x0 + 5, y0 + 6, x0 + 10, y0 + 11), 200, 40, fill=cabbage_light)
        draw.point((x0 + 7, y0 + 7), fill=cabbage_light)
    elif normalized_style == "corn":
        cob = vegetable_rgb or (236, 195, 64)
        cob_dark = _shade(cob, -36)
        cx = x0 + 8
        cy = y0 + 9
        draw.polygon([(cx, y0 + 3), (cx - 5, cy), (cx - 1, cy + 1)], fill=leaf_rgb, outline=leaf_dark)
        draw.polygon([(cx, y0 + 3), (cx + 5, cy), (cx + 1, cy + 1)], fill=leaf_light, outline=leaf_dark)
        draw.polygon([(cx - 1, cy + 1), (cx - 5, y0 + 13), (cx, y0 + 11)], fill=_shade(leaf_rgb, -8), outline=leaf_dark)
        draw.polygon([(cx + 1, cy + 1), (cx + 5, y0 + 13), (cx, y0 + 11)], fill=leaf_rgb, outline=leaf_dark)
        draw.ellipse((cx - 3, y0 + 5, cx + 3, y0 + 12), fill=cob, outline=cob_dark)
        draw.point((cx - 1, y0 + 7), fill=_shade(cob, 28))
        draw.point((cx + 1, y0 + 9), fill=cob_dark)
    elif normalized_style == "tomato":
        tomato = vegetable_rgb or (213, 58, 49)
        tomato_dark = _shade(tomato, -40)
        cx = x0 + 8
        draw.polygon([(cx, y0 + 3), (cx - 5, y0 + 8), (cx - 2, y0 + 12), (cx + 2, y0 + 12), (cx + 5, y0 + 8)], fill=leaf_rgb, outline=leaf_dark)
        draw.point((cx - 3, y0 + 7), fill=leaf_light)
        draw.point((cx + 3, y0 + 8), fill=leaf_light)
        draw.ellipse((cx - 4, y0 + 7, cx + 4, y0 + 14), fill=tomato, outline=tomato_dark)
        draw.point((cx - 1, y0 + 8), fill=_shade(tomato, 42))
        draw.point((cx + 1, y0 + 12), fill=_shade(tomato, -18))
    elif normalized_style == "pumpkin":
        pumpkin = vegetable_rgb or (224, 124, 45)
        pumpkin_dark = _shade(pumpkin, -46)
        pumpkin_light = _shade(pumpkin, 34)
        cx = x0 + 8
        draw.ellipse((cx - 6, y0 + 5, cx + 6, y0 + 14), fill=pumpkin, outline=pumpkin_dark)
        draw.arc((cx - 5, y0 + 6, cx + 5, y0 + 14), 80, 280, fill=_shade(pumpkin, -20))
        draw.arc((cx - 3, y0 + 6, cx + 3, y0 + 14), 260, 100, fill=pumpkin_light)
        draw.line((cx, y0 + 5, cx + 3, y0 + 3), fill=leaf_dark)
        draw.point((cx + 4, y0 + 4), fill=leaf_rgb)
    else:
        carrot = vegetable_rgb or (226, 112, 45)
        carrot_dark = _shade(carrot, -42)
        cx = x0 + 8
        draw.polygon([(cx, y0 + 3), (cx - 4, y0 + 8), (cx - 2, y0 + 11), (cx + 2, y0 + 11), (cx + 4, y0 + 8)], fill=leaf_rgb, outline=leaf_dark)
        draw.point((cx - 3, y0 + 7), fill=leaf_light)
        draw.point((cx + 3, y0 + 7), fill=leaf_light)
        draw.polygon(
            [(cx - 3, y0 + 8), (cx + 3, y0 + 8), (cx + 2, y0 + 13), (cx, y0 + 15), (cx - 2, y0 + 13)],
            fill=carrot,
            outline=carrot_dark,
        )
        draw.line((cx - 1, y0 + 9, cx - 1, y0 + 13), fill=_shade(carrot, -18))
        draw.point((cx + 1, y0 + 9), fill=_shade(carrot, 34))


def draw_pixel_hay_bale(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a reusable one-tile hay bale."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    fill = (218, 178, 72)
    outline = (123, 91, 42)
    draw.rectangle((x0, y0 + 3, x1, y1 - 1), fill=fill, outline=outline)
    draw.line((x0 + 2, y0 + 6, x1 - 2, y0 + 6), fill=(244, 207, 100))
    draw.line((x0 + 4, y0 + 10, x1 - 4, y0 + 10), fill=(166, 120, 51))
    draw.rectangle((x0 + 4, y0 + 3, x0 + 5, y1 - 1), fill=outline)


def draw_pixel_trough(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a reusable two-tile animal trough."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    draw.polygon(
        [(x0, y0 + 5), (x1, y0 + 5), (x1 - 3, y1 - 2), (x0 + 3, y1 - 2)],
        fill=(124, 83, 48),
        outline=(76, 51, 33),
    )
    draw.rectangle((x0 + 3, y0 + 6, x1 - 3, y0 + 9), fill=(86, 150, 175), outline=(54, 93, 112))
    draw.point((x0 + 7, y0 + 7), fill=(168, 217, 226))


def draw_pixel_crate(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a reusable one-tile wooden crate."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    draw.rectangle((x0, y0, x1, y1), fill=(174, 110, 52), outline=(95, 63, 36))
    draw.line((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=(118, 77, 43))
    draw.line((x0 + 2, y1 - 2, x1 - 2, y0 + 2), fill=(118, 77, 43))


def draw_pixel_counter(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (151, 91, 49),
    top_rgb: RGB = (194, 139, 78),
) -> None:
    """Draw a reusable RPG shop counter."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -48)
    wood_light = _shade(wood_rgb, 34)
    top_dark = _shade(top_rgb, -42)
    draw.rectangle((x0 + 1, y0 + 4, x1 - 1, y1 - 2), fill=wood_rgb, outline=wood_dark)
    draw.rectangle((x0 + 1, y0 + 2, x1 - 1, y0 + 7), fill=top_rgb, outline=top_dark)
    draw.line((x0 + 3, y0 + 4, x1 - 3, y0 + 4), fill=_shade(top_rgb, 30))
    for px in range(x0 + 8, x1 - 4, 10):
        draw.line((px, y0 + 8, px, y1 - 3), fill=wood_dark)
        draw.line((px + 1, y0 + 8, px + 1, y1 - 4), fill=wood_light)
    draw.rectangle((x1 - 12, y0 + 8, x1 - 6, y0 + 12), fill=(226, 201, 125), outline=(103, 75, 43))
    draw.point((x1 - 10, y0 + 9), fill=(174, 56, 51))


def draw_pixel_shelf(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (135, 82, 45),
    goods_type: str = "mixed",
) -> None:
    """Draw a reusable RPG shop shelf with visible goods."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -48)
    wood_light = _shade(wood_rgb, 32)
    draw.rectangle((x0 + 1, y0 + 2, x1 - 1, y1 - 2), fill=wood_rgb, outline=wood_dark)
    for yy in (y0 + 6, y0 + 12):
        draw.rectangle((x0 + 2, yy, x1 - 2, yy + 2), fill=wood_dark)
        draw.line((x0 + 4, yy, x1 - 4, yy), fill=wood_light)
    for px in range(x0 + 7, x1 - 6, 14):
        draw.line((px, y0 + 3, px, y1 - 3), fill=_shade(wood_rgb, -30))

    goods = str(goods_type)
    slots = tuple(range(x0 + 6, x1 - 5, 8))
    for index, px in enumerate(slots):
        shelf_y = y0 + 5 if index % 2 == 0 else y0 + 11
        if goods == "jars" or (goods == "mixed" and index % 3 == 0):
            color = ((86, 153, 184), (200, 93, 78), (235, 190, 75))[index % 3]
            draw.rectangle((px, shelf_y - 3, px + 3, shelf_y + 1), fill=color, outline=_shade(color, -45))
            draw.point((px + 1, shelf_y - 2), fill=_shade(color, 55))
        elif goods == "produce" or (goods == "mixed" and index % 3 == 1):
            color = ((215, 67, 55), (95, 169, 77), (224, 139, 45))[index % 3]
            draw.ellipse((px - 1, shelf_y - 2, px + 4, shelf_y + 2), fill=color, outline=_shade(color, -42))
        else:
            color = ((76, 117, 176), (183, 75, 116), (95, 145, 94))[index % 3]
            draw.rectangle((px - 1, shelf_y - 4, px + 3, shelf_y + 1), fill=color, outline=_shade(color, -42))
            draw.line((px + 1, shelf_y - 3, px + 1, shelf_y + 1), fill=_shade(color, 38))


def draw_pixel_produce_bin(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    goods_type: str = "fruit",
    wood_rgb: RGB = (139, 82, 43),
    produce_rgb: RGB | None = None,
) -> None:
    """Draw a reusable two-tile produce bin."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    wood_dark = _shade(wood_rgb, -44)
    draw.polygon(
        [(x0 + 1, y0 + 5), (x1 - 1, y0 + 5), (x1 - 4, y1 - 2), (x0 + 4, y1 - 2)],
        fill=wood_rgb,
        outline=wood_dark,
    )
    draw.line((x0 + 3, y0 + 9, x1 - 4, y0 + 9), fill=_shade(wood_rgb, 28))
    goods = str(goods_type)
    palette = {
        "vegetable": ((67, 152, 74), (99, 184, 86), (45, 116, 63)),
        "grain": ((220, 178, 70), (238, 204, 96), (170, 121, 50)),
        "fruit": ((213, 67, 55), (225, 140, 43), (235, 190, 70)),
    }.get(goods, ((213, 67, 55), (225, 140, 43), (235, 190, 70)))
    if produce_rgb is not None:
        palette = (produce_rgb, _shade(produce_rgb, 28), _shade(produce_rgb, -32))
    for index, px in enumerate(range(x0 + 6, x1 - 5, 6)):
        color = palette[index % len(palette)]
        py = y0 + 5 + (index % 2)
        if goods == "grain":
            draw.line((px, py - 1, px, py + 5), fill=_shade(color, -42))
            draw.point((px - 1, py), fill=color)
            draw.point((px + 1, py + 1), fill=_shade(color, 20))
        else:
            draw.ellipse((px - 2, py - 1, px + 3, py + 4), fill=color, outline=_shade(color, -44))
            draw.point((px - 1, py), fill=_shade(color, 50))


def draw_pixel_sack(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    cloth_rgb: RGB = (191, 158, 95),
) -> None:
    """Draw a reusable one-tile grain sack."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    cloth_dark = _shade(cloth_rgb, -46)
    cloth_light = _shade(cloth_rgb, 32)
    draw.polygon(
        [(x0 + 5, y0 + 2), (x1 - 4, y0 + 2), (x1 - 1, y1 - 3), (x0 + 2, y1 - 1), (x0 + 1, y0 + 8)],
        fill=cloth_rgb,
        outline=cloth_dark,
    )
    draw.line((x0 + 5, y0 + 5, x1 - 4, y0 + 5), fill=cloth_dark)
    draw.point((x0 + 5, y0 + 8), fill=cloth_light)
    draw.point((x0 + 7, y0 + 10), fill=cloth_light)


def draw_pixel_jar(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    glass_rgb: RGB = (92, 155, 178),
) -> None:
    """Draw a reusable one-tile jar."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=3)
    glass_dark = _shade(glass_rgb, -48)
    glass_light = _shade(glass_rgb, 62)
    lid = (98, 69, 43)
    draw.rectangle((x0 + 3, y0 + 1, x1 - 3, y0 + 4), fill=lid, outline=(59, 42, 31))
    draw.ellipse((x0 + 1, y0 + 3, x1 - 1, y1 - 1), fill=glass_rgb, outline=glass_dark)
    draw.rectangle((x0 + 2, y0 + 6, x1 - 2, y1 - 3), fill=_shade(glass_rgb, -8))
    draw.point((x0 + 4, y0 + 6), fill=glass_light)
    draw.point((x0 + 5, y0 + 7), fill=glass_light)


def draw_pixel_pot(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    clay_rgb: RGB = (174, 94, 58),
) -> None:
    """Draw a reusable one-tile clay pot."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=3)
    clay_dark = _shade(clay_rgb, -48)
    clay_light = _shade(clay_rgb, 36)
    draw.ellipse((x0 + 1, y0 + 3, x1 - 1, y0 + 8), fill=clay_rgb, outline=clay_dark)
    draw.polygon(
        [(x0 + 2, y0 + 6), (x1 - 2, y0 + 6), (x1 - 4, y1 - 1), (x0 + 4, y1 - 1)],
        fill=clay_rgb,
        outline=clay_dark,
    )
    draw.line((x0 + 4, y0 + 8, x1 - 4, y0 + 8), fill=clay_light)
    draw.point((x0 + 5, y0 + 7), fill=_shade(clay_rgb, 50))


def draw_pixel_basket(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wicker_rgb: RGB = (180, 121, 62),
) -> None:
    """Draw a reusable one-tile woven basket."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wicker_dark = _shade(wicker_rgb, -48)
    wicker_light = _shade(wicker_rgb, 34)
    weave_shadow = _shade(wicker_rgb, -28)
    draw.arc((x0 + 3, y0 + 1, x1 - 3, y0 + 11), start=180, end=360, fill=wicker_dark, width=2)
    draw.arc((x0 + 4, y0 + 2, x1 - 4, y0 + 10), start=190, end=350, fill=wicker_light, width=1)
    draw.polygon(
        [(x0 + 2, y0 + 8), (x1 - 2, y0 + 8), (x1 - 4, y1 - 2), (x0 + 4, y1 - 2)],
        fill=wicker_rgb,
        outline=wicker_dark,
    )
    draw.ellipse((x0 + 2, y0 + 5, x1 - 2, y0 + 10), fill=wicker_light, outline=wicker_dark)
    draw.line((x0 + 4, y0 + 8, x1 - 4, y0 + 8), fill=_shade(wicker_rgb, 18))
    for px in (x0 + 5, x0 + 8, x1 - 5):
        draw.line((px, y0 + 9, px, y1 - 3), fill=weave_shadow)
    draw.line((x0 + 4, y0 + 11, x1 - 4, y0 + 11), fill=weave_shadow)
    draw.point((x0 + 6, y0 + 7), fill=_shade(wicker_light, 25))


def draw_pixel_rug(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    cloth_rgb: RGB = (168, 74, 82),
    trim_rgb: RGB = (230, 190, 96),
) -> None:
    """Draw a reusable rectangular shop rug."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    cloth_dark = _shade(cloth_rgb, -44)
    trim_dark = _shade(trim_rgb, -46)
    draw.rectangle((x0, y0 + 1, x1, y1 - 1), fill=cloth_rgb, outline=cloth_dark)
    draw.rectangle((x0 + 2, y0 + 3, x1 - 2, y1 - 3), outline=trim_rgb)
    draw.line((x0 + 5, y0 + 5, x1 - 5, y1 - 5), fill=trim_dark)
    draw.line((x0 + 5, y1 - 5, x1 - 5, y0 + 5), fill=trim_dark)
    for px in range(x0 + 2, x1, 5):
        draw.point((px, y0), fill=trim_rgb)
        draw.point((px, y1), fill=trim_rgb)


def draw_pixel_chest(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (139, 82, 43),
    metal_rgb: RGB = (189, 160, 80),
) -> None:
    """Draw a reusable two-tile treasure/storage chest."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    width = x1 - x0 + 1
    if width > 22:
        trim = max(2, width // 8)
        x0 += trim
        x1 -= trim
    wood_dark = _shade(wood_rgb, -48)
    wood_light = _shade(wood_rgb, 35)
    metal_dark = _shade(metal_rgb, -52)
    lid_rgb = _shade(wood_rgb, 16)
    body_y0 = y0 + 6
    lid_y0 = y0 + 1
    lid_y1 = y0 + 7
    draw.rectangle((x0 + 1, body_y0, x1 - 1, y1 - 1), fill=wood_rgb, outline=wood_dark)
    draw.rectangle((x0 + 2, lid_y0, x1 - 2, lid_y1), fill=lid_rgb, outline=wood_dark)
    draw.line((x0 + 2, lid_y1, x1 - 2, lid_y1), fill=wood_dark)
    draw.line((x0 + 4, y0 + 3, x1 - 4, y0 + 3), fill=wood_light)
    draw.line((x0 + 4, y0 + 11, x1 - 4, y0 + 11), fill=_shade(wood_rgb, -18))
    draw.rectangle((x0 + 5, lid_y0, x0 + 7, y1 - 1), fill=metal_rgb, outline=metal_dark)
    draw.rectangle((x1 - 7, lid_y0, x1 - 5, y1 - 1), fill=metal_rgb, outline=metal_dark)
    lock_x = (x0 + x1) // 2
    draw.rectangle((lock_x - 2, y0 + 8, lock_x + 2, y0 + 12), fill=metal_rgb, outline=metal_dark)
    draw.point((lock_x, y0 + 10), fill=_shade(metal_rgb, 42))


def draw_pixel_table(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    table_shape: str = "square",
    wood_rgb: RGB = (139, 82, 43),
) -> None:
    """Draw a reusable tavern table."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    wood_dark = _shade(wood_rgb, -48)
    wood_light = _shade(wood_rgb, 34)
    shape = str(table_shape)
    if shape == "round":
        cx = (x0 + x1) // 2
        cy = y0 + max(6, (y1 - y0) // 2)
        radius_x = max(6, min(15, (x1 - x0) // 2 - 1))
        radius_y = max(4, min(10, (y1 - y0) // 3))
        draw.ellipse((cx - radius_x, cy - radius_y, cx + radius_x, cy + radius_y), fill=wood_rgb, outline=wood_dark)
        draw.line((cx - radius_x + 3, cy - 1, cx + radius_x - 3, cy - 1), fill=wood_light)
        draw.rectangle((cx - 2, cy + radius_y - 1, cx + 2, y1 - 3), fill=wood_dark)
        draw.rectangle((cx - 5, y1 - 4, cx + 5, y1 - 2), fill=wood_dark)
        return
    top_y0 = y0 + 2
    top_y1 = y0 + max(8, min(15, (y1 - y0) // 2))
    if shape == "long":
        draw.rectangle((x0 + 1, top_y0, x1 - 1, top_y1), fill=wood_rgb, outline=wood_dark)
        for px in range(x0 + 9, x1 - 5, 10):
            draw.line((px, top_y0 + 1, px, top_y1 - 1), fill=_shade(wood_rgb, -22))
        draw.line((x0 + 5, top_y0 + 2, x1 - 5, top_y0 + 2), fill=wood_light)
    else:
        side = min(x1 - x0 - 1, y1 - y0 - 5)
        tx0 = (x0 + x1 - side) // 2
        tx1 = tx0 + side
        draw.rectangle((tx0, top_y0, tx1, top_y0 + max(9, side // 2)), fill=wood_rgb, outline=wood_dark)
        draw.line((tx0 + 2, top_y0 + 2, tx1 - 2, top_y0 + 2), fill=wood_light)
        x0, x1 = tx0, tx1
        top_y1 = top_y0 + max(9, side // 2)
    leg_top = top_y1 + 1
    for lx in (x0 + 5, x1 - 7):
        draw.rectangle((lx, leg_top, lx + 3, y1 - 4), fill=wood_dark)
        draw.point((lx + 1, leg_top + 1), fill=wood_light)
    draw.line((x0 + 4, y1 - 3, x1 - 4, y1 - 3), fill=wood_dark)


def draw_pixel_chair(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    facing: str = "down",
    wood_rgb: RGB = (126, 75, 42),
    cushion_rgb: RGB | None = None,
) -> None:
    """Draw a reusable one-tile tavern chair."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -46)
    wood_light = _shade(wood_rgb, 34)
    back_panel = _shade(wood_rgb, 24)
    cushion = cushion_rgb or (183, 101, 48)
    cushion_dark = _shade(cushion, -42)
    cushion_light = _shade(cushion, 38)
    direction = str(facing)

    if direction == "up":
        draw.rectangle((x0 + 3, y0 + 1, x0 + 4, y1 - 2), fill=wood_rgb)
        draw.rectangle((x1 - 4, y0 + 1, x1 - 3, y1 - 2), fill=wood_dark)
        draw.rectangle((x0 + 3, y0 + 1, x1 - 3, y0 + 3), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((x0 + 5, y0 + 4, x1 - 5, y1 - 5), fill=back_panel, outline=wood_dark)
        draw.line((x0 + 6, y0 + 5, x1 - 6, y0 + 5), fill=wood_light)
        draw.rectangle((x0 + 4, y1 - 4, x1 - 4, y1 - 2), fill=wood_rgb, outline=wood_dark)
        draw.line((x0 + 5, y1 - 3, x1 - 5, y1 - 3), fill=wood_light)
    elif direction == "left":
        draw.rectangle((x1 - 4, y0 + 1, x1 - 2, y1 - 2), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((x0 + 4, y0 + 5, x1 - 5, y0 + 7), fill=back_panel, outline=wood_dark)
        draw.rectangle((x0 + 4, y0 + 8, x1 - 5, y1 - 5), fill=cushion, outline=cushion_dark)
        draw.line((x0 + 5, y0 + 9, x1 - 6, y0 + 9), fill=cushion_light)
        draw.rectangle((x0 + 3, y1 - 4, x0 + 4, y1 - 2), fill=wood_dark)
        draw.rectangle((x1 - 5, y1 - 4, x1 - 4, y1 - 2), fill=wood_dark)
        draw.line((x0 + 4, y1 - 4, x1 - 5, y1 - 4), fill=wood_rgb)
    elif direction == "right":
        draw.rectangle((x0 + 2, y0 + 1, x0 + 4, y1 - 2), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((x0 + 5, y0 + 5, x1 - 4, y0 + 7), fill=back_panel, outline=wood_dark)
        draw.rectangle((x0 + 5, y0 + 8, x1 - 4, y1 - 5), fill=cushion, outline=cushion_dark)
        draw.line((x0 + 6, y0 + 9, x1 - 5, y0 + 9), fill=cushion_light)
        draw.rectangle((x0 + 4, y1 - 4, x0 + 5, y1 - 2), fill=wood_dark)
        draw.rectangle((x1 - 4, y1 - 4, x1 - 3, y1 - 2), fill=wood_dark)
        draw.line((x0 + 5, y1 - 4, x1 - 4, y1 - 4), fill=wood_rgb)
    else:
        draw.rectangle((x0 + 3, y0 + 1, x0 + 4, y1 - 3), fill=wood_rgb)
        draw.rectangle((x1 - 4, y0 + 1, x1 - 3, y1 - 3), fill=wood_dark)
        draw.rectangle((x0 + 3, y0 + 1, x1 - 3, y0 + 3), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((x0 + 5, y0 + 4, x1 - 5, y0 + 7), fill=back_panel, outline=wood_dark)
        draw.rectangle((x0 + 5, y0 + 8, x1 - 5, y1 - 5), fill=cushion, outline=cushion_dark)
        draw.line((x0 + 6, y0 + 9, x1 - 6, y0 + 9), fill=cushion_light)
        draw.rectangle((x0 + 4, y1 - 4, x0 + 5, y1 - 2), fill=wood_dark)
        draw.rectangle((x1 - 5, y1 - 4, x1 - 4, y1 - 2), fill=wood_dark)
        draw.line((x0 + 5, y1 - 4, x1 - 5, y1 - 4), fill=wood_rgb)
    draw.point((x0 + 4, y0 + 4), fill=wood_light)
    draw.point((x1 - 4, y1 - 4), fill=wood_dark)


def draw_pixel_stool(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (126, 75, 42),
    cushion_rgb: RGB = (151, 83, 58),
) -> None:
    """Draw a reusable one-tile stool."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -46)
    wood_light = _shade(wood_rgb, 30)
    cushion_dark = _shade(cushion_rgb, -42)
    cushion_light = _shade(cushion_rgb, 38)
    cx = (x0 + x1) // 2
    seat = (cx - 5, y0 + 3, cx + 5, y0 + 9)
    for px, py in ((cx - 5, y0 + 8), (cx + 4, y0 + 8), (cx - 5, y0 + 11), (cx + 4, y0 + 11)):
        draw.rectangle((px, py, px + 1, y1 - 2), fill=wood_dark)
        draw.point((px, py), fill=wood_light)
    draw.rectangle(seat, fill=cushion_rgb, outline=cushion_dark)
    draw.line((seat[0] + 2, seat[1] + 1, seat[2] - 2, seat[1] + 1), fill=cushion_light)
    draw.line((seat[0] + 1, seat[3] - 1, seat[2] - 1, seat[3] - 1), fill=_shade(cushion_rgb, -20))
    draw.line((cx - 5, y0 + 10, cx + 5, y0 + 10), fill=wood_rgb)


def draw_pixel_bed(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    bed_size: str = "single",
    wood_rgb: RGB = (118, 72, 42),
    blanket_rgb: RGB = (97, 132, 173),
    pillow_rgb: RGB = (232, 222, 188),
) -> None:
    """Draw a reusable RPG inn bed."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    wood_dark = _shade(wood_rgb, -48)
    blanket_dark = _shade(blanket_rgb, -44)
    blanket_light = _shade(blanket_rgb, 34)
    pillow_dark = _shade(pillow_rgb, -42)
    if str(bed_size) == "double":
        draw.rectangle((x0, y0 + 2, x1, y1 - 1), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((x0 + 2, y0 + 3, x1 - 2, y0 + 14), fill=pillow_rgb, outline=pillow_dark)
        mid_x = (x0 + x1) // 2
        draw.line((mid_x, y0 + 5, mid_x, y0 + 13), fill=pillow_dark)
        draw.rectangle((x0 + 3, y0 + 15, x1 - 3, y1 - 4), fill=blanket_rgb, outline=blanket_dark)
        for yy in range(y0 + 18, y1 - 6, 7):
            draw.line((x0 + 5, yy, x1 - 5, yy), fill=_shade(blanket_rgb, -18))
        for xx in range(x0 + 8, x1 - 7, 10):
            draw.line((xx, y0 + 16, xx, y1 - 5), fill=_shade(blanket_rgb, 18))
        draw.line((x0 + 4, y0 + 15, x0 + 9, y0 + 19, x0 + 5, y0 + 23), fill=blanket_light)
    else:
        draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 1), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((x0 + 4, y0 + 4, x1 - 4, y0 + 14), fill=pillow_rgb, outline=pillow_dark)
        draw.line((x0 + 7, y0 + 8, x1 - 7, y0 + 8), fill=_shade(pillow_rgb, -16))
        draw.rectangle((x0 + 5, y0 + 15, x1 - 5, y1 - 4), fill=blanket_rgb, outline=blanket_dark)
        for yy in range(y0 + 19, y1 - 6, 6):
            draw.line((x0 + 7, yy, x1 - 7, yy), fill=_shade(blanket_rgb, -18))
        draw.line((x0 + 8, y0 + 16, x0 + 8, y1 - 5), fill=_shade(blanket_rgb, 18))
        draw.line((x0 + 7, y0 + 17, x1 - 7, y0 + 17), fill=blanket_light)
    draw.rectangle((x0 + 1, y1 - 6, x1 - 1, y1 - 2), fill=wood_rgb, outline=wood_dark)
    for px in range(x0 + 5, x1 - 4, 8):
        draw.line((px, y1 - 6, px + 3, y1 - 2), fill=_shade(wood_rgb, -24))
    draw.point((x0 + 5, y1 - 4), fill=_shade(wood_rgb, 36))
    draw.point((x1 - 5, y1 - 4), fill=_shade(wood_rgb, 36))


def draw_pixel_fireplace(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (116, 109, 96),
    fire_state: str = "lit",
    flame_rgb: RGB = (238, 126, 45),
) -> None:
    """Draw a reusable two-tile inn fireplace."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    stone_dark = _shade(stone_rgb, -48)
    stone_light = _shade(stone_rgb, 38)
    draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 1), fill=stone_rgb, outline=stone_dark)
    draw.rectangle((x0 + 1, y0 + 2, x1 - 1, y0 + 6), fill=stone_light, outline=stone_dark)
    for px in range(x0 + 6, x1 - 5, 9):
        draw.line((px, y0 + 7, px, y1 - 2), fill=stone_dark)
    mouth = (x0 + 7, y0 + 8, x1 - 7, y1 - 3)
    draw.rectangle(mouth, fill=(47, 39, 34), outline=(35, 30, 28))
    draw.rectangle((x0 + 5, y1 - 4, x1 - 5, y1 - 2), fill=(86, 55, 36))
    if str(fire_state) == "lit":
        cx = (x0 + x1) // 2
        fy = y1 - 6
        flame_dark = _shade(flame_rgb, -48)
        draw.polygon([(cx, fy - 5), (cx - 5, fy), (cx - 2, fy + 3), (cx, fy + 5), (cx + 4, fy + 2), (cx + 5, fy - 1)], fill=flame_rgb, outline=flame_dark)
        draw.polygon([(cx, fy - 2), (cx - 2, fy + 1), (cx, fy + 3), (cx + 2, fy)], fill=(255, 221, 82))


def draw_pixel_room_divider(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    divider_style: str = "screen",
    wood_rgb: RGB = (121, 76, 43),
    cloth_rgb: RGB = (170, 94, 82),
) -> None:
    """Draw a reusable inn privacy divider."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -48)
    cloth_dark = _shade(cloth_rgb, -42)
    cloth_light = _shade(cloth_rgb, 35)
    panel_count = 3
    panel_w = max(8, (x1 - x0 + 1) // panel_count)
    if str(divider_style) == "curtain":
        draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y0 + 5), fill=wood_rgb, outline=wood_dark)
        for px in range(x0 + 4, x1 - 3, 7):
            draw.line((px, y0 + 6, px - 2, y1 - 2), fill=cloth_dark)
            draw.line((px + 2, y0 + 6, px, y1 - 2), fill=cloth_rgb)
        draw.rectangle((x0 + 2, y0 + 6, x1 - 2, y1 - 2), outline=cloth_dark)
        return
    for index in range(panel_count):
        px0 = x0 + 2 + index * panel_w
        px1 = min(x1 - 2, px0 + panel_w - 3)
        top = y0 + 2 + (1 if index == 1 else 0)
        bottom = y1 - 2 - (1 if index == 1 else 0)
        draw.rectangle((px0, top, px1, bottom), fill=cloth_rgb, outline=wood_dark)
        draw.line((px0 + 2, top + 3, px1 - 2, top + 3), fill=cloth_light)
        draw.line((px0 + 2, bottom - 3, px1 - 2, bottom - 3), fill=cloth_dark)
        draw.line((px0, top, px0, bottom), fill=wood_rgb)
        draw.line((px1, top, px1, bottom), fill=wood_dark)


def draw_pixel_mug(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    ceramic_rgb: RGB = (218, 205, 165),
    drink_rgb: RGB = (125, 77, 42),
) -> None:
    """Draw a reusable one-tile mug."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    ceramic_dark = _shade(ceramic_rgb, -48)
    body_rgb = _shade(ceramic_rgb, -4)
    ceramic_light = _shade(ceramic_rgb, 48)
    cx = (x0 + x1) // 2
    draw.rectangle((cx - 4, y0 + 4, cx + 2, y1 - 3), fill=body_rgb, outline=ceramic_dark)
    draw.rectangle((cx - 3, y0 + 3, cx + 1, y0 + 5), fill=drink_rgb, outline=ceramic_dark)
    draw.point((cx - 2, y0 + 4), fill=_shade(drink_rgb, 34))
    draw.line((cx + 3, y0 + 6, cx + 5, y0 + 6), fill=ceramic_dark)
    draw.line((cx + 5, y0 + 6, cx + 5, y0 + 10), fill=ceramic_dark)
    draw.line((cx + 3, y0 + 10, cx + 5, y0 + 10), fill=ceramic_dark)
    draw.line((cx + 3, y0 + 7, cx + 3, y0 + 9), fill=body_rgb)
    draw.line((cx - 2, y0 + 6, cx - 2, y1 - 5), fill=ceramic_light)
    draw.line((cx - 3, y1 - 4, cx + 1, y1 - 4), fill=_shade(ceramic_rgb, -28))


def draw_pixel_bottle(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    glass_rgb: RGB = (57, 130, 91),
) -> None:
    """Draw a reusable one-tile bottle."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=3)
    glass_dark = _shade(glass_rgb, -52)
    glass_light = _shade(glass_rgb, 52)
    cx = (x0 + x1) // 2
    draw.rectangle((cx - 2, y0 + 1, cx + 2, y0 + 6), fill=glass_rgb, outline=glass_dark)
    draw.rectangle((cx - 4, y0 + 6, cx + 4, y1 - 2), fill=glass_rgb, outline=glass_dark)
    draw.rectangle((cx - 3, y0, cx + 3, y0 + 2), fill=(83, 56, 37), outline=(49, 36, 27))
    draw.point((cx - 2, y0 + 7), fill=glass_light)
    draw.point((cx - 1, y0 + 8), fill=glass_light)


def draw_pixel_plate(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    ceramic_rgb: RGB = (224, 218, 198),
    food_rgb: RGB | None = None,
) -> None:
    """Draw a reusable one-tile plate."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    ceramic_dark = _shade(ceramic_rgb, -44)
    ceramic_light = _shade(ceramic_rgb, 34)
    _draw_row_crown(
        draw,
        x0=x0,
        y0=y0,
        rows=(
            (4, 4, 7),
            (5, 2, 9),
            (6, 1, 10),
            (7, 1, 10),
            (8, 2, 9),
            (9, 4, 7),
        ),
        fill=ceramic_rgb,
        outline=ceramic_dark,
    )
    draw.line((x0 + 3, y0 + 6, x1 - 3, y0 + 6), fill=ceramic_light)
    draw.line((x0 + 3, y0 + 8, x1 - 3, y0 + 8), fill=_shade(ceramic_rgb, -20))
    food = food_rgb or (196, 121, 61)
    _draw_row_crown(
        draw,
        x0=x0,
        y0=y0,
        rows=((6, 5, 6), (7, 4, 7), (8, 5, 6)),
        fill=food,
        outline=_shade(food, -42),
    )


def draw_pixel_bowl(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    ceramic_rgb: RGB = (207, 188, 148),
    contents_rgb: RGB = (190, 119, 62),
) -> None:
    """Draw a reusable one-tile bowl."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    ceramic_dark = _shade(ceramic_rgb, -46)
    ceramic_light = _shade(ceramic_rgb, 38)
    _draw_row_crown(
        draw,
        x0=x0,
        y0=y0,
        rows=((3, 4, 7), (4, 2, 9), (5, 1, 10), (6, 2, 9)),
        fill=contents_rgb,
        outline=ceramic_dark,
    )
    draw.polygon(
        [(x0 + 1, y0 + 6), (x1 - 1, y0 + 6), (x1 - 3, y1 - 3), (x0 + 3, y1 - 3)],
        fill=ceramic_rgb,
        outline=ceramic_dark,
    )
    draw.line((x0 + 3, y0 + 7, x1 - 3, y0 + 7), fill=ceramic_light)
    draw.line((x0 + 4, y1 - 4, x1 - 4, y1 - 4), fill=_shade(ceramic_rgb, -22))
    draw.point((x0 + 5, y0 + 4), fill=_shade(contents_rgb, 36))


def draw_pixel_candle(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wax_rgb: RGB = (238, 222, 171),
    flame_state: str = "lit",
    flame_rgb: RGB = (243, 153, 55),
) -> None:
    """Draw a reusable one-tile candle."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    wax_dark = _shade(wax_rgb, -46)
    cx = (x0 + x1) // 2
    draw.rectangle((cx - 3, y0 + 4, cx + 3, y1 - 3), fill=wax_rgb, outline=wax_dark)
    draw.line((cx - 2, y0 + 4, cx + 2, y0 + 4), fill=_shade(wax_rgb, 34))
    draw.line((cx, y0 + 2, cx, y0 + 5), fill=(54, 42, 35))
    draw.point((cx + 1, y0 + 6), fill=_shade(wax_rgb, 42))
    draw.line((cx - 2, y0 + 11, cx - 1, y0 + 12), fill=wax_dark)
    draw.rectangle((cx - 5, y1 - 3, cx + 5, y1), fill=(111, 79, 51), outline=(66, 50, 37))
    if str(flame_state) == "lit":
        flame_dark = _shade(flame_rgb, -46)
        draw.polygon([(cx, y0), (cx - 3, y0 + 4), (cx, y0 + 7), (cx + 3, y0 + 4)], fill=flame_rgb, outline=flame_dark)
        draw.point((cx, y0 + 3), fill=(255, 229, 92))
    else:
        draw.point((cx - 1, y0 + 2), fill=(70, 70, 68))


def draw_pixel_rock(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a reusable one-tile field rock."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    outline = (72, 79, 76)
    mid = (125, 132, 125)
    light = (158, 164, 153)
    dark = (91, 98, 94)
    shadow = (58, 64, 62)
    silhouette = [
        (x0 + 2, y0 + 9),
        (x0 + 4, y0 + 5),
        (x0 + 8, y0 + 3),
        (x0 + 12, y0 + 5),
        (x1 - 1, y0 + 8),
        (x1 - 3, y1 - 2),
        (x0 + 8, y1),
        (x0 + 3, y1 - 2),
    ]
    draw.polygon(silhouette, fill=mid, outline=outline)
    draw.polygon([(x0 + 4, y0 + 8), (x0 + 8, y0 + 4), (x0 + 11, y0 + 6), (x0 + 8, y0 + 9)], fill=light)
    draw.polygon([(x0 + 8, y0 + 9), (x1 - 2, y0 + 8), (x1 - 4, y1 - 3), (x0 + 9, y1 - 1)], fill=dark)
    draw.line((x0 + 4, y1 - 2, x1 - 4, y1 - 2), fill=shadow)
    draw.point((x0 + 6, y0 + 6), fill=(188, 192, 181))


def draw_pixel_boulder(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (113, 112, 104),
) -> None:
    """Draw a reusable one-tile cave boulder."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -58)
    dark = _shade(stone_rgb, -34)
    light = _shade(stone_rgb, 42)
    shadow = _shade(stone_rgb, -72)
    silhouette = [
        (x0 + 1, y0 + 11),
        (x0 + 3, y0 + 6),
        (x0 + 7, y0 + 3),
        (x0 + 12, y0 + 4),
        (x1 - 1, y0 + 8),
        (x1 - 2, y1 - 3),
        (x0 + 11, y1),
        (x0 + 4, y1 - 1),
    ]
    draw.polygon(silhouette, fill=stone_rgb, outline=outline)
    draw.polygon([(x0 + 4, y0 + 7), (x0 + 7, y0 + 4), (x0 + 10, y0 + 5), (x0 + 8, y0 + 9)], fill=light)
    draw.polygon([(x0 + 9, y0 + 9), (x1 - 2, y0 + 9), (x1 - 3, y1 - 4), (x0 + 10, y1 - 1)], fill=dark)
    draw.line((x0 + 4, y1 - 1, x1 - 4, y1 - 1), fill=shadow)
    draw.point((x0 + 5, y0 + 6), fill=_shade(light, 24))


def draw_pixel_ore_vein(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (103, 101, 94),
    ore_rgb: RGB = (218, 171, 71),
) -> None:
    """Draw a reusable one-tile ore-bearing rock patch."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    outline = _shade(stone_rgb, -54)
    dark = _shade(stone_rgb, -30)
    light = _shade(stone_rgb, 28)
    ore_dark = _shade(ore_rgb, -55)
    ore_light = _shade(ore_rgb, 38)
    rock = [
        (x0 + 1, y0 + 10),
        (x0 + 4, y0 + 6),
        (x0 + 8, y0 + 4),
        (x1 - 2, y0 + 7),
        (x1 - 1, y1 - 3),
        (x0 + 8, y1),
        (x0 + 3, y1 - 2),
    ]
    draw.polygon(rock, fill=stone_rgb, outline=outline)
    draw.polygon([(x0 + 4, y0 + 8), (x0 + 8, y0 + 5), (x0 + 11, y0 + 7), (x0 + 8, y0 + 9)], fill=light)
    draw.polygon([(x0 + 8, y0 + 9), (x1 - 1, y0 + 8), (x1 - 3, y1 - 3), (x0 + 9, y1 - 1)], fill=dark)
    draw.line((x0 + 5, y0 + 11, x0 + 8, y0 + 9, x0 + 11, y0 + 12), fill=ore_dark)
    draw.line((x0 + 6, y0 + 11, x0 + 8, y0 + 10, x0 + 10, y0 + 12), fill=ore_rgb)
    draw.point((x0 + 5, y0 + 12), fill=ore_light)
    draw.point((x1 - 4, y0 + 9), fill=ore_rgb)
    draw.point((x1 - 5, y1 - 4), fill=ore_dark)


def draw_pixel_crystal_cluster(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    crystal_rgb: RGB = (111, 189, 213),
) -> None:
    """Draw a reusable one-tile crystal cluster."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(crystal_rgb, -78)
    dark = _shade(crystal_rgb, -42)
    light = _shade(crystal_rgb, 48)
    base = (78, 76, 72)
    draw.polygon([(x0 + 2, y1 - 4), (x0 + 6, y1 - 7), (x1 - 4, y1 - 7), (x1 - 1, y1 - 3), (x1 - 5, y1), (x0 + 3, y1)], fill=base, outline=(50, 50, 48))
    crystals = [
        [(x0 + 4, y1 - 4), (x0 + 6, y0 + 5), (x0 + 10, y1 - 4), (x0 + 7, y1 - 1)],
        [(x0 + 8, y1 - 4), (x0 + 12, y0 + 2), (x1 - 2, y1 - 5), (x0 + 12, y1)],
        [(x0 + 1, y1 - 4), (x0 + 4, y0 + 9), (x0 + 7, y1 - 5), (x0 + 4, y1)],
    ]
    for polygon in crystals:
        draw.polygon(polygon, fill=crystal_rgb, outline=outline)
    draw.polygon([(x0 + 12, y0 + 2), (x1 - 2, y1 - 5), (x0 + 12, y1)], fill=dark)
    draw.line((x0 + 6, y0 + 6, x0 + 7, y1 - 3), fill=light)
    draw.line((x0 + 12, y0 + 4, x0 + 12, y1 - 3), fill=light)
    draw.point((x0 + 5, y0 + 9), fill=light)


def draw_pixel_stalagmite(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (126, 122, 111),
) -> None:
    """Draw a reusable one-tile floor stalagmite cluster."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -58)
    dark = _shade(stone_rgb, -32)
    light = _shade(stone_rgb, 38)
    draw.ellipse((x0 + 2, y1 - 6, x1 - 2, y1), fill=(65, 64, 61), outline=None)
    spikes = [
        [(x0 + 3, y1 - 2), (x0 + 6, y0 + 7), (x0 + 9, y1 - 2)],
        [(x0 + 7, y1 - 2), (x0 + 10, y0 + 3), (x1 - 2, y1 - 2)],
        [(x0 + 1, y1 - 2), (x0 + 3, y0 + 11), (x0 + 6, y1 - 2)],
    ]
    for polygon in spikes:
        draw.polygon(polygon, fill=stone_rgb, outline=outline)
    draw.polygon([(x0 + 10, y0 + 3), (x1 - 2, y1 - 2), (x0 + 10, y1 - 2)], fill=dark)
    draw.line((x0 + 6, y0 + 8, x0 + 6, y1 - 3), fill=light)
    draw.point((x0 + 10, y0 + 6), fill=_shade(light, 16))


def draw_pixel_torch(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (112, 72, 42),
    flame_rgb: RGB = (244, 153, 45),
) -> None:
    """Draw a reusable one-tile mine torch."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -44)
    flame_dark = _shade(flame_rgb, -46)
    flame_light = (255, 226, 86)
    cx = (x0 + x1) // 2
    draw.line((cx - 1, y0 + 9, cx - 1, y1 - 1), fill=wood_dark)
    draw.line((cx, y0 + 9, cx, y1 - 1), fill=wood_rgb)
    draw.line((cx + 1, y0 + 10, cx + 1, y1 - 1), fill=wood_dark)
    draw.rectangle((cx - 2, y0 + 11, cx + 2, y0 + 12), fill=wood_dark)
    draw.point((cx, y0 + 13), fill=_shade(wood_rgb, 32))
    draw.polygon(
        [(cx, y0 + 1), (cx - 2, y0 + 5), (cx - 1, y0 + 10), (cx, y0 + 12), (cx + 1, y0 + 9), (cx + 2, y0 + 5)],
        fill=flame_rgb,
        outline=flame_dark,
    )
    draw.polygon([(cx, y0 + 4), (cx - 1, y0 + 7), (cx, y0 + 10), (cx + 1, y0 + 7)], fill=flame_light)
    draw.point((cx, y0 + 2), fill=(255, 238, 124))


def draw_pixel_stone_column(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (126, 123, 114),
) -> None:
    """Draw a reusable one-tile dungeon stone column."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -58)
    dark = _shade(stone_rgb, -34)
    light = _shade(stone_rgb, 38)
    cx = (x0 + x1) // 2
    draw.ellipse((cx - 6, y1 - 4, cx + 6, y1), fill=(62, 61, 58), outline=None)
    draw.rectangle((cx - 4, y0 + 5, cx + 4, y1 - 4), fill=stone_rgb, outline=outline)
    draw.rectangle((cx - 6, y0 + 2, cx + 6, y0 + 6), fill=_shade(stone_rgb, 12), outline=outline)
    draw.rectangle((cx - 6, y1 - 8, cx + 6, y1 - 4), fill=_shade(stone_rgb, 6), outline=outline)
    draw.line((cx - 2, y0 + 7, cx - 2, y1 - 6), fill=light)
    draw.line((cx + 3, y0 + 7, cx + 3, y1 - 6), fill=dark)
    for yy in (y0 + 8, y0 + 13):
        draw.line((cx - 4, yy, cx + 4, yy), fill=_shade(stone_rgb, -18))
    draw.point((cx - 3, y0 + 4), fill=_shade(light, 18))


def draw_pixel_archway(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (120, 118, 110),
    shadow_rgb: RGB = (37, 36, 39),
) -> None:
    """Draw a reusable two-tile dungeon archway."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -58)
    dark = _shade(stone_rgb, -34)
    light = _shade(stone_rgb, 36)
    cx = (x0 + x1) // 2
    draw.rectangle((x0 + 2, y0 + 6, x0 + 8, y1 - 1), fill=stone_rgb, outline=outline)
    draw.rectangle((x1 - 8, y0 + 6, x1 - 2, y1 - 1), fill=stone_rgb, outline=outline)
    draw.pieslice((cx - 15, y0 - 2, cx + 15, y0 + 28), 180, 360, fill=stone_rgb, outline=outline)
    draw.rectangle((cx - 15, y0 + 12, cx + 15, y0 + 17), fill=stone_rgb, outline=outline)
    opening_top = min(y0 + 11, y1 - 1)
    draw.pieslice((cx - 9, y0 + 4, cx + 9, y0 + 24), 180, 360, fill=shadow_rgb, outline=_shade(shadow_rgb, -18))
    draw.rectangle((cx - 9, opening_top, cx + 9, y1 - 2), fill=shadow_rgb, outline=_shade(shadow_rgb, -18))
    for px in (x0 + 5, x0 + 8, x1 - 8, x1 - 5):
        draw.line((px, y0 + 8, px, y1 - 3), fill=dark)
    draw.line((x0 + 5, y0 + 8, cx - 4, y0 + 3, x1 - 5, y0 + 8), fill=light)
    draw.point((x0 + 4, y0 + 10), fill=_shade(light, 14))
    draw.point((x1 - 5, y0 + 12), fill=dark)


def draw_pixel_sealed_door(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (134, 87, 45),
    seal_rgb: RGB = (94, 74, 55),
    orientation: str = "horizontal",
) -> None:
    """Draw a reusable sealed dungeon door."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -56)
    dark = _shade(stone_rgb, -30)
    light = _shade(stone_rgb, 34)
    seal_dark = _shade(seal_rgb, -46)
    iron = (60, 58, 54)
    if str(orientation) == "vertical":
        panel = (x0 + 3, y0 + 1, x1 - 3, y1 - 1)
        draw.rectangle(panel, fill=stone_rgb, outline=outline)
        for yy in range(panel[1] + 4, panel[3] - 1, 5):
            draw.line((panel[0] + 2, yy, panel[2] - 2, yy), fill=dark)
        cx = (x0 + x1) // 2
        draw.line((cx, panel[1] + 2, cx, panel[3] - 2), fill=seal_dark, width=2)
        draw.rectangle((panel[0] + 2, panel[1] + 3, panel[2] - 2, panel[1] + 5), fill=iron, outline=seal_dark)
        draw.rectangle((panel[0] + 2, panel[3] - 5, panel[2] - 2, panel[3] - 3), fill=iron, outline=seal_dark)
        draw.rectangle((cx - 3, (y0 + y1) // 2 - 3, cx + 3, (y0 + y1) // 2 + 3), fill=seal_rgb, outline=seal_dark)
        draw.point((cx + 4, (y0 + y1) // 2), fill=light)
    else:
        panel = (x0 + 1, y0 + 2, x1 - 1, y1 - 2)
        draw.rectangle(panel, fill=stone_rgb, outline=outline)
        for xx in range(panel[0] + 5, panel[2] - 1, 5):
            draw.line((xx, panel[1] + 1, xx, panel[3] - 1), fill=dark)
        cy = (y0 + y1) // 2
        draw.rectangle((panel[0] + 2, cy - 3, panel[2] - 2, cy - 1), fill=iron, outline=seal_dark)
        draw.rectangle((panel[0] + 2, cy + 2, panel[2] - 2, cy + 4), fill=iron, outline=seal_dark)
        draw.line((x0 + 4, cy, x1 - 4, cy), fill=seal_dark, width=2)
        draw.rectangle(((x0 + x1) // 2 - 3, cy - 3, (x0 + x1) // 2 + 3, cy + 3), fill=seal_rgb, outline=seal_dark)
    draw.line((panel[0] + 2, panel[1] + 1, panel[2] - 2, panel[1] + 1), fill=light)


def draw_pixel_floor_switch(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    plate_rgb: RGB = (142, 126, 88),
    switch_state: str = "raised",
) -> None:
    """Draw a reusable one-tile floor switch or pressure plate."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    outline = _shade(plate_rgb, -56)
    dark = _shade(plate_rgb, -28)
    light = _shade(plate_rgb, 38)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    base = (x0 + 1, y0 + 2, x1 - 1, y1 - 2)
    if str(switch_state) == "pressed":
        draw.rectangle(base, fill=dark, outline=outline)
        draw.rectangle((base[0] + 2, base[1] + 2, base[2] - 2, base[3] - 2), fill=_shade(plate_rgb, -8), outline=_shade(plate_rgb, -32))
        draw.line((base[0] + 3, base[3] - 1, base[2] - 3, base[3] - 1), fill=light)
    else:
        draw.rectangle(base, fill=plate_rgb, outline=outline)
        draw.line((base[0] + 2, base[1] + 1, base[2] - 2, base[1] + 1), fill=light)
        draw.line((base[0] + 2, base[3] - 1, base[2] - 2, base[3] - 1), fill=dark)
        draw.rectangle((base[0] + 4, base[1] + 3, base[2] - 4, base[3] - 3), fill=_shade(plate_rgb, 16), outline=outline)
    draw.rectangle((cx - 1, cy - 1, cx + 1, cy + 1), fill=_shade(plate_rgb, 35), outline=outline)


def draw_pixel_brazier(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    metal_rgb: RGB = (93, 91, 88),
    flame_rgb: RGB = (238, 126, 45),
    fire_state: str = "lit",
) -> None:
    """Draw a reusable one-tile dungeon brazier."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    metal_dark = _shade(metal_rgb, -48)
    metal_light = _shade(metal_rgb, 38)
    cx = (x0 + x1) // 2
    draw.ellipse((cx - 6, y1 - 5, cx + 6, y1), fill=(54, 52, 49), outline=None)
    stem_top = min(y0 + 9, y1 - 5)
    stem_bottom = max(stem_top, y1 - 4)
    draw.rectangle((cx - 2, stem_top, cx + 2, stem_bottom), fill=metal_rgb, outline=metal_dark)
    draw.rectangle((cx - 5, y1 - 4, cx + 5, y1 - 2), fill=metal_rgb, outline=metal_dark)
    bowl = (cx - 7, y0 + 7, cx + 7, y0 + 14)
    draw.ellipse(bowl, fill=metal_rgb, outline=metal_dark)
    draw.rectangle((cx - 6, y0 + 9, cx + 6, y0 + 13), fill=metal_rgb, outline=metal_dark)
    draw.line((cx - 5, y0 + 8, cx + 5, y0 + 8), fill=metal_light)
    if str(fire_state) == "lit":
        flame_dark = _shade(flame_rgb, -48)
        flame_light = (255, 224, 82)
        draw.polygon([(cx, y0 + 1), (cx - 4, y0 + 7), (cx - 1, y0 + 11), (cx + 3, y0 + 7)], fill=flame_rgb, outline=flame_dark)
        draw.polygon([(cx + 1, y0 + 4), (cx - 1, y0 + 8), (cx + 1, y0 + 10), (cx + 2, y0 + 7)], fill=flame_light)
    else:
        draw.line((cx - 4, y0 + 8, cx + 4, y0 + 8), fill=(55, 54, 52))


def draw_pixel_broken_wall(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (132, 132, 124),
    break_style: str = "cracked",
) -> None:
    """Draw a reusable two-tile broken wall segment."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -58)
    dark = _shade(stone_rgb, -34)
    light = _shade(stone_rgb, 34)
    wall = (x0 + 1, y0 + 2, x1 - 1, y1 - 2)
    draw.rectangle(wall, fill=stone_rgb, outline=outline)
    mortar_y = (wall[1] + wall[3]) // 2
    draw.line((wall[0] + 1, mortar_y, wall[2] - 1, mortar_y), fill=dark)
    for xx in range(wall[0] + 5, wall[2] - 3, 7):
        draw.line((xx, wall[1] + 1, xx, mortar_y - 1), fill=dark)
        draw.line((xx + 3, mortar_y + 1, xx + 3, wall[3] - 1), fill=dark)
    crack = [(x0 + 8, wall[1] + 1), (x0 + 12, wall[1] + 4), (x0 + 10, wall[1] + 7), (x0 + 17, wall[3] - 1)]
    draw.line(crack, fill=(46, 45, 43), width=1)
    if str(break_style) == "gap":
        gap_x = (x0 + x1) // 2
        rubble = [(gap_x - 6, wall[1] + 1), (gap_x + 1, wall[1] + 4), (gap_x + 1, wall[3] - 1), (gap_x - 8, wall[3])]
        draw.polygon(rubble, fill=(45, 44, 42), outline=outline)
        draw.rectangle((gap_x + 2, wall[1] + 2, gap_x + 8, wall[3]), fill=dark, outline=outline)
    draw.line((wall[0] + 2, wall[1] + 1, wall[2] - 3, wall[1] + 1), fill=light)
    draw.rectangle((wall[0] + 1, wall[3] - 1, wall[2] - 1, wall[3] + 1), fill=_shade(stone_rgb, -20), outline=outline)


def draw_pixel_rubble(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (116, 113, 105),
) -> None:
    """Draw a reusable one-tile rubble pile."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -58)
    dark = _shade(stone_rgb, -36)
    light = _shade(stone_rgb, 36)
    draw.ellipse((x0 + 2, y1 - 5, x1 - 2, y1), fill=(57, 55, 52), outline=None)
    stones = (
        (x0 + 2, y1 - 7, x0 + 7, y1 - 2, _shade(stone_rgb, -8)),
        (x0 + 6, y1 - 10, x0 + 12, y1 - 3, stone_rgb),
        (x0 + 10, y1 - 7, x1 - 1, y1 - 1, dark),
        (x0 + 4, y1 - 13, x0 + 9, y1 - 8, _shade(stone_rgb, 8)),
    )
    for sx0, sy0, sx1, sy1, color in stones:
        draw.polygon([(sx0, sy1), ((sx0 + sx1) // 2, sy0), (sx1, sy0 + 2), (sx1 - 1, sy1)], fill=color, outline=outline)
    draw.point((x0 + 6, y1 - 11), fill=light)
    draw.point((x1 - 4, y1 - 5), fill=outline)


def draw_pixel_magic_circle(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    rune_rgb: RGB = (92, 214, 232),
    glow_rgb: RGB = (84, 79, 186),
) -> None:
    """Draw a reusable flat magic-circle floor marking."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    rx = max(6, (x1 - x0) // 2 - 1)
    ry = max(5, (y1 - y0) // 2 - 2)
    rune_dark = _shade(rune_rgb, -56)
    glow = _shade(glow_rgb, 8)
    draw.ellipse((cx - rx - 2, cy - ry - 2, cx + rx + 2, cy + ry + 2), outline=_shade(glow_rgb, -28))
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=glow, width=2)
    draw.ellipse((cx - rx + 4, cy - ry + 3, cx + rx - 4, cy + ry - 3), outline=rune_rgb, width=1)
    draw.line((cx - rx + 2, cy, cx + rx - 2, cy), fill=_shade(glow_rgb, 20))
    draw.line((cx, cy - ry + 1, cx, cy + ry - 1), fill=_shade(glow_rgb, 20))
    draw.line((cx, cy - ry + 3, cx + rx - 4, cy + ry - 2), fill=rune_dark)
    draw.line((cx + rx - 4, cy + ry - 2, cx - rx + 4, cy + ry - 2), fill=rune_dark)
    draw.line((cx - rx + 4, cy + ry - 2, cx, cy - ry + 3), fill=rune_dark)
    for px, py in ((cx, cy - ry), (cx + rx - 2, cy), (cx, cy + ry), (cx - rx + 2, cy)):
        draw.rectangle((px - 1, py - 1, px + 1, py + 1), fill=rune_rgb)


def draw_pixel_ladder(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (128, 86, 45),
    orientation: str = "vertical",
) -> None:
    """Draw a reusable one-tile ladder segment."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -45)
    wood_light = _shade(wood_rgb, 32)
    if str(orientation) == "horizontal":
        rail_y0, rail_y1 = y0 + 5, y1 - 5
        draw.line((x0 + 1, rail_y0, x1 - 1, rail_y0), fill=wood_dark, width=2)
        draw.line((x0 + 1, rail_y1, x1 - 1, rail_y1), fill=wood_dark, width=2)
        for x in range(x0 + 4, x1 - 1, 4):
            draw.line((x, rail_y0 - 1, x, rail_y1 + 1), fill=wood_rgb)
            draw.point((x, rail_y0 - 1), fill=wood_light)
    else:
        rail_x0, rail_x1 = x0 + 5, x1 - 5
        draw.line((rail_x0, y0 + 1, rail_x0, y1 - 1), fill=wood_dark, width=2)
        draw.line((rail_x1, y0 + 1, rail_x1, y1 - 1), fill=wood_dark, width=2)
        for y in range(y0 + 4, y1 - 1, 4):
            draw.line((rail_x0 - 1, y, rail_x1 + 1, y), fill=wood_rgb)
            draw.point((rail_x0 - 1, y), fill=wood_light)


def draw_pixel_rail_track(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (114, 75, 42),
    rail_rgb: RGB = (73, 76, 75),
    track_shape: str = "horizontal",
) -> None:
    """Draw a reusable one-tile mine rail track."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    wood_dark = _shade(wood_rgb, -42)
    rail_light = _shade(rail_rgb, 34)

    def horizontal() -> None:
        for x in range(x0 + 2, x1, 5):
            draw.line((x, y0 + 3, x, y1 - 3), fill=wood_dark)
            draw.line((x + 1, y0 + 4, x + 1, y1 - 4), fill=wood_rgb)
        draw.line((x0 + 1, y0 + 5, x1 - 1, y0 + 5), fill=rail_rgb, width=2)
        draw.line((x0 + 1, y1 - 5, x1 - 1, y1 - 5), fill=rail_rgb, width=2)
        draw.line((x0 + 1, y0 + 4, x1 - 1, y0 + 4), fill=rail_light)

    def vertical() -> None:
        for y in range(y0 + 2, y1, 5):
            draw.line((x0 + 3, y, x1 - 3, y), fill=wood_dark)
            draw.line((x0 + 4, y + 1, x1 - 4, y + 1), fill=wood_rgb)
        draw.line((x0 + 5, y0 + 1, x0 + 5, y1 - 1), fill=rail_rgb, width=2)
        draw.line((x1 - 5, y0 + 1, x1 - 5, y1 - 1), fill=rail_rgb, width=2)
        draw.line((x0 + 4, y0 + 1, x0 + 4, y1 - 1), fill=rail_light)

    shape = str(track_shape)
    if shape == "vertical":
        vertical()
    elif shape == "crossing":
        horizontal()
        vertical()
    elif shape == "corner":
        draw.arc((x0 + 3, y0 + 3, x1 + 7, y1 + 7), 180, 270, fill=rail_rgb, width=2)
        draw.arc((x0 + 7, y0 + 7, x1 + 3, y1 + 3), 180, 270, fill=rail_rgb, width=2)
        draw.line((x0 + 3, y1 - 5, x0 + 9, y1 - 5), fill=wood_rgb)
        draw.line((x0 + 5, y0 + 7, x0 + 5, y1 - 3), fill=wood_dark)
        draw.line((x0 + 7, y0 + 5, x1 - 2, y0 + 5), fill=wood_dark)
    else:
        horizontal()


def draw_pixel_mine_cart(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    body_rgb: RGB = (93, 93, 92),
    ore_rgb: RGB = (198, 145, 58),
    orientation: str = "horizontal",
) -> None:
    """Draw a reusable two-tile mine cart."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    body_dark = _shade(body_rgb, -52)
    body_light = _shade(body_rgb, 38)
    ore_dark = _shade(ore_rgb, -46)
    if str(orientation) == "vertical":
        draw.rectangle((x0 + 5, y0 + 2, x1 - 5, y1 - 2), fill=body_rgb, outline=body_dark)
        draw.polygon([(x0 + 6, y0 + 2), (x1 - 6, y0 + 2), (x1 - 9, y0 + 7), (x0 + 9, y0 + 7)], fill=body_light, outline=body_dark)
        for y in (y0 + 5, y1 - 6):
            draw.ellipse((x0 + 3, y - 2, x0 + 7, y + 2), fill=body_dark)
            draw.ellipse((x1 - 7, y - 2, x1 - 3, y + 2), fill=body_dark)
        for px, py in ((x0 + 10, y0 + 6), (x1 - 10, y0 + 8), (x0 + 12, y0 + 11)):
            draw.rectangle((px - 1, py - 1, px + 2, py + 1), fill=ore_rgb, outline=ore_dark)
        return
    draw.rectangle((x0 + 3, y0 + 7, x1 - 3, y1 - 5), fill=body_rgb, outline=body_dark)
    draw.polygon([(x0 + 5, y0 + 4), (x1 - 5, y0 + 4), (x1 - 8, y0 + 9), (x0 + 8, y0 + 9)], fill=body_light, outline=body_dark)
    draw.line((x0 + 5, y0 + 8, x1 - 5, y0 + 8), fill=_shade(body_rgb, 18))
    for x in (x0 + 8, x1 - 12):
        draw.ellipse((x, y1 - 6, x + 5, y1 - 1), fill=body_dark, outline=(32, 34, 35))
        draw.point((x + 2, y1 - 4), fill=body_light)
    for px, py in ((x0 + 11, y0 + 6), (x0 + 16, y0 + 5), (x1 - 15, y0 + 6), (x1 - 10, y0 + 7)):
        draw.rectangle((px - 1, py - 1, px + 2, py + 1), fill=ore_rgb, outline=ore_dark)


def draw_pixel_wood_support(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    wood_rgb: RGB = (128, 78, 43),
) -> None:
    """Draw a reusable two-tile mine support frame."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -50)
    wood_light = _shade(wood_rgb, 32)
    draw.rectangle((x0 + 3, y0 + 4, x1 - 3, y0 + 9), fill=wood_rgb, outline=wood_dark)
    draw.rectangle((x0 + 5, y0 + 8, x0 + 10, y1 - 1), fill=wood_rgb, outline=wood_dark)
    draw.rectangle((x1 - 10, y0 + 8, x1 - 5, y1 - 1), fill=wood_rgb, outline=wood_dark)
    draw.line((x0 + 6, y0 + 11, x1 - 7, y1 - 3), fill=wood_dark, width=2)
    draw.line((x1 - 7, y0 + 11, x0 + 7, y1 - 3), fill=wood_dark, width=2)
    draw.line((x0 + 5, y0 + 5, x1 - 5, y0 + 5), fill=wood_light)
    draw.point((x0 + 7, y0 + 12), fill=wood_light)
    draw.point((x1 - 8, y0 + 13), fill=wood_light)


def draw_pixel_stairs(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (112, 112, 106),
    stair_direction: str = "down",
) -> None:
    """Draw a reusable two-tile stone stair flight."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    outline = _shade(stone_rgb, -55)
    dark = _shade(stone_rgb, -32)
    light = _shade(stone_rgb, 34)
    draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), fill=stone_rgb, outline=outline)
    direction = str(stair_direction)
    if direction in {"left", "right"}:
        step_count = 5
        for index in range(step_count):
            x = x0 + 4 + index * max(3, (x1 - x0 - 8) // step_count)
            if direction == "left":
                x = x1 - 4 - index * max(3, (x1 - x0 - 8) // step_count)
            draw.line((x, y0 + 4, x, y1 - 4), fill=outline)
            draw.line((x + (1 if direction == "right" else -1), y0 + 5, x + (1 if direction == "right" else -1), y1 - 5), fill=light)
    else:
        step_count = 6
        step_h = max(3, (y1 - y0 - 6) // step_count)
        for index in range(step_count):
            y = y0 + 4 + index * step_h
            draw.rectangle((x0 + 3 + index, y, x1 - 3 - index, y + step_h - 1), fill=_shade(stone_rgb, -index * 4), outline=outline)
            draw.line((x0 + 4 + index, y, x1 - 4 - index, y), fill=light)
        if direction == "up":
            draw.rectangle((x0 + 5, y0 + 4, x1 - 5, y0 + 7), fill=dark, outline=outline)


def draw_pixel_cave_entrance(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (92, 92, 86),
    shadow_rgb: RGB = (31, 31, 33),
) -> None:
    """Draw a reusable four-by-three-tile cave entrance."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    outline = _shade(stone_rgb, -50)
    dark = _shade(stone_rgb, -26)
    light = _shade(stone_rgb, 34)
    hill = [
        (x0 + 2, y1 - 2),
        (x0 + 4, y0 + 20),
        (x0 + 14, y0 + 8),
        (x0 + 26, y0 + 2),
        (x0 + 42, y0 + 4),
        (x1 - 5, y0 + 18),
        (x1 - 2, y1 - 2),
    ]
    draw.polygon(hill, fill=stone_rgb, outline=outline)
    draw.polygon([(x0 + 5, y1 - 2), (x0 + 9, y0 + 23), (x0 + 17, y0 + 10), (x0 + 26, y0 + 3), (x0 + 20, y1 - 2)], fill=light)
    draw.polygon([(x0 + 42, y0 + 4), (x1 - 6, y0 + 18), (x1 - 3, y1 - 2), (x0 + 36, y1 - 2)], fill=dark)
    mouth_x0, mouth_x1 = x0 + 20, x1 - 18
    mouth_y0, mouth_y1 = y0 + 19, y1 - 2
    draw.pieslice((mouth_x0, mouth_y0 - 13, mouth_x1, mouth_y0 + 17), 180, 360, fill=shadow_rgb, outline=outline)
    draw.rectangle((mouth_x0, mouth_y0, mouth_x1, mouth_y1), fill=shadow_rgb, outline=outline)
    draw.rectangle((mouth_x0 + 5, mouth_y0 + 5, mouth_x1 - 5, mouth_y1), fill=_shade(shadow_rgb, -12))
    for px, py, color in (
        (x0 + 12, y0 + 22, dark),
        (x0 + 17, y0 + 15, outline),
        (x0 + 48, y0 + 22, light),
        (x0 + 50, y0 + 34, outline),
        (x0 + 9, y1 - 10, outline),
    ):
        draw.rectangle((px, py, px + 3, py + 2), fill=color)
    draw.line((x0 + 7, y1 - 2, x1 - 7, y1 - 2), fill=_shade(stone_rgb, -62))


def draw_pixel_fountain(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a reusable two-tile village fountain."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    draw.rectangle((x0 + 3, y0 + 5, x1 - 3, y1 - 3), fill=(142, 146, 145), outline=(76, 84, 92))
    draw.rectangle((x0 + 6, y0 + 8, x1 - 6, y1 - 7), fill=(72, 158, 198), outline=(43, 101, 142))
    draw.rectangle((x0 + 12, y0 + 2, x0 + 20, y0 + 11), fill=(163, 166, 158), outline=(86, 91, 88))
    draw.point((x0 + 11, y0 + 7), fill=(189, 235, 246))
    draw.point((x0 + 21, y0 + 7), fill=(189, 235, 246))


def draw_pixel_sign(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a reusable one-tile wooden sign."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    draw.rectangle((x0 + 7, y0 + 12, x0 + 10, y1 - 2), fill=(90, 60, 35))
    draw.rectangle((x0 + 3, y0 + 5, x1 - 3, y0 + 13), fill=(205, 159, 87), outline=(92, 65, 39))
    draw.line((x0 + 6, y0 + 9, x1 - 6, y0 + 9), fill=(92, 65, 39))


def draw_pixel_bridge(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    orientation: str = "horizontal",
) -> None:
    """Draw a reusable bridge segment over path or water tiles."""

    rect = _base_rect(tile_xywh)
    draw.rectangle(rect, fill=(151, 101, 54), outline=(90, 62, 36))
    x0, y0, x1, y1 = rect
    if str(orientation) == "vertical":
        for y in range(y0 + 4, y1, 7):
            draw.line((x0 + 2, y, x1 - 2, y), fill=(202, 151, 83))
        draw.line((x0 + 3, y0, x0 + 3, y1), fill=(80, 54, 34))
        draw.line((x1 - 3, y0, x1 - 3, y1), fill=(80, 54, 34))
    else:
        for x in range(x0 + 4, x1, 7):
            draw.line((x, y0 + 2, x, y1 - 2), fill=(202, 151, 83))
        draw.line((x0, y0 + 3, x1, y0 + 3), fill=(80, 54, 34))
        draw.line((x0, y1 - 3, x1, y1 - 3), fill=(80, 54, 34))


def draw_pixel_fence(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    orientation: str = "horizontal",
) -> None:
    """Draw a reusable wooden fence segment."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    wood = (140, 91, 48)
    post = (104, 68, 38)
    if str(orientation) == "vertical":
        draw.line((x0 + 7, y0 + 1, x0 + 7, y1 - 1), fill=wood, width=2)
        draw.line((x0 + 11, y0 + 1, x0 + 11, y1 - 1), fill=wood, width=2)
        for y in range(y0 + 3, y1, 8):
            draw.rectangle((x0 + 4, y, x0 + 14, y + 2), fill=post)
    else:
        draw.line((x0 + 1, y0 + 7, x1 - 1, y0 + 7), fill=wood, width=2)
        draw.line((x0 + 1, y0 + 11, x1 - 1, y0 + 11), fill=wood, width=2)
        for x in range(x0 + 3, x1, 8):
            draw.rectangle((x, y0 + 4, x + 2, y0 + 14), fill=post)


def draw_pixel_iron_fence(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    orientation: str = "horizontal",
) -> None:
    """Draw a reusable iron fence segment."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    iron = (78, 84, 91)
    iron_dark = (42, 45, 49)
    cap = (124, 130, 136)
    if str(orientation) == "vertical":
        draw.line((x0 + 8, y0 + 1, x0 + 8, y1 - 1), fill=iron_dark)
        draw.line((x0 + 12, y0 + 1, x0 + 12, y1 - 1), fill=iron)
        for y in range(y0 + 2, y1, 5):
            draw.line((x0 + 3, y, x0 + 14, y), fill=iron_dark)
            draw.point((x0 + 2, y), fill=cap)
    else:
        draw.line((x0 + 1, y0 + 8, x1 - 1, y0 + 8), fill=iron_dark)
        draw.line((x0 + 1, y0 + 12, x1 - 1, y0 + 12), fill=iron)
        for x in range(x0 + 2, x1, 5):
            draw.line((x, y0 + 3, x, y0 + 14), fill=iron_dark)
            draw.point((x, y0 + 2), fill=cap)


def draw_pixel_cemetery_gate(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    orientation: str = "horizontal",
) -> None:
    """Draw a reusable cemetery gate tile."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    stone = (124, 128, 124)
    stone_dark = (68, 72, 70)
    iron = (45, 48, 52)
    if str(orientation) == "vertical":
        draw.rectangle((x0 + 4, y0 + 2, x1 - 1, y0 + 4), fill=stone, outline=stone_dark)
        draw.rectangle((x0 + 4, y1 - 4, x1 - 1, y1 - 2), fill=stone, outline=stone_dark)
        draw.line((x0 + 8, y0 + 4, x0 + 8, y1 - 4), fill=iron)
        draw.line((x0 + 12, y0 + 4, x0 + 12, y1 - 4), fill=iron)
        draw.line((x0 + 7, y0 + 8, x1 - 2, y0 + 8), fill=iron)
        draw.line((x0 + 7, y0 + 11, x1 - 2, y0 + 11), fill=iron)
    else:
        draw.rectangle((x0 + 2, y0 + 4, x0 + 4, y1 - 1), fill=stone, outline=stone_dark)
        draw.rectangle((x1 - 4, y0 + 4, x1 - 2, y1 - 1), fill=stone, outline=stone_dark)
        draw.line((x0 + 4, y0 + 8, x1 - 4, y0 + 8), fill=iron)
        draw.line((x0 + 4, y0 + 12, x1 - 4, y0 + 12), fill=iron)
        draw.line((x0 + 8, y0 + 7, x0 + 8, y1 - 2), fill=iron)
        draw.line((x0 + 11, y0 + 7, x0 + 11, y1 - 2), fill=iron)


def draw_pixel_farm_gate(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    orientation: str = "horizontal",
) -> None:
    """Draw a reusable farm gate tile."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    wood = (142, 92, 48)
    dark = (79, 52, 33)
    light = (185, 128, 65)
    shadow = (101, 68, 39)
    if str(orientation) == "vertical":
        draw.rectangle((x0 + 5, y0 + 1, x0 + 10, y0 + 6), fill=dark)
        draw.rectangle((x0 + 5, y1 - 7, x0 + 10, y1 - 2), fill=dark)
        draw.point((x0 + 6, y0 + 2), fill=light)
        draw.point((x0 + 6, y1 - 6), fill=light)
        draw.rectangle((x0 + 7, y0 + 5, x0 + 9, y1 - 6), fill=wood)
        draw.rectangle((x0 + 11, y0 + 5, x0 + 13, y1 - 6), fill=wood)
        draw.line((x0 + 7, y0 + 8, x0 + 13, y0 + 8), fill=light, width=2)
        draw.line((x0 + 7, y1 - 9, x0 + 13, y1 - 9), fill=shadow, width=2)
        draw.line((x0 + 13, y0 + 7, x0 + 7, y1 - 8), fill=wood, width=2)
    else:
        draw.rectangle((x0 + 1, y0 + 4, x0 + 6, y1 - 1), fill=dark)
        draw.rectangle((x1 - 6, y0 + 4, x1 - 1, y1 - 1), fill=dark)
        draw.point((x0 + 2, y0 + 5), fill=light)
        draw.point((x1 - 5, y0 + 5), fill=light)
        draw.rectangle((x0 + 5, y0 + 7, x1 - 6, y0 + 9), fill=light)
        draw.rectangle((x0 + 5, y0 + 11, x1 - 6, y0 + 13), fill=wood)
        draw.line((x0 + 6, y0 + 13, x1 - 7, y0 + 8), fill=shadow, width=2)
        draw.line((x0 + 6, y0 + 12, x1 - 7, y0 + 7), fill=wood)


def draw_pixel_scarecrow(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    shirt_rgb: RGB = (182, 84, 60),
    hat_rgb: RGB = (154, 103, 50),
) -> None:
    """Draw a reusable 1x2-tile scarecrow."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    wood = (105, 68, 38)
    straw = (226, 187, 82)
    shirt_dark = _shade(shirt_rgb, -42)
    draw.line((x0 + 8, y0 + 11, x0 + 8, y1 - 2), fill=wood, width=2)
    draw.line((x0 + 2, y0 + 18, x1 - 2, y0 + 18), fill=wood, width=2)
    draw.rectangle((x0 + 4, y0 + 16, x1 - 4, y0 + 25), fill=shirt_rgb, outline=shirt_dark)
    draw.line((x0 + 2, y0 + 20, x0 + 6, y0 + 24), fill=straw)
    draw.line((x1 - 2, y0 + 20, x1 - 6, y0 + 24), fill=straw)
    draw.rectangle((x0 + 5, y0 + 7, x0 + 11, y0 + 13), fill=(224, 179, 98), outline=(100, 69, 42))
    draw.line((x0 + 3, y0 + 7, x0 + 13, y0 + 7), fill=hat_rgb, width=2)
    draw.polygon([(x0 + 5, y0 + 7), (x0 + 8, y0 + 3), (x0 + 11, y0 + 7)], fill=hat_rgb, outline=(98, 66, 38))
    draw.point((x0 + 6, y0 + 10), fill=(64, 47, 36))
    draw.point((x0 + 10, y0 + 10), fill=(64, 47, 36))


def draw_pixel_barrel(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    barrel_rgb: RGB = (151, 86, 45),
    band_rgb: RGB = (82, 59, 42),
) -> None:
    """Draw a reusable one-tile wooden barrel."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=2)
    barrel_dark = _shade(barrel_rgb, -44)
    barrel_light = _shade(barrel_rgb, 36)
    draw.ellipse((x0 + 1, y0, x1 - 1, y0 + 5), fill=barrel_rgb, outline=barrel_dark)
    draw.rectangle((x0 + 1, y0 + 3, x1 - 1, y1 - 3), fill=barrel_rgb, outline=barrel_dark)
    draw.ellipse((x0 + 1, y1 - 6, x1 - 1, y1 - 1), fill=_shade(barrel_rgb, -8), outline=barrel_dark)
    draw.line((x0 + 2, y0 + 6, x1 - 2, y0 + 6), fill=band_rgb)
    draw.line((x0 + 2, y1 - 6, x1 - 2, y1 - 6), fill=band_rgb)
    draw.line((x0 + 5, y0 + 4, x0 + 4, y1 - 5), fill=barrel_light)
    draw.line((x1 - 5, y0 + 4, x1 - 4, y1 - 5), fill=barrel_dark)
    draw.point((x0 + 6, y0 + 2), fill=barrel_light)


def draw_pixel_bench(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    orientation: str = "horizontal",
    wood_rgb: RGB = (138, 83, 47),
) -> None:
    """Draw a reusable path-side bench."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    wood_dark = _shade(wood_rgb, -48)
    wood_light = _shade(wood_rgb, 32)
    wood_mid = _shade(wood_rgb, 10)
    metal = (61, 65, 61)
    metal_dark = (38, 42, 39)
    if str(orientation) == "vertical":
        cx = (x0 + x1) // 2
        draw.rectangle((cx - 6, y0 + 6, cx - 4, y1 - 7), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((cx + 1, y0 + 4, cx + 5, y1 - 5), fill=wood_mid, outline=wood_dark)
        draw.line((cx - 5, y0 + 8, cx - 5, y1 - 9), fill=wood_light)
        draw.line((cx + 2, y0 + 7, cx + 2, y1 - 7), fill=wood_light)
        for yy in (y0 + 9, y1 - 10):
            draw.rectangle((cx - 8, yy, cx - 7, yy + 4), fill=metal)
            draw.rectangle((cx + 6, yy, cx + 7, yy + 4), fill=metal)
    else:
        cy = (y0 + y1) // 2
        for xx in (x0 + 8, x1 - 10):
            draw.rectangle((xx, cy + 3, xx + 2, cy + 8), fill=metal_dark)
        draw.rectangle((x0 + 6, cy - 6, x1 - 6, cy - 3), fill=wood_rgb, outline=wood_dark)
        draw.rectangle((x0 + 4, cy + 1, x1 - 4, cy + 5), fill=wood_mid, outline=wood_dark)
        draw.line((x0 + 8, cy - 5, x1 - 8, cy - 5), fill=wood_light)
        draw.line((x0 + 7, cy + 2, x1 - 7, cy + 2), fill=wood_light)
        draw.rectangle((x0 + 6, cy + 5, x0 + 8, cy + 8), fill=metal)
        draw.rectangle((x1 - 9, cy + 5, x1 - 7, cy + 8), fill=metal)


def draw_pixel_lamp_post(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    glow_rgb: RGB = (248, 210, 104),
    metal_rgb: RGB = (61, 70, 75),
) -> None:
    """Draw a reusable one-by-two-tile lamp post."""

    x0, y0, _, y1 = _base_rect(tile_xywh)
    metal_dark = _shade(metal_rgb, -34)
    glow_dark = _shade(glow_rgb, -56)
    draw.ellipse((x0 + 3, y1 - 4, x0 + 13, y1 - 1), fill=metal_dark)
    draw.rectangle((x0 + 7, y0 + 10, x0 + 9, y1 - 3), fill=metal_rgb, outline=metal_dark)
    draw.line((x0 + 5, y0 + 14, x0 + 11, y0 + 14), fill=metal_dark)
    draw.rectangle((x0 + 5, y0 + 5, x0 + 11, y0 + 12), fill=glow_rgb, outline=glow_dark)
    draw.polygon([(x0 + 4, y0 + 5), (x0 + 8, y0 + 1), (x0 + 12, y0 + 5)], fill=metal_rgb, outline=metal_dark)
    draw.point((x0 + 7, y0 + 7), fill=(255, 240, 153))
    draw.point((x0 + 9, y0 + 8), fill=(255, 240, 153))


def draw_pixel_notice_board(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    board_rgb: RGB = (177, 119, 62),
    paper_rgb: RGB = (236, 214, 154),
) -> None:
    """Draw a reusable two-tile village notice board."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    board_dark = _shade(board_rgb, -48)
    board_light = _shade(board_rgb, 30)
    paper_dark = _shade(paper_rgb, -40)
    left_post = x0 + 5
    right_post = x1 - 7
    draw.rectangle((left_post, y0 + 2, left_post + 2, y1), fill=board_dark)
    draw.rectangle((right_post, y0 + 2, right_post + 2, y1), fill=board_dark)
    draw.rectangle((x0 + 2, y0 + 1, x1 - 2, y0 + 13), fill=board_rgb, outline=board_dark)
    draw.rectangle((x0 + 1, y0, x1 - 1, y0 + 2), fill=board_dark)
    draw.line((x0 + 4, y0 + 4, x1 - 4, y0 + 4), fill=board_light)
    draw.line((x0 + 4, y0 + 12, x1 - 4, y0 + 12), fill=_shade(board_rgb, -26))
    draw.rectangle((x0 + 6, y0 + 5, x0 + 13, y0 + 11), fill=paper_rgb, outline=paper_dark)
    draw.rectangle((x0 + 17, y0 + 4, x0 + 25, y0 + 10), fill=_shade(paper_rgb, 10), outline=paper_dark)
    draw.point((x0 + 8, y0 + 6), fill=(176, 55, 48))
    draw.point((x0 + 20, y0 + 5), fill=(176, 55, 48))
    draw.line((x0 + 8, y0 + 8, x0 + 12, y0 + 8), fill=paper_dark)
    draw.line((x0 + 18, y0 + 7, x0 + 23, y0 + 7), fill=paper_dark)
    draw.line((x0 + 8, y0 + 10, x0 + 11, y0 + 10), fill=paper_dark)


def draw_pixel_cart(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    facing: str = "right",
    body_rgb: RGB = (151, 88, 45),
) -> None:
    """Draw a reusable two-tile wooden hand cart."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    body_dark = _shade(body_rgb, -48)
    body_light = _shade(body_rgb, 32)
    side_rgb = _shade(body_rgb, -10)
    wheel = (55, 48, 42)
    if str(facing) == "left":
        draw.line((x0 + 5, y0 + 7, x0, y0 + 4), fill=body_dark, width=2)
        draw.line((x0 + 5, y1 - 7, x0, y1 - 4), fill=body_dark, width=2)
        bed = [(x0 + 5, y0 + 5), (x1 - 5, y0 + 3), (x1 - 8, y1 - 5), (x0 + 7, y1 - 3)]
        rail_start = x0 + 8
        rail_end = x1 - 8
    else:
        draw.line((x1 - 5, y0 + 7, x1, y0 + 4), fill=body_dark, width=2)
        draw.line((x1 - 5, y1 - 7, x1, y1 - 4), fill=body_dark, width=2)
        bed = [(x0 + 5, y0 + 3), (x1 - 5, y0 + 5), (x1 - 7, y1 - 3), (x0 + 8, y1 - 5)]
        rail_start = x0 + 8
        rail_end = x1 - 8
    for wx in (x0 + 9, x1 - 13):
        draw.ellipse((wx, y1 - 8, wx + 5, y1 - 3), fill=wheel, outline=(28, 25, 24))
        draw.point((wx + 2, y1 - 6), fill=(126, 108, 78))
    draw.polygon(bed, fill=body_rgb, outline=body_dark)
    draw.line((rail_start, y0 + 6, rail_end, y0 + 7), fill=body_light)
    draw.line((rail_start, y1 - 6, rail_end, y1 - 5), fill=body_dark)
    for px in range(x0 + 11, x1 - 10, 6):
        draw.line((px, y0 + 5, px - 1, y1 - 5), fill=side_rgb)
    draw.rectangle((x0 + 9, y0 + 7, x1 - 10, y0 + 9), fill=_shade(body_rgb, 12))


def draw_pixel_market_stall(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    canopy_rgb: RGB = (194, 72, 66),
    wood_rgb: RGB = (129, 80, 45),
    goods_type: str = "fruit",
) -> None:
    """Draw a reusable 3x2-tile village market stall."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    canopy_dark = _shade(canopy_rgb, -48)
    canopy_light = _shade(canopy_rgb, 36)
    wood_dark = _shade(wood_rgb, -44)
    cloth = (237, 218, 154)
    draw.rectangle((x0 + 5, y0 + 14, x1 - 5, y1 - 6), fill=wood_rgb, outline=wood_dark)
    draw.rectangle((x0 + 8, y1 - 13, x0 + 11, y1 - 2), fill=wood_dark)
    draw.rectangle((x1 - 11, y1 - 13, x1 - 8, y1 - 2), fill=wood_dark)
    draw.polygon(
        [(x0 + 2, y0 + 10), (x0 + 9, y0 + 2), (x1 - 9, y0 + 2), (x1 - 2, y0 + 10), (x1 - 5, y0 + 16), (x0 + 5, y0 + 16)],
        fill=canopy_rgb,
        outline=canopy_dark,
    )
    for stripe_x in range(x0 + 9, x1 - 10, 12):
        draw.rectangle((stripe_x, y0 + 3, stripe_x + 5, y0 + 15), fill=canopy_light)
    draw.line((x0 + 7, y0 + 16, x1 - 7, y0 + 16), fill=canopy_dark)
    draw.rectangle((x0 + 8, y0 + 18, x1 - 8, y0 + 23), fill=cloth, outline=(121, 91, 52))
    goods = str(goods_type)
    if goods == "cloth":
        for gx, color in ((x0 + 12, (85, 137, 190)), (x0 + 23, (188, 91, 148)), (x0 + 34, (84, 153, 100))):
            draw.rectangle((gx, y0 + 18, gx + 7, y0 + 22), fill=color, outline=_shade(color, -40))
    elif goods == "crates":
        for gx in (x0 + 11, x0 + 25, x0 + 36):
            draw.rectangle((gx, y0 + 18, gx + 8, y0 + 23), fill=(174, 110, 52), outline=(95, 63, 36))
            draw.line((gx + 1, y0 + 19, gx + 7, y0 + 22), fill=(118, 77, 43))
    else:
        for gx, color in ((x0 + 12, (224, 62, 58)), (x0 + 20, (232, 181, 54)), (x0 + 28, (58, 154, 74)), (x0 + 36, (224, 62, 58))):
            draw.point((gx, y0 + 19), fill=color)
            draw.point((gx + 1, y0 + 19), fill=color)
            draw.point((gx, y0 + 20), fill=_shade(color, -30))


def draw_pixel_wagon(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    facing: str = "right",
    body_rgb: RGB = (143, 84, 43),
    cover_rgb: RGB | None = None,
) -> None:
    """Draw a reusable 3x2-tile wagon."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    body_outline = _shade(body_rgb, -92)
    body_dark = _shade(body_rgb, -58)
    body_light = _shade(body_rgb, 52)
    body_mid = _shade(body_rgb, 18)
    wheel = (39, 34, 31)
    wheel_outline = (18, 16, 15)
    hub = _shade(body_rgb, 72)
    draw.rectangle((x0 + 7, y1 - 8, x1 - 9, y1 - 5), fill=(46, 58, 38))
    draw.rectangle((x0 + 5, y0 + 14, x1 - 7, y1 - 7), fill=body_outline)
    draw.rectangle((x0 + 7, y0 + 16, x1 - 9, y1 - 9), fill=body_rgb, outline=body_dark)
    draw.line((x0 + 8, y0 + 17, x1 - 10, y0 + 17), fill=body_light)
    draw.line((x0 + 8, y0 + 20, x1 - 10, y0 + 20), fill=body_mid)
    draw.line((x0 + 9, y1 - 11, x1 - 11, y1 - 11), fill=body_dark)
    for px in (x0 + 15, x0 + 23, x0 + 31):
        draw.line((px, y0 + 16, px, y1 - 9), fill=body_dark)
        draw.point((px + 1, y0 + 18), fill=body_light)
    if cover_rgb is not None:
        cover_outline = _shade(cover_rgb, -78)
        cover_dark = _shade(cover_rgb, -42)
        cover_light = _shade(cover_rgb, 34)
        draw.arc((x0 + 7, y0 + 3, x1 - 9, y0 + 28), start=180, end=360, fill=cover_outline, width=3)
        draw.arc((x0 + 9, y0 + 5, x1 - 11, y0 + 26), start=180, end=360, fill=cover_light, width=1)
        draw.rectangle((x0 + 9, y0 + 12, x1 - 11, y0 + 18), fill=cover_rgb, outline=cover_outline)
        draw.line((x0 + 11, y0 + 13, x1 - 13, y0 + 13), fill=cover_light)
        draw.line((x0 + 18, y0 + 8, x0 + 18, y0 + 18), fill=cover_dark)
        draw.line((x1 - 20, y0 + 8, x1 - 20, y0 + 18), fill=cover_dark)
    for wx in (x0 + 10, x0 + 30):
        draw.ellipse((wx - 1, y1 - 13, wx + 8, y1 - 4), fill=wheel_outline)
        draw.ellipse((wx, y1 - 12, wx + 7, y1 - 5), fill=wheel, outline=wheel_outline)
        draw.rectangle((wx + 2, y1 - 10, wx + 5, y1 - 7), fill=hub, outline=body_outline)
    if str(facing) == "left":
        draw.line((x0 + 6, y0 + 20, x0, y0 + 15), fill=body_outline, width=2)
        draw.line((x0 + 6, y1 - 12, x0, y1 - 8), fill=body_outline, width=2)
    else:
        draw.line((x1 - 8, y0 + 20, x1 - 1, y0 + 15), fill=body_outline, width=2)
        draw.line((x1 - 8, y1 - 12, x1 - 1, y1 - 8), fill=body_outline, width=2)


def draw_pixel_statue(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    material_rgb: RGB = (145, 151, 148),
) -> None:
    """Draw a reusable 2x2-tile pedestal statue."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    dark = _shade(material_rgb, -56)
    light = _shade(material_rgb, 34)
    mid = _shade(material_rgb, -20)
    cx = (x0 + x1) // 2
    draw.rectangle((x0 + 5, y1 - 10, x1 - 5, y1 - 2), fill=material_rgb, outline=dark)
    draw.rectangle((x0 + 8, y1 - 16, x1 - 8, y1 - 9), fill=_shade(material_rgb, 10), outline=dark)
    draw.rectangle((cx - 4, y0 + 12, cx + 4, y1 - 16), fill=material_rgb, outline=dark)
    draw.ellipse((cx - 5, y0 + 5, cx + 5, y0 + 15), fill=_shade(material_rgb, 12), outline=dark)
    draw.line((cx - 4, y0 + 19, cx - 10, y0 + 23), fill=dark, width=2)
    draw.line((cx + 4, y0 + 19, cx + 10, y0 + 15), fill=dark, width=2)
    draw.point((cx - 2, y0 + 8), fill=light)
    draw.line((x0 + 9, y1 - 6, x1 - 9, y1 - 6), fill=mid)


def draw_pixel_gazebo(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    roof_rgb: RGB = (154, 82, 74),
    wood_rgb: RGB = (123, 83, 48),
) -> None:
    """Draw a reusable 3x3-tile village gazebo."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    roof_dark = _shade(roof_rgb, -46)
    roof_light = _shade(roof_rgb, 30)
    wood_dark = _shade(wood_rgb, -44)
    floor = (178, 142, 86)
    draw.ellipse((x0 + 8, y1 - 17, x1 - 8, y1 - 4), fill=floor, outline=(111, 82, 51))
    for px in (x0 + 12, x1 - 14):
        draw.rectangle((px, y0 + 18, px + 3, y1 - 9), fill=wood_rgb, outline=wood_dark)
    for px in (x0 + 21, x1 - 23):
        draw.rectangle((px, y0 + 22, px + 2, y1 - 10), fill=_shade(wood_rgb, 10), outline=wood_dark)
    draw.polygon(
        [(x0 + 4, y0 + 18), ((x0 + x1) // 2, y0 + 4), (x1 - 4, y0 + 18), (x1 - 11, y0 + 25), (x0 + 11, y0 + 25)],
        fill=roof_rgb,
        outline=roof_dark,
    )
    draw.line((x0 + 13, y0 + 18, x1 - 13, y0 + 18), fill=roof_light)
    draw.line(((x0 + x1) // 2, y0 + 5, (x0 + x1) // 2, y0 + 24), fill=roof_dark)
    draw.rectangle((x0 + 14, y1 - 15, x1 - 14, y1 - 11), fill=wood_rgb, outline=wood_dark)


def draw_pixel_woodpile(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    log_rgb: RGB = (135, 82, 45),
    stack_variant: str = "low",
) -> None:
    """Draw a reusable stacked woodpile."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    bark = _shade(log_rgb, -44)
    cut = (211, 165, 94)
    rows = 2 if str(stack_variant) == "low" else 3
    base_y = y1 - 6
    for row in range(rows):
        y = base_y - row * 5
        offset = 3 if row % 2 else 0
        for x in range(x0 + 3 + offset, x1 - 7, 10):
            draw.rectangle((x, y, x + 9, y + 4), fill=log_rgb, outline=bark)
            draw.ellipse((x, y, x + 4, y + 4), fill=cut, outline=bark)
            draw.point((x + 2, y + 2), fill=bark)
    draw.line((x0 + 2, y1 - 1, x1 - 2, y1 - 1), fill=(55, 104, 56))


def draw_pixel_pond(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    shape: str = "round",
    water_rgb: RGB = (55, 133, 188),
    rim_rgb: RGB = (74, 129, 74),
) -> None:
    """Draw a reusable variable-size irregular village pond."""

    x0, y0, x1, y1 = _base_rect(tile_xywh, inset=1)
    water_dark = _shade(water_rgb, -42)
    water_light = _shade(water_rgb, 45)
    rim_dark = _shade(rim_rgb, -35)
    shape_id = str(shape)
    if shape_id == "kidney":
        points = [
            (x0 + 8, y0 + 8),
            (x0 + 21, y0 + 3),
            (x1 - 10, y0 + 7),
            (x1 - 5, y0 + 17),
            (x1 - 16, y1 - 9),
            (x0 + 25, y1 - 5),
            (x0 + 12, y1 - 11),
            (x0 + 5, y0 + 21),
        ]
    elif shape_id == "long":
        points = [
            (x0 + 6, y0 + 10),
            (x0 + 19, y0 + 4),
            (x1 - 15, y0 + 5),
            (x1 - 5, y0 + 13),
            (x1 - 8, y1 - 10),
            (x1 - 24, y1 - 5),
            (x0 + 17, y1 - 6),
            (x0 + 5, y1 - 14),
        ]
    else:
        points = [
            (x0 + 10, y0 + 6),
            (x0 + 24, y0 + 3),
            (x1 - 9, y0 + 12),
            (x1 - 5, y0 + 25),
            (x1 - 17, y1 - 7),
            (x0 + 18, y1 - 5),
            (x0 + 5, y1 - 17),
            (x0 + 4, y0 + 18),
        ]
    expanded = [(max(x0, px - 1), max(y0, py - 1)) for px, py in points]
    draw.polygon(expanded, fill=rim_rgb, outline=rim_dark)
    draw.polygon(points, fill=water_rgb, outline=water_dark)
    draw.line((x0 + 13, y0 + 15, x0 + 26, y0 + 15), fill=water_light)
    draw.line((x1 - 27, y1 - 14, x1 - 13, y1 - 14), fill=_shade(water_light, -15))
    for sx, sy in ((x0 + 7, y0 + 12), (x1 - 8, y1 - 12), (x0 + 15, y1 - 8)):
        draw.rectangle((sx, sy, sx + 2, sy + 2), fill=(112, 119, 99), outline=(70, 82, 69))
    for rx, ry in ((x0 + 8, y1 - 10), (x1 - 12, y0 + 12)):
        draw.line((rx, ry, rx, ry - 5), fill=(39, 114, 68))
        draw.point((rx - 1, ry - 3), fill=(68, 153, 79))
        draw.point((rx + 1, ry - 4), fill=(68, 153, 79))


def draw_pixel_winter_overlay(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    target: str,
    snow_rgb: RGB = (239, 246, 248),
    shadow_rgb: RGB = (178, 202, 215),
    coverage: float = 0.5,
    style: str = "patchy",
) -> None:
    """Draw a reusable pixel snow/ice overlay for top-down RPG objects."""

    if float(coverage) <= 0.0:
        return
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    target_id = str(target)
    style_id = str(style)
    snow = tuple(int(v) for v in snow_rgb)
    shadow = tuple(int(v) for v in shadow_rgb)
    heavy = float(coverage) >= 0.66
    medium = float(coverage) >= 0.42

    if target_id in {"tree", "oak", "maple", "fruit_tree"}:
        rows = [(y0 + 5, x0 + 5, x0 + 10), (y0 + 10, x0 + 2, x0 + 13)]
        if medium:
            rows.append((y0 + 15, x0 + 3, x0 + 12))
        if heavy:
            rows.extend([(y0 + 19, x0 + 4, x0 + 11), (y0 + 3, x0 + 7, x0 + 9)])
        for yy, sx, ex in rows:
            draw.line((sx, yy, ex, yy), fill=snow)
            draw.point((sx, yy + 1), fill=shadow)
            draw.point((ex, yy + 1), fill=shadow)
        return
    if target_id == "pine":
        for yy, sx, ex in ((y0 + 8, x0 + 4, x0 + 12), (y0 + 17, x0 + 2, x0 + 14), (y0 + 24, x0 + 3, x0 + 13)):
            draw.line((sx, yy, ex, yy), fill=snow)
            if medium:
                draw.point((sx + 2, yy + 1), fill=shadow)
                draw.point((ex - 2, yy + 1), fill=shadow)
        if heavy:
            draw.point((x0 + 8, y0 + 2), fill=snow)
        return
    if target_id in {"building", "market_stall", "gazebo"}:
        y = y0 + (6 if target_id == "building" else 10 if target_id == "market_stall" else 18)
        draw.line((x0 + 4, y, x1 - 4, y), fill=snow, width=2)
        draw.line((x0 + 7, y + 2, x1 - 7, y + 2), fill=shadow)
        if style_id in {"ridge", "full_cap"} or medium:
            draw.line((x0 + 10, y - 4, x1 - 10, y - 4), fill=snow)
        if heavy:
            for px in range(x0 + 8, x1 - 8, 13):
                draw.rectangle((px, y + 1, px + 3, y + 4), fill=snow)
        return
    if target_id == "wagon":
        draw.line((x0 + 9, y0 + 12, x1 - 12, y0 + 12), fill=snow, width=2)
        draw.line((x0 + 12, y0 + 15, x1 - 15, y0 + 15), fill=shadow)
        if heavy:
            draw.line((x0 + 9, y0 + 19, x1 - 11, y0 + 19), fill=snow)
        return
    if target_id == "bench":
        cy = (y0 + y1) // 2
        if (x1 - x0) >= (y1 - y0):
            draw.line((x0 + 5, cy - 5, x1 - 5, cy - 5), fill=snow)
            if medium:
                draw.line((x0 + 6, cy + 1, x1 - 6, cy + 1), fill=snow)
        else:
            draw.line((x0 + 4, y0 + 5, x0 + 4, y1 - 5), fill=snow)
            if medium:
                draw.line((x0 + 10, y0 + 5, x0 + 10, y1 - 5), fill=snow)
        return
    if target_id == "lamp_post":
        draw.line((x0 + 5, y0 + 5, x0 + 11, y0 + 5), fill=snow)
        if medium:
            draw.point((x0 + 8, y0 + 1), fill=snow)
        return
    if target_id == "pond":
        draw.line((x0 + 11, y0 + 13, x1 - 11, y0 + 13), fill=snow)
        draw.line((x0 + 15, y0 + 19, x1 - 18, y0 + 19), fill=(160, 210, 229))
        if heavy:
            draw.line((x0 + 11, y1 - 13, x1 - 13, y1 - 13), fill=snow)
        return


def draw_pixel_autumn_overlay(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    target: str,
    leaf_rgb: RGB = (177, 126, 55),
    shadow_rgb: RGB = (104, 82, 47),
    accent_rgb: RGB = (154, 78, 51),
    coverage: float = 0.25,
    style: str = "scattered",
) -> None:
    """Draw a reusable subtle fallen-leaf overlay for top-down RPG objects."""

    if float(coverage) <= 0.0:
        return
    x0, y0, x1, y1 = _base_rect(tile_xywh)
    target_id = str(target)
    leaf = tuple(int(v) for v in leaf_rgb)
    shadow = tuple(int(v) for v in shadow_rgb)
    accent = tuple(int(v) for v in accent_rgb)
    medium = float(coverage) >= 0.24
    heavy = float(coverage) >= 0.38

    def leaf_dot(px: int, py: int, fill: RGB) -> None:
        draw.point((px, py), fill=fill)
        draw.point((px + 1, py), fill=fill)
        draw.point((px, py + 1), fill=shadow)

    if target_id in {"tree", "oak", "maple", "fruit_tree", "pine"}:
        points = [
            (x0 + 4, y1 - 5, leaf),
            (x0 + 10, y1 - 3, accent),
        ]
        if medium:
            points.extend([(x0 + 2, y1 - 8, accent), (x0 + 13, y1 - 7, leaf)])
        if heavy:
            points.extend([(x0 + 7, y1 - 9, leaf), (x0 + 12, y1 - 2, accent)])
        for px, py, color in points:
            leaf_dot(px, py, color)
        return

    if target_id == "bench":
        cy = (y0 + y1) // 2
        leaf_dot(x0 + 6, cy + 5, leaf)
        if medium:
            leaf_dot(x1 - 8, cy + 4, accent)
        return

    if target_id == "market_stall":
        leaf_dot(x0 + 9, y1 - 8, leaf)
        if medium:
            leaf_dot(x1 - 13, y1 - 7, accent)
        return

    if target_id == "wagon":
        leaf_dot(x0 + 8, y1 - 7, leaf)
        if medium:
            leaf_dot(x1 - 14, y1 - 8, accent)
        return

    if target_id == "gazebo":
        leaf_dot(x0 + 10, y1 - 10, leaf)
        leaf_dot(x1 - 13, y1 - 9, accent)
        if heavy:
            leaf_dot((x0 + x1) // 2, y1 - 7, leaf)
        return

    if target_id == "pond":
        leaf_dot(x0 + 13, y0 + 15, leaf)
        if medium:
            leaf_dot(x1 - 17, y1 - 15, accent)
        return

    if str(style) == "ground_edge":
        leaf_dot(x0 + 4, y1 - 5, leaf)
        if medium:
            leaf_dot(x1 - 8, y1 - 4, accent)


def _draw_chicken(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    facing: str,
    body_rgb: RGB,
    accent_rgb: RGB,
) -> None:
    x0, y0, _, _ = rect
    outline = (93, 68, 43)
    beak = (230, 168, 48)
    leg = (170, 105, 38)
    comb = accent_rgb
    if facing == "left":
        body_box = (x0 + 5, y0 + 8, x0 + 12, y0 + 13)
        head_box = (x0 + 2, y0 + 5, x0 + 7, y0 + 10)
        beak_points = [(x0 + 2, y0 + 7), (x0, y0 + 8), (x0 + 2, y0 + 9)]
        eye = (x0 + 4, y0 + 7)
        tail = [(x0 + 12, y0 + 8), (x0 + 15, y0 + 6), (x0 + 13, y0 + 11)]
        comb_pts = [(x0 + 3, y0 + 5), (x0 + 4, y0 + 3), (x0 + 5, y0 + 5)]
    else:
        body_box = (x0 + 4, y0 + 8, x0 + 11, y0 + 13)
        head_box = (x0 + 9, y0 + 5, x0 + 14, y0 + 10)
        beak_points = [(x0 + 14, y0 + 7), (x0 + 16, y0 + 8), (x0 + 14, y0 + 9)]
        eye = (x0 + 12, y0 + 7)
        tail = [(x0 + 4, y0 + 8), (x0 + 1, y0 + 6), (x0 + 3, y0 + 11)]
        comb_pts = [(x0 + 11, y0 + 5), (x0 + 12, y0 + 3), (x0 + 13, y0 + 5)]
    draw.polygon(tail, fill=_shade(body_rgb, -20), outline=outline)
    draw.ellipse(body_box, fill=body_rgb, outline=outline)
    draw.ellipse(head_box, fill=_shade(body_rgb, 18), outline=outline)
    draw.polygon(beak_points, fill=beak, outline=(137, 92, 31))
    draw.polygon(comb_pts, fill=comb)
    draw.point(eye, fill=(26, 22, 20))
    for lx in (x0 + 6, x0 + 10):
        draw.line((lx, y0 + 13, lx, y0 + 15), fill=leg)
        draw.line((lx - 1, y0 + 15, lx + 1, y0 + 15), fill=leg)


def _draw_pig(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    facing: str,
    body_rgb: RGB,
) -> None:
    x0, y0, _, _ = rect
    outline = _shade(body_rgb, -70)
    dark = _shade(body_rgb, -35)
    if facing == "left":
        head = (x0 + 1, y0 + 5, x0 + 7, y0 + 11)
        snout = (x0, y0 + 7, x0 + 4, y0 + 10)
        eye = (x0 + 4, y0 + 7)
        ear = [(x0 + 5, y0 + 5), (x0 + 7, y0 + 2), (x0 + 8, y0 + 6)]
        tail_x = x0 + 14
    else:
        head = (x0 + 9, y0 + 5, x0 + 15, y0 + 11)
        snout = (x0 + 12, y0 + 7, x0 + 16, y0 + 10)
        eye = (x0 + 12, y0 + 7)
        ear = [(x0 + 10, y0 + 5), (x0 + 8, y0 + 2), (x0 + 7, y0 + 6)]
        tail_x = x0 + 1
    draw.ellipse((x0 + 3, y0 + 6, x0 + 13, y0 + 13), fill=body_rgb, outline=outline)
    draw.polygon(ear, fill=dark, outline=outline)
    draw.ellipse(head, fill=_shade(body_rgb, 12), outline=outline)
    draw.ellipse(snout, fill=_shade(body_rgb, 30), outline=outline)
    draw.point(eye, fill=(31, 24, 24))
    draw.point(((snout[0] + snout[2]) // 2, y0 + 8), fill=dark)
    for lx in (x0 + 5, x0 + 11):
        draw.rectangle((lx, y0 + 12, lx + 1, y0 + 15), fill=dark)
    draw.arc((tail_x - 1, y0 + 7, tail_x + 4, y0 + 11), start=260, end=80, fill=outline)


def _draw_sheep(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    facing: str,
    body_rgb: RGB,
    accent_rgb: RGB,
) -> None:
    x0, y0, _, _ = rect
    wool_dark = _shade(body_rgb, -42)
    outline = (80, 76, 69)
    face = accent_rgb
    for cx, cy in ((4, 8), (7, 6), (10, 7), (12, 10), (8, 11), (5, 11)):
        draw.ellipse((x0 + cx - 3, y0 + cy - 3, x0 + cx + 3, y0 + cy + 3), fill=body_rgb, outline=wool_dark)
    if facing == "left":
        head = (x0, y0 + 6, x0 + 6, y0 + 12)
        ear = (x0 + 4, y0 + 7, x0 + 7, y0 + 9)
        eye = (x0 + 2, y0 + 8)
    else:
        head = (x0 + 10, y0 + 6, x0 + 16, y0 + 12)
        ear = (x0 + 9, y0 + 7, x0 + 12, y0 + 9)
        eye = (x0 + 13, y0 + 8)
    draw.ellipse(ear, fill=_shade(face, 20), outline=outline)
    draw.ellipse(head, fill=face, outline=outline)
    draw.point(eye, fill=(18, 16, 14))
    for lx in (x0 + 5, x0 + 11):
        draw.rectangle((lx, y0 + 12, lx + 1, y0 + 15), fill=face)


def _draw_cow(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    facing: str,
    body_rgb: RGB,
    accent_rgb: RGB,
    spot_rgb: RGB,
) -> None:
    x0, y0, _, _ = rect
    outline = (53, 50, 47)
    horn = (222, 213, 169)
    hoof = (39, 37, 35)
    # Cows use a 2-tile placement footprint, but the visible sprite is closer
    # to 1.5 tiles wide so it does not look like a stretched animal.
    body = (x0 + 8, y0 + 4, x0 + 23, y0 + 13)
    draw.ellipse(body, fill=body_rgb, outline=outline)
    draw.rectangle((x0 + 10, y0 + 5, x0 + 21, y0 + 13), fill=body_rgb)
    if facing == "left":
        head = (x0 + 4, y0 + 4, x0 + 10, y0 + 11)
        muzzle = (x0 + 3, y0 + 8, x0 + 7, y0 + 12)
        eye = (x0 + 7, y0 + 6)
        horn_pts = [(x0 + 7, y0 + 4), (x0 + 6, y0 + 1), (x0 + 9, y0 + 3)]
        tail = (x0 + 24, y0 + 6, x0 + 27, y0 + 11)
    else:
        head = (x0 + 22, y0 + 4, x0 + 28, y0 + 11)
        muzzle = (x0 + 25, y0 + 8, x0 + 29, y0 + 12)
        eye = (x0 + 25, y0 + 6)
        horn_pts = [(x0 + 25, y0 + 4), (x0 + 26, y0 + 1), (x0 + 23, y0 + 3)]
        tail = (x0 + 4, y0 + 6, x0 + 7, y0 + 11)
    draw.line(tail, fill=outline)
    draw.point((tail[2], tail[3]), fill=outline)
    draw.polygon(horn_pts, fill=horn, outline=(137, 126, 88))
    draw.ellipse(head, fill=body_rgb, outline=outline)
    draw.ellipse(muzzle, fill=accent_rgb, outline=outline)
    draw.point(eye, fill=(18, 16, 14))
    for spot in ((x0 + 10, y0 + 6, x0 + 13, y0 + 10), (x0 + 17, y0 + 6, x0 + 20, y0 + 10)):
        draw.ellipse(spot, fill=spot_rgb)
    for lx in (x0 + 9, x0 + 14, x0 + 20, x0 + 24):
        draw.rectangle((lx, y0 + 12, lx + 1, y0 + 15), fill=outline)
        draw.rectangle((lx, y0 + 14, lx + 1, y0 + 15), fill=hoof)


def draw_pixel_person(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    skin_rgb: RGB = (225, 171, 109),
    shirt_rgb: RGB = (55, 116, 190),
    pants_rgb: RGB = (48, 72, 105),
    hair_rgb: RGB = (82, 50, 33),
    gender_id: str = "male",
    facing: str = "down",
    person_variant_id: str = "adult",
) -> None:
    """Draw a reusable 1-tile pixel person with shared appearance variants."""

    x0, y0, _, _ = _base_rect(tile_xywh)
    outline = (42, 37, 34)
    eye = (23, 21, 20)
    shoe = (44, 39, 37)
    skin = tuple(int(value) for value in skin_rgb)
    shirt = tuple(int(value) for value in shirt_rgb)
    pants = tuple(int(value) for value in pants_rgb)
    hair = tuple(int(value) for value in hair_rgb)
    normalized_gender = "female" if str(gender_id) == "female" else "male"
    normalized_facing = str(facing) if str(facing) in {"down", "up", "left", "right"} else "down"
    variant = normalize_object_variant_id("person", str(person_variant_id))
    if variant == "soldier":
        shirt = (78, 103, 61)
        pants = (54, 72, 51)
    skin_shadow = _shade(skin, -36)
    shirt_shadow = _shade(shirt, -42)
    hair_highlight = _shade(hair, 32)

    draw.rectangle((x0 + 4, y0 + 15, x0 + 11, y0 + 15), fill=(36, 80, 47))

    # Feet and legs stay separated so the sprite reads as a person after scaling.
    if normalized_facing in {"left", "right"}:
        draw.rectangle((x0 + 5, y0 + 12, x0 + 7, y0 + 14), fill=pants, outline=outline)
        draw.rectangle((x0 + 8, y0 + 12, x0 + 10, y0 + 14), fill=pants, outline=outline)
        if normalized_facing == "left":
            draw.rectangle((x0 + 4, y0 + 14, x0 + 7, y0 + 15), fill=shoe)
            draw.rectangle((x0 + 8, y0 + 14, x0 + 10, y0 + 15), fill=shoe)
        else:
            draw.rectangle((x0 + 5, y0 + 14, x0 + 7, y0 + 15), fill=shoe)
            draw.rectangle((x0 + 8, y0 + 14, x0 + 11, y0 + 15), fill=shoe)
    else:
        draw.rectangle((x0 + 4, y0 + 12, x0 + 6, y0 + 14), fill=pants, outline=outline)
        draw.rectangle((x0 + 9, y0 + 12, x0 + 11, y0 + 14), fill=pants, outline=outline)
        draw.rectangle((x0 + 3, y0 + 14, x0 + 6, y0 + 15), fill=shoe)
        draw.rectangle((x0 + 9, y0 + 14, x0 + 12, y0 + 15), fill=shoe)

    if normalized_gender == "female":
        draw.rectangle((x0 + 4, y0 + 8, x0 + 11, y0 + 11), fill=shirt, outline=outline)
        draw.polygon([(x0 + 4, y0 + 11), (x0 + 11, y0 + 11), (x0 + 13, y0 + 14), (x0 + 2, y0 + 14)], fill=shirt, outline=outline)
        draw.line((x0 + 4, y0 + 13, x0 + 11, y0 + 13), fill=shirt_shadow)
    else:
        draw.rectangle((x0 + 4, y0 + 8, x0 + 11, y0 + 13), fill=shirt, outline=outline)
        draw.line((x0 + 4, y0 + 11, x0 + 11, y0 + 11), fill=shirt_shadow)
        draw.rectangle((x0 + 5, y0 + 12, x0 + 10, y0 + 13), fill=pants)

    if variant in {"farmer", "worker", "vendor"}:
        apron = (224, 207, 124) if variant == "farmer" else (232, 184, 78) if variant == "worker" else (236, 228, 205)
        draw.rectangle((x0 + 5, y0 + 9, x0 + 10, y0 + 12), fill=apron)
        draw.line((x0 + 5, y0 + 9, x0 + 10, y0 + 12), fill=_shade(apron, -38))
    elif variant == "soldier":
        camo_dark = (42, 59, 39)
        camo_light = (104, 124, 76)
        for px, py, fill in (
            (x0 + 5, y0 + 9, camo_dark),
            (x0 + 9, y0 + 10, camo_light),
            (x0 + 7, y0 + 12, camo_dark),
            (x0 + 10, y0 + 13, camo_light),
        ):
            draw.point((px, py), fill=fill)

    # Arms/sleeves add definition without increasing the semantic 1-tile footprint.
    if normalized_facing == "left":
        draw.rectangle((x0 + 3, y0 + 9, x0 + 4, y0 + 12), fill=skin, outline=outline)
        draw.rectangle((x0 + 10, y0 + 9, x0 + 11, y0 + 11), fill=shirt_shadow)
    elif normalized_facing == "right":
        draw.rectangle((x0 + 11, y0 + 9, x0 + 12, y0 + 12), fill=skin, outline=outline)
        draw.rectangle((x0 + 4, y0 + 9, x0 + 5, y0 + 11), fill=shirt_shadow)
    else:
        draw.rectangle((x0 + 2, y0 + 9, x0 + 4, y0 + 12), fill=skin, outline=outline)
        draw.rectangle((x0 + 11, y0 + 9, x0 + 13, y0 + 12), fill=skin, outline=outline)

    # Head and hair. Direction-specific silhouettes make facing visible in review sheets.
    if normalized_facing == "left":
        draw.rectangle((x0 + 4, y0 + 3, x0 + 10, y0 + 8), fill=skin, outline=outline)
        draw.rectangle((x0 + 4, y0 + 1, x0 + 10, y0 + 4), fill=hair, outline=outline)
        draw.point((x0 + 4, y0 + 6), fill=eye)
        draw.point((x0 + 3, y0 + 6), fill=skin_shadow)
        draw.point((x0 + 6, y0 + 2), fill=hair_highlight)
        if normalized_gender == "female":
            draw.rectangle((x0 + 9, y0 + 4, x0 + 12, y0 + 9), fill=hair, outline=outline)
            draw.point((x0 + 12, y0 + 9), fill=hair_highlight)
    elif normalized_facing == "right":
        draw.rectangle((x0 + 5, y0 + 3, x0 + 11, y0 + 8), fill=skin, outline=outline)
        draw.rectangle((x0 + 5, y0 + 1, x0 + 11, y0 + 4), fill=hair, outline=outline)
        draw.point((x0 + 11, y0 + 6), fill=eye)
        draw.point((x0 + 12, y0 + 6), fill=skin_shadow)
        draw.point((x0 + 9, y0 + 2), fill=hair_highlight)
        if normalized_gender == "female":
            draw.rectangle((x0 + 3, y0 + 4, x0 + 6, y0 + 9), fill=hair, outline=outline)
            draw.point((x0 + 3, y0 + 9), fill=hair_highlight)
    elif normalized_facing == "up":
        draw.rectangle((x0 + 5, y0 + 3, x0 + 10, y0 + 8), fill=skin_shadow, outline=outline)
        draw.rectangle((x0 + 4, y0 + 1, x0 + 11, y0 + 7), fill=hair, outline=outline)
        draw.line((x0 + 5, y0 + 6, x0 + 10, y0 + 6), fill=_shade(hair, -34))
        draw.point((x0 + 6, y0 + 2), fill=hair_highlight)
        if normalized_gender == "female":
            draw.rectangle((x0 + 3, y0 + 5, x0 + 5, y0 + 10), fill=hair, outline=outline)
            draw.rectangle((x0 + 10, y0 + 5, x0 + 12, y0 + 10), fill=hair, outline=outline)
    else:
        draw.rectangle((x0 + 5, y0 + 3, x0 + 10, y0 + 8), fill=skin, outline=outline)
        draw.rectangle((x0 + 4, y0 + 1, x0 + 11, y0 + 4), fill=hair, outline=outline)
        draw.point((x0 + 6, y0 + 6), fill=eye)
        draw.point((x0 + 9, y0 + 6), fill=eye)
        draw.point((x0 + 8, y0 + 7), fill=skin_shadow)
        draw.point((x0 + 6, y0 + 2), fill=hair_highlight)
        if normalized_gender == "female":
            draw.rectangle((x0 + 3, y0 + 4, x0 + 5, y0 + 9), fill=hair, outline=outline)
            draw.rectangle((x0 + 10, y0 + 4, x0 + 12, y0 + 9), fill=hair, outline=outline)
            draw.point((x0 + 3, y0 + 9), fill=hair_highlight)
            draw.point((x0 + 12, y0 + 9), fill=hair_highlight)

    if variant == "farmer":
        straw = (218, 174, 84)
        draw.line((x0 + 3, y0 + 3, x0 + 12, y0 + 3), fill=(104, 76, 42))
        draw.rectangle((x0 + 5, y0, x0 + 10, y0 + 3), fill=straw, outline=(104, 76, 42))
        draw.line((x0 + 3, y0 + 3, x0 + 12, y0 + 3), fill=straw)
    elif variant == "worker":
        hard_hat = (237, 195, 63)
        draw.rectangle((x0 + 4, y0 + 1, x0 + 11, y0 + 3), fill=hard_hat, outline=(126, 91, 34))
        draw.line((x0 + 5, y0, x0 + 10, y0), fill=hard_hat)
        draw.line((x0 + 7, y0 + 1, x0 + 7, y0 + 3), fill=(126, 91, 34))
    elif variant == "vendor":
        cap = (192, 69, 72)
        draw.rectangle((x0 + 5, y0 + 1, x0 + 11, y0 + 3), fill=cap, outline=(99, 43, 45))
        draw.rectangle((x0 + 10, y0 + 3, x0 + 13, y0 + 4), fill=cap)
    elif variant == "soldier":
        helmet = (77, 96, 58)
        helmet_dark = (38, 54, 36)
        draw.rectangle((x0 + 4, y0 + 1, x0 + 11, y0 + 4), fill=helmet, outline=helmet_dark)
        draw.line((x0 + 3, y0 + 4, x0 + 12, y0 + 4), fill=helmet_dark)
        draw.point((x0 + 6, y0 + 2), fill=(116, 135, 82))


def draw_pixel_animal(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    animal_type: str,
    facing: str = "right",
    body_rgb: RGB | None = None,
    accent_rgb: RGB | None = None,
    spot_rgb: RGB | None = None,
) -> None:
    """Draw a reusable domestic farm animal.

    Chickens, pigs, and sheep are designed for a 1x1-tile footprint. Cows are
    designed for a 2x1-tile footprint so they read as larger than a person.
    """

    animal = str(animal_type)
    normalized_facing = "left" if str(facing) == "left" else "right"
    defaults: dict[str, tuple[RGB, RGB, RGB]] = {
        "chicken": ((242, 221, 160), (200, 48, 48), (84, 56, 39)),
        "pig": ((222, 132, 150), (184, 84, 112), (126, 64, 84)),
        "sheep": ((230, 226, 205), (73, 63, 56), (106, 94, 82)),
        "cow": ((238, 236, 221), (212, 144, 145), (48, 47, 45)),
    }
    base_body, base_accent, base_spot = defaults.get(animal, defaults["sheep"])
    resolved_body = body_rgb or base_body
    resolved_accent = accent_rgb or base_accent
    resolved_spot = spot_rgb or base_spot
    rect = _base_rect(tile_xywh)
    if animal == "chicken":
        _draw_chicken(draw, rect, facing=normalized_facing, body_rgb=resolved_body, accent_rgb=resolved_accent)
    elif animal == "pig":
        _draw_pig(draw, rect, facing=normalized_facing, body_rgb=resolved_body)
    elif animal == "cow":
        _draw_cow(
            draw,
            rect,
            facing=normalized_facing,
            body_rgb=resolved_body,
            accent_rgb=resolved_accent,
            spot_rgb=resolved_spot,
        )
    else:
        _draw_sheep(draw, rect, facing=normalized_facing, body_rgb=resolved_body, accent_rgb=resolved_accent)


def draw_pixel_grave_marker(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    style: str = "rounded",
    stone_rgb: RGB = (158, 164, 158),
    mound_rgb: RGB = (104, 86, 56),
    flower_rgb: RGB | None = None,
) -> None:
    """Draw a reusable 1-tile cemetery grave marker with a small mound."""

    x0, y0, _, _ = _base_rect(tile_xywh)
    style_id = str(style)
    stone_dark = _shade(stone_rgb, -58)
    stone_light = _shade(stone_rgb, 34)
    mound_dark = _shade(mound_rgb, -28)
    draw.ellipse((x0 + 2, y0 + 11, x0 + 14, y0 + 15), fill=mound_rgb, outline=mound_dark)
    draw.line((x0 + 4, y0 + 13, x0 + 12, y0 + 13), fill=_shade(mound_rgb, 18))

    if style_id == "cross":
        draw.rectangle((x0 + 7, y0 + 3, x0 + 9, y0 + 11), fill=stone_rgb, outline=stone_dark)
        draw.rectangle((x0 + 4, y0 + 5, x0 + 12, y0 + 7), fill=stone_rgb, outline=stone_dark)
        draw.point((x0 + 8, y0 + 4), fill=stone_light)
    elif style_id == "obelisk":
        draw.polygon([(x0 + 8, y0 + 2), (x0 + 4, y0 + 7), (x0 + 12, y0 + 7)], fill=stone_rgb, outline=stone_dark)
        draw.rectangle((x0 + 5, y0 + 7, x0 + 11, y0 + 12), fill=stone_rgb, outline=stone_dark)
        draw.line((x0 + 7, y0 + 7, x0 + 7, y0 + 12), fill=stone_light)
    elif style_id == "tablet":
        draw.rectangle((x0 + 4, y0 + 4, x0 + 12, y0 + 12), fill=stone_rgb, outline=stone_dark)
        draw.line((x0 + 6, y0 + 7, x0 + 10, y0 + 7), fill=stone_dark)
        draw.line((x0 + 6, y0 + 9, x0 + 10, y0 + 9), fill=stone_dark)
        draw.point((x0 + 5, y0 + 5), fill=stone_light)
    else:
        draw.rectangle((x0 + 5, y0 + 3, x0 + 11, y0 + 5), fill=stone_rgb, outline=stone_dark)
        draw.rectangle((x0 + 4, y0 + 5, x0 + 12, y0 + 12), fill=stone_rgb, outline=stone_dark)
        draw.point((x0 + 4, y0 + 4), fill=stone_dark)
        draw.point((x0 + 12, y0 + 4), fill=stone_dark)
        draw.line((x0 + 6, y0 + 8, x0 + 10, y0 + 8), fill=stone_dark)
        draw.point((x0 + 6, y0 + 4), fill=stone_light)

    if flower_rgb is not None:
        leaf = (43, 122, 65)
        draw.point((x0 + 12, y0 + 13), fill=leaf)
        draw.point((x0 + 13, y0 + 12), fill=flower_rgb)
        draw.point((x0 + 13, y0 + 13), fill=_shade(flower_rgb, -28))


def draw_pixel_dead_tree(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    bark_rgb: RGB = (82, 63, 50),
) -> None:
    """Draw a reusable leafless 1x2-tile cemetery tree."""

    x0, y0, _, _ = _base_rect(tile_xywh)
    bark_dark = _shade(bark_rgb, -34)
    bark_light = _shade(bark_rgb, 28)
    draw.rectangle((x0 + 7, y0 + 13, x0 + 10, y0 + 31), fill=bark_rgb, outline=bark_dark)
    draw.line((x0 + 8, y0 + 16, x0 + 3, y0 + 9), fill=bark_dark, width=2)
    draw.line((x0 + 9, y0 + 17, x0 + 15, y0 + 10), fill=bark_dark, width=2)
    draw.line((x0 + 8, y0 + 14, x0 + 6, y0 + 4), fill=bark_rgb, width=2)
    draw.line((x0 + 4, y0 + 9, x0 + 1, y0 + 6), fill=bark_rgb)
    draw.line((x0 + 14, y0 + 10, x0 + 16, y0 + 7), fill=bark_rgb)
    draw.line((x0 + 6, y0 + 4, x0 + 3, y0 + 1), fill=bark_dark)
    draw.point((x0 + 9, y0 + 18), fill=bark_light)
    draw.rectangle((x0 + 5, y0 + 29, x0 + 12, y0 + 31), fill=bark_dark)


def draw_pixel_barn(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    body_rgb: RGB = (181, 71, 59),
    roof_rgb: RGB = (126, 50, 50),
    door_state: str = "closed",
) -> None:
    """Draw a reusable front-facing pixel barn."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    outline = (82, 45, 39)
    draw.rectangle((x0 + 5, y0 + 19, x1 - 5, y1 - 2), fill=body_rgb, outline=outline)
    draw.polygon([(x0 + 2, y0 + 20), ((x0 + x1) // 2, y0 + 5), (x1 - 2, y0 + 20)], fill=roof_rgb, outline=outline)
    door_rect = (x0 + 26, y1 - 23, x0 + 44, y1 - 2)
    if str(door_state) == "open":
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


def draw_pixel_chicken_coop(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    body_rgb: RGB = (188, 126, 70),
    roof_rgb: RGB = (137, 62, 50),
    door_state: str = "closed",
) -> None:
    """Draw a reusable front-facing chicken coop."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    draw.rectangle((x0 + 4, y0 + 16, x1 - 4, y1 - 2), fill=body_rgb, outline=(78, 55, 37))
    draw.polygon([(x0 + 2, y0 + 17), ((x0 + x1) // 2, y0 + 7), (x1 - 2, y0 + 17)], fill=roof_rgb, outline=(89, 48, 39))
    if str(door_state) == "open":
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


def _draw_house_roof_texture(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    roof_rgb: RGB,
    roof_style: str,
    direction: str,
) -> None:
    x0, y0, x1, y1 = rect
    roof_dark = _shade(roof_rgb, -40)
    roof_light = _shade(roof_rgb, 24)
    style = str(roof_style)
    if str(direction) == "vertical":
        ridge_x = (x0 + x1) // 2
        draw.line((ridge_x, y0 + 2, ridge_x, y1 - 2), fill=roof_dark)
        for y in range(y0 + 7, y1 - 3, 8):
            draw.line((x0 + 3, y, x1 - 3, y), fill=roof_dark)
        if style == "wood_plank":
            for x in range(x0 + 5, x1 - 3, 7):
                draw.line((x, y0 + 4, x, y1 - 4), fill=_shade(roof_rgb, -25))
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
                draw.line((x0 + 3, y, x1 - 3, y), fill=_shade(roof_rgb, -25))
        elif style == "tile":
            for y in range(y0 + 7, y1 - 4, 7):
                for x in range(x0 + 5, x1 - 5, 9):
                    draw.arc((x - 4, y - 3, x + 4, y + 5), start=180, end=360, fill=roof_dark)
        else:
            draw.line((x0 + 2, y0 + 2, x1 - 2, y0 + 2), fill=roof_light)


def _draw_house_wall_texture(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    body_rgb: RGB,
    wall_style: str,
) -> None:
    x0, y0, x1, y1 = rect
    dark = _shade(body_rgb, -42)
    light = _shade(body_rgb, 24)
    style = str(wall_style)
    if style == "wood":
        for x in range(x0 + 5, x1 - 3, 8):
            draw.line((x, y0 + 2, x, y1 - 2), fill=dark)
        for y in range(y0 + 8, y1 - 3, 14):
            draw.line((x0 + 3, y, x1 - 3, y), fill=_shade(body_rgb, -18))
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
        draw.point((x0 + 9, y1 - 8), fill=_shade(body_rgb, -20))


def _draw_house_window(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    frame = (89, 64, 43)
    glass = (93, 175, 202)
    draw.rectangle(rect, fill=glass, outline=frame)
    draw.line(((x0 + x1) // 2, y0, (x0 + x1) // 2, y1), fill=(225, 240, 232))
    draw.line((x0, (y0 + y1) // 2, x1, (y0 + y1) // 2), fill=(225, 240, 232))
    draw.point((x0 + 2, y0 + 2), fill=(197, 228, 224))


def _draw_house_door(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    *,
    knob_side: str,
    door_state: str,
) -> None:
    x0, y0, x1, y1 = rect
    if str(door_state) == "open":
        draw.rectangle(rect, fill=(36, 30, 27), outline=(54, 38, 28))
        if str(knob_side) == "left":
            panel = [(x0, y0), (x0 - 6, y0 + 3), (x0 - 6, y1 - 3), (x0, y1)]
            knob = (x0 - 4, (y0 + y1) // 2)
        else:
            panel = [(x1, y0), (x1 + 6, y0 + 3), (x1 + 6, y1 - 3), (x1, y1)]
            knob = (x1 + 4, (y0 + y1) // 2)
        draw.polygon(panel, fill=(91, 55, 38), outline=(54, 38, 28))
        draw.point(knob, fill=(237, 203, 96))
    else:
        draw.rectangle(rect, fill=(91, 55, 38), outline=(54, 38, 28))
        knob_x = x0 + 2 if str(knob_side) == "left" else x1 - 2
        draw.point((knob_x, (y0 + y1) // 2), fill=(237, 203, 96))


def _draw_house_sign(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], *, variant: str) -> None:
    draw.rectangle(rect, fill=(236, 205, 116), outline=(99, 76, 43))
    mark = (74, 76, 96) if str(variant) == "inn" else (126, 75, 51)
    x0, y0, x1, y1 = rect
    draw.rectangle((x0 + 5, y0 + 3, x1 - 5, y1 - 3), fill=mark)


def draw_pixel_house(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    roof_rgb: RGB = (179, 72, 55),
    body_rgb: RGB = (211, 176, 119),
    variant: str = "house",
    roof_style: str = "shingle",
    wall_style: str = "stucco",
    door_state: str = "closed",
    draw_ground_shadow: bool = True,
) -> None:
    """Draw a reusable front-facing village house, shop, inn, or tower."""

    variant_id = str(variant)
    if variant_id == "castle":
        draw_pixel_castle(draw, tile_xywh, stone_rgb=body_rgb, door_state=door_state)
        return
    if variant_id == "church":
        draw_pixel_church(draw, tile_xywh, body_rgb=body_rgb, roof_rgb=roof_rgb, door_state=door_state)
        return

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    if draw_ground_shadow:
        draw.rectangle((x0 + 2, y1 - 4, x1 + 2, y1 + 2), fill=(44, 83, 50))
    roof_h = max(12, int((y1 - y0 + 1) * 0.38))
    body_rect = (x0 + 2, y0 + roof_h - 2, x1 - 2, y1 - 2)
    roof_rect = (x0, y0 + 6, x1, y0 + roof_h + 4)
    draw.rectangle(body_rect, fill=body_rgb, outline=(92, 74, 58))
    _draw_house_wall_texture(draw, body_rect, body_rgb=body_rgb, wall_style=wall_style)
    draw.rectangle(roof_rect, fill=roof_rgb, outline=(92, 52, 41))
    _draw_house_roof_texture(draw, roof_rect, roof_rgb=roof_rgb, roof_style=roof_style, direction="horizontal")

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
    if variant_id in {"shop", "inn"}:
        _draw_house_sign(draw, (x0 + 7, y0 + roof_h + 6, x0 + 26, y0 + roof_h + 16), variant=variant_id)
    if variant_id == "tower":
        cx = (x0 + x1) // 2
        draw.rectangle((cx - 5, y0, cx + 5, y0 + 10), fill=body_rgb, outline=(92, 74, 58))
        draw.rectangle((cx - 7, y0 - 3, cx + 7, y0 + 3), fill=roof_rgb, outline=(92, 52, 41))


def draw_pixel_well(draw: ImageDraw.ImageDraw, tile_xywh: TileBox) -> None:
    """Draw a reusable 2x2-tile roofed round well."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    stone = (142, 143, 134)
    stone_dark = (76, 78, 76)
    stone_light = (178, 179, 166)
    wood = (111, 72, 43)
    roof = (154, 86, 58)
    water = (49, 119, 165)
    water_light = (118, 192, 220)

    draw.rectangle((x0 + 7, y0 + 7, x0 + 9, y0 + 23), fill=wood, outline=(78, 51, 34))
    draw.rectangle((x1 - 9, y0 + 7, x1 - 7, y0 + 23), fill=wood, outline=(78, 51, 34))
    draw.polygon(
        [(x0 + 3, y0 + 8), (x0 + 16, y0 + 1), (x1 - 3, y0 + 8), (x1 - 5, y0 + 12), (x0 + 5, y0 + 12)],
        fill=roof,
        outline=(85, 52, 39),
    )
    for rx in range(x0 + 7, x1 - 8, 6):
        draw.line((rx, y0 + 7, rx + 4, y0 + 11), fill=_shade(roof, -35))
    draw.line((x0 + 16, y0 + 8, x0 + 16, y0 + 16), fill=(67, 48, 38))
    draw.rectangle((x0 + 14, y0 + 15, x0 + 18, y0 + 18), fill=(96, 64, 42), outline=(60, 42, 32))

    draw.ellipse((x0 + 5, y0 + 14, x1 - 5, y1 - 2), fill=stone, outline=stone_dark)
    draw.ellipse((x0 + 8, y0 + 15, x1 - 8, y0 + 25), fill=stone_dark)
    draw.ellipse((x0 + 10, y0 + 16, x1 - 10, y0 + 23), fill=water, outline=(39, 89, 126))
    draw.line((x0 + 11, y0 + 18, x0 + 18, y0 + 18), fill=water_light)
    draw.point((x0 + 8, y0 + 19), fill=stone_light)
    draw.point((x1 - 9, y0 + 19), fill=stone_light)
    draw.point((x0 + 13, y1 - 5), fill=stone_light)


def draw_pixel_windmill(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    body_rgb: RGB = (213, 184, 137),
    roof_rgb: RGB = (68, 116, 171),
    blade_pose: str = "plus",
) -> None:
    """Draw a reusable 3x4-tile village windmill landmark."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    body_dark = _shade(body_rgb, -62)
    body_shadow = _shade(body_rgb, -24)
    body_light = _shade(body_rgb, 28)
    roof_dark = _shade(roof_rgb, -44)
    wood = (114, 72, 39)
    wood_dark = (72, 48, 31)
    blade = (230, 218, 176)
    blade_dark = (126, 95, 55)

    draw.polygon([(x0 + 11, y0 + 20), (x1 - 11, y0 + 20), (x1 - 7, y1 - 2), (x0 + 7, y1 - 2)], fill=body_rgb, outline=body_dark)
    draw.line((x0 + 11, y0 + 31, x1 - 11, y0 + 31), fill=body_shadow)
    draw.line((x0 + 10, y0 + 43, x1 - 10, y0 + 43), fill=body_shadow)
    for bx, by in ((x0 + 15, y0 + 27), (x0 + 29, y0 + 38), (x0 + 20, y0 + 50)):
        draw.line((bx, by, bx + 5, by), fill=body_shadow)
    draw.polygon([(x0 + 7, y0 + 21), ((x0 + x1) // 2, y0 + 6), (x1 - 7, y0 + 21)], fill=roof_rgb, outline=roof_dark)
    draw.line((x0 + 18, y0 + 17, x1 - 18, y0 + 17), fill=_shade(roof_rgb, 26))
    draw.rectangle((x0 + 20, y1 - 18, x0 + 28, y1 - 2), fill=wood, outline=wood_dark)
    draw.point((x0 + 26, y1 - 10), fill=(235, 198, 93))
    draw.rectangle((x0 + 11, y0 + 36, x0 + 18, y0 + 45), fill=(93, 175, 202), outline=(63, 91, 104))
    draw.line((x0 + 14, y0 + 36, x0 + 14, y0 + 45), fill=(225, 240, 232))
    draw.rectangle((x1 - 18, y0 + 36, x1 - 11, y0 + 45), fill=(93, 175, 202), outline=(63, 91, 104))
    draw.line((x1 - 15, y0 + 36, x1 - 15, y0 + 45), fill=(225, 240, 232))

    hub_x = (x0 + x1) // 2
    hub_y = y0 + 21
    if str(blade_pose) == "diagonal":
        arms = [
            [(hub_x - 1, hub_y - 2), (hub_x - 14, hub_y - 16), (hub_x - 17, hub_y - 13), (hub_x - 3, hub_y)],
            [(hub_x + 1, hub_y - 2), (hub_x + 16, hub_y - 14), (hub_x + 18, hub_y - 10), (hub_x + 3, hub_y)],
            [(hub_x - 1, hub_y + 2), (hub_x - 16, hub_y + 15), (hub_x - 13, hub_y + 18), (hub_x - 3, hub_y + 3)],
            [(hub_x + 1, hub_y + 2), (hub_x + 14, hub_y + 16), (hub_x + 17, hub_y + 13), (hub_x + 3, hub_y + 3)],
        ]
    else:
        arms = [
            [(hub_x - 2, hub_y - 1), (hub_x - 2, y0 + 1), (hub_x + 2, y0 + 1), (hub_x + 2, hub_y - 1)],
            [(hub_x + 1, hub_y - 2), (x1 - 2, hub_y - 2), (x1 - 2, hub_y + 2), (hub_x + 1, hub_y + 2)],
            [(hub_x - 2, hub_y + 1), (hub_x - 2, y0 + 39), (hub_x + 2, y0 + 39), (hub_x + 2, hub_y + 1)],
            [(hub_x - 1, hub_y - 2), (x0 + 2, hub_y - 2), (x0 + 2, hub_y + 2), (hub_x - 1, hub_y + 2)],
        ]
    for arm in arms:
        draw.polygon(arm, fill=blade, outline=blade_dark)
    draw.rectangle((hub_x - 3, hub_y - 3, hub_x + 3, hub_y + 3), fill=wood, outline=wood_dark)
    draw.point((hub_x, hub_y), fill=(238, 201, 97))


def draw_pixel_castle(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    stone_rgb: RGB = (145, 148, 144),
    door_state: str = "closed",
) -> None:
    """Draw a reusable small RPG castle building."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    stone_dark = _shade(stone_rgb, -56)
    stone_light = _shade(stone_rgb, 28)
    stone_mid = _shade(stone_rgb, -22)
    banner = (126, 65, 102)
    banner_dark = (83, 45, 74)
    roof = (102, 80, 104)
    draw.rectangle((x0 + 3, y0 + 15, x1 - 3, y1 - 2), fill=stone_rgb, outline=stone_dark)
    for tx in (x0 + 2, x1 - 17):
        draw.rectangle((tx, y0 + 8, tx + 15, y1 - 2), fill=stone_rgb, outline=stone_dark)
        for cx in range(tx + 2, tx + 13, 5):
            draw.rectangle((cx, y0 + 5, cx + 3, y0 + 10), fill=stone_rgb, outline=stone_dark)
        draw.rectangle((tx + 5, y0 + 18, tx + 10, y0 + 25), fill=(52, 62, 82), outline=stone_dark)
        draw.rectangle((tx + 6, y0 + 35, tx + 9, y0 + 40), fill=(52, 62, 82), outline=stone_dark)
        draw.line((tx + 3, y0 + 29, tx + 13, y0 + 29), fill=stone_mid)
    for cx in range(x0 + 7, x1 - 8, 8):
        draw.rectangle((cx, y0 + 11, cx + 4, y0 + 16), fill=stone_rgb, outline=stone_dark)
    for bx, by in ((x0 + 21, y0 + 23), (x0 + 34, y0 + 22), (x0 + 48, y0 + 25), (x0 + 25, y0 + 39), (x0 + 54, y0 + 40)):
        draw.line((bx, by, bx + 6, by), fill=stone_mid)
        draw.point((bx + 2, by + 5), fill=stone_dark)
    draw.polygon([(x0 + 20, y0 + 14), (x0 + 29, y0 + 5), (x0 + 38, y0 + 14)], fill=roof, outline=(66, 49, 72))
    draw.line((x0 + 29, y0 + 2, x0 + 29, y0 + 8), fill=(66, 49, 72))
    draw.polygon([(x0 + 30, y0 + 2), (x0 + 37, y0 + 4), (x0 + 30, y0 + 6)], fill=(180, 63, 62), outline=(102, 50, 58))
    draw.rectangle((x0 + 22, y0 + 26, x0 + 27, y0 + 41), fill=banner, outline=banner_dark)
    draw.polygon([(x0 + 22, y0 + 41), (x0 + 24, y0 + 45), (x0 + 27, y0 + 41)], fill=banner, outline=banner_dark)
    draw.rectangle((x1 - 28, y0 + 26, x1 - 23, y0 + 41), fill=banner, outline=banner_dark)
    draw.polygon([(x1 - 28, y0 + 41), (x1 - 25, y0 + 45), (x1 - 23, y0 + 41)], fill=banner, outline=banner_dark)
    for tx, ty in ((x0 + 31, y0 + 35), (x0 + 48, y0 + 35)):
        draw.rectangle((tx, ty, tx + 3, ty + 7), fill=(48, 55, 71), outline=stone_dark)
    for tx in (x0 + 25, x1 - 30):
        draw.point((tx, y1 - 24), fill=(247, 178, 70))
        draw.point((tx + 1, y1 - 23), fill=(210, 81, 54))
        draw.point((tx, y1 - 22), fill=(74, 48, 37))
    gate_x = (x0 + x1) // 2
    if door_state == "open":
        draw.rectangle((gate_x - 7, y1 - 17, gate_x + 7, y1 - 2), fill=(34, 32, 34), outline=(54, 37, 29))
        draw.polygon(
            [(gate_x - 8, y1 - 17), (gate_x - 17, y1 - 14), (gate_x - 17, y1 - 5), (gate_x - 8, y1 - 2)],
            fill=(83, 56, 40),
            outline=(54, 37, 29),
        )
        draw.polygon(
            [(gate_x + 8, y1 - 17), (gate_x + 17, y1 - 14), (gate_x + 17, y1 - 5), (gate_x + 8, y1 - 2)],
            fill=(83, 56, 40),
            outline=(54, 37, 29),
        )
        draw.line((gate_x - 15, y1 - 13, gate_x - 10, y1 - 3), fill=(122, 83, 54))
        draw.line((gate_x + 15, y1 - 13, gate_x + 10, y1 - 3), fill=(122, 83, 54))
    else:
        draw.rectangle((gate_x - 7, y1 - 17, gate_x + 7, y1 - 2), fill=(83, 56, 40), outline=(54, 37, 29))
        draw.rectangle((gate_x - 3, y1 - 13, gate_x + 3, y1 - 2), fill=(56, 42, 36))
        draw.line((gate_x - 6, y1 - 14, gate_x + 6, y1 - 14), fill=(122, 83, 54))
        for sx in range(gate_x - 5, gate_x + 6, 5):
            draw.line((sx, y1 - 16, sx, y1 - 3), fill=(54, 37, 29))
    for px in range(x0 + 10, x1 - 10, 14):
        draw.point((px, y0 + 23), fill=stone_light)
        draw.point((px + 1, y0 + 31), fill=stone_dark)
    draw.line((x0 + 5, y1 - 3, x1 - 5, y1 - 3), fill=stone_dark)


def draw_pixel_church(
    draw: ImageDraw.ImageDraw,
    tile_xywh: TileBox,
    *,
    body_rgb: RGB = (211, 190, 150),
    roof_rgb: RGB = (77, 113, 166),
    door_state: str = "closed",
) -> None:
    """Draw a reusable chapel/church building."""

    x0, y0, x1, y1 = _base_rect(tile_xywh)
    body_dark = _shade(body_rgb, -66)
    body_shadow = _shade(body_rgb, -24)
    body_light = _shade(body_rgb, 30)
    roof_dark = _shade(roof_rgb, -46)
    glass = (98, 180, 204)

    # Broad nave and centered steeple keep the church legible in map-scale thumbnails.
    draw.rectangle((x0 + 8, y0 + 24, x1 - 7, y1 - 2), fill=body_rgb, outline=body_dark)
    draw.polygon([(x0 + 5, y0 + 25), (x0 + 31, y0 + 8), (x1 - 4, y0 + 25)], fill=roof_rgb, outline=roof_dark)
    draw.line((x0 + 16, y0 + 18, x0 + 47, y0 + 18), fill=_shade(roof_rgb, 24))
    for rx in range(x0 + 11, x1 - 10, 10):
        draw.line((rx, y0 + 23, rx + 7, y0 + 17), fill=roof_dark)

    tower = (x0 + 23, y0 + 14, x0 + 40, y1 - 2)
    draw.rectangle(tower, fill=body_rgb, outline=body_dark)
    draw.rectangle((x0 + 25, y0 + 17, x0 + 38, y0 + 20), fill=body_light, outline=body_shadow)
    draw.polygon([(x0 + 20, y0 + 14), (x0 + 31, y0 + 2), (x0 + 43, y0 + 14)], fill=roof_rgb, outline=roof_dark)
    draw.line((x0 + 31, y0, x0 + 31, y0 + 7), fill=(66, 55, 45))
    draw.line((x0 + 28, y0 + 3, x0 + 34, y0 + 3), fill=(66, 55, 45))

    draw.rectangle((x0 + 28, y0 + 23, x0 + 35, y0 + 31), fill=glass, outline=(60, 89, 101))
    draw.point((x0 + 29, y0 + 22), fill=(60, 89, 101))
    draw.point((x0 + 34, y0 + 22), fill=(60, 89, 101))
    draw.line((x0 + 31, y0 + 23, x0 + 31, y0 + 31), fill=(228, 241, 226))
    draw.line((x0 + 28, y0 + 27, x0 + 35, y0 + 27), fill=(228, 241, 226))

    for wx in (x0 + 13, x0 + 47):
        draw.rectangle((wx, y0 + 34, wx + 7, y0 + 44), fill=glass, outline=(60, 89, 101))
        draw.line((wx + 3, y0 + 34, wx + 3, y0 + 44), fill=(228, 241, 226))
        draw.line((wx, y0 + 39, wx + 7, y0 + 39), fill=(72, 122, 142))

    door = (x0 + 28, y1 - 16, x0 + 35, y1 - 2)
    if door_state == "open":
        draw.rectangle(door, fill=(34, 29, 25), outline=(56, 38, 30))
        draw.polygon(
            [(x0 + 36, y1 - 15), (x0 + 42, y1 - 13), (x0 + 42, y1 - 5), (x0 + 36, y1 - 2)],
            fill=(91, 59, 41),
            outline=(56, 38, 30),
        )
        draw.point((x0 + 41, y1 - 9), fill=(237, 203, 96))
    else:
        draw.rectangle(door, fill=(91, 59, 41), outline=(56, 38, 30))
        draw.line((x0 + 31, y1 - 15, x0 + 31, y1 - 2), fill=(56, 38, 30))
        draw.point((x0 + 34, y1 - 9), fill=(237, 203, 96))
    draw.line((x0 + 9, y1 - 2, x1 - 8, y1 - 2), fill=body_shadow)


__all__ = [
    "CANONICAL_TILE_PX",
    "PIXEL_CROP_STYLES",
    "PIXEL_DOMESTIC_ANIMALS",
    "PIXEL_GRAVE_MARKER_STYLES",
    "PIXEL_PERSON_VARIANTS",
    "PIXEL_PRODUCE_GOODS",
    "PIXEL_SHELF_GOODS",
    "PIXEL_TREE_STYLES",
    "PIXEL_VEGETABLE_STYLES",
    "PixelCropStyle",
    "PixelDomesticAnimal",
    "PixelGraveMarkerStyle",
    "PixelPersonVariant",
    "PixelProduceGoods",
    "PixelShelfGoods",
    "PixelTreeStyle",
    "draw_pixel_animal",
    "draw_pixel_archway",
    "draw_pixel_autumn_overlay",
    "draw_pixel_barrel",
    "draw_pixel_basket",
    "draw_pixel_barn",
    "draw_pixel_bench",
    "draw_pixel_boulder",
    "draw_pixel_brazier",
    "draw_pixel_bridge",
    "draw_pixel_broken_wall",
    "draw_pixel_castle",
    "draw_pixel_cave_entrance",
    "draw_pixel_chicken_coop",
    "draw_pixel_church",
    "draw_pixel_crystal_cluster",
    "draw_pixel_crop_row",
    "draw_pixel_cart",
    "draw_pixel_cemetery_gate",
    "draw_pixel_chest",
    "draw_pixel_counter",
    "draw_pixel_crate",
    "draw_pixel_dead_tree",
    "draw_pixel_farm_gate",
    "draw_pixel_fence",
    "draw_pixel_flower_patch",
    "draw_pixel_floor_switch",
    "draw_pixel_fountain",
    "draw_pixel_gazebo",
    "draw_pixel_grave_marker",
    "draw_pixel_hay_bale",
    "draw_pixel_house",
    "draw_pixel_iron_fence",
    "draw_pixel_jar",
    "draw_pixel_lamp_post",
    "draw_pixel_ladder",
    "draw_pixel_magic_circle",
    "draw_pixel_market_stall",
    "draw_pixel_mine_cart",
    "draw_pixel_notice_board",
    "draw_pixel_ore_vein",
    "draw_pixel_person",
    "draw_pixel_pond",
    "draw_pixel_pot",
    "draw_pixel_produce_bin",
    "draw_pixel_rail_track",
    "draw_pixel_rock",
    "draw_pixel_rubble",
    "draw_pixel_rug",
    "draw_pixel_sack",
    "draw_pixel_scarecrow",
    "draw_pixel_sealed_door",
    "draw_pixel_shelf",
    "draw_pixel_sign",
    "draw_pixel_stairs",
    "draw_pixel_stalagmite",
    "draw_pixel_statue",
    "draw_pixel_stone_column",
    "draw_pixel_torch",
    "draw_pixel_tree",
    "draw_pixel_trough",
    "draw_pixel_vegetable_patch",
    "draw_pixel_well",
    "draw_pixel_windmill",
    "draw_pixel_woodpile",
    "draw_pixel_wood_support",
    "draw_pixel_wagon",
    "draw_pixel_winter_overlay",
]
