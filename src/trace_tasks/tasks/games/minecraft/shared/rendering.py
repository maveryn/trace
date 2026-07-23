"""Shared renderer for Minecraft-like block-world games tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.color_distance import resolve_contrasting_palette
from ....shared.text_rendering import fit_font_to_box
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from ...shared.text import draw_game_text_traced as draw_text_traced
from .defaults import DEFAULTS, STYLE_VARIANTS
from .state import (
    MinecraftBlock,
    MinecraftCell,
    MinecraftRenderParams,
    MinecraftRouteOverlay,
    MinecraftTheme,
    RenderedMinecraftScene,
    ladder_entity_id,
    player_entity_id,
    stack_entity_id,
)


Point2 = Tuple[float, float]
BBox = Tuple[float, float, float, float]


def build_games_minecraft_theme(*, style_variant: str) -> MinecraftTheme:
    """Return a visual theme for a Minecraft-like block world."""

    style = str(style_variant)
    if style == "desert":
        return MinecraftTheme(
            ground_rgb=(214, 183, 112),
            ground_alt_rgb=(229, 201, 138),
            water_rgb=(40, 139, 211),
            water_line_rgb=(23, 91, 156),
            outline_rgb=(94, 82, 59),
            support_rgb=(167, 120, 72),
            stone_rgb=(137, 133, 119),
            gold_rgb=(224, 181, 43),
            diamond_rgb=(78, 211, 219),
            ladder_rgb=(114, 71, 35),
            player_rgb=(220, 45, 45),
            arrow_rgb=(176, 43, 43),
        )
    if style == "snow":
        return MinecraftTheme(
            ground_rgb=(228, 237, 241),
            ground_alt_rgb=(210, 225, 231),
            water_rgb=(82, 158, 209),
            water_line_rgb=(48, 106, 161),
            outline_rgb=(78, 91, 101),
            support_rgb=(145, 125, 92),
            stone_rgb=(139, 150, 156),
            gold_rgb=(224, 185, 55),
            diamond_rgb=(85, 219, 226),
            ladder_rgb=(135, 87, 46),
            player_rgb=(210, 42, 56),
            arrow_rgb=(192, 49, 66),
        )
    if style == "cave":
        return MinecraftTheme(
            ground_rgb=(67, 70, 73),
            ground_alt_rgb=(76, 80, 84),
            water_rgb=(32, 100, 155),
            water_line_rgb=(22, 68, 116),
            outline_rgb=(32, 35, 37),
            support_rgb=(111, 92, 72),
            stone_rgb=(109, 114, 117),
            gold_rgb=(214, 173, 50),
            diamond_rgb=(70, 195, 203),
            ladder_rgb=(156, 101, 49),
            player_rgb=(229, 54, 63),
            arrow_rgb=(239, 87, 83),
        )
    if style == "mesa":
        return MinecraftTheme(
            ground_rgb=(174, 97, 61),
            ground_alt_rgb=(191, 118, 72),
            water_rgb=(50, 136, 189),
            water_line_rgb=(30, 84, 139),
            outline_rgb=(88, 51, 42),
            support_rgb=(156, 91, 65),
            stone_rgb=(124, 113, 105),
            gold_rgb=(227, 181, 52),
            diamond_rgb=(69, 205, 212),
            ladder_rgb=(111, 66, 37),
            player_rgb=(223, 47, 47),
            arrow_rgb=(204, 50, 48),
        )
    return MinecraftTheme(
        ground_rgb=(91, 166, 79),
        ground_alt_rgb=(109, 181, 91),
        water_rgb=(38, 134, 214),
        water_line_rgb=(25, 86, 163),
        outline_rgb=(52, 89, 51),
        support_rgb=(132, 97, 65),
        stone_rgb=(129, 133, 126),
        gold_rgb=(224, 184, 45),
        diamond_rgb=(73, 210, 220),
        ladder_rgb=(118, 72, 36),
        player_rgb=(219, 42, 42),
        arrow_rgb=(190, 47, 45),
    )


def resolve_minecraft_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
    grid_width: int,
    grid_depth: int,
    max_stack_height: int = 1,
    player_marker_label: str = "P",
) -> MinecraftRenderParams:
    """Resolve block-world rendering dimensions, font, and canvas jitter."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.layout",
        ),
        unit_scale_meta,
    )
    tile_width = scale_games_px(
        params.get("tile_width_px", group_default(render_defaults, "tile_width_px", DEFAULTS.tile_width_px)),
        unit_scale,
        min_px=29,
    )
    tile_height = scale_games_px(
        params.get("tile_height_px", group_default(render_defaults, "tile_height_px", DEFAULTS.tile_height_px)),
        unit_scale,
        min_px=15,
    )
    cube_height = scale_games_px(
        params.get("cube_height_px", group_default(render_defaults, "cube_height_px", DEFAULTS.cube_height_px)),
        unit_scale,
        min_px=15,
    )
    default_canvas_width = int(group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))
    default_canvas_height = int(group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))
    world_width = float((int(grid_width) + int(grid_depth)) * int(tile_width) / 2.0)
    stack_height = max(1, int(max_stack_height))
    world_height = float((int(grid_width) + int(grid_depth)) * int(tile_height) / 2.0) + float(
        cube_height * stack_height
    )
    canvas_width = int(params.get("canvas_width", min(default_canvas_width, max(520, int(round(world_width + 190.0))))))
    canvas_height = int(
        params.get("canvas_height", min(default_canvas_height, max(430, int(round(world_height + 165.0)))))
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font_family",
        params=params,
    )
    return MinecraftRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        tile_width_px=int(tile_width),
        tile_height_px=int(tile_height),
        cube_height_px=int(cube_height),
        outline_width_px=scale_games_px(
            params.get("outline_width_px", group_default(render_defaults, "outline_width_px", DEFAULTS.outline_width_px)),
            unit_scale,
            min_px=1,
        ),
        player_marker_size_px=scale_games_px(
            params.get(
                "player_marker_size_px",
                group_default(render_defaults, "player_marker_size_px", DEFAULTS.player_marker_size_px),
            ),
            unit_scale,
            min_px=14,
        ),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
        max_stack_height=int(stack_height),
        player_marker_label=str(player_marker_label or "P"),
    )


def _shade(rgb: Sequence[int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(float(v) * float(factor))))) for v in rgb)


def _tint(rgb: Sequence[int], amount: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(float(v) + ((255.0 - float(v)) * float(amount)))))) for v in rgb)


def _bbox_from_points(points: Sequence[Point2]) -> BBox:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (round(min(xs), 3), round(min(ys), 3), round(max(xs), 3), round(max(ys), 3))


def _merge_bbox(*bboxes: BBox) -> BBox:
    xs0 = [float(bbox[0]) for bbox in bboxes]
    ys0 = [float(bbox[1]) for bbox in bboxes]
    xs1 = [float(bbox[2]) for bbox in bboxes]
    ys1 = [float(bbox[3]) for bbox in bboxes]
    return (round(min(xs0), 3), round(min(ys0), 3), round(max(xs1), 3), round(max(ys1), 3))


def _center_of_points(points: Sequence[Point2]) -> Point2:
    return (
        round(sum(float(point[0]) for point in points) / float(len(points)), 3),
        round(sum(float(point[1]) for point in points) / float(len(points)), 3),
    )


def _grid_origin(*, grid_width: int, grid_depth: int, params: MinecraftRenderParams) -> Tuple[float, float]:
    span_x = float((int(grid_width) + int(grid_depth)) * int(params.tile_width_px) / 2.0)
    span_y = float((int(grid_width) + int(grid_depth)) * int(params.tile_height_px) / 2.0)
    stack_height_px = float(max(1, int(params.max_stack_height)) * int(params.cube_height_px))
    ore_mark_pad_px = max(6.0, float(int(params.cube_height_px)) * 0.45)
    origin_x = float((int(params.canvas_width) - span_x) / 2.0) + float(int(grid_depth) * int(params.tile_width_px) / 2.0)
    origin_y = 76.0 + float(max(0, int(params.max_stack_height) - 1) * int(params.cube_height_px))
    content_top = float(origin_y - stack_height_px - ore_mark_pad_px)
    base_bbox = (
        origin_x - float(int(grid_depth) * int(params.tile_width_px) / 2.0),
        content_top,
        origin_x + float(int(grid_width) * int(params.tile_width_px) / 2.0),
        origin_y + span_y,
    )
    jittered, _dx, _dy, _resolved = apply_games_layout_jitter_to_bbox(
        bbox_px=base_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    return (
        float(origin_x + (float(jittered[0]) - float(base_bbox[0]))),
        float(origin_y + (float(jittered[1]) - float(base_bbox[1]))),
    )


def _project(
    *,
    x: float,
    y: float,
    z: float,
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
) -> Point2:
    return (
        float(origin[0]) + (float(x) - float(y)) * float(params.tile_width_px) / 2.0,
        float(origin[1]) + (float(x) + float(y)) * float(params.tile_height_px) / 2.0 - float(z) * float(params.cube_height_px),
    )


def _cell_top_points(*, x: int, y: int, z: float, origin: Tuple[float, float], params: MinecraftRenderParams) -> Tuple[Point2, ...]:
    return (
        _project(x=float(x), y=float(y), z=float(z), origin=origin, params=params),
        _project(x=float(x + 1), y=float(y), z=float(z), origin=origin, params=params),
        _project(x=float(x + 1), y=float(y + 1), z=float(z), origin=origin, params=params),
        _project(x=float(x), y=float(y + 1), z=float(z), origin=origin, params=params),
    )


def _draw_tile(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
    width: int = 1,
) -> BBox:
    points = _cell_top_points(x=int(x), y=int(y), z=0.0, origin=origin, params=params)
    draw.polygon(points, fill=tuple(fill), outline=tuple(outline))
    if int(width) > 1:
        draw.line(points + (points[0],), fill=tuple(outline), width=int(width), joint="curve")
    return _bbox_from_points(points)


def _block_fill(theme: MinecraftTheme, kind: str) -> Tuple[int, int, int]:
    if str(kind) == "gold_ore":
        return tuple(theme.stone_rgb)
    if str(kind) == "diamond_ore":
        return tuple(theme.stone_rgb)
    if str(kind) == "stone":
        return tuple(theme.stone_rgb)
    return tuple(theme.support_rgb)


def _draw_block(
    draw: ImageDraw.ImageDraw,
    *,
    block: MinecraftBlock,
    theme: MinecraftTheme,
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
) -> BBox:
    """Draw one cube with shaded faces and resource markings on the top face."""

    x = int(block.x)
    y = int(block.y)
    z = int(block.z)
    top = _cell_top_points(x=x, y=y, z=float(z + 1), origin=origin, params=params)
    bottom = _cell_top_points(x=x, y=y, z=float(z), origin=origin, params=params)
    right_face = (top[1], top[2], bottom[2], bottom[1])
    left_face = (top[2], top[3], bottom[3], bottom[2])
    fill = _block_fill(theme, str(block.kind))
    draw.polygon(right_face, fill=_shade(fill, 0.80), outline=tuple(theme.outline_rgb))
    draw.polygon(left_face, fill=_shade(fill, 0.66), outline=tuple(theme.outline_rgb))
    draw.polygon(top, fill=_tint(fill, 0.18), outline=tuple(theme.outline_rgb))

    if str(block.kind) == "gold_ore":
        cx = sum(point[0] for point in top) / 4.0
        cy = sum(point[1] for point in top) / 4.0
        for dx, dy in ((-8, -2), (5, 2), (0, -8), (9, -6)):
            draw.ellipse((cx + dx - 3, cy + dy - 3, cx + dx + 3, cy + dy + 3), fill=tuple(theme.gold_rgb))
    elif str(block.kind) == "diamond_ore":
        cx = sum(point[0] for point in top) / 4.0
        cy = sum(point[1] for point in top) / 4.0
        for dx, dy in ((-9, -3), (3, 0), (8, -7), (-1, -10)):
            draw.polygon(
                ((cx + dx, cy + dy - 4), (cx + dx + 4, cy + dy), (cx + dx, cy + dy + 4), (cx + dx - 4, cy + dy)),
                fill=tuple(theme.diamond_rgb),
                outline=tuple(theme.outline_rgb),
            )
    return _merge_bbox(_bbox_from_points(top), _bbox_from_points(bottom))


def _draw_ladder(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    height: int,
    theme: MinecraftTheme,
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
) -> BBox:
    bboxes: list[BBox] = []
    for z in range(max(1, int(height))):
        top = _cell_top_points(x=int(x), y=int(y), z=float(z + 1), origin=origin, params=params)
        bottom = _cell_top_points(x=int(x), y=int(y), z=float(z), origin=origin, params=params)
        face = (top[2], top[3], bottom[3], bottom[2])
        bboxes.append(_bbox_from_points(face))
        left = (
            (float(face[0][0]) * 0.70) + (float(face[1][0]) * 0.30),
            (float(face[0][1]) * 0.70) + (float(face[1][1]) * 0.30),
        )
        right = (
            (float(face[0][0]) * 0.30) + (float(face[1][0]) * 0.70),
            (float(face[0][1]) * 0.30) + (float(face[1][1]) * 0.70),
        )
        left_b = (
            (float(face[2][0]) * 0.30) + (float(face[3][0]) * 0.70),
            (float(face[2][1]) * 0.30) + (float(face[3][1]) * 0.70),
        )
        right_b = (
            (float(face[2][0]) * 0.70) + (float(face[3][0]) * 0.30),
            (float(face[2][1]) * 0.70) + (float(face[3][1]) * 0.30),
        )
        draw.line((left, left_b), fill=tuple(theme.ladder_rgb), width=3)
        draw.line((right, right_b), fill=tuple(theme.ladder_rgb), width=3)
        for t in (0.25, 0.50, 0.75):
            p0 = ((left[0] * (1 - t)) + (left_b[0] * t), (left[1] * (1 - t)) + (left_b[1] * t))
            p1 = ((right[0] * (1 - t)) + (right_b[0] * t), (right[1] * (1 - t)) + (right_b[1] * t))
            draw.line((p0, p1), fill=tuple(theme.ladder_rgb), width=3)
    return _merge_bbox(*bboxes) if bboxes else (0.0, 0.0, 0.0, 0.0)


def _draw_player(
    draw: ImageDraw.ImageDraw,
    *,
    x: int,
    y: int,
    theme: MinecraftTheme,
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
) -> BBox:
    """Draw the optional player marker as a fitted label anchored to a cell center."""

    center = _project(x=float(x) + 0.5, y=float(y) + 0.5, z=0.0, origin=origin, params=params)
    size = float(params.player_marker_size_px)
    label = str(params.player_marker_label or "P")
    marker_width = size if len(label) <= 2 else min(size * 2.7, max(size, 9.0 * float(len(label)) + 12.0))
    bbox = (
        float(center[0] - marker_width / 2.0),
        float(center[1] - size / 2.0),
        float(center[0] + marker_width / 2.0),
        float(center[1] + size / 2.0),
    )
    draw.rounded_rectangle(bbox, radius=5, fill=tuple(theme.player_rgb), outline=tuple(theme.outline_rgb), width=2)
    font = fit_font_to_box(
        draw,
        text=label,
        max_width=max(8.0, marker_width - 8.0),
        max_height=max(8.0, size - 6.0),
        bold=True,
        min_size_px=10,
        max_size_px=18,
        font_family=str(params.font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), label, font=font)
    draw_text_traced(draw,
        (
            float(center[0]) - float(text_bbox[2] - text_bbox[0]) / 2.0 - float(text_bbox[0]),
            float(center[1]) - float(text_bbox[3] - text_bbox[1]) / 2.0 - float(text_bbox[1]),
        ),
        label,
        font=font,
        fill=(255, 255, 255),
     role="readout", required=False,)
    return tuple(round(float(v), 3) for v in bbox)  # type: ignore[return-value]


def _draw_direction_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start_cell: Tuple[int, int],
    end_cell: Tuple[int, int],
    theme: MinecraftTheme,
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
) -> None:
    sx, sy = start_cell
    ex, ey = end_cell
    start = _project(x=float(sx) + 0.5, y=float(sy) + 0.5, z=0.04, origin=origin, params=params)
    end = _project(x=float(ex) + 0.5, y=float(ey) + 0.5, z=0.04, origin=origin, params=params)
    mid = ((float(start[0]) * 0.18) + (float(end[0]) * 0.82), (float(start[1]) * 0.18) + (float(end[1]) * 0.82))
    draw.line((start, mid), fill=tuple(theme.arrow_rgb), width=5)
    dx = float(mid[0] - start[0])
    dy = float(mid[1] - start[1])
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    head = (
        mid,
        (mid[0] - ux * 14.0 + px * 7.0, mid[1] - uy * 14.0 + py * 7.0),
        (mid[0] - ux * 14.0 - px * 7.0, mid[1] - uy * 14.0 - py * 7.0),
    )
    draw.polygon(head, fill=tuple(theme.arrow_rgb))


def _draw_route_overlay(
    draw: ImageDraw.ImageDraw,
    *,
    route: MinecraftRouteOverlay,
    theme: MinecraftTheme,
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
    draw_label: bool = True,
) -> None:
    """Draw one visible route cue across terrain cells."""

    if not route.cells:
        return
    centers = [
        _project(x=float(x) + 0.5, y=float(y) + 0.5, z=0.07, origin=origin, params=params)
        for x, y in route.cells
    ]
    color = tuple(int(v) for v in route.rgb)
    if len(centers) >= 2:
        for index, center in enumerate(centers):
            left = centers[max(0, index - 1)]
            right = centers[min(len(centers) - 1, index + 1)]
            dx = float(right[0] - left[0])
            dy = float(right[1] - left[1])
            length = max(1.0, (dx * dx + dy * dy) ** 0.5)
            px = -dy / length
            py = dx / length
            tie_half = 12.0
            draw.line(
                (
                    (float(center[0]) - px * tie_half, float(center[1]) - py * tie_half),
                    (float(center[0]) + px * tie_half, float(center[1]) + py * tie_half),
                ),
                fill=tuple(theme.outline_rgb),
                width=3,
            )
        draw.line(centers, fill=tuple(theme.outline_rgb), width=11, joint="curve")
        draw.line(centers, fill=color, width=7, joint="curve")
    else:
        cx, cy = centers[0]
        draw.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), fill=color, outline=tuple(theme.outline_rgb), width=2)
    if not bool(draw_label):
        return
    _draw_route_label(draw, route=route, theme=theme, origin=origin, params=params)


def _draw_route_label(
    draw: ImageDraw.ImageDraw,
    *,
    route: MinecraftRouteOverlay,
    theme: MinecraftTheme,
    origin: Tuple[float, float],
    params: MinecraftRenderParams,
) -> None:
    """Draw one route label just before the first route cell."""

    label = str(route.label).strip()
    if not label or not route.cells:
        return
    start_x = float(route.cells[0][0]) + 0.5
    start_y = float(route.cells[0][1]) + 0.5
    if len(route.cells) >= 2:
        next_x = float(route.cells[1][0]) + 0.5
        next_y = float(route.cells[1][1]) + 0.5
        offset_x = start_x - next_x
        offset_y = start_y - next_y
    else:
        offset_x = -1.0
        offset_y = 0.0
    lx, ly = _project(
        x=start_x + (offset_x * 1.05),
        y=start_y + (offset_y * 1.05),
        z=0.45,
        origin=origin,
        params=params,
    )
    color = tuple(int(v) for v in route.rgb)
    label_box = (float(lx - 16.0), float(ly - 16.0), float(lx + 16.0), float(ly + 16.0))
    draw.rounded_rectangle(label_box, radius=6, fill=color, outline=tuple(theme.outline_rgb), width=3)
    font = fit_font_to_box(
        draw,
        text=label,
        max_width=25,
        max_height=24,
        bold=True,
        min_size_px=11,
        max_size_px=19,
        font_family=str(params.font_family) or None,
    )
    text_bbox = draw.textbbox((0, 0), label, font=font)
    draw_text_traced(draw,
        (
            float((label_box[0] + label_box[2]) / 2.0) - float(text_bbox[2] - text_bbox[0]) / 2.0 - float(text_bbox[0]),
            float((label_box[1] + label_box[3]) / 2.0) - float(text_bbox[3] - text_bbox[1]) / 2.0 - float(text_bbox[1]),
        ),
        label,
        font=font,
        fill=(255, 255, 255),
     role="readout", required=False,)


def render_minecraft_block_world_scene(
    *,
    grid_width: int,
    grid_depth: int,
    terrain_cells: Sequence[MinecraftCell],
    blocks: Sequence[MinecraftBlock],
    player_cell: Tuple[int, int] | None,
    style_variant: str,
    background: Image.Image,
    params: MinecraftRenderParams,
    ladder_column: Tuple[int, int, int] | None = None,
    ladder_columns: Sequence[Tuple[int, int, int]] | None = None,
    target_cell: Tuple[int, int] | None = None,
    route_overlays: Sequence[MinecraftRouteOverlay] | None = None,
) -> RenderedMinecraftScene:
    """Render terrain, overlays, stacks, labels, and trace geometry in isometric order."""

    if str(style_variant) not in STYLE_VARIANTS:
        raise ValueError(f"unsupported minecraft style: {style_variant}")
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_games_minecraft_theme(style_variant=str(style_variant))
    origin = _grid_origin(grid_width=int(grid_width), grid_depth=int(grid_depth), params=params)
    route_overlays = tuple(route_overlays or ())
    route_anchor_colors = (
        tuple(theme.ground_rgb),
        tuple(theme.ground_alt_rgb),
        tuple(theme.water_rgb),
        tuple(theme.outline_rgb),
        tuple(theme.support_rgb),
        tuple(theme.stone_rgb),
    )
    resolved_route_rgbs = resolve_contrasting_palette(
        tuple(tuple(int(channel) for channel in route.rgb) for route in route_overlays),
        anchor_colors=route_anchor_colors,
        min_anchor_distance=42.0,
        min_pairwise_distance=28.0,
    )
    resolved_route_overlays = tuple(
        MinecraftRouteOverlay(label=str(route.label), cells=tuple(route.cells), rgb=tuple(rgb))
        for route, rgb in zip(route_overlays, resolved_route_rgbs)
    )

    entity_bboxes: Dict[str, BBox] = {}
    scene_entities: list[Dict[str, Any]] = []
    terrain_by_xy = {(int(cell.x), int(cell.y)): cell for cell in terrain_cells}
    for y in range(int(grid_depth)):
        for x in range(int(grid_width)):
            cell = terrain_by_xy.get((x, y))
            kind = str(cell.kind) if cell is not None else "grass"
            if kind == "water":
                fill = theme.water_rgb
                outline = theme.water_line_rgb
                line_width = 2
            elif kind in {"tunnel_path", "route_path"}:
                fill = _tint(theme.support_rgb, 0.48)
                outline = tuple(theme.arrow_rgb)
                line_width = 2
            else:
                fill = theme.ground_alt_rgb if (x + y) % 2 else theme.ground_rgb
                outline = theme.outline_rgb
                line_width = 1
            bbox = _draw_tile(
                draw,
                x=x,
                y=y,
                fill=fill,
                outline=outline,
                origin=origin,
                params=params,
                width=line_width,
            )
            if cell is not None:
                entity_bboxes[str(cell.cell_id)] = bbox
                scene_entities.append(
                    {
                        "entity_id": str(cell.cell_id),
                        "entity_type": f"minecraft_{kind}_cell",
                        "x": int(x),
                        "y": int(y),
                        "bbox_px": list(bbox),
                    }
                )

    for route in resolved_route_overlays:
        _draw_route_overlay(draw, route=route, theme=theme, origin=origin, params=params, draw_label=False)

    sorted_blocks = sorted(blocks, key=lambda block: (int(block.x) + int(block.y) + int(block.z), int(block.y), int(block.x), int(block.z)))
    stack_bboxes: dict[Tuple[int, int], list[BBox]] = {}
    stack_z_values: dict[Tuple[int, int], list[int]] = {}
    stack_top_points: dict[Tuple[int, int], Tuple[int, Point2]] = {}
    entity_points: Dict[str, Point2] = {}
    for block in sorted_blocks:
        block_top = _cell_top_points(x=int(block.x), y=int(block.y), z=float(int(block.z) + 1), origin=origin, params=params)
        block_point = _center_of_points(block_top)
        bbox = _draw_block(draw, block=block, theme=theme, origin=origin, params=params)
        entity_bboxes[str(block.block_id)] = bbox
        entity_points[str(block.block_id)] = block_point
        stack_key = (int(block.x), int(block.y))
        stack_bboxes.setdefault(stack_key, []).append(bbox)
        stack_z_values.setdefault(stack_key, []).append(int(block.z))
        if stack_key not in stack_top_points or int(block.z) >= int(stack_top_points[stack_key][0]):
            stack_top_points[stack_key] = (int(block.z), block_point)
        scene_entities.append(
            {
                "entity_id": str(block.block_id),
                "entity_type": f"minecraft_{str(block.kind)}_block",
                "x": int(block.x),
                "y": int(block.y),
                "z": int(block.z),
                "kind": str(block.kind),
                "point_px": list(block_point),
                "bbox_px": list(bbox),
            }
        )
    for (stack_x, stack_y), bboxes in sorted(stack_bboxes.items()):
        merged_bbox = _merge_bbox(*bboxes)
        entity_id = stack_entity_id(int(stack_x), int(stack_y))
        z_values = tuple(sorted(set(stack_z_values.get((int(stack_x), int(stack_y)), []))))
        stack_top = stack_top_points.get((int(stack_x), int(stack_y)))
        if stack_top is None:
            stack_point = _center_of_points(
                (
                    (float(merged_bbox[0]), float(merged_bbox[1])),
                    (float(merged_bbox[2]), float(merged_bbox[1])),
                    (float(merged_bbox[2]), float(merged_bbox[3])),
                    (float(merged_bbox[0]), float(merged_bbox[3])),
                )
            )
        else:
            stack_point = stack_top[1]
        entity_bboxes[entity_id] = merged_bbox
        entity_points[entity_id] = stack_point
        scene_entities.append(
            {
                "entity_id": entity_id,
                "entity_type": "minecraft_block_stack",
                "x": int(stack_x),
                "y": int(stack_y),
                "height": int(len(z_values)),
                "z_values": [int(value) for value in z_values],
                "point_px": list(stack_point),
                "bbox_px": list(merged_bbox),
            }
        )

    all_ladder_columns: list[Tuple[int, int, int]] = []
    if ladder_column is not None:
        all_ladder_columns.append(ladder_column)
    all_ladder_columns.extend(tuple(column) for column in (ladder_columns or ()))
    for ladder_index, ladder_column_item in enumerate(all_ladder_columns):
        lx, ly, height = ladder_column_item
        bbox = _draw_ladder(draw, x=int(lx), y=int(ly), height=int(height), theme=theme, origin=origin, params=params)
        entity_id = ladder_entity_id() if int(ladder_index) == 0 else f"{ladder_entity_id()}_{int(ladder_index):02d}"
        entity_bboxes[entity_id] = bbox
        scene_entities.append(
            {
                "entity_id": entity_id,
                "entity_type": "minecraft_ladder",
                "x": int(lx),
                "y": int(ly),
                "height": int(height),
                "bbox_px": list(bbox),
            }
        )

    if player_cell is not None:
        px, py = player_cell
        if target_cell is not None:
            _draw_direction_arrow(
                draw,
                start_cell=(int(px), int(py)),
                end_cell=(int(target_cell[0]), int(target_cell[1])),
                theme=theme,
                origin=origin,
                params=params,
            )
        bbox = _draw_player(draw, x=int(px), y=int(py), theme=theme, origin=origin, params=params)
        entity_bboxes[player_entity_id()] = bbox
        scene_entities.append(
            {
                "entity_id": player_entity_id(),
                "entity_type": "minecraft_player_marker",
                "x": int(px),
                "y": int(py),
                "bbox_px": list(bbox),
            }
        )

    for route in resolved_route_overlays:
        _draw_route_label(draw, route=route, theme=theme, origin=origin, params=params)

    return RenderedMinecraftScene(
        image=image,
        scene_entities=tuple(scene_entities),
        render_map={
            "entity_bboxes_px": {str(key): list(value) for key, value in entity_bboxes.items()},
            "entity_points_px": {str(key): list(value) for key, value in entity_points.items()},
            "grid_width": int(grid_width),
            "grid_depth": int(grid_depth),
            "style_variant": str(style_variant),
            "view_projection": "isometric_block_world",
            "route_overlays": [
                {
                    "label": str(route.label),
                    "cells": [list(cell) for cell in route.cells],
                    "rgb": list(route.rgb),
                }
                for route in resolved_route_overlays
            ],
            "route_color_policy": {
                "min_anchor_distance_lab": 42.0,
                "min_pairwise_distance_lab": 28.0,
                "anchors_rgb": [list(color) for color in route_anchor_colors],
            },
            "text_style": {"font_family": str(params.font_family)},
            "layout_jitter": dict(params.layout_jitter_meta or {}),
        },
    )


__all__ = [
    "MinecraftRenderParams",
    "MinecraftTheme",
    "RenderedMinecraftScene",
    "build_games_minecraft_theme",
    "render_minecraft_block_world_scene",
    "resolve_minecraft_render_params",
]
