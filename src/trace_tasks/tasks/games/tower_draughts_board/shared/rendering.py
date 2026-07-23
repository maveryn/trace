"""Rendering helpers for tower draughts board scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.marking import draw_optional_marker_x, draw_semantic_ellipse_marker, resolve_semantic_marker_style
from trace_tasks.tasks.games.shared.scene_style import draw_panel_scene_chrome, make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import DEFAULTS, RENDER_DEFAULTS
from .rules import cell_id, playable_coords, player_name, stack_id
from .state import BLACK, RED, SCENE_NAMESPACE, STYLE_VARIANTS, StackSpec, TowerDraughtsAxes, TowerDraughtsSample, TowerDraughtsTheme, RenderedTowerDraughtsScene


def theme_for_style(style_variant: str) -> Tuple[TowerDraughtsTheme, dict[str, Any]]:
    """Return a scene-local board/token palette for one style variant."""

    themes: dict[str, TowerDraughtsTheme] = {
        "wood_table": TowerDraughtsTheme(
            board_fill_rgb=(206, 156, 96),
            board_border_rgb=(97, 64, 35),
            light_cell_rgb=(235, 202, 151),
            dark_cell_rgb=(122, 82, 48),
            playable_outline_rgb=(72, 48, 28),
            red_piece_rgb=(207, 54, 58),
            red_piece_outline_rgb=(86, 24, 31),
            black_piece_rgb=(38, 44, 54),
            black_piece_outline_rgb=(225, 231, 238),
            crown_rgb=(249, 218, 87),
            crown_outline_rgb=(82, 55, 16),
        ),
        "ink_board": TowerDraughtsTheme(
            board_fill_rgb=(239, 238, 228),
            board_border_rgb=(52, 53, 58),
            light_cell_rgb=(247, 246, 237),
            dark_cell_rgb=(188, 190, 190),
            playable_outline_rgb=(58, 60, 66),
            red_piece_rgb=(180, 55, 63),
            red_piece_outline_rgb=(73, 28, 35),
            black_piece_rgb=(35, 38, 45),
            black_piece_outline_rgb=(242, 244, 247),
            crown_rgb=(250, 220, 96),
            crown_outline_rgb=(70, 52, 22),
        ),
        "felt_mat": TowerDraughtsTheme(
            board_fill_rgb=(63, 107, 74),
            board_border_rgb=(34, 63, 42),
            light_cell_rgb=(212, 229, 201),
            dark_cell_rgb=(84, 137, 92),
            playable_outline_rgb=(34, 82, 50),
            red_piece_rgb=(214, 74, 75),
            red_piece_outline_rgb=(86, 30, 33),
            black_piece_rgb=(38, 50, 51),
            black_piece_outline_rgb=(233, 245, 236),
            crown_rgb=(255, 229, 102),
            crown_outline_rgb=(82, 65, 17),
        ),
        "night_tokens": TowerDraughtsTheme(
            board_fill_rgb=(34, 42, 60),
            board_border_rgb=(177, 194, 212),
            light_cell_rgb=(73, 89, 117),
            dark_cell_rgb=(25, 32, 48),
            playable_outline_rgb=(164, 191, 217),
            red_piece_rgb=(239, 83, 93),
            red_piece_outline_rgb=(255, 224, 230),
            black_piece_rgb=(15, 20, 31),
            black_piece_outline_rgb=(232, 241, 255),
            crown_rgb=(255, 222, 78),
            crown_outline_rgb=(42, 31, 8),
        ),
        "parchment": TowerDraughtsTheme(
            board_fill_rgb=(229, 208, 166),
            board_border_rgb=(118, 84, 45),
            light_cell_rgb=(248, 232, 194),
            dark_cell_rgb=(150, 101, 62),
            playable_outline_rgb=(93, 67, 39),
            red_piece_rgb=(177, 62, 50),
            red_piece_outline_rgb=(77, 32, 27),
            black_piece_rgb=(52, 48, 43),
            black_piece_outline_rgb=(246, 238, 219),
            crown_rgb=(247, 207, 72),
            crown_outline_rgb=(85, 57, 20),
        ),
    }
    resolved = str(style_variant) if str(style_variant) in themes else "wood_table"
    return themes[resolved], {
        "style_variant": str(resolved),
        "available_styles": list(STYLE_VARIANTS),
        "board_style_policy": "scene_local_tower_draughts_board_palette",
    }


def bbox_from_center(center: Sequence[float], radius: float) -> Tuple[float, float, float, float]:
    """Return a square bbox centered at a point."""

    cx, cy = float(center[0]), float(center[1])
    return (
        round(cx - float(radius), 3),
        round(cy - float(radius), 3),
        round(cx + float(radius), 3),
        round(cy + float(radius), 3),
    )


def draw_stack(
    draw: ImageDraw.ImageDraw,
    *,
    stack: StackSpec,
    cell_bbox: Sequence[float],
    theme: TowerDraughtsTheme,
    disk_radius: float,
) -> tuple[list[float], list[float]]:
    """Draw one visible stack and return top-center point plus stack bbox."""

    cx = 0.5 * (float(cell_bbox[0]) + float(cell_bbox[2]))
    cy = 0.5 * (float(cell_bbox[1]) + float(cell_bbox[3]))
    layer_offset = max(3.0, min(7.0, float(disk_radius) * 0.22))
    top_center = (float(cx), float(cy) - (0.5 * (int(stack.height) - 1) * layer_offset))
    all_bboxes: list[Tuple[float, float, float, float]] = []
    for index, player in enumerate(stack.disks):
        layer_y = float(cy) + (0.5 * (int(stack.height) - 1) * layer_offset) - (float(index) * layer_offset)
        fill = theme.red_piece_rgb if int(player) == RED else theme.black_piece_rgb
        outline = theme.red_piece_outline_rgb if int(player) == RED else theme.black_piece_outline_rgb
        bbox = bbox_from_center((cx, layer_y), float(disk_radius))
        shadow = (bbox[0] + 2.0, bbox[1] + 2.0, bbox[2] + 2.0, bbox[3] + 2.0)
        draw.ellipse(shadow, fill=(0, 0, 0, 46))
        draw.ellipse(
            bbox,
            fill=tuple(fill) + (255,),
            outline=tuple(outline) + (255,),
            width=max(2, int(round(float(disk_radius) * 0.13))),
        )
        inner = bbox_from_center((cx, layer_y), float(disk_radius) * 0.62)
        draw.ellipse(inner, outline=tuple(outline) + (150,), width=max(1, int(round(float(disk_radius) * 0.05))))
        all_bboxes.append(bbox)
    if bool(stack.top_crowned):
        crown_radius = float(disk_radius) * 0.38
        tx, ty = top_center
        points = (
            (tx - crown_radius * 0.95, ty + crown_radius * 0.20),
            (tx - crown_radius * 0.50, ty - crown_radius * 0.52),
            (tx, ty + crown_radius * 0.04),
            (tx + crown_radius * 0.50, ty - crown_radius * 0.52),
            (tx + crown_radius * 0.95, ty + crown_radius * 0.20),
            (tx + crown_radius * 0.72, ty + crown_radius * 0.54),
            (tx - crown_radius * 0.72, ty + crown_radius * 0.54),
        )
        draw.polygon(points, fill=tuple(theme.crown_rgb) + (245,), outline=tuple(theme.crown_outline_rgb) + (255,))
    x0 = min(bbox[0] for bbox in all_bboxes)
    y0 = min(bbox[1] for bbox in all_bboxes)
    x1 = max(bbox[2] for bbox in all_bboxes)
    y1 = max(bbox[3] for bbox in all_bboxes)
    return [round(float(cx), 3), round(float(top_center[1]), 3)], [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


def render_tower_draughts_scene(
    *,
    sample: TowerDraughtsSample,
    axes: TowerDraughtsAxes,
    instance_seed: int,
    params: Mapping[str, Any],
) -> RenderedTowerDraughtsScene:
    """Render one tower draughts board and expose cell/stack geometry maps."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.layout",
        ),
        unit_scale_meta,
    )
    cell_min = int(group_default(RENDER_DEFAULTS, "cell_size_min_px", DEFAULTS.cell_size_min_px))
    cell_max = int(group_default(RENDER_DEFAULTS, "cell_size_max_px", DEFAULTS.cell_size_max_px))
    scaled_min = scale_games_px(cell_min, unit_scale, min_px=48)
    scaled_max = scale_games_px(cell_max, unit_scale, min_px=max(52, scaled_min))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.render")
    cell_size = int(rng.randint(min(scaled_min, scaled_max), max(scaled_min, scaled_max)))
    board_px = int(cell_size) * int(sample.board_size)
    base_canvas_width = int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", DEFAULTS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", DEFAULTS.canvas_height)))
    dynamic_canvas_enabled = bool(params.get("dynamic_canvas_size_enabled", group_default(RENDER_DEFAULTS, "dynamic_canvas_size_enabled", DEFAULTS.dynamic_canvas_size_enabled)))
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(RENDER_DEFAULTS, "canvas_min_width_px", DEFAULTS.canvas_min_width_px))),
                int(round(float(board_px) + (2.0 * float(params.get("canvas_side_padding_px", group_default(RENDER_DEFAULTS, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px)))))),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(RENDER_DEFAULTS, "canvas_min_height_px", DEFAULTS.canvas_min_height_px))),
                int(round(float(board_px) + (2.0 * float(params.get("canvas_vertical_padding_px", group_default(RENDER_DEFAULTS, "canvas_vertical_padding_px", DEFAULTS.canvas_vertical_padding_px)))))),
            ),
        )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_scene_style",
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(RENDER_DEFAULTS, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(RENDER_DEFAULTS, "panel_scene_palette_weights", None)),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=panel_style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme, theme_meta = theme_for_style(str(axes.style_variant))
    board_bbox = (
        round(0.5 * (float(canvas_width) - float(board_px)), 3),
        round(0.5 * (float(canvas_height) - float(board_px)), 3),
        round(0.5 * (float(canvas_width) + float(board_px)), 3),
        round(0.5 * (float(canvas_height) + float(board_px)), 3),
    )
    board_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    frame_width = scale_games_px(
        group_default(RENDER_DEFAULTS, "board_frame_width_px", DEFAULTS.board_frame_width_px),
        unit_scale,
        min_px=3,
    )
    panel_pad = max(18, int(round(float(cell_size) * 0.26)))
    panel_bbox = (
        int(round(float(board_bbox[0]) - panel_pad)),
        int(round(float(board_bbox[1]) - panel_pad)),
        int(round(float(board_bbox[2]) + panel_pad)),
        int(round(float(board_bbox[3]) + panel_pad)),
    )
    draw_panel_scene_chrome(draw, bbox=panel_bbox, style=panel_style, radius=24, border_width=2)
    draw.rounded_rectangle(
        panel_bbox,
        radius=22,
        fill=tuple(theme.board_fill_rgb) + (230,),
        outline=tuple(theme.board_border_rgb) + (255,),
        width=max(2, int(frame_width)),
    )
    cell_bboxes: dict[str, list[float]] = {}
    cell_centers: dict[str, list[float]] = {}
    for row in range(int(sample.board_size)):
        for col in range(int(sample.board_size)):
            resolved_cell_id = cell_id((row, col))
            x0 = float(board_bbox[0]) + (float(col) * float(cell_size))
            y0 = float(board_bbox[1]) + (float(row) * float(cell_size))
            bbox = (x0, y0, x0 + float(cell_size), y0 + float(cell_size))
            playable = (row + col) % 2 == 1
            fill = theme.dark_cell_rgb if playable else theme.light_cell_rgb
            draw.rectangle(
                bbox,
                fill=tuple(fill) + (255,),
                outline=tuple(theme.playable_outline_rgb if playable else theme.board_border_rgb) + (200,),
                width=1,
            )
            cell_bboxes[resolved_cell_id] = [round(float(value), 3) for value in bbox]
            cell_centers[resolved_cell_id] = [round(x0 + (0.5 * float(cell_size)), 3), round(y0 + (0.5 * float(cell_size)), 3)]
    stack_centers: dict[str, list[float]] = {}
    stack_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    disk_radius = max(12.0, float(cell_size) * 0.31)
    stack_by_coord = {tuple(stack.coord): stack for stack in sample.stacks}
    for coord in playable_coords(int(sample.board_size)):
        resolved_cell_id = cell_id(coord)
        stack = stack_by_coord.get(tuple(coord))
        entity = {
            "entity_id": str(resolved_cell_id),
            "entity_type": "tower_draughts_cell",
            "row": int(coord[0]),
            "col": int(coord[1]),
            "playable": True,
            "state": "empty" if stack is None else "occupied",
            "center_px": list(cell_centers[resolved_cell_id]),
            "bbox_px": list(cell_bboxes[resolved_cell_id]),
            "stack_id": "",
        }
        if stack is not None:
            resolved_stack_id = "stack_marked" if sample.marked_coord == tuple(coord) else stack_id(coord)
            center, bbox = draw_stack(
                draw,
                stack=stack,
                cell_bbox=cell_bboxes[resolved_cell_id],
                theme=theme,
                disk_radius=float(disk_radius),
            )
            stack_centers[resolved_stack_id] = list(center)
            stack_bboxes[resolved_stack_id] = list(bbox)
            entity.update(
                {
                    "stack_id": str(resolved_stack_id),
                    "stack_height": int(stack.height),
                    "stack_owner": player_name(int(stack.owner)),
                    "top_crowned": bool(stack.top_crowned),
                    "stack_center_px": list(center),
                    "stack_bbox_px": list(bbox),
                }
            )
        entities.append(entity)
    marker_metadata: dict[str, Any] | None = None
    if sample.marked_coord is not None and "stack_marked" in stack_bboxes:
        marker_width = scale_games_px(
            group_default(RENDER_DEFAULTS, "marker_width_px", DEFAULTS.marker_width_px),
            unit_scale,
            min_px=3,
        )
        marked_bbox = stack_bboxes["stack_marked"]
        marker_pad = max(4.0, float(marker_width) * 1.4)
        marker_bbox = (
            float(marked_bbox[0]) - marker_pad,
            float(marked_bbox[1]) - marker_pad,
            float(marked_bbox[2]) + marker_pad,
            float(marked_bbox[3]) + marker_pad,
        )
        marker_style = resolve_semantic_marker_style(
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.marked_stack",
            role="marked_stack",
            surface_rgbs=(theme.board_fill_rgb, theme.dark_cell_rgb),
            preferred_rgbs=((255, 214, 38), (255, 247, 92), (246, 80, 164), (36, 205, 228)),
        )
        marker_metadata = draw_semantic_ellipse_marker(
            draw,
            marker_bbox,
            style=marker_style,
            width=max(3, int(marker_width)),
            marker_kind="marked_stack_ring",
            extra_metadata={"stack_id": "stack_marked"},
        )
        x_metadata = draw_optional_marker_x(
            draw,
            marked_bbox,
            enabled=True,
            width=max(3, int(round(float(marker_width) * 0.72))),
            inset_fraction=0.25,
            marker_kind="marked_stack_x",
            extra_metadata={"stack_id": "stack_marked"},
        )
        if x_metadata is not None:
            marker_metadata = {**dict(marker_metadata), "overlay_x": dict(x_metadata)}
    render_map = {
        "board_bbox_px": [round(float(value), 3) for value in board_bbox],
        "panel_bbox_px": [float(value) for value in panel_bbox],
        "cell_bboxes_px": dict(cell_bboxes),
        "cell_centers_px": dict(cell_centers),
        "stack_centers_px": dict(stack_centers),
        "stack_bboxes_px": dict(stack_bboxes),
        "marked_stack_marker": marker_metadata,
        "layout_jitter": dict(resolved_jitter),
        "effective_cell_size_px": int(cell_size),
        "effective_disk_radius_px": round(float(disk_radius), 3),
    }
    return RenderedTowerDraughtsScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=render_map,
        style_meta={
            "panel_scene_style": dict(panel_style_meta),
            "tower_draughts_board_style": dict(theme_meta),
        },
        background_meta=dict(background_meta),
    )


__all__ = [
    "bbox_from_center",
    "draw_stack",
    "render_tower_draughts_scene",
    "theme_for_style",
]
