"""Shared Pac-Man maze renderer for games-domain tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.text import draw_centered_game_text as draw_centered_text
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .sampling import PacmanVisualAxes
from .state import (
    Coord,
    PacmanGhost,
    PacmanItem,
    ghost_entity_id,
    item_entity_id,
    pacman_entity_id,
    pellet_entity_id,
    route_entity_id,
    sorted_coords,
)
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata


@dataclass(frozen=True)
class PacmanRenderParams:
    """Resolved render controls for one Pac-Man scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    maze_width_px: int
    maze_height_px: int
    wall_gap_px: int
    wall_outline_width_px: int
    pellet_radius_px: int
    item_radius_px: int
    ghost_radius_px: int
    route_width_px: int
    item_label_font_size_px: int
    font_family: str = ""
    layout_jitter_meta: Dict[str, Any] | None = None


@dataclass(frozen=True)
class PacmanTheme:
    """Resolved Pac-Man visual palette."""

    panel_fill_rgb: Tuple[int, int, int]
    panel_outline_rgb: Tuple[int, int, int]
    wall_fill_rgb: Tuple[int, int, int]
    wall_outline_rgb: Tuple[int, int, int]
    corridor_fill_rgb: Tuple[int, int, int]
    pellet_fill_rgb: Tuple[int, int, int]
    route_fill_rgb: Tuple[int, int, int]
    route_outline_rgb: Tuple[int, int, int]
    pacman_fill_rgb: Tuple[int, int, int]
    pacman_outline_rgb: Tuple[int, int, int]
    item_label_rgb: Tuple[int, int, int]
    item_label_outline_rgb: Tuple[int, int, int]
    item_palette_rgb: Dict[str, Tuple[int, int, int]]
    ghost_palette_rgb: Dict[str, Tuple[int, int, int]]
    ghost_eye_rgb: Tuple[int, int, int]
    ghost_pupil_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedPacmanScene:
    """Rendered Pac-Man image plus trace-friendly geometry."""

    image: Image.Image
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedPacmanTaskContext:
    """Rendered Pac-Man image plus shared style metadata for trace output."""

    image: Image.Image
    rendered_scene: RenderedPacmanScene
    panel_style_meta: Dict[str, Any]
    text_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


def resolve_pacman_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> PacmanRenderParams:
    """Resolve Pac-Man rendering parameters from config/defaults."""

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
    maze_width_px = scale_games_px(
        params.get("maze_width_px", group_default(render_defaults, "maze_width_px", DEFAULTS.maze_width_px)),
        unit_scale,
        min_px=420,
    )
    maze_height_px = scale_games_px(
        params.get("maze_height_px", group_default(render_defaults, "maze_height_px", DEFAULTS.maze_height_px)),
        unit_scale,
        min_px=310,
    )
    default_canvas_width = int(group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width))
    default_canvas_height = int(group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height))
    canvas_width = int(max(620, min(default_canvas_width, int(maze_width_px) + 220)))
    canvas_height = int(max(500, min(default_canvas_height, int(maze_height_px) + 180)))
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font_family",
        params=params,
    )
    return PacmanRenderParams(
        canvas_width=int(params.get("canvas_width", canvas_width)),
        canvas_height=int(params.get("canvas_height", canvas_height)),
        panel_margin_px=scale_games_px(
            params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px)),
            unit_scale,
            min_px=18,
        ),
        maze_width_px=int(maze_width_px),
        maze_height_px=int(maze_height_px),
        wall_gap_px=scale_games_px(
            params.get("wall_gap_px", group_default(render_defaults, "wall_gap_px", DEFAULTS.wall_gap_px)),
            unit_scale,
            min_px=1,
        ),
        wall_outline_width_px=scale_games_px(
            params.get("wall_outline_width_px", group_default(render_defaults, "wall_outline_width_px", DEFAULTS.wall_outline_width_px)),
            unit_scale,
            min_px=1,
        ),
        pellet_radius_px=scale_games_px(
            params.get("pellet_radius_px", group_default(render_defaults, "pellet_radius_px", DEFAULTS.pellet_radius_px)),
            unit_scale,
            min_px=7,
        ),
        item_radius_px=scale_games_px(
            params.get("item_radius_px", group_default(render_defaults, "item_radius_px", DEFAULTS.item_radius_px)),
            unit_scale,
            min_px=9,
        ),
        ghost_radius_px=scale_games_px(
            params.get("ghost_radius_px", group_default(render_defaults, "ghost_radius_px", DEFAULTS.ghost_radius_px)),
            unit_scale,
            min_px=9,
        ),
        route_width_px=scale_games_px(
            params.get("route_width_px", group_default(render_defaults, "route_width_px", DEFAULTS.route_width_px)),
            unit_scale,
            min_px=4,
        ),
        item_label_font_size_px=scale_games_px(
            params.get("item_label_font_size_px", group_default(render_defaults, "item_label_font_size_px", DEFAULTS.item_label_font_size_px)),
            unit_scale,
            min_px=12,
        ),
        font_family=str(font_family),
        layout_jitter_meta=layout_jitter,
    )


def render_pacman_task_context(
    *,
    axes: PacmanVisualAxes,
    scene,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> RenderedPacmanTaskContext:
    """Render the maze inside the shared games canvas treatment."""

    render_params = resolve_pacman_render_params(
        params,
        render_defaults=render_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene_style",
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(render_defaults, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(render_defaults, "panel_scene_palette_weights", None)),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_pacman_scene(
        row_count=int(scene.row_count),
        col_count=int(scene.col_count),
        open_cells=scene.open_cells,
        wall_cells=scene.wall_cells,
        pacman_coord=scene.pacman_coord,
        route_coords=scene.route_coords,
        pellets=scene.pellets,
        items=scene.items,
        ghosts=scene.ghosts,
        background=background,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    text_style_meta = {
        "font_family": str(render_params.font_family),
        "font_asset": get_font_family_record(str(render_params.font_family)).to_trace(),
    }
    return RenderedPacmanTaskContext(
        image=image,
        rendered_scene=rendered_scene,
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=dict(text_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def build_games_pacman_theme(*, style_variant: str) -> PacmanTheme:
    """Return one Pac-Man maze visual theme."""

    style = str(style_variant)
    if style == "neon":
        return PacmanTheme(
            panel_fill_rgb=(11, 12, 31),
            panel_outline_rgb=(138, 117, 255),
            wall_fill_rgb=(46, 65, 239),
            wall_outline_rgb=(145, 214, 255),
            corridor_fill_rgb=(8, 10, 24),
            pellet_fill_rgb=(255, 238, 166),
            route_fill_rgb=(255, 213, 66),
            route_outline_rgb=(91, 47, 178),
            pacman_fill_rgb=(255, 225, 38),
            pacman_outline_rgb=(255, 255, 212),
            item_label_rgb=(255, 255, 255),
            item_label_outline_rgb=(31, 18, 73),
            item_palette_rgb={
                "cherry": (255, 73, 116),
                "strawberry": (255, 92, 70),
                "orange": (255, 156, 51),
                "bell": (255, 224, 61),
                "melon": (67, 235, 143),
                "key": (99, 218, 255),
            },
            ghost_palette_rgb={
                "red": (255, 65, 86),
                "pink": (255, 132, 206),
                "cyan": (73, 220, 255),
                "orange": (255, 168, 67),
                "purple": (183, 124, 255),
            },
            ghost_eye_rgb=(250, 252, 255),
            ghost_pupil_rgb=(30, 43, 150),
        )
    if style == "paper":
        return PacmanTheme(
            panel_fill_rgb=(240, 231, 206),
            panel_outline_rgb=(74, 65, 48),
            wall_fill_rgb=(87, 101, 164),
            wall_outline_rgb=(51, 57, 98),
            corridor_fill_rgb=(250, 245, 229),
            pellet_fill_rgb=(128, 91, 63),
            route_fill_rgb=(222, 166, 53),
            route_outline_rgb=(97, 70, 41),
            pacman_fill_rgb=(225, 178, 48),
            pacman_outline_rgb=(79, 61, 32),
            item_label_rgb=(34, 30, 24),
            item_label_outline_rgb=(255, 248, 224),
            item_palette_rgb={
                "cherry": (190, 70, 70),
                "strawberry": (204, 85, 80),
                "orange": (211, 126, 57),
                "bell": (214, 177, 67),
                "melon": (92, 153, 88),
                "key": (74, 131, 169),
            },
            ghost_palette_rgb={
                "red": (190, 63, 68),
                "pink": (204, 104, 153),
                "cyan": (83, 157, 174),
                "orange": (202, 127, 61),
                "purple": (145, 105, 177),
            },
            ghost_eye_rgb=(251, 246, 228),
            ghost_pupil_rgb=(58, 73, 132),
        )
    if style == "terminal":
        return PacmanTheme(
            panel_fill_rgb=(7, 20, 12),
            panel_outline_rgb=(88, 200, 116),
            wall_fill_rgb=(31, 128, 62),
            wall_outline_rgb=(110, 232, 143),
            corridor_fill_rgb=(2, 15, 9),
            pellet_fill_rgb=(195, 255, 199),
            route_fill_rgb=(166, 239, 116),
            route_outline_rgb=(32, 84, 39),
            pacman_fill_rgb=(220, 255, 91),
            pacman_outline_rgb=(244, 255, 190),
            item_label_rgb=(5, 27, 14),
            item_label_outline_rgb=(223, 255, 224),
            item_palette_rgb={
                "cherry": (118, 234, 141),
                "strawberry": (94, 211, 117),
                "orange": (161, 230, 108),
                "bell": (217, 247, 103),
                "melon": (91, 196, 151),
                "key": (112, 228, 213),
            },
            ghost_palette_rgb={
                "red": (121, 232, 126),
                "pink": (143, 244, 166),
                "cyan": (88, 220, 189),
                "orange": (176, 235, 105),
                "purple": (137, 220, 155),
            },
            ghost_eye_rgb=(224, 255, 228),
            ghost_pupil_rgb=(7, 40, 20),
        )
    if style == "pastel":
        return PacmanTheme(
            panel_fill_rgb=(236, 234, 247),
            panel_outline_rgb=(90, 86, 128),
            wall_fill_rgb=(117, 139, 218),
            wall_outline_rgb=(74, 88, 159),
            corridor_fill_rgb=(248, 247, 255),
            pellet_fill_rgb=(214, 160, 91),
            route_fill_rgb=(246, 191, 76),
            route_outline_rgb=(120, 92, 116),
            pacman_fill_rgb=(249, 212, 75),
            pacman_outline_rgb=(91, 72, 51),
            item_label_rgb=(41, 36, 56),
            item_label_outline_rgb=(255, 250, 235),
            item_palette_rgb={
                "cherry": (222, 104, 132),
                "strawberry": (229, 119, 105),
                "orange": (229, 158, 90),
                "bell": (229, 202, 95),
                "melon": (114, 190, 130),
                "key": (96, 171, 215),
            },
            ghost_palette_rgb={
                "red": (220, 91, 111),
                "pink": (226, 136, 186),
                "cyan": (96, 190, 210),
                "orange": (227, 164, 86),
                "purple": (165, 132, 218),
            },
            ghost_eye_rgb=(255, 250, 252),
            ghost_pupil_rgb=(68, 72, 147),
        )
    return PacmanTheme(
        panel_fill_rgb=(13, 17, 42),
        panel_outline_rgb=(94, 111, 198),
        wall_fill_rgb=(36, 65, 219),
        wall_outline_rgb=(103, 151, 255),
        corridor_fill_rgb=(5, 8, 30),
        pellet_fill_rgb=(255, 226, 171),
        route_fill_rgb=(255, 213, 67),
        route_outline_rgb=(27, 36, 96),
        pacman_fill_rgb=(255, 221, 44),
        pacman_outline_rgb=(255, 246, 171),
        item_label_rgb=(255, 255, 255),
        item_label_outline_rgb=(28, 28, 80),
        item_palette_rgb={
            "cherry": (226, 46, 85),
            "strawberry": (240, 77, 70),
            "orange": (238, 143, 50),
            "bell": (238, 201, 51),
            "melon": (74, 195, 113),
            "key": (76, 180, 229),
        },
        ghost_palette_rgb={
            "red": (233, 46, 66),
            "pink": (235, 112, 186),
            "cyan": (58, 202, 230),
            "orange": (239, 149, 51),
            "purple": (157, 97, 221),
        },
        ghost_eye_rgb=(247, 251, 255),
        ghost_pupil_rgb=(24, 43, 147),
    )


def _grid_geometry(
    *,
    params: PacmanRenderParams,
    row_count: int,
    col_count: int,
) -> Tuple[Tuple[float, float, float, float], float, Dict[Coord, Tuple[float, float]], Dict[str, Any]]:
    """Return maze bbox, cell size, and cell centers."""

    base_bbox = (
        0.5 * (float(params.canvas_width) - float(params.maze_width_px)),
        0.5 * (float(params.canvas_height) - float(params.maze_height_px)),
        0.5 * (float(params.canvas_width) + float(params.maze_width_px)),
        0.5 * (float(params.canvas_height) + float(params.maze_height_px)),
    )
    maze_bbox, _dx, _dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    left, top, right, bottom = maze_bbox
    cell_size = min(float(right - left) / float(col_count), float(bottom - top) / float(row_count))
    used_width = float(cell_size * int(col_count))
    used_height = float(cell_size * int(row_count))
    left = round(float(left + (0.5 * ((right - left) - used_width))), 3)
    top = round(float(top + (0.5 * ((bottom - top) - used_height))), 3)
    right = round(float(left + used_width), 3)
    bottom = round(float(top + used_height), 3)
    centers = {
        (row, col): (
            round(float(left + ((float(col) + 0.5) * cell_size)), 3),
            round(float(top + ((float(row) + 0.5) * cell_size)), 3),
        )
        for row in range(int(row_count))
        for col in range(int(col_count))
    }
    return (left, top, right, bottom), float(cell_size), centers, dict(layout_jitter)


def _cell_bbox(
    coord: Coord,
    *,
    maze_bbox: Sequence[float],
    cell_size: float,
    gap_px: float,
) -> Tuple[float, float, float, float]:
    """Return one inset cell bbox."""

    row, col = int(coord[0]), int(coord[1])
    left, top, _right, _bottom = [float(value) for value in maze_bbox]
    gap = float(gap_px)
    return (
        round(float(left + (float(col) * float(cell_size)) + gap), 3),
        round(float(top + (float(row) * float(cell_size)) + gap), 3),
        round(float(left + ((float(col) + 1.0) * float(cell_size)) - gap), 3),
        round(float(top + ((float(row) + 1.0) * float(cell_size)) - gap), 3),
    )


def _circle_bbox(center: Tuple[float, float], radius: float) -> Tuple[float, float, float, float]:
    """Return a circular entity bbox."""

    cx, cy = float(center[0]), float(center[1])
    return (
        round(float(cx - radius), 3),
        round(float(cy - radius), 3),
        round(float(cx + radius), 3),
        round(float(cy + radius), 3),
    )


def _draw_pacman(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    direction: Tuple[int, int],
    theme: PacmanTheme,
) -> Tuple[float, float, float, float]:
    """Draw Pac-Man with a directional mouth."""

    bbox = _circle_bbox(center, radius)
    draw.ellipse(
        bbox,
        fill=theme.pacman_fill_rgb,
        outline=theme.pacman_outline_rgb,
        width=max(2, int(radius * 0.12)),
    )
    dx, dy = int(direction[0]), int(direction[1])
    angle = math.atan2(float(dy), float(dx))
    if dx == 0 and dy == 0:
        angle = 0.0
    spread = math.radians(33.0)
    points = [
        (float(center[0]), float(center[1])),
        (
            float(center[0] + (float(radius) * 1.12 * math.cos(angle - spread))),
            float(center[1] + (float(radius) * 1.12 * math.sin(angle - spread))),
        ),
        (
            float(center[0] + (float(radius) * 1.12 * math.cos(angle + spread))),
            float(center[1] + (float(radius) * 1.12 * math.sin(angle + spread))),
        ),
    ]
    draw.polygon(points, fill=theme.corridor_fill_rgb)
    eye_angle = float(angle - math.radians(65.0))
    eye_center = (
        float(center[0] + (float(radius) * 0.42 * math.cos(eye_angle))),
        float(center[1] + (float(radius) * 0.42 * math.sin(eye_angle))),
    )
    eye_bbox = _circle_bbox(eye_center, max(2.0, float(radius) * 0.12))
    draw.ellipse(eye_bbox, fill=(24, 24, 24))
    return bbox


def _draw_ghost(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    fill: Tuple[int, int, int],
    theme: PacmanTheme,
) -> Tuple[float, float, float, float]:
    """Draw a compact ghost marker with a body, skirt, and eye geometry."""

    cx, cy = float(center[0]), float(center[1])
    r = float(radius)
    bbox = (
        round(float(cx - r), 3),
        round(float(cy - (0.88 * r)), 3),
        round(float(cx + r), 3),
        round(float(cy + (0.92 * r)), 3),
    )
    body = (
        float(cx - r),
        float(cy - (0.70 * r)),
        float(cx + r),
        float(cy + (0.72 * r)),
    )
    draw.rounded_rectangle(
        body,
        radius=max(3, int(0.55 * r)),
        fill=fill,
        outline=theme.panel_outline_rgb,
        width=max(2, int(0.10 * r)),
    )
    skirt_y = float(cy + (0.34 * r))
    skirt = [
        (float(cx - r), skirt_y),
        (float(cx - r), float(cy + (0.82 * r))),
        (float(cx - (0.55 * r)), float(cy + (0.58 * r))),
        (float(cx - (0.20 * r)), float(cy + (0.82 * r))),
        (float(cx + (0.15 * r)), float(cy + (0.58 * r))),
        (float(cx + (0.55 * r)), float(cy + (0.82 * r))),
        (float(cx + r), float(cy + (0.58 * r))),
        (float(cx + r), skirt_y),
    ]
    draw.polygon(skirt, fill=fill)
    eye_radius = max(2.2, 0.18 * r)
    pupil_radius = max(1.1, 0.08 * r)
    for offset in (-0.33 * r, 0.33 * r):
        eye_center = (float(cx + offset), float(cy - (0.15 * r)))
        eye_bbox = _circle_bbox(eye_center, eye_radius)
        draw.ellipse(eye_bbox, fill=theme.ghost_eye_rgb)
        pupil_bbox = _circle_bbox((float(eye_center[0] + (0.05 * r)), float(eye_center[1])), pupil_radius)
        draw.ellipse(pupil_bbox, fill=theme.ghost_pupil_rgb)
    return bbox


def render_pacman_scene(
    *,
    row_count: int,
    col_count: int,
    open_cells: Sequence[Coord],
    wall_cells: Sequence[Coord],
    pacman_coord: Coord,
    route_coords: Sequence[Coord],
    pellets: Sequence[Coord],
    items: Sequence[PacmanItem],
    ghosts: Sequence[PacmanGhost],
    background: Image.Image,
    scene_variant: str,
    style_variant: str,
    params: PacmanRenderParams,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedPacmanScene:
    """Render one visible Pac-Man maze scene."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    theme = build_games_pacman_theme(style_variant=str(style_variant))
    maze_bbox, cell_size, centers, layout_jitter = _grid_geometry(
        params=params,
        row_count=int(row_count),
        col_count=int(col_count),
    )
    left, top, right, bottom = maze_bbox
    chrome_bbox = (
        int(round(float(left - 28.0))),
        int(round(float(top - 28.0))),
        int(round(float(right + 28.0))),
        int(round(float(bottom + 28.0))),
    )
    if panel_style is not None:
        draw_panel_scene_chrome(
            draw,
            bbox=chrome_bbox,
            style=panel_style,
            radius=24,
            border_width=2,
        )
    draw.rounded_rectangle(
        (
            float(left - 18.0),
            float(top - 18.0),
            float(right + 18.0),
            float(bottom + 18.0),
        ),
        radius=20,
        fill=theme.panel_fill_rgb,
        outline=theme.panel_outline_rgb,
        width=4,
    )

    open_set = {tuple(coord) for coord in open_cells}
    wall_set = {tuple(coord) for coord in wall_cells}
    entities: list[Dict[str, Any]] = []
    entity_bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    entity_points: Dict[str, Tuple[float, float]] = {}

    for coord in sorted_coords(tuple(open_set | wall_set)):
        bbox = _cell_bbox(coord, maze_bbox=maze_bbox, cell_size=float(cell_size), gap_px=float(params.wall_gap_px))
        if coord in wall_set:
            draw.rounded_rectangle(
                bbox,
                radius=max(2, int(cell_size * 0.09)),
                fill=theme.wall_fill_rgb,
                outline=theme.wall_outline_rgb,
                width=max(1, int(params.wall_outline_width_px)),
            )
        else:
            draw.rectangle(bbox, fill=theme.corridor_fill_rgb)

    route_points = [centers[tuple(coord)] for coord in route_coords if tuple(coord) in centers]
    if len(route_points) >= 2:
        draw.line(
            route_points,
            fill=theme.route_outline_rgb,
            width=max(1, int(params.route_width_px) + 4),
            joint="curve",
        )
        draw.line(
            route_points,
            fill=theme.route_fill_rgb,
            width=max(1, int(params.route_width_px)),
            joint="curve",
        )

    pellet_radius = max(3.0, float(params.pellet_radius_px))
    for coord in sorted_coords(pellets):
        center = centers[tuple(coord)]
        bbox = _circle_bbox(center, pellet_radius)
        draw.ellipse(bbox, fill=theme.pellet_fill_rgb)
        entity_id = pellet_entity_id(coord)
        entity_bboxes[entity_id] = bbox
        entity_points[entity_id] = (round(float(center[0]), 3), round(float(center[1]), 3))
        entities.append(
            {
                "entity_id": entity_id,
                "type": "pellet",
                "coord": [int(coord[0]), int(coord[1])],
                "bbox_px": list(bbox),
                "point_px": list(entity_points[entity_id]),
            }
        )

    label_font = load_font(int(params.item_label_font_size_px), bold=True, font_family=str(params.font_family))
    item_radius = max(12.0, float(params.item_radius_px))
    for item in items:
        coord = tuple(item.coord)
        center = centers[coord]
        item_bbox = _circle_bbox(center, item_radius)
        color = theme.item_palette_rgb.get(str(item.kind), next(iter(theme.item_palette_rgb.values())))
        display_text = str(int(item.score_value)) if item.score_value is not None else str(item.label)
        draw.ellipse(
            item_bbox,
            fill=color,
            outline=theme.pacman_outline_rgb,
            width=max(2, int(item_radius * 0.14)),
        )
        draw_centered_text(
            draw,
            text=display_text,
            center=center,
            font=label_font,
            fill=theme.item_label_rgb,
            stroke_fill=theme.item_label_outline_rgb,
            stroke_width=2,
        )
        entity_id = item_entity_id(str(item.label))
        entity_bboxes[entity_id] = item_bbox
        entity_points[entity_id] = (round(float(center[0]), 3), round(float(center[1]), 3))
        entities.append(
            {
                "entity_id": entity_id,
                "type": "bonus_item",
                "label": str(item.label),
                "display_text": display_text,
                "kind": str(item.kind),
                "coord": [int(coord[0]), int(coord[1])],
                "bbox_px": list(item_bbox),
                "point_px": list(entity_points[entity_id]),
            }
        )
        if item.score_value is not None:
            entities[-1]["score_value"] = int(item.score_value)

    ghost_radius = max(12.0, float(params.ghost_radius_px))
    for ghost in ghosts:
        coord = tuple(ghost.coord)
        center = centers[coord]
        color = theme.ghost_palette_rgb.get(str(ghost.color_key), next(iter(theme.ghost_palette_rgb.values())))
        ghost_bbox = _draw_ghost(
            draw,
            center=center,
            radius=ghost_radius,
            fill=color,
            theme=theme,
        )
        entity_id = ghost_entity_id(str(ghost.ghost_id))
        entity_bboxes[entity_id] = ghost_bbox
        entity_points[entity_id] = (round(float(center[0]), 3), round(float(center[1]), 3))
        entities.append(
            {
                "entity_id": entity_id,
                "type": "ghost",
                "color_key": str(ghost.color_key),
                "is_stop_ghost": bool(ghost.is_stop_ghost),
                "coord": [int(coord[0]), int(coord[1])],
                "bbox_px": list(ghost_bbox),
                "point_px": list(entity_points[entity_id]),
            }
        )

    pac_center = centers[tuple(pacman_coord)]
    direction = (1, 0)
    if len(route_coords) > 1:
        next_coord = tuple(route_coords[1])
        direction = (int(next_coord[1]) - int(pacman_coord[1]), int(next_coord[0]) - int(pacman_coord[0]))
    pac_bbox = _draw_pacman(
        draw,
        center=pac_center,
        radius=max(12.0, min(0.42 * float(cell_size), 24.0)),
        direction=direction,
        theme=theme,
    )
    entity_bboxes[pacman_entity_id()] = pac_bbox
    entity_points[pacman_entity_id()] = (round(float(pac_center[0]), 3), round(float(pac_center[1]), 3))
    entities.append(
        {
            "entity_id": pacman_entity_id(),
            "type": "pacman",
            "coord": [int(pacman_coord[0]), int(pacman_coord[1])],
            "bbox_px": list(pac_bbox),
            "point_px": list(entity_points[pacman_entity_id()]),
        }
    )
    for index, coord in enumerate(route_coords):
        center = centers[tuple(coord)]
        route_bbox = _circle_bbox(center, max(4.0, float(params.route_width_px)))
        entities.append(
            {
                "entity_id": route_entity_id(index),
                "type": "route_cell",
                "coord": [int(coord[0]), int(coord[1])],
                "bbox_px": list(route_bbox),
            }
        )

    render_map = {
        "entity_bboxes_px": {str(entity_id): [float(value) for value in bbox] for entity_id, bbox in entity_bboxes.items()},
        "entity_points_px": {str(entity_id): [float(x), float(y)] for entity_id, (x, y) in entity_points.items()},
        "cell_centers_px": {f"r{row}_c{col}": [float(x), float(y)] for (row, col), (x, y) in centers.items()},
        "route_polyline_px": [[float(x), float(y)] for x, y in route_points],
        "maze_bbox_px": [float(left), float(top), float(right), float(bottom)],
        "cell_size_px": float(cell_size),
        "layout_jitter": dict(layout_jitter),
        "font_family": str(params.font_family),
        "text_style": {
            "font_family": str(params.font_family),
        },
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
    }
    return RenderedPacmanScene(
        image=image,
        scene_entities=tuple(entities),
        render_map=render_map,
    )


__all__ = [
    "PacmanRenderParams",
    "PacmanTheme",
    "RenderedPacmanScene",
    "RenderedPacmanTaskContext",
    "build_games_pacman_theme",
    "render_pacman_scene",
    "render_pacman_task_context",
    "resolve_pacman_render_params",
]
