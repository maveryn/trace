"""Rendering helpers for radial hunt board scene tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.marking import (
    draw_optional_marker_x,
    draw_semantic_ellipse_marker,
    resolve_semantic_marker_style,
)
from trace_tasks.tasks.games.shared.scene_style import (
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import DEFAULTS
from .rules import all_coords, all_possible_edges, piece_id, point_id
from .state import (
    CENTER,
    SCENE_NAMESPACE,
    Coord,
    RadialHuntBoardSample,
    RadialHuntBoardTheme,
    RadialHuntBoardVisualAxes,
    RenderedRadialHuntBoardScene,
    SUPPORTED_STYLE_VARIANTS,
)


def theme_for_style(style_variant: str) -> Tuple[RadialHuntBoardTheme, Dict[str, Any]]:
    """Resolve one radial hunt board palette and metadata record."""

    themes: dict[str, RadialHuntBoardTheme] = {
        "ink_rings": RadialHuntBoardTheme(
            board_fill_rgb=(236, 235, 221),
            board_border_rgb=(52, 55, 59),
            edge_rgb=(45, 48, 52),
            point_fill_rgb=(252, 249, 237),
            point_outline_rgb=(43, 45, 49),
            marked_piece_fill_rgb=(28, 40, 61),
            marked_piece_outline_rgb=(250, 250, 245),
            opponent_piece_fill_rgb=(184, 72, 58),
            opponent_piece_outline_rgb=(80, 32, 28),
        ),
        "carved_wood": RadialHuntBoardTheme(
            board_fill_rgb=(226, 188, 126),
            board_border_rgb=(94, 63, 36),
            edge_rgb=(110, 74, 42),
            point_fill_rgb=(250, 228, 177),
            point_outline_rgb=(83, 55, 33),
            marked_piece_fill_rgb=(38, 52, 80),
            marked_piece_outline_rgb=(247, 240, 220),
            opponent_piece_fill_rgb=(166, 75, 47),
            opponent_piece_outline_rgb=(76, 35, 24),
        ),
        "temple_cloth": RadialHuntBoardTheme(
            board_fill_rgb=(211, 228, 202),
            board_border_rgb=(63, 95, 67),
            edge_rgb=(67, 103, 71),
            point_fill_rgb=(242, 249, 229),
            point_outline_rgb=(54, 82, 55),
            marked_piece_fill_rgb=(40, 62, 47),
            marked_piece_outline_rgb=(247, 255, 239),
            opponent_piece_fill_rgb=(191, 67, 89),
            opponent_piece_outline_rgb=(83, 30, 42),
        ),
        "night_gold": RadialHuntBoardTheme(
            board_fill_rgb=(35, 43, 58),
            board_border_rgb=(210, 185, 116),
            edge_rgb=(236, 198, 101),
            point_fill_rgb=(58, 71, 95),
            point_outline_rgb=(242, 221, 156),
            marked_piece_fill_rgb=(241, 246, 255),
            marked_piece_outline_rgb=(12, 20, 36),
            opponent_piece_fill_rgb=(255, 177, 84),
            opponent_piece_outline_rgb=(86, 48, 18),
        ),
        "chalk_circle": RadialHuntBoardTheme(
            board_fill_rgb=(70, 91, 82),
            board_border_rgb=(220, 230, 216),
            edge_rgb=(224, 232, 221),
            point_fill_rgb=(90, 113, 103),
            point_outline_rgb=(238, 243, 232),
            marked_piece_fill_rgb=(247, 249, 236),
            marked_piece_outline_rgb=(28, 45, 39),
            opponent_piece_fill_rgb=(236, 104, 89),
            opponent_piece_outline_rgb=(92, 40, 35),
        ),
    }
    resolved = str(style_variant) if str(style_variant) in themes else "ink_rings"
    return themes[resolved], {
        "style_variant": str(resolved),
        "available_styles": list(SUPPORTED_STYLE_VARIANTS),
        "board_style_policy": "scene_local_radial_hunt_board_palette",
    }


def _bbox(center: Sequence[float], radius: float) -> Tuple[float, float, float, float]:
    """Return a rounded square bbox around a center point."""

    cx, cy = float(center[0]), float(center[1])
    return (
        round(cx - float(radius), 3),
        round(cy - float(radius), 3),
        round(cx + float(radius), 3),
        round(cy + float(radius), 3),
    )


def _coord_angle(spoke: int) -> float:
    """Return the display angle for one radial-board spoke index."""

    return math.radians(-90.0 + (60.0 * float(spoke)))


def render_radial_hunt_board_scene(
    *,
    sample: RadialHuntBoardSample,
    axes: RadialHuntBoardVisualAxes,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedRadialHuntBoardScene:
    """Render one radial board while recording point, piece, and edge geometry."""

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
    max_board_size_px = scale_games_px(
        group_default(render_defaults, "max_board_size_px", DEFAULTS.max_board_size_px),
        unit_scale,
        min_px=300,
    )
    base_canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height)))
    dynamic_canvas_enabled = bool(params.get("dynamic_canvas_size_enabled", group_default(render_defaults, "dynamic_canvas_size_enabled", DEFAULTS.dynamic_canvas_size_enabled)))
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(render_defaults, "canvas_min_width_px", DEFAULTS.canvas_min_width_px))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_side_padding_px", group_default(render_defaults, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px)))))),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(render_defaults, "canvas_min_height_px", DEFAULTS.canvas_min_height_px))),
                int(round(float(max_board_size_px) + (2.0 * float(params.get("canvas_vertical_padding_px", group_default(render_defaults, "canvas_vertical_padding_px", DEFAULTS.canvas_vertical_padding_px)))))),
            ),
        )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_scene_style",
        treatment_weights=params.get("panel_scene_treatment_weights", group_default(render_defaults, "panel_scene_treatment_weights", None)),
        palette_weights=params.get("panel_scene_palette_weights", group_default(render_defaults, "panel_scene_palette_weights", None)),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=panel_style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme, theme_meta = theme_for_style(str(axes.style_variant))

    margin = int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px)))
    max_span = min(float(max_board_size_px), float(canvas_width - (2 * margin)), float(canvas_height - (2 * margin)))
    outer_radius = float(max_span) * 0.5
    board_bbox = (
        round(0.5 * (float(canvas_width) - (2.0 * outer_radius)), 3),
        round(0.5 * (float(canvas_height) - (2.0 * outer_radius)), 3),
        round(0.5 * (float(canvas_width) + (2.0 * outer_radius)), 3),
        round(0.5 * (float(canvas_height) + (2.0 * outer_radius)), 3),
    )
    board_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    center_xy = (
        round(0.5 * (float(board_bbox[0]) + float(board_bbox[2])), 3),
        round(0.5 * (float(board_bbox[1]) + float(board_bbox[3])), 3),
    )
    outer_radius = 0.5 * min(float(board_bbox[2]) - float(board_bbox[0]), float(board_bbox[3]) - float(board_bbox[1]))
    ring_radii = {1: outer_radius / 3.0, 2: (2.0 * outer_radius) / 3.0, 3: outer_radius}
    centers: Dict[Coord, Tuple[float, float]] = {CENTER: center_xy}
    for ring in range(1, 4):
        for spoke in range(6):
            angle = _coord_angle(spoke)
            radius = ring_radii[ring]
            centers[(ring, spoke)] = (
                round(float(center_xy[0]) + (float(radius) * math.cos(angle)), 3),
                round(float(center_xy[1]) + (float(radius) * math.sin(angle)), 3),
            )

    edge_width = scale_games_px(group_default(render_defaults, "edge_width_px", DEFAULTS.edge_width_px), unit_scale, min_px=2)
    point_radius = scale_games_px(group_default(render_defaults, "point_radius_px", DEFAULTS.point_radius_px), unit_scale, min_px=8)
    piece_radius = scale_games_px(group_default(render_defaults, "piece_radius_px", DEFAULTS.piece_radius_px), unit_scale, min_px=14)
    marker_width = scale_games_px(group_default(render_defaults, "marker_width_px", DEFAULTS.marker_width_px), unit_scale, min_px=3)
    board_pad = max(24, int(round(float(piece_radius) * 1.6)))
    panel_bbox = (
        max(4, int(round(float(board_bbox[0]) - board_pad))),
        max(4, int(round(float(board_bbox[1]) - board_pad))),
        min(int(canvas_width) - 4, int(round(float(board_bbox[2]) + board_pad))),
        min(int(canvas_height) - 4, int(round(float(board_bbox[3]) + board_pad))),
    )
    draw_panel_scene_chrome(
        draw,
        bbox=panel_bbox,
        style=panel_style,
        radius=28,
        border_width=max(2, int(round(float(edge_width) * 0.65))),
    )
    graph_bbox = (
        int(round(float(board_bbox[0]) - board_pad)),
        int(round(float(board_bbox[1]) - board_pad)),
        int(round(float(board_bbox[2]) + board_pad)),
        int(round(float(board_bbox[3]) + board_pad)),
    )
    draw.rounded_rectangle(
        graph_bbox,
        radius=max(20, int(round(float(board_pad) * 0.6))),
        fill=tuple(theme.board_fill_rgb) + (226,),
        outline=tuple(theme.board_border_rgb) + (255,),
        width=max(2, int(round(float(edge_width) * 0.7))),
    )

    for ring in range(1, 4):
        radius = ring_radii[ring]
        circle_bbox = (
            float(center_xy[0]) - radius,
            float(center_xy[1]) - radius,
            float(center_xy[0]) + radius,
            float(center_xy[1]) + radius,
        )
        draw.ellipse(circle_bbox, outline=tuple(theme.edge_rgb) + (255,), width=max(2, int(edge_width)))
    for axis in range(3):
        p0 = centers[(3, axis)]
        p1 = centers[(3, (axis + 3) % 6)]
        draw.line([p0, p1], fill=tuple(theme.edge_rgb) + (255,), width=max(2, int(edge_width)))

    entity_bboxes: Dict[str, list[float]] = {}
    entity_points: Dict[str, list[float]] = {}
    point_centers: Dict[str, list[float]] = {}
    point_bboxes: Dict[str, list[float]] = {}
    piece_centers: Dict[str, list[float]] = {}
    piece_bboxes: Dict[str, list[float]] = {}
    entities: list[Dict[str, Any]] = []
    occupied = set(sample.occupied_coords)
    for coord in all_coords():
        current_point_id = point_id(coord)
        center = centers[coord]
        point_bbox = _bbox(center, float(point_radius))
        draw.ellipse(
            point_bbox,
            fill=tuple(theme.point_fill_rgb) + (255,),
            outline=tuple(theme.point_outline_rgb) + (255,),
            width=max(1, int(round(float(edge_width) * 0.45))),
        )
        point_centers[current_point_id] = [float(center[0]), float(center[1])]
        point_bboxes[current_point_id] = [float(value) for value in point_bbox]
        entity_points[current_point_id] = [float(center[0]), float(center[1])]
        entity_bboxes[current_point_id] = [float(value) for value in point_bbox]

    for coord in sorted(occupied):
        center = centers[coord]
        current_piece_id = "piece_marked" if coord == sample.marked_coord else piece_id(coord)
        is_marked = bool(coord == sample.marked_coord)
        fill = theme.marked_piece_fill_rgb if is_marked else theme.opponent_piece_fill_rgb
        outline = theme.marked_piece_outline_rgb if is_marked else theme.opponent_piece_outline_rgb
        piece_bbox = _bbox(center, float(piece_radius))
        draw.ellipse(
            piece_bbox,
            fill=tuple(fill) + (255,),
            outline=tuple(outline) + (255,),
            width=max(2, int(round(float(edge_width) * 0.8))),
        )
        piece_centers[current_piece_id] = [float(center[0]), float(center[1])]
        piece_bboxes[current_piece_id] = [float(value) for value in piece_bbox]
        entity_points[current_piece_id] = [float(center[0]), float(center[1])]
        entity_bboxes[current_piece_id] = [float(value) for value in piece_bbox]

    marker_metadata: dict[str, Any] | None = None
    marked_bbox = piece_bboxes.get("piece_marked")
    if marked_bbox is not None:
        marker_pad = max(5.0, float(marker_width) * 1.45)
        marker_bbox = (
            float(marked_bbox[0]) - marker_pad,
            float(marked_bbox[1]) - marker_pad,
            float(marked_bbox[2]) + marker_pad,
            float(marked_bbox[3]) + marker_pad,
        )
        marker_style = resolve_semantic_marker_style(
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.marked_piece",
            role="marked_piece",
            surface_rgbs=(theme.board_fill_rgb,),
            preferred_rgbs=((255, 214, 38), (255, 247, 92), (246, 80, 164), (36, 205, 228)),
        )
        marker_metadata = draw_semantic_ellipse_marker(
            draw,
            marker_bbox,
            style=marker_style,
            width=max(3, int(marker_width)),
            marker_kind="marked_piece_ring",
            extra_metadata={"piece_id": "piece_marked"},
        )
        x_metadata = draw_optional_marker_x(
            draw,
            marked_bbox,
            enabled=True,
            width=max(3, int(round(float(marker_width) * 0.75))),
            inset_fraction=0.25,
            marker_kind="marked_piece_x",
            extra_metadata={"piece_id": "piece_marked"},
        )
        if x_metadata is not None:
            marker_metadata = {**dict(marker_metadata), "overlay_x": dict(x_metadata)}

    edge_specs = [
        {
            "from": [int(item[0][0]), int(item[0][1])],
            "to": [int(item[1][0]), int(item[1][1])],
            "from_point_id": point_id(item[0]),
            "to_point_id": point_id(item[1]),
        }
        for item in all_possible_edges()
    ]
    for coord in all_coords():
        current_point_id = point_id(coord)
        is_marked = coord == sample.marked_coord
        current_piece_id = "piece_marked" if is_marked else piece_id(coord) if coord in occupied else ""
        entities.append(
            {
                "entity_id": str(current_point_id),
                "entity_type": "radial_hunt_point",
                "ring": int(coord[0]),
                "spoke": int(coord[1]),
                "state": "marked_piece" if is_marked else "occupied" if coord in occupied else "empty",
                "center_px": list(point_centers[current_point_id]),
                "bbox_px": list(point_bboxes[current_point_id]),
                "piece_id": str(current_piece_id),
                "piece_bbox_px": None if not current_piece_id else list(piece_bboxes[str(current_piece_id)]),
            }
        )
    render_map = {
        "board_bbox_px": [float(value) for value in board_bbox],
        "graph_bbox_px": [float(value) for value in graph_bbox],
        "panel_bbox_px": [float(value) for value in panel_bbox],
        "point_centers_px": dict(point_centers),
        "point_bboxes_px": dict(point_bboxes),
        "piece_centers_px": dict(piece_centers),
        "piece_bboxes_px": dict(piece_bboxes),
        "entity_points_px": dict(entity_points),
        "entity_bboxes_px": dict(entity_bboxes),
        "edges": edge_specs,
        "marked_piece_marker": marker_metadata,
        "layout_jitter": dict(resolved_jitter),
        "effective_outer_radius_px": round(float(outer_radius), 3),
        "effective_point_radius_px": int(point_radius),
        "effective_piece_radius_px": int(piece_radius),
        "effective_edge_width_px": int(edge_width),
    }
    return RenderedRadialHuntBoardScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=render_map,
        style_meta={
            "panel_scene_style": dict(panel_style_meta),
            "radial_hunt_board_style": dict(theme_meta),
        },
        background_meta=dict(background_meta),
    )


__all__ = ["render_radial_hunt_board_scene", "theme_for_style"]
