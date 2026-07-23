"""Rendering helpers for irregular-link-board scene tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

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

from .defaults import DEFAULTS
from .rules import all_coords, piece_id, point_id
from .state import (
    STYLE_VARIANTS,
    Coord,
    IrregularLinkBoardAxes,
    IrregularLinkBoardSample,
    IrregularLinkBoardTheme,
    RenderedIrregularLinkBoardScene,
)


def _theme_for_style(style_variant: str) -> Tuple[IrregularLinkBoardTheme, Dict[str, Any]]:
    """Resolve the scene-local palette while preserving piece/link contrast."""

    themes: dict[str, IrregularLinkBoardTheme] = {
        "woodcut": IrregularLinkBoardTheme(
            board_fill_rgb=(229, 200, 150),
            board_border_rgb=(103, 72, 39),
            edge_rgb=(108, 76, 45),
            point_fill_rgb=(250, 232, 188),
            point_outline_rgb=(82, 57, 34),
            marked_piece_fill_rgb=(34, 45, 65),
            marked_piece_outline_rgb=(248, 244, 232),
            blocker_piece_fill_rgb=(181, 65, 55),
            blocker_piece_outline_rgb=(88, 31, 29),
        ),
        "ink_diagram": IrregularLinkBoardTheme(
            board_fill_rgb=(238, 237, 226),
            board_border_rgb=(51, 52, 56),
            edge_rgb=(58, 60, 66),
            point_fill_rgb=(253, 251, 243),
            point_outline_rgb=(44, 46, 52),
            marked_piece_fill_rgb=(29, 40, 56),
            marked_piece_outline_rgb=(250, 251, 255),
            blocker_piece_fill_rgb=(83, 107, 133),
            blocker_piece_outline_rgb=(33, 48, 64),
        ),
        "garden_cloth": IrregularLinkBoardTheme(
            board_fill_rgb=(213, 233, 204),
            board_border_rgb=(63, 101, 68),
            edge_rgb=(71, 113, 76),
            point_fill_rgb=(244, 251, 235),
            point_outline_rgb=(55, 88, 58),
            marked_piece_fill_rgb=(39, 64, 50),
            marked_piece_outline_rgb=(246, 255, 240),
            blocker_piece_fill_rgb=(197, 72, 85),
            blocker_piece_outline_rgb=(94, 33, 43),
        ),
        "night_lines": IrregularLinkBoardTheme(
            board_fill_rgb=(34, 42, 58),
            board_border_rgb=(174, 190, 209),
            edge_rgb=(120, 206, 232),
            point_fill_rgb=(56, 72, 96),
            point_outline_rgb=(195, 221, 236),
            marked_piece_fill_rgb=(241, 246, 255),
            marked_piece_outline_rgb=(15, 23, 42),
            blocker_piece_fill_rgb=(255, 185, 89),
            blocker_piece_outline_rgb=(87, 47, 18),
        ),
        "parchment": IrregularLinkBoardTheme(
            board_fill_rgb=(239, 223, 185),
            board_border_rgb=(118, 89, 49),
            edge_rgb=(125, 92, 55),
            point_fill_rgb=(252, 240, 209),
            point_outline_rgb=(90, 67, 38),
            marked_piece_fill_rgb=(38, 55, 88),
            marked_piece_outline_rgb=(250, 247, 231),
            blocker_piece_fill_rgb=(151, 76, 52),
            blocker_piece_outline_rgb=(72, 35, 26),
        ),
    }
    resolved = str(style_variant) if str(style_variant) in themes else "woodcut"
    return themes[resolved], {
        "style_variant": str(resolved),
        "available_styles": list(STYLE_VARIANTS),
        "board_style_policy": "scene_local_irregular_link_board_palette",
    }


def _bbox(center: Sequence[float], radius: float) -> Tuple[float, float, float, float]:
    cx, cy = float(center[0]), float(center[1])
    return (
        round(cx - float(radius), 3),
        round(cy - float(radius), 3),
        round(cx + float(radius), 3),
        round(cy + float(radius), 3),
    )


def render_irregular_link_board_scene(
    *,
    sample: IrregularLinkBoardSample,
    axes: IrregularLinkBoardAxes,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderedIrregularLinkBoardScene:
    """Render one variable-link board and expose point/piece projection maps.

    The renderer is scene-only: it draws every link, point, piece, and the
    X-marked reference piece without knowing which task will use the rendered
    board. Task files select annotation ids from the returned maps.
    """

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
        min_px=280,
    )
    base_canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height)))
    dynamic_canvas_enabled = bool(
        params.get("dynamic_canvas_size_enabled", group_default(render_defaults, "dynamic_canvas_size_enabled", DEFAULTS.dynamic_canvas_size_enabled))
    )
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
    theme, theme_meta = _theme_for_style(str(axes.style_variant))

    margin = int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px)))
    board_size = int(sample.board_size)
    max_span = min(float(max_board_size_px), float(canvas_width - (2 * margin)), float(canvas_height - (2 * margin)))
    step = max(48.0, float(max_span) / float(board_size - 1))
    board_span = float(step * float(board_size - 1))
    board_bbox = (
        round(0.5 * (float(canvas_width) - board_span), 3),
        round(0.5 * (float(canvas_height) - board_span), 3),
        round(0.5 * (float(canvas_width) + board_span), 3),
        round(0.5 * (float(canvas_height) + board_span), 3),
    )
    board_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    left, top = float(board_bbox[0]), float(board_bbox[1])
    centers: Dict[Coord, Tuple[float, float]] = {
        coord: (round(left + (float(coord[1]) * step), 3), round(top + (float(coord[0]) * step), 3))
        for coord in all_coords(board_size)
    }
    edge_width = scale_games_px(group_default(render_defaults, "edge_width_px", DEFAULTS.edge_width_px), unit_scale, min_px=2)
    point_radius = scale_games_px(group_default(render_defaults, "point_radius_px", DEFAULTS.point_radius_px), unit_scale, min_px=8)
    piece_radius = scale_games_px(group_default(render_defaults, "piece_radius_px", DEFAULTS.piece_radius_px), unit_scale, min_px=14)
    marker_width = scale_games_px(group_default(render_defaults, "marker_width_px", DEFAULTS.marker_width_px), unit_scale, min_px=3)
    board_pad = max(22, int(round(float(piece_radius) * 1.55)))
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
        radius=24,
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
        radius=max(16, int(round(float(board_pad) * 0.55))),
        fill=tuple(theme.board_fill_rgb) + (226,),
        outline=tuple(theme.board_border_rgb) + (255,),
        width=max(2, int(round(float(edge_width) * 0.7))),
    )

    for link in sample.edges:
        draw.line([centers[link[0]], centers[link[1]]], fill=tuple(theme.edge_rgb) + (255,), width=max(2, int(edge_width)))

    entity_bboxes: Dict[str, List[float]] = {}
    entity_points: Dict[str, List[float]] = {}
    point_centers: Dict[str, List[float]] = {}
    point_bboxes: Dict[str, List[float]] = {}
    piece_centers: Dict[str, List[float]] = {}
    piece_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    occupied = set(sample.occupied_coords)
    for coord in all_coords(board_size):
        pid = point_id(coord)
        center = centers[coord]
        pbbox = _bbox(center, float(point_radius))
        draw.ellipse(
            pbbox,
            fill=tuple(theme.point_fill_rgb) + (255,),
            outline=tuple(theme.point_outline_rgb) + (255,),
            width=max(1, int(round(float(edge_width) * 0.45))),
        )
        point_centers[pid] = [float(center[0]), float(center[1])]
        point_bboxes[pid] = [float(v) for v in pbbox]
        entity_points[pid] = [float(center[0]), float(center[1])]
        entity_bboxes[pid] = [float(v) for v in pbbox]

    for coord in sorted(occupied):
        center = centers[coord]
        rendered_piece_id = "piece_marked" if coord == sample.marked_coord else piece_id(coord)
        is_marked = bool(coord == sample.marked_coord)
        fill = theme.marked_piece_fill_rgb if is_marked else theme.blocker_piece_fill_rgb
        outline = theme.marked_piece_outline_rgb if is_marked else theme.blocker_piece_outline_rgb
        rendered_piece_bbox = _bbox(center, float(piece_radius))
        draw.ellipse(
            rendered_piece_bbox,
            fill=tuple(fill) + (255,),
            outline=tuple(outline) + (255,),
            width=max(2, int(round(float(edge_width) * 0.8))),
        )
        piece_centers[rendered_piece_id] = [float(center[0]), float(center[1])]
        piece_bboxes[rendered_piece_id] = [float(v) for v in rendered_piece_bbox]
        entity_points[rendered_piece_id] = [float(center[0]), float(center[1])]
        entity_bboxes[rendered_piece_id] = [float(v) for v in rendered_piece_bbox]

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
            "from": [int(link[0][0]), int(link[0][1])],
            "to": [int(link[1][0]), int(link[1][1])],
            "from_point_id": point_id(link[0]),
            "to_point_id": point_id(link[1]),
        }
        for link in sample.edges
    ]
    for coord in all_coords(board_size):
        pid = point_id(coord)
        is_marked = coord == sample.marked_coord
        rendered_piece_id = "piece_marked" if is_marked else piece_id(coord) if coord in occupied else ""
        entities.append(
            {
                "entity_id": str(pid),
                "entity_type": "irregular_link_point",
                "row": int(coord[0]),
                "col": int(coord[1]),
                "state": "marked_piece" if is_marked else "occupied" if coord in occupied else "empty",
                "center_px": list(point_centers[pid]),
                "bbox_px": list(point_bboxes[pid]),
                "piece_id": str(rendered_piece_id),
                "piece_bbox_px": None if not rendered_piece_id else list(piece_bboxes[str(rendered_piece_id)]),
            }
        )
    render_map = {
        "board_bbox_px": [float(v) for v in board_bbox],
        "graph_bbox_px": [float(v) for v in graph_bbox],
        "panel_bbox_px": [float(v) for v in panel_bbox],
        "point_centers_px": dict(point_centers),
        "point_bboxes_px": dict(point_bboxes),
        "piece_centers_px": dict(piece_centers),
        "piece_bboxes_px": dict(piece_bboxes),
        "entity_points_px": dict(entity_points),
        "entity_bboxes_px": dict(entity_bboxes),
        "edges": edge_specs,
        "marked_piece_marker": marker_metadata,
        "layout_jitter": dict(resolved_jitter),
        "effective_board_step_px": round(float(step), 3),
        "effective_point_radius_px": int(point_radius),
        "effective_piece_radius_px": int(piece_radius),
        "effective_edge_width_px": int(edge_width),
    }
    return RenderedIrregularLinkBoardScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=render_map,
        style_meta={
            "panel_scene_style": dict(panel_style_meta),
            "irregular_link_board_style": dict(theme_meta),
        },
        background_meta=dict(background_meta),
    )


__all__ = ["render_irregular_link_board_scene"]
