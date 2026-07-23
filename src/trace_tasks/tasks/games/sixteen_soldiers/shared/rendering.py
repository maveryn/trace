"""Renderer for Sixteen Soldiers games scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.config_defaults import group_default

from ...shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from ...shared.marking import draw_optional_marker_x, draw_semantic_ellipse_marker, resolve_semantic_marker_style
from ...shared.scene_style import GamePanelSceneStyle, draw_panel_scene_chrome, game_panel_scene_style_metadata
from .defaults import DEFAULTS
from .rules import (
    EDGES,
    POINT_COORDS,
    board_to_dict,
    piece_to_entity_id,
    player_name,
    point_coord,
    point_id_from_coord,
)
from .state import (
    BLUE,
    EMPTY,
    RED,
    Board,
    PointId,
    RenderedSixteenSoldiersScene,
    SixteenSoldiersTheme,
)


@dataclass(frozen=True)
class SixteenSoldiersRenderParams:
    """Resolved render controls for one Sixteen Soldiers scene."""

    canvas_width: int
    canvas_height: int
    panel_margin_px: int
    max_board_width_px: int
    max_board_height_px: int
    edge_width_px: int
    point_radius_px: int
    piece_radius_px: int
    marker_width_px: int
    layout_jitter_meta: Dict[str, Any] | None = None
    instance_seed: int = 0


def _theme_for_style(style_variant: str) -> SixteenSoldiersTheme:
    """Return one readable Sixteen Soldiers board palette."""

    styles: dict[str, SixteenSoldiersTheme] = {
        "ground_court": SixteenSoldiersTheme(
            board_fill_rgb=(222, 190, 139),
            board_border_rgb=(105, 72, 41),
            edge_rgb=(88, 62, 39),
            point_fill_rgb=(248, 226, 179),
            point_outline_rgb=(76, 53, 35),
            red_piece_fill_rgb=(192, 40, 50),
            red_piece_outline_rgb=(92, 22, 30),
            blue_piece_fill_rgb=(38, 91, 176),
            blue_piece_outline_rgb=(17, 47, 96),
        ),
        "ink_court": SixteenSoldiersTheme(
            board_fill_rgb=(237, 235, 224),
            board_border_rgb=(50, 51, 55),
            edge_rgb=(51, 52, 58),
            point_fill_rgb=(252, 250, 239),
            point_outline_rgb=(43, 44, 49),
            red_piece_fill_rgb=(202, 48, 58),
            red_piece_outline_rgb=(86, 24, 31),
            blue_piece_fill_rgb=(39, 102, 181),
            blue_piece_outline_rgb=(19, 52, 96),
        ),
        "cloth_board": SixteenSoldiersTheme(
            board_fill_rgb=(212, 228, 201),
            board_border_rgb=(65, 99, 67),
            edge_rgb=(60, 104, 70),
            point_fill_rgb=(242, 250, 234),
            point_outline_rgb=(48, 79, 53),
            red_piece_fill_rgb=(205, 52, 63),
            red_piece_outline_rgb=(99, 25, 35),
            blue_piece_fill_rgb=(43, 99, 168),
            blue_piece_outline_rgb=(21, 51, 89),
        ),
        "slate_court": SixteenSoldiersTheme(
            board_fill_rgb=(202, 214, 224),
            board_border_rgb=(58, 68, 80),
            edge_rgb=(67, 78, 91),
            point_fill_rgb=(235, 242, 248),
            point_outline_rgb=(47, 57, 69),
            red_piece_fill_rgb=(205, 53, 68),
            red_piece_outline_rgb=(96, 24, 35),
            blue_piece_fill_rgb=(45, 96, 185),
            blue_piece_outline_rgb=(22, 47, 98),
        ),
        "sand_court": SixteenSoldiersTheme(
            board_fill_rgb=(239, 219, 176),
            board_border_rgb=(125, 90, 49),
            edge_rgb=(136, 98, 55),
            point_fill_rgb=(253, 238, 204),
            point_outline_rgb=(92, 66, 38),
            red_piece_fill_rgb=(188, 45, 48),
            red_piece_outline_rgb=(88, 22, 25),
            blue_piece_fill_rgb=(39, 92, 158),
            blue_piece_outline_rgb=(19, 47, 84),
        ),
    }
    return styles.get(str(style_variant), styles["ground_court"])


def _bbox_from_center(center: Sequence[float], radius: float) -> Tuple[float, float, float, float]:
    """Return one bbox centered at a point."""

    cx, cy = float(center[0]), float(center[1])
    return (
        round(float(cx - radius), 3),
        round(float(cy - radius), 3),
        round(float(cx + radius), 3),
        round(float(cy + radius), 3),
    )


def resolve_sixteen_soldiers_render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> SixteenSoldiersRenderParams:
    """Resolve render parameters from scene config/defaults."""

    unit_scale, unit_scale_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="games.sixteen_soldiers.unit_size",
    )
    layout_jitter = attach_games_unit_size_jitter(
        resolve_games_layout_jitter(
            params,
            render_defaults,
            instance_seed=int(instance_seed),
            namespace="games.sixteen_soldiers.layout",
        ),
        unit_scale_meta,
    )
    base_canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", DEFAULTS.canvas_width)))
    base_canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", DEFAULTS.canvas_height)))
    max_board_width_px = scale_games_px(
        params.get("max_board_width_px", group_default(render_defaults, "max_board_width_px", DEFAULTS.max_board_width_px)),
        unit_scale,
        min_px=280,
    )
    max_board_height_px = scale_games_px(
        params.get("max_board_height_px", group_default(render_defaults, "max_board_height_px", DEFAULTS.max_board_height_px)),
        unit_scale,
        min_px=520,
    )
    dynamic_canvas_enabled = bool(
        params.get(
            "dynamic_canvas_size_enabled",
            group_default(render_defaults, "dynamic_canvas_size_enabled", DEFAULTS.dynamic_canvas_size_enabled),
        )
    )
    canvas_width = int(base_canvas_width)
    canvas_height = int(base_canvas_height)
    if dynamic_canvas_enabled and params.get("canvas_width") is None:
        canvas_width = min(
            int(base_canvas_width),
            max(
                int(params.get("canvas_min_width_px", group_default(render_defaults, "canvas_min_width_px", DEFAULTS.canvas_min_width_px))),
                int(
                    round(
                        float(max_board_width_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_side_padding_px",
                                    group_default(render_defaults, "canvas_side_padding_px", DEFAULTS.canvas_side_padding_px),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    if dynamic_canvas_enabled and params.get("canvas_height") is None:
        canvas_height = min(
            int(base_canvas_height),
            max(
                int(params.get("canvas_min_height_px", group_default(render_defaults, "canvas_min_height_px", DEFAULTS.canvas_min_height_px))),
                int(
                    round(
                        float(max_board_height_px)
                        + (
                            2.0
                            * float(
                                params.get(
                                    "canvas_vertical_padding_px",
                                    group_default(render_defaults, "canvas_vertical_padding_px", DEFAULTS.canvas_vertical_padding_px),
                                )
                            )
                        )
                    )
                ),
            ),
        )
    return SixteenSoldiersRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        panel_margin_px=int(params.get("panel_margin_px", group_default(render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px))),
        max_board_width_px=int(max_board_width_px),
        max_board_height_px=int(max_board_height_px),
        edge_width_px=scale_games_px(params.get("edge_width_px", group_default(render_defaults, "edge_width_px", DEFAULTS.edge_width_px)), unit_scale, min_px=2),
        point_radius_px=scale_games_px(params.get("point_radius_px", group_default(render_defaults, "point_radius_px", DEFAULTS.point_radius_px)), unit_scale, min_px=4),
        piece_radius_px=scale_games_px(params.get("piece_radius_px", group_default(render_defaults, "piece_radius_px", DEFAULTS.piece_radius_px)), unit_scale, min_px=13),
        marker_width_px=scale_games_px(params.get("marker_width_px", group_default(render_defaults, "marker_width_px", DEFAULTS.marker_width_px)), unit_scale, min_px=3),
        layout_jitter_meta=layout_jitter,
        instance_seed=int(instance_seed),
    )


def render_sixteen_soldiers_scene(
    *,
    board: Board,
    background: Image.Image,
    style_variant: str,
    params: SixteenSoldiersRenderParams,
    marked_point_id: PointId,
    panel_style: GamePanelSceneStyle | None = None,
) -> RenderedSixteenSoldiersScene:
    """Render one Sixteen Soldiers board state."""

    image = background.convert("RGBA")
    draw = ImageDraw.Draw(image, "RGBA")
    theme = _theme_for_style(str(style_variant))
    values = board_to_dict(board)

    max_width = min(
        int(params.max_board_width_px),
        int(params.canvas_width) - (2 * int(params.panel_margin_px)),
    )
    max_height = min(
        int(params.max_board_height_px),
        int(params.canvas_height) - (2 * int(params.panel_margin_px)),
    )
    step = max(42.0, min(float(max_width) / 4.0, float(max_height) / 8.0))
    board_span_x = float(4.0 * step)
    board_span_y = float(8.0 * step)
    board_bbox = (
        round(float(0.5 * (int(params.canvas_width) - board_span_x)), 3),
        round(float(0.5 * (int(params.canvas_height) - board_span_y)), 3),
        round(float(0.5 * (int(params.canvas_width) + board_span_x)), 3),
        round(float(0.5 * (int(params.canvas_height) + board_span_y)), 3),
    )
    board_bbox, dx, dy, layout_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=board_bbox,
        canvas_width=int(params.canvas_width),
        canvas_height=int(params.canvas_height),
        jitter=params.layout_jitter_meta,
    )
    left, top = float(board_bbox[0]), float(board_bbox[1])
    centers_by_point_id: Dict[PointId, Tuple[float, float]] = {}
    for coord in POINT_COORDS:
        point_id = point_id_from_coord(coord)
        row, col = int(coord[0]), int(coord[1])
        centers_by_point_id[point_id] = (
            round(float(left + (float(col) * step)), 3),
            round(float(top + (float(row) * step)), 3),
        )

    board_pad = max(18, int(round(float(params.piece_radius_px) * 1.35)))
    panel_bbox: Tuple[int, int, int, int] | None = None
    if panel_style is not None:
        panel_bbox = (
            max(4, int(round(board_bbox[0])) - board_pad),
            max(4, int(round(board_bbox[1])) - board_pad),
            min(int(params.canvas_width) - 4, int(round(board_bbox[2])) + board_pad),
            min(int(params.canvas_height) - 4, int(round(board_bbox[3])) + board_pad),
        )
        draw_panel_scene_chrome(
            draw,
            bbox=panel_bbox,
            style=panel_style,
            radius=24,
            border_width=max(2, int(round(float(params.edge_width_px) * 0.65))),
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
        fill=tuple(int(v) for v in theme.board_fill_rgb) + (226,),
        outline=tuple(int(v) for v in theme.board_border_rgb) + (255,),
        width=max(2, int(round(float(params.edge_width_px) * 0.7))),
    )

    edge_specs: list[dict[str, Any]] = []
    for a, b in EDGES:
        p0 = centers_by_point_id[str(a)]
        p1 = centers_by_point_id[str(b)]
        draw.line(
            [p0, p1],
            fill=tuple(int(v) for v in theme.edge_rgb) + (255,),
            width=max(2, int(params.edge_width_px)),
        )
        edge_specs.append(
            {
                "from_point_id": str(a),
                "to_point_id": str(b),
                "from_coord": [int(value) for value in point_coord(a)],
                "to_coord": [int(value) for value in point_coord(b)],
            }
        )

    point_centers_px: Dict[str, List[float]] = {}
    point_bboxes_px: Dict[str, List[float]] = {}
    piece_bboxes_px: Dict[str, List[float]] = {}
    piece_centers_px: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []

    for coord in POINT_COORDS:
        point_id = point_id_from_coord(coord)
        center = centers_by_point_id[point_id]
        point_bbox = _bbox_from_center(center, float(params.point_radius_px))
        draw.ellipse(
            point_bbox,
            fill=tuple(int(v) for v in theme.point_fill_rgb) + (255,),
            outline=tuple(int(v) for v in theme.point_outline_rgb) + (255,),
            width=max(1, int(round(float(params.edge_width_px) * 0.45))),
        )
        point_centers_px[point_id] = [float(center[0]), float(center[1])]
        point_bboxes_px[point_id] = [float(v) for v in point_bbox]

    marker_metadata: dict[str, Any] | None = None
    for coord in POINT_COORDS:
        point_id = point_id_from_coord(coord)
        value = int(values[point_id])
        piece_bbox = None
        piece_id = None
        if value in {RED, BLUE}:
            center = centers_by_point_id[point_id]
            piece_bbox = _bbox_from_center(center, float(params.piece_radius_px))
            fill = theme.red_piece_fill_rgb if value == RED else theme.blue_piece_fill_rgb
            outline = theme.red_piece_outline_rgb if value == RED else theme.blue_piece_outline_rgb
            shadow_offset = max(1, int(round(float(params.piece_radius_px) * 0.12)))
            draw.ellipse(
                (
                    float(piece_bbox[0]) + shadow_offset,
                    float(piece_bbox[1]) + shadow_offset,
                    float(piece_bbox[2]) + shadow_offset,
                    float(piece_bbox[3]) + shadow_offset,
                ),
                fill=(0, 0, 0, 34),
            )
            draw.ellipse(
                piece_bbox,
                fill=tuple(int(v) for v in fill) + (255,),
                outline=tuple(int(v) for v in outline) + (255,),
                width=max(2, int(round(float(params.edge_width_px) * 0.9))),
            )
            highlight_r = max(2.0, float(params.piece_radius_px) * 0.18)
            draw.ellipse(
                (
                    float(center[0]) - (0.42 * float(params.piece_radius_px)),
                    float(center[1]) - (0.44 * float(params.piece_radius_px)),
                    float(center[0]) - (0.42 * float(params.piece_radius_px)) + highlight_r,
                    float(center[1]) - (0.44 * float(params.piece_radius_px)) + highlight_r,
                ),
                fill=(255, 255, 255, 82),
            )
            piece_id = piece_to_entity_id(point_id)
            piece_bboxes_px[piece_id] = [float(v) for v in piece_bbox]
            piece_centers_px[piece_id] = [float(center[0]), float(center[1])]
        row, col = point_coord(point_id)
        scene_entities.append(
            {
                "entity_id": str(point_id),
                "entity_type": "sixteen_soldiers_point",
                "row": int(row),
                "col": int(col),
                "state": "empty" if value == EMPTY else player_name(value),
                "center_px": [float(v) for v in point_centers_px[point_id]],
                "bbox_px": [float(v) for v in point_bboxes_px[point_id]],
                "piece_id": piece_id,
                "piece_bbox_px": None if piece_bbox is None else [float(v) for v in piece_bbox],
                "is_marked": bool(str(point_id) == str(marked_point_id)),
            }
        )

    marked_piece_id = piece_to_entity_id(str(marked_point_id))
    if marked_piece_id in piece_bboxes_px:
        marked_bbox = piece_bboxes_px[marked_piece_id]
        marker_pad = max(5.0, float(params.marker_width_px) * 1.45)
        marker_bbox = (
            float(marked_bbox[0]) - marker_pad,
            float(marked_bbox[1]) - marker_pad,
            float(marked_bbox[2]) + marker_pad,
            float(marked_bbox[3]) + marker_pad,
        )
        marker_style = resolve_semantic_marker_style(
            instance_seed=int(params.instance_seed),
            namespace="games.sixteen_soldiers.marked_piece",
            role="marked_piece",
            surface_rgbs=(theme.board_fill_rgb,),
            preferred_rgbs=((255, 214, 38), (255, 247, 92), (246, 80, 164), (36, 205, 228)),
        )
        marker_metadata = draw_semantic_ellipse_marker(
            draw,
            marker_bbox,
            style=marker_style,
            width=max(3, int(params.marker_width_px)),
            marker_kind="marked_piece_ring",
        )
        x_metadata = draw_optional_marker_x(
            draw,
            marked_bbox,
            enabled=True,
            width=max(3, int(round(float(params.marker_width_px) * 0.75))),
            inset_fraction=0.25,
            marker_kind="marked_piece_x",
            extra_metadata={"piece_id": str(marked_piece_id)},
        )
        if x_metadata is not None:
            marker_metadata = {**dict(marker_metadata), "overlay_x": dict(x_metadata)}

    render_map = {
        "board_bbox_px": [float(v) for v in board_bbox],
        "graph_bbox_px": [float(v) for v in graph_bbox],
        "point_centers_px": dict(point_centers_px),
        "point_bboxes_px": dict(point_bboxes_px),
        "piece_centers_px": dict(piece_centers_px),
        "piece_bboxes_px": dict(piece_bboxes_px),
        "edges": edge_specs,
        "marked_piece_id": piece_to_entity_id(str(marked_point_id)),
        "marked_point_id": str(marked_point_id),
        "layout_jitter": {**dict(layout_jitter), "board_dx_px": float(dx), "board_dy_px": float(dy)},
        "scene_panel_bbox_px": None if panel_bbox is None else [int(value) for value in panel_bbox],
        "panel_scene_style": None if panel_style is None else game_panel_scene_style_metadata(panel_style),
        "style_variant": str(style_variant),
        "marker_metadata": marker_metadata,
    }
    return RenderedSixteenSoldiersScene(
        image=image,
        scene_entities=tuple(scene_entities),
        render_map=render_map,
    )


__all__ = [
    "RenderedSixteenSoldiersScene",
    "SixteenSoldiersRenderParams",
    "resolve_sixteen_soldiers_render_params",
    "render_sixteen_soldiers_scene",
]
