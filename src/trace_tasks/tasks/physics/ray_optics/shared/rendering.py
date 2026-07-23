"""Rendering helpers for graph-paper ray-optics scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.physics.shared.style import build_physics_optics_theme
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text
from trace_tasks.tasks.shared.text_rendering import (
    load_font,
    resolve_text_stroke_fill,
)

from .state import (
    OpticsBounceSpec,
    OpticsMirrorSpec,
    OpticsTargetSpec,
    RenderedOpticsScene,
)


def _cell_bbox(
    *,
    board_left_px: float,
    board_top_px: float,
    cell_size_px: float,
    col: int,
    row: int,
) -> List[float]:
    """Return the pixel bbox for one board cell."""

    left = float(board_left_px + (int(col) * cell_size_px))
    top = float(board_top_px + (int(row) * cell_size_px))
    return [
        round(left, 3),
        round(top, 3),
        round(left + cell_size_px, 3),
        round(top + cell_size_px, 3),
    ]


def _cell_center(
    *,
    board_left_px: float,
    board_top_px: float,
    cell_size_px: float,
    col: int,
    row: int,
) -> Tuple[float, float]:
    """Return the pixel center for one board cell."""

    bbox = _cell_bbox(
        board_left_px=float(board_left_px),
        board_top_px=float(board_top_px),
        cell_size_px=float(cell_size_px),
        col=int(col),
        row=int(row),
    )
    return (float(0.5 * (bbox[0] + bbox[2])), float(0.5 * (bbox[1] + bbox[3])))


def _graph_origin(
    *,
    board_left_px: float,
    board_top_px: float,
    board_rows: int,
    cell_size_px: float,
) -> Tuple[float, float]:
    """Return the pixel-space origin for graph-paper point coordinates."""

    return (
        float(board_left_px + (0.5 * cell_size_px)),
        float(board_top_px + ((float(board_rows) - 0.5) * cell_size_px)),
    )


def render_optics_ray_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    scene_variant: str,
    ray_event_kind: str,
    source_row: int,
    mirrors: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    bounce_cells: Sequence[Tuple[int, int]],
    ray_polyline_cells: Sequence[Tuple[int, int]],
    source_point_px: Tuple[float, float],
    exit_point_px: Tuple[float, float],
    annotation_entity_ids: Sequence[str],
    diagram_style: Any | None = None,
    font_family: str | None = None,
) -> RenderedOpticsScene:
    """Render one graph-paper optics scene with hidden full-path semantics."""

    canvas = background.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    theme = build_physics_optics_theme(str(accent_color_name), diagram_style=diagram_style)
    board_left = float(render_defaults["board_left_px"])
    board_top = float(render_defaults["board_top_px"])
    board_cols = int(render_defaults["board_cols"])
    board_rows = int(render_defaults["board_rows"])
    cell_size = float(render_defaults["cell_size_px"])
    board_right = float(board_left + (board_cols * cell_size))
    board_bottom = float(board_top + (board_rows * cell_size))
    grid_width = max(1, int(render_defaults["board_grid_width_px"]))
    outline_width = max(1, int(render_defaults["board_outline_width_px"]))
    mirror_width = max(1, int(render_defaults["mirror_width_px"]))
    mirror_padding = float(render_defaults["mirror_padding_px"])
    ray_width = max(1, int(render_defaults["ray_width_px"]))
    ray_head_length = float(render_defaults["ray_head_length_px"])
    ray_head_width = float(render_defaults["ray_head_width_px"])
    target_radius = max(10.0, float(render_defaults["target_radius_px"]))
    target_font = load_font(
        max(16, int(round(cell_size * 0.36))),
        bold=False,
        font_family=font_family,
    )
    label_stroke_width = max(1, int(render_defaults["label_stroke_width_px"]))

    board_bbox = [
        round(board_left, 3),
        round(board_top, 3),
        round(board_right, 3),
        round(board_bottom, 3),
    ]
    graph_origin_px = _graph_origin(
        board_left_px=board_left,
        board_top_px=board_top,
        board_rows=int(board_rows),
        cell_size_px=cell_size,
    )

    for col in range(board_cols + 1):
        x = float(board_left + (col * cell_size))
        draw.line(
            [(x, board_top), (x, board_bottom)],
            fill=tuple(int(v) for v in theme.board_grid_rgb),
            width=grid_width,
        )
    for row in range(board_rows + 1):
        y = float(board_top + (row * cell_size))
        draw.line(
            [(board_left, y), (board_right, y)],
            fill=tuple(int(v) for v in theme.board_grid_rgb),
            width=grid_width,
        )

    axis_color = tuple(int(v) for v in theme.board_outline_rgb)
    draw.line(
        [(board_left, board_bottom), (board_right, board_bottom)],
        fill=axis_color,
        width=outline_width,
    )
    draw.line(
        [(board_left, board_bottom), (board_left, board_top)],
        fill=axis_color,
        width=outline_width,
    )
    draw.line(
        [(board_right, board_top), (board_right, board_bottom)],
        fill=axis_color,
        width=grid_width,
    )
    draw.line(
        [(board_left, board_top), (board_right, board_top)],
        fill=axis_color,
        width=grid_width,
    )

    draw_arrow(
        draw,
        start=(float(board_right - (0.6 * cell_size)), float(board_bottom)),
        end=(float(board_right + (0.55 * cell_size)), float(board_bottom)),
        fill=axis_color,
        width=max(2, int(outline_width)),
        head_length_px=max(10.0, 0.28 * cell_size),
        head_width_px=max(10.0, 0.22 * cell_size),
    )
    draw_arrow(
        draw,
        start=(float(board_left), float(board_top + (0.6 * cell_size))),
        end=(float(board_left), float(board_top - (0.55 * cell_size))),
        fill=axis_color,
        width=max(2, int(outline_width)),
        head_length_px=max(10.0, 0.28 * cell_size),
        head_width_px=max(10.0, 0.22 * cell_size),
    )

    label_fill = tuple(int(v) for v in theme.target_text_rgb)
    label_stroke = tuple(int(v) for v in resolve_text_stroke_fill(theme.target_text_rgb))
    for col in range(board_cols):
        center = _cell_center(
            board_left_px=board_left,
            board_top_px=board_top,
            cell_size_px=cell_size,
            col=int(col),
            row=int(board_rows - 1),
        )
        draw_centered_text(
            draw,
            text=str(int(col)),
            center=(float(center[0]), float(board_bottom + (0.42 * cell_size))),
            font=target_font,
            fill=label_fill,
            stroke_fill=label_stroke,
            stroke_width=int(label_stroke_width),
        )
    for row in range(board_rows):
        graph_y = int((int(board_rows) - 1) - int(row))
        center = _cell_center(
            board_left_px=board_left,
            board_top_px=board_top,
            cell_size_px=cell_size,
            col=0,
            row=int(row),
        )
        draw_centered_text(
            draw,
            text=str(graph_y),
            center=(float(board_left - (0.42 * cell_size)), float(center[1])),
            font=target_font,
            fill=label_fill,
            stroke_fill=label_stroke,
            stroke_width=int(label_stroke_width),
        )

    scene_entities: List[Dict[str, Any]] = []
    mirror_specs: List[OpticsMirrorSpec] = []
    target_specs: List[OpticsTargetSpec] = []
    bounce_specs: List[OpticsBounceSpec] = []

    source_center_y = _cell_center(
        board_left_px=board_left,
        board_top_px=board_top,
        cell_size_px=cell_size,
        col=0,
        row=int(source_row),
    )[1]
    initial_ray_start = (float(source_point_px[0]), float(source_center_y))
    initial_ray_end = (
        float(board_left + (0.38 * cell_size)),
        float(source_center_y),
    )
    draw_arrow(
        draw,
        start=initial_ray_start,
        end=initial_ray_end,
        fill=tuple(int(v) for v in theme.ray_rgb),
        width=ray_width,
        head_length_px=ray_head_length,
        head_width_px=ray_head_width,
    )
    source_bbox = [
        round(float(min(initial_ray_start[0], initial_ray_end[0]) - (0.15 * cell_size)), 3),
        round(float(initial_ray_start[1] - (0.18 * cell_size)), 3),
        round(float(max(initial_ray_start[0], initial_ray_end[0]) + (0.15 * cell_size)), 3),
        round(float(initial_ray_start[1] + (0.18 * cell_size)), 3),
    ]
    scene_entities.append(
        {
            "entity_id": "source_direction",
            "entity_type": "physics_optics_source_direction",
            "bbox_px": list(source_bbox),
            "meta": {"row": int(source_row), "entry_direction": "E"},
        }
    )

    for index, item in enumerate(mirrors, start=1):
        col = int(item["col"])
        row = int(item["row"])
        orientation = str(item["orientation"])
        hit = bool(item.get("hit", False))
        bbox = _cell_bbox(
            board_left_px=board_left,
            board_top_px=board_top,
            cell_size_px=cell_size,
            col=col,
            row=row,
        )
        left, top, right, bottom = [float(v) for v in bbox]
        if orientation == "/":
            start = (float(left + mirror_padding), float(bottom - mirror_padding))
            end = (float(right - mirror_padding), float(top + mirror_padding))
        else:
            start = (float(left + mirror_padding), float(top + mirror_padding))
            end = (float(right - mirror_padding), float(bottom - mirror_padding))
        draw.line(
            [start, end],
            fill=tuple(int(v) for v in theme.mirror_rgb),
            width=mirror_width,
        )
        mirror_id = f"mirror_{int(index)}"
        mirror_specs.append(
            OpticsMirrorSpec(
                mirror_id=mirror_id,
                col=col,
                row=row,
                orientation=orientation,
                hit=bool(hit),
                bbox_px=list(bbox),
            )
        )
        scene_entities.append(
            {
                "entity_id": mirror_id,
                "entity_type": "physics_optics_mirror",
                "bbox_px": list(bbox),
                "meta": {
                    "col": int(col),
                    "row": int(row),
                    "orientation": str(orientation),
                    "hit": bool(hit),
                },
            }
        )

    for index, (col, row) in enumerate(bounce_cells, start=1):
        point_px = _cell_center(
            board_left_px=board_left,
            board_top_px=board_top,
            cell_size_px=cell_size,
            col=int(col),
            row=int(row),
        )
        bbox = [
            round(float(point_px[0] - 8.0), 3),
            round(float(point_px[1] - 8.0), 3),
            round(float(point_px[0] + 8.0), 3),
            round(float(point_px[1] + 8.0), 3),
        ]
        bounce_id = f"bounce_{int(index)}"
        bounce_specs.append(
            OpticsBounceSpec(
                bounce_id=bounce_id,
                col=int(col),
                row=int(row),
                bbox_px=list(bbox),
                point_px=[round(float(point_px[0]), 3), round(float(point_px[1]), 3)],
            )
        )
        scene_entities.append(
            {
                "entity_id": bounce_id,
                "entity_type": "physics_optics_bounce_point",
                "bbox_px": list(bbox),
                "meta": {
                    "col": int(col),
                    "row": int(row),
                },
            }
        )

    for item in targets:
        col = int(item["col"])
        row = int(item["row"])
        hit = bool(item.get("hit", False))
        center = _cell_center(
            board_left_px=board_left,
            board_top_px=board_top,
            cell_size_px=cell_size,
            col=col,
            row=row,
        )
        bbox = [
            round(float(center[0] - target_radius), 3),
            round(float(center[1] - target_radius), 3),
            round(float(center[0] + target_radius), 3),
            round(float(center[1] + target_radius), 3),
        ]
        draw.ellipse(
            tuple(bbox),
            fill=tuple(int(v) for v in theme.target_fill_rgb),
            outline=tuple(int(v) for v in theme.target_outline_rgb),
            width=max(2, int(round(0.08 * cell_size))),
        )
        target_id = str(item["target_id"])
        target_specs.append(
            OpticsTargetSpec(
                target_id=target_id,
                col=col,
                row=row,
                hit=bool(hit),
                bbox_px=list(bbox),
                point_px=[round(float(center[0]), 3), round(float(center[1]), 3)],
            )
        )
        scene_entities.append(
            {
                "entity_id": target_id,
                "entity_type": "physics_optics_target_point",
                "bbox_px": list(bbox),
                "meta": {"col": col, "row": row, "hit": bool(hit)},
            }
        )

    render_map = {
        "accent_color_name": str(accent_color_name),
        "scene_variant": str(scene_variant),
        "ray_event_kind": str(ray_event_kind),
        "board_bbox_px": list(board_bbox),
        "graph_origin_px": [
            round(float(graph_origin_px[0]), 3),
            round(float(graph_origin_px[1]), 3),
        ],
        "graph_spacing_px": int(round(cell_size)),
        "source_direction_bbox_px": list(source_bbox),
        "source_direction_px": [
            [round(float(initial_ray_start[0]), 3), round(float(initial_ray_start[1]), 3)],
            [round(float(initial_ray_end[0]), 3), round(float(initial_ray_end[1]), 3)],
        ],
        "mirror_bboxes_px": {spec.mirror_id: list(spec.bbox_px) for spec in mirror_specs},
        "target_point_map_px": {
            spec.target_id: list(spec.point_px) for spec in target_specs
        },
        "bounce_point_map_px": {
            spec.bounce_id: list(spec.point_px) for spec in bounce_specs
        },
        "ray_polyline_px": [
            [round(float(x), 3), round(float(y), 3)]
            for x, y in (
                [tuple(float(v) for v in source_point_px)]
                + [
                    _cell_center(
                        board_left_px=board_left,
                        board_top_px=board_top,
                        cell_size_px=cell_size,
                        col=int(col),
                        row=int(row),
                    )
                    for col, row in ray_polyline_cells
                ]
                + [tuple(float(v) for v in exit_point_px)]
            )
        ],
        "annotation_entity_ids": [str(item) for item in annotation_entity_ids],
    }
    if diagram_style is not None:
        render_map["technical_diagram_frame_mode"] = str(
            getattr(diagram_style, "frame_mode", "none")
        )
    return RenderedOpticsScene(
        image=canvas,
        mirror_specs=list(mirror_specs),
        target_specs=list(target_specs),
        bounce_specs=list(bounce_specs),
        graph_origin_px=[
            round(float(graph_origin_px[0]), 3),
            round(float(graph_origin_px[1]), 3),
        ],
        graph_spacing_px=int(round(cell_size)),
        annotation_entity_ids=[str(item) for item in annotation_entity_ids],
        render_map=render_map,
        scene_entities=list(scene_entities),
    )


__all__ = ["render_optics_ray_scene"]
