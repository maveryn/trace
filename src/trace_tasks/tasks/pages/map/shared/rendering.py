"""Rendering helpers for printed map scene packages."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.drawing import draw_rounded_rect
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter
from trace_tasks.tasks.pages.shared.diagram.common import (
    draw_diagram_text_in_box,
    resolve_diagrams_int_param,
    resolve_diagrams_rgb_triple,
    resolve_jittered_diagram_panel_geometry,
    round_diagram_bbox,
)

from .defaults import POST_IMAGE_BACKGROUND_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, RENDERING_DEFAULTS
from .state import BBox, MapRenderParams, MapSceneCase, Point, RenderedMapBundle, RenderedMapScene


ZONE_FILLS: Tuple[Tuple[int, int, int], ...] = (
    (222, 237, 220),
    (221, 233, 243),
    (239, 229, 209),
    (235, 223, 236),
)


def resolve_map_render_params(
    params: Mapping[str, Any],
    *,
    instance_seed: int | None = None,
) -> MapRenderParams:
    """Resolve rendering params for printed campus-map scenes."""

    def _int(key: str, fallback: int) -> int:
        return resolve_diagrams_int_param(
            params,
            RENDERING_DEFAULTS,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="pages.map",
        )

    def _triple(key: str, fallback: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return resolve_diagrams_rgb_triple(
            params,
            RENDERING_DEFAULTS,
            key,
            fallback,
            instance_seed=instance_seed,
            namespace="pages.map",
        )

    layout_jitter_meta = resolve_layout_jitter(
        params,
        RENDERING_DEFAULTS,
        instance_seed=instance_seed,
        namespace="pages.map.layout",
    )

    return MapRenderParams(
        canvas_width=_int("canvas_width", 1280),
        canvas_height=_int("canvas_height", 900),
        outer_margin_px=_int("outer_margin_px", 46),
        panel_padding_px=_int("panel_padding_px", 26),
        panel_corner_radius_px=_int("panel_corner_radius_px", 20),
        title_font_size_px=_int("title_font_size_px", 30),
        title_band_height_px=_int("title_band_height_px", 70),
        map_corner_radius_px=_int("map_corner_radius_px", 24),
        map_border_width_px=_int("map_border_width_px", 3),
        path_width_px=_int("path_width_px", 12),
        highlighted_path_width_px=_int("highlighted_path_width_px", 18),
        landmark_width_px=_int("landmark_width_px", 126),
        landmark_height_px=_int("landmark_height_px", 58),
        landmark_corner_radius_px=_int("landmark_corner_radius_px", 12),
        landmark_border_width_px=_int("landmark_border_width_px", 3),
        landmark_label_font_size_px=_int("landmark_label_font_size_px", 18),
        zone_label_font_size_px=_int("zone_label_font_size_px", 24),
        legend_font_size_px=_int("legend_font_size_px", 17),
        compass_font_size_px=_int("compass_font_size_px", 18),
        panel_fill_rgb=_triple("panel_fill_rgb", (252, 252, 249)),
        panel_border_rgb=_triple("panel_border_rgb", (77, 88, 99)),
        title_color_rgb=_triple("title_color_rgb", (34, 40, 48)),
        map_fill_rgb=_triple("map_fill_rgb", (246, 247, 240)),
        map_border_rgb=_triple("map_border_rgb", (86, 96, 91)),
        zone_border_rgb=_triple("zone_border_rgb", (196, 202, 188)),
        path_rgb=_triple("path_rgb", (184, 186, 176)),
        highlighted_path_rgb=_triple("highlighted_path_rgb", (205, 92, 46)),
        landmark_fill_rgb=_triple("landmark_fill_rgb", (255, 255, 252)),
        landmark_border_rgb=_triple("landmark_border_rgb", (72, 88, 104)),
        landmark_label_rgb=_triple("landmark_label_rgb", (28, 34, 42)),
        label_stroke_rgb=_triple("label_stroke_rgb", (255, 255, 255)),
        zone_label_rgb=_triple("zone_label_rgb", (72, 84, 78)),
        compass_rgb=_triple("compass_rgb", (45, 52, 60)),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def _grid_point(*, cell_col: int, cell_row: int, map_bbox: BBox, grid_cols: int, grid_rows: int) -> Point:
    left, top, right, bottom = [float(value) for value in map_bbox]
    usable_left = float(left + 82.0)
    usable_top = float(top + 72.0)
    usable_right = float(right - 82.0)
    usable_bottom = float(bottom - 86.0)
    x = usable_left + ((float(cell_col) + 0.5) * (usable_right - usable_left) / float(grid_cols))
    y = usable_top + ((float(cell_row) + 0.5) * (usable_bottom - usable_top) / float(grid_rows))
    return (float(x), float(y))


def _landmark_bbox(center: Point, *, render_params: MapRenderParams) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    half_w = 0.5 * float(render_params.landmark_width_px)
    half_h = 0.5 * float(render_params.landmark_height_px)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _line_bbox(start: Point, end: Point, *, width: int) -> List[float]:
    pad = 0.5 * float(width)
    return round_diagram_bbox(
        (
            min(float(start[0]), float(end[0])) - pad,
            min(float(start[1]), float(end[1])) - pad,
            max(float(start[0]), float(end[0])) + pad,
            max(float(start[1]), float(end[1])) + pad,
        )
    )


def _zone_bbox_from_cell_bounds(
    *,
    cell_bounds: Sequence[int],
    map_bbox: BBox,
    grid_cols: int,
    grid_rows: int,
) -> BBox:
    col0, row0, col1, row1 = [int(value) for value in cell_bounds]
    left, top, right, bottom = [float(value) for value in map_bbox]
    width = float(right - left)
    height = float(bottom - top)
    return (
        float(left + (width * float(col0) / float(grid_cols))),
        float(top + (height * float(row0) / float(grid_rows))),
        float(left + (width * float(col1 + 1) / float(grid_cols))),
        float(top + (height * float(row1 + 1) / float(grid_rows))),
    )


def _draw_compass(draw: ImageDraw.ImageDraw, *, center: Point, render_params: MapRenderParams) -> None:
    cx, cy = float(center[0]), float(center[1])
    color = tuple(int(value) for value in render_params.compass_rgb)
    draw.line([(cx, cy + 26), (cx, cy - 26)], fill=color, width=3)
    draw.line([(cx - 19, cy), (cx + 19, cy)], fill=color, width=3)
    draw.polygon([(cx, cy - 36), (cx - 8, cy - 21), (cx + 8, cy - 21)], fill=color)
    draw_diagram_text_in_box(
        draw,
        bbox=(cx - 18, cy - 62, cx + 18, cy - 36),
        text="N",
        font_size_px=int(render_params.compass_font_size_px),
        bold=True,
        fill=color,
        stroke_fill=render_params.label_stroke_rgb,
        padding_px=1,
    )


def _boxes_overlap(first: BBox, second: BBox, *, pad: float = 8.0) -> bool:
    return not (
        float(first[2]) + float(pad) <= float(second[0])
        or float(second[2]) + float(pad) <= float(first[0])
        or float(first[3]) + float(pad) <= float(second[1])
        or float(second[3]) + float(pad) <= float(first[1])
    )


def _zone_label_candidates(zone_bbox: BBox) -> List[BBox]:
    left, top, right, bottom = [float(value) for value in zone_bbox]
    center_x = 0.5 * (left + right)
    width = min(320.0, max(180.0, float(right - left) - 44.0))
    half_width = 0.5 * width
    return [
        (center_x - half_width, top + 16.0, center_x + half_width, top + 58.0),
        (center_x - half_width, bottom - 74.0, center_x + half_width, bottom - 32.0),
        (left + 22.0, top + 16.0, min(right - 22.0, left + 22.0 + width), top + 58.0),
        (max(left + 22.0, right - 22.0 - width), top + 16.0, right - 22.0, top + 58.0),
        (left + 22.0, bottom - 74.0, min(right - 22.0, left + 22.0 + width), bottom - 32.0),
        (max(left + 22.0, right - 22.0 - width), bottom - 74.0, right - 22.0, bottom - 32.0),
    ]


def _choose_zone_label_bbox(*, zone_bbox: BBox, obstacles: Sequence[BBox]) -> BBox:
    candidates = _zone_label_candidates(zone_bbox)
    for candidate in candidates:
        if all(not _boxes_overlap(candidate, obstacle) for obstacle in obstacles):
            return candidate
    return candidates[0]


def render_map_scene(
    background: Image.Image,
    *,
    scene_title: str,
    grid_cols: int,
    grid_rows: int,
    zone_specs: Sequence[Mapping[str, object]],
    landmark_specs: Sequence[Mapping[str, object]],
    path_specs: Sequence[Mapping[str, object]],
    highlighted_route_landmark_ids: Sequence[str],
    render_params: MapRenderParams,
) -> RenderedMapScene:
    """Render one static printed campus/facility map."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    entities: List[Dict[str, object]] = []
    landmark_bbox_map: Dict[str, List[float]] = {}
    landmark_label_bbox_map: Dict[str, List[float]] = {}
    zone_label_bbox_map: Dict[str, List[float]] = {}
    path_bbox_map: Dict[str, List[float]] = {}
    highlighted_route_bbox_map: Dict[str, List[float]] = {}

    panel_bbox, title_bbox, content_bbox, layout_jitter_meta = resolve_jittered_diagram_panel_geometry(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        outer_margin_px=int(render_params.outer_margin_px),
        title_band_height_px=int(render_params.title_band_height_px),
        panel_padding_px=int(render_params.panel_padding_px),
        layout_jitter_meta=render_params.layout_jitter_meta,
    )
    draw_rounded_rect(
        draw,
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=2,
    )
    title_text_bbox = draw_diagram_text_in_box(
        draw,
        bbox=title_bbox,
        text=str(scene_title),
        font_size_px=int(render_params.title_font_size_px),
        bold=True,
        fill=render_params.title_color_rgb,
        stroke_fill=render_params.panel_fill_rgb,
        padding_px=12,
    )
    entities.append({"entity_id": "map_panel", "entity_type": "map_panel", "bbox_xyxy": round_diagram_bbox(panel_bbox)})
    entities.append(
        {
            "entity_id": "map_title",
            "entity_type": "map_title",
            "bbox_xyxy": list(title_text_bbox),
            "text": str(scene_title),
        }
    )

    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    map_bbox = (
        float(content_left),
        float(content_top),
        float(content_right),
        float(content_bottom - 10.0),
    )
    draw_rounded_rect(
        draw,
        map_bbox,
        radius=int(render_params.map_corner_radius_px),
        fill=render_params.map_fill_rgb,
        outline=render_params.map_border_rgb,
        width=int(render_params.map_border_width_px),
    )
    entities.append({"entity_id": "printed_map", "entity_type": "printed_map", "bbox_xyxy": round_diagram_bbox(map_bbox)})

    centers_by_landmark_id: Dict[str, Point] = {}
    landmark_obstacles: List[BBox] = []
    for landmark_spec in landmark_specs:
        center = _grid_point(
            cell_col=int(landmark_spec["grid_col"]),
            cell_row=int(landmark_spec["grid_row"]),
            map_bbox=map_bbox,
            grid_cols=int(grid_cols),
            grid_rows=int(grid_rows),
        )
        centers_by_landmark_id[str(landmark_spec["landmark_id"])] = center
        landmark_obstacles.append(_landmark_bbox(center, render_params=render_params))

    compass_center = (float(map_bbox[2]) - 58.0, float(map_bbox[1]) + 82.0)
    compass_obstacle = (
        float(compass_center[0]) - 52.0,
        float(compass_center[1]) - 70.0,
        float(compass_center[0]) + 52.0,
        float(compass_center[1]) + 36.0,
    )
    label_obstacles = [*landmark_obstacles, compass_obstacle]

    for index, zone_spec in enumerate(zone_specs):
        zone_bbox = _zone_bbox_from_cell_bounds(
            cell_bounds=zone_spec["cell_bounds"],
            map_bbox=map_bbox,
            grid_cols=int(grid_cols),
            grid_rows=int(grid_rows),
        )
        fill = ZONE_FILLS[index % len(ZONE_FILLS)]
        draw.rectangle(zone_bbox, fill=fill, outline=tuple(int(value) for value in render_params.zone_border_rgb), width=1)
        zone_label_bbox_id = f"{str(zone_spec['zone_id'])}_label_bbox"
        zone_label_slot = _choose_zone_label_bbox(zone_bbox=zone_bbox, obstacles=label_obstacles)
        label_bbox = draw_diagram_text_in_box(
            draw,
            bbox=zone_label_slot,
            text=str(zone_spec["zone_label"]),
            font_size_px=int(render_params.zone_label_font_size_px),
            bold=True,
            fill=render_params.zone_label_rgb,
            stroke_fill=fill,
            padding_px=2,
        )
        zone_label_bbox_map[zone_label_bbox_id] = list(label_bbox)
        entities.append(
            {
                "entity_id": str(zone_spec["zone_id"]),
                "entity_type": "map_zone",
                "bbox_xyxy": round_diagram_bbox(zone_bbox),
                "text": str(zone_spec["zone_label"]),
            }
        )
        entities.append(
            {
                "entity_id": zone_label_bbox_id,
                "entity_type": "map_zone_label",
                "bbox_xyxy": list(label_bbox),
                "zone_id": str(zone_spec["zone_id"]),
                "text": str(zone_spec["zone_label"]),
            }
        )

    # Re-draw the border over zone fills so the page reads as one printed map.
    draw.rounded_rectangle(
        map_bbox,
        radius=int(render_params.map_corner_radius_px),
        fill=None,
        outline=tuple(int(value) for value in render_params.map_border_rgb),
        width=int(render_params.map_border_width_px),
    )

    for path_spec in path_specs:
        source = centers_by_landmark_id[str(path_spec["source_landmark_id"])]
        target = centers_by_landmark_id[str(path_spec["target_landmark_id"])]
        draw.line(
            [source, target],
            fill=tuple(int(value) for value in render_params.path_rgb),
            width=int(render_params.path_width_px),
        )
        bbox = _line_bbox(source, target, width=int(render_params.path_width_px))
        path_bbox_map[str(path_spec["path_bbox_id"])] = bbox
        entities.append(
            {
                "entity_id": str(path_spec["path_id"]),
                "entity_type": "map_path_segment",
                "bbox_xyxy": list(bbox),
                "source_landmark_id": str(path_spec["source_landmark_id"]),
                "target_landmark_id": str(path_spec["target_landmark_id"]),
            }
        )

    highlighted_ids = [str(item) for item in highlighted_route_landmark_ids]
    for index, (source_id, target_id) in enumerate(zip(highlighted_ids, highlighted_ids[1:])):
        source = centers_by_landmark_id[str(source_id)]
        target = centers_by_landmark_id[str(target_id)]
        draw.line(
            [source, target],
            fill=tuple(int(value) for value in render_params.highlighted_path_rgb),
            width=int(render_params.highlighted_path_width_px),
        )
        bbox = _line_bbox(source, target, width=int(render_params.highlighted_path_width_px))
        segment_id = f"highlighted_route_segment_{index}"
        highlighted_route_bbox_map[segment_id] = list(bbox)
        entities.append(
            {
                "entity_id": segment_id,
                "entity_type": "map_highlighted_route_segment",
                "bbox_xyxy": list(bbox),
                "source_landmark_id": str(source_id),
                "target_landmark_id": str(target_id),
            }
        )

    for landmark_spec in landmark_specs:
        landmark_id = str(landmark_spec["landmark_id"])
        landmark_bbox_id = str(landmark_spec["landmark_bbox_id"])
        landmark_label_bbox_id = str(landmark_spec["landmark_label_bbox_id"])
        bbox = _landmark_bbox(centers_by_landmark_id[landmark_id], render_params=render_params)
        draw_rounded_rect(
            draw,
            bbox,
            radius=int(render_params.landmark_corner_radius_px),
            fill=render_params.landmark_fill_rgb,
            outline=render_params.landmark_border_rgb,
            width=int(render_params.landmark_border_width_px),
        )
        label_bbox = draw_diagram_text_in_box(
            draw,
            bbox=bbox,
            text=str(landmark_spec["landmark_label"]),
            font_size_px=int(render_params.landmark_label_font_size_px),
            bold=True,
            fill=render_params.landmark_label_rgb,
            stroke_fill=render_params.label_stroke_rgb,
            padding_px=8,
        )
        landmark_bbox_map[landmark_bbox_id] = round_diagram_bbox(bbox)
        landmark_label_bbox_map[landmark_label_bbox_id] = list(label_bbox)
        entities.append(
            {
                "entity_id": landmark_bbox_id,
                "entity_type": "map_landmark",
                "bbox_xyxy": round_diagram_bbox(bbox),
                "landmark_id": landmark_id,
                "zone_id": str(landmark_spec["zone_id"]),
                "text": str(landmark_spec["landmark_label"]),
            }
        )
        entities.append(
            {
                "entity_id": landmark_label_bbox_id,
                "entity_type": "map_landmark_label",
                "bbox_xyxy": list(label_bbox),
                "landmark_id": landmark_id,
                "text": str(landmark_spec["landmark_label"]),
            }
        )

    _draw_compass(draw, center=compass_center, render_params=render_params)

    return RenderedMapScene(
        image=image,
        entities=entities,
        panel_bbox_px=round_diagram_bbox(panel_bbox),
        title_bbox_px=list(title_text_bbox),
        map_bbox_px=round_diagram_bbox(map_bbox),
        landmark_bbox_map=landmark_bbox_map,
        landmark_label_bbox_map=landmark_label_bbox_map,
        zone_label_bbox_map=zone_label_bbox_map,
        path_bbox_map=path_bbox_map,
        highlighted_route_bbox_map=highlighted_route_bbox_map,
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def render_map_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: MapSceneCase,
    highlighted_route_landmark_ids: Sequence[str],
) -> RenderedMapBundle:
    """Render one sampled map case and attach post-image style metadata."""

    render_params = resolve_map_render_params(params, instance_seed=int(instance_seed))
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    rendered_scene = render_map_scene(
        background,
        scene_title=str(case.scene_title),
        grid_cols=int(case.grid_cols),
        grid_rows=int(case.grid_rows),
        zone_specs=tuple(case.zone_specs),
        landmark_specs=tuple(case.landmark_specs),
        path_specs=tuple(case.path_specs),
        highlighted_route_landmark_ids=[str(item) for item in highlighted_route_landmark_ids],
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedMapBundle(
        image=image,
        render_params=render_params,
        rendered_scene=rendered_scene,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "render_map_case",
    "render_map_scene",
    "resolve_map_render_params",
]
