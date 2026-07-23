"""Rendering and trace entity extraction for pedigree-chart scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import load_font
from ....shared.text_legibility import draw_text_traced
from ...shared.graph_scene import GraphRenderParams
from .state import (
    GENERATION_LABELS,
    SUPPORTED_PEDIGREE_SCENE_VARIANTS,
    BBox,
    Point,
    PedigreeSample,
    RenderedPedigreeFamily,
    RenderedPedigreePerson,
    RenderedPedigreeScene,
    _person_map,
)

def _text_bbox(draw: ImageDraw.ImageDraw, xy: Point, text: str, font) -> BBox:
    bbox = draw.textbbox((int(xy[0]), int(xy[1])), str(text), font=font)
    return tuple(int(value) for value in bbox)  # type: ignore[return-value]


def _center_text(draw: ImageDraw.ImageDraw, center_xy: Point, text: str, font, fill: Tuple[int, int, int]) -> BBox:
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    width = int(text_bbox[2] - text_bbox[0])
    height = int(text_bbox[3] - text_bbox[1])
    xy = (int(center_xy[0] - (width / 2)), int(center_xy[1] - (height / 2)))
    draw_text_traced(
        draw,
        xy,
        str(text),
        fill=tuple(int(value) for value in fill),
        font=font,
        role="graph_pedigree_label_text",
        required=False,
    )
    return _text_bbox(draw, xy, str(text), font)


def _resolve_centers(
    sample: PedigreeSample,
    *,
    render_params: GraphRenderParams,
    bottom_reserved_px: int = 0,
) -> Tuple[Dict[str, Point], Dict[int, int], Dict[int, Tuple[int, int]]]:
    """Lay out pedigree rows with stable generation spacing and bounded horizontal extents."""

    generations = sorted({int(person.generation_index) for person in sample.people})
    panel_left = int(render_params.outer_margin_px)
    panel_top = int(render_params.outer_margin_px)
    panel_right = int(render_params.canvas_width - render_params.outer_margin_px)
    panel_bottom = int(render_params.canvas_height - render_params.outer_margin_px)
    content_left = int(panel_left + render_params.panel_padding_px + 76)
    content_right = int(panel_right - render_params.panel_padding_px - 22)
    content_top = int(panel_top + render_params.panel_padding_px + 78)
    content_bottom = int(panel_bottom - render_params.panel_padding_px - 42 - max(0, int(bottom_reserved_px)))
    row_count = max(1, len(generations))
    if row_count == 1:
        row_y = {generations[0]: int((content_top + content_bottom) / 2)}
    else:
        row_y = {
            int(generation): int(round(content_top + ((content_bottom - content_top) * index / (row_count - 1))))
            for index, generation in enumerate(generations)
        }
    centers: Dict[str, Point] = {}
    row_extents: Dict[int, Tuple[int, int]] = {}
    for generation in generations:
        people = [person for person in sample.people if int(person.generation_index) == int(generation)]
        count = len(people)
        if count == 1:
            xs = [int((content_left + content_right) / 2)]
        else:
            available_width = int(content_right - content_left)
            spacing = min(112.0, float(available_width) / float(max(1, count - 1)))
            row_width = spacing * float(count - 1)
            start_x = (float(content_left + content_right) / 2.0) - (row_width / 2.0)
            xs = [int(round(start_x + (spacing * index))) for index in range(count)]
        for person, x in zip(people, xs):
            centers[str(person.person_id)] = (int(x), int(row_y[int(generation)]))
        row_extents[int(generation)] = (int(content_left), int(content_right))
    return centers, row_y, row_extents


def _draw_person_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    center_xy: Point,
    sex: str,
    affected: bool,
    render_params: GraphRenderParams,
) -> BBox:
    radius = int(render_params.node_radius_px)
    x, y = int(center_xy[0]), int(center_xy[1])
    bbox = (int(x - radius), int(y - radius), int(x + radius), int(y + radius))
    fill_rgb = tuple(int(value) for value in (render_params.node_fill_rgb if bool(affected) else render_params.panel_fill_rgb))
    outline_rgb = tuple(int(value) for value in render_params.node_border_rgb)
    if str(sex) == "female":
        draw.ellipse(
            bbox,
            fill=fill_rgb,
            outline=outline_rgb,
            width=int(render_params.node_border_width_px),
        )
    else:
        draw.rectangle(
            bbox,
            fill=fill_rgb,
            outline=outline_rgb,
            width=int(render_params.node_border_width_px),
        )
    return tuple(int(value) for value in bbox)  # type: ignore[return-value]


def render_pedigree_chart_scene(
    *,
    sample: PedigreeSample,
    render_params: GraphRenderParams,
    scene_variant: str,
    scene_title: str,
    base_image: Image.Image,
    highlighted_person_ids: Sequence[str] = (),
    bottom_reserved_px: int = 0,
) -> RenderedPedigreeScene:
    """Render one pedigree chart and trace all symbol projections."""

    variant = str(scene_variant)
    if variant not in set(SUPPORTED_PEDIGREE_SCENE_VARIANTS):
        variant = "classic_pedigree"
    image = base_image
    draw = ImageDraw.Draw(image)
    panel_left = int(render_params.outer_margin_px)
    panel_top = int(render_params.outer_margin_px)
    panel_right = int(render_params.canvas_width - render_params.outer_margin_px)
    panel_bottom = int(render_params.canvas_height - render_params.outer_margin_px)
    panel_bbox = (panel_left, panel_top, panel_right, panel_bottom)
    draw.rounded_rectangle(
        panel_bbox,
        radius=int(render_params.panel_corner_radius_px),
        fill=tuple(int(value) for value in render_params.panel_fill_rgb),
        outline=tuple(int(value) for value in render_params.panel_border_rgb),
        width=2,
    )

    title_font = load_font(
        int(render_params.panel_title_font_size_px),
        bold=True,
        font_family=str(render_params.font_family),
    )
    label_font = load_font(
        int(render_params.label_font_size_px),
        bold=True,
        font_family=str(render_params.font_family),
    )
    small_font = load_font(
        max(12, int(round(render_params.label_font_size_px * 0.78))),
        bold=True,
        font_family=str(render_params.font_family),
    )
    draw_text_traced(
        draw,
        (int(panel_left + render_params.panel_padding_px), int(panel_top + render_params.panel_padding_px)),
        str(scene_title),
        fill=tuple(int(value) for value in render_params.title_color_rgb),
        font=title_font,
        role="graph_pedigree_title_text",
        required=False,
    )
    subtitle = "Squares are male; circles are female"
    draw_text_traced(
        draw,
        (
            int(panel_left + render_params.panel_padding_px),
            int(panel_top + render_params.panel_padding_px + render_params.panel_title_font_size_px + 8),
        ),
        subtitle,
        fill=tuple(int(value) for value in render_params.edge_color_rgb),
        font=small_font,
        role="graph_pedigree_context_text",
        required=False,
    )

    centers, row_y, row_extents = _resolve_centers(
        sample,
        render_params=render_params,
        bottom_reserved_px=int(bottom_reserved_px),
    )
    if variant == "row_guided_pedigree":
        for generation, y in sorted(row_y.items()):
            left, right = row_extents[int(generation)]
            band_height = int(max(42, render_params.node_radius_px * 2 + 22))
            draw.rounded_rectangle(
                (int(left - 46), int(y - band_height / 2), int(right + 8), int(y + band_height / 2)),
                radius=12,
                fill=tuple(max(0, min(255, int(value))) for value in (246, 248, 252)),
                outline=None,
            )
    elif variant == "paper_pedigree":
        for generation, y in sorted(row_y.items()):
            left, right = row_extents[int(generation)]
            draw.line(
                (int(left - 44), int(y), int(right + 8), int(y)),
                fill=tuple(int(value) for value in render_params.panel_border_rgb),
                width=1,
            )

    people_by_id = _person_map(sample)
    connector_color = tuple(int(value) for value in render_params.edge_color_rgb)
    connector_width = max(2, int(render_params.edge_width_px))
    rendered_families: List[RenderedPedigreeFamily] = []
    for family in sample.families:
        parent_a, parent_b = str(family.parent_ids[0]), str(family.parent_ids[1])
        if parent_a not in centers or parent_b not in centers:
            continue
        pa = centers[parent_a]
        pb = centers[parent_b]
        radius = int(render_params.node_radius_px)
        spouse_start = (int(pa[0] + radius), int(pa[1]))
        spouse_end = (int(pb[0] - radius), int(pb[1]))
        if pa[0] > pb[0]:
            spouse_start = (int(pa[0] - radius), int(pa[1]))
            spouse_end = (int(pb[0] + radius), int(pb[1]))
        draw.line((spouse_start[0], spouse_start[1], spouse_end[0], spouse_end[1]), fill=connector_color, width=connector_width)
        descent_segments: List[Tuple[Point, Point]] = []
        child_centers = [centers[str(child_id)] for child_id in family.child_ids if str(child_id) in centers]
        if child_centers:
            parent_mid = (int(round((pa[0] + pb[0]) / 2)), int(pa[1]))
            child_y = int(child_centers[0][1])
            mid_y = int(round((parent_mid[1] + child_y) / 2))
            vertical_parent = ((int(parent_mid[0]), int(parent_mid[1])), (int(parent_mid[0]), int(mid_y)))
            draw.line((vertical_parent[0][0], vertical_parent[0][1], vertical_parent[1][0], vertical_parent[1][1]), fill=connector_color, width=connector_width)
            descent_segments.append(vertical_parent)
            child_xs = [int(point[0]) for point in child_centers]
            left_x = int(min(child_xs))
            right_x = int(max(child_xs))
            if left_x != right_x:
                sibship = ((left_x, mid_y), (right_x, mid_y))
                draw.line((left_x, mid_y, right_x, mid_y), fill=connector_color, width=connector_width)
                descent_segments.append(sibship)
            for child_center in child_centers:
                child_segment = ((int(child_center[0]), int(mid_y)), (int(child_center[0]), int(child_center[1] - radius)))
                draw.line(
                    (child_segment[0][0], child_segment[0][1], child_segment[1][0], child_segment[1][1]),
                    fill=connector_color,
                    width=connector_width,
                )
                descent_segments.append(child_segment)
        rendered_families.append(
            RenderedPedigreeFamily(
                pedigree_id=str(family.pedigree_id),
                parent_ids=tuple(str(parent_id) for parent_id in family.parent_ids),
                child_ids=tuple(str(child_id) for child_id in family.child_ids),
                spouse_segment_px=(spouse_start, spouse_end),
                descent_segments_px=tuple(descent_segments),
            )
        )

    generation_label_bboxes: Dict[str, BBox] = {}
    for generation, y in sorted(row_y.items()):
        generation_label = str(GENERATION_LABELS[int(generation)])
        generation_label_bboxes[generation_label] = _center_text(
            draw,
            (int(panel_left + render_params.panel_padding_px + 30), int(y)),
            generation_label,
            small_font,
            tuple(int(value) for value in render_params.title_color_rgb),
        )

    rendered_people: List[RenderedPedigreePerson] = []
    highlighted_set = {str(person_id) for person_id in highlighted_person_ids}
    for person in sample.people:
        center = centers[str(person.person_id)]
        symbol_bbox = _draw_person_symbol(
            draw,
            center_xy=center,
            sex=str(person.sex),
            affected=bool(person.affected),
            render_params=render_params,
        )
        label_y = int(center[1] + render_params.node_radius_px + 13)
        label_bbox = _center_text(
            draw,
            (int(center[0]), int(label_y)),
            str(person.label),
            label_font,
            tuple(int(value) for value in render_params.title_color_rgb),
        )
        rendered_people.append(
            RenderedPedigreePerson(
                person_id=str(person.person_id),
                label=str(person.label),
                generation_index=int(person.generation_index),
                generation_label=str(person.generation_label),
                sex=str(person.sex),
                affected=bool(person.affected),
                center_xy=(int(center[0]), int(center[1])),
                symbol_bbox_xyxy=tuple(int(value) for value in symbol_bbox),
                label_bbox_xyxy=tuple(int(value) for value in label_bbox),
            )
        )

    if highlighted_set:
        highlight_rgb = (211, 118, 36)
        for rendered_person in rendered_people:
            if str(rendered_person.person_id) not in highlighted_set:
                continue
            x0, y0, x1, y1 = rendered_person.symbol_bbox_xyxy
            pad = max(5, int(round(render_params.node_radius_px * 0.28)))
            highlight_bbox = (int(x0 - pad), int(y0 - pad), int(x1 + pad), int(y1 + pad))
            if str(rendered_person.sex) == "female":
                draw.ellipse(highlight_bbox, outline=highlight_rgb, width=3)
            else:
                draw.rounded_rectangle(highlight_bbox, radius=6, outline=highlight_rgb, width=3)

    panel_geometry = {
        "canvas_size": [int(render_params.canvas_width), int(render_params.canvas_height)],
        "panel_bbox_xyxy": [int(value) for value in panel_bbox],
        "generation_row_y": {str(GENERATION_LABELS[int(key)]): int(value) for key, value in sorted(row_y.items())},
        "highlighted_person_ids": sorted(highlighted_set),
        "bottom_reserved_px": int(max(0, int(bottom_reserved_px))),
    }
    return RenderedPedigreeScene(
        image=image,
        panel_geometry=dict(panel_geometry),
        people=tuple(rendered_people),
        families=tuple(rendered_families),
        scene_variant=str(variant),
        resolved_label_font_size_px=int(render_params.label_font_size_px),
        resolved_label_stroke_width_px=0,
        generation_label_bboxes={
            str(key): tuple(int(value) for value in bbox)
            for key, bbox in generation_label_bboxes.items()
        },
    )


def pedigree_scene_entities(sample: PedigreeSample, rendered_scene: RenderedPedigreeScene) -> Tuple[Dict[str, Any], ...]:
    """Return trace entity records for rendered pedigree people."""

    rendered_by_id = {str(person.person_id): person for person in rendered_scene.people}
    entities: List[Dict[str, Any]] = []
    for person in sample.people:
        rendered = rendered_by_id[str(person.person_id)]
        entities.append(
            {
                "entity_id": str(person.person_id),
                "entity_kind": "pedigree_person",
                "label": str(person.label),
                "generation_index": int(person.generation_index),
                "generation_label": str(person.generation_label),
                "sex": str(person.sex),
                "affected": bool(person.affected),
                "is_counted": str(person.person_id) in set(sample.counted_person_ids),
                "center_xy": [int(rendered.center_xy[0]), int(rendered.center_xy[1])],
                "symbol_bbox_xyxy": [int(value) for value in rendered.symbol_bbox_xyxy],
                "label_bbox_xyxy": [int(value) for value in rendered.label_bbox_xyxy],
            }
        )
    return tuple(entities)


def pedigree_connector_relations(sample: PedigreeSample, rendered_scene: RenderedPedigreeScene) -> Tuple[Dict[str, Any], ...]:
    """Return trace relation records for pedigree family connectors."""

    rendered_by_id = {str(family.pedigree_id): family for family in rendered_scene.families}
    relations: List[Dict[str, Any]] = []
    for family in sample.families:
        rendered = rendered_by_id.get(str(family.pedigree_id))
        relations.append(
            {
                "pedigree_id": str(family.pedigree_id),
                "parent_ids": list(family.parent_ids),
                "child_ids": list(family.child_ids),
                "spouse_segment_px": [list(point) for point in rendered.spouse_segment_px] if rendered is not None else [],
                "descent_segments_px": [
                    [list(point) for point in segment]
                    for segment in (rendered.descent_segments_px if rendered is not None else ())
                ],
            }
        )
    return tuple(relations)



__all__ = [
    "pedigree_connector_relations",
    "pedigree_scene_entities",
    "render_pedigree_chart_scene",
]
