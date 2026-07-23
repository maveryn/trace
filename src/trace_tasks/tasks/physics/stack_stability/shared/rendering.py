"""Rendering primitives for stack-stability brick-stack diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many as bbox_union
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.render_variation import resolve_render_int
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .layout import bbox_from_center, expand_bbox
from .state import (
    SCENE_NAMESPACE,
    RenderedStack,
    RenderedStackScene,
    StackCandidateSpec,
    StackSceneSpec,
    StackTaskDefaults,
)


def _draw_dashed_vertical_line(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y0: float,
    y1: float,
    fill: Tuple[int, int, int],
    width: int,
    dash_px: float = 10.0,
    gap_px: float = 7.0,
) -> None:
    top = min(float(y0), float(y1))
    bottom = max(float(y0), float(y1))
    y = top
    while y < bottom:
        y_next = min(bottom, y + float(dash_px))
        draw.line(
            [(float(x), float(y)), (float(x), float(y_next))],
            fill=fill,
            width=int(width),
        )
        y = y_next + float(gap_px)


def _lighten(rgb: Tuple[int, int, int], amount: int = 28) -> Tuple[int, int, int]:
    return tuple(min(255, int(value) + int(amount)) for value in rgb)


def resolve_stack_render_defaults(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    defaults: StackTaskDefaults,
) -> Dict[str, int]:
    """Resolve all scene-local render integers with deterministic jitter."""

    keys = (
        "board_left_px",
        "board_top_px",
        "board_right_margin_px",
        "board_bottom_margin_px",
        "cell_gap_x_px",
        "cell_gap_y_px",
        "brick_width_px",
        "brick_height_px",
        "brick_gap_px",
        "label_font_size_px",
        "title_font_size_px",
        "small_font_size_px",
        "label_stroke_width_px",
        "support_width_px",
        "projection_width_px",
        "com_radius_px",
    )
    return {
        str(key): resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(getattr(defaults, str(key))),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        )
        for key in keys
    }


def _render_candidate(
    *,
    draw: ImageDraw.ImageDraw,
    candidate: StackCandidateSpec,
    cell_bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    style: Any,
    font_family: str,
    instance_seed: int,
) -> RenderedStack:
    """Render one option cell and return projected geometry for its stack."""

    cell_left, cell_top, cell_right, cell_bottom = [float(value) for value in cell_bbox]
    brick_width = float(render_defaults["brick_width_px"])
    brick_height = float(render_defaults["brick_height_px"])
    brick_gap = float(render_defaults["brick_gap_px"])
    rng = spawn_rng(
        int(instance_seed),
        f"{SCENE_NAMESPACE}.candidate.{candidate.label}",
    )
    base_x = float((cell_left + cell_right) / 2.0 + rng.randint(-10, 10))
    ground_y = float(cell_bottom - 34.0 + rng.randint(-3, 4))
    top_margin = float(cell_top + 54.0)
    row_count = len(candidate.row_offsets)
    stack_height = row_count * brick_height + max(0, row_count - 1) * brick_gap
    if ground_y - stack_height < top_margin:
        ground_y = top_margin + stack_height

    label_font = load_font(
        int(render_defaults["label_font_size_px"]),
        bold=True,
        font_family=font_family,
    )
    small_font = load_font(
        int(render_defaults["small_font_size_px"]),
        bold=True,
        font_family=font_family,
    )
    label_rgb = tuple(int(v) for v in style.label_rgb)
    guide_rgb = tuple(int(v) for v in style.guide_rgb)
    support_rgb = tuple(int(v) for v in style.stroke_rgb)
    projection_rgb = tuple(int(v) for v in style.accent_rgb)
    com_fill = (223, 48, 54)
    com_outline = (112, 30, 34)

    draw.rounded_rectangle(
        [cell_left, cell_top, cell_right, cell_bottom],
        radius=14,
        fill=tuple(int(v) for v in style.panel_alt_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=2,
    )
    draw.line(
        [(cell_left + 22, ground_y), (cell_right - 22, ground_y)],
        fill=guide_rgb,
        width=2,
    )

    label_center = (cell_left + 34, cell_top + 30)
    label_bbox = bbox_from_center(label_center, 20, 18)
    draw.rounded_rectangle(
        label_bbox,
        radius=8,
        fill=tuple(int(v) for v in style.label_fill_rgb),
        outline=tuple(int(v) for v in style.label_border_rgb),
        width=2,
    )
    draw_centered_text(
        draw,
        text=str(candidate.label),
        center=label_center,
        font=label_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=int(render_defaults["label_stroke_width_px"]),
    )

    brick_bboxes: list[list[float]] = []
    brick_centers: list[Tuple[float, float]] = []
    for row_index, offset_units in enumerate(candidate.row_offsets):
        cx = float(base_x + float(offset_units) * brick_width)
        y1 = float(ground_y - int(row_index) * (brick_height + brick_gap))
        y0 = float(y1 - brick_height)
        bbox = [
            round(float(cx - brick_width / 2.0), 3),
            round(float(y0), 3),
            round(float(cx + brick_width / 2.0), 3),
            round(float(y1), 3),
        ]
        brick_bboxes.append(bbox)
        brick_centers.append((float(cx), float((y0 + y1) / 2.0)))
        draw.rounded_rectangle(
            bbox,
            radius=5,
            fill=tuple(int(v) for v in candidate.brick_fill_rgb),
            outline=tuple(int(v) for v in candidate.brick_outline_rgb),
            width=3,
        )
        top_highlight_y = float(y0 + 6)
        draw.line(
            [(bbox[0] + 9, top_highlight_y), (bbox[2] - 9, top_highlight_y)],
            fill=_lighten(candidate.brick_fill_rgb, 34),
            width=2,
        )
        mid_x = float((bbox[0] + bbox[2]) / 2.0)
        draw.line(
            [(mid_x, bbox[1] + 7), (mid_x, bbox[3] - 7)],
            fill=tuple(int(v) for v in candidate.brick_outline_rgb),
            width=1,
        )

    stack_bbox = expand_bbox(bbox_union(*brick_bboxes), 3.0)
    bottom_bbox = brick_bboxes[0]
    support_y = float(ground_y + 12.0)
    support_bbox = [
        round(float(bottom_bbox[0]), 3),
        round(float(support_y - 8.0), 3),
        round(float(bottom_bbox[2]), 3),
        round(float(support_y + 8.0), 3),
    ]
    support_width = int(render_defaults["support_width_px"])
    draw.line(
        [(bottom_bbox[0], support_y), (bottom_bbox[2], support_y)],
        fill=support_rgb,
        width=support_width,
    )
    draw.line(
        [(bottom_bbox[0], support_y - 10), (bottom_bbox[0], support_y + 10)],
        fill=support_rgb,
        width=support_width,
    )
    draw.line(
        [(bottom_bbox[2], support_y - 10), (bottom_bbox[2], support_y + 10)],
        fill=support_rgb,
        width=support_width,
    )

    com_x = float(sum(point[0] for point in brick_centers) / len(brick_centers))
    com_y = float(sum(point[1] for point in brick_centers) / len(brick_centers))
    projection_y = float(support_y)
    projection_width = int(render_defaults["projection_width_px"])
    _draw_dashed_vertical_line(
        draw,
        x=com_x,
        y0=com_y,
        y1=projection_y,
        fill=projection_rgb,
        width=projection_width,
    )
    com_radius = int(render_defaults["com_radius_px"])
    com_bbox = bbox_from_center((com_x, com_y), com_radius, com_radius)
    draw.ellipse(com_bbox, fill=com_fill, outline=com_outline, width=3)
    draw.line(
        [(com_x - com_radius + 3, com_y), (com_x + com_radius - 3, com_y)],
        fill=(255, 255, 255),
        width=2,
    )
    draw.line(
        [(com_x, com_y - com_radius + 3), (com_x, com_y + com_radius - 3)],
        fill=(255, 255, 255),
        width=2,
    )
    draw_centered_text(
        draw,
        text="COM",
        center=(min(cell_right - 35.0, max(cell_left + 35.0, com_x + 35.0)), com_y - 18.0),
        font=small_font,
        fill=com_fill,
        stroke_fill=resolve_text_stroke_fill(com_fill),
        stroke_width=1,
    )
    projection_point_bbox = bbox_from_center((com_x, projection_y), 5.0, 5.0)
    draw.ellipse(projection_point_bbox, fill=projection_rgb, outline=support_rgb, width=2)

    projection_bbox = [
        round(float(com_x - projection_width - 5), 3),
        round(float(min(com_y, projection_y) - 5), 3),
        round(float(com_x + projection_width + 5), 3),
        round(float(max(com_y, projection_y) + 5), 3),
    ]
    com_offset_units = float(
        (com_x - float((bottom_bbox[0] + bottom_bbox[2]) / 2.0)) / brick_width
    )
    return RenderedStack(
        label=str(candidate.label),
        status=str(candidate.status),
        tip_direction=candidate.tip_direction,
        brick_bboxes_px=tuple(list(bbox) for bbox in brick_bboxes),
        stack_bbox_px=list(stack_bbox),
        support_bbox_px=list(support_bbox),
        center_of_mass_point_px=[round(float(com_x), 3), round(float(com_y), 3)],
        center_of_mass_bbox_px=list(com_bbox),
        projection_point_px=[round(float(com_x), 3), round(float(projection_y), 3)],
        projection_bbox_px=list(projection_bbox),
        support_left_px=float(bottom_bbox[0]),
        support_right_px=float(bottom_bbox[2]),
        com_offset_units=round(float(com_offset_units), 4),
    )


def render_stack_scene(
    *,
    image: Image.Image,
    spec: StackSceneSpec,
    render_defaults: Mapping[str, Any],
    font_family: str,
    style: Any,
    instance_seed: int,
) -> RenderedStackScene:
    """Render the full six-option board and project selected visual witnesses."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    title_font = load_font(
        int(render_defaults["title_font_size_px"]),
        bold=True,
        font_family=font_family,
    )
    label_rgb = tuple(int(v) for v in style.label_rgb)
    board_left = float(render_defaults["board_left_px"])
    board_top = float(render_defaults["board_top_px"])
    board_right = float(width - int(render_defaults["board_right_margin_px"]))
    board_bottom = float(height - int(render_defaults["board_bottom_margin_px"]))
    board_bbox = [board_left, board_top, board_right, board_bottom]
    draw.rounded_rectangle(
        board_bbox,
        radius=20,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    draw_centered_text(
        draw,
        text="center-of-mass stability checks",
        center=((board_left + board_right) / 2.0, board_top + 26.0),
        font=title_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )

    gap_x = float(render_defaults["cell_gap_x_px"])
    gap_y = float(render_defaults["cell_gap_y_px"])
    inner_left = board_left + 24.0
    inner_right = board_right - 24.0
    inner_top = board_top + 58.0
    inner_bottom = board_bottom - 24.0
    cell_width = float((inner_right - inner_left - 2.0 * gap_x) / 3.0)
    cell_height = float((inner_bottom - inner_top - gap_y) / 2.0)
    rendered_stacks: dict[str, RenderedStack] = {}
    entities: list[dict[str, Any]] = []
    for index, candidate in enumerate(spec.candidates):
        row = int(index // 3)
        col = int(index % 3)
        cell_left = inner_left + col * (cell_width + gap_x)
        cell_top = inner_top + row * (cell_height + gap_y)
        cell_bbox = [
            round(float(cell_left), 3),
            round(float(cell_top), 3),
            round(float(cell_left + cell_width), 3),
            round(float(cell_top + cell_height), 3),
        ]
        rendered = _render_candidate(
            draw=draw,
            candidate=candidate,
            cell_bbox=cell_bbox,
            render_defaults=render_defaults,
            style=style,
            font_family=str(font_family),
            instance_seed=int(instance_seed),
        )
        rendered_stacks[str(candidate.label)] = rendered
        entities.append(
            {
                "entity_id": f"stack_{candidate.label}",
                "entity_type": "brick_stack_option",
                "bbox_px": list(rendered.stack_bbox_px),
                "meta": {
                    "option_letter": str(candidate.label),
                    "status": str(candidate.status),
                    "tip_direction": candidate.tip_direction,
                    "is_correct": str(candidate.label) == str(spec.correct_option_letter),
                    "row_offsets": [
                        round(float(value), 4) for value in candidate.row_offsets
                    ],
                    "com_offset_units": float(rendered.com_offset_units),
                    "support_left_px": round(float(rendered.support_left_px), 3),
                    "support_right_px": round(float(rendered.support_right_px), 3),
                    "com_x_px": rendered.center_of_mass_point_px[0],
                },
            }
        )

    selected = rendered_stacks[str(spec.correct_option_letter)]
    selected_witness_bboxes = {
        "center_of_mass": list(selected.center_of_mass_bbox_px),
        "projection": list(selected.projection_bbox_px),
        "support_footprint": list(selected.support_bbox_px),
    }
    annotation_bbox = expand_bbox(
        bbox_union(
            selected.stack_bbox_px,
            selected.center_of_mass_bbox_px,
            selected.projection_bbox_px,
            selected.support_bbox_px,
        ),
        4.0,
    )
    render_map = {
        "target_status": str(spec.target_status),
        "correct_option_letter": str(spec.correct_option_letter),
        "candidate_statuses": {
            str(label): str(rendered.status)
            for label, rendered in sorted(rendered_stacks.items())
        },
        "candidate_tip_directions": {
            str(label): rendered.tip_direction
            for label, rendered in sorted(rendered_stacks.items())
        },
        "candidate_com_points_px": {
            str(label): list(rendered.center_of_mass_point_px)
            for label, rendered in sorted(rendered_stacks.items())
        },
        "candidate_projection_points_px": {
            str(label): list(rendered.projection_point_px)
            for label, rendered in sorted(rendered_stacks.items())
        },
        "candidate_support_bboxes_px": {
            str(label): list(rendered.support_bbox_px)
            for label, rendered in sorted(rendered_stacks.items())
        },
        "candidate_stack_bboxes_px": {
            str(label): list(rendered.stack_bbox_px)
            for label, rendered in sorted(rendered_stacks.items())
        },
        "annotation_bbox_px": list(annotation_bbox),
        "selected_witness_bboxes_px": dict(selected_witness_bboxes),
    }
    return RenderedStackScene(
        image=image,
        annotation_bbox_px=list(annotation_bbox),
        scene_entities=[dict(entity) for entity in entities],
        render_map=dict(render_map),
    )


__all__ = ["render_stack_scene", "resolve_stack_render_defaults"]
