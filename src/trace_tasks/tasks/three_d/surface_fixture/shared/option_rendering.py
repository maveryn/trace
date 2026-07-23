"""Option-grid rendering for surface-fixture visual MCQ tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.three_d.shared.canvas import (
    bbox_transform,
    entities_transform,
    point_transform,
)

from .rendering import RenderedSurfaceFixture, bbox_union, render_surface_fixture


@dataclass(frozen=True)
class RenderedSurfaceFixtureOptionGrid:
    """Rendered 2x2 option-grid surface fixture scene."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    option_panel_bboxes_px: Dict[str, List[float]]
    option_label_bboxes_px: Dict[str, List[float]]
    option_counts_by_label: Dict[str, int]
    element_bboxes_px: Dict[str, List[float]]
    element_centers_px: Dict[str, List[float]]
    option_records: List[Dict[str, Any]]


@dataclass(frozen=True)
class RenderedSurfaceFixtureRecolorBoardMatch:
    """Rendered original-board plus 2x2 candidate-board recolor scene."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    original_panel_bbox_px: List[float]
    original_label_bbox_px: List[float]
    option_panel_bboxes_px: Dict[str, List[float]]
    option_label_bboxes_px: Dict[str, List[float]]
    original_color_counts: Dict[str, int]
    option_color_counts_by_label: Dict[str, Dict[str, int]]
    element_bboxes_px: Dict[str, List[float]]
    element_centers_px: Dict[str, List[float]]
    option_records: List[Dict[str, Any]]


@dataclass(frozen=True)
class RenderedSurfaceFixtureColorFrequencyOptions:
    """Rendered single fixture panel plus six text options naming colors."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    fixture_panel_bbox_px: List[float]
    option_bboxes_px: Dict[str, List[float]]
    option_label_bboxes_px: Dict[str, List[float]]
    option_text_bboxes_px: Dict[str, List[float]]
    element_bboxes_px: Dict[str, List[float]]
    element_centers_px: Dict[str, List[float]]
    option_records: List[Dict[str, Any]]


def _panel_positions(width: int, height: int) -> Dict[str, List[int]]:
    margin_x = max(42, int(round(float(width) * 0.048)))
    margin_y = max(34, int(round(float(height) * 0.045)))
    gap_x = max(26, int(round(float(width) * 0.030)))
    gap_y = max(26, int(round(float(height) * 0.034)))
    panel_w = int((int(width) - 2 * margin_x - gap_x) // 2)
    panel_h = int((int(height) - 2 * margin_y - gap_y) // 2)
    return {
        "A": [margin_x, margin_y, margin_x + panel_w, margin_y + panel_h],
        "B": [margin_x + panel_w + gap_x, margin_y, margin_x + 2 * panel_w + gap_x, margin_y + panel_h],
        "C": [margin_x, margin_y + panel_h + gap_y, margin_x + panel_w, margin_y + 2 * panel_h + gap_y],
        "D": [
            margin_x + panel_w + gap_x,
            margin_y + panel_h + gap_y,
            margin_x + 2 * panel_w + gap_x,
            margin_y + 2 * panel_h + gap_y,
        ],
    }


def _recolor_board_positions(width: int, height: int) -> Tuple[List[int], Dict[str, List[int]]]:
    margin_x = max(42, int(round(float(width) * 0.048)))
    top_margin = max(26, int(round(float(height) * 0.030)))
    bottom_margin = max(28, int(round(float(height) * 0.032)))
    gap_x = max(24, int(round(float(width) * 0.028)))
    gap_y = max(20, int(round(float(height) * 0.026)))
    original_w = min(int(round(float(width) * 0.50)), int(width - 2 * margin_x))
    original_h = max(220, int(round(float(height) * 0.285)))
    original_x0 = int((int(width) - int(original_w)) // 2)
    original_y0 = int(top_margin)
    original_bbox = [original_x0, original_y0, original_x0 + int(original_w), original_y0 + int(original_h)]
    option_top = int(original_bbox[3] + gap_y)
    available_h = int(height - option_top - bottom_margin - gap_y)
    panel_w = int((int(width) - 2 * margin_x - gap_x) // 2)
    panel_h = int(available_h // 2)
    return original_bbox, {
        "A": [margin_x, option_top, margin_x + panel_w, option_top + panel_h],
        "B": [margin_x + panel_w + gap_x, option_top, margin_x + 2 * panel_w + gap_x, option_top + panel_h],
        "C": [margin_x, option_top + panel_h + gap_y, margin_x + panel_w, option_top + 2 * panel_h + gap_y],
        "D": [
            margin_x + panel_w + gap_x,
            option_top + panel_h + gap_y,
            margin_x + 2 * panel_w + gap_x,
            option_top + 2 * panel_h + gap_y,
        ],
    }


def _color_frequency_positions(width: int, height: int) -> Tuple[List[int], Dict[str, List[int]]]:
    margin_x = max(42, int(round(float(width) * 0.050)))
    margin_y = max(30, int(round(float(height) * 0.040)))
    gap_y = max(24, int(round(float(height) * 0.030)))
    option_gap_x = max(14, int(round(float(width) * 0.018)))
    option_gap_y = max(14, int(round(float(height) * 0.020)))
    option_area_h = max(150, int(round(float(height) * 0.245)))
    panel_bottom = int(height - margin_y - option_area_h - gap_y)
    fixture_bbox = [margin_x, margin_y, int(width - margin_x), int(panel_bottom)]
    option_top = int(panel_bottom + gap_y)
    option_bottom = int(height - margin_y)
    option_w = int((int(width) - 2 * margin_x - 2 * option_gap_x) // 3)
    option_h = int((option_bottom - option_top - option_gap_y) // 2)
    labels = ("A", "B", "C", "D", "E", "F")
    bboxes: Dict[str, List[int]] = {}
    for index, label in enumerate(labels):
        row = int(index // 3)
        col = int(index % 3)
        x0 = int(margin_x + col * (option_w + option_gap_x))
        y0 = int(option_top + row * (option_h + option_gap_y))
        bboxes[str(label)] = [x0, y0, x0 + option_w, y0 + option_h]
    return fixture_bbox, bboxes


def _draw_option_label(draw: ImageDraw.ImageDraw, *, label: str, panel_bbox: Sequence[float]) -> List[float]:
    badge_size = 38
    x0 = float(panel_bbox[0]) + 12.0
    y0 = float(panel_bbox[1]) + 12.0
    badge = [x0, y0, x0 + badge_size, y0 + badge_size]
    draw.rounded_rectangle(tuple(badge), radius=7, fill=(28, 36, 50), outline=(255, 255, 255), width=2)
    font = load_font(24, bold=True)
    text_bbox = draw.textbbox((0, 0), str(label), font=font, stroke_width=0)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    draw.text(
        (
            x0 + badge_size * 0.5 - text_w * 0.5,
            y0 + badge_size * 0.5 - text_h * 0.5 - 1.0,
        ),
        str(label),
        font=font,
        fill=(255, 255, 255),
    )
    return [round(float(value), 3) for value in badge]


def _draw_color_text_option(
    draw: ImageDraw.ImageDraw,
    *,
    label: str,
    color_name: str,
    option_bbox: Sequence[int],
) -> Tuple[List[float], List[float], List[float]]:
    x0, y0, x1, y1 = [int(value) for value in option_bbox]
    bbox = [float(x0), float(y0), float(x1), float(y1)]
    draw.rounded_rectangle(tuple(bbox), radius=8, fill=(248, 250, 251), outline=(72, 84, 96), width=2)
    label_bbox = _draw_option_label(draw, label=str(label), panel_bbox=bbox)
    font = load_font(30, bold=True)
    color_text = str(color_name).replace("_", " ")
    text_bbox = draw.textbbox((0, 0), color_text, font=font, stroke_width=0)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    text_x = float(label_bbox[2]) + max(18.0, float(x1 - x0) * 0.065)
    max_text_x = float(x1) - 16.0 - text_w
    text_x = min(text_x, max_text_x)
    text_y = float(y0 + y1) * 0.5 - text_h * 0.5 - 1.0
    draw.text((text_x, text_y), color_text, font=font, fill=(24, 30, 38))
    option_text_bbox = [
        round(float(text_x), 3),
        round(float(text_y), 3),
        round(float(text_x + text_w), 3),
        round(float(text_y + text_h), 3),
    ]
    return [round(float(value), 3) for value in bbox], list(label_bbox), option_text_bbox


def _draw_panel_badge(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    panel_bbox: Sequence[float],
    fill: Sequence[int] = (28, 36, 50),
) -> List[float]:
    font = load_font(22, bold=True)
    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    x0 = float(panel_bbox[0]) + 12.0
    y0 = float(panel_bbox[1]) + 12.0
    badge = [x0, y0, x0 + max(46.0, text_w + 22.0), y0 + 38.0]
    draw.rounded_rectangle(tuple(badge), radius=7, fill=tuple(int(value) for value in fill), outline=(255, 255, 255), width=2)
    draw.text(
        (x0 + 11.0, y0 + 19.0 - text_h * 0.5 - 1.0),
        str(text),
        font=font,
        fill=(255, 255, 255),
    )
    return [round(float(value), 3) for value in badge]


def _offset_entities(
    rendered: RenderedSurfaceFixture,
    *,
    label: str,
    scale_x: float,
    scale_y: float,
    dx: float,
    dy: float,
) -> List[Dict[str, Any]]:
    entities: List[Dict[str, Any]] = []
    for entity in entities_transform(
        rendered.entities,
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        offset_x=float(dx),
        offset_y=float(dy),
    ):
        updated = dict(entity)
        entity_id = str(updated.get("entity_id", "entity"))
        updated["entity_id"] = f"option_{label}.{entity_id}"
        attrs = dict(updated.get("attrs", {}))
        attrs["option_label"] = str(label)
        updated["attrs"] = attrs
        entities.append(updated)
    return entities


def _render_source_panel_into_bbox(
    image: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: Any,
    panel_bbox: Sequence[int],
) -> Tuple[RenderedSurfaceFixture, float, float, int, int]:
    """Render a canonical fixture panel and downsample it into a composite panel."""

    x0, y0, x1, y1 = [int(value) for value in panel_bbox]
    panel_w = int(x1 - x0)
    panel_h = int(y1 - y0)
    source_background = Image.new(
        "RGB",
        (int(render_params.canvas_width), int(render_params.canvas_height)),
        (246, 249, 250),
    )
    rendered_panel = render_surface_fixture(
        source_background,
        dataset=dict(dataset),
        render_params=render_params,
    )
    resized_panel = rendered_panel.image.resize((int(panel_w), int(panel_h)), Image.Resampling.LANCZOS)
    image.paste(resized_panel, (x0, y0))
    return (
        rendered_panel,
        float(panel_w) / float(render_params.canvas_width),
        float(panel_h) / float(render_params.canvas_height),
        int(x0),
        int(y0),
    )


def render_surface_fixture_option_grid(
    background: Image.Image,
    *,
    option_datasets: Mapping[str, Mapping[str, Any]],
    render_params: Any,
) -> RenderedSurfaceFixtureOptionGrid:
    """Render four labeled surface-fixture options into one 2x2 canvas."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    panel_positions = _panel_positions(int(width), int(height))
    labels = tuple(label for label in ("A", "B", "C", "D") if label in option_datasets)
    if labels != ("A", "B", "C", "D"):
        raise ValueError("surface fixture option grid requires labels A, B, C, D")

    option_panel_bboxes: Dict[str, List[float]] = {}
    option_label_bboxes: Dict[str, List[float]] = {}
    option_counts_by_label: Dict[str, int] = {}
    element_bboxes: Dict[str, List[float]] = {}
    element_centers: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    option_records: List[Dict[str, Any]] = []
    scene_bboxes: List[List[float]] = []

    for label in labels:
        panel_bbox = panel_positions[str(label)]
        x0, y0, x1, y1 = [int(value) for value in panel_bbox]
        panel_w = int(x1 - x0)
        panel_h = int(y1 - y0)
        rendered_panel, scale_x, scale_y, x0, y0 = _render_source_panel_into_bbox(
            image,
            dataset=dict(option_datasets[str(label)]),
            render_params=render_params,
            panel_bbox=panel_bbox,
        )

        panel_bbox_float = [float(x0), float(y0), float(x1), float(y1)]
        draw.rounded_rectangle(tuple(panel_bbox_float), radius=8, outline=(59, 72, 86), width=3)
        draw.rounded_rectangle(
            (panel_bbox_float[0] + 3, panel_bbox_float[1] + 3, panel_bbox_float[2] - 3, panel_bbox_float[3] - 3),
            radius=6,
            outline=(236, 242, 245),
            width=1,
        )
        label_bbox = _draw_option_label(draw, label=str(label), panel_bbox=panel_bbox_float)

        option_panel_bboxes[str(label)] = [round(float(value), 3) for value in panel_bbox_float]
        option_label_bboxes[str(label)] = list(label_bbox)
        option_counts_by_label[str(label)] = int(option_datasets[str(label)]["answer_value"])
        entities.append(
            {
                "entity_id": f"option_{label}",
                "entity_type": "three_d_surface_fixture_option_panel",
                "bbox_px": list(option_panel_bboxes[str(label)]),
                "attrs": {
                    "option_label": str(label),
                    "visible_element_count": int(option_counts_by_label[str(label)]),
                },
            }
        )
        entities.extend(
            _offset_entities(
                rendered_panel,
                label=str(label),
                scale_x=float(scale_x),
                scale_y=float(scale_y),
                dx=float(x0),
                dy=float(y0),
            )
        )

        for element_id, bbox in rendered_panel.element_bboxes_px.items():
            key = f"{label}:{element_id}"
            element_bboxes[key] = bbox_transform(bbox, scale_x=scale_x, scale_y=scale_y, offset_x=x0, offset_y=y0)
        for element_id, center in rendered_panel.element_centers_px.items():
            key = f"{label}:{element_id}"
            element_centers[key] = point_transform(center, scale_x=scale_x, scale_y=scale_y, offset_x=x0, offset_y=y0)

        scene_bbox = bbox_transform(rendered_panel.scene_bbox_px, scale_x=scale_x, scale_y=scale_y, offset_x=x0, offset_y=y0)
        scene_bboxes.append(list(scene_bbox))
        option_records.append(
            {
                "label": str(label),
                "panel_bbox_px": list(option_panel_bboxes[str(label)]),
                "option_label_bbox_px": list(option_label_bboxes[str(label)]),
                "visible_element_count": int(option_counts_by_label[str(label)]),
                "scene_bbox_px": list(scene_bbox),
                "source_canvas_preset": str(getattr(render_params, "canvas_preset", "unknown")),
                "source_canvas_width": int(render_params.canvas_width),
                "source_canvas_height": int(render_params.canvas_height),
                "panel_scale_x": round(float(scale_x), 8),
                "panel_scale_y": round(float(scale_y), 8),
                "element_ids": [f"{label}:{element_id}" for element_id in rendered_panel.element_bboxes_px.keys()],
            }
        )

    return RenderedSurfaceFixtureOptionGrid(
        image=image,
        entities=entities,
        scene_bbox_px=bbox_union(*scene_bboxes),
        option_panel_bboxes_px=dict(option_panel_bboxes),
        option_label_bboxes_px=dict(option_label_bboxes),
        option_counts_by_label=dict(option_counts_by_label),
        element_bboxes_px=dict(element_bboxes),
        element_centers_px=dict(element_centers),
        option_records=list(option_records),
    )


def _paste_rendered_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    dataset: Mapping[str, Any],
    render_params: Any,
    panel_bbox: Sequence[int],
    label: str,
    label_text: str,
    element_prefix: str,
    badge_fill: Sequence[int] = (28, 36, 50),
) -> Tuple[RenderedSurfaceFixture, List[float], List[float], float, float]:
    """Render one fixture panel into a precomputed composite-scene rectangle."""

    x0, y0, x1, y1 = [int(value) for value in panel_bbox]
    panel_w = int(x1 - x0)
    panel_h = int(y1 - y0)
    rendered_panel, scale_x, scale_y, x0, y0 = _render_source_panel_into_bbox(
        image,
        dataset=dict(dataset),
        render_params=render_params,
        panel_bbox=panel_bbox,
    )
    panel_bbox_float = [float(x0), float(y0), float(x1), float(y1)]
    draw.rounded_rectangle(tuple(panel_bbox_float), radius=8, outline=(59, 72, 86), width=3)
    draw.rounded_rectangle(
        (panel_bbox_float[0] + 3, panel_bbox_float[1] + 3, panel_bbox_float[2] - 3, panel_bbox_float[3] - 3),
        radius=6,
        outline=(236, 242, 245),
        width=1,
    )
    if len(str(label_text)) == 1:
        label_bbox = _draw_option_label(draw, label=str(label_text), panel_bbox=panel_bbox_float)
    else:
        label_bbox = _draw_panel_badge(draw, text=str(label_text), panel_bbox=panel_bbox_float, fill=badge_fill)
    return rendered_panel, [round(float(value), 3) for value in panel_bbox_float], list(label_bbox), float(scale_x), float(scale_y)


def render_surface_fixture_recolor_board_match(
    background: Image.Image,
    *,
    original_dataset: Mapping[str, Any],
    option_datasets: Mapping[str, Mapping[str, Any]],
    render_params: Any,
) -> RenderedSurfaceFixtureRecolorBoardMatch:
    """Render one original board over four labeled candidate boards."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    original_position, option_positions = _recolor_board_positions(int(width), int(height))
    labels = tuple(label for label in ("A", "B", "C", "D") if label in option_datasets)
    if labels != ("A", "B", "C", "D"):
        raise ValueError("surface fixture recolor board match requires labels A, B, C, D")

    entities: List[Dict[str, Any]] = []
    element_bboxes: Dict[str, List[float]] = {}
    element_centers: Dict[str, List[float]] = {}
    option_panel_bboxes: Dict[str, List[float]] = {}
    option_label_bboxes: Dict[str, List[float]] = {}
    option_color_counts_by_label: Dict[str, Dict[str, int]] = {}
    option_records: List[Dict[str, Any]] = []
    scene_bboxes: List[List[float]] = []

    original_rendered, original_panel_bbox, original_label_bbox, original_scale_x, original_scale_y = _paste_rendered_panel(
        image,
        draw,
        dataset=original_dataset,
        render_params=render_params,
        panel_bbox=original_position,
        label="original",
        label_text="Original",
        element_prefix="original",
        badge_fill=(82, 94, 108),
    )
    ox0, oy0 = int(original_position[0]), int(original_position[1])
    original_scene_bbox = bbox_transform(
        original_rendered.scene_bbox_px,
        scale_x=original_scale_x,
        scale_y=original_scale_y,
        offset_x=ox0,
        offset_y=oy0,
    )
    scene_bboxes.append(list(original_scene_bbox))
    entities.append(
        {
            "entity_id": "original_board",
            "entity_type": "three_d_surface_fixture_original_board",
            "bbox_px": list(original_panel_bbox),
            "attrs": {"color_counts": dict(original_dataset.get("color_counts", {}))},
        }
    )
    entities.extend(
        _offset_entities(
            original_rendered,
            label="original",
            scale_x=float(original_scale_x),
            scale_y=float(original_scale_y),
            dx=float(ox0),
            dy=float(oy0),
        )
    )
    for element_id, bbox in original_rendered.element_bboxes_px.items():
        element_bboxes[f"original:{element_id}"] = bbox_transform(
            bbox,
            scale_x=original_scale_x,
            scale_y=original_scale_y,
            offset_x=ox0,
            offset_y=oy0,
        )
    for element_id, center in original_rendered.element_centers_px.items():
        element_centers[f"original:{element_id}"] = point_transform(
            center,
            scale_x=original_scale_x,
            scale_y=original_scale_y,
            offset_x=ox0,
            offset_y=oy0,
        )

    for label in labels:
        panel_position = option_positions[str(label)]
        rendered_panel, panel_bbox, label_bbox, scale_x, scale_y = _paste_rendered_panel(
            image,
            draw,
            dataset=dict(option_datasets[str(label)]),
            render_params=render_params,
            panel_bbox=panel_position,
            label=str(label),
            label_text=str(label),
            element_prefix=str(label),
        )
        x0, y0 = int(panel_position[0]), int(panel_position[1])
        option_panel_bboxes[str(label)] = list(panel_bbox)
        option_label_bboxes[str(label)] = list(label_bbox)
        option_counts = {str(color): int(count) for color, count in option_datasets[str(label)].get("color_counts", {}).items()}
        option_color_counts_by_label[str(label)] = dict(option_counts)
        entities.append(
            {
                "entity_id": f"option_{label}",
                "entity_type": "three_d_surface_fixture_recolor_option_panel",
                "bbox_px": list(panel_bbox),
                "attrs": {
                    "option_label": str(label),
                    "color_counts": dict(option_counts),
                },
            }
        )
        entities.extend(
            _offset_entities(
                rendered_panel,
                label=str(label),
                scale_x=float(scale_x),
                scale_y=float(scale_y),
                dx=float(x0),
                dy=float(y0),
            )
        )
        for element_id, bbox in rendered_panel.element_bboxes_px.items():
            element_bboxes[f"{label}:{element_id}"] = bbox_transform(
                bbox,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=x0,
                offset_y=y0,
            )
        for element_id, center in rendered_panel.element_centers_px.items():
            element_centers[f"{label}:{element_id}"] = point_transform(
                center,
                scale_x=scale_x,
                scale_y=scale_y,
                offset_x=x0,
                offset_y=y0,
            )

        scene_bbox = bbox_transform(rendered_panel.scene_bbox_px, scale_x=scale_x, scale_y=scale_y, offset_x=x0, offset_y=y0)
        scene_bboxes.append(list(scene_bbox))
        option_records.append(
            {
                "label": str(label),
                "panel_bbox_px": list(panel_bbox),
                "option_label_bbox_px": list(label_bbox),
                "color_counts": dict(option_counts),
                "scene_bbox_px": list(scene_bbox),
                "source_canvas_preset": str(getattr(render_params, "canvas_preset", "unknown")),
                "source_canvas_width": int(render_params.canvas_width),
                "source_canvas_height": int(render_params.canvas_height),
                "panel_scale_x": round(float(scale_x), 8),
                "panel_scale_y": round(float(scale_y), 8),
                "element_ids": [f"{label}:{element_id}" for element_id in rendered_panel.element_bboxes_px.keys()],
            }
        )

    return RenderedSurfaceFixtureRecolorBoardMatch(
        image=image,
        entities=entities,
        scene_bbox_px=bbox_union(*scene_bboxes),
        original_panel_bbox_px=list(original_panel_bbox),
        original_label_bbox_px=list(original_label_bbox),
        option_panel_bboxes_px=dict(option_panel_bboxes),
        option_label_bboxes_px=dict(option_label_bboxes),
        original_color_counts={str(color): int(count) for color, count in original_dataset.get("color_counts", {}).items()},
        option_color_counts_by_label=dict(option_color_counts_by_label),
        element_bboxes_px=dict(element_bboxes),
        element_centers_px=dict(element_centers),
        option_records=list(option_records),
    )


def render_surface_fixture_color_frequency_options(
    background: Image.Image,
    *,
    dataset: Mapping[str, Any],
    render_params: Any,
) -> RenderedSurfaceFixtureColorFrequencyOptions:
    """Render one colored fixture and six labeled text options naming colors."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    fixture_position, option_positions = _color_frequency_positions(int(width), int(height))
    rendered_panel, fixture_bbox, fixture_label_bbox, scale_x, scale_y = _paste_rendered_panel(
        image,
        draw,
        dataset=dict(dataset),
        render_params=render_params,
        panel_bbox=fixture_position,
        label="fixture",
        label_text="Fixture",
        element_prefix="fixture",
        badge_fill=(82, 94, 108),
    )
    x0, y0 = int(fixture_position[0]), int(fixture_position[1])
    option_records: List[Dict[str, Any]] = []
    option_bboxes: Dict[str, List[float]] = {}
    option_label_bboxes: Dict[str, List[float]] = {}
    option_text_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    element_bboxes: Dict[str, List[float]] = {}
    element_centers: Dict[str, List[float]] = {}

    scene_bbox = bbox_transform(
        rendered_panel.scene_bbox_px,
        scale_x=float(scale_x),
        scale_y=float(scale_y),
        offset_x=float(x0),
        offset_y=float(y0),
    )
    entities.append(
        {
            "entity_id": "fixture_panel",
            "entity_type": "three_d_surface_fixture_color_frequency_panel",
            "bbox_px": list(fixture_bbox),
            "attrs": {
                "option_color_counts": dict(dataset.get("option_color_counts", {})),
            },
        }
    )
    entities.extend(
        _offset_entities(
            rendered_panel,
            label="fixture",
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            dx=float(x0),
            dy=float(y0),
        )
    )
    for element_id, bbox in rendered_panel.element_bboxes_px.items():
        element_bboxes[str(element_id)] = bbox_transform(
            bbox,
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            offset_x=float(x0),
            offset_y=float(y0),
        )
    for element_id, center in rendered_panel.element_centers_px.items():
        element_centers[str(element_id)] = point_transform(
            center,
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            offset_x=float(x0),
            offset_y=float(y0),
        )

    records = [dict(record) for record in dataset.get("option_records", [])]
    if len(records) != 6:
        raise ValueError("color-frequency option renderer requires six option records")
    for record in records:
        label = str(record["label"])
        option_bbox, label_bbox, text_bbox = _draw_color_text_option(
            draw,
            label=str(label),
            color_name=str(record["color_name"]),
            option_bbox=option_positions[str(label)],
        )
        option_bboxes[str(label)] = list(option_bbox)
        option_label_bboxes[str(label)] = list(label_bbox)
        option_text_bboxes[str(label)] = list(text_bbox)
        option_record = dict(record)
        option_record["option_bbox_px"] = list(option_bbox)
        option_record["option_label_bbox_px"] = list(label_bbox)
        option_record["option_text_bbox_px"] = list(text_bbox)
        option_records.append(option_record)
        entities.append(
            {
                "entity_id": f"text_option_{label}",
                "entity_type": "three_d_surface_fixture_text_option",
                "bbox_px": list(option_bbox),
                "attrs": {
                    "option_label": str(label),
                    "color_name": str(record["color_name"]),
                    "option_text": str(record["color_name"]).replace("_", " "),
                    "visible_count": int(record["visible_count"]),
                },
            }
        )
    all_bboxes = [list(scene_bbox), list(fixture_bbox), *[list(bbox) for bbox in option_bboxes.values()]]
    return RenderedSurfaceFixtureColorFrequencyOptions(
        image=image,
        entities=entities,
        scene_bbox_px=bbox_union(*all_bboxes),
        fixture_panel_bbox_px=list(fixture_bbox),
        option_bboxes_px=dict(option_bboxes),
        option_label_bboxes_px=dict(option_label_bboxes),
        option_text_bboxes_px=dict(option_text_bboxes),
        element_bboxes_px=dict(element_bboxes),
        element_centers_px=dict(element_centers),
        option_records=list(option_records),
    )


__all__ = [
    "RenderedSurfaceFixtureColorFrequencyOptions",
    "RenderedSurfaceFixtureOptionGrid",
    "RenderedSurfaceFixtureRecolorBoardMatch",
    "render_surface_fixture_color_frequency_options",
    "render_surface_fixture_option_grid",
    "render_surface_fixture_recolor_board_match",
]
