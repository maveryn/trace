"""Rendering helpers for graph adjacency representations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.drawing import draw_rounded_rect
from ....shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from ....shared.text_legibility import (
    ReadableTextStyle,
    draw_centered_readable_text,
    draw_readable_text,
    resolve_readable_text_style,
    text_legibility_summary_from_records,
)
from ....shared.text_rendering import fit_font_to_box, load_font
from .state import (
    AdjacencyGraphSample,
    AdjacencyRepresentationRender,
    canonical_undirected_edge,
    matrix_cell_key,
)


def _draw_text_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    bbox: Sequence[float],
    font_size_px: int,
    style: ReadableTextStyle,
    records: List[Dict[str, Any]],
    font_family: str,
    bold: bool = True,
    role_context: Mapping[str, Any] | None = None,
) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(x1 - x0 - 4)),
        max_height=max(1.0, float(y1 - y0 - 4)),
        bold=bool(bold),
        font_family=str(font_family or ""),
        min_size_px=8,
        max_size_px=int(font_size_px),
        fill_ratio=0.90,
    )
    record = draw_centered_readable_text(
        draw,
        text=str(text),
        center=(0.5 * (x0 + x1), 0.5 * (y0 + y1)),
        font=font,
        style=style,
        stroke_width=1,
        extra_metadata=dict(role_context or {}),
    )
    records.append(dict(record))
    return [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]


def _adjacency_panel_style(layout_seed: int, *, namespace: str) -> Dict[str, Any]:
    """Return one deterministic table/list visual style."""

    rng = spawn_rng(int(layout_seed), f"graph.adjacency.{str(namespace)}.panel_style")
    style_ids = ("clean_card", "cool_sheet", "warm_ledger", "mint_index", "ink_header")
    style_id = style_ids[int(rng.randrange(len(style_ids)))]
    styles: Dict[str, Dict[str, Tuple[int, int, int]]] = {
        "clean_card": {
            "panel_fill": (255, 255, 255),
            "panel_border": (197, 207, 222),
            "grid": (205, 214, 228),
            "row_fill": (248, 250, 253),
            "header_fill": (229, 236, 248),
            "neighbor_fill": (239, 244, 252),
            "accent_fill": (25, 62, 150),
        },
        "cool_sheet": {
            "panel_fill": (248, 251, 255),
            "panel_border": (169, 190, 218),
            "grid": (192, 207, 228),
            "row_fill": (240, 246, 254),
            "header_fill": (219, 232, 250),
            "neighbor_fill": (231, 241, 253),
            "accent_fill": (24, 76, 122),
        },
        "warm_ledger": {
            "panel_fill": (255, 252, 245),
            "panel_border": (210, 192, 165),
            "grid": (220, 204, 180),
            "row_fill": (252, 246, 236),
            "header_fill": (244, 231, 207),
            "neighbor_fill": (250, 239, 220),
            "accent_fill": (76, 49, 28),
        },
        "mint_index": {
            "panel_fill": (249, 253, 250),
            "panel_border": (177, 206, 190),
            "grid": (199, 222, 210),
            "row_fill": (241, 249, 244),
            "header_fill": (224, 240, 231),
            "neighbor_fill": (235, 247, 240),
            "accent_fill": (20, 76, 53),
        },
        "ink_header": {
            "panel_fill": (250, 251, 253),
            "panel_border": (178, 185, 197),
            "grid": (199, 205, 216),
            "row_fill": (243, 245, 249),
            "header_fill": (229, 232, 239),
            "neighbor_fill": (238, 242, 248),
            "accent_fill": (49, 61, 82),
        },
    }
    resolved: Dict[str, Any] = dict(styles[str(style_id)])
    resolved["panel_style_id"] = str(style_id)
    return resolved


def _resolve_adjacency_font(*, layout_seed: int, namespace: str, font_family: str | None = None) -> Tuple[str, Dict[str, Any]]:
    family = str(font_family or "").strip()
    if not family:
        family = sample_font_family(
            role="readout",
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.font_family",
            params={},
        )
    record = get_font_family_record(str(family))
    return str(family), dict(record.to_trace())


def _adjacency_text_styles(
    *,
    layout_seed: int,
    namespace: str,
    style: Mapping[str, Sequence[int]],
) -> Dict[str, ReadableTextStyle]:
    """Resolve non-semantic text styles for one adjacency representation panel."""

    panel_fill = tuple(int(v) for v in style["panel_fill"])
    header_fill = tuple(int(v) for v in style["header_fill"])
    row_fill = tuple(int(v) for v in style["row_fill"])
    neighbor_fill = tuple(int(v) for v in style["neighbor_fill"])
    accent_fill = tuple(int(v) for v in style["accent_fill"])
    return {
        "title": resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.title_text",
            role="graph_panel_title_text",
            surface_rgbs=(panel_fill,),
            preferred_rgbs=((45, 55, 72), (24, 31, 44)),
            min_contrast_ratio=4.5,
            min_lab_distance=28.0,
        ),
        "subtitle": resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.subtitle_text",
            role="graph_panel_subtitle_text",
            surface_rgbs=(panel_fill,),
            preferred_rgbs=((80, 88, 103), (45, 55, 72)),
            min_contrast_ratio=4.5,
            min_lab_distance=28.0,
        ),
        "row_label": resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.row_label_text",
            role="adjacency_row_label_text",
            surface_rgbs=(accent_fill,),
            preferred_rgbs=((255, 255, 255), (245, 248, 255), (10, 14, 22)),
            min_contrast_ratio=7.0,
            min_lab_distance=38.0,
        ),
        "header_label": resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.header_label_text",
            role="adjacency_header_label_text",
            surface_rgbs=(header_fill,),
            preferred_rgbs=((36, 43, 57), (24, 31, 44)),
            min_contrast_ratio=7.0,
            min_lab_distance=38.0,
        ),
        "cell_value": resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.cell_value_text",
            role="adjacency_cell_value_text",
            surface_rgbs=(panel_fill, row_fill, neighbor_fill, (255, 255, 255)),
            preferred_rgbs=((36, 43, 57), (24, 31, 44)),
            min_contrast_ratio=7.0,
            min_lab_distance=38.0,
        ),
        "muted_cell": resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.muted_cell_value_text",
            role="adjacency_cell_value_text",
            surface_rgbs=(panel_fill, (255, 255, 255)),
            preferred_rgbs=((74, 82, 96), (54, 54, 52)),
            min_contrast_ratio=7.0,
            min_lab_distance=38.0,
        ),
        "context": resolve_readable_text_style(
            instance_seed=int(layout_seed),
            namespace=f"graph.adjacency.{str(namespace)}.context_text",
            role="non_answer_context_text",
            surface_rgbs=(panel_fill,),
            preferred_rgbs=((90, 98, 112), (80, 88, 103)),
            min_contrast_ratio=3.0,
            min_lab_distance=18.0,
            required=False,
        ),
    }


def _draw_header_context(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: Sequence[float],
    layout_seed: int,
    namespace: str,
    font_family: str,
    style: Mapping[str, Sequence[int]],
    text_style: ReadableTextStyle,
    records: List[Dict[str, Any]],
    probability: float,
) -> List[Dict[str, Any]]:
    """Draw optional non-answer header text chips for document-like context."""

    rng = spawn_rng(int(layout_seed), f"graph.adjacency.{str(namespace)}.context")
    if float(probability) <= 0.0 or rng.random() > min(1.0, max(0.0, float(probability))):
        return []
    chip_phrases = (
        "draft table",
        "node index",
        "scan rows",
        "network note",
        "panel ref",
        "working copy",
    )
    x0, y0, x1, _ = [float(value) for value in panel_bbox]
    chip_count = 1 + int(rng.random() < 0.35)
    font = load_font(11, bold=True, font_family=str(font_family or ""))
    elements: List[Dict[str, Any]] = []
    cursor_right = float(x1 - 24.0)
    for _index in range(chip_count):
        text = str(rng.choice(chip_phrases))
        try:
            raw = draw.textbbox((0, 0), text, font=font, stroke_width=1)
            text_w = float(raw[2] - raw[0])
            text_h = float(raw[3] - raw[1])
        except Exception:
            text_w = float(8 * len(text))
            text_h = 12.0
        pad_x = 8.0
        pad_y = 5.0
        chip_w = float(text_w + (2.0 * pad_x))
        chip_h = float(text_h + (2.0 * pad_y))
        left = float(cursor_right - chip_w)
        if left < x0 + 420.0:
            break
        top = float(y0 + 18.0 + (rng.randint(0, 1) * 20.0))
        chip_bbox = (left, top, left + chip_w, top + chip_h)
        draw_rounded_rect(
            draw,
            chip_bbox,
            radius=7,
            fill=tuple(int(v) for v in style["panel_fill"]),
            outline=tuple(int(v) for v in style["panel_border"]),
            width=1,
        )
        record = draw_readable_text(
            draw,
            xy=(left + pad_x, top + pad_y - 1.0),
            text=text,
            font=font,
            style=text_style,
            stroke_width=1,
            extra_metadata={"answer_excluded": True, "kind": "adjacency_context_chip"},
        )
        records.append(dict(record))
        elements.append(
            {
                "role": "non_answer_context_text",
                "kind": "adjacency_context_chip",
                "text": text,
                "bbox_xyxy": [round(float(v), 3) for v in chip_bbox],
                "font_family": str(font_family or ""),
            }
        )
        cursor_right = float(left - 8.0)
    return elements


def render_adjacency_list_panel(
    *,
    sample: AdjacencyGraphSample,
    base_image: Image.Image,
    title: str,
    subtitle: str,
    font_size_px: int = 20,
    layout_seed: int = 0,
    font_family: str | None = None,
    context_text_probability: float = 0.35,
) -> AdjacencyRepresentationRender:
    """Render an adjacency-list panel and return row anchors for task annotations."""

    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    style = _adjacency_panel_style(int(layout_seed), namespace="list")
    resolved_font_family, font_asset = _resolve_adjacency_font(
        layout_seed=int(layout_seed),
        namespace="list",
        font_family=font_family,
    )
    text_styles = _adjacency_text_styles(layout_seed=int(layout_seed), namespace="list", style=style)
    text_records: List[Dict[str, Any]] = []
    width, height = image.size
    panel_bbox = [36.0, 34.0, float(width - 36), float(height - 34)]
    draw_rounded_rect(draw, tuple(panel_bbox), radius=20, fill=style["panel_fill"], outline=style["panel_border"], width=2)
    context_elements = _draw_header_context(
        draw,
        panel_bbox=panel_bbox,
        layout_seed=int(layout_seed),
        namespace="list",
        font_family=str(resolved_font_family),
        style=style,
        text_style=text_styles["context"],
        records=text_records,
        probability=float(context_text_probability),
    )
    title_font = load_font(24, bold=True, font_family=str(resolved_font_family))
    body_font = load_font(16, bold=False, font_family=str(resolved_font_family))
    text_records.append(draw_readable_text(
        draw,
        xy=(62, 52),
        text=str(title),
        font=title_font,
        style=text_styles["title"],
        stroke_width=1,
    ))
    text_records.append(draw_readable_text(
        draw,
        xy=(62, 84),
        text=str(subtitle),
        font=body_font,
        style=text_styles["subtitle"],
        stroke_width=1,
    ))

    labels = tuple(str(label) for label in sample.labels)
    row_h = min(56, max(40, int((height - 140) / max(1, len(labels)))))
    y0 = 120
    x_label0 = 70
    label_w = 88
    x_neighbors0 = x_label0 + label_w + 38
    node_bboxes: Dict[str, List[float]] = {}
    row_bboxes: Dict[str, List[float]] = {}
    for row_index, label in enumerate(labels):
        top = float(y0 + (row_index * row_h))
        bottom = float(top + row_h - 8)
        row_bbox = [float(x_label0 - 14), float(top), float(width - 70), float(bottom)]
        row_bboxes[str(label)] = [round(value, 3) for value in row_bbox]
        draw_rounded_rect(draw, tuple(row_bbox), radius=10, fill=style["row_fill"], outline=style["grid"], width=1)
        label_bbox = [float(x_label0), float(top + 6), float(x_label0 + label_w), float(bottom - 6)]
        draw_rounded_rect(draw, tuple(label_bbox), radius=14, fill=style["accent_fill"], outline=(35, 74, 166), width=1)
        node_bboxes[str(label)] = _draw_text_bbox(
            draw,
            text=str(label),
            bbox=label_bbox,
            font_size_px=int(font_size_px),
            style=text_styles["row_label"],
            records=text_records,
            font_family=str(resolved_font_family),
            role_context={"label": str(label), "kind": "row_label"},
        )
        colon_font = load_font(int(font_size_px), bold=True, font_family=str(resolved_font_family))
        text_records.append(draw_centered_readable_text(
            draw,
            text=":",
            center=(float(x_label0 + label_w + 18), 0.5 * (label_bbox[1] + label_bbox[3])),
            font=colon_font,
            style=text_styles["context"],
            stroke_width=0,
            extra_metadata={"kind": "adjacency_separator", "answer_excluded": True},
        ))
        cursor = float(x_neighbors0)
        targets = tuple(str(target) for target in sample.adjacency.get(str(label), ()))
        if not targets:
            none_bbox = [cursor, float(top + 6), cursor + 72.0, float(bottom - 6)]
            _draw_text_bbox(
                draw,
                text="none",
                bbox=none_bbox,
                font_size_px=max(14, int(font_size_px) - 2),
                style=text_styles["muted_cell"],
                records=text_records,
                font_family=str(resolved_font_family),
                bold=False,
                role_context={"source_label": str(label), "kind": "empty_neighbor_list"},
            )
            continue
        for target in targets:
            chip_w = max(48.0, min(96.0, 22.0 + (12.0 * len(str(target)))))
            chip_bbox = [cursor, float(top + 6), cursor + chip_w, float(bottom - 6)]
            draw_rounded_rect(draw, tuple(chip_bbox), radius=12, fill=style["neighbor_fill"], outline=style["panel_border"], width=1)
            _draw_text_bbox(
                draw,
                text=str(target),
                bbox=chip_bbox,
                font_size_px=max(13, int(font_size_px) - 2),
                style=text_styles["cell_value"],
                records=text_records,
                font_family=str(resolved_font_family),
                bold=True,
                role_context={"source_label": str(label), "target_label": str(target), "kind": "neighbor_label"},
            )
            cursor += chip_w + 10.0

    panel_geometry = {
        "canvas_size": [int(width), int(height)],
        "panel_bbox": list(panel_bbox),
        "row_bboxes": dict(row_bboxes),
        "text_legibility": text_legibility_summary_from_records(text_records, drawn_text_record_count=len(text_records)),
        "font_family": str(resolved_font_family),
        "font_asset": dict(font_asset),
        "font_asset_version": str(font_asset_version()),
        "font_exclusion_reason": "readout font pool; no scene-local exclusion",
    }
    if context_elements:
        panel_geometry["context_text_elements"] = [dict(element) for element in context_elements]
    return AdjacencyRepresentationRender(
        image=image,
        representation_variant="adjacency_list_panel",
        panel_bbox=list(panel_bbox),
        node_label_bboxes=dict(node_bboxes),
        row_label_bboxes=dict(node_bboxes),
        column_label_bboxes={},
        cell_bboxes={},
        panel_geometry=panel_geometry,
        style_meta={
            "panel_style": "adjacency_list_card",
            "adjacency_panel_style_id": str(style["panel_style_id"]),
            "font_family": str(resolved_font_family),
            "font_asset": dict(font_asset),
            "font_asset_version": str(font_asset_version()),
            "font_exclusion_reason": "readout font pool; no scene-local exclusion",
        },
    )


def render_adjacency_matrix_panel(
    *,
    sample: AdjacencyGraphSample,
    base_image: Image.Image,
    title: str,
    subtitle: str,
    weighted: bool,
    font_size_px: int = 19,
    layout_seed: int = 0,
    font_family: str | None = None,
    context_text_probability: float = 0.35,
) -> AdjacencyRepresentationRender:
    """Render an adjacency matrix or weighted adjacency matrix."""

    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    style = _adjacency_panel_style(int(layout_seed), namespace="matrix")
    resolved_font_family, font_asset = _resolve_adjacency_font(
        layout_seed=int(layout_seed),
        namespace="matrix",
        font_family=font_family,
    )
    text_styles = _adjacency_text_styles(layout_seed=int(layout_seed), namespace="matrix", style=style)
    text_records: List[Dict[str, Any]] = []
    width, height = image.size
    panel_bbox = [34.0, 30.0, float(width - 34), float(height - 30)]
    draw_rounded_rect(draw, tuple(panel_bbox), radius=20, fill=style["panel_fill"], outline=style["panel_border"], width=2)
    context_elements = _draw_header_context(
        draw,
        panel_bbox=panel_bbox,
        layout_seed=int(layout_seed),
        namespace="matrix",
        font_family=str(resolved_font_family),
        style=style,
        text_style=text_styles["context"],
        records=text_records,
        probability=float(context_text_probability),
    )
    title_font = load_font(24, bold=True, font_family=str(resolved_font_family))
    body_font = load_font(16, bold=False, font_family=str(resolved_font_family))
    text_records.append(draw_readable_text(
        draw,
        xy=(60, 48),
        text=str(title),
        font=title_font,
        style=text_styles["title"],
        stroke_width=1,
    ))
    text_records.append(draw_readable_text(
        draw,
        xy=(60, 80),
        text=str(subtitle),
        font=body_font,
        style=text_styles["subtitle"],
        stroke_width=1,
    ))

    labels = tuple(str(label) for label in sample.labels)
    n = len(labels)
    available_w = float(width - 120)
    available_h = float(height - 145)
    cell = int(max(34, min(62, available_w / float(n + 1), available_h / float(n + 1))))
    grid_w = cell * (n + 1)
    grid_h = cell * (n + 1)
    left = int(round((width - grid_w) / 2))
    top = 120
    row_bboxes: Dict[str, List[float]] = {}
    col_bboxes: Dict[str, List[float]] = {}
    cell_bboxes: Dict[str, List[float]] = {}

    for row in range(n + 1):
        for col in range(n + 1):
            x0 = float(left + (col * cell))
            y0 = float(top + (row * cell))
            bbox = [x0, y0, x0 + cell, y0 + cell]
            fill = style["header_fill"] if row == 0 or col == 0 else (255, 255, 255)
            draw.rectangle(tuple(bbox), fill=fill, outline=style["grid"], width=1)
            if row == 0 and col == 0:
                continue
            if row == 0:
                label = labels[col - 1]
                col_bboxes[str(label)] = [round(value, 3) for value in bbox]
                _draw_text_bbox(
                    draw,
                    text=str(label),
                    bbox=bbox,
                    font_size_px=int(font_size_px),
                    style=text_styles["header_label"],
                    records=text_records,
                    font_family=str(resolved_font_family),
                    role_context={"label": str(label), "kind": "column_header_label"},
                )
                continue
            if col == 0:
                label = labels[row - 1]
                row_bboxes[str(label)] = [round(value, 3) for value in bbox]
                _draw_text_bbox(
                    draw,
                    text=str(label),
                    bbox=bbox,
                    font_size_px=int(font_size_px),
                    style=text_styles["header_label"],
                    records=text_records,
                    font_family=str(resolved_font_family),
                    role_context={"label": str(label), "kind": "row_header_label"},
                )
                continue
            row_label = labels[row - 1]
            col_label = labels[col - 1]
            cell_bboxes[matrix_cell_key(row_label, col_label)] = [round(value, 3) for value in bbox]
            if row_label == col_label:
                text = "-"
                value_style = text_styles["muted_cell"]
            else:
                edge_key = (str(row_label), str(col_label)) if sample.directed else canonical_undirected_edge(row_label, col_label)
                if bool(weighted):
                    text = str(sample.weights.get(edge_key, ""))
                else:
                    text = "1" if edge_key in set(sample.edges) else "0"
                value_style = text_styles["cell_value"] if text not in {"", "0"} else text_styles["muted_cell"]
            if str(text):
                _draw_text_bbox(
                    draw,
                    text=str(text),
                    bbox=bbox,
                    font_size_px=max(13, int(font_size_px)),
                    style=value_style,
                    records=text_records,
                    font_family=str(resolved_font_family),
                    bold=True,
                    role_context={"row_label": str(row_label), "column_label": str(col_label), "kind": "matrix_cell_value"},
                )

    panel_geometry = {
        "canvas_size": [int(width), int(height)],
        "panel_bbox": list(panel_bbox),
        "matrix_origin_px": [int(left), int(top)],
        "cell_size_px": int(cell),
        "text_legibility": text_legibility_summary_from_records(text_records, drawn_text_record_count=len(text_records)),
        "font_family": str(resolved_font_family),
        "font_asset": dict(font_asset),
        "font_asset_version": str(font_asset_version()),
        "font_exclusion_reason": "readout font pool; no scene-local exclusion",
    }
    if context_elements:
        panel_geometry["context_text_elements"] = [dict(element) for element in context_elements]
    return AdjacencyRepresentationRender(
        image=image,
        representation_variant="adjacency_matrix_panel",
        panel_bbox=list(panel_bbox),
        node_label_bboxes=dict(row_bboxes),
        row_label_bboxes=dict(row_bboxes),
        column_label_bboxes=dict(col_bboxes),
        cell_bboxes=dict(cell_bboxes),
        panel_geometry=panel_geometry,
        style_meta={
            "panel_style": "weighted_adjacency_matrix" if bool(weighted) else "adjacency_matrix",
            "adjacency_panel_style_id": str(style["panel_style_id"]),
            "font_family": str(resolved_font_family),
            "font_asset": dict(font_asset),
            "font_asset_version": str(font_asset_version()),
            "font_exclusion_reason": "readout font pool; no scene-local exclusion",
        },
    )


def render_component_adjacency_panel(
    *,
    sample: AdjacencyGraphSample,
    base_image,
    representation_variant: str,
    directed: bool,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    defaults: Any,
    layout_seed: int,
) -> AdjacencyRepresentationRender:
    """Render the shared adjacency panel for a component-count task.

    This helper owns only the scene rendering choice between list and matrix.
    The public task owns directedness selection, answer binding, and annotation
    witness semantics before calling it.
    """

    font_size_px = int(params.get("label_font_size_px", group_default(render_defaults, "label_font_size_px", defaults.label_font_size_px)))
    context_probability = float(params.get("context_text_probability", group_default(render_defaults, "context_text_probability", 0.35)))
    font_family = params.get("font_family")
    if str(representation_variant) == "adjacency_matrix_panel":
        return render_adjacency_matrix_panel(
            sample=sample,
            base_image=base_image,
            title="Directed Adjacency Matrix" if bool(directed) else "Undirected Adjacency Matrix",
            subtitle="Rows point to columns." if bool(directed) else "The matrix is symmetric.",
            weighted=False,
            font_size_px=font_size_px,
            layout_seed=int(layout_seed),
            font_family=font_family,
            context_text_probability=context_probability,
        )
    return render_adjacency_list_panel(
        sample=sample,
        base_image=base_image,
        title="Directed Adjacency List" if bool(directed) else "Undirected Adjacency List",
        subtitle="Rows list outgoing neighbors." if bool(directed) else "Rows list adjacent nodes.",
        font_size_px=font_size_px,
        layout_seed=int(layout_seed),
        font_family=font_family,
        context_text_probability=context_probability,
    )


__all__ = [
    "render_component_adjacency_panel",
    "render_adjacency_list_panel",
    "render_adjacency_matrix_panel",
]
