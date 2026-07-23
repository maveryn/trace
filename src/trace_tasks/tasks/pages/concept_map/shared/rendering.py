"""Rendering helpers for concept-map scene packages."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.pages.shared.page_semantic_assets import render_page_semantic_asset_rgba
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_dashed_line, draw_rounded_rect
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font

from .defaults import MARKERS, PALETTES, POST_IMAGE_BACKGROUND_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, RENDERING_DEFAULTS
from .sampling import _bbox_center
from .state import ConceptMapCase, RenderedConceptMap


def _line_bbox(a: tuple[float, float], b: tuple[float, float], pad: float) -> list[float]:
    return [
        round(min(float(a[0]), float(b[0])) - float(pad), 3),
        round(min(float(a[1]), float(b[1])) - float(pad), 3),
        round(max(float(a[0]), float(b[0])) + float(pad), 3),
        round(max(float(a[1]), float(b[1])) + float(pad), 3),
    ]


def _round_bbox(bbox: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in bbox]


def _edge_point_toward(source_bbox: Sequence[float], target_bbox: Sequence[float]) -> tuple[float, float]:
    """Return the point where the center-to-center ray exits the source bbox."""

    sx, sy = _bbox_center(source_bbox)
    tx, ty = _bbox_center(target_bbox)
    dx = float(tx - sx)
    dy = float(ty - sy)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return float(sx), float(sy)

    half_w = max(1.0, float(source_bbox[2] - source_bbox[0]) * 0.5)
    half_h = max(1.0, float(source_bbox[3] - source_bbox[1]) * 0.5)
    x_scale = half_w / abs(dx) if abs(dx) >= 1e-6 else float("inf")
    y_scale = half_h / abs(dy) if abs(dy) >= 1e-6 else float("inf")
    scale = min(float(x_scale), float(y_scale))
    return float(sx + dx * scale), float(sy + dy * scale)


def _connector_points(source_bbox: Sequence[float], target_bbox: Sequence[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return edge-to-edge connector endpoints between two node bboxes."""

    return _edge_point_toward(source_bbox, target_bbox), _edge_point_toward(target_bbox, source_bbox)


def _draw_node(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    node: Mapping[str, Any],
    fill: Sequence[int],
    outline: Sequence[int],
    text_fill: Sequence[int],
    font_size: int,
    radius: int,
    border_width: int,
    bold: bool,
    marker: Mapping[str, Any] | None = None,
) -> list[float]:
    """Draw one finalized node box, preserving the sampled shape and label fit."""

    bbox = [float(value) for value in node["bbox"]]
    shape = str(node.get("shape", "rounded_rect"))
    if shape in {"circle", "ellipse"}:
        draw.ellipse(tuple(bbox), fill=tuple(fill), outline=tuple(outline), width=int(border_width))
    else:
        draw_rounded_rect(
            draw,
            tuple(bbox),
            radius=int((bbox[3] - bbox[1]) / 2.0) if shape == "pill" else int(radius),
            fill=fill,
            outline=outline,
            width=int(border_width),
        )
    if marker is not None:
        mx0 = float(bbox[0] + 8.0)
        my0 = float(bbox[1] + 0.5 * (bbox[3] - bbox[1]) - 5.5)
        mx1 = float(mx0 + 11.0)
        my1 = float(my0 + 11.0)
        marker_img = render_page_semantic_asset_rgba(
            str(marker["marker_id"]),
            size_px=(max(1, int(round(mx1 - mx0))), max(1, int(round(my1 - my0)))),
            tint_rgb=tuple(int(value) for value in text_fill),
        )
        image.paste(marker_img, (int(round(mx0)), int(round(my0))), marker_img)
    inset = 25.0 if marker is not None else 10.0
    shape_width_factor = 0.90 if shape == "circle" else 0.84 if shape == "ellipse" else 1.0
    font = fit_font_to_box(
        draw,
        text=str(node["label"]),
        max_width=max(12.0, float(bbox[2] - bbox[0] - inset - 8.0) * shape_width_factor),
        max_height=max(12.0, float(bbox[3] - bbox[1] - 8.0)),
        bold=bool(bold),
        min_size_px=8,
        max_size_px=int(font_size),
        fill_ratio=0.94,
    )
    draw_centered_text(
        draw,
        text=str(node["label"]),
        center=(float((bbox[0] + bbox[2] + inset) / 2.0), float((bbox[1] + bbox[3]) / 2.0)),
        font=font,
        fill=text_fill,
        stroke_fill=fill,
        stroke_width=0,
    )
    return _round_bbox(bbox)


def render_concept_map_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    case: ConceptMapCase,
) -> RenderedConceptMap:
    """Render a resolved concept map and post-image effects."""

    background, background_meta = make_background_canvas(
        canvas_width=int(case.scene["canvas_width"]),
        canvas_height=int(case.scene["canvas_height"]),
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
    )
    rendered, render_map = _render_scene(base_image=background, scene=case.scene)
    image, post_noise_meta = apply_post_image_noise(
        rendered,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedConceptMap(
        image=image,
        render_map=dict(render_map),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def _render_scene(
    *,
    base_image: Image.Image,
    scene: Mapping[str, Any],
) -> tuple[Image.Image, Dict[str, Any]]:
    """Draw the full concept map and collect node/connector boxes."""

    image = base_image.copy()
    draw = ImageDraw.Draw(image)
    palette = PALETTES[str(scene["style_variant"])]
    panel = [float(value) for value in scene["panel_bbox"]]
    title_bbox = [float(value) for value in scene["title_bbox"]]
    panel_radius = int(group_default(RENDERING_DEFAULTS, "panel_corner_radius_px", 24))
    node_radius = int(group_default(RENDERING_DEFAULTS, "node_corner_radius_px", 14))
    border_width = int(group_default(RENDERING_DEFAULTS, "node_border_width_px", 2))
    connector_width = int(group_default(RENDERING_DEFAULTS, "connector_width_px", 3))
    draw_rounded_rect(
        draw,
        tuple(panel),
        radius=int(panel_radius),
        fill=palette["panel_fill"],
        outline=palette["panel_outline"],
        width=2,
    )
    title_font = load_font(int(group_default(RENDERING_DEFAULTS, "title_font_size_px", 30)), bold=True)
    draw_centered_text(
        draw,
        text=str(scene["scene_title"]),
        center=_bbox_center(title_bbox),
        font=title_font,
        fill=palette["node_text"],
        stroke_fill=palette["panel_fill"],
        stroke_width=0,
    )

    node_bboxes: Dict[str, list[float]] = {}
    connector_bboxes: Dict[str, list[float]] = {}
    marker_by_id = {str(marker["marker_id"]): marker for marker in MARKERS}
    branch_fills = list(palette["branch_fills"])
    central = scene["branches"][0]["central"]
    central_bbox = [float(value) for value in central["bbox"]]

    # Connector layer first, so unrelated nodes/cells always cover crossing connector paths.
    for branch_index, branch in enumerate(scene["branches"]):
        branch_bbox = [float(value) for value in branch["bbox"]]
        start, end = _connector_points(central_bbox, branch_bbox)
        connector_id = f"central_to_{branch['branch_id']}"
        if int(branch_index) % 3 == 1:
            draw_dashed_line(
                draw,
                start=start,
                end=end,
                fill=palette["connector"],
                width=int(connector_width),
                dash_px=14,
                gap_px=7,
            )
        else:
            draw.line([start, end], fill=tuple(palette["connector"]), width=int(connector_width))
        connector_bboxes[connector_id] = _line_bbox(start, end, connector_width + 3)
        for child in branch["children"]:
            child_bbox = [float(value) for value in child["bbox"]]
            child_start, child_end = _connector_points(branch_bbox, child_bbox)
            connector_id = f"{branch['branch_id']}_to_{child['node_id']}"
            draw.line([child_start, child_end], fill=tuple(palette["connector"]), width=max(1, int(connector_width - 1)))
            connector_bboxes[connector_id] = _line_bbox(child_start, child_end, connector_width + 2)

    for branch_index, branch in enumerate(scene["branches"]):
        branch_color = branch_fills[int(branch_index) % len(branch_fills)]
        node_bboxes[str(branch["branch_id"])] = _draw_node(
            image,
            draw,
            node=branch,
            fill=branch_color,
            outline=palette["connector"],
            text_fill=palette["node_text"],
            font_size=int(group_default(RENDERING_DEFAULTS, "branch_font_size_px", 16)),
            radius=int(node_radius),
            border_width=int(border_width),
            bold=True,
        )
        for child in branch["children"]:
            child_fill = tuple(min(255, int(channel) + 28) for channel in branch_color)
            node_bboxes[str(child["node_id"])] = _draw_node(
                image,
                draw,
                node=child,
                fill=child_fill,
                outline=palette["connector"],
                text_fill=palette["node_text"],
                font_size=int(group_default(RENDERING_DEFAULTS, "child_font_size_px", 13)),
                radius=int(max(8, node_radius - 3)),
                border_width=1,
                bold=False,
                marker=marker_by_id[str(child["marker_id"])],
            )
    node_bboxes["central"] = _draw_node(
        image,
        draw,
        node=central,
        fill=palette["central_fill"],
        outline=palette["central_outline"],
        text_fill=palette["central_text"],
        font_size=int(group_default(RENDERING_DEFAULTS, "central_font_size_px", 20)),
        radius=int(node_radius + 5),
        border_width=2,
        bold=True,
    )
    return image, {
        "node_bboxes_px": dict(node_bboxes),
        "connector_bboxes_px": dict(connector_bboxes),
    }
