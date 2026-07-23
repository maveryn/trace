"""Rendering primitives for contour-density chart scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.cartesian.axes import (
    draw_axis_lines,
    draw_horizontal_value_grid_ticks,
    draw_plot_frame,
    draw_vertical_value_grid_ticks,
)
from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_xy
from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.charts.contour_density.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.charts.contour_density.shared.sampling import bbox
from trace_tasks.tasks.charts.contour_density.shared.state import BBox, ContourDataset, Reference, Region, RenderParams, RenderedContourScene, RGB


def _render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", params.get("_sample_cursor", 0)) or 0)
    except Exception:
        return 0


def _resolve_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(
        resolve_render_int(
            params,
            RENDER_DEFAULTS,
            str(key),
            int(fallback),
            instance_seed=_render_style_seed(params),
            namespace=SCENE_NAMESPACE,
        )
    )


def _resolve_rgb(params: Mapping[str, Any], key: str, fallback: RGB) -> RGB:
    return resolve_render_rgb(
        params,
        RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=_render_style_seed(params),
        namespace=SCENE_NAMESPACE,
    )


def resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    params = {**dict(params), "_render_style_seed": int(instance_seed)}
    outer = _resolve_int(params, "outer_margin_px", 48)
    left, right, top, bottom, jitter = apply_layout_jitter_to_margins(
        left_px=_resolve_int(params, "plot_margin_left_px", 92),
        right_px=_resolve_int(params, "plot_margin_right_px", 72),
        top_px=_resolve_int(params, "plot_margin_top_px", 82),
        bottom_px=_resolve_int(params, "plot_margin_bottom_px", 86),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    if outer:
        left = max(int(left), int(outer))
        right = max(int(right), int(outer))
    return RenderParams(
        canvas_width=_resolve_int(params, "canvas_width", 1280),
        canvas_height=_resolve_int(params, "canvas_height", 820),
        margin_left=int(left),
        margin_right=int(right),
        margin_top=int(top),
        margin_bottom=int(bottom),
        grid_line_width=_resolve_int(params, "grid_line_width_px", 1),
        axis_line_width=_resolve_int(params, "axis_line_width_px", 2),
        tick_font_size=_resolve_int(params, "tick_font_size_px", 15),
        label_font_size=_resolve_int(params, "label_font_size_px", 18),
        title_font_size=_resolve_int(params, "title_font_size_px", 26),
        marker_radius=_resolve_int(params, "reference_marker_radius_px", 8),
        text_rgb=_resolve_rgb(params, "text_color_rgb", (35, 40, 52)),
        muted_rgb=_resolve_rgb(params, "muted_text_rgb", (83, 91, 105)),
        axis_rgb=_resolve_rgb(params, "axis_color_rgb", (55, 60, 70)),
        grid_rgb=_resolve_rgb(params, "grid_color_rgb", (223, 227, 235)),
        plot_fill_rgb=_resolve_rgb(params, "plot_fill_rgb", (255, 255, 255)),
        reference_rgb=_resolve_rgb(params, "reference_rgb", (24, 26, 30)),
        text_stroke_rgb=_resolve_rgb(params, "text_stroke_rgb", (255, 255, 255)),
        layout_jitter=dict(jitter),
    )


def plot_bbox(rp: RenderParams) -> BBox:
    return tuple(
        plot_bbox_from_margins(
            canvas_width=float(rp.canvas_width),
            canvas_height=float(rp.canvas_height),
            margin_left_px=float(rp.margin_left),
            margin_right_px=float(rp.margin_right),
            margin_top_px=float(rp.margin_top),
            margin_bottom_px=float(rp.margin_bottom),
        )
    )


def scale_point(x_value: float, y_value: float, *, plot_box: BBox) -> Tuple[float, float]:
    return project_xy(
        x_value=float(x_value),
        y_value=float(y_value),
        plot_bbox=plot_box,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
    )


def region_bbox(region: Region, *, plot_box: BBox, scale: float = 1.0) -> List[float]:
    cx, cy = scale_point(region.center_x, region.center_y, plot_box=plot_box)
    x0, y0, x1, y1 = (float(value) for value in plot_box)
    rx = (float(region.radius_x) * float(scale) / 100.0) * (x1 - x0)
    ry = (float(region.radius_y) * float(scale) / 100.0) * (y1 - y0)
    return bbox([cx - rx, cy - ry, cx + rx, cy + ry])


def draw_axes(draw: ImageDraw.ImageDraw, *, plot_box: BBox, rp: RenderParams) -> None:
    x0, y0, x1, y1 = (float(value) for value in plot_box)
    draw_plot_frame(draw, plot_box, fill=rp.plot_fill_rgb, outline=rp.grid_rgb, width=1)
    tick_font = load_font(int(rp.tick_font_size), bold=False)
    tick_values = (0, 25, 50, 75, 100)
    x_tick_positions = draw_vertical_value_grid_ticks(
        draw,
        plot_box,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=rp.grid_rgb,
        axis_rgb=rp.axis_rgb,
        grid_width_px=max(1, int(rp.grid_line_width)),
        tick_width_px=1,
        tick_length_px=0.0,
    )
    y_tick_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_box,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=rp.grid_rgb,
        axis_rgb=rp.axis_rgb,
        grid_width_px=max(1, int(rp.grid_line_width)),
        tick_width_px=1,
        tick_length_px=0.0,
    )
    for tick in tick_values:
        sx = float(x_tick_positions[float(tick)])
        sy = float(y_tick_positions[float(tick)])
        draw_text_traced(draw, (sx - 8.0, y1 + 10.0), str(tick), font=tick_font, fill=rp.muted_rgb, role="readout", required=False)
        draw_text_traced(draw, (x0 - 36.0, sy - 8.0), str(tick), font=tick_font, fill=rp.muted_rgb, role="readout", required=False)
    draw_axis_lines(draw, plot_box, axis_rgb=rp.axis_rgb, axis_width_px=max(1, int(rp.axis_line_width)))


def lighten(color: RGB, amount: float) -> RGB:
    return tuple(int(round(float(channel) + ((255.0 - float(channel)) * float(amount)))) for channel in color)


def draw_region(
    draw: ImageDraw.ImageDraw,
    *,
    region: Region,
    dataset: ContourDataset,
    plot_box: BBox,
    rp: RenderParams,
) -> List[float]:
    """Draw one density region without deciding whether that region is task-relevant."""

    outer_bbox = region_bbox(region, plot_box=plot_box, scale=1.9)
    density_level = max(1, min(5, int(region.density_level)))
    ring_count = int(density_level)
    if str(dataset.scene_variant) == "filled_density":
        fill = lighten(region.color_rgb, max(0.16, 0.82 - (0.12 * float(density_level))))
        draw.ellipse(outer_bbox, fill=fill, outline=region.color_rgb, width=2)
    for ring_index in range(ring_count, 0, -1):
        scale = 0.42 + ((1.45 / float(max(1, ring_count))) * float(ring_index))
        mark_bbox = region_bbox(region, plot_box=plot_box, scale=scale)
        width = 1 + (1 if ring_index <= 2 or ring_index == ring_count else 0)
        draw.ellipse(mark_bbox, outline=region.color_rgb, width=width)
    if str(dataset.scene_variant) == "scatter_contour":
        rng = spawn_rng(int(round(region.center_x * 1000 + region.center_y * 37)), f"{SCENE_NAMESPACE}.scatter_points.{region.region_id}")
        point_count = 5 + (3 * int(density_level))
        for _ in range(point_count):
            x_value = region.center_x + rng.uniform(-0.8 * region.radius_x, 0.8 * region.radius_x)
            y_value = region.center_y + rng.uniform(-0.8 * region.radius_y, 0.8 * region.radius_y)
            px, py = scale_point(x_value, y_value, plot_box=plot_box)
            pr = 2.8
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=region.color_rgb, outline=(255, 255, 255), width=1)
    label_font = load_font(int(rp.label_font_size), bold=dense_fit_bold())
    label = str(region.option_label or region.label)
    cx, cy = scale_point(region.center_x, region.center_y, plot_box=plot_box)
    draw_text_traced(
        draw,
        (cx + 7.0, cy - 8.0),
        label,
        font=label_font,
        fill=rp.text_rgb,
        stroke_fill=rp.text_stroke_rgb,
        stroke_width=dense_stroke_width(),
        role="readout",
        required=False,
    )
    return outer_bbox


def draw_density_threshold_guide(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: ContourDataset,
    plot_box: BBox,
    rp: RenderParams,
) -> Dict[str, Any]:
    if dataset.threshold_guide is None:
        return {}
    label = str(dataset.threshold_guide.label)
    font = load_font(int(rp.tick_font_size), bold=False)
    x0, y0, x1, _ = (float(value) for value in plot_box)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    width = float(text_bbox[2] - text_bbox[0]) + 22.0
    height = float(text_bbox[3] - text_bbox[1]) + 14.0
    bx1 = x1 - 8.0
    bx0 = max(x0 + 8.0, bx1 - width)
    by0 = max(8.0, y0 - height - 12.0)
    by1 = by0 + height
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=(255, 255, 255), outline=rp.grid_rgb, width=1)
    draw_text_traced(draw, (bx0 + 11.0, by0 + 6.0), label, font=font, fill=rp.text_rgb, role="readout", required=False)
    return {"density_threshold_guide_bbox_px": bbox([bx0, by0, bx1, by1]), "density_threshold_guide_text": label}


def draw_reference(
    draw: ImageDraw.ImageDraw,
    *,
    reference: Reference | None,
    plot_box: BBox,
    rp: RenderParams,
) -> Dict[str, List[float]]:
    if reference is None:
        return {}
    x0, y0, x1, y1 = (float(value) for value in plot_box)
    if str(reference.kind) == "point":
        cx, cy = scale_point(reference.x_value, reference.y_value, plot_box=plot_box)
        radius = max(12.0, float(rp.marker_radius) * 1.35)
        ref_bbox = bbox([cx - radius, cy - radius, cx + radius, cy + radius])
        draw.ellipse(ref_bbox, fill=rp.reference_rgb, outline=(255, 255, 255), width=2)
        ref_font = load_font(max(12, int(round(radius * 1.15))), bold=False)
        text_bbox = draw.textbbox((0, 0), "R", font=ref_font)
        text_w = float(text_bbox[2] - text_bbox[0])
        text_h = float(text_bbox[3] - text_bbox[1])
        draw_text_traced(
            draw,
            (cx - (text_w / 2.0), cy - (text_h / 2.0) - 1.0),
            "R",
            font=ref_font,
            fill=(255, 255, 255),
            role="readout",
            required=False,
        )
        return {"reference": list(ref_bbox)}
    if str(reference.kind) == "vertical_line":
        sx, _ = scale_point(reference.x_value, 0.0, plot_box=plot_box)
        draw.line([sx, y0, sx, y1], fill=rp.reference_rgb, width=3)
        return {"reference": bbox([sx - 2.0, y0, sx + 2.0, y1])}
    if str(reference.kind) == "horizontal_line":
        _, sy = scale_point(0.0, reference.y_value, plot_box=plot_box)
        draw.line([x0, sy, x1, sy], fill=rp.reference_rgb, width=3)
        return {"reference": bbox([x0, sy - 2.0, x1, sy + 2.0])}
    raise ValueError(f"unsupported reference kind: {reference.kind}")


def _render_dataset(dataset: ContourDataset, *, params: Mapping[str, Any], instance_seed: int) -> RenderedContourScene:
    """Render the full contour scene and record neutral projected geometry maps."""

    params = {**dict(params), "_render_style_seed": int(instance_seed)}
    rp = resolve_render_params(params, instance_seed=int(instance_seed))
    protected_colors = tuple(tuple(int(channel) for channel in region.color_rgb) for region in dataset.regions)
    rp, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="contour_density",
        render_params=rp,
        protected_colors=protected_colors,
    )
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_box = plot_bbox(rp)
    draw_axes(draw, plot_box=plot_box, rp=rp)
    threshold_guide_meta = draw_density_threshold_guide(draw, dataset=dataset, plot_box=plot_box, rp=rp)

    entities: List[Dict[str, Any]] = []
    region_bboxes: Dict[str, List[float]] = {}
    option_bboxes: Dict[str, List[float]] = {}
    for region in dataset.regions:
        mark_bbox = draw_region(draw, region=region, dataset=dataset, plot_box=plot_box, rp=rp)
        region_bboxes[str(region.region_id)] = list(mark_bbox)
        if str(region.option_label):
            option_bboxes[str(region.option_label)] = list(mark_bbox)
        entities.append(
            {
                "entity_id": str(region.region_id),
                "entity_type": "contour_density_region",
                "bbox_px": list(mark_bbox),
                "attrs": {
                    "label": str(region.label),
                    "option_label": str(region.option_label),
                    "center": [round(float(region.center_x), 3), round(float(region.center_y), 3)],
                    "density": round(float(region.density), 3),
                    "density_level": int(region.density_level),
                    "radius": [round(float(region.radius_x), 3), round(float(region.radius_y), 3)],
                },
            }
        )
    reference_bboxes = draw_reference(draw, reference=dataset.reference, plot_box=plot_box, rp=rp)
    if reference_bboxes:
        entities.append(
            {
                "entity_id": "reference",
                "entity_type": "contour_density_reference",
                "bbox_px": list(reference_bboxes["reference"]),
                "attrs": {
                    "kind": str(dataset.reference.kind if dataset.reference else ""),
                    "x_value": round(float(dataset.reference.x_value if dataset.reference else 0), 3),
                    "y_value": round(float(dataset.reference.y_value if dataset.reference else 0), 3),
                },
            }
        )
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedContourScene(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=bbox(plot_box),
        region_bboxes=dict(region_bboxes),
        option_bboxes=dict(option_bboxes),
        reference_bboxes=dict(reference_bboxes),
        render_meta={
            "background_style": dict(background_meta),
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "layout_jitter": dict(rp.layout_jitter),
            **dict(threshold_guide_meta),
        },
    )


def render_dataset(dataset: ContourDataset, *, params: Mapping[str, Any], instance_seed: int) -> tuple[RenderedContourScene, str]:
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = _render_dataset(dataset, params=params, instance_seed=int(instance_seed))
    return rendered, str(chart_font_family)


def font_assets_for_family(chart_font_family: str) -> dict[str, Any]:
    return chart_font_asset_metadata(str(chart_font_family))


__all__ = [
    "font_assets_for_family",
    "plot_bbox",
    "region_bbox",
    "render_dataset",
    "scale_point",
]
