"""Shared runtime for coordinate-plane locus label tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.text_rendering import load_font, symbol_safe_font_for_text
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.geometry.shared.background_defaults import load_geometry_background_defaults
from trace_tasks.tasks.geometry.shared.coordinate_panel_grid import (
    CoordinatePanelConfig,
    coordinate_panel_layout,
    draw_coordinate_panel_grid,
    graph_point_to_panel_pixel,
    panel_bbox_for_index,
    plot_bbox_for_panel,
)
from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_COORDINATE_GRID,
    geometry_coordinate_panel_style_from_diagram_style,
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)

SCENE_ID = "coordinate_plane"
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame, graph_units_to_pixel, scale_point
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.geometry.shared.option_count import panel_grid_shape_for_option_count, resolve_geometry_option_count
from trace_tasks.tasks.geometry.shared.point_labels import draw_labeled_points
from trace_tasks.tasks.geometry.shared.single_object_scene import finalize_graph_scene_image, make_graph_scene_canvas, resolve_graph_scene_context
from .spatial_primitives import (
    _draw_marker,
    _marker_bbox,
    _probability_map,
    _resolve_label_pool,
    _resolve_marker_colors,
    _sample_marker_style,
)
from .defaults import resolve_int_param as _resolve_int_param
from .output import build_option_letter_prompt_artifacts


GraphPoint = Tuple[int, int]
PixelPoint = Tuple[float, float]
Color = Tuple[int, int, int]
BBox = Tuple[int, int, int, int]

POINT_REGION_FAMILIES: Tuple[str, ...] = (
    "circle_region",
    "annulus_region",
    "vertical_strip_region",
    "half_plane_intersection_region",
)
PANEL_REGION_FAMILIES: Tuple[str, ...] = (
    "circle_panel",
    "vertical_strip_panel",
    "horizontal_halfplane_panel",
    "two_inequality_panel",
)
DEFAULT_LABEL_POOL: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

_SCENE_DEFAULTS = get_scene_defaults("geometry", "coordinate_plane")
_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id=SCENE_ID)
_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)
@dataclass(frozen=True)
class _TaskDefaults:
    canvas_size_min: int = 680
    canvas_size_max: int = 760
    graph_cells_min: int = 18
    graph_cells_max: int = 20
    graph_abs_max: int = 6
    candidate_count: int = 6
    marker_radius_px: int = 7
    marker_radius_px_min: int = 6
    marker_radius_px_max: int = 9
    locus_marker_radius_px: int = 11
    locus_marker_radius_px_min: int = 10
    locus_marker_radius_px_max: int = 14
    locus_label_font_size_min: int = 28
    locus_label_font_size_max: int = 42
    locus_label_offset_px: int = 24
    label_font_size_min: int = 16
    label_font_size_max: int = 28
    label_stroke_width: int = 1
    label_offset_px: int = 15
    panel_canvas_width: int = 1040
    panel_canvas_height: int = 760
    panel_canvas_width_4: int = 900
    panel_canvas_height_4: int = 840
    panel_grid_min: int = -6
    panel_grid_max: int = 6
    panel_count: int = 6
    panel_top_reserved_px: int = 58
    panel_marker_radius_px: int = 4
    panel_marker_radius_px_min: int = 3
    panel_marker_radius_px_max: int = 5
    condition_label_font_size: int = 18


@dataclass(frozen=True)
class _RegionSpec:
    kind: str
    params: Dict[str, int]
    condition_text: str


@dataclass(frozen=True)
class _ResolvedQuery:
    operation_key: str
    query_probabilities: Dict[str, float]
    winner_label: str
    winner_label_probabilities: Dict[str, float]
    label_pool: Tuple[str, ...]


@dataclass(frozen=True)
class _PointScene:
    region: _RegionSpec
    candidate_points_by_label: Dict[str, GraphPoint]
    candidate_bboxes_by_label: Dict[str, List[int]]
    candidate_points_px_by_label: Dict[str, PixelPoint]
    center_point_px: PixelPoint | None
    marker_meta: Dict[str, Any]
    image: Image.Image
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    render_spec_extra: Dict[str, Any]
    option_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _PanelSpec:
    label: str
    region: _RegionSpec
    panel_bbox: List[int]
    plot_bbox: List[int]
    is_answer: bool


@dataclass(frozen=True)
class _PanelScene:
    condition_text: str
    panels_by_label: Dict[str, _PanelSpec]
    condition_label_bbox_px: List[int]
    panel_style_meta: Dict[str, Any]
    marker_meta: Dict[str, Any]
    image: Image.Image
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    option_count_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_REGION_FILL: Color = (82, 142, 218)
_REGION_OUTLINE: Color = (32, 83, 138)
_BOUNDARY_DASH_FILL: Color = (42, 84, 124)


def _squared_term(variable: str, center: int) -> str:
    """Return a readable squared coordinate term for a shifted center."""

    if int(center) == 0:
        return f"{variable}²"
    sign = "-" if int(center) > 0 else "+"
    return f"({variable} {sign} {abs(int(center))})²"


def _circle_condition_text(*, cx: int, cy: int, radius: int) -> str:
    return f"{_squared_term('x', int(cx))} + {_squared_term('y', int(cy))} ≤ {int(radius) ** 2}"


def _annulus_condition_text(*, cx: int, cy: int, inner_radius: int, outer_radius: int) -> str:
    center_expr = f"{_squared_term('x', int(cx))} + {_squared_term('y', int(cy))}"
    return f"{int(inner_radius) ** 2} ≤ {center_expr} ≤ {int(outer_radius) ** 2}"


def _interval_condition_text(*, variable: str, lower: int, upper: int) -> str:
    return f"{int(lower)} ≤ {variable} ≤ {int(upper)}"


def _halfplane_condition_text(*, variable: str, op: str, value: int) -> str:
    return f"{variable} {op} {int(value)}"


def _intersection_condition_text(params: Mapping[str, int]) -> str:
    parts: list[str] = []
    if "x_min" in params:
        parts.append(_halfplane_condition_text(variable="x", op="≥", value=int(params["x_min"])))
    if "x_max" in params:
        parts.append(_halfplane_condition_text(variable="x", op="≤", value=int(params["x_max"])))
    if "y_min" in params:
        parts.append(_halfplane_condition_text(variable="y", op="≥", value=int(params["y_min"])))
    if "y_max" in params:
        parts.append(_halfplane_condition_text(variable="y", op="≤", value=int(params["y_max"])))
    return " and ".join(parts)


def _split_defaults_for_task(config_key: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    return split_scene_generation_rendering_prompt_defaults(
        _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
        task_id=str(config_key),
    )


def _select_winner_label(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    label_pool: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    labels = tuple(str(label) for label in label_pool)
    explicit = params.get("winner_label", params.get("answer_label"))
    if explicit is not None:
        label = str(explicit)
        if label not in set(labels):
            raise ValueError(f"winner_label={label!r} is not in label pool {labels!r}")
        return label, {label: 1.0}
    rng = spawn_rng(int(instance_seed), f"{namespace}.winner_label")
    return str(uniform_choice(rng, labels)), _probability_map(labels)


def _candidate_labels_for_selection(query: _ResolvedQuery, *, candidate_count: int) -> Tuple[str, ...]:
    labels = tuple(query.label_pool[: int(candidate_count)])
    if str(query.winner_label) not in set(labels):
        raise ValueError("winner_label must be inside the active contiguous candidate label set")
    return labels


def _region_contains(region: _RegionSpec, point: GraphPoint) -> bool:
    x, y = int(point[0]), int(point[1])
    params = dict(region.params)
    if str(region.kind) in {"circle", "annulus"}:
        cx = int(params.get("cx", 0))
        cy = int(params.get("cy", 0))
        d2 = (int(x) - cx) ** 2 + (int(y) - cy) ** 2
        if str(region.kind) == "circle":
            return int(d2) <= int(params["r"]) ** 2
        return int(params["inner_r"]) ** 2 <= int(d2) <= int(params["outer_r"]) ** 2
    if "x_min" in params and int(x) < int(params["x_min"]):
        return False
    if "x_max" in params and int(x) > int(params["x_max"]):
        return False
    if "y_min" in params and int(y) < int(params["y_min"]):
        return False
    if "y_max" in params and int(y) > int(params["y_max"]):
        return False
    return True


def _region_boundary_point(region: _RegionSpec, point: GraphPoint) -> bool:
    x, y = int(point[0]), int(point[1])
    params = dict(region.params)
    if str(region.kind) == "circle":
        cx = int(params.get("cx", 0))
        cy = int(params.get("cy", 0))
        return ((x - cx) ** 2 + (y - cy) ** 2) == int(params["r"]) ** 2
    if str(region.kind) == "annulus":
        cx = int(params.get("cx", 0))
        cy = int(params.get("cy", 0))
        d2 = (x - cx) ** 2 + (y - cy) ** 2
        return int(d2) in {int(params["inner_r"]) ** 2, int(params["outer_r"]) ** 2}
    return any(
        (
            "x_min" in params and x == int(params["x_min"]),
            "x_max" in params and x == int(params["x_max"]),
            "y_min" in params and y == int(params["y_min"]),
            "y_max" in params and y == int(params["y_max"]),
        )
    )


def _sample_region_for_point_query(operation_key: str, rng, *, max_abs: int) -> _RegionSpec:
    circle_cases = (
        _RegionSpec("circle", {"cx": 0, "cy": 0, "r": 4}, "inside circle centered at O"),
        _RegionSpec("circle", {"cx": 1, "cy": -1, "r": 3}, "inside circle centered at O"),
        _RegionSpec("circle", {"cx": -1, "cy": 2, "r": 3}, "inside circle centered at O"),
        _RegionSpec("circle", {"cx": 2, "cy": 1, "r": 3}, "inside circle centered at O"),
    )
    annulus_cases = (
        _RegionSpec("annulus", {"cx": 0, "cy": 0, "inner_r": 2, "outer_r": 5}, "inside the shaded ring"),
        _RegionSpec("annulus", {"cx": 1, "cy": -1, "inner_r": 2, "outer_r": 4}, "inside the shaded ring"),
        _RegionSpec("annulus", {"cx": -1, "cy": 1, "inner_r": 2, "outer_r": 4}, "inside the shaded ring"),
    )
    strip_cases = (
        _RegionSpec("vertical_strip", {"x_min": -2, "x_max": 2}, "-2 <= x <= 2"),
        _RegionSpec("vertical_strip", {"x_min": -4, "x_max": -1}, "-4 <= x <= -1"),
        _RegionSpec("vertical_strip", {"x_min": 1, "x_max": 4}, "1 <= x <= 4"),
    )
    intersection_cases = (
        _RegionSpec("half_plane_intersection", {"x_min": -1, "y_max": 2}, "x >= -1 and y <= 2"),
        _RegionSpec("half_plane_intersection", {"x_min": 1, "y_max": 1}, "x >= 1 and y <= 1"),
        _RegionSpec("half_plane_intersection", {"x_min": -2, "y_max": 0}, "x >= -2 and y <= 0"),
    )
    if str(operation_key) == "circle_region":
        return rng.choice(circle_cases)
    if str(operation_key) == "annulus_region":
        return rng.choice(annulus_cases)
    if str(operation_key) == "vertical_strip_region":
        return rng.choice(strip_cases)
    if str(operation_key) == "half_plane_intersection_region":
        return rng.choice(intersection_cases)
    raise ValueError(f"unsupported locus point query: {operation_key}")


def _grid_points(max_abs: int) -> Tuple[GraphPoint, ...]:
    return tuple((x, y) for x in range(-int(max_abs), int(max_abs) + 1) for y in range(-int(max_abs), int(max_abs) + 1))


def _sample_candidate_points(
    *,
    region: _RegionSpec,
    query: _ResolvedQuery,
    candidate_labels: Sequence[str],
    rng,
    max_abs: int,
) -> Dict[str, GraphPoint]:
    available = [point for point in _grid_points(int(max_abs)) if not _region_boundary_point(region, point)]
    inside = [point for point in available if _region_contains(region, point)]
    outside = [point for point in available if not _region_contains(region, point)]
    if not inside or len(outside) < int(len(candidate_labels) - 1):
        raise RuntimeError("locus region does not have enough candidate support")
    target = tuple(rng.choice(inside))
    rng.shuffle(outside)
    candidate_points_by_label: Dict[str, GraphPoint] = {str(query.winner_label): target}
    occupied = {target}
    for label in candidate_labels:
        if str(label) == str(query.winner_label):
            continue
        for point in outside:
            if tuple(point) in occupied:
                continue
            candidate_points_by_label[str(label)] = tuple(point)
            occupied.add(tuple(point))
            break
        if str(label) not in candidate_points_by_label:
            raise RuntimeError("failed to sample locus-region distractor")
    return dict(candidate_points_by_label)


def _scaled_bbox(bbox: Sequence[int], scale: int) -> Tuple[int, int, int, int]:
    return tuple(int(value) * int(scale) for value in bbox)  # type: ignore[return-value]


def _clip_bbox(bbox: Sequence[float], clip: Sequence[int]) -> Tuple[int, int, int, int] | None:
    left = max(int(round(float(bbox[0]))), int(clip[0]))
    top = max(int(round(float(bbox[1]))), int(clip[1]))
    right = min(int(round(float(bbox[2]))), int(clip[2]))
    bottom = min(int(round(float(bbox[3]))), int(clip[3]))
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom)


def _apply_region_mask(image: Image.Image, mask: Image.Image, *, color: Color) -> None:
    overlay = Image.new("RGBA", image.size, (int(color[0]), int(color[1]), int(color[2]), 0))
    overlay.putalpha(mask)
    composed = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    image.paste(composed)


def _draw_scaled_line(draw: ImageDraw.ImageDraw, points: Sequence[PixelPoint], *, fill: Color, width: int) -> None:
    draw.line([(float(x), float(y)) for x, y in points], fill=fill, width=max(1, int(width)))


def _project_single(point: Tuple[float, float], *, context: Any) -> PixelPoint:
    return scale_point(
        graph_units_to_pixel(point, graph_origin=context.graph_origin, spacing=int(context.graph_spacing)),
        int(context.scene_scale),
    )


def _single_region_bbox(region: _RegionSpec, *, context: Any, radius_key: str) -> Tuple[int, int, int, int]:
    params = dict(region.params)
    cx = int(params.get("cx", 0))
    cy = int(params.get("cy", 0))
    radius = int(params[radius_key])
    left_top = _project_single((cx - radius, cy + radius), context=context)
    right_bottom = _project_single((cx + radius, cy - radius), context=context)
    return (
        int(round(float(left_top[0]))),
        int(round(float(left_top[1]))),
        int(round(float(right_bottom[0]))),
        int(round(float(right_bottom[1]))),
    )


def _constraint_rect(region: _RegionSpec, *, grid_min: int, grid_max: int) -> Tuple[int, int, int, int]:
    params = dict(region.params)
    return (
        int(params.get("x_min", grid_min)),
        int(params.get("y_min", grid_min)),
        int(params.get("x_max", grid_max)),
        int(params.get("y_max", grid_max)),
    )


def _draw_single_region(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    region: _RegionSpec,
    context: Any,
    max_abs: int,
) -> None:
    """Draw one locus region on the resolved graph-paper canvas."""

    scale = int(context.scene_scale)
    clip = _scaled_bbox(context.graph_panel_layout.content_bbox_px, int(scale))
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if str(region.kind) == "circle":
        bbox = _clip_bbox(_single_region_bbox(region, context=context, radius_key="r"), clip)
        if bbox is not None:
            mask_draw.ellipse(bbox, fill=78)
            _apply_region_mask(image, mask, color=_REGION_FILL)
            draw.ellipse(bbox, outline=_REGION_OUTLINE, width=max(2, 2 * scale))
        return
    if str(region.kind) == "annulus":
        outer = _clip_bbox(_single_region_bbox(region, context=context, radius_key="outer_r"), clip)
        inner = _clip_bbox(_single_region_bbox(region, context=context, radius_key="inner_r"), clip)
        if outer is not None:
            mask_draw.ellipse(outer, fill=82)
            if inner is not None:
                mask_draw.ellipse(inner, fill=0)
            _apply_region_mask(image, mask, color=_REGION_FILL)
            draw.ellipse(outer, outline=_REGION_OUTLINE, width=max(2, 2 * scale))
            if inner is not None:
                draw.ellipse(inner, outline=_REGION_OUTLINE, width=max(2, 2 * scale))
        return

    x_min, y_min, x_max, y_max = _constraint_rect(region, grid_min=-int(max_abs), grid_max=int(max_abs))
    top_left = _project_single((x_min, y_max), context=context)
    bottom_right = _project_single((x_max, y_min), context=context)
    rect = _clip_bbox((top_left[0], top_left[1], bottom_right[0], bottom_right[1]), clip)
    if rect is None:
        return
    mask_draw.rectangle(rect, fill=76)
    _apply_region_mask(image, mask, color=_REGION_FILL)
    params = dict(region.params)
    line_width = max(2, 2 * scale)
    for key in ("x_min", "x_max"):
        if key not in params:
            continue
        x_value = int(params[key])
        start = _project_single((x_value, -int(max_abs)), context=context)
        end = _project_single((x_value, int(max_abs)), context=context)
        _draw_scaled_line(draw, [(start[0], clip[1]), (end[0], clip[3])], fill=_BOUNDARY_DASH_FILL, width=line_width)
    for key in ("y_min", "y_max"):
        if key not in params:
            continue
        y_value = int(params[key])
        start = _project_single((-int(max_abs), y_value), context=context)
        end = _project_single((int(max_abs), y_value), context=context)
        _draw_scaled_line(draw, [(clip[0], start[1]), (clip[2], end[1])], fill=_BOUNDARY_DASH_FILL, width=line_width)


def _draw_center_marker(
    draw: ImageDraw.ImageDraw,
    *,
    context: Any,
    region: _RegionSpec,
    marker_radius: int,
    label_font_size_px: int,
    label_offset_px: int,
    label_stroke_width: int,
    color: Color,
) -> PixelPoint | None:
    """Mark circle centers only when the visible condition uses them."""

    if str(region.kind) not in {"circle", "annulus"}:
        return None
    center = (int(region.params.get("cx", 0)), int(region.params.get("cy", 0)))
    point_px = graph_units_to_pixel(center, graph_origin=context.graph_origin, spacing=int(context.graph_spacing))
    render_px = scale_point(point_px, int(context.scene_scale))
    _draw_marker(
        draw,
        render_px,
        style="cross",
        color=color,
        radius=max(4, int(marker_radius) * int(context.scene_scale)),
        width=max(2, int(context.scene_scale) * 2),
    )
    draw_labeled_points(
        draw,
        points=[render_px],
        labels=["O"],
        label_offset_px=float(label_offset_px) * float(context.scene_scale),
        font_size_px=int(label_font_size_px),
        text_stroke_width=int(label_stroke_width) * int(context.scene_scale),
        blocked_points=[render_px],
        blocked_point_clearance_px=float(marker_radius * int(context.scene_scale) + 8),
        marker_radius_px=0,
        marker_color=color,
        label_color=color,
        label_stroke_color=(255, 255, 255),
        canvas_size=int(context.canvas_size) * int(context.scene_scale),
    )
    return (float(point_px[0]), float(point_px[1]))


def _render_point_scene(
    query: _ResolvedQuery,
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> _PointScene:
    """Render one locus region and lettered candidate points after sampling."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    max_abs = _resolve_int_param(params, generation_defaults, "locus_graph_abs_max", _DEFAULTS.graph_abs_max)
    candidate_count, option_count_probabilities = resolve_geometry_option_count(
        params=params,
        gen_defaults=generation_defaults,
        field_name="locus_candidate_count",
        supported_counts=(4, 6),
        task_id=str(namespace),
        instance_seed=int(instance_seed),
    )
    if int(candidate_count) > len(query.label_pool):
        raise ValueError("locus_candidate_count cannot exceed candidate label pool length")
    candidate_labels = _candidate_labels_for_selection(query, candidate_count=int(candidate_count))
    region = _sample_region_for_point_query(str(query.operation_key), rng, max_abs=int(max_abs))
    candidate_points_by_label = _sample_candidate_points(
        region=region,
        query=query,
        candidate_labels=candidate_labels,
        rng=rng,
        max_abs=int(max_abs),
    )
    context = resolve_graph_scene_context(
        rng,
        instance_seed=int(instance_seed),
        scene_id=SCENE_ID,
        params=params,
        render_defaults=rendering_defaults,
        background_defaults=_BACKGROUND_DEFAULTS,
        fallback_canvas_min=_resolve_int_param(params, rendering_defaults, "locus_canvas_size_min", _DEFAULTS.canvas_size_min),
        fallback_canvas_max=_resolve_int_param(params, rendering_defaults, "locus_canvas_size_max", _DEFAULTS.canvas_size_max),
        fallback_cells_min=_resolve_int_param(params, rendering_defaults, "locus_graph_cells_min", _DEFAULTS.graph_cells_min),
        fallback_cells_max=_resolve_int_param(params, rendering_defaults, "locus_graph_cells_max", _DEFAULTS.graph_cells_max),
        require_graph_paper_background=True,
        graph_style_overrides={
            "origin_fraction_x": 0.5,
            "origin_fraction_y": 0.5,
            "axis_scale_labels_enabled": True,
            "axis_scale_label_max_abs": max(6, int(max_abs)),
            "origin_label_enabled": False,
        },
    )
    image, draw, background_meta = make_graph_scene_canvas(
        instance_seed=int(instance_seed),
        context=context,
        background_defaults=_BACKGROUND_DEFAULTS,
        require_graph_paper=True,
    )
    _draw_single_region(image, draw, region=region, context=context, max_abs=int(max_abs))

    generic_marker_radius = _resolve_int_param(params, rendering_defaults, "marker_radius_px", _DEFAULTS.marker_radius_px)
    marker_radius = _resolve_int_param(params, rendering_defaults, "locus_marker_radius_px", _DEFAULTS.locus_marker_radius_px)
    marker_radius = max(
        _resolve_int_param(
            params,
            rendering_defaults,
            "locus_marker_radius_px_min",
            max(_DEFAULTS.locus_marker_radius_px_min, int(generic_marker_radius)),
        ),
        min(
            _resolve_int_param(params, rendering_defaults, "locus_marker_radius_px_max", _DEFAULTS.locus_marker_radius_px_max),
            int(marker_radius),
        ),
    )
    generic_label_min = _resolve_int_param(params, rendering_defaults, "label_font_size_min", _DEFAULTS.label_font_size_min)
    generic_label_max = _resolve_int_param(params, rendering_defaults, "label_font_size_max", _DEFAULTS.label_font_size_max)
    label_min = _resolve_int_param(
        params,
        rendering_defaults,
        "locus_label_font_size_min",
        max(_DEFAULTS.locus_label_font_size_min, int(generic_label_min)),
    )
    label_max = _resolve_int_param(
        params,
        rendering_defaults,
        "locus_label_font_size_max",
        max(_DEFAULTS.locus_label_font_size_max, int(generic_label_max)),
    )
    label_font_size_px = int(
        max(
            int(label_min),
            min(
                int(label_max),
                int(round(float(context.graph_spacing) * 1.12)),
            ),
        )
    )
    label_offset_px = _resolve_int_param(params, rendering_defaults, "locus_label_offset_px", _DEFAULTS.locus_label_offset_px)
    label_stroke_width = _resolve_int_param(params, rendering_defaults, "label_stroke_width", _DEFAULTS.label_stroke_width)
    _, candidate_color, color_meta = _resolve_marker_colors(rng)
    candidate_style = _sample_marker_style(rng, params=params, defaults=rendering_defaults, key="candidate_marker_style")
    center_point_px: PixelPoint | None = None
    candidate_points_px_by_label = {
        str(label): graph_units_to_pixel(point, graph_origin=context.graph_origin, spacing=int(context.graph_spacing))
        for label, point in candidate_points_by_label.items()
    }
    render_radius = int(marker_radius) * int(context.scene_scale)
    for label in candidate_labels:
        _draw_marker(
            draw,
            scale_point(candidate_points_px_by_label[str(label)], int(context.scene_scale)),
            style=str(candidate_style),
            color=candidate_color,
            radius=int(render_radius),
            width=max(2, int(context.scene_scale) * 2),
        )
    blocked_points = [scale_point(candidate_points_px_by_label[str(label)], int(context.scene_scale)) for label in candidate_labels]
    draw_labeled_points(
        draw,
        points=[scale_point(candidate_points_px_by_label[str(label)], int(context.scene_scale)) for label in candidate_labels],
        labels=list(candidate_labels),
        label_offset_px=float(label_offset_px) * float(context.scene_scale),
        font_size_px=int(label_font_size_px) * int(context.scene_scale),
        text_stroke_width=int(label_stroke_width) * int(context.scene_scale),
        blocked_points=blocked_points,
        blocked_point_clearance_px=float(render_radius + 7),
        marker_radius_px=0,
        marker_color=candidate_color,
        label_color=candidate_color,
        label_stroke_color=(255, 255, 255),
        canvas_size=int(context.canvas_size) * int(context.scene_scale),
    )
    candidate_bboxes_by_label = {
        str(label): _marker_bbox(
            candidate_points_px_by_label[str(label)],
            radius=int(marker_radius),
            canvas_width=int(context.canvas_size),
            canvas_height=int(context.canvas_size),
        )
        for label in candidate_labels
    }
    image, background_meta_final, post_noise_meta = finalize_graph_scene_image(
        image,
        instance_seed=int(instance_seed),
        context=context,
        background_meta=background_meta,
        noise_defaults=_NOISE_DEFAULTS,
    )
    marker_meta = {
        "candidate_marker_style": str(candidate_style),
        "marker_radius_px": int(marker_radius),
        "region_fill": list(_REGION_FILL),
        "region_outline": list(_REGION_OUTLINE),
        "region_fill_alpha": 78,
        **dict(color_meta),
    }
    return _PointScene(
        region=region,
        candidate_points_by_label=dict(candidate_points_by_label),
        candidate_bboxes_by_label=dict(candidate_bboxes_by_label),
        candidate_points_px_by_label=dict(candidate_points_px_by_label),
        center_point_px=center_point_px,
        marker_meta=dict(marker_meta),
        image=image,
        background_meta=dict(background_meta_final),
        post_noise_meta=dict(post_noise_meta),
        render_spec_extra={
            "canvas_size": int(context.canvas_size),
            "coord_space": "pixel",
            "graph_coordinate_frame": dict(context.graph_frame),
            "graph_paper_grid": graph_paper_grid_from_frame(context.graph_frame),
            "scene_scale": int(context.scene_scale),
            **dict(context.graph_layout_metadata),
        },
        option_count_probabilities=dict(option_count_probabilities),
    )

def _circle_region(cx: int, cy: int, radius: int) -> _RegionSpec:
    return _RegionSpec(
        "circle",
        {"cx": int(cx), "cy": int(cy), "r": int(radius)},
        _circle_condition_text(cx=int(cx), cy=int(cy), radius=int(radius)),
    )


def _annulus_region(cx: int, cy: int, inner_radius: int, outer_radius: int) -> _RegionSpec:
    return _RegionSpec(
        "annulus",
        {"cx": int(cx), "cy": int(cy), "inner_r": int(inner_radius), "outer_r": int(outer_radius)},
        _annulus_condition_text(
            cx=int(cx),
            cy=int(cy),
            inner_radius=int(inner_radius),
            outer_radius=int(outer_radius),
        ),
    )


def _vertical_strip_region(x_min: int, x_max: int) -> _RegionSpec:
    return _RegionSpec(
        "vertical_strip",
        {"x_min": int(x_min), "x_max": int(x_max)},
        _interval_condition_text(variable="x", lower=int(x_min), upper=int(x_max)),
    )


def _horizontal_strip_region(y_min: int, y_max: int) -> _RegionSpec:
    return _RegionSpec(
        "horizontal_strip",
        {"y_min": int(y_min), "y_max": int(y_max)},
        _interval_condition_text(variable="y", lower=int(y_min), upper=int(y_max)),
    )


def _halfplane_region(kind: str, params: Mapping[str, int], *, variable: str, op: str, value: int) -> _RegionSpec:
    return _RegionSpec(
        str(kind),
        {str(key): int(val) for key, val in params.items()},
        _halfplane_condition_text(variable=str(variable), op=str(op), value=int(value)),
    )


def _intersection_region(params: Mapping[str, int]) -> _RegionSpec:
    return _RegionSpec(
        "half_plane_intersection",
        {str(key): int(value) for key, value in params.items()},
        _intersection_condition_text({str(key): int(value) for key, value in params.items()}),
    )


def _same_region(left: _RegionSpec, right: _RegionSpec) -> bool:
    return str(left.kind) == str(right.kind) and dict(left.params) == dict(right.params)


def _panel_target_region(operation_key: str, rng) -> _RegionSpec:
    """Sample the correct locus panel shape for a semantic panel query.

    The helper returns only identity-free region geometry; the public task has
    already chosen the query and answer label, and this function never handles
    task ids, query ids, or final answer construction.
    """
    if str(operation_key) == "circle_panel":
        return rng.choice(
            (
                _circle_region(0, 0, 4),
                _circle_region(1, -1, 3),
                _circle_region(-1, 1, 3),
                _circle_region(2, 0, 3),
                _circle_region(0, -2, 3),
            )
        )
    if str(operation_key) == "vertical_strip_panel":
        return rng.choice(
            (
                _vertical_strip_region(-2, 2),
                _vertical_strip_region(-4, -1),
                _vertical_strip_region(-3, 1),
                _vertical_strip_region(1, 4),
                _vertical_strip_region(0, 3),
            )
        )
    if str(operation_key) == "horizontal_halfplane_panel":
        return rng.choice(
            (
                _halfplane_region("upper_halfplane", {"y_min": -1}, variable="y", op="≥", value=-1),
                _halfplane_region("upper_halfplane", {"y_min": 1}, variable="y", op="≥", value=1),
                _halfplane_region("upper_halfplane", {"y_min": -3}, variable="y", op="≥", value=-3),
                _halfplane_region("lower_halfplane", {"y_max": 2}, variable="y", op="≤", value=2),
                _halfplane_region("lower_halfplane", {"y_max": 0}, variable="y", op="≤", value=0),
            )
        )
    if str(operation_key) == "two_inequality_panel":
        return rng.choice(
            (
                _intersection_region({"x_min": -1, "y_max": 3}),
                _intersection_region({"x_max": -1, "y_max": 2}),
                _intersection_region({"x_min": 1, "y_min": -2}),
                _intersection_region({"x_max": 2, "y_min": 1}),
                _intersection_region({"x_min": -3, "y_max": 0}),
            )
        )
    raise ValueError(f"unsupported locus panel query: {operation_key}")


def _panel_distractor_regions(operation_key: str) -> Tuple[_RegionSpec, ...]:
    """Return reusable distractor region families for the selected panel grammar.

    Distractors are geometry primitives compatible with the visual panel layout;
    winner placement, option count, and annotation binding stay outside this
    shared helper.
    """
    if str(operation_key) == "circle_panel":
        return (
            _circle_region(0, 0, 3),
            _circle_region(1, 0, 4),
            _circle_region(-1, 2, 3),
            _annulus_region(0, 0, 2, 4),
            _annulus_region(1, -1, 1, 4),
            _vertical_strip_region(-2, 2),
            _horizontal_strip_region(-2, 2),
            _halfplane_region("upper_halfplane", {"y_min": -1}, variable="y", op="≥", value=-1),
        )
    if str(operation_key) == "vertical_strip_panel":
        return (
            _horizontal_strip_region(-2, 2),
            _vertical_strip_region(-3, 1),
            _vertical_strip_region(0, 4),
            _halfplane_region("right_halfplane", {"x_min": -2}, variable="x", op="≥", value=-2),
            _halfplane_region("left_halfplane", {"x_max": 2}, variable="x", op="≤", value=2),
            _circle_region(0, 0, 3),
            _intersection_region({"x_min": -1, "y_max": 3}),
        )
    if str(operation_key) == "horizontal_halfplane_panel":
        return (
            _halfplane_region("lower_halfplane", {"y_max": -1}, variable="y", op="≤", value=-1),
            _halfplane_region("upper_halfplane", {"y_min": 2}, variable="y", op="≥", value=2),
            _halfplane_region("right_halfplane", {"x_min": -1}, variable="x", op="≥", value=-1),
            _halfplane_region("left_halfplane", {"x_max": -1}, variable="x", op="≤", value=-1),
            _horizontal_strip_region(-1, 3),
            _vertical_strip_region(-2, 2),
            _circle_region(0, 0, 3),
        )
    if str(operation_key) == "two_inequality_panel":
        return (
            _intersection_region({"x_max": -1, "y_max": 3}),
            _intersection_region({"x_min": -1, "y_min": 3}),
            _intersection_region({"x_min": 1, "y_max": 3}),
            _intersection_region({"x_min": -1, "y_max": 1}),
            _intersection_region({"x_max": 2, "y_min": -2}),
            _vertical_strip_region(-1, 3),
            _horizontal_strip_region(-2, 2),
            _circle_region(0, 0, 3),
        )
    raise ValueError(f"unsupported locus panel query: {operation_key}")


def _panel_region_bbox(
    region: _RegionSpec,
    *,
    plot_bbox: BBox,
    config: CoordinatePanelConfig,
    radius_key: str,
) -> Tuple[int, int, int, int]:
    params = dict(region.params)
    cx = int(params.get("cx", 0))
    cy = int(params.get("cy", 0))
    radius = int(params[radius_key])
    left_top = graph_point_to_panel_pixel((cx - radius, cy + radius), plot_bbox=plot_bbox, config=config)
    right_bottom = graph_point_to_panel_pixel((cx + radius, cy - radius), plot_bbox=plot_bbox, config=config)
    return (
        int(round(float(left_top[0]))),
        int(round(float(left_top[1]))),
        int(round(float(right_bottom[0]))),
        int(round(float(right_bottom[1]))),
    )


def _draw_panel_region(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    region: _RegionSpec,
    plot_bbox: BBox,
    config: CoordinatePanelConfig,
) -> None:
    """Draw a locus region inside one already laid-out panel plot."""

    clip = tuple(int(value) for value in plot_bbox)
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    if str(region.kind) == "circle":
        bbox = _clip_bbox(_panel_region_bbox(region, plot_bbox=plot_bbox, config=config, radius_key="r"), clip)
        if bbox is not None:
            mask_draw.ellipse(bbox, fill=76)
            _apply_region_mask(image, mask, color=_REGION_FILL)
            draw.ellipse(bbox, outline=_REGION_OUTLINE, width=2)
        return
    if str(region.kind) == "annulus":
        outer = _clip_bbox(_panel_region_bbox(region, plot_bbox=plot_bbox, config=config, radius_key="outer_r"), clip)
        inner = _clip_bbox(_panel_region_bbox(region, plot_bbox=plot_bbox, config=config, radius_key="inner_r"), clip)
        if outer is not None:
            mask_draw.ellipse(outer, fill=76)
            if inner is not None:
                mask_draw.ellipse(inner, fill=0)
            _apply_region_mask(image, mask, color=_REGION_FILL)
            draw.ellipse(outer, outline=_REGION_OUTLINE, width=2)
            if inner is not None:
                draw.ellipse(inner, outline=_REGION_OUTLINE, width=2)
        return
    x_min, y_min, x_max, y_max = _constraint_rect(region, grid_min=int(config.grid_min), grid_max=int(config.grid_max))
    top_left = graph_point_to_panel_pixel((x_min, y_max), plot_bbox=plot_bbox, config=config)
    bottom_right = graph_point_to_panel_pixel((x_max, y_min), plot_bbox=plot_bbox, config=config)
    rect = _clip_bbox((top_left[0], top_left[1], bottom_right[0], bottom_right[1]), clip)
    if rect is None:
        return
    mask_draw.rectangle(rect, fill=74)
    _apply_region_mask(image, mask, color=_REGION_FILL)
    params = dict(region.params)
    for key in ("x_min", "x_max"):
        if key not in params:
            continue
        x_value = int(params[key])
        x_px, _ = graph_point_to_panel_pixel((x_value, 0), plot_bbox=plot_bbox, config=config)
        draw.line([(float(x_px), clip[1]), (float(x_px), clip[3])], fill=_BOUNDARY_DASH_FILL, width=2)
    for key in ("y_min", "y_max"):
        if key not in params:
            continue
        _, y_px = graph_point_to_panel_pixel((0, int(params[key])), plot_bbox=plot_bbox, config=config)
        draw.line([(clip[0], float(y_px)), (clip[2], float(y_px))], fill=_BOUNDARY_DASH_FILL, width=2)


def _draw_condition_box(
    draw: ImageDraw.ImageDraw,
    *,
    canvas_width: int,
    text: str,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> List[int]:
    font_size = _resolve_int_param(params, rendering_defaults, "locus_condition_label_font_size", _DEFAULTS.condition_label_font_size)
    label_text = f"Condition: {text}"
    font = symbol_safe_font_for_text(label_text, load_font(max(12, int(font_size)), bold=True))
    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    pad_x = 12
    pad_y = 6
    width = int(text_bbox[2] - text_bbox[0]) + (2 * pad_x)
    height = int(text_bbox[3] - text_bbox[1]) + (2 * pad_y)
    left = int(round((int(canvas_width) - width) / 2.0))
    top = 10
    box = [left, top, left + width, top + height]
    draw.rounded_rectangle(box, radius=6, fill=(255, 255, 255), outline=(82, 96, 116), width=2)
    draw_text_traced(draw,(left + pad_x, top + pad_y), label_text, fill=(30, 43, 62), font=font, role="readout", required=False)
    return [int(value) for value in box]


def _render_panel_scene(
    query: _ResolvedQuery,
    *,
    namespace: str,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> _PanelScene:
    """Render the option panel grid and unique matching condition panel."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    panel_count, option_count_probabilities = resolve_geometry_option_count(
        params=params,
        gen_defaults=generation_defaults,
        field_name="locus_panel_count",
        supported_counts=(4, 6),
        task_id=str(namespace),
        instance_seed=int(instance_seed),
    )
    if int(panel_count) > len(query.label_pool):
        raise ValueError("locus_panel_count cannot exceed panel label pool length")
    label_pool = tuple(query.label_pool[: int(panel_count)])
    if str(query.winner_label) not in set(label_pool):
        raise ValueError("winner_label must be inside the active contiguous panel label set")

    target_region = _panel_target_region(str(query.operation_key), rng)
    distractors = [
        region
        for region in _panel_distractor_regions(str(query.operation_key))
        if not _same_region(region, target_region)
    ]
    if len(distractors) < int(panel_count) - 1:
        raise RuntimeError("locus panel distractor bank is too small for the selected panel count")
    rng.shuffle(distractors)
    region_by_label: Dict[str, _RegionSpec] = {}
    distractor_iter = iter(distractors)
    for label in label_pool:
        region_by_label[str(label)] = target_region if str(label) == str(query.winner_label) else next(distractor_iter)

    columns, rows = panel_grid_shape_for_option_count(int(panel_count))
    panel_config = CoordinatePanelConfig(
        grid_min=_resolve_int_param(params, rendering_defaults, "locus_panel_grid_min", _DEFAULTS.panel_grid_min),
        grid_max=_resolve_int_param(params, rendering_defaults, "locus_panel_grid_max", _DEFAULTS.panel_grid_max),
        columns=int(columns),
        rows=int(rows),
    )
    if int(panel_count) == 4:
        canvas_width = _resolve_int_param(
            params,
            rendering_defaults,
            "locus_panel_canvas_width_4",
            _DEFAULTS.panel_canvas_width_4,
        )
        canvas_height = _resolve_int_param(
            params,
            rendering_defaults,
            "locus_panel_canvas_height_4",
            _DEFAULTS.panel_canvas_height_4,
        )
    else:
        canvas_width = _resolve_int_param(params, rendering_defaults, "locus_panel_canvas_width", _DEFAULTS.panel_canvas_width)
        canvas_height = _resolve_int_param(params, rendering_defaults, "locus_panel_canvas_height", _DEFAULTS.panel_canvas_height)
    top_reserved = _resolve_int_param(params, rendering_defaults, "locus_panel_top_reserved_px", _DEFAULTS.panel_top_reserved_px)
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        scene_id=SCENE_ID,
        instance_seed=int(instance_seed),
        params=params,
        style_profile=GEOMETRY_STYLE_PROFILE_COORDINATE_GRID,
        namespace_suffix="coordinate_plane_locus_panel_background",
    )
    draw = ImageDraw.Draw(image)
    condition_label_bbox = _draw_condition_box(
        draw,
        canvas_width=int(canvas_width),
        text=str(target_region.condition_text),
        params=params,
        rendering_defaults=rendering_defaults,
    )
    style = geometry_coordinate_panel_style_from_diagram_style(diagram_style)
    panel_style_meta = {
        "style": style.to_trace_dict(),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_meta),
    }
    layout = coordinate_panel_layout(
        int(canvas_width),
        max(320, int(canvas_height) - int(top_reserved)),
        config=panel_config,
    )
    layout["margin_y"] = int(layout["margin_y"]) + int(top_reserved)
    panels_by_label: Dict[str, _PanelSpec] = {}
    for index, label in enumerate(label_pool):
        panel_bbox = panel_bbox_for_index(layout, int(index), config=panel_config)
        plot_bbox = plot_bbox_for_panel(panel_bbox)
        draw_coordinate_panel_grid(
            draw,
            panel_bbox=panel_bbox,
            plot_bbox=plot_bbox,
            label=str(label),
            config=panel_config,
            style=style,
        )
        _draw_panel_region(
            image,
            draw,
            region=region_by_label[str(label)],
            plot_bbox=plot_bbox,
            config=panel_config,
        )
        panels_by_label[str(label)] = _PanelSpec(
            label=str(label),
            region=region_by_label[str(label)],
            panel_bbox=[int(value) for value in panel_bbox],
            plot_bbox=[int(value) for value in plot_bbox],
            is_answer=str(label) == str(query.winner_label),
        )
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    return _PanelScene(
        condition_text=str(target_region.condition_text),
        panels_by_label=dict(panels_by_label),
        condition_label_bbox_px=list(condition_label_bbox),
        panel_style_meta=dict(panel_style_meta),
        marker_meta={
            "region_fill": list(_REGION_FILL),
            "region_outline": list(_REGION_OUTLINE),
            "region_fill_alpha": 76,
        },
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        option_count_probabilities=dict(option_count_probabilities),
    )


def _region_trace(region: _RegionSpec) -> Dict[str, Any]:
    return {
        "kind": str(region.kind),
        "params": {str(key): int(value) for key, value in region.params.items()},
        "condition_text": str(region.condition_text),
    }




def _point_trace_payload(
    *,
    query: _ResolvedQuery,
    rendered: _PointScene,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_value: List[float],
) -> Dict[str, Any]:
    """Assemble trace payload for a scalar point locus answer."""

    candidate_trace = {
        str(label): {
            "point_graph": [int(value) for value in rendered.candidate_points_by_label[str(label)]],
            "point_px": [float(value) for value in rendered.candidate_points_px_by_label[str(label)]],
            "bbox_px": list(rendered.candidate_bboxes_by_label[str(label)]),
            "inside_region": bool(_region_contains(rendered.region, rendered.candidate_points_by_label[str(label)])),
            "is_answer": str(label) == str(query.winner_label),
        }
        for label in sorted(rendered.candidate_points_by_label)
    }
    return {
        "scene_ir": {
            "scene_kind": "geometry_coordinate_locus_region",
            "entities": [
                {
                    "entity_type": "candidate_point",
                    "label": str(label),
                    **dict(payload),
                }
                for label, payload in candidate_trace.items()
            ],
            "relations": {
                "scene_id": SCENE_ID,
                "operation_key": str(query.operation_key),
                "operation_key_probabilities": dict(query.query_probabilities),
                "winner_label": str(query.winner_label),
                "region": _region_trace(rendered.region),
            },
        },
        "query_spec": {
            "operation_key": str(query.operation_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "scene_id": SCENE_ID,
                "operation_key": str(query.operation_key),
                "operation_key_probabilities": dict(query.query_probabilities),
                "winner_label": str(query.winner_label),
                "winner_label_probabilities": dict(query.winner_label_probabilities),
                "candidate_label_pool": list(query.label_pool),
                "locus_candidate_count_probabilities": dict(rendered.option_count_probabilities),
            },
        },
        "render_spec": {
            **dict(rendered.render_spec_extra),
            "scene_id": SCENE_ID,
            "marker_style": dict(rendered.marker_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
            "background_style": dict(rendered.background_meta),
            "locus_candidate_count_probabilities": dict(rendered.option_count_probabilities),
        },
        "render_map": {
            "coord_space": "pixel",
            "candidate_points_graph_by_label": {
                str(label): [int(value) for value in point] for label, point in rendered.candidate_points_by_label.items()
            },
            "candidate_points_px_by_label": {
                str(label): [float(value) for value in point] for label, point in rendered.candidate_points_px_by_label.items()
            },
            "candidate_bboxes_px_by_label": dict(rendered.candidate_bboxes_by_label),
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "operation_key": str(query.operation_key),
            "answer_type": "option_letter",
            "answer_value": str(query.winner_label),
            "region": _region_trace(rendered.region),
            "center_point_px": list(rendered.center_point_px) if rendered.center_point_px is not None else None,
            "candidate_points_by_label": dict(candidate_trace),
            "operation_key_probabilities": dict(query.query_probabilities),
            "locus_candidate_count_probabilities": dict(rendered.option_count_probabilities),
        },
        "witness_symbolic": {
            "type": "coordinate_locus_point_membership",
            "answer_label": str(query.winner_label),
            "region": _region_trace(rendered.region),
            "candidate_points_by_label": dict(candidate_trace),
        },
        "projected_annotation": {
            "type": "point",
            "point": list(annotation_value),
            "pixel_point": list(annotation_value),
            "candidate_points_px_by_label": {
                str(label): [float(value) for value in point]
                for label, point in rendered.candidate_points_px_by_label.items()
            },
        },
    }


def _panel_trace_payload(
    *,
    query: _ResolvedQuery,
    rendered: _PanelScene,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    annotation_value: List[int],
) -> Dict[str, Any]:
    """Assemble trace payload for a scalar panel bounding-box answer."""

    panels_trace = {
        str(label): {
            "label": str(label),
            "region": _region_trace(spec.region),
            "panel_bbox": list(spec.panel_bbox),
            "plot_bbox": list(spec.plot_bbox),
            "is_answer": bool(spec.is_answer),
        }
        for label, spec in rendered.panels_by_label.items()
    }

    return {
        "scene_ir": {
            "scene_kind": "geometry_coordinate_locus_panel_grid",
            "entities": [dict(panels_trace[str(label)]) for label in sorted(panels_trace)],
            "relations": {
                "scene_id": SCENE_ID,
                "operation_key": str(query.operation_key),
                "operation_key_probabilities": dict(query.query_probabilities),
                "winner_label": str(query.winner_label),
                "condition_text": str(rendered.condition_text),
            },
        },
        "query_spec": {
            "operation_key": str(query.operation_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "scene_id": SCENE_ID,
                "operation_key": str(query.operation_key),
                "operation_key_probabilities": dict(query.query_probabilities),
                "winner_label": str(query.winner_label),
                "winner_label_probabilities": dict(query.winner_label_probabilities),
                "candidate_label_pool": list(query.label_pool),
                "locus_panel_count_probabilities": dict(rendered.option_count_probabilities),
            },
        },
        "render_spec": {
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "coord_space": "pixel",
            "scene_id": SCENE_ID,
            "panel_count": int(len(rendered.panels_by_label)),
            "locus_panel_count_probabilities": dict(rendered.option_count_probabilities),
            "condition_label_bbox_px": list(rendered.condition_label_bbox_px),
            "panel_style": dict(rendered.panel_style_meta),
            "marker_style": dict(rendered.marker_meta),
            "background_style": dict(rendered.background_meta),
            "post_image_noise": dict(rendered.post_noise_meta),
        },
        "render_map": {
            "coord_space": "pixel",
            "panel_bboxes": {str(label): list(spec.panel_bbox) for label, spec in rendered.panels_by_label.items()},
            "plot_bboxes": {str(label): list(spec.plot_bbox) for label, spec in rendered.panels_by_label.items()},
        },
        "execution_trace": {
            "scene_id": SCENE_ID,
            "operation_key": str(query.operation_key),
            "answer_type": "option_letter",
            "answer_value": str(query.winner_label),
            "condition_text": str(rendered.condition_text),
            "panels_by_label": dict(panels_trace),
            "operation_key_probabilities": dict(query.query_probabilities),
            "locus_panel_count_probabilities": dict(rendered.option_count_probabilities),
        },
        "witness_symbolic": {
            "type": "coordinate_locus_panel_match",
            "answer_label": str(query.winner_label),
            "condition_text": str(rendered.condition_text),
            "panels_by_label": dict(panels_trace),
        },
        "projected_annotation": {
            "type": "bbox",
            "bbox": list(annotation_value),
            "panel_bbox_by_label": {str(label): list(spec.panel_bbox) for label, spec in rendered.panels_by_label.items()},
        },
    }


def _candidate_point_annotation(rendered: _PointScene, label: str) -> List[float]:
    point = rendered.candidate_points_px_by_label[str(label)]
    return [float(point[0]), float(point[1])]


@dataclass(frozen=True)
class LocusPointArtifacts:
    query: _ResolvedQuery
    rendered: _PointScene
    prompt_artifacts: Any
    annotation_value: List[float]
    trace_payload: Dict[str, Any]


@dataclass(frozen=True)
class LocusPanelArtifacts:
    query: _ResolvedQuery
    rendered: _PanelScene
    prompt_artifacts: Any
    annotation_value: List[int]
    trace_payload: Dict[str, Any]


def build_locus_point_artifacts(
    *,
    namespace: str,
    config_key: str,
    semantic_operation_key: str,
    semantic_query_probabilities: Mapping[str, float],
    prompt_query_key: str,
    winner_label: str,
    winner_label_probabilities: Mapping[str, float],
    label_pool: Sequence[str],
    instance_seed: int,
    params: Dict[str, Any],
) -> LocusPointArtifacts:
    """Resolve render, prompt, scalar annotation, and trace for point selection."""

    generation_defaults, rendering_defaults, prompt_defaults_all = _split_defaults_for_task(str(config_key))
    query = _ResolvedQuery(
        operation_key=str(semantic_operation_key),
        query_probabilities={str(key): float(value) for key, value in semantic_query_probabilities.items()},
        winner_label=str(winner_label),
        winner_label_probabilities={str(key): float(value) for key, value in winner_label_probabilities.items()},
        label_pool=tuple(str(label) for label in label_pool),
    )
    rendered = _render_point_scene(
        query,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
    )
    annotation_value = _candidate_point_annotation(rendered, str(query.winner_label))
    prompt_defaults, prompt_artifacts = build_option_letter_prompt_artifacts(
        prompt_defaults_all=prompt_defaults_all,
        config_key=str(config_key),
        scene_key_fallback=SCENE_ID,
        prompt_query_key=str(prompt_query_key),
        annotation_hint_key="annotation_hint_candidate_point",
        annotation_value=annotation_value,
        instance_seed=int(instance_seed),
    )
    trace_payload = _point_trace_payload(
        query=query,
        rendered=rendered,
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
        annotation_value=annotation_value,
    )
    return LocusPointArtifacts(
        query=query,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation_value=annotation_value,
        trace_payload=trace_payload,
    )


def build_locus_panel_artifacts(
    *,
    namespace: str,
    config_key: str,
    semantic_operation_key: str,
    semantic_query_probabilities: Mapping[str, float],
    prompt_query_key: str,
    winner_label: str,
    winner_label_probabilities: Mapping[str, float],
    label_pool: Sequence[str],
    instance_seed: int,
    params: Dict[str, Any],
) -> LocusPanelArtifacts:
    """Resolve render, prompt, scalar bbox annotation, and trace for panel matching."""

    generation_defaults, rendering_defaults, prompt_defaults_all = _split_defaults_for_task(str(config_key))
    query = _ResolvedQuery(
        operation_key=str(semantic_operation_key),
        query_probabilities={str(key): float(value) for key, value in semantic_query_probabilities.items()},
        winner_label=str(winner_label),
        winner_label_probabilities={str(key): float(value) for key, value in winner_label_probabilities.items()},
        label_pool=tuple(str(label) for label in label_pool),
    )
    rendered = _render_panel_scene(
        query,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
    )
    annotation_value = list(rendered.panels_by_label[str(query.winner_label)].panel_bbox)
    prompt_defaults, prompt_artifacts = build_option_letter_prompt_artifacts(
        prompt_defaults_all=prompt_defaults_all,
        config_key=str(config_key),
        scene_key_fallback=SCENE_ID,
        prompt_query_key=str(prompt_query_key),
        annotation_hint_key="annotation_hint_selected_panel_bbox",
        annotation_value=annotation_value,
        instance_seed=int(instance_seed),
    )
    trace_payload = _panel_trace_payload(
        query=query,
        rendered=rendered,
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
        annotation_value=annotation_value,
    )
    return LocusPanelArtifacts(
        query=query,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        annotation_value=annotation_value,
        trace_payload=trace_payload,
    )
