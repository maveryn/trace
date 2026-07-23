"""Rendering primitives for coordinate-panel scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
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
from trace_tasks.tasks.geometry.shared.option_count import panel_grid_shape_for_option_count

from .defaults import resolve_int_param
from .construction import (
    classify_point_set,
    is_ambiguous_for_prompt,
    sample_panel_points,
    sample_segment_pair,
    sample_transform_panel_points,
    shape_distractor_kinds,
)
from .state import (
    Color,
    PanelDefaults,
    PanelScene,
    PanelSpec,
    PixelPoint,
    SegmentPanelScene,
    SegmentPanelSpec,
    TransformPanelScene,
    TransformPanelSpec,
)

SCENE_ID = "coordinate_panels"

MARKER_STYLES: Tuple[str, ...] = ("filled_circle", "ring", "cross", "diamond", "square")
MARKER_COLOR_PALETTES: Tuple[Tuple[Color, Color], ...] = (
    ((32, 92, 166), (206, 92, 38)),
    ((38, 123, 96), (129, 77, 172)),
    ((19, 119, 150), (190, 67, 104)),
    ((97, 100, 36), (178, 83, 43)),
    ((54, 94, 132), (185, 104, 28)),
    ((105, 76, 151), (34, 130, 121)),
)
DEFAULTS = PanelDefaults()


def _sample_marker_style(rng, *, params: Mapping[str, Any], defaults: Mapping[str, Any], key: str) -> str:
    explicit = params.get(str(key), defaults.get(str(key), None))
    if explicit is not None:
        style = str(explicit)
        if style not in set(MARKER_STYLES):
            raise ValueError(f"{key}={style!r} is not supported")
        return style
    return str(rng.choice(MARKER_STYLES))


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    point: PixelPoint,
    *,
    style: str,
    color: Color,
    radius: int,
    outline: Color = (255, 255, 255),
    width: int = 2,
) -> None:
    x_value, y_value = float(point[0]), float(point[1])
    radius_px = max(2, int(radius))
    line_width = max(1, int(width))
    fill = tuple(int(value) for value in color)
    stroke = tuple(int(value) for value in outline)
    if str(style) == "ring":
        draw.ellipse([x_value - radius_px, y_value - radius_px, x_value + radius_px, y_value + radius_px], fill=stroke, outline=fill, width=line_width)
    elif str(style) == "cross":
        draw.line([(x_value - radius_px, y_value - radius_px), (x_value + radius_px, y_value + radius_px)], fill=fill, width=line_width)
        draw.line([(x_value - radius_px, y_value + radius_px), (x_value + radius_px, y_value - radius_px)], fill=fill, width=line_width)
    elif str(style) == "diamond":
        draw.polygon(
            [(x_value, y_value - radius_px), (x_value + radius_px, y_value), (x_value, y_value + radius_px), (x_value - radius_px, y_value)],
            fill=fill,
            outline=stroke,
        )
    elif str(style) == "square":
        draw.rectangle(
            [x_value - radius_px, y_value - radius_px, x_value + radius_px, y_value + radius_px],
            fill=fill,
            outline=stroke,
            width=line_width,
        )
    else:
        draw.ellipse([x_value - radius_px, y_value - radius_px, x_value + radius_px, y_value + radius_px], fill=fill, outline=stroke, width=line_width)


def _resolve_marker_colors(rng) -> Tuple[Color, Color, Dict[str, Any]]:
    known_color, candidate_color = rng.choice(MARKER_COLOR_PALETTES)
    return tuple(known_color), tuple(candidate_color), {
        "palette": [list(known_color), list(candidate_color)],
        "known_color": list(known_color),
        "candidate_color": list(candidate_color),
    }


def _visible_panel_labels(
    *,
    label_pool: Sequence[str],
    panel_count: int,
    winner_label: str,
) -> Tuple[str, ...]:
    if int(panel_count) > len(tuple(label_pool)):
        raise ValueError("panel_count cannot exceed panel label pool length")
    visible_labels = tuple(str(label) for label in tuple(label_pool)[: int(panel_count)])
    if str(winner_label) not in set(visible_labels):
        visible_labels = tuple([str(winner_label), *[label for label in visible_labels if label != str(winner_label)]])
        visible_labels = visible_labels[: int(panel_count)]
    return tuple(visible_labels)


def _panel_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    label_pool: Sequence[str],
    panel_count: int,
    winner_label: str,
) -> Tuple[Any, Tuple[str, ...], CoordinatePanelConfig, Any, ImageDraw.ImageDraw, Any, Dict[str, Any], Dict[str, Any], str, Color, Color, int, Dict[str, Any], int, int]:
    """Resolve shared canvas, grid, style, color, and marker state for all panel variants."""

    rng = spawn_rng(int(instance_seed), "coordinate_panels.panel_scene")
    max_abs = resolve_int_param(params, generation_defaults, "panel_graph_abs_max", DEFAULTS.panel_graph_abs_max)
    visible_labels = _visible_panel_labels(label_pool=label_pool, panel_count=int(panel_count), winner_label=str(winner_label))
    columns, rows = panel_grid_shape_for_option_count(int(panel_count))
    panel_config = CoordinatePanelConfig(
        grid_min=resolve_int_param(params, rendering_defaults, "panel_grid_min", DEFAULTS.panel_grid_min),
        grid_max=resolve_int_param(params, rendering_defaults, "panel_grid_max", DEFAULTS.panel_grid_max),
        columns=int(columns),
        rows=int(rows),
    )
    canvas_width = resolve_int_param(params, rendering_defaults, "panel_canvas_width", DEFAULTS.panel_canvas_width)
    canvas_height = resolve_int_param(params, rendering_defaults, "panel_canvas_height", DEFAULTS.panel_canvas_height)
    image, background_meta, diagram_style, diagram_style_meta = prepare_geometry_diagram_style_and_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        scene_id=SCENE_ID,
        instance_seed=int(instance_seed),
        params=params,
        style_profile=GEOMETRY_STYLE_PROFILE_COORDINATE_GRID,
        namespace_suffix="coordinate_panels_background",
    )
    draw = ImageDraw.Draw(image)
    style = geometry_coordinate_panel_style_from_diagram_style(diagram_style)
    panel_style_meta = {
        "style": style.to_trace_dict(),
        "technical_diagram_style": geometry_diagram_style_metadata(diagram_style),
        "technical_diagram_style_resolution": dict(diagram_style_meta),
    }
    marker_style = _sample_marker_style(rng, params=params, defaults=rendering_defaults, key="panel_marker_style")
    known_color, marker_color, color_meta = _resolve_marker_colors(rng)
    marker_radius = resolve_int_param(params, rendering_defaults, "panel_marker_radius_px", DEFAULTS.panel_marker_radius_px)
    marker_radius = max(
        resolve_int_param(params, rendering_defaults, "panel_marker_radius_px_min", DEFAULTS.panel_marker_radius_px_min),
        min(
            resolve_int_param(params, rendering_defaults, "panel_marker_radius_px_max", DEFAULTS.panel_marker_radius_px_max),
            int(marker_radius),
        ),
    )
    return (
        rng,
        visible_labels,
        panel_config,
        image,
        draw,
        style,
        panel_style_meta,
        background_meta,
        marker_style,
        known_color,
        marker_color,
        marker_radius,
        color_meta,
        int(canvas_width),
        int(canvas_height),
    )


def render_panel_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    target_kind: str,
    winner_label: str,
    label_pool: Sequence[str],
    panel_count: int,
    option_count_probabilities: Mapping[str, float],
    noise_defaults: Mapping[str, Any],
) -> PanelScene:
    """Render labeled panels after the public task resolves the target shape and answer label."""

    (
        rng,
        visible_labels,
        panel_config,
        image,
        draw,
        style,
        panel_style_meta,
        background_meta,
        marker_style,
        _,
        marker_color,
        marker_radius,
        color_meta,
        canvas_width,
        canvas_height,
    ) = _panel_render_context(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        label_pool=label_pool,
        panel_count=int(panel_count),
        winner_label=str(winner_label),
    )
    max_abs = resolve_int_param(params, generation_defaults, "panel_graph_abs_max", DEFAULTS.panel_graph_abs_max)
    distractor_kinds = shape_distractor_kinds(str(target_kind), rng=rng, count=int(panel_count) - 1)
    kind_by_label: Dict[str, str] = {}
    distractor_iter = iter(distractor_kinds)
    for label in visible_labels:
        kind_by_label[str(label)] = str(target_kind) if str(label) == str(winner_label) else str(next(distractor_iter))

    layout = coordinate_panel_layout(int(canvas_width), int(canvas_height), config=panel_config)
    panels_by_label: Dict[str, PanelSpec] = {}
    for index, label in enumerate(visible_labels):
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
        points = sample_panel_points(str(kind_by_label[str(label)]), rng=rng, max_abs=int(max_abs))
        classified_kind = classify_point_set(points)
        if str(kind_by_label[str(label)]) != "other" and str(classified_kind) != str(kind_by_label[str(label)]):
            raise RuntimeError(f"panel {label} sampled as {kind_by_label[str(label)]} but classified {classified_kind}")
        if str(label) != str(winner_label) and is_ambiguous_for_prompt(str(classified_kind), str(target_kind)):
            raise RuntimeError("sampled ambiguous distractor panel")
        points_px = tuple(
            graph_point_to_panel_pixel(point, plot_bbox=plot_bbox, config=panel_config)
            for point in points
        )
        for point_px in points_px:
            _draw_marker(
                draw,
                point_px,
                style=str(marker_style),
                color=marker_color,
                radius=int(marker_radius),
                width=2,
            )
        panels_by_label[str(label)] = PanelSpec(
            label=str(label),
            points=tuple(points),
            points_px=tuple((float(point[0]), float(point[1])) for point in points_px),
            classified_kind=str(classified_kind),
            panel_bbox=[int(value) for value in panel_bbox],
            plot_bbox=[int(value) for value in plot_bbox],
        )

    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    marker_meta = {
        "panel_marker_style": str(marker_style),
        "panel_marker_color": list(marker_color),
        "panel_marker_radius_px": int(marker_radius),
        "color_selection": dict(color_meta),
    }
    return PanelScene(
        panels_by_label=dict(panels_by_label),
        marker_meta=dict(marker_meta),
        panel_style_meta=dict(panel_style_meta),
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        option_count_probabilities={str(key): float(value) for key, value in option_count_probabilities.items()},
    )


def render_segment_panel_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    relation_kind: str,
    winner_label: str,
    label_pool: Sequence[str],
    panel_count: int,
    option_count_probabilities: Mapping[str, float],
    noise_defaults: Mapping[str, Any],
) -> SegmentPanelScene:
    """Render labeled panels containing two relation-tested line segments."""

    (
        rng,
        visible_labels,
        panel_config,
        image,
        draw,
        style,
        panel_style_meta,
        background_meta,
        marker_style,
        first_color,
        second_color,
        marker_radius,
        color_meta,
        canvas_width,
        canvas_height,
    ) = _panel_render_context(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        label_pool=label_pool,
        panel_count=int(panel_count),
        winner_label=str(winner_label),
    )
    max_abs = resolve_int_param(params, generation_defaults, "panel_graph_abs_max", DEFAULTS.panel_graph_abs_max)
    layout = coordinate_panel_layout(int(canvas_width), int(canvas_height), config=panel_config)
    panels_by_label: Dict[str, SegmentPanelSpec] = {}
    for index, label in enumerate(visible_labels):
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
        should_match = str(label) == str(winner_label)
        segments = sample_segment_pair(str(relation_kind), rng=rng, max_abs=int(max_abs), should_match=bool(should_match))
        segments_px = tuple(
            (
                graph_point_to_panel_pixel(segment[0], plot_bbox=plot_bbox, config=panel_config),
                graph_point_to_panel_pixel(segment[1], plot_bbox=plot_bbox, config=panel_config),
            )
            for segment in segments
        )
        for segment_index, segment_px in enumerate(segments_px):
            color = first_color if int(segment_index) == 0 else second_color
            draw.line([tuple(segment_px[0]), tuple(segment_px[1])], fill=color, width=4)
            for point_px in segment_px:
                _draw_marker(
                    draw,
                    point_px,
                    style=str(marker_style),
                    color=color,
                    radius=int(marker_radius),
                    width=2,
                )
        flags = {
            "parallel": bool((segments[0][1][0] - segments[0][0][0]) * (segments[1][1][1] - segments[1][0][1]) == (segments[0][1][1] - segments[0][0][1]) * (segments[1][1][0] - segments[1][0][0])),
            "perpendicular": bool((segments[0][1][0] - segments[0][0][0]) * (segments[1][1][0] - segments[1][0][0]) + (segments[0][1][1] - segments[0][0][1]) * (segments[1][1][1] - segments[1][0][1]) == 0),
            "equal_length": bool(
                (segments[0][1][0] - segments[0][0][0]) ** 2 + (segments[0][1][1] - segments[0][0][1]) ** 2
                == (segments[1][1][0] - segments[1][0][0]) ** 2 + (segments[1][1][1] - segments[1][0][1]) ** 2
            ),
        }
        panels_by_label[str(label)] = SegmentPanelSpec(
            label=str(label),
            segments_graph=tuple(segments),
            segments_px=tuple(
                (
                    (float(segment[0][0]), float(segment[0][1])),
                    (float(segment[1][0]), float(segment[1][1])),
                )
                for segment in segments_px
            ),
            relation_flags=dict(flags),
            panel_bbox=[int(value) for value in panel_bbox],
            plot_bbox=[int(value) for value in plot_bbox],
        )

    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    marker_meta = {
        "panel_marker_style": str(marker_style),
        "segment_colors": {"first": list(first_color), "second": list(second_color)},
        "panel_marker_radius_px": int(marker_radius),
        "color_selection": dict(color_meta),
    }
    return SegmentPanelScene(
        panels_by_label=dict(panels_by_label),
        marker_meta=dict(marker_meta),
        panel_style_meta=dict(panel_style_meta),
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        option_count_probabilities={str(key): float(value) for key, value in option_count_probabilities.items()},
    )


def render_quadrilateral_panel_task_scene(*, query: Any, **kwargs: Any) -> PanelScene:
    """Adapt resolved query state to the quadrilateral panel renderer."""

    return render_panel_scene(
        target_kind=str(query.kind_value),
        winner_label=str(query.winner_label),
        label_pool=query.label_pool,
        panel_count=int(query.panel_count),
        option_count_probabilities=query.panel_count_probabilities,
        **kwargs,
    )


def render_segment_relation_panel_task_scene(*, query: Any, **kwargs: Any) -> SegmentPanelScene:
    """Adapt resolved query state to the segment-relation panel renderer."""

    return render_segment_panel_scene(
        relation_kind=str(query.kind_value),
        winner_label=str(query.winner_label),
        label_pool=query.label_pool,
        panel_count=int(query.panel_count),
        option_count_probabilities=query.panel_count_probabilities,
        **kwargs,
    )


def render_transform_panel_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    transform_kind: str,
    winner_label: str,
    label_pool: Sequence[str],
    panel_count: int,
    point_count: int,
    option_count_probabilities: Mapping[str, float],
    noise_defaults: Mapping[str, Any],
) -> TransformPanelScene:
    """Render labeled panels containing source and candidate transformed point sets."""

    (
        rng,
        visible_labels,
        panel_config,
        image,
        draw,
        style,
        panel_style_meta,
        background_meta,
        marker_style,
        source_color,
        candidate_color,
        marker_radius,
        color_meta,
        canvas_width,
        canvas_height,
    ) = _panel_render_context(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        label_pool=label_pool,
        panel_count=int(panel_count),
        winner_label=str(winner_label),
    )
    max_abs = resolve_int_param(params, generation_defaults, "panel_graph_abs_max", DEFAULTS.panel_graph_abs_max)
    layout = coordinate_panel_layout(int(canvas_width), int(canvas_height), config=panel_config)
    panels_by_label: Dict[str, TransformPanelSpec] = {}
    for index, label in enumerate(visible_labels):
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
        should_match = str(label) == str(winner_label)
        source_points, candidate_points, _, flags = sample_transform_panel_points(
            str(transform_kind),
            rng=rng,
            max_abs=int(max_abs),
            point_count=int(point_count),
            should_match=bool(should_match),
        )
        source_points_px = tuple(
            graph_point_to_panel_pixel(point, plot_bbox=plot_bbox, config=panel_config)
            for point in source_points
        )
        candidate_points_px = tuple(
            graph_point_to_panel_pixel(point, plot_bbox=plot_bbox, config=panel_config)
            for point in candidate_points
        )
        for point_px in source_points_px:
            _draw_marker(
                draw,
                point_px,
                style="filled_circle",
                color=source_color,
                radius=int(marker_radius),
                width=2,
            )
        for point_px in candidate_points_px:
            _draw_marker(
                draw,
                point_px,
                style=str(marker_style),
                color=candidate_color,
                radius=int(marker_radius),
                width=2,
            )
        panels_by_label[str(label)] = TransformPanelSpec(
            label=str(label),
            source_points_graph=tuple(source_points),
            candidate_points_graph=tuple(candidate_points),
            source_points_px=tuple((float(point[0]), float(point[1])) for point in source_points_px),
            candidate_points_px=tuple((float(point[0]), float(point[1])) for point in candidate_points_px),
            transform_flags=dict(flags),
            panel_bbox=[int(value) for value in panel_bbox],
            plot_bbox=[int(value) for value in plot_bbox],
        )

    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    marker_meta = {
        "source_marker_style": "filled_circle",
        "candidate_marker_style": str(marker_style),
        "source_marker_color": list(source_color),
        "candidate_marker_color": list(candidate_color),
        "panel_marker_radius_px": int(marker_radius),
        "color_selection": dict(color_meta),
    }
    return TransformPanelScene(
        panels_by_label=dict(panels_by_label),
        marker_meta=dict(marker_meta),
        panel_style_meta=dict(panel_style_meta),
        image=image,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        option_count_probabilities={str(key): float(value) for key, value in option_count_probabilities.items()},
    )


def render_point_set_transform_panel_task_scene(*, query: Any, **kwargs: Any) -> TransformPanelScene:
    """Adapt resolved query state to the point-set transform panel renderer."""

    return render_transform_panel_scene(
        transform_kind=str(query.kind_value),
        winner_label=str(query.winner_label),
        label_pool=query.label_pool,
        panel_count=int(query.panel_count),
        point_count=int(query.extra_axes["point_count"]),
        option_count_probabilities=query.panel_count_probabilities,
        **kwargs,
    )


__all__ = [
    "render_panel_scene",
    "render_point_set_transform_panel_task_scene",
    "render_quadrilateral_panel_task_scene",
    "render_segment_panel_scene",
    "render_segment_relation_panel_task_scene",
    "render_transform_panel_scene",
]
