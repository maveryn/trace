"""Geometry graphing average-rate task over marked points on a plotted function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import ImageDraw

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)

SCENE_ID = "function_graph"
from ...shared.drawing import draw_centered_text
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.text_rendering import load_font, resolve_scene_label_font_size_px
from ..shared.background_defaults import load_geometry_background_defaults
from .shared.projection import draw_function_polyline, graph_units_to_pixel_float
from ..shared.graph_rendering import graph_paper_grid_from_frame
from ..shared.noise_defaults import load_geometry_noise_defaults
from ..shared.shape_style import (
    extract_background_anchor_colors,
    sample_geometry_shape_style,
)
from ..shared.single_object_scene import (
    GraphSceneContext,
    finalize_graph_scene_image,
    make_graph_scene_canvas,
    resolve_graph_scene_context,
)
from .shared.prompts import prompt_asset_slot
from .shared.sampling import resolve_rate_target


TASK_ID = "task_geometry__function_graph__average_rate_value"
AVERAGE_RATE_BETWEEN_MARKED_POINTS = "average_rate_between_marked_points"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("single",)
DEFAULT_RATE_SUPPORT: Tuple[float, ...] = (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0)

POST_IMAGE_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)

GraphPoint = Tuple[float, float]


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallbacks for the average-rate graphing task."""

    canvas_size_min: int = 660
    canvas_size_max: int = 740
    graph_cells_min: int = 20
    graph_cells_max: int = 20
    line_width: int = 4
    marker_radius: int = 7
    label_font_size_min: int = 18
    label_font_size_max: int = 26
    average_rate_support: Tuple[float, ...] = DEFAULT_RATE_SUPPORT


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved query and balanced average-rate target."""

    query_id: str
    target_rate: float
    query_id_probabilities: Dict[str, float]
    target_rate_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _SampledRateScene:
    """Sampled plotted-function scene before raster rendering."""

    polyline_graph: Tuple[GraphPoint, ...]
    point_a: GraphPoint
    point_b: GraphPoint
    answer_value: float
    scene_entities: list[Dict[str, Any]]
    render_map: Dict[str, Any]
    execution_trace: Dict[str, Any]


@dataclass(frozen=True)
class _RenderedRateScene:
    """Rendered scene and metadata-backed annotation artifacts."""

    answer_value: float
    annotation_type: str
    annotation_value: Dict[str, list[float]]
    projected_annotation: Dict[str, Any]
    witness_symbolic: Dict[str, Any]
    required_annotation_labels: list[str]
    scene_entities: list[Dict[str, Any]]
    render_map: Dict[str, Any]
    execution_trace: Dict[str, Any]


_DEFAULTS = _TaskDefaults()
_SCENE_DEFAULTS = get_scene_defaults("geometry", "function_graph")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _rate_key(value: float) -> str:
    """Return the stable metadata key for one one-decimal rate value."""

    return f"{float(value):.1f}"


def _float_tuple_default(
    defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[float],
) -> Tuple[float, ...]:
    """Resolve one numeric sequence from scene-package defaults."""

    raw_value = defaults.get(str(key), fallback)
    if not isinstance(raw_value, Sequence) or isinstance(raw_value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence of numbers for {TASK_ID}")
    values = tuple(round(float(value), 1) for value in raw_value)
    if not values:
        raise ValueError(f"{key} cannot be empty for {TASK_ID}")
    if any(abs(float(value)) <= 1e-9 for value in values):
        raise ValueError(f"{key} cannot include zero for {TASK_ID}")
    return values


def _target_rate_support() -> Tuple[float, ...]:
    """Return configured answer support."""

    return _float_tuple_default(
        _GEN_DEFAULTS,
        "average_rate_support",
        _DEFAULTS.average_rate_support,
    )


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve the task's single public query and target average rate."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="single",
        task_id=TASK_ID,
    )
    target_rate, probabilities = resolve_rate_target(instance_seed=int(instance_seed), params=task_params)

    return _ResolvedQuery(
        query_id=str(query_id),
        target_rate=float(target_rate),
        query_id_probabilities=dict(query_probabilities),
        target_rate_probabilities=dict(sorted(probabilities.items())),
    )


def _sample_endpoint_y(rng, *, target_rate: float, delta_x: int) -> Tuple[float, float]:
    """Sample integer endpoint y-values that realize the requested rate."""

    delta_y = int(round(float(target_rate) * float(delta_x)))
    if abs(float(delta_y) - (float(target_rate) * float(delta_x))) > 1e-9:
        raise ValueError(f"target_rate {target_rate} is incompatible with delta_x {delta_x}")
    y0_min = max(-4, -4 - int(delta_y))
    y0_max = min(4, 4 - int(delta_y))
    if int(y0_min) > int(y0_max):
        raise RuntimeError(f"failed to sample endpoint y-values for target_rate={target_rate}")
    y0 = int(rng.randint(int(y0_min), int(y0_max)))
    return float(y0), float(y0 + int(delta_y))


def _sample_midpoint_y(rng, *, y0: float, y1: float) -> float:
    """Sample a non-collinear middle y-value within the visible grid."""

    secant_mid = 0.5 * (float(y0) + float(y1))
    candidates = [
        float(secant_mid + offset)
        for offset in (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)
        if -4.0 <= float(secant_mid + offset) <= 4.0
    ]
    if not candidates:
        candidates = [float(secant_mid)]
    return float(rng.choice(candidates))


def _sample_outer_y(rng, *, anchor_y: float) -> float:
    """Sample one outer function endpoint without leaving the graph window."""

    candidates = [float(value) for value in range(-4, 5) if abs(float(value) - float(anchor_y)) >= 1.0]
    if not candidates:
        candidates = [float(anchor_y)]
    return float(rng.choice(candidates))


def _sample_rate_scene(rng, query: _ResolvedQuery) -> _SampledRateScene:
    """Sample a piecewise-linear function with two marked endpoints A and B."""

    delta_x = 4
    x0 = float(rng.choice((-4, -2, 0)))
    x1 = float(x0 + delta_x)
    y0, y1 = _sample_endpoint_y(rng, target_rate=float(query.target_rate), delta_x=int(delta_x))
    x_mid = float(x0 + (0.5 * delta_x))
    y_mid = _sample_midpoint_y(rng, y0=float(y0), y1=float(y1))
    x_left = float(x0 - 4.0)
    x_right = float(x1 + 4.0)
    y_left = _sample_outer_y(rng, anchor_y=float(y0))
    y_right = _sample_outer_y(rng, anchor_y=float(y1))
    point_a = (float(x0), float(y0))
    point_b = (float(x1), float(y1))
    polyline = (
        (float(x_left), float(y_left)),
        point_a,
        (float(x_mid), float(y_mid)),
        point_b,
        (float(x_right), float(y_right)),
    )
    answer_value = round((float(y1) - float(y0)) / (float(x1) - float(x0)), 1)
    scene_entities = [
        {
            "entity_id": "function_graph",
            "entity_type": "function_graph",
            "draw_kind": "piecewise_linear",
            "polyline_graph": [[float(x), float(y)] for x, y in polyline],
        },
        {
            "entity_id": "marked_point_A",
            "entity_type": "marked_graph_point",
            "label": "A",
            "graph_point": [float(point_a[0]), float(point_a[1])],
        },
        {
            "entity_id": "marked_point_B",
            "entity_type": "marked_graph_point",
            "label": "B",
            "graph_point": [float(point_b[0]), float(point_b[1])],
        },
    ]
    render_map = {
        "marked_points_graph": {
            "A": [float(point_a[0]), float(point_a[1])],
            "B": [float(point_b[0]), float(point_b[1])],
        },
    }
    execution_trace = {
        "question_format": "average_rate_between_marked_points",
        "point_a_graph": [float(point_a[0]), float(point_a[1])],
        "point_b_graph": [float(point_b[0]), float(point_b[1])],
        "delta_x": float(x1 - x0),
        "delta_y": float(y1 - y0),
        "average_rate": float(answer_value),
        "average_rate_formula": "(y_B - y_A) / (x_B - x_A)",
    }
    return _SampledRateScene(
        polyline_graph=tuple(polyline),
        point_a=point_a,
        point_b=point_b,
        answer_value=float(answer_value),
        scene_entities=list(scene_entities),
        render_map=dict(render_map),
        execution_trace=dict(execution_trace),
    )


def _pixel_point(point: GraphPoint, *, context: GraphSceneContext) -> Tuple[float, float]:
    """Project one graph coordinate into canonical pixel coordinates."""

    return graph_units_to_pixel_float(
        point,
        graph_origin=context.graph_origin,
        graph_spacing=int(context.graph_spacing),
    )


def _draw_marked_point(
    draw: ImageDraw.ImageDraw,
    *,
    canonical_point: Tuple[float, float],
    label: str,
    label_direction: int,
    context: GraphSceneContext,
    marker_radius: int,
    label_font_size_px: int,
    marker_color: Sequence[int],
    label_color: Sequence[int],
    label_stroke_color: Sequence[int],
) -> list[float]:
    """Draw a marked endpoint and return the rendered label bbox."""

    scale = max(1, int(context.scene_scale))
    render_point = (float(canonical_point[0]) * float(scale), float(canonical_point[1]) * float(scale))
    radius = int(max(3, int(marker_radius))) * int(scale)
    draw.ellipse(
        (
            float(render_point[0] - radius),
            float(render_point[1] - radius),
            float(render_point[0] + radius),
            float(render_point[1] + radius),
        ),
        fill=tuple(int(value) for value in marker_color),
        outline=tuple(int(value) for value in label_stroke_color),
        width=max(1, int(2 * scale)),
    )
    offset_x = float(int(label_direction) * 18 * scale)
    offset_y = float(-18 * scale)
    font = load_font(int(label_font_size_px), bold=True)
    return draw_centered_text(
        draw,
        text=str(label),
        center=(float(render_point[0] + offset_x), float(render_point[1] + offset_y)),
        font=font,
        fill=tuple(int(value) for value in label_color),
        stroke_fill=tuple(int(value) for value in label_stroke_color),
        stroke_width=max(1, int(scale)),
    )


def _render_scene(
    draw: ImageDraw.ImageDraw,
    *,
    context: GraphSceneContext,
    sampled_scene: _SampledRateScene,
    shape_style,
    line_width: int,
    marker_radius: int,
    label_font_size_px: int,
) -> _RenderedRateScene:
    """Render the plotted function, marked endpoints, and annotation payload."""

    render_polyline = draw_function_polyline(
        draw,
        polyline_graph=sampled_scene.polyline_graph,
        graph_origin=context.graph_origin,
        graph_spacing=int(context.graph_spacing),
        scene_scale=int(context.scene_scale),
        line_width=int(line_width),
        line_color=shape_style.line_color,
    )
    point_a_pixel = _pixel_point(sampled_scene.point_a, context=context)
    point_b_pixel = _pixel_point(sampled_scene.point_b, context=context)
    scale = max(1, int(context.scene_scale))
    label_bbox_a = _draw_marked_point(
        draw,
        canonical_point=point_a_pixel,
        label="A",
        label_direction=-1,
        context=context,
        marker_radius=int(marker_radius),
        label_font_size_px=int(label_font_size_px),
        marker_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
    )
    label_bbox_b = _draw_marked_point(
        draw,
        canonical_point=point_b_pixel,
        label="B",
        label_direction=1,
        context=context,
        marker_radius=int(marker_radius),
        label_font_size_px=int(label_font_size_px),
        marker_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
    )

    render_map = dict(sampled_scene.render_map)
    render_map.update(
        {
            "function_polyline_pixel": [
                [round(float(_pixel_point(point, context=context)[0]), 3), round(float(_pixel_point(point, context=context)[1]), 3)]
                for point in sampled_scene.polyline_graph
            ],
            "function_polyline_render": [
                [round(float(point[0]), 3), round(float(point[1]), 3)]
                for point in render_polyline
            ],
            "marked_points_pixel": {
                "A": [round(float(point_a_pixel[0]), 3), round(float(point_a_pixel[1]), 3)],
                "B": [round(float(point_b_pixel[0]), 3), round(float(point_b_pixel[1]), 3)],
            },
            "marked_point_label_bboxes": {
                "A": [round(float(value) / float(scale), 3) for value in label_bbox_a],
                "B": [round(float(value) / float(scale), 3) for value in label_bbox_b],
            },
        }
    )
    points_by_label = {
        "A": point_a_pixel,
        "B": point_b_pixel,
    }
    annotation_value = {
        "A": [round(float(points_by_label["A"][0]), 3), round(float(points_by_label["A"][1]), 3)],
        "B": [round(float(points_by_label["B"][0]), 3), round(float(points_by_label["B"][1]), 3)],
    }
    projected_annotation = {
        "type": "point_map",
        "point_map": dict(annotation_value),
        "pixel_point_map": dict(annotation_value),
    }
    witness_symbolic = {
        "type": "marked_average_rate_points",
        "points_graph": {
            "A": [float(sampled_scene.point_a[0]), float(sampled_scene.point_a[1])],
            "B": [float(sampled_scene.point_b[0]), float(sampled_scene.point_b[1])],
        },
        "points_pixel": dict(annotation_value),
        "average_rate": float(sampled_scene.answer_value),
    }
    witness_symbolic.update(
        {
            "point_a_graph": [float(sampled_scene.point_a[0]), float(sampled_scene.point_a[1])],
            "point_b_graph": [float(sampled_scene.point_b[0]), float(sampled_scene.point_b[1])],
            "average_rate": float(sampled_scene.answer_value),
        }
    )
    return _RenderedRateScene(
        answer_value=float(sampled_scene.answer_value),
        annotation_type="point_map",
        annotation_value=dict(annotation_value),
        projected_annotation=dict(projected_annotation),
        witness_symbolic=dict(witness_symbolic),
        required_annotation_labels=["A", "B"],
        scene_entities=list(sampled_scene.scene_entities),
        render_map=dict(render_map),
        execution_trace=dict(sampled_scene.execution_trace),
    )




@register_task
class GeometryGraphingAverageRateValueTask:
    """Compute average rate of change between two marked points on a function graph."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a marked-secants graph with A/B annotation bound from the same trace."""

        del max_attempts
        query = _resolve_query(int(instance_seed), params=params)
        rng = spawn_rng(int(instance_seed), f"{self.task_id}.scene")

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
            ),
            context=f"prompt defaults for {self.task_id}",
        )

        scene_context = resolve_graph_scene_context(
            rng,
            instance_seed=int(instance_seed),
            scene_id=SCENE_ID,
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
            fallback_canvas_min=_DEFAULTS.canvas_size_min,
            fallback_canvas_max=_DEFAULTS.canvas_size_max,
            fallback_cells_min=_DEFAULTS.graph_cells_min,
            fallback_cells_max=_DEFAULTS.graph_cells_max,
            require_graph_paper_background=True,
            graph_style_overrides={
                "origin_fraction_x": 0.5,
                "origin_fraction_y": 0.5,
                "axis_scale_label_max_abs": 8,
                "origin_label_enabled": False,
            },
        )
        image, draw, background_meta = make_graph_scene_canvas(
            instance_seed=int(instance_seed),
            context=scene_context,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
            require_graph_paper=True,
        )
        line_width = int(
            params.get("line_width", group_default(_RENDER_DEFAULTS, "line_width", _DEFAULTS.line_width))
        ) * int(scene_context.scene_scale)
        marker_radius = int(
            params.get("marker_radius", group_default(_RENDER_DEFAULTS, "marker_radius", _DEFAULTS.marker_radius))
        )
        label_font_size_px = int(
            params.get(
                "label_font_size_px",
                resolve_scene_label_font_size_px(
                    canvas_size=int(scene_context.canvas_size),
                    graph_spacing=int(scene_context.graph_spacing),
                    scene_scale=int(scene_context.scene_scale),
                    min_px=int(group_default(_RENDER_DEFAULTS, "label_font_size_min", _DEFAULTS.label_font_size_min)),
                    max_px=int(group_default(_RENDER_DEFAULTS, "label_font_size_max", _DEFAULTS.label_font_size_max)),
                ),
            )
        )
        shape_style = sample_geometry_shape_style(
            rng,
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            anchor_colors=extract_background_anchor_colors(background_meta),
        )
        sampled_scene = _sample_rate_scene(rng, query)
        rendered_scene = _render_scene(
            draw,
            context=scene_context,
            sampled_scene=sampled_scene,
            shape_style=shape_style,
            line_width=int(line_width),
            marker_radius=int(marker_radius),
            label_font_size_px=int(label_font_size_px),
        )
        image, background_meta_final, post_noise_meta = finalize_graph_scene_image(
            image,
            instance_seed=int(instance_seed),
            context=scene_context,
            background_meta=background_meta,
            noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            query_key=AVERAGE_RATE_BETWEEN_MARKED_POINTS,
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "object_description": prompt_asset_slot(prompt_defaults, "object_description_average_rate"),
                "point_label_start": "A",
                "point_label_end": "B",
                "annotation_hint": prompt_asset_slot(prompt_defaults, "annotation_hint_average_rate_points"),
                "answer_hint": prompt_asset_slot(prompt_defaults, "answer_hint_number_one_decimal"),
                "json_example": prompt_asset_slot(prompt_defaults, "json_example_average_rate_points"),
                "json_example_answer_only": prompt_asset_slot(
                    prompt_defaults,
                    "json_example_average_rate_points_answer_only",
                ),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        answer_gt = TypedValue(type="number", value=float(round(rendered_scene.answer_value, 1)))
        annotation_gt = TypedValue(type=str(rendered_scene.annotation_type), value=dict(rendered_scene.annotation_value))
        query_params = {
            "query_id": str(query.query_id),
            "query_id_probabilities": dict(query.query_id_probabilities),
            "target_rate": float(query.target_rate),
            "target_rate_probabilities": dict(query.target_rate_probabilities),
        }
        execution_trace = {
            "query_id": str(query.query_id),
            "query_id_probabilities": dict(query.query_id_probabilities),
            "target_rate": float(query.target_rate),
            "target_rate_probabilities": dict(query.target_rate_probabilities),
            "answer_type": "number",
            "answer_value": float(answer_gt.value),
            "required_annotation_labels": list(rendered_scene.required_annotation_labels),
            **dict(rendered_scene.execution_trace),
        }
        trace_payload = {
            "scene_ir": {
                "scene_kind": "geometry_graphing_average_rate",
                "entities": list(rendered_scene.scene_entities),
                "relations": {
                    "query_id": str(query.query_id),
                    "target_rate": float(query.target_rate),
                },
            },
            "query_spec": {
                "query_id": str(query.query_id),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": dict(query_params),
            },
            "render_spec": {
                "canvas_size": int(scene_context.canvas_size),
                "coord_space": "pixel",
                "background_style": dict(background_meta_final),
                "post_image_noise": dict(post_noise_meta),
                "shape_style": dict(shape_style.to_trace_dict()),
                "graph_coordinate_frame": dict(scene_context.graph_frame),
                "graph_paper_grid": graph_paper_grid_from_frame(scene_context.graph_frame),
                **dict(scene_context.graph_layout_metadata),
                "scene_variant": "piecewise_linear",
            },
            "render_map": dict(rendered_scene.render_map),
            "execution_trace": dict(execution_trace),
            "witness_symbolic": dict(rendered_scene.witness_symbolic),
            "projected_annotation": dict(rendered_scene.projected_annotation),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            query_id=str(query.query_id),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )
