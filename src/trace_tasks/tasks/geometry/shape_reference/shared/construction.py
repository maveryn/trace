"""Geometry transformation matching task on one shared graph-paper scene."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font, resolve_scene_label_font_size_px
from trace_tasks.tasks.shared.geometry_primitives import Point, point_inside_square_canvas
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_dashed_line
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from trace_tasks.tasks.geometry.shared.comparison import resolve_comparison_winner_label
from trace_tasks.tasks.geometry.shared.background_defaults import load_geometry_background_defaults
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame, scale_point
from trace_tasks.tasks.geometry.shared.labeled_point_annotation import graph_point_set_annotation_artifacts
from trace_tasks.tasks.geometry.shared.multi_polygon_scene import PolygonSceneObject, draw_polygon_objects
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.geometry.shared.option_count import resolve_geometry_option_count
from trace_tasks.tasks.geometry.shared.point_labels import draw_labeled_points
from trace_tasks.tasks.geometry.shared.polygon_scene_helpers import (
    draw_reference_polygon,
    graph_polygon_inside_canvas,
    pixel_point_from_graph_units,
    pixel_polygon_from_graph_units,
)

SCENE_ID = "shape_reference"
from trace_tasks.tasks.geometry.shared.polygon_transformations import (
    Polygon,
    RIGID_TRANSFORM_RECIPE_IDS,
    apply_rigid_transform_recipe,
    ordered_vertex_label_map,
    sample_asymmetric_polygon_template,
    translate_polygon,
)
from trace_tasks.tasks.geometry.shared.render_variation import sample_int_render_param
from trace_tasks.tasks.geometry.shared.shape_style import extract_background_anchor_colors, sample_geometry_shape_style
from trace_tasks.tasks.geometry.shared.single_object_scene import (
    GraphSceneContext,
    finalize_graph_scene_image,
    make_graph_scene_canvas,
    resolve_graph_scene_context,
)


SCENE_NAMESPACE = "shape_reference_transform_selection"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("triangle", "quadrilateral")
TRANSFORM_RULES: Tuple[str, ...] = ("translation", "reflection", "rotation")
COMPATIBILITY: Dict[str, Sequence[str]] = {
    "triangle": TRANSFORM_RULES,
    "quadrilateral": TRANSFORM_RULES,
}

POST_IMAGE_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)

_TRANSFORM_RECIPE_IDENTITY = "identity"
_TRANSFORM_RECIPE_REFLECT_VERTICAL = "reflect_vertical"
_TRANSFORM_RECIPE_REFLECT_HORIZONTAL = "reflect_horizontal"
_TRANSFORM_RECIPE_ROTATE_90_CW = "rotate_90_cw"
_TRANSFORM_RECIPE_ROTATE_90_CCW = "rotate_90_ccw"
_TRANSFORM_RECIPE_ROTATE_180 = "rotate_180"

_LOCAL_TRANSFORM_RECIPES: Tuple[str, ...] = tuple(RIGID_TRANSFORM_RECIPE_IDS)



@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for geometry transformation scenes."""

    canvas_size_min: int = 720
    canvas_size_max: int = 800
    graph_cells_min: int = 34
    graph_cells_max: int = 38
    line_width: int = 4
    line_width_min: int = 3
    line_width_max: int = 5
    label_font_size_min: int = 16
    label_font_size_max: int = 28
    label_stroke_width: int = 1
    label_stroke_width_min: int = 1
    label_stroke_width_max: int = 1
    object_label_offset_px: int = 14
    reference_label_gap_px: int = 20
    cue_label_gap_px: int = 18
    cue_dash_px: int = 10
    cue_gap_px: int = 8
    cue_arrow_head_length_px: int = 18
    cue_arrow_head_width_px: int = 14
    cue_point_radius_px: int = 4
    cue_line_padding_px: int = 18
    candidate_label_pool: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    translation_vectors: Tuple[Tuple[int, int], ...] = (
        (10, 0),
        (10, 4),
        (10, -4),
        (15, 0),
        (15, 4),
        (15, -4),
    )
    candidate_slots: Tuple[Tuple[int, int], ...] = (
        (6, 6),
        (13, 6),
        (6, 0),
        (13, 0),
        (6, -6),
        (13, -6),
    )
    translation_cue_gap_graph: int = 4
    candidate_min_gap_graph: float = 0.75
    reflection_axis_x: int = 0
    reflection_line_y_min: int = -9
    reflection_line_y_max: int = 9


@dataclass(frozen=True)
class _RotationMode:
    """One supported rotation query flavor."""

    mode_id: str
    quarter_turns: int
    prompt_label: str
    allowed_slot_indices: Tuple[int, ...]


_ROTATION_MODES: Tuple[_RotationMode, ...] = (
    _RotationMode(
        mode_id="quarter_turn_clockwise",
        quarter_turns=1,
        prompt_label="90° clockwise rotation",
        allowed_slot_indices=(0, 1),
    ),
    _RotationMode(
        mode_id="half_turn",
        quarter_turns=2,
        prompt_label="180° rotation",
        allowed_slot_indices=(0, 1, 2, 3, 4, 5),
    ),
    _RotationMode(
        mode_id="quarter_turn_counterclockwise",
        quarter_turns=3,
        prompt_label="90° counterclockwise rotation",
        allowed_slot_indices=(4, 5),
    ),
)


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved scene/query axes and answer-label support for one instance."""

    scene_variant: str
    transform_rule: str
    winner_label: str
    scene_variant_probabilities: Dict[str, float]
    transform_rule_probabilities: Dict[str, float]
    winner_label_probabilities: Dict[str, float]
    candidate_label_pool: Tuple[str, ...]
    candidate_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _RenderedTransformationScene:
    """Task-local scene package with geometry, cues, and trace payloads."""

    reference_vertices_graph: Polygon
    winner_vertices_graph: Polygon
    reference_vertices_px: Polygon
    winner_vertices_px: Polygon
    candidate_vertices_graph_by_label: Dict[str, Polygon]
    candidate_vertices_px_by_label: Dict[str, Polygon]
    candidate_centers_graph_by_label: Dict[str, Point]
    candidate_centers_px_by_label: Dict[str, Point]
    winner_label: str
    reference_center_graph: Point
    cue_kind: str
    cue_trace: Dict[str, Any]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    annotation: Dict[str, Any]
    answer_value: str
    object_label_centers: Dict[str, List[float]]
    required_annotation_labels: List[str]
    rotation_mode: str | None
    rotation_prompt_label: str | None
    translation_vector: Tuple[int, int] | None


@dataclass(frozen=True)
class TransformationSceneBundle:
    """Rendered transform-selection scene plus neutral metadata for public tasks."""

    resolved: _ResolvedQuery
    rendered_scene: _RenderedTransformationScene
    image: Any
    context: GraphSceneContext
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    shape_style_trace: Dict[str, Any]
    line_width: int
    label_font_size_px: int
    label_stroke_width_px: int


_DEFAULTS = _TaskDefaults()
_SCENE_DEFAULTS = get_scene_defaults("geometry", "shape_reference")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=SCENE_NAMESPACE,
)
_WINNER_LABEL_BALANCE_SALT = 49017


def _apply_local_transform(template: Polygon, *, recipe: str) -> Polygon:
    """Apply one local rigid transform recipe around the origin."""
    return apply_rigid_transform_recipe(template, recipe=str(recipe))


def _winner_recipe_for_rule(*, transform_rule: str, rotation_mode: _RotationMode | None) -> str:
    """Return the correct local transform recipe for the requested rule."""

    normalized_rule = str(transform_rule)
    if normalized_rule == "translation":
        return _TRANSFORM_RECIPE_IDENTITY
    if normalized_rule == "reflection":
        return _TRANSFORM_RECIPE_REFLECT_VERTICAL
    if normalized_rule == "rotation":
        if rotation_mode is None:
            raise ValueError("rotation requires one resolved rotation_mode")
        quarter_turns = int(rotation_mode.quarter_turns) % 4
        if quarter_turns == 1:
            return _TRANSFORM_RECIPE_ROTATE_90_CW
        if quarter_turns == 2:
            return _TRANSFORM_RECIPE_ROTATE_180
        if quarter_turns == 3:
            return _TRANSFORM_RECIPE_ROTATE_90_CCW
    raise ValueError(f"unsupported transform_rule: {transform_rule}")


def _local_distractor_recipes(*, winner_recipe: str) -> Tuple[str, ...]:
    """Return the five distractor transform recipes for the non-winning candidates."""

    return tuple(recipe for recipe in _LOCAL_TRANSFORM_RECIPES if str(recipe) != str(winner_recipe))


def _reference_center_for_rotation(slot_center: Point, *, rotation_mode: _RotationMode) -> Point:
    """Return the inverse-rotated reference center for one winner slot."""

    if int(rotation_mode.quarter_turns) % 4 == 1:
        return (-float(slot_center[1]), float(slot_center[0]))
    if int(rotation_mode.quarter_turns) % 4 == 2:
        return (-float(slot_center[0]), -float(slot_center[1]))
    if int(rotation_mode.quarter_turns) % 4 == 3:
        return (float(slot_center[1]), -float(slot_center[0]))
    raise ValueError("rotation transformation requires a non-zero quarter turn")


def _translation_vector_support(params: Mapping[str, Any]) -> Tuple[Tuple[int, int], ...]:
    """Resolve the supported translation vectors from params/defaults."""

    raw_support = params.get("translation_vectors", group_default(_GEN_DEFAULTS, "translation_vectors", _DEFAULTS.translation_vectors))
    support: List[Tuple[int, int]] = []
    for value in raw_support:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError("translation_vectors entries must be [dx, dy] pairs")
        support.append((int(value[0]), int(value[1])))
    if not support:
        raise ValueError("translation_vectors support must be non-empty")
    return tuple(support)


def _candidate_slot_support(params: Mapping[str, Any], *, candidate_count: int) -> Tuple[Tuple[int, int], ...]:
    """Resolve the visible candidate slot centers used by the transformation scene."""

    raw_support = params.get("candidate_slots", group_default(_GEN_DEFAULTS, "candidate_slots", _DEFAULTS.candidate_slots))
    slots: List[Tuple[int, int]] = []
    for value in raw_support:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError("candidate_slots entries must be [x, y] graph-unit pairs")
        slot = (int(value[0]), int(value[1]))
        if slot not in slots:
            slots.append(slot)
    if len(slots) < int(candidate_count):
        raise ValueError("geometry transformation candidate_slots cannot provide the visible candidate count")
    return tuple(slots[: int(candidate_count)])


def _visible_candidate_labels(label_pool: Sequence[str], *, winner_label: str, candidate_count: int) -> Tuple[str, ...]:
    labels = tuple(str(label) for label in label_pool[: int(candidate_count)])
    if str(winner_label) in set(labels):
        return labels
    return tuple([str(winner_label), *[label for label in labels if str(label) != str(winner_label)]])[: int(candidate_count)]


def _polygon_graph_bbox(vertices: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Return one graph-coordinate bbox for a polygon."""

    min_x = min(float(point[0]) for point in vertices)
    max_x = max(float(point[0]) for point in vertices)
    min_y = min(float(point[1]) for point in vertices)
    max_y = max(float(point[1]) for point in vertices)
    return (float(min_x), float(min_y), float(max_x), float(max_y))


def _candidate_bboxes_have_clearance(
    polygons_by_label: Mapping[str, Sequence[Point]],
    *,
    min_gap_graph: float,
) -> bool:
    """Return true when candidate polygon bboxes do not overlap or crowd."""

    labels = sorted(str(label) for label in polygons_by_label.keys())
    boxes = {
        str(label): _polygon_graph_bbox(polygons_by_label[str(label)])
        for label in labels
    }
    gap = float(min_gap_graph)
    for left_index, left_label in enumerate(labels):
        left = boxes[str(left_label)]
        for right_label in labels[left_index + 1 :]:
            right = boxes[str(right_label)]
            separated = (
                float(left[2]) + gap <= float(right[0])
                or float(right[2]) + gap <= float(left[0])
                or float(left[3]) + gap <= float(right[1])
                or float(right[3]) + gap <= float(left[1])
            )
            if not bool(separated):
                return False
    return True


def _translation_cue_stays_left_of_y_axis(
    *,
    vector_anchor: Tuple[int, int],
    translation_vector: Tuple[int, int],
) -> bool:
    """Return true when the translation cue does not touch or cross x=0."""

    start_x = int(vector_anchor[0])
    end_x = int(vector_anchor[0]) + int(translation_vector[0])
    return max(int(start_x), int(end_x)) < 0


def _translation_cue_anchor_for_reference(
    reference_vertices_graph: Sequence[Point],
    *,
    translation_vector: Tuple[int, int],
    context: GraphSceneContext,
    padding_px: float,
    cue_gap_graph: int,
) -> Tuple[int, int] | None:
    """Place the translation cue above the Reference while keeping it left of x=0."""

    min_x, _min_y, max_x, max_y = _polygon_graph_bbox(reference_vertices_graph)
    dx = int(translation_vector[0])
    dy = int(translation_vector[1])
    reference_mid_x = (float(min_x) + float(max_x)) / 2.0
    start_x = int(round(float(reference_mid_x) - (float(dx) / 2.0)))
    if int(dx) >= 0:
        start_x = min(int(start_x), -int(dx) - 1)
    else:
        start_x = min(int(start_x), -1)

    start_y = int(math.ceil(float(max_y) + float(cue_gap_graph) + float(max(0, -int(dy)))))
    anchor = (int(start_x), int(start_y))
    end = (int(anchor[0]) + int(dx), int(anchor[1]) + int(dy))
    if not _translation_cue_stays_left_of_y_axis(
        vector_anchor=anchor,
        translation_vector=(int(dx), int(dy)),
    ):
        return None
    if min(int(anchor[1]), int(end[1])) <= float(max_y):
        return None
    cue_points_ok = all(
        point_inside_square_canvas(
            pixel_point_from_graph_units(point, context=context),
            canvas_size=int(context.canvas_size),
            padding=float(padding_px),
        )
        for point in (anchor, end)
    )
    if not cue_points_ok:
        return None
    return anchor


def _draw_translation_cue(
    draw,
    *,
    context: GraphSceneContext,
    line_width: int,
    head_length_px: int,
    head_width_px: int,
    vector_anchor: Tuple[int, int],
    translation_vector: Tuple[int, int],
    color: Sequence[int],
) -> Dict[str, Any]:
    """Draw the translation vector cue and return trace metadata."""

    start_graph = (int(vector_anchor[0]), int(vector_anchor[1]))
    end_graph = (
        int(vector_anchor[0]) + int(translation_vector[0]),
        int(vector_anchor[1]) + int(translation_vector[1]),
    )
    start_px = scale_point(pixel_point_from_graph_units(start_graph, context=context), int(context.scene_scale))
    end_px = scale_point(pixel_point_from_graph_units(end_graph, context=context), int(context.scene_scale))
    draw_arrow(
        draw,
        start=start_px,
        end=end_px,
        fill=tuple(int(value) for value in color),
        width=int(line_width),
        head_length_px=float(head_length_px),
        head_width_px=float(head_width_px),
    )
    return {
        "type": "translation_vector",
        "start_graph": [int(start_graph[0]), int(start_graph[1])],
        "end_graph": [int(end_graph[0]), int(end_graph[1])],
        "vector_graph": [int(translation_vector[0]), int(translation_vector[1])],
        "start_px": [round(float(start_px[0]) / float(max(1, int(context.scene_scale))), 3), round(float(start_px[1]) / float(max(1, int(context.scene_scale))), 3)],
        "end_px": [round(float(end_px[0]) / float(max(1, int(context.scene_scale))), 3), round(float(end_px[1]) / float(max(1, int(context.scene_scale))), 3)],
    }


def _draw_reflection_cue(
    draw,
    *,
    context: GraphSceneContext,
    line_width: int,
    dash_px: int,
    gap_px: int,
    label_font_size_px: int,
    label_stroke_width: int,
    cue_label_gap_px: int,
    axis_x: int,
    y_min: int,
    y_max: int,
    color: Sequence[int],
    label_color: Sequence[int],
    label_stroke_color: Sequence[int],
) -> Dict[str, Any]:
    """Draw the vertical reflection line `L` and return trace metadata."""

    start_graph = (int(axis_x), int(y_min))
    end_graph = (int(axis_x), int(y_max))
    start_px = scale_point(pixel_point_from_graph_units(start_graph, context=context), int(context.scene_scale))
    end_px = scale_point(pixel_point_from_graph_units(end_graph, context=context), int(context.scene_scale))
    draw_dashed_line(
        draw,
        start=start_px,
        end=end_px,
        fill=tuple(int(value) for value in color),
        width=max(1, int(line_width)),
        dash_px=float(dash_px),
        gap_px=float(gap_px),
    )
    label_center = (
        float(start_px[0]) + float(cue_label_gap_px),
        float(min(start_px[1], end_px[1])) + float(label_font_size_px),
    )
    font = load_font(int(label_font_size_px), bold=True)
    draw_text_centered(
        draw,
        text="L",
        center=label_center,
        font=font,
        fill=tuple(int(value) for value in label_color),
        stroke_fill=tuple(int(value) for value in label_stroke_color),
        stroke_width=int(label_stroke_width),
    )
    return {
        "type": "reflection_line",
        "axis_kind": "vertical",
        "line_label": "L",
        "x_graph": int(axis_x),
        "y_graph_range": [int(y_min), int(y_max)],
        "start_px": [round(float(start_px[0]) / float(max(1, int(context.scene_scale))), 3), round(float(start_px[1]) / float(max(1, int(context.scene_scale))), 3)],
        "end_px": [round(float(end_px[0]) / float(max(1, int(context.scene_scale))), 3), round(float(end_px[1]) / float(max(1, int(context.scene_scale))), 3)],
    }


def _draw_rotation_cue(
    draw,
    *,
    context: GraphSceneContext,
    label_font_size_px: int,
    label_stroke_width: int,
    point_radius_px: int,
    color: Sequence[int],
    label_color: Sequence[int],
    label_stroke_color: Sequence[int],
) -> Dict[str, Any]:
    """Draw the rotation-center point `O` and return trace metadata."""

    center_graph = (0, 0)
    center_px = pixel_point_from_graph_units(center_graph, context=context)
    scaled_center = scale_point(center_px, int(context.scene_scale))
    draw_labeled_points(
        draw,
        points=[scaled_center],
        labels=["O"],
        label_offset_px=float(max(12, int(label_font_size_px))),
        font_size_px=int(label_font_size_px),
        text_stroke_width=int(label_stroke_width),
        marker_radius_px=max(1, int(point_radius_px)),
        marker_color=tuple(int(value) for value in color),
        label_color=tuple(int(value) for value in label_color),
        label_stroke_color=tuple(int(value) for value in label_stroke_color),
        canvas_size=int(context.canvas_size) * int(context.scene_scale),
    )
    return {
        "type": "rotation_center",
        "point_label": "O",
        "center_graph": [0, 0],
        "center_px": [round(float(center_px[0]), 3), round(float(center_px[1]), 3)],
    }


def _decoupled_winner_label_params(*, params: Mapping[str, Any]) -> Mapping[str, Any]:
    """No-op hook for winner-label cycling call sites."""

    return params


def _resolve_axes(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve scene and transform axes plus balanced answer-label support."""

    axis_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.axes")
    scene_supported = [str(value) for value in SUPPORTED_SCENE_VARIANTS]
    rule_supported = [str(value) for value in TRANSFORM_RULES]
    explicit_rule = params.get("transform_rule")
    if explicit_rule is None:
        raise ValueError("shape-reference transformation rule must be resolved by the public task")
    transform_rule = str(explicit_rule)
    if transform_rule not in set(rule_supported):
        raise ValueError(f"unsupported transform_rule: {transform_rule}")
    explicit_scene = params.get("scene_variant")
    if explicit_scene is not None:
        scene_variant = str(explicit_scene)
        if scene_variant not in set(scene_supported):
            raise ValueError(f"unsupported scene_variant: {scene_variant}")
        if transform_rule not in set(str(value) for value in COMPATIBILITY.get(scene_variant, ())):
            raise ValueError(f"incompatible scene/rule combination: {scene_variant} + {transform_rule}")
        scene_probs = {scene: (1.0 if scene == scene_variant else 0.0) for scene in scene_supported}
    else:
        selected_scene, restricted_scene_probs = resolve_variant(
            axis_rng,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            supported_variants=[scene for scene in scene_supported if transform_rule in set(COMPATIBILITY.get(scene, ()))],
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
        )
        scene_variant = apply_balanced_variant_sampling(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            selected_variant=str(selected_scene),
            variant_probabilities=restricted_scene_probs,
            supported_variants=[scene for scene in scene_supported if transform_rule in set(COMPATIBILITY.get(scene, ()))],
            balance_flag_key="balanced_scene_variant_sampling",
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
            sampling_namespace=f"{SCENE_NAMESPACE}.scene_variant.{transform_rule}",
        )
        scene_probs = {scene: float(restricted_scene_probs.get(scene, 0.0)) for scene in scene_supported}
    rule_probs = {rule: (1.0 if rule == transform_rule else 0.0) for rule in rule_supported}
    label_pool = tuple(
        str(label).upper()
        for label in params.get(
            "candidate_label_pool",
            group_default(_GEN_DEFAULTS, "candidate_label_pool", _DEFAULTS.candidate_label_pool),
        )
    )
    if len(label_pool) not in {4, 6} or len(set(label_pool)) != len(label_pool):
        raise ValueError("geometry transformation requires four or six unique candidate labels")
    supported_counts = tuple(count for count in (4, 6) if int(count) <= len(label_pool))
    candidate_count, candidate_count_probabilities = resolve_geometry_option_count(
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        field_name="candidate_count",
        supported_counts=supported_counts,
        task_id=SCENE_NAMESPACE,
        instance_seed=int(instance_seed),
    )
    winner_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.winner_label")
    winner_label, winner_probs = resolve_comparison_winner_label(
        winner_rng,
        instance_seed=int(instance_seed),
        params=_decoupled_winner_label_params(params=params),
        gen_defaults=_GEN_DEFAULTS,
        label_pool=label_pool,
        selection_namespace=(
            f"{SCENE_NAMESPACE}.winner_label.{str(scene_variant)}.{str(transform_rule)}"
        ),
    )
    visible_labels = _visible_candidate_labels(label_pool, winner_label=str(winner_label), candidate_count=int(candidate_count))
    return _ResolvedQuery(
        scene_variant=str(scene_variant),
        transform_rule=str(transform_rule),
        winner_label=str(winner_label),
        scene_variant_probabilities=dict(scene_probs),
        transform_rule_probabilities=dict(rule_probs),
        winner_label_probabilities=dict(winner_probs),
        candidate_label_pool=tuple(visible_labels),
        candidate_count_probabilities=dict(candidate_count_probabilities),
    )


def _sample_transformation_scene(
    rng,
    *,
    query: _ResolvedQuery,
    context: GraphSceneContext,
    padding_px: float,
    line_width: int,
    label_font_size_px: int,
    label_stroke_width: int,
    reference_label_gap_px: int,
    cue_label_gap_px: int,
    cue_dash_px: int,
    cue_gap_px: int,
    cue_arrow_head_length_px: int,
    cue_arrow_head_width_px: int,
    cue_point_radius_px: int,
    object_label_offset_px: float,
    draw,
    shape_style,
    render_canvas_size: int,
    params: Mapping[str, Any],
) -> _RenderedTransformationScene:
    """Sample and render one full transformation-match scene."""

    template = sample_asymmetric_polygon_template(str(query.scene_variant), rng, profile="compact")
    slots = _candidate_slot_support(params, candidate_count=len(query.candidate_label_pool))
    translation_vectors = _translation_vector_support(params)

    winner_slot_index: int | None = None
    reference_center_graph: Point | None = None
    rotation_mode: _RotationMode | None = None
    translation_vector: Tuple[int, int] | None = None
    translation_vector_anchor: Tuple[int, int] | None = None

    slot_indices = list(range(len(slots)))
    rng.shuffle(slot_indices)
    if str(query.transform_rule) == "translation":
        candidate_pairs: List[Tuple[int, Tuple[int, int], Point, Tuple[int, int]]] = []
        for slot_index in slot_indices:
            slot_center = slots[int(slot_index)]
            for dx, dy in translation_vectors:
                candidate_reference = (
                    float(slot_center[0]) - float(dx),
                    float(slot_center[1]) - float(dy),
                )
                if float(candidate_reference[0]) > -2.0:
                    continue
                reference_vertices_candidate = translate_polygon(
                    template,
                    dx=int(candidate_reference[0]),
                    dy=int(candidate_reference[1]),
                )
                if not graph_polygon_inside_canvas(
                    reference_vertices_candidate,
                    context=context,
                    padding_px=float(padding_px),
                ):
                    continue
                cue_start = _translation_cue_anchor_for_reference(
                    reference_vertices_candidate,
                    translation_vector=(int(dx), int(dy)),
                    context=context,
                    padding_px=float(padding_px),
                    cue_gap_graph=int(
                        params.get(
                            "translation_cue_gap_graph",
                            group_default(_GEN_DEFAULTS, "translation_cue_gap_graph", _DEFAULTS.translation_cue_gap_graph),
                        )
                    ),
                )
                if cue_start is None:
                    continue
                candidate_pairs.append((int(slot_index), (int(dx), int(dy)), candidate_reference, cue_start))
        if not candidate_pairs:
            raise ValueError("no feasible translation slot/vector pair for current scene context")
        winner_slot_index, translation_vector, reference_center_graph, translation_vector_anchor = rng.choice(candidate_pairs)
    elif str(query.transform_rule) == "reflection":
        axis_x = int(params.get("reflection_axis_x", group_default(_GEN_DEFAULTS, "reflection_axis_x", _DEFAULTS.reflection_axis_x)))
        for slot_index in slot_indices:
            slot_center = slots[int(slot_index)]
            candidate_reference = (
                float((2 * int(axis_x)) - int(slot_center[0])),
                float(slot_center[1]),
            )
            if float(candidate_reference[0]) >= float(axis_x):
                continue
            reference_vertices_graph = translate_polygon(
                template,
                dx=int(candidate_reference[0]),
                dy=int(candidate_reference[1]),
            )
            if graph_polygon_inside_canvas(reference_vertices_graph, context=context, padding_px=float(padding_px)):
                winner_slot_index = int(slot_index)
                reference_center_graph = candidate_reference
                break
        if winner_slot_index is None or reference_center_graph is None:
            raise ValueError("no feasible reflection slot for current scene context")
    else:
        rotation_modes = list(_ROTATION_MODES)
        rng.shuffle(rotation_modes)
        for mode in rotation_modes:
            allowed = [index for index in mode.allowed_slot_indices if int(index) < len(slots)]
            rng.shuffle(allowed)
            for slot_index in allowed:
                slot_center = slots[int(slot_index)]
                candidate_reference = _reference_center_for_rotation(slot_center, rotation_mode=mode)
                if float(candidate_reference[0]) > -2.0:
                    continue
                reference_vertices_graph = translate_polygon(
                    template,
                    dx=int(candidate_reference[0]),
                    dy=int(candidate_reference[1]),
                )
                if graph_polygon_inside_canvas(reference_vertices_graph, context=context, padding_px=float(padding_px)):
                    winner_slot_index = int(slot_index)
                    reference_center_graph = candidate_reference
                    rotation_mode = mode
                    break
            if winner_slot_index is not None:
                break
        if winner_slot_index is None or reference_center_graph is None or rotation_mode is None:
            raise ValueError("no feasible rotation slot for current scene context")

    reference_vertices_graph = translate_polygon(
        template,
        dx=int(reference_center_graph[0]),
        dy=int(reference_center_graph[1]),
    )
    reference_vertices_px = pixel_polygon_from_graph_units(reference_vertices_graph, context=context)

    winner_recipe = _winner_recipe_for_rule(transform_rule=str(query.transform_rule), rotation_mode=rotation_mode)
    distractor_recipes = list(_local_distractor_recipes(winner_recipe=str(winner_recipe)))
    rng.shuffle(distractor_recipes)

    labels = list(query.candidate_label_pool)
    other_labels = [label for label in labels if str(label) != str(query.winner_label)]
    rng.shuffle(other_labels)

    candidate_vertices_graph_by_label: Dict[str, Polygon] = {}
    candidate_vertices_px_by_label: Dict[str, Polygon] = {}
    candidate_centers_graph_by_label: Dict[str, Point] = {}
    candidate_centers_px_by_label: Dict[str, Point] = {}
    objects: List[PolygonSceneObject] = []
    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "reference_polygon",
            "entity_type": "reference_polygon",
            "scene_variant": str(query.scene_variant),
            "vertices_graph": [[int(round(point[0])), int(round(point[1]))] for point in reference_vertices_graph],
            "center_graph": [round(float(reference_center_graph[0]), 3), round(float(reference_center_graph[1]), 3)],
        }
    ]

    label_by_slot_index: Dict[int, str] = {}
    for slot_index in range(len(slots)):
        if int(slot_index) == int(winner_slot_index):
            label_by_slot_index[int(slot_index)] = str(query.winner_label)
        else:
            label_by_slot_index[int(slot_index)] = str(other_labels.pop())

    for slot_index, slot_center in enumerate(slots):
        label = str(label_by_slot_index[int(slot_index)])
        if int(slot_index) == int(winner_slot_index):
            local_vertices = _apply_local_transform(template, recipe=str(winner_recipe))
        else:
            local_vertices = _apply_local_transform(template, recipe=str(distractor_recipes.pop()))
        candidate_vertices_graph = translate_polygon(
            local_vertices,
            dx=int(slot_center[0]),
            dy=int(slot_center[1]),
        )
        if not graph_polygon_inside_canvas(candidate_vertices_graph, context=context, padding_px=float(padding_px)):
            raise ValueError("candidate polygon fell outside the graph-paper canvas")
        candidate_vertices_px = pixel_polygon_from_graph_units(candidate_vertices_graph, context=context)
        candidate_vertices_graph_by_label[str(label)] = tuple(candidate_vertices_graph)
        candidate_vertices_px_by_label[str(label)] = tuple(candidate_vertices_px)
        candidate_centers_graph_by_label[str(label)] = (float(slot_center[0]), float(slot_center[1]))
        candidate_centers_px_by_label[str(label)] = pixel_point_from_graph_units(slot_center, context=context)
        objects.append(
            PolygonSceneObject(
                label=str(label),
                vertices=tuple(candidate_vertices_px),
                center=pixel_point_from_graph_units(slot_center, context=context),
            )
        )
        scene_entities.append(
            {
                "entity_id": f"candidate_{label}",
                "entity_type": "candidate_polygon",
                "label": str(label),
                "scene_variant": str(query.scene_variant),
                "vertices_graph": [[int(round(point[0])), int(round(point[1]))] for point in candidate_vertices_graph],
                "center_graph": [int(slot_center[0]), int(slot_center[1])],
            }
        )

    candidate_min_gap_graph = float(
        params.get(
            "candidate_min_gap_graph",
            group_default(_GEN_DEFAULTS, "candidate_min_gap_graph", _DEFAULTS.candidate_min_gap_graph),
        )
    )
    if not _candidate_bboxes_have_clearance(
        candidate_vertices_graph_by_label,
        min_gap_graph=float(candidate_min_gap_graph),
    ):
        raise ValueError("candidate polygons are too close or overlapping")

    object_label_centers = draw_polygon_objects(
        draw,
        objects=objects,
        scene_scale=int(context.scene_scale),
        line_width=int(line_width),
        label_font_size_px=int(label_font_size_px),
        label_stroke_width=int(label_stroke_width),
        object_label_offset_px=float(object_label_offset_px),
        render_canvas_size=int(render_canvas_size),
        shape_style=shape_style,
    )
    draw_reference_polygon(
        draw,
        vertices_px=reference_vertices_px,
        scene_scale=int(context.scene_scale),
        line_width=int(line_width),
        label_font_size_px=int(label_font_size_px),
        label_stroke_width=int(label_stroke_width),
        label_gap_px=float(reference_label_gap_px),
        line_color=shape_style.line_color,
        label_color=shape_style.label_color,
        label_stroke_color=shape_style.label_stroke_color,
    )

    if str(query.transform_rule) == "translation":
        if translation_vector_anchor is None:
            raise ValueError("translation requires one resolved cue anchor")
        cue_trace = _draw_translation_cue(
            draw,
            context=context,
            line_width=int(line_width),
            head_length_px=int(cue_arrow_head_length_px),
            head_width_px=int(cue_arrow_head_width_px),
            vector_anchor=translation_vector_anchor,
            translation_vector=translation_vector if translation_vector is not None else (0, 0),
            color=shape_style.line_color,
        )
    elif str(query.transform_rule) == "reflection":
        cue_trace = _draw_reflection_cue(
            draw,
            context=context,
            line_width=int(line_width),
            dash_px=int(cue_dash_px) * int(context.scene_scale),
            gap_px=int(cue_gap_px) * int(context.scene_scale),
            label_font_size_px=int(label_font_size_px),
            label_stroke_width=int(label_stroke_width),
            cue_label_gap_px=int(cue_label_gap_px) * int(context.scene_scale),
            axis_x=int(params.get("reflection_axis_x", group_default(_GEN_DEFAULTS, "reflection_axis_x", _DEFAULTS.reflection_axis_x))),
            y_min=int(params.get("reflection_line_y_min", group_default(_GEN_DEFAULTS, "reflection_line_y_min", _DEFAULTS.reflection_line_y_min))),
            y_max=int(params.get("reflection_line_y_max", group_default(_GEN_DEFAULTS, "reflection_line_y_max", _DEFAULTS.reflection_line_y_max))),
            color=shape_style.line_color,
            label_color=shape_style.label_color,
            label_stroke_color=shape_style.label_stroke_color,
        )
    else:
        cue_trace = _draw_rotation_cue(
            draw,
            context=context,
            label_font_size_px=int(label_font_size_px),
            label_stroke_width=int(label_stroke_width),
            point_radius_px=max(1, int(cue_point_radius_px) * int(context.scene_scale)),
            color=shape_style.line_color,
            label_color=shape_style.label_color,
            label_stroke_color=shape_style.label_stroke_color,
        )
        cue_trace["rotation_mode"] = str(rotation_mode.mode_id) if rotation_mode is not None else None
        cue_trace["rotation_instruction"] = str(rotation_mode.prompt_label) if rotation_mode is not None else None

    scene_entities.append(dict(cue_trace))

    winner_vertices_px = candidate_vertices_px_by_label[str(query.winner_label)]
    annotation = graph_point_set_annotation_artifacts(
        points_by_label=ordered_vertex_label_map(winner_vertices_px),
        graph_origin=context.graph_origin,
        graph_spacing=int(context.graph_spacing),
        witness_type="winning_transformed_polygon_vertices",
    )
    required_labels = list(annotation.get("required_labels", []))

    render_map = {
        "image_id": "img0",
        "reference_vertices_graph": [[int(round(point[0])), int(round(point[1]))] for point in reference_vertices_graph],
        "candidate_vertices_graph_by_label": {
            str(label): [[int(round(point[0])), int(round(point[1]))] for point in vertices]
            for label, vertices in candidate_vertices_graph_by_label.items()
        },
        "candidate_centers_graph_by_label": {
            str(label): [round(float(center[0]), 3), round(float(center[1]), 3)]
            for label, center in candidate_centers_graph_by_label.items()
        },
        "winner_label": str(query.winner_label),
        "object_label_centers": dict(object_label_centers),
        "cue": dict(cue_trace),
    }
    if translation_vector is not None:
        render_map["translation_vector_graph"] = [int(translation_vector[0]), int(translation_vector[1])]

    return _RenderedTransformationScene(
        reference_vertices_graph=tuple(reference_vertices_graph),
        winner_vertices_graph=tuple(candidate_vertices_graph_by_label[str(query.winner_label)]),
        reference_vertices_px=tuple(reference_vertices_px),
        winner_vertices_px=tuple(winner_vertices_px),
        candidate_vertices_graph_by_label=dict(candidate_vertices_graph_by_label),
        candidate_vertices_px_by_label=dict(candidate_vertices_px_by_label),
        candidate_centers_graph_by_label=dict(candidate_centers_graph_by_label),
        candidate_centers_px_by_label=dict(candidate_centers_px_by_label),
        winner_label=str(query.winner_label),
        reference_center_graph=(float(reference_center_graph[0]), float(reference_center_graph[1])),
        cue_kind=str(cue_trace["type"]),
        cue_trace=dict(cue_trace),
        scene_entities=scene_entities,
        render_map=render_map,
        annotation=annotation,
        answer_value=str(query.winner_label),
        object_label_centers=dict(object_label_centers),
        required_annotation_labels=required_labels,
        rotation_mode=(str(rotation_mode.mode_id) if rotation_mode is not None else None),
        rotation_prompt_label=(str(rotation_mode.prompt_label) if rotation_mode is not None else None),
        translation_vector=translation_vector,
    )


def compose_transformation_scene(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
    transform_rule: str,
    seed_namespace: str,
) -> TransformationSceneBundle:
    """Sample, render, and finalize one transform-selection gallery scene."""

    runtime_params = dict(params)
    runtime_params["transform_rule"] = str(transform_rule)
    resolved = _resolve_axes(int(instance_seed), params=runtime_params)
    scene_rng = spawn_rng(int(instance_seed), f"{seed_namespace}.scene")

    last_error: Exception | None = None
    for _ in range(max(1, int(max_attempts))):
        context = resolve_graph_scene_context(
            scene_rng,
            instance_seed=int(instance_seed),
            scene_id=SCENE_ID,
            params=runtime_params,
            render_defaults=_RENDER_DEFAULTS,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
            fallback_canvas_min=_DEFAULTS.canvas_size_min,
            fallback_canvas_max=_DEFAULTS.canvas_size_max,
            fallback_cells_min=_DEFAULTS.graph_cells_min,
            fallback_cells_max=_DEFAULTS.graph_cells_max,
        )
        line_width = sample_int_render_param(
            scene_rng,
            params=runtime_params,
            render_defaults=_RENDER_DEFAULTS,
            key="line_width",
            fallback=_DEFAULTS.line_width,
            minimum_value=1,
        )
        label_font_size_px = int(
            runtime_params.get(
                "label_font_size_px",
                resolve_scene_label_font_size_px(
                    canvas_size=int(context.canvas_size),
                    graph_spacing=int(context.graph_spacing),
                    scene_scale=int(context.scene_scale),
                    min_px=int(group_default(_RENDER_DEFAULTS, "label_font_size_min", _DEFAULTS.label_font_size_min)),
                    max_px=int(group_default(_RENDER_DEFAULTS, "label_font_size_max", _DEFAULTS.label_font_size_max)),
                ),
            )
        )
        label_stroke_width = sample_int_render_param(
            scene_rng,
            params=runtime_params,
            render_defaults=_RENDER_DEFAULTS,
            key="label_stroke_width",
            fallback=_DEFAULTS.label_stroke_width,
            minimum_value=1,
        )
        label_stroke_width_px = max(1, int(label_stroke_width) * int(context.scene_scale))
        image, draw, background_meta = make_graph_scene_canvas(
            instance_seed=int(instance_seed),
            context=context,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
        )
        shape_style = sample_geometry_shape_style(
            scene_rng,
            params=runtime_params,
            render_defaults=_RENDER_DEFAULTS,
            anchor_colors=extract_background_anchor_colors(background_meta),
        )
        padding_px = float(
            runtime_params.get(
                "cue_line_padding_px",
                group_default(_RENDER_DEFAULTS, "cue_line_padding_px", _DEFAULTS.cue_line_padding_px),
            )
        )
        try:
            rendered_scene = _sample_transformation_scene(
                scene_rng,
                query=resolved,
                context=context,
                padding_px=float(padding_px),
                line_width=int(line_width) * int(context.scene_scale),
                label_font_size_px=int(label_font_size_px),
                label_stroke_width=int(label_stroke_width_px),
                reference_label_gap_px=int(
                    runtime_params.get(
                        "reference_label_gap_px",
                        group_default(_RENDER_DEFAULTS, "reference_label_gap_px", _DEFAULTS.reference_label_gap_px),
                    )
                ),
                cue_label_gap_px=int(
                    runtime_params.get(
                        "cue_label_gap_px",
                        group_default(_RENDER_DEFAULTS, "cue_label_gap_px", _DEFAULTS.cue_label_gap_px),
                    )
                ),
                cue_dash_px=int(runtime_params.get("cue_dash_px", group_default(_RENDER_DEFAULTS, "cue_dash_px", _DEFAULTS.cue_dash_px))),
                cue_gap_px=int(runtime_params.get("cue_gap_px", group_default(_RENDER_DEFAULTS, "cue_gap_px", _DEFAULTS.cue_gap_px))),
                cue_arrow_head_length_px=int(
                    runtime_params.get(
                        "cue_arrow_head_length_px",
                        group_default(_RENDER_DEFAULTS, "cue_arrow_head_length_px", _DEFAULTS.cue_arrow_head_length_px),
                    )
                )
                * int(context.scene_scale),
                cue_arrow_head_width_px=int(
                    runtime_params.get(
                        "cue_arrow_head_width_px",
                        group_default(_RENDER_DEFAULTS, "cue_arrow_head_width_px", _DEFAULTS.cue_arrow_head_width_px),
                    )
                )
                * int(context.scene_scale),
                cue_point_radius_px=int(
                    runtime_params.get(
                        "cue_point_radius_px",
                        group_default(_RENDER_DEFAULTS, "cue_point_radius_px", _DEFAULTS.cue_point_radius_px),
                    )
                ),
                object_label_offset_px=float(
                    runtime_params.get(
                        "object_label_offset_px",
                        group_default(_RENDER_DEFAULTS, "object_label_offset_px", _DEFAULTS.object_label_offset_px),
                    )
                ),
                draw=draw,
                shape_style=shape_style,
                render_canvas_size=int(context.canvas_size) * int(context.scene_scale),
                params=runtime_params,
            )
            annotation_value = rendered_scene.annotation.get("annotation_value", [])
            if not isinstance(annotation_value, list) or not annotation_value:
                raise RuntimeError("geometry transformation annotation must include winning polygon pixel points")
            final_image, background_meta_final, post_noise_meta = finalize_graph_scene_image(
                image,
                instance_seed=int(instance_seed),
                context=context,
                background_meta=background_meta,
                noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
            )
            return TransformationSceneBundle(
                resolved=resolved,
                rendered_scene=rendered_scene,
                image=final_image,
                context=context,
                background_meta=dict(background_meta_final),
                post_noise_meta=dict(post_noise_meta),
                shape_style_trace=dict(shape_style.to_trace_dict()),
                line_width=int(line_width),
                label_font_size_px=int(label_font_size_px),
                label_stroke_width_px=int(label_stroke_width_px),
            )
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"failed to compose shape-reference transform scene for {transform_rule}") from last_error
