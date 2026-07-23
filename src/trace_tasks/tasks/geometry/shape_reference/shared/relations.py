"""Geometry relation-matching task on one shared graph-paper scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, required_group_defaults, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.text_rendering import resolve_scene_label_font_size_px
from trace_tasks.tasks.shared.variant_sampling import (
    apply_balanced_variant_sampling,
    resolve_variant,
)

SCENE_ID = "shape_reference"
from trace_tasks.tasks.geometry.shared.comparison import resolve_comparison_winner_label
from trace_tasks.tasks.geometry.shared.background_defaults import load_geometry_background_defaults
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame
from trace_tasks.tasks.geometry.shared.labeled_point_annotation import graph_point_set_annotation_artifacts
from trace_tasks.tasks.geometry.shared.multi_polygon_scene import PolygonSceneObject, draw_polygon_objects
from trace_tasks.tasks.geometry.shared.noise_defaults import load_geometry_noise_defaults
from trace_tasks.tasks.geometry.shared.option_count import resolve_geometry_option_count
from trace_tasks.tasks.geometry.shared.polygon_scene_helpers import (
    draw_reference_polygon,
    graph_polygon_inside_canvas,
    pixel_point_from_graph_units,
    pixel_polygon_from_graph_units,
)
from trace_tasks.tasks.geometry.shared.polygon_transformations import (
    Polygon,
    RIGID_TRANSFORM_RECIPE_IDS,
    apply_rigid_transform_recipe,
    ordered_vertex_label_map,
    polygons_are_congruent,
    polygons_are_similar,
    sample_asymmetric_polygon_template,
    scale_polygon,
    scale_polygon_non_uniform,
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

SCENE_NAMESPACE = "shape_reference_relation_match"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("triangle", "quadrilateral")
RELATION_RULES: Tuple[str, ...] = ("congruent", "similar")
COMPATIBILITY: Dict[str, Sequence[str]] = {
    "triangle": RELATION_RULES,
    "quadrilateral": RELATION_RULES,
}

POST_IMAGE_BACKGROUND_DEFAULTS = load_geometry_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_geometry_noise_defaults(scene_id=SCENE_ID)

_RIGID_RECIPES: Tuple[str, ...] = tuple(RIGID_TRANSFORM_RECIPE_IDS)



@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for geometry similarity scenes."""

    canvas_size_min: int = 720
    canvas_size_max: int = 800
    graph_cells_min: int = 44
    graph_cells_max: int = 44
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
    cue_line_padding_px: int = 18
    reference_center: Tuple[int, int] = (-11, 1)
    candidate_label_pool: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
    candidate_slots: Tuple[Tuple[int, int], ...] = (
        (4, 9),
        (16, 9),
        (4, 0),
        (16, 0),
        (4, -9),
        (16, -9),
    )
    candidate_min_gap_graph: float = 0.5
    similar_scale_support: Tuple[int, ...] = (1, 2)
    distractor_non_uniform_pairs: Tuple[Tuple[int, int], ...] = (
        (2, 1),
        (1, 2),
    )


@dataclass(frozen=True)
class _ResolvedQuery:
    """Resolved scene and relation axes plus answer-label support for one instance."""

    scene_variant: str
    relation_rule: str
    winner_label: str
    scene_variant_probabilities: Dict[str, float]
    relation_rule_probabilities: Dict[str, float]
    winner_label_probabilities: Dict[str, float]
    candidate_label_pool: Tuple[str, ...]
    candidate_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class _RenderedSimilarityScene:
    """Task-local scene package with all rendered polygons and trace artifacts."""

    reference_vertices_graph: Polygon
    reference_vertices_px: Polygon
    candidate_vertices_graph_by_label: Dict[str, Polygon]
    candidate_vertices_px_by_label: Dict[str, Polygon]
    candidate_centers_graph_by_label: Dict[str, Tuple[float, float]]
    candidate_centers_px_by_label: Dict[str, Tuple[float, float]]
    winner_label: str
    winner_vertices_graph: Polygon
    winner_vertices_px: Polygon
    annotation: Dict[str, Any]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    object_label_centers: Dict[str, List[float]]
    required_annotation_labels: List[str]


@dataclass(frozen=True)
class SimilaritySceneBundle:
    """Rendered relation-match scene plus neutral metadata for public tasks."""

    resolved: _ResolvedQuery
    rendered_scene: _RenderedSimilarityScene
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
_WINNER_LABEL_BALANCE_SALT = 29017


def _full_probability_map(supported: Sequence[str], probabilities: Mapping[str, float]) -> Dict[str, float]:
    """Expand one restricted probability map over the full supported variant domain."""

    positive = {str(key): float(value) for key, value in probabilities.items()}
    return {
        str(key): float(positive.get(str(key), 0.0))
        for key in supported
    }


def _candidate_slot_support(params: Mapping[str, Any], *, candidate_count: int) -> Tuple[Tuple[int, int], ...]:
    """Resolve the visible candidate slot centers used by the relation scene."""

    raw_support = params.get("candidate_slots", group_default(_GEN_DEFAULTS, "candidate_slots", _DEFAULTS.candidate_slots))
    slots: List[Tuple[int, int]] = []
    for value in raw_support:
        if not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError("candidate_slots entries must be [x, y] graph-unit pairs")
        slot = (int(value[0]), int(value[1]))
        if slot not in slots:
            slots.append(slot)
    if len(slots) < int(candidate_count):
        raise ValueError("geometry relation matching cannot provide the visible candidate count")
    return tuple(slots[: int(candidate_count)])


def _visible_candidate_labels(label_pool: Sequence[str], *, winner_label: str, candidate_count: int) -> Tuple[str, ...]:
    labels = tuple(str(label) for label in label_pool[: int(candidate_count)])
    if str(winner_label) in set(labels):
        return labels
    return tuple([str(winner_label), *[label for label in labels if str(label) != str(winner_label)]])[: int(candidate_count)]


def _polygon_graph_bbox(vertices: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return one graph-coordinate bbox for a polygon."""

    min_x = min(float(point[0]) for point in vertices)
    max_x = max(float(point[0]) for point in vertices)
    min_y = min(float(point[1]) for point in vertices)
    max_y = max(float(point[1]) for point in vertices)
    return (float(min_x), float(min_y), float(max_x), float(max_y))


def _candidate_bboxes_have_clearance(
    polygons_by_label: Mapping[str, Sequence[Tuple[float, float]]],
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


def _decoupled_winner_label_params(*, params: Mapping[str, Any]) -> Mapping[str, Any]:
    """No-op hook for winner-label cycling call sites."""

    return params


def _resolve_axes(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
    """Resolve scene and relation axes plus balanced answer-label support."""

    axis_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.axes")
    scene_supported = [str(value) for value in SUPPORTED_SCENE_VARIANTS]
    relation_supported = [str(value) for value in RELATION_RULES]
    compatibility_map = {
        str(scene): tuple(str(rule) for rule in rules)
        for scene, rules in COMPATIBILITY.items()
    }
    explicit_scene = params.get("scene_variant")
    explicit_relation = params.get("relation_rule")
    if explicit_scene is not None and str(explicit_scene) not in set(scene_supported):
        raise ValueError(f"unsupported scene_variant: {explicit_scene}")
    if explicit_relation is not None and str(explicit_relation) not in set(relation_supported):
        raise ValueError(f"unsupported relation_rule: {explicit_relation}")

    if explicit_relation is None:
        raise ValueError("shape-reference relation rule must be resolved by the public task")
    relation_rule = str(explicit_relation)
    relation_probs = _full_probability_map(relation_supported, {relation_rule: 1.0})

    if explicit_scene is not None:
        scene_variant = str(explicit_scene)
        if str(relation_rule) not in set(compatibility_map.get(scene_variant, ())):
            raise ValueError(f"incompatible scene/relation combination: {scene_variant} + {relation_rule}")
        scene_probs = _full_probability_map(scene_supported, {scene_variant: 1.0})
    else:
        allowed_scenes = [
            scene
            for scene in scene_supported
            if str(relation_rule) in set(compatibility_map.get(scene, ()))
        ]
        selected_scene, restricted_scene_probs = resolve_variant(
            axis_rng,
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            supported_variants=allowed_scenes,
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
        )
        scene_variant = apply_balanced_variant_sampling(
            instance_seed=int(instance_seed),
            params=params,
            gen_defaults=_GEN_DEFAULTS,
            selected_variant=str(selected_scene),
            variant_probabilities=restricted_scene_probs,
            supported_variants=allowed_scenes,
            balance_flag_key="balanced_scene_variant_sampling",
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
            sampling_namespace=f"{SCENE_NAMESPACE}.scene_variant.{relation_rule}",
        )
        scene_probs = _full_probability_map(scene_supported, restricted_scene_probs)

    label_pool = tuple(
        str(label).upper()
        for label in params.get(
            "candidate_label_pool",
            group_default(_GEN_DEFAULTS, "candidate_label_pool", _DEFAULTS.candidate_label_pool),
        )
    )
    if len(label_pool) not in {4, 6} or len(set(label_pool)) != len(label_pool):
        raise ValueError("geometry relation matching requires four or six unique candidate labels")
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
        selection_namespace=f"{SCENE_NAMESPACE}.winner_label.{scene_variant}.{relation_rule}",
    )
    visible_labels = _visible_candidate_labels(
        label_pool,
        winner_label=str(winner_label),
        candidate_count=int(candidate_count),
    )
    return _ResolvedQuery(
        scene_variant=str(scene_variant),
        relation_rule=str(relation_rule),
        winner_label=str(winner_label),
        scene_variant_probabilities=dict(scene_probs),
        relation_rule_probabilities=dict(relation_probs),
        winner_label_probabilities=dict(winner_probs),
        candidate_label_pool=tuple(visible_labels),
        candidate_count_probabilities=dict(candidate_count_probabilities),
    )


def _candidate_is_match(
    *,
    relation_rule: str,
    reference_vertices_graph: Sequence[Tuple[float, float]],
    candidate_vertices_graph: Sequence[Tuple[float, float]],
) -> bool:
    """Return whether one candidate satisfies the requested similarity predicate."""

    if str(relation_rule) == "congruent":
        return bool(polygons_are_congruent(reference_vertices_graph, candidate_vertices_graph))
    if str(relation_rule) == "similar":
        return bool(polygons_are_similar(reference_vertices_graph, candidate_vertices_graph))
    raise ValueError(f"unsupported relation_rule: {relation_rule}")


def _build_matching_candidate(
    rng,
    *,
    relation_rule: str,
    template: Polygon,
    slot_center: Tuple[int, int],
    force_large_scale: bool,
    similar_scale_support: Sequence[int],
) -> Polygon:
    """Build one matching candidate polygon by construction."""

    scale_factor = 1
    if str(relation_rule) == "similar":
        allowed_scales = [int(value) for value in similar_scale_support if int(value) > 0]
        if not allowed_scales:
            raise ValueError("similar_scale_support must be non-empty")
        if bool(force_large_scale) and 2 in allowed_scales:
            scale_factor = 2
        else:
            scale_factor = int(rng.choice(list(allowed_scales)))
    base_vertices = scale_polygon(template, factor=int(scale_factor))
    rigid_recipe = str(rng.choice(list(_RIGID_RECIPES)))
    transformed = apply_rigid_transform_recipe(base_vertices, recipe=str(rigid_recipe))
    return translate_polygon(transformed, dx=int(slot_center[0]), dy=int(slot_center[1]))


def _build_distractor_candidate(
    rng,
    *,
    relation_rule: str,
    template: Polygon,
    slot_center: Tuple[int, int],
    non_uniform_pairs: Sequence[Tuple[int, int]],
) -> Polygon:
    """Build one distractor candidate polygon."""

    if str(relation_rule) == "congruent" and bool(rng.randint(0, 1)):
        base_vertices = scale_polygon(template, factor=2)
    else:
        scale_x, scale_y = rng.choice(list(non_uniform_pairs))
        base_vertices = scale_polygon_non_uniform(template, scale_x=int(scale_x), scale_y=int(scale_y))
    rigid_recipe = str(rng.choice(list(_RIGID_RECIPES)))
    transformed = apply_rigid_transform_recipe(base_vertices, recipe=str(rigid_recipe))
    return translate_polygon(transformed, dx=int(slot_center[0]), dy=int(slot_center[1]))


def _sample_similarity_scene(
    rng,
    *,
    query: _ResolvedQuery,
    context: GraphSceneContext,
    padding_px: float,
    line_width: int,
    label_font_size_px: int,
    label_stroke_width: int,
    reference_label_gap_px: int,
    object_label_offset_px: float,
    draw,
    shape_style,
    render_canvas_size: int,
    params: Mapping[str, Any],
) -> _RenderedSimilarityScene:
    """Sample and render one full relation-match scene."""

    template = sample_asymmetric_polygon_template(str(query.scene_variant), rng, profile="compact")
    reference_center = tuple(
        int(value)
        for value in params.get("reference_center", group_default(_GEN_DEFAULTS, "reference_center", _DEFAULTS.reference_center))
    )
    reference_vertices_graph = translate_polygon(template, dx=int(reference_center[0]), dy=int(reference_center[1]))
    if not graph_polygon_inside_canvas(reference_vertices_graph, context=context, padding_px=float(padding_px)):
        raise ValueError("reference polygon fell outside the graph-paper canvas")
    reference_vertices_px = pixel_polygon_from_graph_units(reference_vertices_graph, context=context)

    candidate_slots = _candidate_slot_support(params, candidate_count=len(query.candidate_label_pool))
    winner_label = str(query.winner_label)
    highlight_large_label = None

    candidate_vertices_graph_by_label: Dict[str, Polygon] = {}
    candidate_vertices_px_by_label: Dict[str, Polygon] = {}
    candidate_centers_graph_by_label: Dict[str, Tuple[float, float]] = {}
    candidate_centers_px_by_label: Dict[str, Tuple[float, float]] = {}
    objects: List[PolygonSceneObject] = []
    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "reference_polygon",
            "entity_type": "reference_polygon",
            "scene_variant": str(query.scene_variant),
            "vertices_graph": [[int(round(point[0])), int(round(point[1]))] for point in reference_vertices_graph],
            "center_graph": [int(reference_center[0]), int(reference_center[1])],
        }
    ]

    similar_scale_support = tuple(
        int(value)
        for value in params.get(
            "similar_scale_support",
            group_default(_GEN_DEFAULTS, "similar_scale_support", _DEFAULTS.similar_scale_support),
        )
    )
    non_uniform_pairs = tuple(
        (int(value[0]), int(value[1]))
        for value in params.get(
            "distractor_non_uniform_pairs",
            group_default(_GEN_DEFAULTS, "distractor_non_uniform_pairs", _DEFAULTS.distractor_non_uniform_pairs),
        )
    )

    for label, slot_center in zip(query.candidate_label_pool, candidate_slots):
        is_match = bool(str(label) == winner_label)
        candidate_vertices_graph: Polygon | None = None
        for _ in range(24):
            if is_match:
                candidate_vertices_graph = _build_matching_candidate(
                    rng,
                    relation_rule=str(query.relation_rule),
                    template=template,
                    slot_center=slot_center,
                    force_large_scale=bool(str(label) == str(highlight_large_label)),
                    similar_scale_support=similar_scale_support,
                )
                if not _candidate_is_match(
                    relation_rule=str(query.relation_rule),
                    reference_vertices_graph=reference_vertices_graph,
                    candidate_vertices_graph=candidate_vertices_graph,
                ):
                    candidate_vertices_graph = None
                    continue
            else:
                candidate_vertices_graph = _build_distractor_candidate(
                    rng,
                    relation_rule=str(query.relation_rule),
                    template=template,
                    slot_center=slot_center,
                    non_uniform_pairs=non_uniform_pairs,
                )
                if _candidate_is_match(
                    relation_rule=str(query.relation_rule),
                    reference_vertices_graph=reference_vertices_graph,
                    candidate_vertices_graph=candidate_vertices_graph,
                ):
                    candidate_vertices_graph = None
                    continue
            if graph_polygon_inside_canvas(candidate_vertices_graph, context=context, padding_px=float(padding_px)):
                break
            candidate_vertices_graph = None
        if candidate_vertices_graph is None:
            raise ValueError("failed to place a similarity candidate inside the graph-paper canvas")

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
                "matches_query": bool(is_match),
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

    winner_vertices_px = candidate_vertices_px_by_label[winner_label]
    annotation = graph_point_set_annotation_artifacts(
        points_by_label=ordered_vertex_label_map(winner_vertices_px),
        graph_origin=context.graph_origin,
        graph_spacing=int(context.graph_spacing),
        witness_type="winning_relation_polygon_vertices",
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
        "object_label_centers": dict(object_label_centers),
        "winner_label": winner_label,
    }

    return _RenderedSimilarityScene(
        reference_vertices_graph=tuple(reference_vertices_graph),
        reference_vertices_px=tuple(reference_vertices_px),
        candidate_vertices_graph_by_label=dict(candidate_vertices_graph_by_label),
        candidate_vertices_px_by_label=dict(candidate_vertices_px_by_label),
        candidate_centers_graph_by_label=dict(candidate_centers_graph_by_label),
        candidate_centers_px_by_label=dict(candidate_centers_px_by_label),
        winner_label=winner_label,
        winner_vertices_graph=tuple(candidate_vertices_graph_by_label[winner_label]),
        winner_vertices_px=tuple(winner_vertices_px),
        annotation=annotation,
        scene_entities=list(scene_entities),
        render_map=render_map,
        object_label_centers=dict(object_label_centers),
        required_annotation_labels=required_labels,
    )


def compose_similarity_scene(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
    relation_rule: str,
    seed_namespace: str,
) -> SimilaritySceneBundle:
    """Sample, render, and finalize one relation-match gallery scene."""

    runtime_params = dict(params)
    runtime_params["relation_rule"] = str(relation_rule)
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
            rendered_scene = _sample_similarity_scene(
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
            final_image, background_meta_final, post_noise_meta = finalize_graph_scene_image(
                image,
                instance_seed=int(instance_seed),
                context=context,
                background_meta=background_meta,
                noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
            )
            return SimilaritySceneBundle(
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

    raise RuntimeError(f"failed to compose shape-reference relation scene for {relation_rule}") from last_error
