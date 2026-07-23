"""Runtime assembly for coordinate-plane relation-count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.config_defaults import group_default, required_group_defaults
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from trace_tasks.tasks.shared.text_rendering import resolve_scene_label_font_size_px
from trace_tasks.tasks.geometry.shared.graph_rendering import graph_paper_grid_from_frame
from trace_tasks.tasks.geometry.shared.render_variation import sample_int_render_param
from trace_tasks.tasks.geometry.shared.shape_style import extract_background_anchor_colors, sample_geometry_shape_style
from trace_tasks.tasks.geometry.shared.single_object_scene import (
    finalize_graph_scene_image,
    make_graph_scene_canvas,
    resolve_graph_scene_context,
)
from trace_tasks.core.seed import spawn_rng

from .state import (
    _DEFAULTS,
    _GEN_DEFAULTS,
    _PROMPT_DEFAULTS,
    _RENDER_DEFAULTS,
    POST_IMAGE_BACKGROUND_DEFAULTS,
    POST_IMAGE_NOISE_DEFAULTS,
    _ResolvedQuery,
    _RenderedCoordinateScene,
)
from .relations import (
    _execution_trace_for_trace,
    _selection_params_for_trace,
    _resolve_count_target,
    _resolve_scene_render_params,
    _sample_collinear_count_scene,
    _sample_point_in_shape_scene,
    _sample_quadrant_count_scene,
    _sample_segment_count_scene,
    _scene_relations_for_trace,
)


@dataclass(frozen=True)
class RelationArtifacts:
    """Prompt/render/trace artifacts before final public task binding."""

    query: _ResolvedQuery
    rendered_scene: _RenderedCoordinateScene
    prompt_artifacts: Any
    image: Any
    trace_payload: Dict[str, Any]
    task_versions: Dict[str, str]
    text_style: Dict[str, int]
    point_style: Dict[str, int]


def relation_query(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    operation_key: str,
    operation_key_probabilities: Mapping[str, float],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
) -> _ResolvedQuery:
    """Resolve the count target and scene-local query state."""

    count_rng = spawn_rng(int(instance_seed), f"coordinate_plane_relation.{scene_variant}.target_count")
    target_count, target_probabilities = _resolve_count_target(
        count_rng,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        operation_key=str(operation_key),
        params=params,
    )
    label_pool = tuple()
    if str(scene_variant) == "quadrant_points":
        label_pool = tuple(
            str(value).upper()
            for value in params.get(
                "quadrant_candidate_labels",
                group_default(_GEN_DEFAULTS, "quadrant_candidate_labels", _DEFAULTS.quadrant_candidate_labels),
            )
        )
    return _ResolvedQuery(
        scene_variant=str(scene_variant),
        operation_key=str(operation_key),
        scene_variant_probabilities={str(key): float(value) for key, value in scene_variant_probabilities.items()},
        operation_key_probabilities={str(key): float(value) for key, value in operation_key_probabilities.items()},
        target_count=int(target_count),
        target_count_probabilities=dict(target_probabilities),
        label_pool=tuple(label_pool),
    )


def _render_relation_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    query: _ResolvedQuery,
    max_attempts: int,
) -> tuple[Any, Any, Dict[str, Any], Dict[str, Any], Any, _RenderedCoordinateScene, int, int, int, int]:
    """Render the selected relation-count scene after target-count resolution."""

    scene_rng = spawn_rng(int(instance_seed), f"coordinate_plane_relation.{query.scene_variant}.scene")
    last_error: Exception | None = None

    for _ in range(max(1, int(max_attempts))):
        scene_render_params = _resolve_scene_render_params(str(query.scene_variant), params=params)
        context = resolve_graph_scene_context(
            scene_rng,
            instance_seed=int(instance_seed),
            scene_id="coordinate_plane",
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
            fallback_canvas_min=int(scene_render_params["canvas_size_min"]),
            fallback_canvas_max=int(scene_render_params["canvas_size_max"]),
            fallback_cells_min=int(scene_render_params["graph_cells_min"]),
            fallback_cells_max=int(scene_render_params["graph_cells_max"]),
            graph_style_overrides={
                "origin_fraction_x": float(scene_render_params.get("graph_origin_fraction_x", 0.5)),
                "origin_fraction_y": float(scene_render_params.get("graph_origin_fraction_y", 0.5)),
            },
        )
        line_width = sample_int_render_param(
            scene_rng,
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            key="line_width",
            fallback=_DEFAULTS.line_width,
            minimum_value=1,
        )
        point_radius_px = sample_int_render_param(
            scene_rng,
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            key="point_radius_px",
            fallback=_DEFAULTS.point_radius_px,
            minimum_value=1,
        )
        label_font_size_px = int(
            params.get(
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
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            key="label_stroke_width",
            fallback=_DEFAULTS.label_stroke_width,
            minimum_value=1,
        )
        label_stroke_width_scene = max(1, int(label_stroke_width) * int(context.scene_scale))
        image, draw, background_meta = make_graph_scene_canvas(
            instance_seed=int(instance_seed),
            context=context,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
        )
        shape_style = sample_geometry_shape_style(
            scene_rng,
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            anchor_colors=extract_background_anchor_colors(background_meta),
        )
        try:
            if str(query.scene_variant) == "segment_set":
                rendered_scene = _sample_segment_count_scene(
                    scene_rng,
                    query=query,
                    context=context,
                    draw=draw,
                    line_width=int(line_width) * int(context.scene_scale),
                    point_radius_px=int(point_radius_px) * int(context.scene_scale),
                    label_font_size_px=int(label_font_size_px),
                    label_stroke_width=int(label_stroke_width_scene),
                    label_offset_px=float(params.get("label_offset_px", group_default(_RENDER_DEFAULTS, "label_offset_px", _DEFAULTS.label_offset_px))),
                    shape_style=shape_style,
                    params=params,
                    render_canvas_size=int(context.canvas_size) * int(context.scene_scale),
                )
            elif str(query.scene_variant) == "line_points":
                rendered_scene = _sample_collinear_count_scene(
                    scene_rng,
                    query=query,
                    context=context,
                    draw=draw,
                    point_radius_px=int(point_radius_px) * int(context.scene_scale),
                    label_font_size_px=int(label_font_size_px),
                    label_stroke_width=int(label_stroke_width_scene),
                    label_offset_px=float(params.get("label_offset_px", group_default(_RENDER_DEFAULTS, "label_offset_px", _DEFAULTS.label_offset_px))),
                    point_radius_scale=float(scene_render_params.get("point_radius_scale", 1.4)),
                    shape_style=shape_style,
                    params=params,
                    render_canvas_size=int(context.canvas_size) * int(context.scene_scale),
                )
            elif str(query.scene_variant) == "quadrant_points":
                rendered_scene = _sample_quadrant_count_scene(
                    scene_rng,
                    query=query,
                    context=context,
                    draw=draw,
                    point_radius_px=int(point_radius_px) * int(context.scene_scale),
                    label_font_size_px=int(label_font_size_px),
                    label_stroke_width=int(label_stroke_width_scene),
                    label_offset_px=float(params.get("label_offset_px", group_default(_RENDER_DEFAULTS, "label_offset_px", _DEFAULTS.label_offset_px))),
                    point_radius_scale=float(scene_render_params.get("point_radius_scale", 1.5)),
                    reference_cross_scale=float(scene_render_params.get("reference_cross_scale", 2.6)),
                    shape_style=shape_style,
                    render_canvas_size=int(context.canvas_size) * int(context.scene_scale),
                )
            elif str(query.scene_variant) == "polygon_lattice":
                rendered_scene = _sample_point_in_shape_scene(
                    scene_rng,
                    query=query,
                    context=context,
                    draw=draw,
                    line_width=int(line_width) * int(context.scene_scale),
                    point_radius_px=int(point_radius_px) * int(context.scene_scale),
                    label_font_size_px=int(label_font_size_px),
                    label_stroke_width=int(label_stroke_width_scene),
                    label_offset_px=float(params.get("label_offset_px", group_default(_RENDER_DEFAULTS, "label_offset_px", _DEFAULTS.label_offset_px))),
                    shape_style=shape_style,
                    params=params,
                    render_canvas_size=int(context.canvas_size) * int(context.scene_scale),
                )
            else:
                raise ValueError(f"unsupported coordinate scene_variant: {query.scene_variant}")
            return (
                image,
                context,
                background_meta,
                shape_style.to_trace_dict(),
                rendered_scene,
                rendered_scene,
                int(line_width),
                int(point_radius_px),
                int(label_font_size_px),
                int(label_stroke_width_scene),
            )
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError("failed to generate coordinate-plane relation scene") from last_error


def build_relation_artifacts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    query: _ResolvedQuery,
    output_operation_key: str,
    output_query_probabilities: Mapping[str, float],
    prompt_query_key: str,
    max_attempts: int,
) -> RelationArtifacts:
    """Render one relation-count scene and build prompt/trace artifacts."""

    (
        image,
        context,
        background_meta,
        shape_style_trace,
        rendered_scene,
        _,
        line_width,
        point_radius_px,
        label_font_size_px,
        label_stroke_width_scene,
    ) = _render_relation_scene(
        instance_seed=int(instance_seed),
        params=params,
        query=query,
        max_attempts=int(max_attempts),
    )
    image, background_meta_final, post_noise_meta = finalize_graph_scene_image(
        image,
        instance_seed=int(instance_seed),
        context=context,
        background_meta=background_meta,
        noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
    )

    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="prompt defaults for coordinate-plane relation count",
    )
    prompt_selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id="coordinate_plane",
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    selection_params = _selection_params_for_trace(query)
    selection_params["semantic_operation"] = str(query.operation_key)
    selection_params["operation_key"] = str(output_operation_key)
    selection_params["operation_key_probabilities"] = {
        str(key): float(value) for key, value in output_query_probabilities.items()
    }
    execution_trace = _execution_trace_for_trace(query, rendered_scene)
    execution_trace["semantic_operation"] = str(query.operation_key)
    execution_trace["operation_key"] = str(output_operation_key)
    execution_trace["operation_key_probabilities"] = dict(selection_params["operation_key_probabilities"])
    scene_relations = _scene_relations_for_trace(query, rendered_scene)
    scene_relations["semantic_operation"] = str(query.operation_key)
    scene_relations["operation_key"] = str(output_operation_key)
    trace_payload = {
        "scene_ir": {
            "scene_kind": "geometry_coordinate_relation",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": dict(scene_relations),
        },
        "query_spec": {
            "operation_key": str(output_operation_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": dict(selection_params),
        },
        "render_spec": {
            "canvas_size": int(context.canvas_size),
            "coord_space": "pixel",
            "background_style": dict(background_meta_final),
            "post_image_noise": dict(post_noise_meta),
            "shape_style": dict(shape_style_trace),
            "text_style": {
                "font_size_px": int(label_font_size_px),
                "stroke_width_px": int(label_stroke_width_scene),
            },
            "point_style": {
                "radius_px": int(point_radius_px),
            },
            "graph_coordinate_frame": dict(context.graph_frame),
            "graph_paper_grid": graph_paper_grid_from_frame(context.graph_frame),
            **dict(context.graph_layout_metadata),
            "scene_variant": str(query.scene_variant),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": dict(execution_trace),
        "witness_symbolic": dict(rendered_scene.witness_symbolic),
        "projected_annotation": dict(rendered_scene.projected_annotation),
    }
    return RelationArtifacts(
        query=query,
        rendered_scene=rendered_scene,
        prompt_artifacts=prompt_artifacts,
        image=image,
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        text_style={
            "font_size_px": int(label_font_size_px),
            "stroke_width_px": int(label_stroke_width_scene),
        },
        point_style={"radius_px": int(point_radius_px)},
    )


__all__ = ["RelationArtifacts", "build_relation_artifacts", "relation_query"]
