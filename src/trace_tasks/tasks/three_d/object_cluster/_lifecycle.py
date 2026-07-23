"""Neutral render/prompt/output lifecycle for object-cluster public tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.scene_config import get_domain_defaults, get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.background import make_background_canvas
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import (
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from trace_tasks.tasks.three_d.shared.canvas import render_params_canvas_metadata
from trace_tasks.tasks.three_d.shared.object_scene import (
    render_object_scene_3d,
    resolve_object_scene_render_params,
)

from .shared.defaults import PROMPT_COLOR_RGB, SCENE_ID
from .shared.objects import clip_object_bboxes_to_canvas, rendered_bboxes_are_valid, rendered_layout_stats
from .shared.relations import semantic_color_label
from .shared.state import BuildRequest


_DOMAIN_DEFAULTS = get_domain_defaults("three_d")
_VISUAL_DEFAULTS = _DOMAIN_DEFAULTS.get("visual", {}) if isinstance(_DOMAIN_DEFAULTS, Mapping) else {}
_BACKGROUND_DEFAULTS = _VISUAL_DEFAULTS.get("background", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}
_NOISE_DEFAULTS = _VISUAL_DEFAULTS.get("noise", {}) if isinstance(_VISUAL_DEFAULTS, Mapping) else {}


def _task_defaults(task_identifier: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Resolve scene defaults for one public task without branching on its identity."""

    scene_defaults = get_scene_defaults("three_d", SCENE_ID)
    gen_defaults, render_defaults, prompt_defaults = split_scene_generation_rendering_prompt_defaults(
        scene_defaults if isinstance(scene_defaults, Mapping) else {},
        task_id=str(task_identifier),
    )
    return dict(gen_defaults), dict(render_defaults), dict(prompt_defaults)


def run_object_cluster_lifecycle(
    instance_seed: int,
    *,
    params: Dict[str, Any],
    max_attempts: int,
    task_identifier: str,
    build_request: BuildRequest,
) -> TaskOutput:
    """Retry one task-local request builder, then render and bind output."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = (
            int(instance_seed)
            if attempt_index == 0
            else int(spawn_rng(int(instance_seed), f"{task_identifier}.attempt_seed.{attempt_index}").randrange(1, 2**62))
        )
        try:
            return _run_once(
                int(attempt_seed),
                params=params,
                task_identifier=str(task_identifier),
                build_request=build_request,
            )
        except Exception as exc:  # pragma: no cover - stochastic retry guard.
            last_error = exc
    raise RuntimeError(f"{task_identifier} failed to generate a valid scene after {max_attempts} attempts: {last_error}")


def _run_once(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    task_identifier: str,
    build_request: BuildRequest,
) -> TaskOutput:
    """Build a request, render its scene, project annotation, and return TaskOutput."""

    gen_defaults, render_defaults, prompt_defaults_config = _task_defaults(str(task_identifier))
    render_params = resolve_object_scene_render_params(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{task_identifier}.canvas",
    )
    request = build_request(
        int(instance_seed),
        params,
        gen_defaults,
        prompt_defaults_config,
        render_params,
    )
    dataset = dict(request.dataset)
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=params,
        default_config=_BACKGROUND_DEFAULTS,
    )
    rendered = render_object_scene_3d(
        background,
        dataset=dataset,
        render_params=render_params,
        draw_candidate_labels=False,
        compute_single_annotation=False,
    )
    if not rendered_bboxes_are_valid(
        rendered.object_bboxes_px,
        rendered.object_centers_px,
        width=int(rendered.image.width),
        height=int(rendered.image.height),
        object_count=int(dataset["object_count"]),
        object_specs=[dict(spec) for spec in dataset["object_specs"]],
        annotation_object_ids=[str(object_id) for object_id in dataset["target_object_ids"]],
    ):
        raise ValueError("rendered object cluster failed readability constraints")
    raw_object_bboxes_px = {str(object_id): list(bbox) for object_id, bbox in rendered.object_bboxes_px.items()}
    public_object_bboxes_px = clip_object_bboxes_to_canvas(
        raw_object_bboxes_px,
        width=int(rendered.image.width),
        height=int(rendered.image.height),
    )
    layout_stats = rendered_layout_stats(
        object_bboxes_px=public_object_bboxes_px,
        object_centers_px=rendered.object_centers_px,
        width=int(rendered.image.width),
        height=int(rendered.image.height),
        object_specs=[dict(spec) for spec in dataset["object_specs"]],
    )
    raw_layout_stats = rendered_layout_stats(
        object_bboxes_px=raw_object_bboxes_px,
        object_centers_px=rendered.object_centers_px,
        width=int(rendered.image.width),
        height=int(rendered.image.height),
        object_specs=[dict(spec) for spec in dataset["object_specs"]],
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )

    target_object_ids = [str(object_id) for object_id in dataset["target_object_ids"]]
    target_bboxes = [list(public_object_bboxes_px[str(object_id)]) for object_id in target_object_ids]
    if bool(request.keyed_annotation):
        role_object_ids = dataset.get("role_object_ids", {})
        left_ids = [str(object_id) for object_id in role_object_ids.get("left_operand", [])]
        right_ids = [str(object_id) for object_id in role_object_ids.get("right_operand", [])]
        annotation_value = {
            "left_operand": [list(public_object_bboxes_px[str(object_id)]) for object_id in left_ids],
            "right_operand": [list(public_object_bboxes_px[str(object_id)]) for object_id in right_ids],
        }
        annotation_gt = TypedValue(type="bbox_set_map", value=dict(annotation_value))
        projected_annotation = {
            "type": "bbox_set_map",
            "bbox_set_map": dict(annotation_value),
            "pixel_bbox_set_map": dict(annotation_value),
        }
        annotation_render_map = {
            "operand_object_bboxes_px": {
                "left_operand": {str(object_id): list(public_object_bboxes_px[str(object_id)]) for object_id in left_ids},
                "right_operand": {str(object_id): list(public_object_bboxes_px[str(object_id)]) for object_id in right_ids},
            }
        }
    else:
        annotation_gt = TypedValue(type="bbox_set", value=[list(bbox) for bbox in target_bboxes])
        projected_annotation = {
            "type": "bbox_set",
            "bbox_set": [list(bbox) for bbox in target_bboxes],
            "pixel_bbox_set": [list(bbox) for bbox in target_bboxes],
        }
        annotation_render_map = {}

    prompt_defaults = required_group_defaults(
        prompt_defaults_config,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context=f"prompt defaults for {task_identifier}",
    )
    target_spec = dict(dataset.get("target_spec", {}))
    target_color_name = str(dataset.get("target_color_name") or "")
    target_color_names = [str(color) for color in dataset.get("target_color_names", [])]
    prompt_slots = {
        "target_shape_type": str(dataset.get("target_shape_type") or ""),
        "target_object_name": str(dataset.get("target_object_name") or ""),
        "target_object_plural": str(dataset.get("target_object_plural") or ""),
        "target_object_union_phrase": str(dataset.get("target_object_union_phrase") or ""),
        "target_color_name": semantic_color_label(target_color_name) if target_color_name in PROMPT_COLOR_RGB else target_color_name,
        "target_color_names": ", ".join(semantic_color_label(color) if color in PROMPT_COLOR_RGB else color for color in target_color_names),
        "target_property_phrase": str(target_spec.get("target_property_prompt_phrase") or dataset.get("target_property_phrase") or ""),
        "target_property_singular": str(dataset.get("target_property_singular") or ""),
        "left_operand_phrase": str(target_spec.get("left_operand_prompt_phrase") or dataset.get("left_operand_phrase") or ""),
        "right_operand_phrase": str(target_spec.get("right_operand_prompt_phrase") or dataset.get("right_operand_phrase") or ""),
        **dict(request.prompt_slots),
    }
    prompt_selection = render_scene_prompt_variants(
        domain="three_d",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(request.prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=prompt_slots,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    answer_value = int(dataset["answer_value"])
    answer_gt = TypedValue(type="integer", value=int(answer_value))

    render_map = {
        "image_id": "img0",
        "scene_bbox_px": list(rendered.scene_bbox_px),
        "room_bbox_px": list(rendered.room_bbox_px),
        "object_bboxes_px": dict(public_object_bboxes_px),
        "raw_object_bboxes_px": dict(raw_object_bboxes_px),
        "object_centers_px": dict(rendered.object_centers_px),
        "target_object_bboxes_px": {
            str(object_id): list(public_object_bboxes_px[str(object_id)])
            for object_id in target_object_ids
        },
        "target_object_centers_px": {
            str(object_id): list(rendered.object_centers_px[str(object_id)])
            for object_id in target_object_ids
        },
    }
    render_map.update(annotation_render_map)
    trace_payload = {
        "scene_ir": {
            "scene_kind": str(dataset["scene_kind"]),
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(request.scene_variant),
                "object_count": int(dataset["object_count"]),
                "countable_object_count": int(dataset["countable_object_count"]),
                "target_spec": dict(dataset["target_spec"]),
                "target_property_phrase": str(dataset["target_property_phrase"]),
                "target_count": int(dataset["target_count"]),
                "answer_value": int(answer_value),
                "target_object_ids": list(target_object_ids),
                "role_object_ids": dict(dataset["role_object_ids"]),
                "cluster_count": int(dataset["cluster_count"]),
                "cluster_compactness": float(dataset["cluster_compactness"]),
                "composition_offset": dict(dataset["composition_offset"]),
            },
        },
        "query_spec": {
            "query_id": str(request.external_query),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "query_id": str(request.external_query),
                "internal_query_id": str(request.prompt_query_key),
                "query_id_probabilities": dict(request.query_probabilities),
                "scene_variant": str(request.scene_variant),
                "scene_variant_probabilities": dict(request.scene_probabilities),
                "cluster_count": int(dataset["cluster_count"]),
                "cluster_compactness": float(dataset["cluster_compactness"]),
                "composition_offset": dict(dataset["composition_offset"]),
                "object_count": int(dataset["object_count"]),
                "object_count_probabilities": dict(request.count_probabilities.get("object_count_probabilities", {})),
                "target_count": int(dataset["target_count"]),
                "target_count_probabilities": dict(request.count_probabilities.get("target_count_probabilities", {})),
                "left_operand_count": request.count_probabilities.get("left_operand_count"),
                "left_operand_count_probabilities": dict(request.count_probabilities.get("left_operand_count_probabilities", {})),
                "right_operand_count": request.count_probabilities.get("right_operand_count"),
                "right_operand_count_probabilities": dict(request.count_probabilities.get("right_operand_count_probabilities", {})),
                "target_spec": dict(dataset["target_spec"]),
                "target_shape_type": dataset.get("target_shape_type"),
                "target_shape_types": list(dataset.get("target_shape_types", [])),
                "target_shape_type_probabilities": dict(request.count_probabilities.get("target_shape_probabilities", {})),
                "target_color_name": dataset.get("target_color_name"),
                "target_color_names": list(dataset.get("target_color_names", [])),
                "target_color_name_probabilities": dict(request.count_probabilities.get("target_color_probabilities", {})),
                "cluster_object_pool_size": request.count_probabilities.get("cluster_object_pool_size"),
                "semantic_color_palette": {str(key): list(value) for key, value in sorted(PROMPT_COLOR_RGB.items())},
            },
        },
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "scene_canvas_preset": str(render_params.canvas_preset),
            "scene_canvas_width": int(render_params.canvas_width),
            "scene_canvas_height": int(render_params.canvas_height),
            "scene_canvas_policy": str(render_params.canvas_policy),
            **render_params_canvas_metadata(render_params),
            "final_canvas_width": int(image.width),
            "final_canvas_height": int(image.height),
            "final_canvas_pixels": int(image.width) * int(image.height),
            "coord_space": "pixel",
            "scene_variant": str(request.scene_variant),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "camera": dict(dataset["camera"]),
            "projection_frame": dict(dataset["projection_frame"]),
            "room_extent": float(render_params.room_extent),
            "full_bleed_floor": bool(render_params.full_bleed_floor),
            "semantic_color_palette": {str(key): list(value) for key, value in sorted(PROMPT_COLOR_RGB.items())},
            "cluster_layout": dict(dataset["cluster_layout"]),
            "composition_offset": dict(dataset["composition_offset"]),
            "rendered_layout_stats": dict(layout_stats),
            "raw_rendered_layout_stats": dict(raw_layout_stats),
        },
        "render_map": dict(render_map),
        "execution_trace": {
            "query_id": str(request.external_query),
            "internal_query_id": str(request.prompt_query_key),
            "scene_variant": str(request.scene_variant),
            "object_count": int(dataset["object_count"]),
            "countable_object_count": int(dataset["countable_object_count"]),
            "target_count": int(dataset["target_count"]),
            "distractor_count": int(dataset.get("distractor_count", max(0, int(dataset["object_count"]) - int(dataset["target_count"])))),
            "cluster_composition_mode": dataset.get("cluster_composition_mode"),
            "cluster_count": int(dataset["cluster_count"]),
            "cluster_compactness": float(dataset["cluster_compactness"]),
            "cluster_layout": dict(dataset["cluster_layout"]),
            "composition_offset": dict(dataset["composition_offset"]),
            "rendered_layout_stats": dict(layout_stats),
            "answer_value": int(answer_value),
            "target_spec": dict(dataset["target_spec"]),
            "target_shape_type": dataset.get("target_shape_type"),
            "target_shape_types": list(dataset.get("target_shape_types", [])),
            "target_object_name": dataset.get("target_object_name"),
            "target_object_plural": dataset.get("target_object_plural"),
            "target_object_union_phrase": dataset.get("target_object_union_phrase"),
            "target_color_name": dataset.get("target_color_name"),
            "target_color_names": list(dataset.get("target_color_names", [])),
            "singleton_shape_types": list(dataset.get("singleton_shape_types", [])),
            "left_operand_phrase": dataset.get("left_operand_phrase"),
            "right_operand_phrase": dataset.get("right_operand_phrase"),
            "arithmetic_operation": dataset.get("arithmetic_operation"),
            "target_property_phrase": str(dataset["target_property_phrase"]),
            "target_object_ids": list(target_object_ids),
            "counted_object_ids": list(dataset.get("counted_object_ids", target_object_ids)),
            "role_object_ids": dict(dataset["role_object_ids"]),
            "object_specs": [dict(spec) for spec in dataset["object_specs"]],
            "shape_counts": dict(dataset["shape_counts"]),
            "color_counts": dict(dataset["color_counts"]),
            "property_counts": dict(dataset["property_counts"]),
            "count_role_counts": dict(dataset["count_role_counts"]),
            "camera": dict(dataset["camera"]),
            "projection_frame": dict(dataset["projection_frame"]),
            "question_format": str(request.external_query),
            "internal_question_format": str(request.prompt_query_key),
            "solver_trace": dict(dataset["solver_trace"]),
        },
        "witness_symbolic": {
            "type": "object_cluster_count_arithmetic_operand_sets" if bool(request.keyed_annotation) else "object_cluster_counted_object_set",
            "object_ids": list(target_object_ids),
            "role_object_ids": dict(dataset["role_object_ids"]),
            "target_spec": dict(dataset["target_spec"]),
            "target_property_phrase": str(dataset["target_property_phrase"]),
            "answer_value": int(answer_value),
        },
        "projected_annotation": dict(projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(request.external_query),
    )


__all__ = [
    "run_object_cluster_lifecycle",
]
