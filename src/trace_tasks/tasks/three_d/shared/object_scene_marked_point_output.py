"""Task-output assembly helpers for 3D marked-point object-scene tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from PIL import Image

from .canvas import render_params_canvas_metadata
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import required_group_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from .object_scene import (
    POINT_LABELS,
    SCENE_ID,
    SUPPORTED_SCENE_VARIANTS,
    _RenderParams,
    _camera_yaw_band_for_instance,
    _resolve_render_params,
    render_object_scene_3d,
)
from .task_support import resolve_axis_variant as _resolve_axis_variant
from .task_support import resolve_count as _resolve_count


def build_marked_point_object_scene_output(
    *,
    objective_name: str,
    task_domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    dataset: Mapping[str, Any],
    branch_key: str,
    scene_variant: str,
    point_count: int,
    render_params: _RenderParams,
    prompt_defaults_config: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    scene_probabilities: Mapping[str, float],
    point_count_probabilities: Mapping[str, float],
    dynamic_slots: Mapping[str, Any],
    scene_kind: str,
    count_params: Mapping[str, Any],
    relation_fields: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    draw_marked_points_fn: Callable[..., tuple[Image.Image, Dict[str, Any], list[Dict[str, Any]]]],
    bbox_union_fn: Callable[..., list[float]],
    include_reference_render_map: bool = False,
) -> TaskOutput:
    """Render one marked-point object-scene instance and assemble its point annotation payload."""
    background, background_meta = make_background_canvas(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=background_defaults,
    )
    rendered_scene = render_object_scene_3d(
        background,
        dataset=dataset,
        render_params=render_params,
        draw_candidate_labels=False,
        compute_single_annotation=False,
    )
    marked_image, marker_render_map, marker_entities = draw_marked_points_fn(
        rendered_scene.image,
        marked_points=dataset["marked_points"],
        render_params=render_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        marked_image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=noise_defaults,
    )

    prompt_defaults = required_group_defaults(
        prompt_defaults_config,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context=f"prompt defaults for {objective_name}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(task_domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    answer_label = str(dataset["answer_label"])
    answer_gt = TypedValue(type="option_letter", value=str(answer_label))
    annotation_point = list(dataset["answer_point_px"])
    annotation_gt = TypedValue(type="point", value=list(annotation_point))
    scene_bbox = bbox_union_fn(
        rendered_scene.scene_bbox_px,
        *[bbox for bbox in marker_render_map["marked_point_bboxes_px"].values()],
    )
    scene_entities = [*rendered_scene.entities, *marker_entities]
    answer_support = [str(label) for label in POINT_LABELS[: int(point_count)]]

    query_params: Dict[str, Any] = {
        "query_id": str(branch_key),
        "query_id_probabilities": dict(query_probabilities),
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probabilities),
        "point_count": int(point_count),
        "point_count_probabilities": dict(point_count_probabilities),
        "answer_support": list(answer_support),
    }
    query_params.update(dict(count_params))

    render_map: Dict[str, Any] = {
        "image_id": "img0",
        "scene_bbox_px": list(scene_bbox),
        "room_bbox_px": list(rendered_scene.room_bbox_px),
        "object_bboxes_px": {str(key): list(value) for key, value in rendered_scene.object_bboxes_px.items()},
        "object_centers_px": {str(key): list(value) for key, value in rendered_scene.object_centers_px.items()},
        "context_object_bboxes_px": {
            str(key): list(value) for key, value in rendered_scene.context_object_bboxes_px.items()
        },
        "context_object_centers_px": {
            str(key): list(value) for key, value in rendered_scene.context_object_centers_px.items()
        },
        **dict(marker_render_map),
        "selected_point_px": list(dataset["answer_point_px"]),
    }
    if include_reference_render_map:
        reference_id = str(dataset["reference_object_id"])
        render_map["reference_object_bbox_px"] = list(rendered_scene.object_bboxes_px[reference_id])
        render_map["reference_object_center_px"] = list(rendered_scene.object_centers_px[reference_id])

    execution_trace: Dict[str, Any] = {
        "query_id": str(branch_key),
        "scene_variant": str(scene_variant),
        "point_count": int(point_count),
        "point_specs": [],
        "marked_points": [dict(point) for point in dataset["marked_points"]],
        "context_object_specs": [dict(spec) for spec in dataset["context_object_specs"]],
        "object_specs": [dict(spec) for spec in dataset["object_specs"]],
        "answer_label": str(answer_label),
        "answer_point_id": str(dataset["answer_point_id"]),
        "answer_marker_id": str(dataset["answer_marker_id"]),
        "answer_point_px": list(dataset["answer_point_px"]),
        "camera": dict(dataset["camera"]),
        "projection_frame": dict(dataset["projection_frame"]),
        "question_format": str(branch_key),
        "view_family": "synthetic_perspective_3d_marked_points",
        "solver_trace": dict(dataset["solver_trace"]),
    }
    execution_trace.update(dict(execution_extra))

    trace_payload = {
        "scene_ir": {
            "scene_kind": str(scene_kind),
            "entities": [dict(entity) for entity in scene_entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "point_count": int(point_count),
                "answer_point_id": str(dataset["answer_point_id"]),
                "answer_label": str(answer_label),
                "view_family": "synthetic_perspective_3d_marked_points",
                **dict(relation_fields),
            },
        },
        "query_spec": {
            "query_id": str(branch_key),
            "template_id": str(prompt_defaults["bundle_id"]),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": query_params,
        },
        "render_spec": {
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(image.height),
            "scene_canvas_preset": str(render_params.canvas_preset),
            "scene_canvas_width": int(render_params.canvas_width),
            "scene_canvas_height": int(render_params.canvas_height),
            "scene_canvas_policy": str(render_params.canvas_policy),
            **render_params_canvas_metadata(render_params),
            "final_canvas_width": int(image.width),
            "final_canvas_height": int(image.height),
            "final_canvas_pixels": int(image.width) * int(image.height),
            "coord_space": "pixel",
            "scene_variant": str(scene_variant),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "camera": dict(dataset["camera"]),
            "projection_frame": dict(dataset["projection_frame"]),
            "label_font_size_px": int(render_params.label_font_size_px),
            "marker_radius_px": int(max(12.0, float(render_params.marker_radius_px) * 0.66)),
        },
        "render_map": render_map,
        "execution_trace": execution_trace,
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": {
            "type": "point",
            "point": list(annotation_point),
            "pixel_point": list(annotation_point),
        },
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
        query_id=str(branch_key),
    )


def build_marked_point_object_scene_task_output(
    *,
    public_name: str,
    public_domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    branch_options: Sequence[str],
    generation_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults_config: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    exact_point_count: int,
    dataset_builder: Callable[..., Mapping[str, Any]],
    dynamic_slots_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    scene_kind: str,
    relation_fields_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    execution_extra_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    witness_symbolic_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    draw_marked_points_fn: Callable[..., tuple[Image.Image, Dict[str, Any], list[Dict[str, Any]]]],
    bbox_union_fn: Callable[..., list[float]],
    camera_yaw_band: Tuple[float, float] | None = None,
    object_count_default: int = 6,
    object_count_lower: int = 5,
    object_count_upper: int = 7,
    context_object_count_default: int = 1,
    context_object_count_lower: int = 1,
    context_object_count_upper: int = 2,
) -> TaskOutput:
    """Resolve shared marked-point generation axes and assemble one task output."""

    branch_key, branch_probabilities = _resolve_axis_variant(
        params,
        task_id=str(public_name),
        gen_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(value) for value in branch_options),
        explicit_key="query_id",
        weights_key="query_id_weights",
        balance_flag_key="balanced_query_id_sampling",
        axis_namespace="query_id",
    )
    scene_variant, scene_probabilities = _resolve_axis_variant(
        params,
        task_id=str(public_name),
        gen_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )
    point_count, point_count_probabilities = _resolve_count(
        params,
        task_id=str(public_name),
        gen_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        key="point_count",
        default_min=int(exact_point_count),
        default_max=int(exact_point_count),
        lower=int(exact_point_count),
        upper=int(exact_point_count),
    )
    object_count, object_count_probabilities = _resolve_count(
        params,
        task_id=str(public_name),
        gen_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        key="object_count",
        default_min=int(object_count_default),
        default_max=int(object_count_default),
        lower=int(object_count_lower),
        upper=int(object_count_upper),
    )
    context_object_count, context_object_count_probabilities = _resolve_count(
        params,
        task_id=str(public_name),
        gen_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        key="context_object_count",
        default_min=int(context_object_count_default),
        default_max=int(context_object_count_default),
        lower=int(context_object_count_lower),
        upper=int(context_object_count_upper),
    )
    render_params = _resolve_render_params(
        params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{public_name}.canvas",
    )
    dataset = dataset_builder(
        branch_key=str(branch_key),
        scene_variant=str(scene_variant),
        point_count=int(point_count),
        object_count=int(object_count),
        context_object_count=int(context_object_count),
        render_params=render_params,
        instance_seed=int(instance_seed),
        camera_yaw_band=camera_yaw_band,
    )
    return build_marked_point_object_scene_output(
        objective_name=str(public_name),
        task_domain=str(public_domain),
        instance_seed=int(instance_seed),
        params=params,
        dataset=dataset,
        branch_key=str(branch_key),
        scene_variant=str(scene_variant),
        point_count=int(point_count),
        render_params=render_params,
        prompt_defaults_config=prompt_defaults_config,
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        query_probabilities=branch_probabilities,
        scene_probabilities=scene_probabilities,
        point_count_probabilities=point_count_probabilities,
        dynamic_slots=dynamic_slots_fn(dataset),
        scene_kind=str(scene_kind),
        count_params={
            "object_count": int(object_count),
            "object_count_probabilities": dict(object_count_probabilities),
            "context_object_count": int(context_object_count),
            "context_object_count_probabilities": dict(context_object_count_probabilities),
        },
        relation_fields=relation_fields_fn(dataset),
        execution_extra=execution_extra_fn(dataset),
        witness_symbolic=witness_symbolic_fn(dataset),
        draw_marked_points_fn=draw_marked_points_fn,
        bbox_union_fn=bbox_union_fn,
    )


def generate_marked_point_object_scene_task_with_retries(
    *,
    public_name: str,
    public_domain: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    branch_options: Sequence[str],
    generation_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults_config: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    exact_point_count: int,
    dataset_builder: Callable[..., Mapping[str, Any]],
    dynamic_slots_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    scene_kind: str,
    relation_fields_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    execution_extra_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    witness_symbolic_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    draw_marked_points_fn: Callable[..., tuple[Image.Image, Dict[str, Any], list[Dict[str, Any]]]],
    bbox_union_fn: Callable[..., list[float]],
) -> TaskOutput:
    """Retry marked-point scene sampling with a stable camera yaw band for the public instance seed."""

    last_error: Exception | None = None
    camera_yaw_band = _camera_yaw_band_for_instance(int(instance_seed))
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = (
            int(instance_seed)
            if attempt_index == 0
            else int(spawn_rng(int(instance_seed), f"{public_name}.attempt_seed.{attempt_index}").randrange(1, 2**62))
        )
        try:
            return build_marked_point_object_scene_task_output(
                public_name=str(public_name),
                public_domain=str(public_domain),
                instance_seed=int(attempt_seed),
                params=params,
                branch_options=branch_options,
                generation_defaults=generation_defaults,
                render_defaults=render_defaults,
                prompt_defaults_config=prompt_defaults_config,
                background_defaults=background_defaults,
                noise_defaults=noise_defaults,
                exact_point_count=int(exact_point_count),
                dataset_builder=dataset_builder,
                dynamic_slots_fn=dynamic_slots_fn,
                scene_kind=str(scene_kind),
                relation_fields_fn=relation_fields_fn,
                execution_extra_fn=execution_extra_fn,
                witness_symbolic_fn=witness_symbolic_fn,
                draw_marked_points_fn=draw_marked_points_fn,
                bbox_union_fn=bbox_union_fn,
                camera_yaw_band=camera_yaw_band,
            )
        except Exception as exc:  # pragma: no cover - unlucky sampling fallback.
            last_error = exc
    raise RuntimeError(f"{public_name} failed to generate a valid scene after {max_attempts} attempts: {last_error}")


__all__ = [
    "build_marked_point_object_scene_output",
    "build_marked_point_object_scene_task_output",
    "generate_marked_point_object_scene_task_with_retries",
]
