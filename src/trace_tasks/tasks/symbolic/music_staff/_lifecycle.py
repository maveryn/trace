"""Neutral lifecycle plumbing for symbolic music-staff public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ...base import TaskOutput
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ..shared.unit_size_jitter import with_symbolic_unit_size_jitter
from ..shared.visual_defaults import load_symbolic_noise_defaults

from .shared.annotations import bbox_center_point, project_bbox_map, project_bbox_set, round_bbox
from .shared.components import render_music_scene
from .shared.prompts import build_music_staff_prompt
from .shared.state import DOMAIN, SCENE_ID, MusicStaffDataset
from .shared.styles import resolve_background, resolve_render_params


POST_IMAGE_NOISE_DEFAULTS = load_symbolic_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class MusicStaffRuntime:
    """Private lifecycle inputs supplied by a public music-staff task file."""

    public_id: str
    supported_query_ids: tuple[str, ...]
    prompt_key: str
    annotation_schema: str
    dataset_builder: Callable[..., MusicStaffDataset]
    gen_defaults: Mapping[str, Any]
    render_defaults: Mapping[str, Any]
    prompt_defaults: Mapping[str, Any]
    dataset_branch_key: str | None = None
    prompt_query_key: str | None = None
    annotation_role_names: tuple[str, ...] = ()


def load_music_staff_defaults(task_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load scene-package defaults for one public music-staff task."""

    return load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=str(task_id))


def _coerce_answer(dataset: MusicStaffDataset) -> int | str:
    if str(dataset.answer_type) == "integer":
        return int(dataset.answer_value)
    return str(dataset.answer_value)


def _project_annotation(
    *,
    item_bboxes: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
    annotation_schema: str,
    role_names: Sequence[str] = (),
) -> tuple[TypedValue, dict[str, Any]]:
    """Project task-owned notation item ids into the requested annotation schema."""

    schema = str(annotation_schema)
    if schema == "point":
        if len(tuple(item_ids)) != 1:
            raise RuntimeError("point annotation requires exactly one music-staff item id")
        value = bbox_center_point(item_bboxes[str(tuple(item_ids)[0])])
        return TypedValue(type="point", value=value), {"type": "point", "point": value, "pixel_point": value, "value": value}
    if schema == "bbox":
        if len(tuple(item_ids)) != 1:
            raise RuntimeError("bbox annotation requires exactly one music-staff item id")
        value = round_bbox(item_bboxes[str(tuple(item_ids)[0])])
        return TypedValue(type="bbox", value=value), {"type": "bbox", "bbox": value, "value": value}
    if schema == "bbox_map":
        roles = tuple(str(role) for role in role_names)
        ids = tuple(str(item) for item in item_ids)
        if len(roles) != len(ids):
            raise RuntimeError("bbox_map annotation roles must match music-staff item ids")
        value = project_bbox_map(item_bboxes, dict(zip(roles, ids)))
        return TypedValue(type="bbox_map", value=value), {"type": "bbox_map", "bbox_map": value, "value": value}
    value = project_bbox_set(item_bboxes, item_ids)
    return TypedValue(type="bbox_set", value=value), {"type": "bbox_set", "bbox_set": value, "value": value}


def build_music_staff_output(
    *,
    task_id: str,
    dataset: MusicStaffDataset,
    public_query_id: str,
    selected_branch_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    prompt_key: str,
    prompt_query_key: str,
    annotation_schema: str,
    annotation_role_names: Sequence[str] = (),
) -> TaskOutput:
    """Render the scene and assemble a TaskOutput from task-owned bindings."""

    render_params = resolve_render_params(params, render_defaults, instance_seed=int(instance_seed))
    background, background_meta, scene_style, scene_style_meta = resolve_background(
        instance_seed=int(instance_seed),
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
    )
    rendered_scene = render_music_scene(
        background,
        spec=dataset.spec,
        render_params=render_params,
        scene_style=scene_style,
        instance_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt, prompt_variants, prompt_meta = build_music_staff_prompt(
        prompt_defaults=prompt_defaults,
        scene_variant=str(dataset.scene_variant),
        prompt_key=str(prompt_key),
        branch_key=str(prompt_query_key),
        prompt_slots=dict(dataset.prompt_slots),
        instance_seed=int(instance_seed),
        output_modes=PROMPT_OUTPUT_MODES,
    )

    answer_value = _coerce_answer(dataset)
    answer_gt = TypedValue(type=str(dataset.answer_type), value=answer_value)
    annotation_gt, projected_annotation = _project_annotation(
        item_bboxes=rendered_scene.item_bboxes,
        item_ids=tuple(dataset.annotation_item_ids),
        annotation_schema=str(annotation_schema),
        role_names=tuple(annotation_role_names),
    )
    query_params = {
        "query_id": str(public_query_id),
        "internal_query_id": str(dataset.branch_key),
        "prompt_query_key": str(prompt_query_key),
        "query_id_probabilities": dict(selected_branch_probabilities),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "target_answer_support": list(dataset.target_answer_support),
    }
    trace_payload = {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "query_id": str(public_query_id),
                "internal_query_id": str(dataset.branch_key),
                "prompt_query_key": str(prompt_query_key),
                "scene_id": SCENE_ID,
                "scene_variant": str(dataset.scene_variant),
                "answer_value": answer_value,
            },
        },
        "query_spec": {
            "query_id": str(public_query_id),
            "internal_query_id": str(dataset.branch_key),
            "prompt_query_key": str(prompt_query_key),
            "template_id": str(prompt_meta["bundle_id"]),
            "prompt_variant": dict(prompt_meta["prompt_variant"]),
            "prompt_variant_active_key": str(prompt_meta["prompt_variant_active_key"]),
            "prompt_variants": dict(prompt_meta["prompt_variants_for_trace"]),
            "params": dict(query_params),
        },
        "render_spec": {
            "scene_id": SCENE_ID,
            "canvas_width": int(render_params.canvas_width),
            "canvas_height": int(render_params.canvas_height),
            "coord_space": "pixel",
            "scene_variant": str(dataset.scene_variant),
            "scene_style": dict(scene_style_meta),
            "music_style": dict(rendered_scene.style_metadata),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
            "scene_bbox_px": list(rendered_scene.scene_bbox_px),
            "unit_size_jitter": dict(render_params.unit_size_jitter),
            "layout_jitter": dict(rendered_scene.layout_jitter),
        },
        "render_map": with_symbolic_unit_size_jitter(
            {
                "image_id": "img0",
                "scene_bbox_px": list(rendered_scene.scene_bbox_px),
                "item_bboxes_px": {str(key): list(value) for key, value in rendered_scene.item_bboxes.items()},
                "annotation_source": "item_bboxes_px",
                "layout_jitter": dict(rendered_scene.layout_jitter),
            },
            render_params.unit_size_jitter,
        ),
        "execution_trace": {
            **dict(query_params),
            "answer_value": answer_value,
            "answer_type": str(dataset.answer_type),
            "annotation_item_ids": [str(item) for item in dataset.annotation_item_ids],
            "annotation_schema": str(annotation_gt.type),
            "notation_metadata": dict(dataset.metadata),
            "question_format": str(dataset.branch_key),
        },
        "witness_symbolic": annotation_gt.to_dict(),
        "projected_annotation": dict(projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_gt.to_dict(),
    }
    return TaskOutput(
        prompt=str(prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_query_id),
        prompt_variants=dict(prompt_variants),
    )


def generate_music_staff_task(
    *,
    task_id: str,
    selected_branch: str,
    selected_branch_probabilities: Mapping[str, float],
    scene_variant: str,
    scene_variant_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    prompt_key: str,
    annotation_schema: str,
    dataset_builder: Callable[..., MusicStaffDataset],
    dataset_branch_key: str | None = None,
    prompt_query_key: str | None = None,
    annotation_role_names: Sequence[str] = (),
) -> TaskOutput:
    """Retry task-owned dataset construction, then render and package it."""

    last_error: Exception | None = None
    dataset: MusicStaffDataset | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            dataset = dataset_builder(
                branch_key=str(dataset_branch_key or selected_branch),
                instance_seed=int(instance_seed) + int(attempt_index),
                scene_variant=str(scene_variant),
                params=params,
                gen_defaults=gen_defaults,
            )
            break
        except Exception as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError("failed to generate music-staff symbolic instance") from last_error

    return build_music_staff_output(
        task_id=str(task_id),
        dataset=dataset,
        public_query_id=str(selected_branch),
        selected_branch_probabilities=dict(selected_branch_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=params,
        prompt_key=str(prompt_key),
        prompt_query_key=str(prompt_query_key or dataset.branch_key),
        annotation_schema=str(annotation_schema),
        annotation_role_names=tuple(annotation_role_names),
    )


def run_music_staff_runtime(
    runtime: MusicStaffRuntime,
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Apply common query/scene selection before invoking task-owned builders."""

    from ...shared.fixed_query import select_task_query_id

    from .shared.sampling import resolve_scene_variant

    selected_branch, branch_probs, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(runtime.supported_query_ids),
        default_query_id=tuple(runtime.supported_query_ids)[0],
        task_id=str(runtime.public_id),
    )
    scene_variant, scene_probs = resolve_scene_variant(
        task_params,
        runtime.gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(runtime.public_id),
    )
    return generate_music_staff_task(
        task_id=str(runtime.public_id),
        selected_branch=str(selected_branch),
        selected_branch_probabilities=branch_probs,
        scene_variant=str(scene_variant),
        scene_variant_probabilities=scene_probs,
        params=task_params,
        gen_defaults=runtime.gen_defaults,
        render_defaults=runtime.render_defaults,
        prompt_defaults=runtime.prompt_defaults,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        prompt_key=str(runtime.prompt_key),
        prompt_query_key=runtime.prompt_query_key,
        dataset_branch_key=runtime.dataset_branch_key,
        annotation_schema=str(runtime.annotation_schema),
        dataset_builder=runtime.dataset_builder,
        annotation_role_names=tuple(runtime.annotation_role_names),
    )


__all__ = [
    "DOMAIN",
    "MusicStaffRuntime",
    "SCENE_ID",
    "build_music_staff_output",
    "generate_music_staff_task",
    "load_music_staff_defaults",
    "run_music_staff_runtime",
]
