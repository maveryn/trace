"""Neutral visual and prompt plumbing for color-gradient public tasks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.puzzles.shared.scene_style import (
    make_puzzle_scene_background,
    resolve_puzzle_scene_style,
)
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import with_puzzle_unit_size_jitter
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family

from .shared.defaults import (
    font_trace_record,
    post_noise_policy_trace,
    resolve_render_params,
    sample_label_font,
    sample_weighted_variant,
)
from .shared.annotations import item_bbox_annotation
from .shared.prompts import (
    object_description_for_scene,
    render_color_gradient_prompt_artifacts,
)
from .shared.output import build_trace_payload, json_ready
from .shared.state import DOMAIN, SCENE_ID, SCENE_VARIANTS, RenderParams, RenderedScene

_NOISE_DEFAULTS = load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.0)


def retry_color_gradient_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    build_once: Callable[..., TaskOutput],
) -> TaskOutput:
    """Run neutral retry plumbing around a task-owned construction hook."""

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            return build_once(
                params=params,
                instance_seed=attempt_seed,
                attempt_limit=int(max_attempts),
            )
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("color-gradient generation failed")


def run_color_gradient_case(
    *,
    task_id: str,
    supported_query_ids: tuple[str, ...],
    namespace: str,
    params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    build_output: Callable[..., TaskOutput],
) -> TaskOutput:
    """Load defaults, select the fixed public branch, and retry construction."""

    generation_defaults, rendering_defaults, prompt_defaults = (
        load_scene_generation_rendering_prompt_defaults(
            DOMAIN,
            SCENE_ID,
            task_id=str(task_id),
        )
    )
    public_branch, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=supported_query_ids,
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(task_id),
        namespace=f"{namespace}.branch",
    )
    if str(public_branch) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported color-gradient query_id: {public_branch}")

    return retry_color_gradient_case(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        build_once=lambda **kwargs: build_output(
            **kwargs,
            generation_defaults=generation_defaults,
            rendering_defaults=rendering_defaults,
            prompt_defaults=prompt_defaults,
            public_branch=str(public_branch),
            branch_probabilities=dict(branch_probabilities),
        ),
    )


def prepare_color_gradient_visual_case(
    *,
    dataset: Any,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    render_scene: Callable[..., RenderedScene],
    prompt_task_key: str,
    prompt_query_key: str,
    completion_prompt: bool,
) -> dict[str, Any]:
    """Resolve shared visual axes, render the scene, and build prompt artifacts."""

    scene_variant, scene_probabilities = sample_weighted_variant(
        params=params,
        generation_defaults=generation_defaults,
        instance_seed=int(instance_seed),
        support=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        namespace=f"{namespace}.scene_variant",
    )
    render_params = resolve_render_params(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
    )
    scene_style, scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    render_params = replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in scene_style.panel_fill_rgb),
        panel_border_rgb=tuple(int(value) for value in scene_style.panel_border_rgb),
        swatch_border_rgb=tuple(int(value) for value in scene_style.grid_rgb),
        notebook_line_rgb=tuple(int(value) for value in scene_style.notebook_line_rgb),
    )
    font_family = sample_label_font(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
        namespace=namespace,
    )
    background, background_meta = make_puzzle_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=scene_style,
    )
    with temporary_default_font_family(str(font_family)):
        rendered_scene = render_scene(
            background=background,
            dataset=dataset,
            scene_variant=str(scene_variant),
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=_NOISE_DEFAULTS,
    )
    object_description = object_description_for_scene(
        str(scene_variant),
        completion=bool(completion_prompt),
    )
    prompt_artifacts = render_color_gradient_prompt_artifacts(
        prompt_defaults={
            **dict(prompt_defaults),
            "object_description": str(object_description),
        },
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    return {
        "image": image,
        "rendered_scene": rendered_scene,
        "render_params": render_params,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probabilities),
        "background_meta": dict(background_meta),
        "scene_style_meta": dict(scene_style_meta),
        "post_noise_meta": dict(post_noise_meta),
        "font_family": str(font_family),
        "prompt_artifacts": prompt_artifacts,
    }


def render_spec_from_visual(
    visual: Mapping[str, Any],
    *,
    label_scope: str,
) -> dict[str, Any]:
    """Build the common render-spec trace section from rendered visual metadata."""

    render_params: RenderParams = visual["render_params"]
    rendered_scene = visual["rendered_scene"]
    return {
        "scene_id": SCENE_ID,
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(visual["scene_variant"]),
        "background_style": dict(visual["background_meta"]),
        "scene_style": dict(visual["scene_style_meta"]),
        "post_image_noise": dict(visual["post_noise_meta"]),
        "post_image_noise_policy": post_noise_policy_trace(),
        "scene_bbox_px": [int(value) for value in rendered_scene.scene_bbox_px],
        "render_params": {
            "swatch_size_px": int(render_params.swatch_size_px),
            "swatch_gap_px": int(render_params.swatch_gap_px),
            "label_chip_size_px": int(render_params.label_chip_size_px),
        },
        "label_style": {
            "font": font_trace_record(
                str(visual["font_family"]),
                scope=str(label_scope),
            ),
            "chip_fill_rgb": [255, 255, 255],
            "chip_outline_rgb": [36, 42, 52],
            "label_fill_rgb": [28, 32, 38],
        },
        "unit_size_jitter": dict(render_params.unit_size_jitter),
    }


def render_map_from_visual(visual: Mapping[str, Any]) -> dict[str, Any]:
    """Build the common pixel render map for a rendered color-gradient scene."""

    render_params: RenderParams = visual["render_params"]
    rendered_scene = visual["rendered_scene"]
    return with_puzzle_unit_size_jitter(
        {
            "image_id": "img0",
            "scene_bbox_px": [int(value) for value in rendered_scene.scene_bbox_px],
            "cell_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.cell_bbox_map.items()
            },
            "item_bboxes_px": {
                str(key): list(value)
                for key, value in rendered_scene.item_bbox_map.items()
            },
            "annotation_source": "item_bboxes_px",
        },
        render_params.unit_size_jitter,
    )


def build_color_gradient_trace_payload(
    *,
    visual: Mapping[str, Any],
    query_spec: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    answer_gt: TypedValue,
    annotation_artifacts: Any,
    prompt_query_key: str,
    label_scope: str,
    relation_fields: Mapping[str, Any],
    execution_fields: Mapping[str, Any],
) -> dict[str, Any]:
    """Wrap task-owned fields in the common color-gradient trace envelope."""

    rendered_scene = visual["rendered_scene"]
    return build_trace_payload(
        scene_ir={
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "public_query": "single",
                "prompt_query_key": str(prompt_query_key),
                "scene_id": SCENE_ID,
                "scene_variant": str(visual["scene_variant"]),
                **dict(relation_fields),
            },
        },
        query_spec=dict(query_spec),
        render_spec=render_spec_from_visual(visual, label_scope=str(label_scope)),
        render_map=render_map_from_visual(visual),
        execution_trace={
            "question_format": str(prompt_query_key),
            **dict(execution_fields),
        },
        witness_symbolic={
            "type": str(annotation_artifacts.annotation_type),
            "value": annotation_artifacts.value,
        },
        projected_annotation=dict(annotation_artifacts.projected_annotation),
        answer_gt=answer_gt.__dict__,
        annotation_gt=annotation_artifacts.annotation_gt.__dict__,
        prompt_defaults=prompt_defaults,
        prompt_artifacts=visual["prompt_artifacts"],
    )


def _dataset_field_map(
    dataset: Any,
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    return {
        str(field_name): json_ready(getattr(dataset, str(field_name)))
        for field_name in field_names
    }


def build_color_gradient_label_task_output(
    *,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    public_branch: str,
    branch_probabilities: Mapping[str, float],
    attempt_limit: int,
    namespace: str,
    prompt_task_key: str,
    prompt_query_key: str,
    completion_prompt: bool,
    label_scope: str,
    sample_dataset: Callable[..., Any],
    render_scene: Callable[..., RenderedScene],
    answer_label_field: str,
    annotation_item_field: str,
    query_field_names: tuple[str, ...],
    relation_field_names: tuple[str, ...],
    execution_field_names: tuple[str, ...],
    support_item_field: str,
    context_item_field: str | None = None,
) -> TaskOutput:
    """Use task-owned hooks to build one color-gradient label task output."""

    dataset = sample_dataset(
        params=params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        namespace=f"{namespace}.sample",
    )
    visual = prepare_color_gradient_visual_case(
        dataset=dataset,
        params=params,
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        namespace=namespace,
        render_scene=render_scene,
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        completion_prompt=bool(completion_prompt),
    )
    rendered_scene = visual["rendered_scene"]
    answer_value = str(getattr(dataset, str(answer_label_field)))
    annotation_item_id = str(getattr(dataset, str(annotation_item_field)))
    answer_gt = TypedValue(type="option_letter", value=answer_value)
    annotation_artifacts = item_bbox_annotation(
        rendered_scene.item_bbox_map,
        annotation_item_id,
    )
    prompt_artifacts = visual["prompt_artifacts"]
    base_params = {
        "query_id_probabilities": dict(branch_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "scene_id": SCENE_ID,
        "scene_variant": str(visual["scene_variant"]),
        "scene_variant_probabilities": dict(visual["scene_variant_probabilities"]),
        "attempt_limit": int(attempt_limit),
    }
    resolved_params = {
        **base_params,
        **_dataset_field_map(dataset, query_field_names),
    }
    spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(public_branch),
        params=resolved_params,
    )
    support_item_id = str(getattr(dataset, str(support_item_field)))
    execution_payload = {
        **resolved_params,
        **_dataset_field_map(dataset, execution_field_names),
        "supporting_item_ids": [support_item_id],
        "answer_value": answer_value,
    }
    if context_item_field is not None:
        execution_payload["context_item_ids"] = [
            str(getattr(dataset, str(context_item_field)))
        ]

    payload = build_color_gradient_trace_payload(
        visual=visual,
        query_spec=spec,
        prompt_defaults=prompt_defaults,
        answer_gt=answer_gt,
        annotation_artifacts=annotation_artifacts,
        prompt_query_key=str(prompt_query_key),
        label_scope=str(label_scope),
        relation_fields=_dataset_field_map(dataset, relation_field_names),
        execution_fields=execution_payload,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=visual["image"],
        image_id="img0",
        trace_payload=payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "build_color_gradient_label_task_output",
    "build_color_gradient_trace_payload",
    "prepare_color_gradient_visual_case",
    "render_map_from_visual",
    "render_spec_from_visual",
    "retry_color_gradient_case",
    "run_color_gradient_case",
]
