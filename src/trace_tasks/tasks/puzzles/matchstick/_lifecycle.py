"""Scene-private plumbing for matchstick rendering and visual axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.puzzles.shared.scene_style import resolve_puzzle_scene_style
from trace_tasks.tasks.puzzles.shared.visual_defaults import load_puzzle_noise_defaults
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.variant_sampling import resolve_variant

from .shared.annotations import bbox_map_artifacts
from .shared.output import (
    build_matchstick_trace_payload,
    build_render_map,
    build_render_spec,
)
from .shared.prompts import build_matchstick_prompt_artifacts
from .shared.rendering import (
    font_trace_record,
    make_scene_background,
    resolve_render_params,
    sample_matchstick_font,
)
from .shared.rules import number_text, number_transition_allowed
from .shared.state import (
    NumberDataset,
    RenderParams,
    RenderedScene,
    SCENE_ID,
    SCENE_VARIANTS,
)


@dataclass(frozen=True)
class MatchstickRenderContext:
    """Rendered image and metadata shared by matchstick public tasks."""

    rendered_scene: RenderedScene
    image: Image.Image
    render_params: RenderParams
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    font_meta: Dict[str, Any]


@dataclass(frozen=True)
class BoundMatchstickOutput:
    """Task-owned answer, annotation, and trace fields after rendering."""

    prompt_query_key: str
    prompt_dynamic_slots: Dict[str, Any]
    query_params: Dict[str, Any]
    answer_gt: TypedValue
    annotation_artifacts: AnnotationArtifacts
    annotation_source: str
    scene_relations: Dict[str, Any]
    execution_trace: Dict[str, Any]
    witness_symbolic: Dict[str, Any]


DatasetBuilder = Callable[..., Any]
OutputBinder = Callable[..., BoundMatchstickOutput]


def build_bound_output(
    *,
    dataset: Any,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
    answer_gt: TypedValue,
    annotation_artifacts: AnnotationArtifacts,
    annotation_source: str,
    prompt_query_key: str,
    scene_extra: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    prompt_dynamic_slots: Mapping[str, Any] | None = None,
) -> BoundMatchstickOutput:
    """Build common bound-output fields around task-specific trace extras."""

    common = {
        "query_id": str(query_id),
        "scene_id": SCENE_ID,
        "scene_variant": str(dataset.scene_variant),
        "answer_label": str(dataset.answer_label),
    }
    return BoundMatchstickOutput(
        prompt_query_key=str(prompt_query_key),
        prompt_dynamic_slots=dict(prompt_dynamic_slots or {}),
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "scene_id": SCENE_ID,
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            "option_count": int(dataset.option_count),
            "answer_label": str(dataset.answer_label),
            **dict(scene_extra),
        },
        answer_gt=answer_gt,
        annotation_artifacts=annotation_artifacts,
        annotation_source=str(annotation_source),
        scene_relations={**common, **dict(scene_extra)},
        execution_trace={
            **common,
            "query_id_probabilities": dict(query_probabilities),
            "question_format": str(query_id),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
            "option_count": int(dataset.option_count),
            "answer_value": str(dataset.answer_label),
            **dict(execution_extra),
        },
        witness_symbolic=dict(witness_symbolic),
    )


def bind_number_transform_output(
    *,
    dataset: NumberDataset,
    context: MatchstickRenderContext,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
    stick_delta: int,
) -> BoundMatchstickOutput:
    """Bind source/candidate bboxes and per-option reachability for number edits."""

    roles = {
        "source_number": "source_panel",
        "selected_option": f"option_{dataset.answer_label}",
    }
    annotation = bbox_map_artifacts(context.rendered_scene.item_bbox_map, roles)
    option_specs = [
        {
            "option_label": str(option.label),
            "number": number_text(int(option.value)),
            "is_reachable": bool(
                number_transition_allowed(
                    int(dataset.source_number),
                    int(option.value),
                    stick_delta=int(stick_delta),
                )
            ),
            "is_correct": bool(option.is_correct),
        }
        for option in dataset.option_specs
    ]
    return build_bound_output(
        dataset=dataset,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        scene_variant_probabilities=scene_variant_probabilities,
        prompt_query_key=str(query_id),
        answer_gt=TypedValue(type="option_letter", value=str(dataset.answer_label)),
        annotation_artifacts=annotation,
        annotation_source="keyed_item_bboxes_px",
        scene_extra={
            "source_number": number_text(int(dataset.source_number)),
            "answer_number": number_text(int(dataset.answer_number)),
        },
        execution_extra={
            "source_number": number_text(int(dataset.source_number)),
            "answer_number": number_text(int(dataset.answer_number)),
            "changed_digit_index": int(dataset.changed_digit_index),
            "removed_segment_keys": list(dataset.removed_segment_keys),
            "added_segment_keys": list(dataset.added_segment_keys),
            "supporting_item_ids": ["source_panel", f"option_{dataset.answer_label}"],
            "annotation_role_item_ids": dict(roles),
            "option_specs": option_specs,
        },
        witness_symbolic={"type": "bbox_map", "value": dict(annotation.value)},
    )


def select_scene_variant(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[str, Dict[str, float]]:
    """Select one nonsemantic matchstick material variant."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.scene_variant")
    return resolve_variant(
        rng,
        params=params,
        gen_defaults=generation_defaults,
        supported_variants=SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )


def render_matchstick_context(
    *,
    dataset: Any,
    render_scene: Callable[..., RenderedScene],
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> MatchstickRenderContext:
    """Render a task-owned dataset with the common matchstick canvas pipeline."""

    render_params, background, background_meta, font_family, font_meta = prepare_render_inputs(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
        namespace=str(namespace),
    )
    from trace_tasks.tasks.shared.text_rendering import temporary_default_font_family

    with temporary_default_font_family(str(font_family)):
        rendered_scene = render_scene(
            background=background,
            dataset=dataset,
            render_params=render_params,
        )
    image, post_noise_meta = apply_matchstick_post_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
    )
    return MatchstickRenderContext(
        rendered_scene=rendered_scene,
        image=image,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        font_meta=dict(font_meta),
    )


def run_matchstick_public_task(
    *,
    task_id: str,
    supported_query_ids: Sequence[str],
    task_prompt_key: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    dataset_builder: DatasetBuilder,
    render_scene: Callable[..., RenderedScene],
    output_binder: OutputBinder,
) -> TaskOutput:
    """Run shared lifecycle steps around task-owned dataset/output callbacks."""

    from trace_tasks.tasks.shared.fixed_query import select_task_query_id

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(tuple(supported_query_ids)[0]),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    scene_variant, scene_variant_probabilities = select_scene_variant(
        instance_seed=int(instance_seed),
        params=task_params,
        generation_defaults=generation_defaults,
        namespace=str(task_id),
    )
    dataset = None
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        try:
            dataset = dataset_builder(
                query_id=str(query_id),
                scene_variant=str(scene_variant),
                params=task_params,
                generation_defaults=generation_defaults,
                namespace=str(task_id),
                instance_seed=int(instance_seed) + int(attempt_index),
            )
            break
        except RuntimeError as exc:
            last_error = exc
    if dataset is None:
        raise RuntimeError(f"failed to generate matchstick task {task_id}") from last_error

    context = render_matchstick_context(
        dataset=dataset,
        render_scene=render_scene,
        instance_seed=int(instance_seed),
        params=task_params,
        rendering_defaults=rendering_defaults,
        namespace=str(task_id),
    )
    bound = output_binder(
        dataset=dataset,
        context=context,
        query_id=str(query_id),
        query_probabilities=dict(query_probabilities),
        scene_variant_probabilities=dict(scene_variant_probabilities),
    )
    return finalize_matchstick_output(
        context=context,
        task_prompt_key=str(task_prompt_key),
        prompt_query_key=str(bound.prompt_query_key),
        query_id=str(query_id),
        query_params=bound.query_params,
        prompt_dynamic_slots=bound.prompt_dynamic_slots,
        answer_gt=bound.answer_gt,
        annotation_artifacts=bound.annotation_artifacts,
        annotation_source=str(bound.annotation_source),
        scene_relations=bound.scene_relations,
        execution_trace=bound.execution_trace,
        witness_symbolic=bound.witness_symbolic,
        instance_seed=int(instance_seed),
    )


def prepare_render_inputs(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[RenderParams, Image.Image, Dict[str, Any], str, Dict[str, Any]]:
    """Resolve dimensions, background, and font for one rendered instance."""

    render_params = resolve_render_params(params, rendering_defaults)
    font_family = sample_matchstick_font(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=rendering_defaults,
        namespace=str(namespace),
    )
    font_meta = font_trace_record(str(font_family))
    scene_style, _scene_style_meta = resolve_puzzle_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    background, background_meta = make_scene_background(
        render_params=render_params,
        style=scene_style,
    )
    return render_params, background, dict(background_meta), str(font_family), dict(font_meta)


def apply_matchstick_post_noise(
    image: Image.Image,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> tuple[Image.Image, Dict[str, Any]]:
    """Apply scene-configured post-image noise after semantic rendering."""

    return apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=load_puzzle_noise_defaults(scene_id=SCENE_ID, apply_prob=0.15),
    )


def finalize_matchstick_output(
    *,
    context: MatchstickRenderContext,
    task_prompt_key: str,
    prompt_query_key: str,
    query_id: str,
    query_params: Mapping[str, Any],
    answer_gt: TypedValue,
    annotation_artifacts: AnnotationArtifacts,
    annotation_source: str,
    scene_relations: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    instance_seed: int,
    prompt_dynamic_slots: Mapping[str, Any] | None = None,
) -> TaskOutput:
    """Assemble prompt, trace payload, and TaskOutput from task-owned bindings."""

    prompt_defaults, prompt_artifacts = build_matchstick_prompt_artifacts(
        task_prompt_key=str(task_prompt_key),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(prompt_dynamic_slots or {}),
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params=dict(query_params),
    )
    render_spec = build_render_spec(
        rendered_scene=context.rendered_scene,
        render_params=context.render_params,
        scene_variant=str(scene_relations.get("scene_variant", "")),
        background_meta=context.background_meta,
        post_noise_meta=context.post_noise_meta,
        font_meta=context.font_meta,
    )
    render_map = build_render_map(
        rendered_scene=context.rendered_scene,
        annotation_source=str(annotation_source),
    )
    trace_payload = build_matchstick_trace_payload(
        scene_ir={
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in context.rendered_scene.entities],
            "relations": dict(scene_relations),
        },
        query_spec=query_spec,
        render_spec=render_spec,
        render_map=render_map,
        execution_trace=dict(execution_trace),
        witness_symbolic=dict(witness_symbolic),
        projected_annotation=annotation_artifacts.projected_annotation,
        answer_gt=answer_gt.to_dict(),
        annotation_gt=annotation_artifacts.annotation_gt.to_dict(),
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=context.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = [
    "apply_matchstick_post_noise",
    "bind_number_transform_output",
    "BoundMatchstickOutput",
    "build_bound_output",
    "finalize_matchstick_output",
    "prepare_render_inputs",
    "render_matchstick_context",
    "run_matchstick_public_task",
    "select_scene_variant",
]
