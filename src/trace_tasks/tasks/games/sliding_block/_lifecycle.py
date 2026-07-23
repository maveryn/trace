"""Private neutral lifecycle plumbing for sliding-block public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.output import build_common_trace_sections, common_query_params
from .shared.prompts import build_sliding_block_prompt_artifacts
from .shared.rendering import apply_panel_style, render_map_payload, render_sliding_block_scene, resolve_render_params
from .shared.sampling import resolve_exit_side, resolve_scene_variant
from .shared.state import SCENE_ID, RenderedSlidingBlockScene


AnnotationBuilder = Callable[[RenderedSlidingBlockScene], AnnotationArtifacts]
ObjectiveBuilder = Callable[[int, Mapping[str, Any], str, str], "SlidingBlockObjective"]


@dataclass(frozen=True)
class SlidingBlockObjective:
    """Task-owned objective result consumed by neutral scene plumbing."""

    dataset: Mapping[str, Any]
    answer_gt: TypedValue
    answer_block_ids: list[str]
    render_mode: str
    annotation_source: str
    prompt_query_key: str
    prompt_default_prefix: str
    build_annotation: AnnotationBuilder
    prompt_dynamic_values: Mapping[str, Any] = field(default_factory=dict)
    trace_extra_params: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


def run_sliding_block_lifecycle(
    *,
    namespace: str,
    domain: str,
    supported_queries: tuple[str, ...],
    default_query: str,
    task_params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    build_objective: ObjectiveBuilder,
) -> TaskOutput:
    """Run shared axes, rendering, prompt, trace, and output assembly."""

    selected_query, query_probabilities, resolved_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=task_params,
        supported_query_ids=supported_queries,
        default_query_id=str(default_query),
        namespace=f"{namespace}.public_query",
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(resolved_params, instance_seed=int(instance_seed))
    exit_side, exit_side_probabilities = resolve_exit_side(resolved_params, instance_seed=int(instance_seed))

    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + (1009 * int(attempt_index))
        try:
            objective = build_objective(int(attempt_seed), resolved_params, str(exit_side), str(selected_query))
        except ValueError as exc:
            last_error = exc
            continue
        break
    else:
        raise RuntimeError(f"{namespace} failed to construct a valid sliding-block instance: {last_error}") from last_error

    render_params = resolve_render_params(resolved_params, instance_seed=int(instance_seed))
    render_params, background, background_meta, style_meta = apply_panel_style(
        render_params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.background",
    )
    rendered_scene = render_sliding_block_scene(
        background,
        dataset=objective.dataset,
        scene_variant=str(scene_variant),
        render_params=render_params,
        render_mode=str(objective.render_mode),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=resolved_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = objective.build_annotation(rendered_scene)
    prompt_defaults, prompt_artifacts = build_sliding_block_prompt_artifacts(
        prompt_query_key=str(objective.prompt_query_key),
        prompt_default_prefix=str(objective.prompt_default_prefix),
        answer_type=str(objective.answer_gt.type),
        instance_seed=int(instance_seed),
        dynamic_values=dict(objective.prompt_dynamic_values),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query),
        params=common_query_params(
            scene_variant=str(scene_variant),
            scene_variant_probabilities=scene_variant_probabilities,
            exit_side=str(exit_side),
            exit_side_probabilities=exit_side_probabilities,
            dataset=objective.dataset,
            extra_params={
                **dict(objective.trace_extra_params),
                "query_id_probabilities": dict(query_probabilities),
            },
        ),
    )
    render_map = render_map_payload(
        rendered_scene=rendered_scene,
        render_params=render_params,
        annotation_source=str(objective.annotation_source),
    )
    trace_payload = build_common_trace_sections(
        rendered_scene=rendered_scene,
        render_params=render_params,
        render_map=render_map,
        scene_variant=str(scene_variant),
        exit_side=str(exit_side),
        dataset=objective.dataset,
        answer_value=objective.answer_gt.value,
        answer_block_ids=[str(block_id) for block_id in objective.answer_block_ids],
        background_meta=background_meta,
        scene_style_meta=style_meta,
        post_noise_meta=post_noise_meta,
        annotation_payload=annotation_artifacts.projected_annotation,
    )
    trace_payload["query_spec"] = query_spec
    trace_payload["answer_gt"] = objective.answer_gt.to_dict()
    trace_payload["annotation_gt"] = annotation_artifacts.annotation_gt.to_dict()
    trace_payload["execution_trace"]["question_format"] = str(objective.prompt_query_key)
    trace_payload["execution_trace"].update(dict(objective.execution_extra))
    trace_payload["render_spec"]["prompt_defaults_bundle_id"] = str(prompt_defaults["bundle_id"])

    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=objective.answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = ["SlidingBlockObjective", "run_sliding_block_lifecycle"]
