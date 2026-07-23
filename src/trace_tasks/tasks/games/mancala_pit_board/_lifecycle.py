"""Private neutral lifecycle helpers for Mancala pit-board public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Mapping

from PIL import Image

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import MancalaAnnotationBundle
from .shared.output import build_mancala_common_trace_params, build_mancala_execution_trace, build_mancala_trace_payload
from .shared.prompts import (
    MancalaPromptContext,
    MancalaPromptSlots,
    build_mancala_prompt_artifacts,
    format_mancala_json_examples,
)
from .shared.rendering import render_mancala_scene
from .shared.sampling import resolve_mancala_scene_axes
from .shared.state import SCENE_ID, MancalaSample, MancalaSceneAxes, RenderedMancalaScene


AnnotationBuilder = Callable[[RenderedMancalaScene], MancalaAnnotationBundle]
AttemptBuilder = Callable[[Any, MancalaSceneAxes], "MancalaAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], MancalaSceneAxes, Mapping[str, Any]],
    "MancalaObjectivePlan",
]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class MancalaAttemptResult:
    """Task-owned Mancala sample plus answer, prompt, trace, and annotation hooks."""

    answer_gt: TypedValue
    sample: MancalaSample
    prompt_slots: MancalaPromptSlots
    build_annotation: AnnotationBuilder
    execution_trace: Mapping[str, Any]
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    relations_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MancalaObjectivePlan:
    """Prepared task-owned Mancala objective for one generated instance."""

    attempt_namespace: str
    construct_attempt: AttemptBuilder


@dataclass(frozen=True)
class MancalaLifecycleResult:
    """Rendered prompt/image/annotation payload returned to public task files."""

    prompt: str
    prompt_variants: Mapping[str, str]
    annotation_bundle: MancalaAnnotationBundle
    image: Image.Image
    trace_payload: Mapping[str, Any]


class MancalaSingleQueryTaskBase:
    """Private base for Mancala public tasks that expose the single-query sentinel."""

    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = ("single",)
    _namespace: str
    _prepare_objective: ObjectivePreparer


def build_mancala_attempt_result(
    *,
    answer_gt: TypedValue,
    sample: MancalaSample,
    prompt_slots: MancalaPromptSlots,
    build_annotation: AnnotationBuilder,
    selected_query_id: str,
    annotation_entity_ids: Mapping[str, Any],
    extra_execution_fields: Mapping[str, Any],
    extra_query_params: Mapping[str, Any],
    relations_extra: Mapping[str, Any],
) -> MancalaAttemptResult:
    """Assemble neutral attempt plumbing around task-owned bindings."""

    execution_trace = build_mancala_execution_trace(
        branch_field_name="query_id",
        selected_branch=str(selected_query_id),
        sample=sample,
        answer=answer_gt.value,
        extra_fields=extra_execution_fields,
    )
    execution_trace["annotation_entity_ids"] = dict(annotation_entity_ids)
    return MancalaAttemptResult(
        answer_gt=answer_gt,
        sample=sample,
        prompt_slots=prompt_slots,
        build_annotation=build_annotation,
        execution_trace=execution_trace,
        extra_query_params=dict(extra_query_params),
        relations_extra=dict(relations_extra),
    )


@lru_cache(maxsize=None)
def mancala_task_defaults(task_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Load and cache generation/rendering/prompt defaults for one Mancala task."""

    return load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=str(task_id),
    )


def select_mancala_single_query(
    *,
    task_id: str,
    instance_seed: int,
    params: Mapping[str, Any] | None,
    namespace: str,
) -> tuple[str, Mapping[str, float], Mapping[str, Any]]:
    """Select and validate the public single-query sentinel for a Mancala task."""

    selected, probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        supported_query_ids=("single",),
        default_query_id="single",
        task_id=str(task_id),
        namespace=f"{namespace}.query",
    )
    return str(selected), dict(probabilities), dict(task_params)


def render_mancala_lifecycle(
    *,
    domain: str,
    selected_query_id: str,
    branch_probabilities: Mapping[str, float],
    task_params: Mapping[str, Any],
    axes: MancalaSceneAxes,
    attempt: MancalaAttemptResult,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> MancalaLifecycleResult:
    """Run neutral render, prompt, noise, and trace assembly around task bindings."""

    rendered_scene = render_mancala_scene(
        sample=attempt.sample,
        axes=axes,
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        namespace=str(namespace),
    )
    annotation_bundle = attempt.build_annotation(rendered_scene)
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    json_example, json_example_answer_only = format_mancala_json_examples(
        annotation=attempt.prompt_slots.example_annotation,
        answer=attempt.prompt_slots.example_answer,
    )
    prompt_defaults_used, prompt_artifacts = build_mancala_prompt_artifacts(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        context=MancalaPromptContext(
            prompt_query_key=str(attempt.prompt_slots.prompt_query_key),
            answer_hint_key=str(attempt.prompt_slots.answer_hint_key),
            annotation_hint_key=str(attempt.prompt_slots.annotation_hint_key),
            scene_variant=str(axes.scene_variant),
            json_example=str(json_example),
            json_example_answer_only=str(json_example_answer_only),
        ),
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=build_mancala_common_trace_params(
            axes=axes,
            branch_probabilities=branch_probabilities,
            extra_params={
                "prompt_query_key": str(attempt.prompt_slots.prompt_query_key),
                **dict(attempt.extra_query_params),
            },
        ),
    )
    trace_payload = build_mancala_trace_payload(
        annotation_bundle=annotation_bundle,
        axes=axes,
        rendered_scene=rendered_scene,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=query_spec,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        selected_branch=str(selected_query_id),
        branch_field_name="query_id",
        execution_trace=attempt.execution_trace,
        relations_extra=attempt.relations_extra,
    )
    return MancalaLifecycleResult(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        annotation_bundle=annotation_bundle,
        image=image,
        trace_payload=dict(trace_payload),
    )


def run_mancala_lifecycle(
    *,
    task_id: str,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    namespace: str,
) -> TaskOutput:
    """Run shared Mancala plumbing around task-owned objective hooks."""

    selected_query_id, branch_probabilities, task_params = select_mancala_single_query(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        params=dict(params),
        namespace=str(namespace),
    )
    axes = resolve_mancala_scene_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=str(namespace),
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query_id),
        dict(branch_probabilities),
        axes,
        gen_defaults,
    )
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue
        lifecycle = render_mancala_lifecycle(
            domain=str(domain),
            selected_query_id=str(selected_query_id),
            branch_probabilities=branch_probabilities,
            task_params=task_params,
            axes=axes,
            attempt=attempt,
            prompt_defaults=prompt_defaults,
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        return TaskOutput(
            prompt=str(lifecycle.prompt),
            prompt_variants=dict(lifecycle.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=lifecycle.annotation_bundle.annotation_gt,
            image=lifecycle.image,
            image_id="img0",
            trace_payload=dict(lifecycle.trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
        )
    raise RuntimeError(f"{task_id} failed to generate after {max_attempts} attempts")


def run_mancala_registered_task(
    task_obj: Any,
    instance_seed: int,
    *,
    params: Mapping[str, Any] | None = None,
    max_attempts: int = 100,
) -> TaskOutput:
    """Run the private Mancala lifecycle for a registered public task instance."""

    gen_defaults, render_defaults, prompt_defaults = mancala_task_defaults(str(task_obj.task_id))
    return run_mancala_lifecycle(
        task_id=str(task_obj.task_id),
        domain=str(task_obj.domain),
        gen_defaults=gen_defaults,
        render_defaults=render_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=dict(params or {}),
        max_attempts=int(max_attempts),
        prepare_objective=task_obj._prepare_objective,
        namespace=str(task_obj._namespace),
    )


__all__ = [
    "MancalaAttemptResult",
    "MancalaLifecycleResult",
    "MancalaObjectivePlan",
    "MancalaSingleQueryTaskBase",
    "build_mancala_attempt_result",
    "render_mancala_lifecycle",
    "run_mancala_lifecycle",
    "run_mancala_registered_task",
    "select_mancala_single_query",
]
