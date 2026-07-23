"""Private neutral lifecycle plumbing for solitaire public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import GEN_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS
from .shared.output import build_solitaire_trace_payload, solitaire_trace_params
from .shared.prompts import build_solitaire_prompt
from .shared.rendering import render_solitaire_scene
from .shared.state import SUPPORTED_PANEL_STYLE_VARIANTS, SUPPORTED_SCENE_VARIANTS
from ..shared.sampling import resolve_games_named_axis
from .shared.state import DOMAIN, SCENE_ID, RenderedSolitaireScene, SolitaireSample


AnnotationBuilder = Callable[[SolitaireSample, RenderedSolitaireScene], AnnotationArtifacts]
ObjectiveBuilder = Callable[[Any, Mapping[str, Any], str, int], "SolitaireObjective"]


def _sample_scene_variant(*, namespace: str, instance_seed: int, params: Mapping[str, Any]):
    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported_variants=SUPPORTED_SCENE_VARIANTS,
    )


def _sample_panel_style_variant(*, namespace: str, instance_seed: int, params: Mapping[str, Any]):
    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=GEN_DEFAULTS,
        namespace="style_variant",
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported_variants=SUPPORTED_PANEL_STYLE_VARIANTS,
    )


@dataclass(frozen=True)
class SolitaireObjective:
    """Task-owned solitaire sample, answer, annotation, and prompt binding."""

    sample: SolitaireSample
    answer_gt: TypedValue
    prompt_query_key: str
    build_annotation: AnnotationBuilder
    json_example: str
    json_example_answer_only: str
    prompt_slots: Mapping[str, Any] | None = None


class SolitaireLifecycleTask:
    """Default public metadata shared by solitaire task classes."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)


def run_solitaire_lifecycle(
    *,
    namespace: str,
    prompt_query_key: str,
    supported_queries: tuple[str, ...],
    default_query: str,
    task_params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    build_objective: ObjectiveBuilder,
) -> TaskOutput:
    """Run common query, render, prompt, annotation, and output plumbing."""

    selected_public_query, public_query_probabilities, resolved_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=task_params,
        supported_query_ids=supported_queries,
        default_query_id=str(default_query),
        task_id=str(namespace),
        namespace=f"{namespace}.public_query",
    )
    scene_variant, scene_variant_probabilities = _sample_scene_variant(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=resolved_params,
    )
    style_variant, style_variant_probabilities = _sample_panel_style_variant(
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        params=resolved_params,
    )

    last_error: Exception | None = None
    objective: SolitaireObjective | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{str(namespace)}.attempt.{int(attempt_index)}")
        try:
            objective = build_objective(rng, resolved_params, str(scene_variant), int(instance_seed))
        except ValueError as exc:
            last_error = exc
            continue
        break
    if objective is None:
        raise RuntimeError(f"{namespace} failed to construct a valid solitaire tableau: {last_error}") from last_error

    rendered = render_solitaire_scene(
        sample=objective.sample,
        namespace=str(namespace),
        style_variant=str(style_variant),
        instance_seed=int(instance_seed),
        params=resolved_params,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=resolved_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = objective.build_annotation(objective.sample, rendered)
    prompt, prompt_variants, prompt_meta = build_solitaire_prompt(
        str(prompt_query_key),
        json_example=str(objective.json_example),
        json_example_answer_only=str(objective.json_example_answer_only),
        instance_seed=int(instance_seed),
        prompt_slots=objective.prompt_slots,
    )
    trace_params = solitaire_trace_params(
        sample=objective.sample,
        prompt_query_key=str(prompt_query_key),
        scene_variant_probabilities=dict(scene_variant_probabilities),
        style_variant=str(style_variant),
        style_variant_probabilities=dict(style_variant_probabilities),
        public_query_probabilities=dict(public_query_probabilities),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_meta["prompt_artifacts"],
        query_id=str(selected_public_query),
        params=trace_params,
    )
    trace_payload = build_solitaire_trace_payload(
        sample=objective.sample,
        rendered=rendered,
        prompt_query_key=str(prompt_query_key),
        style_variant=str(style_variant),
        annotation_artifacts=annotation_artifacts,
        background_meta=rendered.background_meta,
        post_noise_meta=post_noise_meta,
    )
    trace_payload["query_spec"] = dict(query_spec)
    trace_payload["answer_gt"] = objective.answer_gt.to_dict()
    trace_payload["annotation_gt"] = annotation_artifacts.annotation_gt.to_dict()
    trace_payload["execution_trace"]["query_id"] = str(selected_public_query)
    trace_payload["execution_trace"]["prompt_query_key"] = str(prompt_query_key)
    trace_payload["render_spec"]["prompt_defaults_bundle_id"] = str(prompt_meta["bundle_id"])

    return TaskOutput(
        prompt=str(prompt),
        prompt_variants=dict(prompt_variants),
        answer_gt=objective.answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_public_query),
    )
