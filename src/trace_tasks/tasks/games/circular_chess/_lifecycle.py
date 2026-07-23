"""Scene-private lifecycle plumbing for Circular Chess public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import annotation_from_evaluation
from .shared.defaults import SCENE_ID
from .shared.output import common_trace_sections
from .shared.prompts import build_circular_chess_prompt_artifacts
from .shared.rendering import (
    render_circular_chess_scene,
    resolve_circular_chess_render_params,
    resolve_scene_background,
    text_style_metadata,
)
from .shared.sampling import resolve_circular_chess_scene_axes
from .shared.state import CircularChessSample, Coord


AttemptBuilder = Callable[[Any], CircularChessSample]
SampleParamsBuilder = Callable[[CircularChessSample], Mapping[str, Any]]
SceneObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], Any],
    "CircularChessObjectivePlan",
]


@dataclass(frozen=True)
class CircularChessOutputPlan:
    """Task-owned values needed by neutral render/prompt/output plumbing."""

    selected_query_id: str
    query_probabilities: Mapping[str, float]
    task_params: Mapping[str, Any]
    scene_variant: str
    style_variant: str
    prompt_target_color: str
    prompt_marked_piece_present: bool
    prompt_example_answer: int
    render_marked_coord: Coord | None
    render_target_coord: Coord | None
    query_spec_params: Mapping[str, Any]
    execution_updates: Mapping[str, Any]
    relation_updates: Mapping[str, Any]


@dataclass(frozen=True)
class CircularChessObjectivePlan:
    """Task-owned objective hooks prepared after query and scene axes resolve."""

    attempt_namespace: str
    construct_attempt: AttemptBuilder
    prompt_target_color: str
    prompt_marked_piece_present: bool
    prompt_example_answer: int
    render_marked_coord: Callable[[CircularChessSample], Coord | None]
    render_target_coord: Callable[[CircularChessSample], Coord | None]
    query_spec_params: SampleParamsBuilder
    execution_updates: SampleParamsBuilder
    relation_updates: SampleParamsBuilder


def sample_with_retries(
    *,
    task_id: str,
    instance_seed: int,
    max_attempts: int,
    attempt_namespace: str,
    construct_attempt: AttemptBuilder,
) -> CircularChessSample:
    """Run task-owned Circular Chess construction attempts with stable retry seeds."""

    last_error: ValueError | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            return construct_attempt(rng)
        except ValueError as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{task_id} failed to generate a valid scene after {max_attempts} attempts") from last_error


def build_circular_chess_task_output(
    *,
    task_id: str,
    domain: str,
    instance_seed: int,
    sample: CircularChessSample,
    output_plan: CircularChessOutputPlan,
    post_image_noise_defaults: Mapping[str, Any],
) -> TaskOutput:
    """Assemble common Circular Chess render, prompt, annotation, and trace payload."""

    render_params = resolve_circular_chess_render_params(
        output_plan.task_params,
        instance_seed=int(instance_seed),
    )
    background, background_meta, panel_style, panel_style_meta = resolve_scene_background(
        params=output_plan.task_params,
        render_params=render_params,
        instance_seed=int(instance_seed),
    )
    rendered = render_circular_chess_scene(
        board=sample.board,
        background=background,
        style_variant=str(output_plan.style_variant),
        params=render_params,
        marked_coord=output_plan.render_marked_coord,
        target_coord=output_plan.render_target_coord,
        panel_style=panel_style,
    )
    annotation_type, annotation_value, projected_annotation = annotation_from_evaluation(
        evaluation=sample.evaluation,
        render_map=rendered.render_map,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=output_plan.task_params,
        default_config=post_image_noise_defaults,
    )
    _prompt_defaults, prompt_artifacts = build_circular_chess_prompt_artifacts(
        domain=str(domain),
        prompt_query_key=str(output_plan.selected_query_id),
        scene_variant=str(output_plan.scene_variant),
        target_color=str(output_plan.prompt_target_color),
        marked_piece_present=bool(output_plan.prompt_marked_piece_present),
        example_answer=int(output_plan.prompt_example_answer),
        instance_seed=int(instance_seed),
    )
    answer_gt = TypedValue(type="integer", value=int(sample.evaluation.answer))
    annotation_gt = TypedValue(type=str(annotation_type), value=[list(item) for item in annotation_value])
    trace_payload = common_trace_sections(
        sample=sample,
        image_size=(int(image.size[0]), int(image.size[1])),
        render_map=dict(rendered.render_map),
        scene_entities=tuple(rendered.scene_entities),
        panel_style_meta=dict(panel_style_meta),
        text_style_meta=text_style_metadata(str(render_params.font_family)),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )
    trace_payload["query_spec"] = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(output_plan.selected_query_id),
        params={
            "query_id": str(output_plan.selected_query_id),
            "query_id_probabilities": dict(output_plan.query_probabilities),
            **dict(output_plan.query_spec_params),
        },
    )
    trace_payload["execution_trace"].update(
        {
            "query_id": str(output_plan.selected_query_id),
            "answer": int(answer_gt.value),
            **dict(output_plan.execution_updates),
        }
    )
    trace_payload["scene_ir"]["relations"].update(
        {
            "query_id": str(output_plan.selected_query_id),
            **dict(output_plan.relation_updates),
        }
    )
    trace_payload["projected_annotation"] = dict(projected_annotation)
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
        query_id=str(output_plan.selected_query_id),
    )


def run_circular_chess_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: tuple[str, ...],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: SceneObjectivePreparer,
    post_image_noise_defaults: Mapping[str, Any],
) -> TaskOutput:
    """Run neutral Circular Chess query, scene, retry, render, and output plumbing."""

    selected_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    scene_axes = resolve_circular_chess_scene_axes(int(instance_seed), params=task_params)
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        str(selected_query_id),
        query_probabilities,
        scene_axes,
    )
    sample = sample_with_retries(
        task_id=str(task_id),
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        attempt_namespace=str(objective.attempt_namespace),
        construct_attempt=objective.construct_attempt,
    )
    return build_circular_chess_task_output(
        task_id=str(task_id),
        domain=str(domain),
        instance_seed=int(instance_seed),
        sample=sample,
        output_plan=CircularChessOutputPlan(
            selected_query_id=str(selected_query_id),
            query_probabilities=dict(query_probabilities),
            task_params=dict(task_params),
            scene_variant=str(scene_axes.scene_variant),
            style_variant=str(scene_axes.style_variant),
            prompt_target_color=str(objective.prompt_target_color),
            prompt_marked_piece_present=bool(objective.prompt_marked_piece_present),
            prompt_example_answer=int(objective.prompt_example_answer),
            render_marked_coord=objective.render_marked_coord(sample),
            render_target_coord=objective.render_target_coord(sample),
            query_spec_params=objective.query_spec_params(sample),
            execution_updates=objective.execution_updates(sample),
            relation_updates=objective.relation_updates(sample),
        ),
        post_image_noise_defaults=post_image_noise_defaults,
    )


__all__ = [
    "CircularChessObjectivePlan",
    "CircularChessOutputPlan",
    "build_circular_chess_task_output",
    "run_circular_chess_lifecycle",
    "sample_with_retries",
]
