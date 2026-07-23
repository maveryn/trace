"""Private neutral lifecycle for radial hunt board public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import radial_hunt_point_set_annotation
from .shared.output import (
    annotation_point_ids,
    build_radial_hunt_common_trace_params,
    build_radial_hunt_trace_payload,
)
from .shared.prompts import build_radial_hunt_prompt_artifacts
from .shared.rendering import render_radial_hunt_board_scene
from .shared.rules import validate_radial_hunt_board_sample
from .shared.sampling import resolve_radial_hunt_board_visual_axes
from .shared.state import SCENE_ID, RadialHuntBoardSample, RadialHuntBoardVisualAxes


AttemptBuilder = Callable[[Any, RadialHuntBoardVisualAxes], RadialHuntBoardSample]
ObjectivePreparer = Callable[[int, Mapping[str, Any], Mapping[str, Any]], "ObjectiveRadialHuntBoardPlan"]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class ObjectiveRadialHuntBoardPlan:
    """Task-owned objective hooks for one radial hunt board instance."""

    attempt_namespace: str
    prompt_query_key: str
    rule_slot_name: str
    annotation_trace_key: str
    common_params: Mapping[str, Any] = field(default_factory=dict)
    construct_attempt: AttemptBuilder | None = None


def run_radial_hunt_board_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
    namespace: str,
) -> TaskOutput:
    """Run query selection, retry, render, prompt, and TaskOutput plumbing."""

    selected_query_id, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{namespace}.query",
    )
    visual_axes = resolve_radial_hunt_board_visual_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
    )
    objective = prepare_objective(int(instance_seed), task_params, gen_defaults)
    if objective.construct_attempt is None:
        raise ValueError("radial hunt board objective plan must provide construct_attempt")

    sample = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, visual_axes)
            validate_radial_hunt_board_sample(sample)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate after {max_attempts} attempts")

    rendered = render_radial_hunt_board_scene(
        sample=sample,
        axes=visual_axes,
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        namespace=str(namespace),
    )
    annotation_ids = annotation_point_ids(sample)
    annotation_artifacts = radial_hunt_point_set_annotation(rendered, annotation_ids)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_radial_hunt_prompt_artifacts(
        domain=str(domain),
        scene_variant=str(visual_axes.scene_variant),
        prompt_query_key=str(objective.prompt_query_key),
        rule_slot_name=str(objective.rule_slot_name),
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
    )
    common_params = build_radial_hunt_common_trace_params(
        axes=visual_axes,
        branch_probabilities=branch_probabilities,
        extra_params={
            "prompt_query_key": str(objective.prompt_query_key),
            **dict(objective.common_params),
        },
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=common_params,
    )
    trace_payload = build_radial_hunt_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=annotation_ids,
        sample=sample,
        axes=visual_axes,
        rendered_scene=rendered,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=query_spec,
        background_meta=rendered.background_meta,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        annotation_trace_key=str(objective.annotation_trace_key),
        relations_extra={"prompt_query_key": str(objective.prompt_query_key)},
        execution_extra=dict(objective.common_params),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        annotation_gt=TypedValue(
            type=str(annotation_artifacts.annotation_type),
            value=annotation_artifacts.value,
        ),
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


__all__ = [
    "ObjectiveRadialHuntBoardPlan",
    "run_radial_hunt_board_lifecycle",
]
