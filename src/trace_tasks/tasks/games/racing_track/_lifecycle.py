"""Scene-private lifecycle orchestration for racing-track public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.defaults import SCENE_ID
from .shared.output import build_racing_track_common_trace_payload, common_racing_track_trace_params
from .shared.prompts import build_racing_track_prompt_artifacts
from .shared.rendering import (
    RacingTrackRenderParams,
    RenderedRacingTrackTaskContext,
    render_racing_track_task_context,
    resolve_racing_track_render_params,
)
from .shared.rules import validate_racing_track_state
from .shared.sampling import RacingTrackVisualAxes, resolve_racing_track_visual_axes
from .shared.state import RacingTrackSceneState


AnnotationBuilder = Callable[[RenderedRacingTrackTaskContext], AnnotationArtifacts]
AttemptBuilder = Callable[[Any], "AttemptRacingTrackResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], Mapping[str, float], str, RacingTrackVisualAxes, RacingTrackRenderParams],
    "ObjectiveRacingTrackPlan",
]


@dataclass(frozen=True)
class AttemptRacingTrackResult:
    """Task-owned result of one constructed racing-track attempt."""

    state: RacingTrackSceneState
    answer_gt: TypedValue
    build_annotation: AnnotationBuilder
    annotation_entity_ids: tuple[str, ...]
    witness_type: str
    marked_car_id: str = ""
    dynamic_prompt_slots: Mapping[str, Any] = field(default_factory=dict)
    query_params: Mapping[str, Any] = field(default_factory=dict)
    relations_extra: Mapping[str, Any] = field(default_factory=dict)
    execution_extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveRacingTrackPlan:
    """Prepared task-owned racing-track objective hooks for one instance."""

    attempt_namespace: str
    prompt_query_key: str
    object_description_key: str
    rule_text_key: str
    answer_hint_key: str
    annotation_hint_key: str
    json_example: str
    json_example_answer_only: str
    query_params: Mapping[str, Any]
    construct_attempt: AttemptBuilder


def run_racing_track_lifecycle(
    *,
    task_id: str,
    domain: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run common racing-track query, render, prompt, and output plumbing."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    visual_axes = resolve_racing_track_visual_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace_root=f"{SCENE_ID}.visual",
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"games.{SCENE_ID}.font",
        params=task_params,
    )
    render_params = resolve_racing_track_render_params(
        task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        font_family=str(font_family),
        namespace=f"games.{SCENE_ID}.layout_jitter",
    )
    objective = prepare_objective(
        int(instance_seed),
        task_params,
        query_probabilities,
        query_id,
        visual_axes,
        render_params,
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            attempt = objective.construct_attempt(rng)
            validate_racing_track_state(attempt.state)
        except ValueError:
            continue

        rendered_context = render_racing_track_task_context(
            state=attempt.state,
            params=task_params,
            render_defaults=render_defaults,
            render_params=render_params,
            instance_seed=int(instance_seed),
            namespace=f"games.{SCENE_ID}.render",
            marked_car_id=str(attempt.marked_car_id or "") or None,
        )
        annotation_artifacts = attempt.build_annotation(rendered_context)
        prompt_defaults, prompt_artifacts = build_racing_track_prompt_artifacts(
            domain=str(domain),
            prompt_query_key=str(objective.prompt_query_key),
            object_description_key=str(objective.object_description_key),
            rule_text_key=str(objective.rule_text_key),
            instance_seed=int(instance_seed),
            answer_hint_key=str(objective.answer_hint_key),
            annotation_hint_key=str(objective.annotation_hint_key),
            json_example=str(objective.json_example),
            json_example_answer_only=str(objective.json_example_answer_only),
            dynamic_slots=dict(attempt.dynamic_prompt_slots),
        )
        merged_query_params = {
            **dict(objective.query_params),
            **dict(attempt.query_params),
        }
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(query_id),
            params=common_racing_track_trace_params(
                visual_axes,
                attempt.state,
                prompt_query_key=str(objective.prompt_query_key),
                query_id_probabilities=query_probabilities,
                extra_params=merged_query_params,
            ),
        )
        trace_payload = build_racing_track_common_trace_payload(
            axes=visual_axes,
            state=attempt.state,
            rendered_context=rendered_context,
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            query_spec=query_spec,
            witness_type=str(attempt.witness_type),
            relations_extra=dict(attempt.relations_extra),
            execution_extra={
                "answer": attempt.answer_gt.value,
                **dict(attempt.execution_extra),
            },
        )
        trace_payload["scene_ir"]["relations"]["query_id"] = str(query_id)
        trace_payload["render_spec"]["query_id"] = str(query_id)
        trace_payload["execution_trace"]["query_id"] = str(query_id)
        font_record = get_font_family_record(str(render_params.font_family)).to_trace()
        trace_payload["render_spec"]["font_assets"] = {
            "readout_font_family": {
                **dict(font_record),
                "font_role": "readout",
            },
        }
        trace_payload["query_spec"]["template_id"] = str(prompt_defaults["bundle_id"])
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=rendered_context.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid racing-track scene after {max_attempts} attempts")


__all__ = [
    "AttemptRacingTrackResult",
    "ObjectiveRacingTrackPlan",
    "run_racing_track_lifecycle",
]
