"""Private neutral lifecycle for Sixteen Soldiers public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import (
    build_sixteen_soldiers_common_trace_params,
    build_sixteen_soldiers_trace_payload,
)
from .shared.prompts import build_sixteen_soldiers_prompt_artifacts
from .shared.rendering import (
    render_sixteen_soldiers_scene,
    resolve_sixteen_soldiers_render_params,
)
from .shared.sampling import resolve_sixteen_soldiers_visual_axes
from .shared.state import SCENE_ID, SixteenSoldiersSample, SixteenSoldiersTargetAxis, SixteenSoldiersVisualAxes


AttemptBuilder = Callable[[Any, SixteenSoldiersVisualAxes], SixteenSoldiersSample]
AnnotationProjector = Callable[[Any, Sequence[str]], AnnotationArtifacts]
AnnotationEntityIdResolver = Callable[[SixteenSoldiersSample], Sequence[str]]
SampleValidator = Callable[[SixteenSoldiersSample], None]
ObjectivePreparer = Callable[[int, Mapping[str, Any], Mapping[str, Any]], "ObjectiveSixteenSoldiersPlan"]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class ObjectiveSixteenSoldiersPlan:
    """Task-owned objective hooks for one Sixteen Soldiers instance."""

    attempt_namespace: str
    prompt_query_key: str
    rule_slot_name: str
    annotation_kind: str
    target_axis: SixteenSoldiersTargetAxis
    common_params: Mapping[str, Any] = field(default_factory=dict)
    construct_attempt: AttemptBuilder | None = None
    annotation_projector: AnnotationProjector | None = None
    annotation_entity_ids: AnnotationEntityIdResolver | None = None
    validate_sample: SampleValidator | None = None


def _resolve_panel_treatments(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Return an optional allowlist of shared panel treatments."""

    raw = params.get("panel_scene_treatments", group_default(render_defaults, "panel_scene_treatments", None))
    if isinstance(raw, str):
        return (str(raw),)
    if raw is None:
        return None
    return tuple(str(item) for item in raw)


def run_sixteen_soldiers_lifecycle(
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
    visual_axes = resolve_sixteen_soldiers_visual_axes(
        int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace_root=str(namespace),
    )
    objective = prepare_objective(int(instance_seed), task_params, gen_defaults)
    if objective.construct_attempt is None:
        raise ValueError("Sixteen Soldiers objective plan must provide construct_attempt")
    if objective.annotation_projector is None:
        raise ValueError("Sixteen Soldiers objective plan must provide annotation_projector")
    if objective.annotation_entity_ids is None:
        raise ValueError("Sixteen Soldiers objective plan must provide annotation_entity_ids")
    if objective.validate_sample is None:
        raise ValueError("Sixteen Soldiers objective plan must provide validate_sample")

    sample = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, visual_axes)
            objective.validate_sample(sample)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate after {max_attempts} attempts")

    render_params = resolve_sixteen_soldiers_render_params(
        task_params,
        render_defaults,
        instance_seed=int(instance_seed),
    )
    panel_style, panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.sixteen_soldiers.panel_scene_style",
        treatments=_resolve_panel_treatments(task_params, render_defaults),
        treatment_weights=task_params.get(
            "panel_scene_treatment_weights",
            group_default(render_defaults, "panel_scene_treatment_weights", None),
        ),
        palette_weights=task_params.get(
            "panel_scene_palette_weights",
            group_default(render_defaults, "panel_scene_palette_weights", None),
        ),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered = render_sixteen_soldiers_scene(
        board=sample.board,
        background=background,
        style_variant=str(visual_axes.style_variant),
        params=render_params,
        marked_point_id=str(sample.marked_point_id),
        panel_style=panel_style,
    )
    rendered.render_map["panel_scene_style"] = dict(panel_style_meta)

    annotation_ids = tuple(str(point_id) for point_id in sample.annotation_point_ids)
    annotation_artifacts = objective.annotation_projector(rendered, annotation_ids)
    annotation_entity_ids = tuple(str(entity_id) for entity_id in objective.annotation_entity_ids(sample))
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults_used, prompt_artifacts = build_sixteen_soldiers_prompt_artifacts(
        domain=str(domain),
        scene_variant=str(visual_axes.scene_variant),
        prompt_query_key=str(objective.prompt_query_key),
        rule_slot_name=str(objective.rule_slot_name),
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
    )
    common_params = build_sixteen_soldiers_common_trace_params(
        axes=visual_axes,
        target_axis=objective.target_axis,
        branch_probabilities=branch_probabilities,
        prompt_query_key=str(objective.prompt_query_key),
        extra_params=dict(objective.common_params),
    )
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params=common_params,
    )
    trace_payload = build_sixteen_soldiers_trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=annotation_entity_ids,
        annotation_kind=str(objective.annotation_kind),
        sample=sample,
        axes=visual_axes,
        rendered_scene=rendered,
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=prompt_query_spec,
        background_meta=background_meta,
        post_noise_meta=post_noise_meta,
        image_size=(int(image.size[0]), int(image.size[1])),
        prompt_query_key=str(objective.prompt_query_key),
        execution_extra=dict(objective.common_params),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


__all__ = [
    "ObjectiveSixteenSoldiersPlan",
    "run_sixteen_soldiers_lifecycle",
]
