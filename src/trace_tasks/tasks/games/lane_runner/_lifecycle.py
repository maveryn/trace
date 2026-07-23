"""Private neutral lifecycle plumbing for lane-runner public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import build_lane_runner_common_trace_params, build_lane_runner_trace_payload
from .shared.prompts import LaneRunnerPromptContext, build_lane_runner_prompt_artifacts
from .shared.rendering import RenderedLaneRunnerScene, render_lane_runner_scene
from .shared.sampling import resolve_lane_runner_render_params, resolve_lane_runner_scene_axes
from .shared.state import SCENE_ID, LaneRunnerSceneAxes


AnnotationBuilder = Callable[[RenderedLaneRunnerScene], AnnotationArtifacts]
AttemptBuilder = Callable[[Any, LaneRunnerSceneAxes], "LaneRunnerAttemptResult"]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float], LaneRunnerSceneAxes],
    "LaneRunnerObjectivePlan",
]
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@dataclass(frozen=True)
class LaneRunnerAttemptResult:
    """Task-owned generated sample plus render, answer, and annotation hooks."""

    answer_gt: TypedValue
    render_inputs: Mapping[str, Any]
    build_annotation: AnnotationBuilder
    annotation_entity_ids: Sequence[str]
    scene_kind: str
    execution_trace: Mapping[str, Any]
    relations_extra: Mapping[str, Any] = field(default_factory=dict)
    witness_type: str = "object_set"


@dataclass(frozen=True)
class LaneRunnerObjectivePlan:
    """Prepared task-owned lane-runner objective for one generated instance."""

    prompt_query_key: str
    object_description_suffix: str
    rule_slot_name: str
    attempt_namespace: str
    construct_attempt: AttemptBuilder
    extra_query_params: Mapping[str, Any] = field(default_factory=dict)
    option_count: int | None = None


def _allowed_panel_treatments(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> tuple[str, ...] | None:
    raw = params.get("panel_scene_treatments", group_default(render_defaults, "panel_scene_treatments", None))
    if isinstance(raw, str):
        return (str(raw),)
    if raw is None:
        return None
    return tuple(str(item) for item in raw)


def run_lane_runner_lifecycle(
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
    """Run shared lane-runner plumbing around task-owned objective hooks."""

    selected_query_id, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{namespace}.query",
    )
    axes = resolve_lane_runner_scene_axes(
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
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=task_params,
    )
    render_params = resolve_lane_runner_render_params(
        task_params,
        axes=axes,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        font_family=str(font_family),
        namespace=str(namespace),
        option_count=objective.option_count,
    )

    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            attempt = objective.construct_attempt(rng, axes)
        except ValueError:
            continue

        panel_style, panel_style_meta = resolve_game_panel_scene_style(
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.panel_scene_style",
            treatments=_allowed_panel_treatments(task_params, render_defaults),
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
        rendered_scene = render_lane_runner_scene(
            background=background,
            style_variant=str(axes.style_variant),
            params=render_params,
            panel_style=panel_style,
            **dict(attempt.render_inputs),
        )
        annotation_artifacts = attempt.build_annotation(rendered_scene)
        image, post_noise_meta = apply_post_image_noise(
            rendered_scene.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        prompt_defaults_used, prompt_artifacts = build_lane_runner_prompt_artifacts(
            domain=str(domain),
            prompt_defaults=prompt_defaults,
            context=LaneRunnerPromptContext(
                scene_variant=str(axes.scene_variant),
                prompt_query_key=str(objective.prompt_query_key),
                object_description_suffix=str(objective.object_description_suffix),
                rule_slot_name=str(objective.rule_slot_name),
                answer_type=str(attempt.answer_gt.type),
            ),
            instance_seed=int(instance_seed),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(selected_query_id),
            params=build_lane_runner_common_trace_params(
                axes=axes,
                branch_probabilities=branch_probabilities,
                extra_params={
                    **dict(objective.extra_query_params),
                    "prompt_query_key": str(objective.prompt_query_key),
                },
            ),
        )
        font_record = get_font_family_record(str(font_family)).to_trace()
        trace_payload = build_lane_runner_trace_payload(
            annotation_artifacts=annotation_artifacts,
            annotation_entity_ids=tuple(str(entity_id) for entity_id in attempt.annotation_entity_ids),
            axes=axes,
            rendered_scene=rendered_scene,
            prompt_defaults=prompt_defaults_used,
            prompt_query_spec=query_spec,
            background_meta=background_meta,
            post_noise_meta=post_noise_meta,
            image_size=(int(image.size[0]), int(image.size[1])),
            scene_kind=str(attempt.scene_kind),
            selected_branch=str(selected_query_id),
            branch_field_name="query_id",
            execution_trace=attempt.execution_trace,
            relations_extra=attempt.relations_extra,
            render_spec_extra={
                "cell_size_px": int(render_params.cell_size_px),
                "panel_scene_style": dict(panel_style_meta),
                "font_assets": {
                    "readout_font_family": {
                        **dict(font_record),
                        "font_role": "readout",
                    },
                },
            },
            witness_type=str(attempt.witness_type),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
            answer_gt=attempt.answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=dict(trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query_id),
        )

    raise RuntimeError(f"{task_id} failed to generate a valid lane-runner scene after {max_attempts} attempts")


__all__ = [
    "LaneRunnerAttemptResult",
    "LaneRunnerObjectivePlan",
    "run_lane_runner_lifecycle",
]
