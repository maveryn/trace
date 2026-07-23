"""Private neutral lifecycle plumbing for space-shooter public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import entity_bbox_set
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, RENDER_DEFAULTS
from .shared.output import build_space_shooter_trace_payload, space_shooter_trace_params
from .shared.prompts import build_space_shooter_prompt
from .shared.rendering import render_space_shooter_scene, resolve_space_shooter_render_params
from .shared.sampling import resolve_scene_axes
from .shared.state import DOMAIN, SCENE_ID, SceneAxes, SpaceShooterSample


AnnotationBuilder = Callable[[SpaceShooterSample, Any], AnnotationArtifacts]
ObjectiveBuilder = Callable[[Any, Mapping[str, Any], SceneAxes, int], "SpaceShooterObjective"]


@dataclass(frozen=True)
class SpaceShooterObjective:
    """Task-owned sample, answer, annotation, and prompt binding."""

    sample: SpaceShooterSample
    answer_gt: TypedValue
    prompt_query_key: str
    build_annotation: AnnotationBuilder = entity_bbox_set
    json_example: str = ""
    json_example_answer_only: str = ""
    show_enemy_labels: bool = True
    visible_enemy_label_ids: tuple[str, ...] | None = None


class SpaceShooterLifecycleTask:
    """Default public metadata shared by space-shooter task classes."""

    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (DEFAULT_QUERY_ID,)


def _resolve_panel_style(*, instance_seed: int, params: Mapping[str, Any]):
    allowed_raw = params.get("panel_scene_treatments", group_default(RENDER_DEFAULTS, "panel_scene_treatments", None))
    if isinstance(allowed_raw, str):
        allowed = (str(allowed_raw),)
    elif allowed_raw is None:
        allowed = None
    else:
        allowed = tuple(str(item) for item in allowed_raw)
    return resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace="games.space_shooter.panel_scene_style",
        treatments=allowed,
        treatment_weights=params.get(
            "panel_scene_treatment_weights",
            group_default(RENDER_DEFAULTS, "panel_scene_treatment_weights", None),
        ),
        palette_weights=params.get(
            "panel_scene_palette_weights",
            group_default(RENDER_DEFAULTS, "panel_scene_palette_weights", None),
        ),
    )


def run_space_shooter_lifecycle(
    *,
    namespace: str,
    prompt_query_key: str,
    supported_queries: tuple[str, ...],
    default_query: str,
    task_params: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    build_objective: ObjectiveBuilder,
    highlight_player_lane: bool = False,
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
    axes = resolve_scene_axes(namespace=str(namespace), instance_seed=int(instance_seed), params=resolved_params)
    render_params = resolve_space_shooter_render_params(resolved_params, instance_seed=int(instance_seed))
    last_error: Exception | None = None
    objective: SpaceShooterObjective | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{str(namespace)}.attempt.{int(attempt_index)}")
        try:
            objective = build_objective(rng, resolved_params, axes, int(instance_seed))
        except ValueError as exc:
            last_error = exc
            continue
        break
    if objective is None:
        raise RuntimeError(f"{namespace} failed to construct a valid space-shooter scene: {last_error}") from last_error

    panel_style, panel_style_meta = _resolve_panel_style(instance_seed=int(instance_seed), params=resolved_params)
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered = render_space_shooter_scene(
        lane_count=int(objective.sample.lane_count),
        player_lane=int(objective.sample.player_lane),
        enemies=objective.sample.enemies,
        projectiles=objective.sample.projectiles,
        background=background,
        style_variant=str(axes.style_variant),
        params=render_params,
        highlight_player_lane=bool(highlight_player_lane),
        show_enemy_labels=bool(objective.show_enemy_labels),
        visible_enemy_label_ids=objective.visible_enemy_label_ids,
        panel_style=panel_style,
    )
    if rendered.render_map.get("panel_scene_style") is None:
        rendered.render_map["panel_scene_style"] = dict(panel_style_meta)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=resolved_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_artifacts = objective.build_annotation(objective.sample, rendered)
    prompt, prompt_variants, prompt_meta = build_space_shooter_prompt(
        str(prompt_query_key),
        json_example=str(objective.json_example),
        json_example_answer_only=str(objective.json_example_answer_only),
        instance_seed=int(instance_seed),
    )
    trace_params = space_shooter_trace_params(
        sample=objective.sample,
        prompt_query_key=str(prompt_query_key),
        scene_variant_probabilities=dict(axes.scene_variant_probabilities),
        style_variant=str(axes.style_variant),
        style_variant_probabilities=dict(axes.style_variant_probabilities),
        lane_count_probabilities=dict(axes.lane_count_probabilities),
        enemy_count_probabilities=dict(axes.enemy_count_probabilities),
        public_query_probabilities=dict(public_query_probabilities),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_meta["prompt_artifacts"],
        query_id=str(selected_public_query),
        params=trace_params,
    )
    trace_payload = build_space_shooter_trace_payload(
        sample=objective.sample,
        rendered=rendered,
        prompt_query_key=str(prompt_query_key),
        style_variant=str(axes.style_variant),
        annotation_artifacts=annotation_artifacts,
        background_meta=background_meta,
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
