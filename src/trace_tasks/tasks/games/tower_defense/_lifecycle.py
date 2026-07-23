"""Scene-private lifecycle orchestration for tower-defense public tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.annotations import (
    path_node_point_annotation,
    path_node_point_set_annotation,
    tower_bbox_set_annotation,
    tower_point_annotation,
)
from .shared.defaults import GEN_DEFAULTS, POST_IMAGE_NOISE_DEFAULTS, PROMPT_DEFAULTS
from .shared.prompts import build_tower_defense_prompt_artifacts
from .shared.rendering import render_tower_defense_scene, resolve_tower_defense_render_params
from .shared.rules import enemy_entity_id, visible_tower_trace
from .shared.sampling import axis_support_metadata, resolve_tower_defense_axes
from .shared.state import SCENE_ID, SCENE_NAMESPACE, TowerDefenseAxes, TowerDefenseRenderParams, TowerDefenseSample
from ..shared.scene_style import make_panel_scene_background, resolve_game_panel_scene_style


AttemptBuilder = Callable[[Any, TowerDefenseAxes, TowerDefenseRenderParams, Mapping[str, Any]], TowerDefenseSample]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "TowerDefenseObjectivePlan"]


@dataclass(frozen=True)
class TowerDefenseObjectivePlan:
    """Task-owned hooks for one tower-defense objective."""

    attempt_namespace: str
    prompt_query_key: str
    annotation_kind: str
    tower_count_support_key: str
    tower_count_fallback: Sequence[int]
    target_answer_support_key: str
    target_answer_fallback: Sequence[int]
    construct_attempt: AttemptBuilder
    answer_type: str = "integer"
    json_example_answer: int | str = 2
    path_count_must_cover_target: bool = False
    tower_count_must_cover_target: bool = False
    trace_params: Mapping[str, Any] = field(default_factory=dict)


def _annotation_for_objective(
    *,
    rendered_scene: Any,
    sample: TowerDefenseSample,
    annotation_kind: str,
) -> AnnotationArtifacts:
    """Project the task-owned witness ids through the requested annotation family."""

    if str(annotation_kind) == "tower_bbox_set":
        return tower_bbox_set_annotation(rendered_scene, sample.annotation_entity_ids)
    if str(annotation_kind) == "path_point_set":
        return path_node_point_set_annotation(rendered_scene, sample.annotation_entity_ids)
    if str(annotation_kind) == "path_point":
        if len(sample.annotation_entity_ids) != 1:
            raise ValueError("path_point annotation requires exactly one witness id")
        return path_node_point_annotation(rendered_scene, sample.annotation_entity_ids[0])
    if str(annotation_kind) == "tower_point":
        if len(sample.annotation_entity_ids) != 1:
            raise ValueError("tower_point annotation requires exactly one witness id")
        return tower_point_annotation(rendered_scene, sample.annotation_entity_ids[0])
    raise ValueError(f"unsupported tower-defense annotation kind: {annotation_kind}")


def _trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    axes: TowerDefenseAxes,
    sample: TowerDefenseSample,
    rendered_scene: Any,
    image_size: tuple[int, int],
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    prompt_query_key: str,
    query_id: str,
    query_probabilities: Mapping[str, float],
    query_params: Mapping[str, Any],
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the verifier trace after the public task binds the objective."""

    return {
        "scene_ir": {
            "scene_kind": f"games_tower_defense_{str(axes.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_id": SCENE_ID,
                "scene_variant": str(axes.scene_variant),
                "query_id": str(query_id),
                "prompt_query_key": str(prompt_query_key),
                "style_variant": str(axes.style_variant),
                "tower_count": len(sample.towers),
                "path_segment_count": len(sample.path_points_px),
                "marked_enemy_id": enemy_entity_id() if sample.enemy is not None else None,
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.scene_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "map_width_px": int(sample.map_width_px),
            "map_height_px": int(sample.map_height_px),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style", {})),
            "tower_defense_style": dict(rendered_scene.render_map.get("tower_defense_style", {})),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(axes.scene_variant),
            "query_id": str(query_id),
            "prompt_query_key": str(prompt_query_key),
            "query_id_probabilities": dict(query_probabilities),
            "style_variant": str(axes.style_variant),
            "map_width_px": int(sample.map_width_px),
            "map_height_px": int(sample.map_height_px),
            "path_points_px_local": [
                [round(float(point[0]), 3), round(float(point[1]), 3)]
                for point in sample.path_points_px
            ],
            "labeled_path_enemy_options": [
                {
                    "label": str(label),
                    "path_index": int(index),
                    "entity_id": f"path_segment_{int(index):02d}",
                }
                for index, label in sample.labeled_path_enemy_options
            ],
            "exit_path_index": int(len(sample.path_points_px) - 1),
            "marked_enemy": (
                {
                    "enemy_id": str(sample.enemy.enemy_id),
                    "center_px_local": [
                        round(float(sample.enemy.center_px[0]), 3),
                        round(float(sample.enemy.center_px[1]), 3),
                    ],
                    "path_index": int(sample.enemy.path_index),
                }
                if sample.enemy is not None
                else None
            ),
            "towers": list(visible_tower_trace(sample.towers)),
            "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            "sample_metadata": dict(sample.metadata),
            "construction_mode": str(sample.construction_mode),
            "answer": sample.answer,
            "target_answer": int(sample.target_answer),
            **dict(query_params),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults.get("bundle_id", ""))},
    }


def run_tower_defense_lifecycle(
    *,
    task_id: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run query selection, construction retries, rendering, prompting, and output assembly."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    objective = prepare_objective(int(instance_seed), task_params, str(query_id), query_probabilities)
    render_params = resolve_tower_defense_render_params(task_params, instance_seed=int(instance_seed))
    axes = resolve_tower_defense_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        namespace_root=SCENE_NAMESPACE,
        tower_count_support_key=str(objective.tower_count_support_key),
        tower_count_fallback=tuple(int(value) for value in objective.tower_count_fallback),
        target_answer_support_key=str(objective.target_answer_support_key),
        target_answer_fallback=tuple(int(value) for value in objective.target_answer_fallback),
        path_count_must_cover_target=bool(objective.path_count_must_cover_target),
        tower_count_must_cover_target=bool(objective.tower_count_must_cover_target),
    )

    sample: TowerDefenseSample | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, axes, render_params, task_params)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate a valid tower-defense map after {max_attempts} attempts")

    panel_style, _panel_style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.panel_scene_style",
        treatment_weights=GEN_DEFAULTS.get("panel_scene_treatment_weights", None),
        palette_weights=GEN_DEFAULTS.get("panel_scene_palette_weights", None),
    )
    background, background_meta = make_panel_scene_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=panel_style,
    )
    rendered_scene = render_tower_defense_scene(
        path_points_px=sample.path_points_px,
        towers=sample.towers,
        enemy=sample.enemy,
        labeled_path_enemy_options=sample.labeled_path_enemy_options,
        show_exit_marker=bool(sample.show_exit_marker),
        background=background,
        style_variant=str(axes.style_variant),
        params=render_params,
        panel_style=panel_style,
    )
    annotation_artifacts = _annotation_for_objective(
        rendered_scene=rendered_scene,
        sample=sample,
        annotation_kind=str(objective.annotation_kind),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=task_params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    prompt_defaults, prompt_artifacts = build_tower_defense_prompt_artifacts(
        prompt_defaults=PROMPT_DEFAULTS,
        scene_variant=str(axes.scene_variant),
        prompt_query_key=str(objective.prompt_query_key),
        annotation_type=str(annotation_artifacts.annotation_type),
        example_answer=objective.json_example_answer,
        instance_seed=int(instance_seed),
    )
    query_params = {
        **axis_support_metadata(axes),
        "query_id_probabilities": dict(query_probabilities),
        "prompt_query_key": str(objective.prompt_query_key),
        "answer": sample.answer,
        **dict(objective.trace_params),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params=query_params,
    )
    trace_payload = _trace_payload(
        annotation_artifacts=annotation_artifacts,
        axes=axes,
        sample=sample,
        rendered_scene=rendered_scene,
        image_size=(int(image.size[0]), int(image.size[1])),
        prompt_defaults=prompt_defaults,
        prompt_query_spec=prompt_query_spec,
        prompt_query_key=str(objective.prompt_query_key),
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        query_params=query_params,
        background_meta=background_meta,
        post_noise_meta=post_noise_meta,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(objective.answer_type), value=sample.answer),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
    )


__all__ = ["TowerDefenseObjectivePlan", "run_tower_defense_lifecycle"]
