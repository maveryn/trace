"""Private lifecycle plumbing for tower draughts board public tasks."""

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

from .shared.annotations import cell_bbox_set_annotation, marked_or_stack_bbox_set_annotation, stack_bbox_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, PROMPT_DEFAULTS
from .shared.prompts import build_tower_draughts_prompt_artifacts
from .shared.rendering import render_tower_draughts_scene
from .shared.rules import cell_id, player_name, stack_id
from .shared.sampling import MaxCountResolver, axis_support_metadata, resolve_tower_draughts_axes
from .shared.state import SCENE_ID, SCENE_NAMESPACE, TowerDraughtsAxes, TowerDraughtsSample


AttemptBuilder = Callable[[Any, TowerDraughtsAxes], TowerDraughtsSample]
ObjectivePreparer = Callable[[int, Mapping[str, Any], str, Mapping[str, float]], "TowerDraughtsObjectivePlan"]


@dataclass(frozen=True)
class TowerDraughtsObjectivePlan:
    """Task-owned hooks for one tower draughts board objective."""

    attempt_namespace: str
    prompt_query_key: str
    answer_hint_key: str
    annotation_hint_key: str
    annotation_kind: str
    json_example: str
    json_example_answer_only: str
    target_answer_support_key: str
    target_answer_fallback: Sequence[int]
    max_count_for_board: MaxCountResolver
    construct_attempt: AttemptBuilder
    force_crowned_for_large_target: bool = False
    trace_params: Mapping[str, Any] = field(default_factory=dict)


def _annotation_for_objective(
    *,
    rendered_scene: Any,
    sample: TowerDraughtsSample,
    annotation_kind: str,
) -> AnnotationArtifacts:
    """Project task-owned board coordinates through the requested bbox family."""

    if str(annotation_kind) == "stack_bbox_set":
        return stack_bbox_set_annotation(rendered_scene, sample.annotation_coords)
    if str(annotation_kind) == "marked_or_stack_bbox_set":
        return marked_or_stack_bbox_set_annotation(
            rendered_scene,
            sample.annotation_coords,
            marked_coord=sample.marked_coord,
        )
    if str(annotation_kind) == "cell_bbox_set":
        return cell_bbox_set_annotation(rendered_scene, sample.annotation_coords)
    raise ValueError(f"unsupported tower draughts annotation kind: {annotation_kind}")


def _annotation_entity_ids(*, sample: TowerDraughtsSample, annotation_kind: str) -> tuple[str, ...]:
    """Return stable render entity ids for the task-owned annotation coordinates."""

    if str(annotation_kind) == "cell_bbox_set":
        return tuple(cell_id(coord) for coord in sample.annotation_coords)
    return tuple(
        "stack_marked" if sample.marked_coord is not None and tuple(coord) == tuple(sample.marked_coord) else stack_id(coord)
        for coord in sample.annotation_coords
    )


def _stack_trace(sample: TowerDraughtsSample) -> list[dict[str, Any]]:
    """Serialize visible stacks for verifier trace payloads."""

    return [
        {
            "coord": [int(stack.coord[0]), int(stack.coord[1])],
            "height": int(stack.height),
            "disks": [player_name(int(player)) for player in stack.disks],
            "owner": player_name(int(stack.owner)),
            "top_crowned": bool(stack.top_crowned),
        }
        for stack in sample.stacks
    ]


def _trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: TowerDraughtsAxes,
    sample: TowerDraughtsSample,
    rendered_scene: Any,
    image_size: tuple[int, int],
    prompt_defaults: Mapping[str, Any],
    prompt_query_spec: Mapping[str, Any],
    prompt_query_key: str,
    query_id: str,
    query_probabilities: Mapping[str, float],
    query_params: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble one trace payload after the public task binds the objective."""

    metadata = dict(sample.metadata)
    return {
        "scene_ir": {
            "scene_kind": "games_tower_draughts_board",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "prompt_query_key": str(prompt_query_key),
                "style_variant": str(sample.style_variant),
                "board_size": int(sample.board_size),
                "marked_stack_id": "stack_marked" if sample.marked_coord is not None else "",
                "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "style_variant": str(sample.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.style_meta.get("panel_scene_style", {})),
            "tower_draughts_board_style": dict(rendered_scene.style_meta.get("tower_draughts_board_style", {})),
            "effective_cell_size_px": int(rendered_scene.render_map["effective_cell_size_px"]),
            "effective_disk_radius_px": float(rendered_scene.render_map["effective_disk_radius_px"]),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "query_id": str(query_id),
            "prompt_query_key": str(prompt_query_key),
            "query_id_probabilities": dict(query_probabilities),
            "style_variant": str(sample.style_variant),
            "board_size": int(sample.board_size),
            "construction_mode": str(sample.construction_mode),
            "answer": int(sample.answer),
            "target_answer": int(sample.answer),
            "target_answer_support": [int(value) for value in axes.target_answer_support],
            "target_player": player_name(int(sample.target_player)),
            "marked_player": player_name(int(sample.marked_player)),
            "top_kind": str(sample.top_kind),
            "marked_coord": None if sample.marked_coord is None else [int(sample.marked_coord[0]), int(sample.marked_coord[1])],
            "stacks": _stack_trace(sample),
            "annotation_coords": [[int(coord[0]), int(coord[1])] for coord in sample.annotation_coords],
            "annotation_entity_ids": [str(entity_id) for entity_id in annotation_entity_ids],
            "legal_destinations": list(metadata.get("legal_destinations", [])),
            "captured_stacks": list(metadata.get("captured_stacks", [])),
            **dict(query_params),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(rendered_scene.background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults.get("bundle_id", ""))},
    }


def run_tower_draughts_lifecycle(
    *,
    task_id: str,
    supported_query_ids: Sequence[str],
    default_query_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    prepare_objective: ObjectivePreparer,
) -> TaskOutput:
    """Run branch selection, construction, rendering, prompting, and output assembly."""

    query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=dict(params),
        supported_query_ids=tuple(str(value) for value in supported_query_ids),
        default_query_id=str(default_query_id),
        task_id=str(task_id),
        namespace=f"{task_id}.query",
    )
    objective = prepare_objective(int(instance_seed), task_params, str(query_id), query_probabilities)
    axes = resolve_tower_draughts_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        target_answer_support_key=str(objective.target_answer_support_key),
        target_answer_fallback=tuple(int(value) for value in objective.target_answer_fallback),
        max_count_for_board=objective.max_count_for_board,
        force_crowned_for_large_target=bool(objective.force_crowned_for_large_target),
    )

    sample: TowerDraughtsSample | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(int(instance_seed), f"{objective.attempt_namespace}.attempt.{int(attempt_index)}")
        try:
            sample = objective.construct_attempt(rng, axes)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(f"{task_id} failed to generate a valid tower draughts board after {max_attempts} attempts")

    rendered_scene = render_tower_draughts_scene(
        sample=sample,
        axes=axes,
        instance_seed=int(instance_seed),
        params=task_params,
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
    annotation_entity_ids = _annotation_entity_ids(sample=sample, annotation_kind=str(objective.annotation_kind))
    prompt_defaults_used, prompt_artifacts = build_tower_draughts_prompt_artifacts(
        prompt_defaults=PROMPT_DEFAULTS,
        prompt_query_key=str(objective.prompt_query_key),
        answer_hint_key=str(objective.answer_hint_key),
        annotation_hint_key=str(objective.annotation_hint_key),
        json_example=str(objective.json_example),
        json_example_answer_only=str(objective.json_example_answer_only),
        target_player=int(sample.target_player),
        marked_player=int(sample.marked_player),
        instance_seed=int(instance_seed),
    )
    query_params = {
        **axis_support_metadata(axes),
        "query_id_probabilities": dict(query_probabilities),
        "prompt_query_key": str(objective.prompt_query_key),
        "answer": int(sample.answer),
        **dict(objective.trace_params),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(query_id),
        params=query_params,
    )
    trace_payload = _trace_payload(
        annotation_artifacts=annotation_artifacts,
        annotation_entity_ids=annotation_entity_ids,
        axes=axes,
        sample=sample,
        rendered_scene=rendered_scene,
        image_size=(int(image.size[0]), int(image.size[1])),
        prompt_defaults=prompt_defaults_used,
        prompt_query_spec=prompt_query_spec,
        prompt_query_key=str(objective.prompt_query_key),
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        query_params=query_params,
        post_noise_meta=post_noise_meta,
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
        query_id=str(query_id),
    )


__all__ = ["TowerDraughtsObjectivePlan", "run_tower_draughts_lifecycle"]
