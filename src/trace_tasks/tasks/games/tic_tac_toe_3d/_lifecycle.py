"""Private lifecycle plumbing for 3D Tic-Tac-Toe public tasks."""

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

from .shared.annotations import cell_bbox_set_annotation, cell_point_set_annotation
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS, PROMPT_DEFAULTS
from .shared.prompts import build_tic_tac_toe_3d_prompt_artifacts
from .shared.rendering import render_tic_tac_toe_3d_scene
from .shared.rules import WINNING_LINES, board_trace, coord_id
from .shared.sampling import axis_support_metadata, resolve_tic_tac_toe_3d_axes
from .shared.state import (
    BOARD_SIZE,
    LAYERS,
    SCENE_ID,
    SCENE_NAMESPACE,
    TicTacToe3DAxes,
    TicTacToe3DSample,
)

AttemptBuilder = Callable[[Any, TicTacToe3DAxes], TicTacToe3DSample]
ObjectivePreparer = Callable[
    [int, Mapping[str, Any], str, Mapping[str, float]], "TicTacToe3DObjectivePlan"
]


@dataclass(frozen=True)
class TicTacToe3DObjectivePlan:
    """Task-owned objective hooks for one generated 3D Tic-Tac-Toe board."""

    attempt_namespace: str
    prompt_query_key: str
    answer_hint_key: str
    annotation_hint_key: str
    annotation_kind: str
    json_example: str
    json_example_answer_only: str
    construct_attempt: AttemptBuilder
    prompt_dynamic_slots: Mapping[str, Any] = field(default_factory=dict)
    trace_params: Mapping[str, Any] = field(default_factory=dict)


def prepare_option_move_objective_from_semantics(
    *,
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    target_player_by_branch: Mapping[str, str],
    sample_scene: Callable[..., TicTacToe3DSample],
    attempt_prefix: str,
    branch_trace_key: str,
    json_example: str,
    json_example_answer_only: str,
    option_labels: Sequence[str],
) -> TicTacToe3DObjectivePlan:
    """Build a win/block option objective from public task semantics."""

    if str(selected_branch) not in set(str(key) for key in target_player_by_branch):
        raise ValueError(f"unsupported 3D Tic-Tac-Toe move branch: {selected_branch}")
    target_player = str(target_player_by_branch[str(selected_branch)])

    def construct_move_options(rng, axes):
        return sample_scene(
            rng=rng,
            target_player=target_player,
            option_count=int(axes.option_count),
            answer_option_index=int(axes.answer_option_index),
        )

    return TicTacToe3DObjectivePlan(
        attempt_namespace=f"games.tic_tac_toe_3d.{attempt_prefix}.{target_player}",
        prompt_query_key=str(selected_branch),
        answer_hint_key=f"answer_hint_{selected_branch}",
        annotation_hint_key=f"annotation_hint_{selected_branch}",
        annotation_kind="cell_bbox_set",
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
        construct_attempt=construct_move_options,
        trace_params={
            "target_player": target_player,
            str(branch_trace_key): str(selected_branch),
            f"{branch_trace_key}_probabilities": dict(branch_probabilities),
            "available_option_labels": [str(label) for label in option_labels],
        },
    )


def _annotation_for_objective(
    *,
    rendered_scene: Any,
    sample: TicTacToe3DSample,
    annotation_kind: str,
) -> AnnotationArtifacts:
    """Project task-owned board coordinates through the requested shape family."""

    if str(annotation_kind) == "cell_bbox_set":
        return cell_bbox_set_annotation(rendered_scene, sample.annotation_coords)
    if str(annotation_kind) == "cell_point_set":
        return cell_point_set_annotation(rendered_scene, sample.annotation_coords)
    raise ValueError(f"unsupported 3D Tic-Tac-Toe annotation kind: {annotation_kind}")


def _trace_payload(
    *,
    annotation_artifacts: AnnotationArtifacts,
    annotation_entity_ids: Sequence[str],
    axes: TicTacToe3DAxes,
    sample: TicTacToe3DSample,
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
    """Assemble one trace payload after the public task binds the objective."""

    return {
        "scene_ir": {
            "scene_kind": "games_tic_tac_toe_3d_board",
            "entities": [dict(entity) for entity in rendered_scene.entities],
            "relations": {
                "scene_id": SCENE_ID,
                "query_id": str(query_id),
                "prompt_query_key": str(prompt_query_key),
                "layout_variant": str(axes.layout_variant),
                "style_variant": str(axes.style_variant),
                "board_size": int(BOARD_SIZE),
                "target_player": str(sample.target_player),
                "target_layer": str(sample.target_layer),
                "annotation_entity_ids": [
                    str(entity_id) for entity_id in annotation_entity_ids
                ],
            },
        },
        "query_spec": dict(prompt_query_spec),
        "render_spec": {
            "scene_variant": str(axes.layout_variant),
            "layout_variant": str(axes.layout_variant),
            "style_variant": str(axes.style_variant),
            "canvas_width": int(image_size[0]),
            "canvas_height": int(image_size[1]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(
                rendered_scene.style_meta.get("panel_scene_style", {})
            ),
            "tic_tac_toe_3d_board_style": dict(
                rendered_scene.style_meta.get("tic_tac_toe_3d_board_style", {})
            ),
            "text_style": dict(rendered_scene.style_meta.get("text_style", {})),
            "effective_cell_size_px": int(
                rendered_scene.render_map["effective_cell_size_px"]
            ),
            "effective_mark_size_px": int(
                rendered_scene.render_map["effective_mark_size_px"]
            ),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "query_id": str(query_id),
            "prompt_query_key": str(prompt_query_key),
            "query_id_probabilities": dict(query_probabilities),
            "board_size": int(BOARD_SIZE),
            "winning_line_count": int(len(WINNING_LINES)),
            "board_layers": board_trace(sample.board),
            "layer_names": [str(label) for _key, label in LAYERS],
            "target_player": str(sample.target_player),
            "target_layer": str(sample.target_layer),
            "answer": sample.answer,
            "answer_cell": (
                None
                if sample.answer_cell is None
                else [int(v) for v in sample.answer_cell]
            ),
            "support_cells": [
                [int(v) for v in coord] for coord in sample.support_cells
            ],
            "option_cells": [[int(v) for v in coord] for coord in sample.option_cells],
            "annotation_coords": [
                [int(v) for v in coord] for coord in sample.annotation_coords
            ],
            "annotation_entity_ids": [
                str(entity_id) for entity_id in annotation_entity_ids
            ],
            **dict(query_params),
            **dict(sample.metadata),
        },
        "witness_symbolic": {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in annotation_entity_ids],
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "prompt_metadata": {"bundle_id": str(prompt_defaults.get("bundle_id", ""))},
    }


def run_tic_tac_toe_3d_lifecycle(
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
    axes = resolve_tic_tac_toe_3d_axes(
        instance_seed=int(instance_seed),
        params=task_params,
        namespace_root=SCENE_NAMESPACE,
    )
    objective = prepare_objective(
        int(instance_seed), task_params, str(query_id), query_probabilities
    )
    sample = None
    for attempt_index in range(max(1, int(max_attempts))):
        rng = spawn_rng(
            int(instance_seed),
            f"{objective.attempt_namespace}.attempt.{int(attempt_index)}",
        )
        try:
            sample = objective.construct_attempt(rng, axes)
        except ValueError:
            continue
        break
    if sample is None:
        raise RuntimeError(
            f"{task_id} failed to generate a valid scene after {max_attempts} attempts"
        )

    rendered_scene = render_tic_tac_toe_3d_scene(
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
    annotation_entity_ids = tuple(coord_id(coord) for coord in sample.annotation_coords)
    prompt_defaults_used, prompt_artifacts = build_tic_tac_toe_3d_prompt_artifacts(
        prompt_defaults=PROMPT_DEFAULTS,
        prompt_query_key=str(objective.prompt_query_key),
        answer_hint_key=str(objective.answer_hint_key),
        annotation_hint_key=str(objective.annotation_hint_key),
        json_example=str(objective.json_example),
        json_example_answer_only=str(objective.json_example_answer_only),
        target_layer_label=str(
            objective.prompt_dynamic_slots.get(
                "target_layer_label", sample.target_layer
            )
        ),
        target_player=str(
            objective.prompt_dynamic_slots.get("target_player", sample.target_player)
        ),
        instance_seed=int(instance_seed),
    )
    query_params = {
        **axis_support_metadata(task_params, axes),
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
        background_meta=rendered_scene.background_meta,
        post_noise_meta=post_noise_meta,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type=str(sample.answer_type), value=sample.answer),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=image,
        image_id="img0",
        trace_payload=dict(trace_payload),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(query_id),
    )


__all__ = ["TicTacToe3DObjectivePlan", "run_tic_tac_toe_3d_lifecycle"]
