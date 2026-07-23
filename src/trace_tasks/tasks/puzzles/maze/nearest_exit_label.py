"""Public maze task for selecting the nearest exit to START."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    build_maze_task_output,
    prepare_maze_visual_case,
    resolve_maze_public_branch,
    resolve_maze_scene_variant,
    retry_maze_generation,
)
from .shared.annotations import single_item_point
from .shared.sampling import sample_nearest_exit_maze
from .shared.state import DOMAIN, SCENE_ID

TASK_ID = "task_puzzles__maze__nearest_exit_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "nearest_exit_label_query"
PROMPT_QUERY_KEY = "nearest_exit_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.nearest_exit_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesMazeNearestExitLabelTask:
    """Return the reachable exit with the shortest corridor path from START."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'topology')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one nearest-exit label task with scalar point annotation."""

        return retry_maze_generation(
            build_case=_build_nearest_exit_label_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=max_attempts,
        )


def _build_nearest_exit_label_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Sample four reachable exits, solve path lengths, and bind the nearest."""

    selected_branch, branch_probabilities, task_params = resolve_maze_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        task_id=TASK_ID,
        namespace=_NAMESPACE_BASE,
    )
    scene_variant, scene_variant_probabilities = resolve_maze_scene_variant(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=_NAMESPACE_BASE,
    )
    dataset = sample_nearest_exit_maze(
        scene_variant=str(scene_variant),
        params=task_params,
        instance_seed=int(instance_seed),
        generation_defaults=_GEN_DEFAULTS,
        max_attempts=int(max_attempts),
    )
    visual = prepare_maze_visual_case(
        dataset=dataset,
        params=task_params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_dynamic_slots={},
        namespace=_NAMESPACE_BASE,
    )
    supporting_item_ids = [str(value) for value in dataset["supporting_item_ids"]]
    if supporting_item_ids != [str(dataset["nearest_exit_item_id"])]:
        raise ValueError("nearest-exit annotation item drifted from solved nearest exit")
    annotation_gt, projected_annotation, witness_symbolic = single_item_point(
        visual["rendered_scene"].item_point_map,
        supporting_item_ids[0],
    )
    answer_value = str(dataset["answer_value"])
    nearest_label = str(dataset["nearest_label"])
    if answer_value != nearest_label:
        raise ValueError("nearest-exit answer drifted from solved path lengths")
    answer_gt = TypedValue(type="string", value=answer_value)
    return build_maze_task_output(
        dataset=dataset,
        visual=visual,
        public_query_id=str(selected_branch),
        branch_probabilities=branch_probabilities,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        prompt_query_key=PROMPT_QUERY_KEY,
        semantic_params={
            "nearest_exit_label": nearest_label,
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
        relation_fields={
            "nearest_exit_label": nearest_label,
            "nearest_exit_item_id": str(dataset["nearest_exit_item_id"]),
            "nearest_exit_cell": list(dataset["nearest_exit_cell"]),
            "nearest_exit_path_length_edges": int(dataset["nearest_exit_path_length_edges"]),
        },
        execution_fields={
            "nearest_exit_label": nearest_label,
            "nearest_exit_item_id": str(dataset["nearest_exit_item_id"]),
            "nearest_exit_cell": list(dataset["nearest_exit_cell"]),
            "nearest_exit_path_cells": [list(cell) for cell in dataset["nearest_exit_path_cells"]],
            "nearest_exit_path_length_edges": int(dataset["nearest_exit_path_length_edges"]),
            "nearest_exit_margin_edges": int(dataset["nearest_exit_margin_edges"]),
            "nearest_exit_min_gap_edges": int(dataset["nearest_exit_min_gap_edges"]),
            "exit_path_lengths_by_label": dict(dataset["exit_path_lengths_by_label"]),
            "exit_paths_by_label": {
                str(label): [list(cell) for cell in path]
                for label, path in dataset["exit_paths_by_label"].items()
            },
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
    )


__all__ = [
    "PROMPT_QUERY_KEY",
    "PuzzlesMazeNearestExitLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
