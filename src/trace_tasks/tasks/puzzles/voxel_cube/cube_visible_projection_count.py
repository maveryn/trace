"""Public voxel-cube task for counting filled projection cells."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import VoxelBinding, VoxelCubeSceneTask
from .shared.annotations import bbox_set_annotation
from .shared.rendering import render_projection_count_scene
from .shared.sampling import sample_visible_projection_dataset
from .shared.state import DOMAIN, SCENE_ID, ProjectionCountDataset

TASK_ID = "task_puzzles__voxel_cube__cube_visible_projection_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "cube_visible_projection_count_query"
PROMPT_QUERY_KEY = "visible_projection_count"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.cube_visible_projection_count"


def _sample_case(params, generation_defaults, rng):
    """Construct one target projection-cell count dataset."""

    return sample_visible_projection_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _prompt_query_key(_dataset: ProjectionCountDataset) -> str:
    """Return this task's prompt query key."""

    return PROMPT_QUERY_KEY


def _bind_output(
    dataset: ProjectionCountDataset,
    visual,
    selected_query,
    branch_probabilities,
):
    """Bind the projection-cell count and filled-cell bbox set."""

    _ = (str(selected_query), dict(branch_probabilities))
    cell_map = visual["rendered_scene"].projection_cell_bbox_map
    bboxes = [
        cell_map[f"{int(row)}_{int(col)}"]
        for row, col in dataset.projection.filled_cells
    ]
    annotation_gt, projected_annotation, witness_symbolic = bbox_set_annotation(
        bboxes,
        role="filled_projection_cells",
    )
    return VoxelBinding(
        answer_gt=TypedValue(type="integer", value=int(dataset.answer_value)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params=dict(dataset.semantic_params),
        execution_fields={
            "annotation_policy": "bbox_set_filled_projection_cells",
            "filled_projection_cells": [
                [int(row), int(col)] for row, col in dataset.projection.filled_cells
            ],
        },
    )


@register_task
class PuzzlesVoxelCubeCubeVisibleProjectionCountTask(VoxelCubeSceneTask):
    """Count cells filled by one orthographic projection."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'transformation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_case)
    render_builder = staticmethod(render_projection_count_scene)
    prompt_query_key_resolver = staticmethod(_prompt_query_key)
    output_binder = staticmethod(_bind_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one voxel projection-cell count case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output


__all__ = [
    "PuzzlesVoxelCubeCubeVisibleProjectionCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
