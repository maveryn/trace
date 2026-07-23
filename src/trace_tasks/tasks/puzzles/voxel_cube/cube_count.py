"""Public voxel-cube task for counting unit cubes."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import VoxelBinding, VoxelCubeSceneTask
from .shared.annotations import scalar_bbox_annotation
from .shared.rendering import render_single_stack_scene
from .shared.sampling import sample_cube_count_dataset
from .shared.state import DOMAIN, SCENE_ID, CountDataset

TASK_ID = "task_puzzles__voxel_cube__cube_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "cube_count_query"
PROMPT_QUERY_KEY = "cube_count"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.cube_count"


def _sample_case(params, generation_defaults, rng):
    """Construct one voxel unit-cube count dataset."""

    return sample_cube_count_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _prompt_query_key(_dataset: CountDataset) -> str:
    """Return this task's prompt query key."""

    return PROMPT_QUERY_KEY


def _bind_output(dataset: CountDataset, visual, selected_query, branch_probabilities):
    """Bind the cube-count answer and one structure bbox witness."""

    _ = (str(selected_query), dict(branch_probabilities))
    annotation_gt, projected_annotation, witness_symbolic = scalar_bbox_annotation(
        visual["rendered_scene"].stack_bbox_px,
        role="voxel_structure",
    )
    return VoxelBinding(
        answer_gt=TypedValue(type="integer", value=int(dataset.answer_value)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params={"answer_schema": "integer_count"},
        execution_fields={
            "annotation_policy": "scalar_bbox_voxel_structure",
        },
    )


@register_task
class PuzzlesVoxelCubeCubeCountTask(VoxelCubeSceneTask):
    """Count unit cubes in the rendered voxel structure."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_case)
    render_builder = staticmethod(render_single_stack_scene)
    prompt_query_key_resolver = staticmethod(_prompt_query_key)
    output_binder = staticmethod(_bind_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one voxel cube-count case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output


__all__ = ["PuzzlesVoxelCubeCubeCountTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
