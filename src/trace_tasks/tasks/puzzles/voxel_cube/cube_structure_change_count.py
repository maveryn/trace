"""Public voxel-cube task for counting cube additions or removals."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import VoxelBinding, VoxelCubeSceneTask
from .shared.annotations import bbox_set_annotation
from .shared.rendering import render_change_scene
from .shared.sampling import sample_structure_change_dataset
from .shared.state import DOMAIN, SCENE_ID, ChangeDataset

TASK_ID = "task_puzzles__voxel_cube__cube_structure_change_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "cube_structure_change_count_query"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.cube_structure_change_count"
_PROMPT_QUERY_BY_CHANGE_TYPE = {
    "missing_to_complete": "missing_to_complete_cuboid_count",
    "removed": "removed_cube_count",
}


def _sample_case(params, generation_defaults, rng):
    """Construct one before/after voxel change dataset."""

    return sample_structure_change_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _prompt_query_key(dataset: ChangeDataset) -> str:
    """Return the prompt query key for the sampled change operation."""

    change_type = str(dataset.semantic_params.get("change_type", ""))
    if change_type not in _PROMPT_QUERY_BY_CHANGE_TYPE:
        raise ValueError(f"unsupported voxel change_type: {change_type}")
    return str(_PROMPT_QUERY_BY_CHANGE_TYPE[change_type])


def _bind_output(dataset: ChangeDataset, visual, selected_query, branch_probabilities):
    """Bind the integer change count and both compared structures."""

    _ = (str(selected_query), dict(branch_probabilities))
    rendered = visual["rendered_scene"]
    bboxes = [rendered.reference_stack_bbox_px, rendered.changed_stack_bbox_px]
    if any(bbox is None for bbox in bboxes):
        raise ValueError("change task requires two rendered structure boxes")
    annotation_gt, projected_annotation, witness_symbolic = bbox_set_annotation(
        [bbox for bbox in bboxes if bbox is not None],
        role="compared_voxel_structures",
    )
    return VoxelBinding(
        answer_gt=TypedValue(type="integer", value=int(dataset.answer_value)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params=dict(dataset.semantic_params),
        execution_fields={
            "annotation_policy": "bbox_set_reference_and_changed_structures",
        },
    )


@register_task
class PuzzlesVoxelCubeCubeStructureChangeCountTask(VoxelCubeSceneTask):
    """Count cubes that differ between two rendered voxel structures."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'matching')
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_case)
    render_builder = staticmethod(render_change_scene)
    prompt_query_key_resolver = staticmethod(_prompt_query_key)
    output_binder = staticmethod(_bind_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one voxel structure-change count case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output


__all__ = [
    "PuzzlesVoxelCubeCubeStructureChangeCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
