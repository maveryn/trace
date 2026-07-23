"""Public voxel-cube task for choosing a matching projection panel."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import VoxelBinding, VoxelCubeSceneTask
from .shared.annotations import option_bbox_annotation
from .shared.rendering import render_projection_match_scene
from .shared.sampling import sample_projection_match_dataset
from .shared.state import DOMAIN, SCENE_ID, ProjectionMatchDataset

TASK_ID = "task_puzzles__voxel_cube__cube_projection_match_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "cube_projection_match_label_query"
PROMPT_QUERY_KEY = "projection_match_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.cube_projection_match_label"


def _sample_case(params, generation_defaults, rng):
    """Construct one projection matching option dataset."""

    return sample_projection_match_dataset(
        params=params,
        generation_defaults=generation_defaults,
        rng=rng,
    )


def _prompt_query_key(_dataset: ProjectionMatchDataset) -> str:
    """Return this task's prompt query key."""

    return PROMPT_QUERY_KEY


def _bind_output(
    dataset: ProjectionMatchDataset,
    visual,
    selected_query,
    branch_probabilities,
):
    """Bind the selected projection option label and panel bbox."""

    _ = (str(selected_query), dict(branch_probabilities))
    annotation_gt, projected_annotation, witness_symbolic = option_bbox_annotation(
        visual["rendered_scene"].option_panel_bbox_map,
        dataset.answer_label,
    )
    return VoxelBinding(
        answer_gt=TypedValue(type="option_letter", value=str(dataset.answer_label)),
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        semantic_params=dict(dataset.semantic_params),
        execution_fields={
            "annotation_policy": "scalar_bbox_selected_projection_option",
        },
    )


@register_task
class PuzzlesVoxelCubeCubeProjectionMatchLabelTask(VoxelCubeSceneTask):
    """Choose the option panel matching one orthographic projection."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    supported_query_ids = SUPPORTED_QUERY_IDS
    prompt_task_key = PROMPT_TASK_KEY
    namespace = _NAMESPACE_BASE
    sample_builder = staticmethod(_sample_case)
    render_builder = staticmethod(render_projection_match_scene)
    prompt_query_key_resolver = staticmethod(_prompt_query_key)
    output_binder = staticmethod(_bind_output)

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one voxel projection match-label case."""

        output = super().generate(
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )
        return output


__all__ = [
    "PuzzlesVoxelCubeCubeProjectionMatchLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
