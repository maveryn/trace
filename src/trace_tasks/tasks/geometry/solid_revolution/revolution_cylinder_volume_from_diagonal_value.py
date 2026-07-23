"""Cylinder volume from a rotating rectangle with a visible diagonal."""

from __future__ import annotations

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import _run_cylinder_diagonal
from .shared.defaults import DOMAIN, SCENE_ID

TASK_ID = "task_geometry__solid_revolution__revolution_cylinder_volume_from_diagonal_value"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
ANNOTATION_KEYS = (
    "source_diagram_bbox",
    "resulting_solid_bbox",
)


def _build_diagonal_cylinder_contract() -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return public task-owned query and annotation contract fields."""

    return TASK_ID, SUPPORTED_QUERY_IDS, ANNOTATION_KEYS


@register_task
class GeometrySolidRevolutionCylinderVolumeFromDiagonalValueTask:
    """Solve cylinder volume after deriving diameter from a rectangle diagonal."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = QUERY_ID

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        return _run_cylinder_diagonal(
            task_id=TASK_ID,
            query_id=QUERY_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            annotation_keys=ANNOTATION_KEYS,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
        )


__all__ = [
    "ANNOTATION_KEYS",
    "GeometrySolidRevolutionCylinderVolumeFromDiagonalValueTask",
    "SCENE_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
