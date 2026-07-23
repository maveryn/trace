"""Solve a missing cuboid dimension from volume and two known dimensions."""

from __future__ import annotations

from typing import Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import segment_annotation_artifacts

from ._lifecycle import RectangularSolidObjectivePlan, run_rectangular_solid_public_entry
from .shared.construction import resolve_cuboid_missing_dimension
from .shared.defaults import DOMAIN
from .shared.rendering import render_cuboid_measure_scene
from .shared.state import RenderedRectangularSolidScene


TASK_ID = "task_geometry__rectangular_solid__cuboid_volume_missing_dimension_value"
QUERY_ID_MISSING_LENGTH = "missing_length_from_volume"
QUERY_ID_MISSING_WIDTH = "missing_width_from_volume"
QUERY_ID_MISSING_HEIGHT = "missing_height_from_volume"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (
    QUERY_ID_MISSING_LENGTH,
    QUERY_ID_MISSING_WIDTH,
    QUERY_ID_MISSING_HEIGHT,
)
DEFAULT_QUERY_ID = QUERY_ID_MISSING_LENGTH
PROMPT_TASK_KEY = "cuboid_volume_missing_dimension_value"
TARGET_ROLE_BY_QUERY_ID = {
    QUERY_ID_MISSING_LENGTH: "length",
    QUERY_ID_MISSING_WIDTH: "width",
    QUERY_ID_MISSING_HEIGHT: "height",
}


def _target_dimension_segment_annotation(rendered: RenderedRectangularSolidScene):
    """Return one segment witness for the dimension marked with ?."""

    target_role = str(rendered.render_map["target_role"])
    artifacts = segment_annotation_artifacts(rendered.render_map["dimension_segments"][target_role])
    return artifacts.annotation_gt, dict(artifacts.projected_annotation)


def _prepare_missing_dimension_objective(
    instance_seed,
    task_params: Mapping[str, object],
    selected_branch,
    branch_probabilities,
):
    """Bind the hidden cuboid dimension requested by the selected branch."""

    target_role = TARGET_ROLE_BY_QUERY_ID[str(selected_branch)]
    problem = resolve_cuboid_missing_dimension(
        instance_seed=int(instance_seed),
        params=task_params,
        target_role=str(target_role),
        sampling_label=f"{TASK_ID}.{selected_branch}",
    )
    trace_values = {
        "target_role": str(problem.target_role),
        "length": int(problem.length),
        "width": int(problem.width),
        "height": int(problem.height),
        "volume": int(problem.volume),
        "surface_area": int(problem.surface_area),
    }
    return RectangularSolidObjectivePlan(
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_branch_key=str(selected_branch),
        problem=problem,
        render_scene=render_cuboid_measure_scene,
        bind_annotation=_target_dimension_segment_annotation,
        answer_gt=TypedValue(type="integer", value=int(problem.answer)),
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "target_role": str(problem.target_role),
            "cuboid_case_probabilities": dict(problem.case_probabilities),
            "answer_support_probabilities": dict(problem.answer_support_probabilities),
            **dict(trace_values),
        },
        trace_values=trace_values,
    )


@register_task
class GeometryRectangularSolidCuboidVolumeMissingDimensionValueTask:
    """Solve one hidden cuboid dimension from the visible volume."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_missing_dimension_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate one cuboid missing-dimension problem."""

        return run_rectangular_solid_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
