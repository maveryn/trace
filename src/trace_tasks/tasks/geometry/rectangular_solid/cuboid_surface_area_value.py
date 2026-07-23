"""Compute total surface area of a labeled cuboid."""

from __future__ import annotations

from typing import Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts

from ._lifecycle import RectangularSolidObjectivePlan, run_rectangular_solid_public_entry
from .shared.construction import resolve_cuboid_surface_area
from .shared.defaults import DOMAIN
from .shared.rendering import render_cuboid_measure_scene
from .shared.state import RenderedRectangularSolidScene


TASK_ID = "task_geometry__rectangular_solid__cuboid_surface_area_value"
QUERY_ID_SURFACE_AREA = "single"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (QUERY_ID_SURFACE_AREA,)
DEFAULT_QUERY_ID = QUERY_ID_SURFACE_AREA
PROMPT_TASK_KEY = "cuboid_surface_area_value"


def _cuboid_bbox_annotation(rendered: RenderedRectangularSolidScene):
    """Return one bbox witness for the cuboid whose surface area is requested."""

    artifacts = bbox_annotation_artifacts(rendered.render_map["cuboid_bbox"])
    return artifacts.annotation_gt, dict(artifacts.projected_annotation)


def _prepare_surface_area_objective(
    instance_seed,
    task_params: Mapping[str, object],
    selected_branch,
    branch_probabilities,
):
    """Bind the total surface-area objective for one cuboid."""

    problem = resolve_cuboid_surface_area(
        instance_seed=int(instance_seed),
        params=task_params,
        sampling_label=f"{TASK_ID}.{selected_branch}",
    )
    trace_values = {
        "target_role": "surface_area",
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
        bind_annotation=_cuboid_bbox_annotation,
        answer_gt=TypedValue(type="integer", value=int(problem.answer)),
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "target_role": "surface_area",
            "cuboid_case_probabilities": dict(problem.case_probabilities),
            "answer_support_probabilities": dict(problem.answer_support_probabilities),
            **dict(trace_values),
        },
        trace_values=trace_values,
    )


@register_task
class GeometryRectangularSolidCuboidSurfaceAreaValueTask:
    """Compute total surface area from visible cuboid dimensions."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_surface_area_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate one cuboid surface-area problem."""

        return run_rectangular_solid_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
