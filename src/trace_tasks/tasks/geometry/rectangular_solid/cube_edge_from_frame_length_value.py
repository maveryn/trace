"""Compute cube edge length from a highlighted partial frame length."""

from __future__ import annotations

from typing import Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts

from ._lifecycle import RectangularSolidObjectivePlan, run_rectangular_solid_public_entry
from .shared.construction import resolve_cube_frame_edge
from .shared.defaults import DOMAIN
from .shared.rendering import render_cube_frame_scene
from .shared.state import RenderedRectangularSolidScene


TASK_ID = "task_geometry__rectangular_solid__cube_edge_from_frame_length_value"
QUERY_ID_CUBE_EDGE_PARTIAL_FRAME = "cube_edge_from_partial_frame"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (QUERY_ID_CUBE_EDGE_PARTIAL_FRAME,)
DEFAULT_QUERY_ID = QUERY_ID_CUBE_EDGE_PARTIAL_FRAME
PROMPT_TASK_KEY = "cube_edge_from_frame_length_value"


def _highlighted_frame_bbox_annotation(rendered: RenderedRectangularSolidScene):
    """Return one bbox witness for the highlighted partial frame path."""

    artifacts = bbox_annotation_artifacts(rendered.annotation_keyed_bboxes["given_length_region_bbox"])
    return artifacts.annotation_gt, dict(artifacts.projected_annotation)


def _prepare_frame_edge_objective(
    instance_seed,
    task_params: Mapping[str, object],
    selected_branch,
    branch_probabilities,
):
    """Bind highlighted-frame edge-length solving."""

    problem = resolve_cube_frame_edge(
        instance_seed=int(instance_seed),
        params=task_params,
        frame_mode="partial",
        sampling_label=f"{TASK_ID}.{selected_branch}",
    )
    trace_values = {
        "target_role": "cube_edge",
        "frame_mode": str(problem.frame_mode),
        "cube_edge": int(problem.cube_edge),
        "visible_frame_edge_count": int(problem.visible_frame_edge_count),
        "frame_length": int(problem.frame_length),
    }
    return RectangularSolidObjectivePlan(
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_branch_key=str(selected_branch),
        problem=problem,
        render_scene=render_cube_frame_scene,
        bind_annotation=_highlighted_frame_bbox_annotation,
        answer_gt=TypedValue(type="integer", value=int(problem.answer)),
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "frame_mode": str(problem.frame_mode),
            "case_probabilities": dict(problem.case_probabilities),
            "answer_support_probabilities": dict(problem.answer_support_probabilities),
            **dict(trace_values),
        },
        trace_values=trace_values,
    )


@register_task
class GeometryRectangularSolidCubeEdgeFromFrameLengthValueTask:
    """Compute cube edge length from a highlighted wire-frame length."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_frame_edge_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate one cube-frame edge-length problem."""

        return run_rectangular_solid_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
