"""Compute an open-box base dimension from a corner-cut net."""

from __future__ import annotations

from typing import Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import segment_annotation_artifacts

from ._lifecycle import RectangularSolidObjectivePlan, run_rectangular_solid_public_entry
from .shared.construction import resolve_open_box_dimension
from .shared.defaults import DOMAIN
from .shared.rendering import render_open_box_net_scene
from .shared.state import RenderedRectangularSolidScene


TASK_ID = "task_geometry__rectangular_solid__open_box_net_dimension_value"
QUERY_ID_OPEN_BOX_DIMENSION = "open_box_dimension_from_corner_cut"
SUPPORTED_QUERY_IDS: tuple[str, ...] = (QUERY_ID_OPEN_BOX_DIMENSION,)
DEFAULT_QUERY_ID = QUERY_ID_OPEN_BOX_DIMENSION
PROMPT_TASK_KEY = "open_box_net_dimension_value"


def _target_dimension_segment_annotation(rendered: RenderedRectangularSolidScene):
    """Return one segment witness for the marked base dimension."""

    artifacts = segment_annotation_artifacts(rendered.render_map["target_dimension_segment"])
    return artifacts.annotation_gt, dict(artifacts.projected_annotation)


def _open_box_trace_fields(problem) -> dict[str, int | str]:
    """Return task-specific numeric fields for the corner-cut net."""

    return {
        "target_role": str(problem.target_role),
        "sheet_length": int(problem.sheet_length),
        "sheet_width": int(problem.sheet_width),
        "cut_size": int(problem.cut_size),
        "base_length": int(problem.base_length),
        "base_width": int(problem.base_width),
        "height": int(problem.cut_size),
        "open_box_volume": int(problem.open_box_volume),
    }


def _prepare_open_box_objective(
    instance_seed,
    task_params: Mapping[str, object],
    selected_branch,
    branch_probabilities,
):
    """Bind resulting base-dimension solving for the net."""

    net_problem = resolve_open_box_dimension(
        instance_seed=int(instance_seed),
        params=task_params,
        sampling_label=f"{TASK_ID}.{QUERY_ID_OPEN_BOX_DIMENSION}",
    )
    trace_values = _open_box_trace_fields(net_problem)
    return RectangularSolidObjectivePlan(
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_branch_key=QUERY_ID_OPEN_BOX_DIMENSION,
        problem=net_problem,
        render_scene=render_open_box_net_scene,
        bind_annotation=_target_dimension_segment_annotation,
        answer_gt=TypedValue(type="integer", value=int(net_problem.answer)),
        query_params={
            "query_id_probabilities": dict(branch_probabilities),
            "open_box_case_probabilities": dict(net_problem.case_probabilities),
            "answer_support_probabilities": dict(net_problem.answer_support_probabilities),
            **trace_values,
        },
        trace_values=trace_values,
    )


@register_task
class GeometryRectangularSolidOpenBoxNetDimensionValueTask:
    """Compute a resulting base dimension from an open-box net."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = DEFAULT_QUERY_ID
    prepare_objective = staticmethod(_prepare_open_box_objective)

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        """Generate one open-box net measurement problem."""

        return run_rectangular_solid_public_entry(
            self,
            int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )
