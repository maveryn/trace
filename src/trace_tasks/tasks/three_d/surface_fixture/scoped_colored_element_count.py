"""Count colored fixture elements inside one row or column."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import explicit_query_id_param

from ._lifecycle import ResolvedSurfaceFixtureAxes, SurfaceFixturePlan, run_surface_fixture_lifecycle
from .shared.metrics import build_scoped_color_surface_data
from .shared.state import COLOR_READOUT_SCENE_VARIANTS, SCENE_ID


TASK_ID = "task_three_d__surface_fixture__scoped_colored_element_count"
ROW_QUERY_ID = "row_scoped_color_count"
COLUMN_QUERY_ID = "column_scoped_color_count"
DEFAULT_QUERY_ID = ROW_QUERY_ID
SCOPE_AXIS_BY_QUERY_ID = {
    ROW_QUERY_ID: "row",
    COLUMN_QUERY_ID: "column",
}
PROMPT_QUERY_KEY = "scoped_element_color_count"
SUPPORTED_QUERY_IDS = (ROW_QUERY_ID, COLUMN_QUERY_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "three_d",
    SCENE_ID,
    task_id=TASK_ID,
)


def _params_with_scope_query(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize legacy/debug `scope_axis` params into the public query id."""

    normalized = dict(params)
    if explicit_query_id_param(normalized, allow_default=True) is not None:
        return normalized
    scope_axis = normalized.get("scope_axis")
    if scope_axis is None:
        return normalized
    axis_text = str(scope_axis)
    query_by_axis = {str(axis): str(query_id) for query_id, axis in SCOPE_AXIS_BY_QUERY_ID.items()}
    if axis_text not in query_by_axis:
        raise ValueError(f"unsupported scope_axis for {TASK_ID}: {axis_text}")
    normalized["query_id"] = str(query_by_axis[axis_text])
    return normalized


def _prepare_scoped_color_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    axes: ResolvedSurfaceFixtureAxes,
    _branch_probabilities: Mapping[str, float],
    selected_branch: str,
) -> SurfaceFixturePlan:
    """Bind the row-or-column scoped color-count objective."""

    scope_axis = str(SCOPE_AXIS_BY_QUERY_ID[str(selected_branch)])
    scoped_params = dict(params)
    explicit_scope_axis = scoped_params.get("scope_axis")
    if explicit_scope_axis is not None and str(explicit_scope_axis) != scope_axis:
        raise ValueError(
            f"scope_axis={explicit_scope_axis!r} conflicts with query_id={selected_branch!r}"
        )
    scoped_params["scope_axis"] = str(scope_axis)

    dataset, answer_probabilities = build_scoped_color_surface_data(
        namespace=f"{TASK_ID}.objective",
        scene_variant=axes.scene_variant,
        element_type=axes.element_type,
        instance_seed=int(instance_seed),
        params=scoped_params,
        gen_defaults=_GEN_DEFAULTS,
    )
    target_color = str(dataset["target_color_name"])
    return SurfaceFixturePlan(
        dataset=dataset,
        answer_gt=TypedValue(type="integer", value=int(dataset["answer_value"])),
        target_element_ids=tuple(str(element_id) for element_id in dataset["target_element_ids"]),
        answer_value_probabilities=dict(answer_probabilities),
        object_description="a fixture surface arranged in rows and columns with repeated colored surface elements",
        objective_params={
            "scope_axis": str(dataset["scope_axis"]),
            "scope_index": int(dataset["scope_index"]),
            "scope_phrase": str(dataset["scope_phrase"]),
            "target_color_name": target_color,
        },
        execution_extra={
            "scope_axis": str(dataset["scope_axis"]),
            "scope_index": int(dataset["scope_index"]),
            "scope_phrase": str(dataset["scope_phrase"]),
            "target_color_name": target_color,
        },
    )


@register_task
class ThreeDSurfaceFixtureScopedColoredElementCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "three_d"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_surface_fixture_lifecycle(
            public_name=TASK_ID,
            domain_name=self.domain,
            prompt_query_key=PROMPT_QUERY_KEY,
            supported_branches=SUPPORTED_QUERY_IDS,
            default_branch=DEFAULT_QUERY_ID,
            supported_scenes=COLOR_READOUT_SCENE_VARIANTS,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=_params_with_scope_query(params),
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_scoped_color_objective,
        )


__all__ = [
    "COLUMN_QUERY_ID",
    "DEFAULT_QUERY_ID",
    "ROW_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "ThreeDSurfaceFixtureScopedColoredElementCountTask",
]
