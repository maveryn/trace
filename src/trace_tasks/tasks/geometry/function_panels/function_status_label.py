"""Select the panel that represents a function."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions

from ._lifecycle import build_property_components, property_defaults, property_plan
from .shared.state import RULE_FUNCTION_TEST, SCENE_ID


TASK_ID = "task_geometry__function_panels__function_status_label"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_DEFAULTS = property_defaults(TASK_ID)


def _prepare_function_status(_branch_name: str, _params: Mapping[str, Any]):
    """Bind the function-test objective and prompt key."""

    return property_plan(
        prompt_key="function_status_label",
        rule_kind=RULE_FUNCTION_TEST,
        sign_kind=None,
        defaults=_DEFAULTS,
    )


@register_task
class GeometryFunctionPanelsFunctionStatusLabelTask:
    """Choose the only panel that passes the vertical-line test."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'matching')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        del max_attempts
        branch_name, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
        )
        components = build_property_components(
            instance_seed=int(instance_seed),
            params=task_params,
            branch_name=str(branch_name),
            branch_probabilities=branch_probabilities,
            namespace=TASK_ID,
            plan=_prepare_function_status(str(branch_name), task_params),
        )
        return TaskOutput(
            prompt=str(components.prompt),
            answer_gt=TypedValue(type="option_letter", value=str(components.trace_payload["execution_trace"]["answer_label"])),
            annotation_gt=components.annotation.annotation_gt,
            image=components.image,
            image_id="img_0",
            trace_payload=dict(components.trace_payload),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(branch_name),
            prompt_variants=dict(components.prompt_variants),
        )


__all__ = ["GeometryFunctionPanelsFunctionStatusLabelTask"]
