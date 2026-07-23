"""Navigation-flow task for identifying a target reached by a navigation path."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import DOMAIN, SCENE_VARIANTS
from .shared.rendering import render_navigation_flow_case
from .shared.sampling import build_navigation_flow_case
from .shared.state import MENU_SURFACE, RIBBON_SURFACE, SIDEBAR_SURFACE


TASK_ID = "task_pages__navigation_flow__navigation_path_target_label"
MENU_PATH_TARGET_QUERY_ID = "menu_path_target_label"
SIDEBAR_TREE_TARGET_QUERY_ID = "sidebar_tree_target_label"
RIBBON_GROUP_COMMAND_QUERY_ID = "ribbon_group_command_label"
SUPPORTED_QUERY_IDS = (
    MENU_PATH_TARGET_QUERY_ID,
    SIDEBAR_TREE_TARGET_QUERY_ID,
    RIBBON_GROUP_COMMAND_QUERY_ID,
)

_SURFACE_BY_QUERY_ID = {
    MENU_PATH_TARGET_QUERY_ID: MENU_SURFACE,
    SIDEBAR_TREE_TARGET_QUERY_ID: SIDEBAR_SURFACE,
    RIBBON_GROUP_COMMAND_QUERY_ID: RIBBON_SURFACE,
}
_QUERY_ID_BY_SURFACE = {str(surface): str(query_id) for query_id, surface in _SURFACE_BY_QUERY_ID.items()}

_QUESTION_FORMATS_BY_QUERY_ID = {
    MENU_PATH_TARGET_QUERY_ID: "navigation_flow_menu_path_target_lookup",
    SIDEBAR_TREE_TARGET_QUERY_ID: "navigation_flow_sidebar_tree_target_lookup",
    RIBBON_GROUP_COMMAND_QUERY_ID: "navigation_flow_ribbon_group_command_lookup",
}


@register_task
class PagesNavigationFlowNavigationPathTargetLabelTask:
    """Identify the candidate reached by a visible navigation path."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, object], max_attempts: int):
        """Bind one semantic navigation-surface branch to the path-target lookup contract."""

        del max_attempts
        requested_params = dict(params)
        if "navigation_surface" in requested_params and "query_id" not in requested_params and "query_variant" not in requested_params:
            requested_surface = str(requested_params["navigation_surface"])
            if requested_surface not in _QUERY_ID_BY_SURFACE:
                raise ValueError(f"unsupported navigation_surface for {TASK_ID}: {requested_surface}")
            requested_params["query_id"] = _QUERY_ID_BY_SURFACE[requested_surface]
        selected_branch, branch_probabilities, task_params = _lifecycle.choose_public_branch(
            instance_seed=int(instance_seed),
            params=requested_params,
            supported=SUPPORTED_QUERY_IDS,
            default=MENU_PATH_TARGET_QUERY_ID,
            public_task=TASK_ID,
        )
        selected_query_id = str(selected_branch)
        navigation_surface = str(_SURFACE_BY_QUERY_ID[selected_query_id])
        if "navigation_surface" in task_params and str(task_params["navigation_surface"]) != navigation_surface:
            raise ValueError("navigation_surface must match query_id for navigation_path_target_label")
        task_params = dict(task_params)
        task_params["navigation_surface"] = navigation_surface
        surface_probabilities = {
            MENU_SURFACE: 1.0 if navigation_surface == MENU_SURFACE else 0.0,
            SIDEBAR_SURFACE: 1.0 if navigation_surface == SIDEBAR_SURFACE else 0.0,
            RIBBON_SURFACE: 1.0 if navigation_surface == RIBBON_SURFACE else 0.0,
        }
        case = build_navigation_flow_case(
            instance_seed=int(instance_seed),
            params=task_params,
            navigation_surface=navigation_surface,
            namespace="navigation_path",
        )
        case = replace(case, surface_probabilities=dict(surface_probabilities))
        rendered = render_navigation_flow_case(
            instance_seed=int(instance_seed),
            params=task_params,
            case=case,
            namespace="navigation_path",
        )
        prompt_binding, answer_binding = _lifecycle.bind_navigation_answer(
            case=case,
            rendered=rendered,
            selected_branch=selected_query_id,
            branch_probabilities=branch_probabilities,
            prompt_branch_key=selected_query_id,
            question_format=_QUESTION_FORMATS_BY_QUERY_ID[selected_query_id],
        )
        return _lifecycle.build_navigation_flow_response(
            instance_seed=int(instance_seed),
            public_task_id=TASK_ID,
            case=case,
            rendered=rendered,
            prompt_binding=prompt_binding,
            answer_binding=answer_binding,
        )


__all__ = [
    "MENU_PATH_TARGET_QUERY_ID",
    "RIBBON_GROUP_COMMAND_QUERY_ID",
    "SCENE_VARIANTS",
    "SIDEBAR_TREE_TARGET_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesNavigationFlowNavigationPathTargetLabelTask",
]
