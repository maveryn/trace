"""Annotation helpers for navigation-flow page scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .state import MENU_SURFACE, RIBBON_SURFACE, SIDEBAR_SURFACE, NavigationFlowCase, RenderedNavigationFlow


def bbox_list(bbox: tuple[float, float, float, float]) -> List[float]:
    """Return a JSON-stable pixel box."""

    return [round(float(value), 3) for value in bbox]


def annotation_role_names(navigation_surface: str) -> Tuple[str, str, str]:
    """Return role names for the selected surface annotation map."""

    if str(navigation_surface) == MENU_SURFACE:
        return ("menu_root", "menu_group", "target_command")
    if str(navigation_surface) == SIDEBAR_SURFACE:
        return ("sidebar_section", "sidebar_group", "target_item")
    if str(navigation_surface) != RIBBON_SURFACE:
        raise ValueError(f"unsupported navigation surface: {navigation_surface}")
    return ("ribbon_tab", "ribbon_group", "target_command")


def support_ids_for_path(case: NavigationFlowCase) -> Tuple[str, str]:
    """Return visible support ids for the target path."""

    path = tuple(str(value) for value in case.path_labels)
    if str(case.navigation_surface) == MENU_SURFACE:
        menu_values = sorted(
            {str(control.path_keys[0]) for control in case.controls},
            key=lambda value: next(
                control.order_index for control in case.controls if str(control.path_keys[0]) == value
            ),
        )
        menu_index = menu_values.index(path[0])
        group_values = sorted(
            {
                str(control.path_keys[2])
                for control in case.controls
                if str(control.path_keys[0]) == path[0] and str(control.path_keys[1]) == path[1]
            },
            key=lambda value: next(
                control.order_index
                for control in case.controls
                if str(control.path_keys[0]) == path[0]
                and str(control.path_keys[1]) == path[1]
                and str(control.path_keys[2]) == value
            ),
        )
        group_index = group_values.index(path[2])
        submenu_values = sorted(
            {str(control.path_keys[1]) for control in case.controls if str(control.path_keys[0]) == path[0]},
            key=lambda value: next(
                control.order_index
                for control in case.controls
                if str(control.path_keys[0]) == path[0] and str(control.path_keys[1]) == value
            ),
        )
        submenu_index = submenu_values.index(path[1])
        return (
            f"support_menu_{menu_index}",
            f"support_menu_{menu_index}_submenu_{submenu_index}_group_{group_index}",
        )
    if str(case.navigation_surface) == SIDEBAR_SURFACE:
        section_values = sorted(
            {str(control.path_keys[0]) for control in case.controls},
            key=lambda value: next(
                control.order_index for control in case.controls if str(control.path_keys[0]) == value
            ),
        )
        section_index = section_values.index(path[0])
        group_values = sorted(
            {
                str(control.path_keys[1])
                for control in case.controls
                if str(control.path_keys[0]) == path[0]
            },
            key=lambda value: next(
                control.order_index
                for control in case.controls
                if str(control.path_keys[0]) == path[0] and str(control.path_keys[1]) == value
            ),
        )
        group_index = group_values.index(path[1])
        return (
            f"support_sidebar_section_{section_index}",
            f"support_sidebar_section_{section_index}_group_{group_index}",
        )
    tab_values = sorted(
        {str(control.path_keys[0]) for control in case.controls},
        key=lambda value: next(control.order_index for control in case.controls if str(control.path_keys[0]) == value),
    )
    tab_index = tab_values.index(path[0])
    group_values = sorted(
        {str(control.path_keys[1]) for control in case.controls if str(control.path_keys[0]) == path[0]},
        key=lambda value: next(
            control.order_index
            for control in case.controls
            if str(control.path_keys[0]) == path[0] and str(control.path_keys[1]) == value
        ),
    )
    group_index = group_values.index(path[1])
    return (f"support_ribbon_tab_{tab_index}", f"support_ribbon_tab_{tab_index}_group_{group_index}")


def annotation_bbox_map(case: NavigationFlowCase, rendered: RenderedNavigationFlow) -> Dict[str, List[float]]:
    """Build the keyed annotation map from finalized render geometry."""

    support_ids = support_ids_for_path(case)
    first_role, second_role, target_role = annotation_role_names(str(case.navigation_surface))
    target_box = list(rendered.control_bboxes_by_id[str(case.target_control_id)])
    return {
        str(first_role): list(rendered.support_bboxes_by_id[str(support_ids[0])]),
        str(second_role): list(rendered.support_bboxes_by_id[str(support_ids[1])]),
        str(target_role): list(target_box),
    }


def target_annotation_bbox(case: NavigationFlowCase, rendered: RenderedNavigationFlow) -> List[float]:
    """Return the public annotation bbox for the selected target control."""

    return list(rendered.control_bboxes_by_id[str(case.target_control_id)])


def annotation_role_support_ids(case: NavigationFlowCase) -> Dict[str, str]:
    """Map annotation roles to support or control identifiers."""

    support_ids = support_ids_for_path(case)
    first_role, second_role, target_role = annotation_role_names(str(case.navigation_surface))
    return {
        str(first_role): str(support_ids[0]),
        str(second_role): str(support_ids[1]),
        str(target_role): str(case.target_control_id),
    }


def control_entities(rendered: RenderedNavigationFlow) -> List[Dict[str, Any]]:
    """Return scene entities for visible candidate controls."""

    return [
        {
            "entity_id": str(record["control_id"]),
            "entity_type": "gui_control",
            "attrs": {
                "candidate_label": str(record["candidate_label"]),
                "role": str(record["role"]),
                "display_text": str(record["display_text"]),
                "nav_kind": str(record["nav_kind"]),
                "path_keys": [str(value) for value in record["path_keys"]],
                "bbox_px": [float(value) for value in record["bbox_px"]],
            },
        }
        for record in rendered.control_records
    ]
