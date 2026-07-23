"""Navigation-flow task for selecting a control in the same group as a reference."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import DOMAIN, NAMESPACE_ROOT, SCENE_VARIANTS
from .shared.rendering import render_navigation_flow_case
from .shared.sampling import build_navigation_flow_case, resolve_navigation_surface
from .shared.state import ControlSpec, MENU_SURFACE, RIBBON_SURFACE, SIDEBAR_SURFACE


TASK_ID = "task_pages__navigation_flow__same_group_target_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "same_group_target_label"


def _group_key_for_control(control: ControlSpec) -> Tuple[str, ...]:
    """Return the visible group key that defines same-group membership."""

    if str(control.nav_kind) == MENU_SURFACE:
        return tuple(str(value) for value in control.path_keys[:3])
    if str(control.nav_kind) in {SIDEBAR_SURFACE, RIBBON_SURFACE}:
        return tuple(str(value) for value in control.path_keys[:2])
    raise ValueError(f"unsupported navigation surface: {control.nav_kind}")


def _select_same_group_pair(
    *,
    controls: Tuple[ControlSpec, ...],
    instance_seed: int,
) -> tuple[ControlSpec, ControlSpec, Tuple[str, ...], Tuple[str, ...]]:
    """Select a reference control and the unique other control in its visible group."""

    by_group: Dict[Tuple[str, ...], list[ControlSpec]] = {}
    for control in controls:
        by_group.setdefault(_group_key_for_control(control), []).append(control)
    groups = [
        tuple(group_key)
        for group_key, group_controls in sorted(by_group.items())
        if len(group_controls) >= 2
    ]
    if not groups:
        raise ValueError("same_group_target_label requires at least one group with two controls")

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.same_group_target_label.pair")
    group_key = groups[int(rng.randrange(len(groups)))]
    group_controls = sorted(by_group[group_key], key=lambda item: int(item.order_index))
    reference = group_controls[int(rng.randrange(len(group_controls)))]
    answer_candidates = [control for control in group_controls if str(control.control_id) != str(reference.control_id)]
    target = answer_candidates[int(rng.randrange(len(answer_candidates)))]
    return reference, target, tuple(group_key), tuple(str(control.control_id) for control in group_controls)


def _forced_pair_params(params: Mapping[str, object]) -> Dict[str, object]:
    """Force two controls per visible group so the same-group target is unique."""

    task_params = dict(params)
    task_params["menu_command_count"] = 2
    task_params["ribbon_command_count"] = 2
    return task_params


def _add_same_group_trace(
    *,
    output,
    reference: ControlSpec,
    target: ControlSpec,
    group_key: Tuple[str, ...],
    group_control_ids: Tuple[str, ...],
) -> None:
    """Record reference-control details without changing the public annotation."""

    trace = output.trace_payload
    render_map = trace.get("render_map", {})
    control_bboxes = render_map.get("control_bboxes_by_id", {})
    badge_bboxes = render_map.get("candidate_label_badge_bboxes_by_id", {})
    reference_box = list(control_bboxes[str(reference.control_id)])
    reference_badge_box = list(badge_bboxes[str(reference.control_id)])
    target_box = list(control_bboxes[str(target.control_id)])
    extra = {
        "reference_control_id": str(reference.control_id),
        "reference_label": str(reference.candidate_label),
        "reference_control_bbox_px": list(reference_box),
        "reference_candidate_label_bbox_px": list(reference_badge_box),
        "same_group_key": [str(value) for value in group_key],
        "same_group_control_ids": [str(value) for value in group_control_ids],
        "same_group_answer_control_id": str(target.control_id),
        "same_group_answer_bbox_px": list(target_box),
        "same_group_controls_per_group": 2,
    }
    for path in (
        ("query_spec", "params"),
        ("scene_ir", "relations"),
        ("render_map",),
        ("execution_trace",),
        ("witness_symbolic",),
    ):
        cursor = trace
        for key in path:
            cursor = cursor.setdefault(str(key), {})
        cursor.update(extra)

@register_task
class PagesNavigationFlowSameGroupTargetLabelTask:
    """Identify the other lettered control in the same group as a named reference letter."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, object], max_attempts: int):
        """Sample a two-control group, name one unique letter, and ask for its group mate."""

        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.choose_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        task_params = _forced_pair_params(task_params)
        navigation_surface, surface_probabilities = resolve_navigation_surface(
            params=task_params,
            instance_seed=int(instance_seed),
            namespace="same_group_target.surface",
        )
        case = build_navigation_flow_case(
            instance_seed=int(instance_seed),
            params=task_params,
            navigation_surface=str(navigation_surface),
            namespace="same_group_target",
        )
        reference, target, group_key, group_control_ids = _select_same_group_pair(
            controls=tuple(case.controls),
            instance_seed=int(instance_seed),
        )
        case = replace(
            case,
            target_control_id=str(target.control_id),
            target_label=str(target.candidate_label),
            path_labels=tuple(str(value) for value in target.path_keys),
            path_display=" > ".join(str(value) for value in target.path_keys),
            command_label=str(target.path_keys[-1]),
            surface_probabilities=dict(surface_probabilities),
        )
        rendered = render_navigation_flow_case(
            instance_seed=int(instance_seed),
            params=task_params,
            case=case,
            namespace="same_group_target",
        )
        prompt_binding = _lifecycle.NavigationPromptBinding(
            prompt_branch_key=PROMPT_QUERY_KEY,
            dynamic_slots={
                "reference_label": str(reference.candidate_label),
            },
        )
        answer_binding = _lifecycle.option_letter_binding(
            annotation_value=list(rendered.control_bboxes_by_id[str(target.control_id)]),
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            answer_value=str(target.candidate_label),
            prompt_branch_key=PROMPT_QUERY_KEY,
            question_format="navigation_flow_same_group_target_lookup",
        )
        output = _lifecycle.build_navigation_flow_response(
            instance_seed=int(instance_seed),
            public_task_id=TASK_ID,
            case=case,
            rendered=rendered,
            prompt_binding=prompt_binding,
            answer_binding=answer_binding,
        )
        _add_same_group_trace(
            output=output,
            reference=reference,
            target=target,
            group_key=group_key,
            group_control_ids=group_control_ids,
        )
        return output


__all__ = [
    "PROMPT_QUERY_KEY",
    "SCENE_VARIANTS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesNavigationFlowSameGroupTargetLabelTask",
]
