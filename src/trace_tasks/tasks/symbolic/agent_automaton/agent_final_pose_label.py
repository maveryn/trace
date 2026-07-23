"""Choose the final pose of a symbolic turning-agent automaton."""

from __future__ import annotations

from ...registry import register_task
from ..shared.common import get_int_param as _get_int

from ._lifecycle import AgentOptionDataset, bbox_map_option_result, make_agent_option_binding, pose_option_records, pose_record, run_bound_agent_option_lifecycle
from .shared.rules import SCENE_ID
from .shared.sampling import choose_pose_options
from .shared.state import AgentRenderBundle, AgentSimulationSample


TASK_ID = "task_symbolic__agent_automaton__agent_final_pose_label"
SUPPORTED_QUERY_IDS = ("single",)


def _build_final_pose_options(
    *,
    run: AgentSimulationSample,
    params,
    gen_defaults,
    instance_seed,
    rule_variant,
):
    """Bind the simulated terminal pose to one of the visual options."""

    import random

    option_specs, answer_label = choose_pose_options(
        rng=random.Random(int(instance_seed) ^ 0xF17A1),
        params=dict(params),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.answer_option",
        rows=int(run.rows),
        cols=int(run.cols),
        final_row=int(run.final_row),
        final_col=int(run.final_col),
        final_direction=int(run.final_direction),
        option_count=_get_int(params, gen_defaults, "pose_option_count", 4),
    )
    return tuple(option_specs), str(answer_label)


def _build_final_pose_objective(dataset: AgentOptionDataset, render_bundle: AgentRenderBundle):
    """Bind the answer option and minimal role-keyed annotation."""

    correct_option = next(option for option in dataset.option_specs if bool(option.is_correct))
    role_item_ids = {"start_marker": "initial_agent", "selected_option": str(correct_option.option_id)}
    return bbox_map_option_result(
        answer_label=str(dataset.answer_label),
        item_bboxes=render_bundle.rendered.item_bboxes,
        annotation_role_item_ids=role_item_ids,
        execution_fields={
            **pose_record(dataset, use_direction_names=True),
            "option_specs": pose_option_records(dataset.option_specs),
        },
        render_map_extra={"selected_option_card_bbox_px": list(render_bundle.rendered.item_bboxes[str(correct_option.option_id)])},
    )


_BINDING = make_agent_option_binding(
    TASK_ID,
    "agent_final_pose_label",
    "agent_final_pose_query",
    "agent_final_steps_min",
    "agent_final_steps_max",
    "pose",
    _build_final_pose_options,
    _build_final_pose_objective,
)


@register_task
class SymbolicAutomatonAgentFinalPoseLabelTask:
    """Choose the final agent pose after simulating a turning automaton."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "symbolic"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one final-pose option task with local answer binding."""

        return run_bound_agent_option_lifecycle(
            _BINDING,
            params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
