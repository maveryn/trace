"""Choose the future grid of a symbolic turning-agent automaton."""

from __future__ import annotations

from ....core.seed import spawn_rng
from ...registry import register_task
from ..shared.common import get_int_param as _get_int

from ._lifecycle import AgentOptionDataset, bbox_map_option_result, grid_option_records, make_agent_option_binding, pose_record, run_bound_agent_option_lifecycle, step_trace_records
from .shared.rules import SCENE_ID, state_count_for_rule
from .shared.sampling import choose_grid_options
from .shared.state import AgentRenderBundle, AgentSimulationSample


TASK_ID = "task_symbolic__agent_automaton__future_grid_label"
SUPPORTED_QUERY_IDS = ("single",)


def _build_future_grid_options(
    *,
    run: AgentSimulationSample,
    params,
    gen_defaults,
    instance_seed,
    rule_variant,
):
    """Bind the simulated final grid to one of the visual options."""

    option_specs, answer_label = choose_grid_options(
        rng=spawn_rng(int(instance_seed), f"{TASK_ID}.options"),
        params=dict(params),
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.answer_option",
        final_grid=run.final_grid,
        state_count=state_count_for_rule(str(rule_variant)),
        option_count=_get_int(params, gen_defaults, "grid_option_count", 4),
    )
    return tuple(option_specs), str(answer_label)


def _build_future_grid_objective(dataset: AgentOptionDataset, render_bundle: AgentRenderBundle):
    """Bind the selected future-grid option and its source grid witness."""

    correct_option = next(option for option in dataset.option_specs if bool(option.is_correct))
    role_item_ids = {"source_grid": "source_grid", "selected_option": str(correct_option.option_id)}
    return bbox_map_option_result(
        answer_label=str(dataset.answer_label),
        item_bboxes=render_bundle.rendered.item_bboxes,
        annotation_role_item_ids=role_item_ids,
        execution_fields={
            **pose_record(dataset, use_direction_names=False),
            "path_trace": step_trace_records(dataset.traces),
            "option_specs": grid_option_records(dataset.option_specs),
        },
        render_map_extra={"annotation_role_item_ids": dict(role_item_ids)},
    )


_BINDING = make_agent_option_binding(
    TASK_ID,
    "agent_future_grid_label",
    "agent_future_grid_query",
    "agent_future_steps_min",
    "agent_future_steps_max",
    "grid",
    _build_future_grid_options,
    _build_future_grid_objective,
    prompt_key_suffix="agent_future_grid_label",
)


@register_task
class SymbolicAutomatonAgentFutureGridLabelTask:
    """Choose the option showing the full grid after simulating the agent."""

    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = "symbolic"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate one future-grid option task with local answer binding."""

        return run_bound_agent_option_lifecycle(
            _BINDING,
            params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
