"""Compute the maximum flow value in a directed capacity network."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ._lifecycle import FlowNetworkObjectivePlan, run_flow_network_plan
from .shared.prompts import PROMPT_BUNDLE_ID as FLOW_PROMPT_BUNDLE_ID
from .shared.sampling import integer_probability_map, resolve_flow_network_axes, resolve_integer_axis
from .shared.state import FlowNetworkDefaults, FlowNetworkSceneBundle, SCENE_ID


TASK_ID = "task_graph__flow_network__max_flow_value"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "max_flow_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
SAMPLING_NAMESPACE = "graph.flow_network.max_flow_value"

_DEFAULTS = FlowNetworkDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "graph",
    SCENE_ID,
    task_id=TASK_ID,
)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, "bundle_id", FLOW_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _resolve_objective_axes(
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[Any, int, Dict[str, float]]:
    """Resolve max-flow answer support before neutral scene sampling."""

    flow_min = int(
        params.get(
            "max_flow_value_min",
            group_default(_GEN_DEFAULTS, "max_flow_value_min", int(_DEFAULTS.max_flow_value_min)),
        )
    )
    flow_max = int(
        params.get(
            "max_flow_value_max",
            group_default(_GEN_DEFAULTS, "max_flow_value_max", int(_DEFAULTS.max_flow_value_max)),
        )
    )
    flow_support = tuple(range(max(1, int(flow_min)), int(flow_max) + 1))
    target_answer, target_answer_probabilities = resolve_integer_axis(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.target_answer",
        support=flow_support,
        explicit_key="target_answer",
    )
    cut_min = int(
        params.get(
            "max_flow_cut_edge_count_min",
            group_default(
                _GEN_DEFAULTS,
                "max_flow_cut_edge_count_min",
                int(_DEFAULTS.max_flow_cut_edge_count_min),
            ),
        )
    )
    cut_max = int(
        params.get(
            "max_flow_cut_edge_count_max",
            group_default(
                _GEN_DEFAULTS,
                "max_flow_cut_edge_count_max",
                int(_DEFAULTS.max_flow_cut_edge_count_max),
            ),
        )
    )
    cut_support = tuple(range(max(1, int(cut_min)), max(int(cut_min), int(cut_max)) + 1))
    feasible_cut_support = tuple(
        int(value)
        for value in cut_support
        if int(value) <= int(target_answer)
        and int(target_answer) <= int(value) * int(_DEFAULTS.cut_capacity_part_max)
    )
    target_cut_edge_count, target_cut_edge_count_probabilities = resolve_integer_axis(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.target_cut_edge_count.{target_answer}",
        support=feasible_cut_support,
        explicit_key="target_cut_edge_count",
    )
    distractor_min = int(
        params.get(
            "max_flow_distractor_edge_min",
            group_default(
                _GEN_DEFAULTS,
                "max_flow_distractor_edge_min",
                int(_DEFAULTS.max_flow_distractor_edge_min),
            ),
        )
    )
    distractor_max = int(
        params.get(
            "max_flow_distractor_edge_max",
            group_default(
                _GEN_DEFAULTS,
                "max_flow_distractor_edge_max",
                int(_DEFAULTS.max_flow_distractor_edge_max),
            ),
        )
    )
    distractor_support = tuple(range(max(0, int(distractor_min)), max(int(distractor_min), int(distractor_max)) + 1))
    axes = resolve_flow_network_axes(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=SAMPLING_NAMESPACE,
        target_cut_edge_count=int(target_cut_edge_count),
        target_cut_edge_count_probabilities=target_cut_edge_count_probabilities,
        target_flow_value=int(target_answer),
        target_flow_value_probabilities=integer_probability_map((int(target_answer),), selected=int(target_answer)),
        distractor_support=distractor_support,
        defaults=_DEFAULTS,
    )
    return axes, int(target_answer), dict(target_answer_probabilities)


def _extra_trace_fields(bundle: FlowNetworkSceneBundle) -> Mapping[str, int]:
    """Expose the conventional max-flow trace alias from the public objective."""

    return {"max_flow_value": int(bundle.flow_sample.original_max_flow_value)}


def _build_objective_plan() -> FlowNetworkObjectivePlan:
    """Bind max-flow answer support, prompt key, and trace aliases."""

    return FlowNetworkObjectivePlan(
        owner_id=TASK_ID,
        supported_branch_names=SUPPORTED_QUERY_IDS,
        default_branch_name=QUERY_ID,
        prompt_query_key=PROMPT_QUERY_KEY,
        sampling_namespace=SAMPLING_NAMESPACE,
        resolve_objective_axes=_resolve_objective_axes,
        extra_trace_fields=_extra_trace_fields,
    )


@register_task
class GraphFlowNetworkMaxFlowValueTask:
    """Answer maximum-flow value questions on a directed capacity graph."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'aggregation', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> FlowNetworkObjectivePlan:
        """Return this task's local objective plan."""

        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Resolve the max-flow objective locally, then render and bind the final task output."""

        return run_flow_network_plan(
            plan=self._build_objective_plan(),
            domain=self.domain,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_bundle_id=PROMPT_BUNDLE_ID,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
            noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
            defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            params=dict(params),
        )


__all__ = ["GraphFlowNetworkMaxFlowValueTask", "TASK_ID"]
