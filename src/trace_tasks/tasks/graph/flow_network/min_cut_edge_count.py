"""Count edges in the unique minimum cut of a capacity network."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ._lifecycle import FlowNetworkObjectivePlan, run_flow_network_plan
from .shared.prompts import PROMPT_BUNDLE_ID as FLOW_PROMPT_BUNDLE_ID
from .shared.sampling import integer_probability_map, resolve_flow_network_axes, resolve_integer_axis
from .shared.state import FlowNetworkDefaults, SCENE_ID


TASK_ID = "task_graph__flow_network__min_cut_edge_count"
QUERY_ID = "single"
PROMPT_QUERY_KEY = "minimum_cut_edge_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
SAMPLING_NAMESPACE = "graph.flow_network.min_cut_edge_count"

_DEFAULTS = FlowNetworkDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "graph",
    SCENE_ID,
    task_id=TASK_ID,
)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, "bundle_id", FLOW_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


@register_task
class GraphFlowNetworkMinCutEdgeCountTask:
    """Answer minimum-cut edge-count questions on a directed capacity graph."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one min-cut count instance through neutral scene lifecycle plumbing."""

        return run_flow_network_plan(
            plan=_build_objective_plan(),
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


def _resolve_objective_axes(
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[Any, int, Dict[str, float]]:
    """Resolve min-cut edge-count answer support before scene sampling."""

    cut_count_min = int(
        params.get(
            "min_cut_edge_count_min",
            group_default(
                _GEN_DEFAULTS,
                "min_cut_edge_count_min",
                int(_DEFAULTS.min_cut_edge_count_min),
            ),
        )
    )
    cut_count_max = int(
        params.get(
            "min_cut_edge_count_max",
            group_default(
                _GEN_DEFAULTS,
                "min_cut_edge_count_max",
                int(_DEFAULTS.min_cut_edge_count_max),
            ),
        )
    )
    cut_count_support = tuple(range(max(1, int(cut_count_min)), int(cut_count_max) + 1))
    target_answer, target_answer_probabilities = resolve_integer_axis(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.target_answer",
        support=cut_count_support,
        explicit_key="target_answer",
    )
    feasible_flow_support = tuple(
        range(
            max(4, int(target_answer)),
            min(12, int(_DEFAULTS.cut_capacity_part_max) * int(target_answer)) + 1,
        )
    )
    target_flow_value, target_flow_value_probabilities = resolve_integer_axis(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.target_flow_value.{target_answer}",
        support=feasible_flow_support,
        explicit_key="target_flow_value",
    )
    distractor_min = int(
        params.get(
            "distractor_edge_min",
            group_default(_GEN_DEFAULTS, "distractor_edge_min", int(_DEFAULTS.distractor_edge_min)),
        )
    )
    distractor_max = int(
        params.get(
            "distractor_edge_max",
            group_default(_GEN_DEFAULTS, "distractor_edge_max", int(_DEFAULTS.distractor_edge_max)),
        )
    )
    distractor_support = tuple(range(max(0, int(distractor_min)), max(int(distractor_min), int(distractor_max)) + 1))
    axes = resolve_flow_network_axes(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=SAMPLING_NAMESPACE,
        target_cut_edge_count=int(target_answer),
        target_cut_edge_count_probabilities=integer_probability_map((int(target_answer),), selected=int(target_answer)),
        target_flow_value=int(target_flow_value),
        target_flow_value_probabilities=target_flow_value_probabilities,
        distractor_support=distractor_support,
        defaults=_DEFAULTS,
    )
    return axes, int(target_answer), dict(target_answer_probabilities)


def _build_objective_plan() -> FlowNetworkObjectivePlan:
    """Bind min-cut count answer support and prompt key."""

    return FlowNetworkObjectivePlan(
        owner_id=TASK_ID,
        supported_branch_names=SUPPORTED_QUERY_IDS,
        default_branch_name=QUERY_ID,
        prompt_query_key=PROMPT_QUERY_KEY,
        sampling_namespace=SAMPLING_NAMESPACE,
        resolve_objective_axes=_resolve_objective_axes,
    )


__all__ = ["GraphFlowNetworkMinCutEdgeCountTask", "TASK_ID"]
