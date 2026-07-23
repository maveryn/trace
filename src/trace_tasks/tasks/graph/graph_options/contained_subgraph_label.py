"""Select the option graph contained in the larger target graph."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ._lifecycle import GraphOptionsObjectivePlan, run_graph_options_plan
from .shared.prompts import PROMPT_BUNDLE_ID as GRAPH_OPTIONS_PROMPT_BUNDLE_ID
from .shared.sampling import build_contained_subgraph_dataset
from .shared.state import GraphOptionsDefaults, SCENE_ID


TASK_ID = "task_graph__graph_options__contained_subgraph_label"
QUERY_ID = "single"
PROMPT_KEY = "contained_subgraph_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
SAMPLING_NAMESPACE = "graph.graph_options.contained_subgraph_label"

_DEFAULTS = GraphOptionsDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "graph",
    SCENE_ID,
    task_id=TASK_ID,
)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, "bundle_id", GRAPH_OPTIONS_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _object_description(edge_mode: str) -> str:
    """Return objective-specific scene wording from prompt defaults."""

    if str(edge_mode) == "directed":
        return "a Target Graph above four labeled directed-graph options; arrow directions matter"
    return "a Target Graph above four labeled graph options"


@register_task
class GraphRelationGraphOptionsContainedSubgraphLabelTask:
    """Select the option graph contained in the displayed target graph."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'matching')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_plan(self) -> GraphOptionsObjectivePlan:
        """Return the task-owned objective hooks for contained-subgraph matching."""

        return GraphOptionsObjectivePlan(
            owner_id=TASK_ID,
            supported_branch_names=SUPPORTED_QUERY_IDS,
            default_branch_name=QUERY_ID,
            prompt_key=PROMPT_KEY,
            prompt_bundle_id=PROMPT_BUNDLE_ID,
            sampling_namespace=SAMPLING_NAMESPACE,
            dataset_factory=build_contained_subgraph_dataset,
            defaults=_DEFAULTS,
            object_description=_object_description,
        )

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one contained-subgraph option-selection instance."""

        return run_graph_options_plan(
            plan=self._build_plan(),
            domain=self.domain,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
            noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
        )


__all__ = ["GraphRelationGraphOptionsContainedSubgraphLabelTask", "TASK_ID"]
