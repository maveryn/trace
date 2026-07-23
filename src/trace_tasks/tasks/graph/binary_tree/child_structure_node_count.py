"""Count binary-tree nodes matching child-structure predicates."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ._lifecycle import BinaryTreeCountPlan, run_binary_tree_count_plan
from .shared.defaults import BinaryTreeDefaults
from .shared.state import SCENE_ID


TASK_ID = "task_graph__binary_tree__child_structure_node_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "internal_node_count",
    "leaf_node_count",
    "single_child_node_count",
    "two_child_node_count",
)
_COUNT_MODE_BY_QUERY = {
    "internal_node_count": "internal",
    "leaf_node_count": "leaf",
    "single_child_node_count": "single_child",
    "two_child_node_count": "two_child",
}

_DEFAULTS = BinaryTreeDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "graph",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _build_objective_plan() -> BinaryTreeCountPlan:
    """Bind the public child-structure queries to semantic count modes."""

    return BinaryTreeCountPlan(
        owner_id=TASK_ID,
        supported_branch_names=SUPPORTED_QUERY_IDS,
        default_branch_name=SUPPORTED_QUERY_IDS[0],
        count_mode_by_branch=_COUNT_MODE_BY_QUERY,
    )


@register_task
class GraphCountingBinaryTreeChildStructureNodeCountTask:
    """Public owner for binary-tree child-structure node-count queries."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> BinaryTreeCountPlan:
        """Return this task's local count objective plan."""

        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one count instance through neutral scene lifecycle plumbing."""

        return run_binary_tree_count_plan(
            plan=self._build_objective_plan(),
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            defaults=_DEFAULTS,
            background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS,
            noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
        )


__all__ = ["GraphCountingBinaryTreeChildStructureNodeCountTask", "TASK_ID"]
