"""Return the label at a position in a binary-tree traversal order."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ._lifecycle import BinaryTreeTraversalPlan, run_binary_tree_traversal_plan
from .shared.defaults import BinaryTreeDefaults
from .shared.state import SCENE_ID


TASK_ID = "task_graph__binary_tree__traversal_kth_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "preorder_kth_node_label",
    "inorder_kth_node_label",
    "postorder_kth_node_label",
    "level_order_kth_node_label",
)
_TRAVERSAL_ORDER_BY_QUERY = {
    "preorder_kth_node_label": "preorder",
    "inorder_kth_node_label": "inorder",
    "postorder_kth_node_label": "postorder",
    "level_order_kth_node_label": "level_order",
}

_DEFAULTS = BinaryTreeDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "graph",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _build_objective_plan() -> BinaryTreeTraversalPlan:
    """Bind public traversal queries to the requested traversal order."""

    return BinaryTreeTraversalPlan(
        owner_id=TASK_ID,
        supported_branch_names=SUPPORTED_QUERY_IDS,
        default_branch_name=SUPPORTED_QUERY_IDS[0],
        traversal_order_by_branch=_TRAVERSAL_ORDER_BY_QUERY,
    )


@register_task
class GraphOrderBinaryTreeTraversalLabelTask:
    """Public owner for binary-tree traversal-position label queries."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> BinaryTreeTraversalPlan:
        """Return this task's local traversal objective plan."""

        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one traversal instance through neutral scene lifecycle plumbing."""

        return run_binary_tree_traversal_plan(
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


__all__ = ["GraphOrderBinaryTreeTraversalLabelTask", "TASK_ID"]
