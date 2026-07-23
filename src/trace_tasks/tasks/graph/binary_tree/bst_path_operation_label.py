"""Identify the terminal node for a binary-search-tree path operation."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ._lifecycle import BinaryTreeOperationPlan, run_binary_tree_operation_plan
from .shared.defaults import BinaryTreeDefaults
from .shared.state import SCENE_ID


TASK_ID = "task_graph__binary_tree__bst_path_operation_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "bst_search_terminal_label",
    "bst_insert_parent_label",
)
_OPERATION_KIND_BY_QUERY = {
    "bst_search_terminal_label": "bst_search_terminal",
    "bst_insert_parent_label": "bst_insert_parent",
}

_DEFAULTS = BinaryTreeDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "graph",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _build_objective_plan() -> BinaryTreeOperationPlan:
    """Bind BST operation queries to semantic path-operation kinds."""

    return BinaryTreeOperationPlan(
        owner_id=TASK_ID,
        supported_branch_names=SUPPORTED_QUERY_IDS,
        default_branch_name=SUPPORTED_QUERY_IDS[0],
        operation_kind_by_branch=_OPERATION_KIND_BY_QUERY,
        scene_title="Binary Search Tree",
        object_description_key="object_description_bst",
        prompt_family="operation_path",
    )


@register_task
class GraphRelationBstPathOperationLabelTask:
    """Public owner for BST search and insert path-operation queries."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update')
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> BinaryTreeOperationPlan:
        """Return this task's local BST operation objective plan."""

        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one BST operation instance through neutral scene lifecycle plumbing."""

        return run_binary_tree_operation_plan(
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


__all__ = ["GraphRelationBstPathOperationLabelTask", "TASK_ID"]
