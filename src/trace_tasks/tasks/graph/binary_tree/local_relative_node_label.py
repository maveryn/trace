"""Identify local relatives in a binary-tree diagram."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
from ._lifecycle import BinaryTreeRelationPlan, run_binary_tree_relation_plan
from .shared.defaults import BinaryTreeDefaults
from .shared.state import SCENE_ID


TASK_ID = "task_graph__binary_tree__local_relative_node_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "parent_label",
    "left_child_label",
    "right_child_label",
    "sibling_label",
)
_RELATION_KIND_BY_QUERY = {
    "parent_label": "parent",
    "left_child_label": "left_child",
    "right_child_label": "right_child",
    "sibling_label": "sibling",
}
_ANNOTATION_ROLES_BY_QUERY = {
    "parent_label": ("child", "parent"),
    "left_child_label": ("parent", "left_child"),
    "right_child_label": ("parent", "right_child"),
    "sibling_label": ("node", "sibling"),
}

_DEFAULTS = BinaryTreeDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "graph",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _build_objective_plan() -> BinaryTreeRelationPlan:
    """Bind local-relative queries to semantic relation and annotation roles."""

    return BinaryTreeRelationPlan(
        owner_id=TASK_ID,
        supported_branch_names=SUPPORTED_QUERY_IDS,
        default_branch_name=SUPPORTED_QUERY_IDS[0],
        relation_kind_by_branch=_RELATION_KIND_BY_QUERY,
        annotation_roles_by_branch=_ANNOTATION_ROLES_BY_QUERY,
    )


@register_task
class GraphRelationBinaryTreeLocalRelativeNodeLabelTask:
    """Public owner for parent, child, and sibling label queries."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = "graph"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_objective_plan(self) -> BinaryTreeRelationPlan:
        """Return this task's local relation objective plan."""

        return _build_objective_plan()

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one relation instance through neutral scene lifecycle plumbing."""

        return run_binary_tree_relation_plan(
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


__all__ = ["GraphRelationBinaryTreeLocalRelativeNodeLabelTask", "TASK_ID"]
