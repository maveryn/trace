"""Count strongly connected components from a directed adjacency panel."""
from __future__ import annotations
from typing import Any, Dict
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ._lifecycle import ComponentCountPlan, run_component_count_lifecycle
from .shared.prompts import PROMPT_BUNDLE_ID as ADJACENCY_PROMPT_BUNDLE_ID
from .shared.sampling import ComponentCountDefaults
from .shared.state import SCENE_ID
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
TASK_ID = 'task_graph__adjacency__directed_strong_component_count'
PUBLIC_QUERY_ID = 'single'
PROMPT_KEY = 'directed_strong_component_count'
QUERY_ID = PUBLIC_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_DEFAULTS = ComponentCountDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults('graph', SCENE_ID, task_id=TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, 'bundle_id', ADJACENCY_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)

def _prepare_directed_component_count_objective(axes: Any) -> ComponentCountPlan:
    """Return directed component semantics for the public task."""
    return ComponentCountPlan(directed=True, object_description=f"a directed graph as an adjacency {('matrix' if axes.scene_variant == 'adjacency_matrix_panel' else 'list')}")

@register_task
class GraphCountingAdjacencyDirectedStrongComponentCountTask:
    """Count strongly connected components in a directed adjacency representation."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a directed SCC count through scene-private plumbing."""
        return run_component_count_lifecycle(task_id=TASK_ID, domain=self.domain, scene_id=SCENE_ID, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=QUERY_ID, gen_defaults=_GEN_DEFAULTS, render_defaults=_RENDER_DEFAULTS, prompt_bundle_id=PROMPT_BUNDLE_ID, prompt_key=PROMPT_KEY, background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS, noise_defaults=POST_IMAGE_NOISE_DEFAULTS, defaults=_DEFAULTS, instance_seed=int(instance_seed), params=params, prepare_objective=_prepare_directed_component_count_objective)
__all__ = ['GraphCountingAdjacencyDirectedStrongComponentCountTask']
