"""DFA accepted-string label task over automaton diagrams."""
from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ._lifecycle import AcceptanceObjectivePlan, run_acceptance_lifecycle
from .shared.prompts import PROMPT_BUNDLE_ID as AUTOMATON_PROMPT_BUNDLE_ID
from .shared.sampling import AcceptanceDefaults
from .shared.state import SCENE_ID
from ..shared.visual_defaults import load_graph_scene_background_defaults, load_graph_scene_noise_defaults
TASK_ID = 'task_graph__automaton__dfa_accepted_string_label'
QUERY_ID = 'dfa_accepted_string_label'
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
TASK_PROMPT_KEY = 'accepted_string_label_query'
_DEFAULTS = AcceptanceDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults('graph', SCENE_ID, task_id=TASK_ID)
PROMPT_BUNDLE_ID = str(group_default(_PROMPT_DEFAULTS, 'bundle_id', AUTOMATON_PROMPT_BUNDLE_ID))
POST_IMAGE_BACKGROUND_DEFAULTS = load_graph_scene_background_defaults(scene_id=SCENE_ID)
POST_IMAGE_NOISE_DEFAULTS = load_graph_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)

def _prepare_dfa_acceptance_objective(_instance_seed: int, _params: Mapping[str, Any], _query_probabilities: Mapping[str, float], _query_id: str) -> AcceptanceObjectivePlan:
    """Return the deterministic automaton semantic contract for this task."""
    return AcceptanceObjectivePlan(automaton_kind='dfa', task_prompt_key=TASK_PROMPT_KEY, object_description='a state-transition diagram with a start arrow, double-ring accepting states, visible transition labels, and labeled candidate input strings', acceptance_rule='a candidate string is accepted when the deterministic path from the start state ends in a double-ring accepting state after all symbols are consumed')

@register_task
class GraphRelationAutomatonDfaAcceptedStringLabelTask:
    """Choose which candidate input string is accepted by a deterministic automaton."""
    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update', 'matching')
    domain = 'graph'
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one DFA accepted-string instance through the scene lifecycle."""
        return run_acceptance_lifecycle(task_id=TASK_ID, domain=self.domain, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=QUERY_ID, gen_defaults=_GEN_DEFAULTS, render_defaults=_RENDER_DEFAULTS, prompt_bundle_id=PROMPT_BUNDLE_ID, post_image_background_defaults=POST_IMAGE_BACKGROUND_DEFAULTS, post_image_noise_defaults=POST_IMAGE_NOISE_DEFAULTS, defaults=_DEFAULTS, instance_seed=int(instance_seed), params=params, max_attempts=int(max_attempts), prepare_objective=_prepare_dfa_acceptance_objective)
__all__ = ['GraphRelationAutomatonDfaAcceptedStringLabelTask', 'TASK_ID']
