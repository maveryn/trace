"""Count merges made by one shown 2048 move."""
from __future__ import annotations
from typing import Any, Dict
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from ._lifecycle import Objective2048Plan, prepare_merge_source_integer_objective, run_2048_lifecycle
from .shared.state import SCENE_ID
TASK_ID = 'task_games__2048__merge_count'
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = 'merge_count'
SUPPORTED_QUERY_IDS = (QUERY_ID,)
MERGE_COUNT_SUPPORT = (0, 1, 2, 3, 4)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults('games', SCENE_ID, task_id=TASK_ID)

def _prepare_merge_count_objective(instance_seed: int, params: Dict[str, Any], _query_probabilities: Dict[str, float], _query_id: str) -> Objective2048Plan:
    return prepare_merge_source_integer_objective(instance_seed=int(instance_seed), params=params, gen_defaults=_GEN_DEFAULTS, support_key='merge_count_support', fallback_support=MERGE_COUNT_SUPPORT, target_namespace=f'{TASK_ID}.target_answer', attempt_namespace='games.2048.merge_count', construction_mode='single_move_merge_count', merge_values_for_target=lambda target, rng: tuple((int(rng.choice((4, 8, 16, 32))) for _idx in range(int(target)))), result_matches_target=lambda result, target: len(result.merge_pairs) == int(target))

@register_task
class Games2048MergeCountTask:
    """Count merges made by one shown 2048 move."""
    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_2048_lifecycle(task_id=TASK_ID, domain=self.domain, prompt_query_key=PROMPT_QUERY_KEY, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=QUERY_ID, gen_defaults=_GEN_DEFAULTS, instance_seed=int(instance_seed), params=params, max_attempts=int(max_attempts), prepare_objective=_prepare_merge_count_objective)
__all__ = ['Games2048MergeCountTask']
