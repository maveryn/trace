"""Count Backgammon destination points matching one move predicate."""
from __future__ import annotations
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from ._lifecycle import BackgammonObjectivePlan, integer_count_attempt_result, resolve_backgammon_count_target, run_backgammon_lifecycle
from .shared.sampling import sample_destination_count_scene
from .shared.state import DESTINATION_STATUS_BLOCKED, DESTINATION_STATUS_HIT, DESTINATION_STATUS_LEGAL, SCENE_ID
TASK_ID = 'task_games__backgammon__destination_count'
LEGAL_COUNT_SUPPORT = (1, 2, 3, 4, 5)
HIT_COUNT_SUPPORT = (0, 1, 2, 3, 4, 5)
BLOCKED_COUNT_SUPPORT = (0, 1, 2, 3, 4)
SOURCE_COUNT_MIN = 2
SOURCE_COUNT_MAX = 4
DISTRACTOR_STACK_PROBABILITY = 0.08
DESTINATION_QUERY_SPECS: dict[str, tuple[str, str, tuple[int, ...]]] = {'legal_move_count': (DESTINATION_STATUS_LEGAL, 'legal_count_support', LEGAL_COUNT_SUPPORT), 'hit_move_count': (DESTINATION_STATUS_HIT, 'hit_count_support', HIT_COUNT_SUPPORT), 'blocked_destination_count': (DESTINATION_STATUS_BLOCKED, 'blocked_count_support', BLOCKED_COUNT_SUPPORT)}
SUPPORTED_QUERY_IDS = tuple(DESTINATION_QUERY_SPECS)

def _prepare_destination_objective(instance_seed, task_params, query_id):
    """Resolve the destination predicate and bind exact-count sample construction."""
    destination_status, support_key, fallback_support = DESTINATION_QUERY_SPECS[str(query_id)]
    target = resolve_backgammon_count_target(instance_seed=int(instance_seed), task_params=task_params, task_id=TASK_ID, support_key=support_key, fallback_support=fallback_support, namespace=f'backgammon.destination_count.{str(query_id)}.target_answer')

    def construct_attempt(rng, axes):
        sample = sample_destination_count_scene(rng, axes=axes, destination_status=str(destination_status), target_answer=int(target.target_answer), source_count_min=SOURCE_COUNT_MIN, source_count_max=SOURCE_COUNT_MAX, distractor_stack_probability=DISTRACTOR_STACK_PROBABILITY)
        return integer_count_attempt_result(sample=sample, target_points=tuple((int(point) for point in sample.target_points or sample.target_destinations)), construction_mode='exact_destination_count')
    return BackgammonObjectivePlan(attempt_namespace='games.backgammon.destination_count', query_params={'target_answer_support': [int(value) for value in target.target_answer_support], 'target_answer_probabilities': dict(target.target_answer_probabilities), 'source_count_min': int(SOURCE_COUNT_MIN), 'source_count_max': int(SOURCE_COUNT_MAX), 'distractor_stack_probability': float(DISTRACTOR_STACK_PROBABILITY)}, construct_attempt=construct_attempt)

@register_task
class GamesBackgammonDestinationCountTask:
    """Count legal, hit, or blocked destination points for the shown dice."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate a destination-count board by binding one move predicate locally."""
        return run_backgammon_lifecycle(task_id=self.task_id, supported_query_ids=SUPPORTED_QUERY_IDS, instance_seed=int(instance_seed), params=params, max_attempts=int(max_attempts), prepare_objective=_prepare_destination_objective)
__all__ = ['GamesBackgammonDestinationCountTask']
