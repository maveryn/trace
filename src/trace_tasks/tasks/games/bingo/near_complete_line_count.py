from __future__ import annotations
from trace_tasks.tasks.registry import register_task
from ._lifecycle import BingoAttemptResult, BingoObjectivePlan, bingo_target_trace_params, resolve_bingo_task_float_param, resolve_bingo_task_integer_target, run_bingo_lifecycle
from .shared.annotations import near_complete_gap_cell_ids
from .shared.rules import build_near_complete_line_card_state
from .shared.defaults import SCENE_ID
TASK_ID = 'task_games__bingo__near_complete_line_count'
NEAR_COMPLETE_QUERY_LINE_AXES = {'near_complete_row_count': 'row', 'near_complete_column_count': 'column'}
SUPPORTED_QUERY_IDS = tuple(NEAR_COMPLETE_QUERY_LINE_AXES)
NEAR_COMPLETE_LINE_COUNT_SUPPORT = (0, 1, 2, 3, 4)

def _prepare_near_complete_objective(instance_seed, task_params, query_id, query_probabilities):
    """Resolve near-complete row/column semantics and bind exact-gap construction."""
    line_axis = NEAR_COMPLETE_QUERY_LINE_AXES[str(query_id)]
    target_axis = resolve_bingo_task_integer_target(instance_seed=int(instance_seed), task_id=TASK_ID, task_params=task_params, support_key='near_complete_line_count_support', fallback_support=NEAR_COMPLETE_LINE_COUNT_SUPPORT, namespace=f'{TASK_ID}.{str(query_id)}.target_answer')
    distractor_mark_prob = resolve_bingo_task_float_param(task_id=TASK_ID, task_params=task_params, key='axis_distractor_mark_prob', fallback=0.2)

    def construct_attempt(rng, _axes):
        card_state = build_near_complete_line_card_state(rng=rng, line_axis=str(line_axis), near_complete_line_count=int(target_axis.target_answer), distractor_mark_prob=float(distractor_mark_prob))
        annotation_cell_ids = near_complete_gap_cell_ids(card_state)
        return BingoAttemptResult(card_state=card_state, answer_value=int(target_axis.target_answer), annotation_cell_ids=annotation_cell_ids, execution_extra={'line_axis': str(line_axis), 'query_id_probabilities': dict(query_probabilities), 'axis_distractor_mark_prob': float(distractor_mark_prob)})
    return BingoObjectivePlan(attempt_namespace=f'games.bingo.{TASK_ID}', prompt_query_key=str(query_id), query_params={'line_axis': str(line_axis), 'line_axis_probabilities': {str(line_axis): 1.0}, **bingo_target_trace_params(target_axis), 'axis_distractor_mark_prob': float(distractor_mark_prob)}, construct_attempt=construct_attempt)

@register_task
class GamesBingoNearCompleteLineCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_bingo_lifecycle(task_id=TASK_ID, supported_query_ids=SUPPORTED_QUERY_IDS, instance_seed=int(instance_seed), params=params, max_attempts=max_attempts, prepare_objective=_prepare_near_complete_objective)
