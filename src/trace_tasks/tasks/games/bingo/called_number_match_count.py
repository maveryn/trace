"""Count CALLED-list numbers that appear on the visible Bingo card."""
from __future__ import annotations
from trace_tasks.tasks.registry import register_task
from ._lifecycle import BingoAttemptResult, BingoObjectivePlan, bingo_named_count_trace_params, bingo_target_trace_params, resolve_bingo_task_integer_target, run_bingo_lifecycle
from .shared.annotations import called_number_cell_ids
from .shared.rules import build_called_number_match_card_state
TASK_ID = 'task_games__bingo__called_number_match_count'
QUERY_ID = 'called_number_match_count'
SUPPORTED_QUERY_IDS = (QUERY_ID,)
CALLED_NUMBER_MATCH_COUNT_SUPPORT = (0, 1, 2, 3, 4, 5)
CALLED_NUMBER_COUNT_SUPPORT = (5, 6, 7, 8)

def _prepare_called_number_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve called-list count axes and bind card-match construction."""
    target_axis = resolve_bingo_task_integer_target(instance_seed=int(instance_seed), task_id=TASK_ID, task_params=task_params, support_key='called_number_match_count_support', fallback_support=CALLED_NUMBER_MATCH_COUNT_SUPPORT, namespace=f'{TASK_ID}.target_answer')
    called_count_axis = resolve_bingo_task_integer_target(instance_seed=int(instance_seed), task_id=TASK_ID, task_params=task_params, support_key='called_number_count_support', fallback_support=CALLED_NUMBER_COUNT_SUPPORT, namespace=f'{TASK_ID}.called_number_count', explicit_key='called_number_count', balanced_flag_key='balanced_called_number_count_sampling')

    def construct_attempt(rng, _axes):
        card_state = build_called_number_match_card_state(rng=rng, present_called_count=int(target_axis.target_answer), called_number_count=int(called_count_axis.target_answer))
        annotation_cell_ids = called_number_cell_ids(card_state)
        visible_numbers = {int(value) for row in card_state.numbers_grid for value in row}
        absent_numbers = tuple(int(value) for value in card_state.called_numbers if int(value) not in visible_numbers)
        return BingoAttemptResult(card_state=card_state, answer_value=int(target_axis.target_answer), annotation_cell_ids=annotation_cell_ids, annotation_type='bbox_set', execution_extra={'called_number_count': int(called_count_axis.target_answer), 'called_absent_numbers': [int(value) for value in absent_numbers]})
    return BingoObjectivePlan(attempt_namespace=f'games.bingo.{TASK_ID}', prompt_query_key=QUERY_ID, show_called_panel=True, query_params={**bingo_target_trace_params(target_axis), **bingo_named_count_trace_params('called_number_count', called_count_axis)}, construct_attempt=construct_attempt)

@register_task
class GamesBingoCalledNumberMatchCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition')
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_bingo_lifecycle(task_id=TASK_ID, supported_query_ids=SUPPORTED_QUERY_IDS, instance_seed=int(instance_seed), params=params, max_attempts=int(max_attempts), prepare_objective=_prepare_called_number_objective)
