"""Identify the one completed BINGO column."""
from __future__ import annotations
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ._lifecycle import BingoAttemptResult, BingoObjectivePlan, resolve_bingo_task_float_param, run_bingo_lifecycle
from .shared.defaults import SCENE_ID
from .shared.rules import build_completed_column_label_card_state
from .shared.state import BINGO_BOARD_SIZE, BINGO_COLUMN_LABELS, cell_id
TASK_ID = 'task_games__bingo__completed_column_label'
QUERY_ID = 'completed_column_label'
SUPPORTED_QUERY_IDS = (QUERY_ID,)

def _resolve_target_column_label(instance_seed: int, task_params) -> tuple[str, dict[str, float]]:
    """Resolve the task-owned B/I/N/G/O target column label."""
    gen_defaults, _render_defaults, _prompt_defaults = load_scene_generation_rendering_prompt_defaults('games', SCENE_ID, task_id=TASK_ID)
    rng = spawn_rng(int(instance_seed), f'{TASK_ID}.target_column_label')
    selected, probabilities = resolve_variant(rng, params=task_params, gen_defaults=gen_defaults, supported_variants=BINGO_COLUMN_LABELS, explicit_key='target_column_label', weights_key='target_column_label_weights')
    selected = apply_balanced_variant_sampling(instance_seed=int(instance_seed), params=task_params, gen_defaults=gen_defaults, selected_variant=str(selected), variant_probabilities=probabilities, supported_variants=BINGO_COLUMN_LABELS, balance_flag_key='balanced_target_column_label_sampling', explicit_key='target_column_label', weights_key='target_column_label_weights', sampling_namespace=f'{TASK_ID}.target_column_label')
    return (str(selected), dict(probabilities))

def _prepare_completed_column_objective(instance_seed, task_params, _query_id, _query_probabilities):
    """Resolve target column label and bind exact completed-column construction."""
    target_label, target_label_probabilities = _resolve_target_column_label(int(instance_seed), task_params)
    target_column_index = int(BINGO_COLUMN_LABELS.index(str(target_label)))
    distractor_mark_prob = resolve_bingo_task_float_param(task_id=TASK_ID, task_params=task_params, key='column_distractor_mark_prob', fallback=0.2)
    top_cell_id = cell_id(row_index=0, column_index=int(target_column_index))
    bottom_cell_id = cell_id(row_index=BINGO_BOARD_SIZE - 1, column_index=int(target_column_index))

    def construct_attempt(rng, _axes):
        card_state = build_completed_column_label_card_state(rng=rng, target_column_label=str(target_label), distractor_mark_prob=float(distractor_mark_prob))
        return BingoAttemptResult(card_state=card_state, answer_value=str(target_label), annotation_cell_ids=(top_cell_id, bottom_cell_id), annotation_type='segment', annotation_cell_id_pairs=((top_cell_id, bottom_cell_id),), execution_extra={'target_column_label': str(target_label), 'target_column_index': int(target_column_index), 'target_column_label_probabilities': dict(target_label_probabilities), 'column_distractor_mark_prob': float(distractor_mark_prob)})
    return BingoObjectivePlan(attempt_namespace=f'games.bingo.{TASK_ID}', prompt_query_key=QUERY_ID, query_params={'target_column_label': str(target_label), 'target_column_index': int(target_column_index), 'target_column_label_probabilities': dict(target_label_probabilities), 'column_distractor_mark_prob': float(distractor_mark_prob)}, construct_attempt=construct_attempt)

@register_task
class GamesBingoCompletedColumnLabelTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering',)
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_bingo_lifecycle(task_id=TASK_ID, supported_query_ids=SUPPORTED_QUERY_IDS, instance_seed=int(instance_seed), params=params, max_attempts=int(max_attempts), prepare_objective=_prepare_completed_column_objective)
