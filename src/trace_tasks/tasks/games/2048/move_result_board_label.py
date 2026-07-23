"""Choose the labeled candidate board matching one shown 2048 move result."""
from __future__ import annotations
from typing import Any, Dict
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from ._lifecycle import Attempt2048Result, Objective2048Plan, all_direction_results, run_2048_lifecycle
from .shared.annotations import entity_bboxes_for_ids
from .shared.rules import board_key, simulate_2048_move
from .shared.sampling import board_for_merge_values, build_result_board_options, resolve_2048_result_board_target
from .shared.state import SCENE_ID, SUPPORTED_2048_RESULT_BOARD_LABELS, Sample2048
TASK_ID = 'task_games__2048__move_result_board_label'
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = 'move_result_board_label'
SUPPORTED_QUERY_IDS = (QUERY_ID,)
RESULT_BOARD_OPTION_COUNT_SUPPORT = (4, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults('games', SCENE_ID, task_id=TASK_ID)

def _prepare_result_board_objective(instance_seed: int, params: Dict[str, Any], _query_probabilities: Dict[str, float], _query_id: str) -> Objective2048Plan:
    """Resolve option labels/counts and bind a constructor with one correct result board."""
    target_axis = resolve_2048_result_board_target(int(instance_seed), params=params, gen_defaults=_GEN_DEFAULTS, fallback_labels=SUPPORTED_2048_RESULT_BOARD_LABELS, fallback_option_counts=RESULT_BOARD_OPTION_COUNT_SUPPORT, namespace=TASK_ID)
    target_label = str(target_axis.target_label)
    labels = tuple((str(label) for label in target_axis.target_label_support))

    def construct_attempt(rng, axes) -> Attempt2048Result:
        """Build a board/options set where exactly one option equals the post-move board."""
        merge_count = int(rng.randint(0, 3))
        merge_values = tuple((int(rng.choice((4, 8, 16, 32, 64))) for _idx in range(merge_count)))
        board = board_for_merge_values(rng=rng, direction=str(axes.move_direction), merge_values=merge_values, force_slide_when_no_merge=merge_count == 0)
        result = simulate_2048_move(board, str(axes.move_direction))
        if not result.moved:
            raise ValueError('constructed board has no visible move result')
        all_results = all_direction_results(board)
        option_boards = build_result_board_options(rng=rng, board=board, result=result, all_results=all_results, target_label=target_label, labels=labels)
        matching_labels = [str(label) for label, option_board in option_boards.items() if board_key(option_board) == board_key(result.after)]
        if len(matching_labels) != 1:
            raise ValueError('result-board options must contain exactly one correct board')
        answer = str(matching_labels[0])
        annotation_ids = (f'result_option_{answer}',)
        sample = Sample2048(scene_variant=str(axes.scene_variant), style_variant=str(axes.style_variant), board=board, move_direction=str(axes.move_direction), move_result=result, all_move_results=all_results, construction_mode='single_move_full_result_board_mcq', result_option_boards=option_boards)
        return Attempt2048Result(sample=sample, answer_gt=TypedValue(type='string', value=answer), annotation_entity_ids=annotation_ids, build_annotation=lambda rendered: bbox_annotation_artifacts(entity_bboxes_for_ids(rendered.rendered_scene, annotation_ids)[0]))
    return Objective2048Plan(attempt_namespace='games.2048.move_result_board_label', query_params={'target_label': str(target_axis.target_label), 'target_label_support': [str(value) for value in target_axis.target_label_support], 'target_label_probabilities': dict(target_axis.target_label_probabilities), 'result_board_option_count': int(target_axis.result_board_option_count), 'result_board_option_count_support': [int(value) for value in target_axis.result_board_option_count_support], 'result_board_option_count_probabilities': dict(target_axis.result_board_option_count_probabilities)}, construct_attempt=construct_attempt)

@register_task
class Games2048MoveResultBoardLabelTask:
    """Choose the labeled candidate board matching one shown 2048 move result."""
    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_2048_lifecycle(task_id=TASK_ID, domain=self.domain, prompt_query_key=PROMPT_QUERY_KEY, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=QUERY_ID, gen_defaults=_GEN_DEFAULTS, instance_seed=int(instance_seed), params=params, max_attempts=int(max_attempts), prepare_objective=_prepare_result_board_objective, render_param_overrides={'dynamic_canvas_size_enabled': False, 'canvas_width': 900, 'canvas_height': 900})
__all__ = ['Games2048MoveResultBoardLabelTask']
