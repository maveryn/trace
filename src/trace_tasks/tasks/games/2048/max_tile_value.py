"""Return the largest tile value after one shown 2048 move."""
from __future__ import annotations
from typing import Any, Dict
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.annotation_artifacts import bbox_set_annotation_artifacts
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from ._lifecycle import Attempt2048Result, Objective2048Plan, all_direction_results, run_2048_lifecycle
from .shared.annotations import entity_bboxes_for_ids
from .shared.rules import board_max_tile, simulate_2048_move, unique_max_result
from .shared.sampling import board_for_merge_values, resolve_2048_axes, resolve_2048_integer_target
from .shared.state import SCENE_ID, Sample2048, coord_to_cell_id
TASK_ID = 'task_games__2048__max_tile_value'
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = 'max_tile_value'
SUPPORTED_QUERY_IDS = (QUERY_ID,)
MAX_TILE_VALUE_SUPPORT = (16, 32, 64, 128, 256)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults('games', SCENE_ID, task_id=TASK_ID)

def _prepare_max_tile_objective(instance_seed: int, params: Dict[str, Any], query_probabilities: Dict[str, float], _query_id: str) -> Objective2048Plan:
    """Resolve the target max tile and bind a constructor that creates it uniquely."""
    target_axis = resolve_2048_integer_target(int(instance_seed), params=params, gen_defaults=_GEN_DEFAULTS, support_key='max_tile_value_support', fallback_support=MAX_TILE_VALUE_SUPPORT, namespace=f'{TASK_ID}.target_answer')
    target = int(target_axis.target_answer)
    if target < 4 or target & target - 1 != 0:
        raise ValueError(f'max tile target must be a power of two >= 4: {target}')

    def construct_attempt(rng, axes) -> Attempt2048Result:
        return _construct_max_tile_attempt(rng=rng, axes=axes, target=target)
    return Objective2048Plan(attempt_namespace='games.2048.max_tile_value', query_params={'target_answer': int(target_axis.target_answer), 'target_answer_support': [int(value) for value in target_axis.target_answer_support], 'target_answer_probabilities': dict(target_axis.target_answer_probabilities)}, construct_attempt=construct_attempt)

def _construct_max_tile_attempt(*, rng: Any, axes: Any, target: int) -> Attempt2048Result:
    """Construct one move whose post-move board has one uniquely largest target tile."""
    board = board_for_merge_values(rng=rng, direction=str(axes.move_direction), merge_values=(int(target),), max_clutter_value=max(2, int(target) // 2))
    result = simulate_2048_move(board, str(axes.move_direction))
    if not result.moved or int(board_max_tile(result.after)) != int(target) or (not unique_max_result(result)):
        raise ValueError('constructed move does not produce the requested unique maximum tile')
    annotation_ids = tuple((coord_to_cell_id(source) for dest, sources in result.result_sources.items() if int(result.after[int(dest[0])][int(dest[1])]) == int(target) for source in sources))
    sample = Sample2048(scene_variant=str(axes.scene_variant), style_variant=str(axes.style_variant), board=board, move_direction=str(axes.move_direction), move_result=result, all_move_results=all_direction_results(board), construction_mode='single_move_max_tile_value')
    return Attempt2048Result(sample=sample, answer_gt=TypedValue(type='integer', value=int(target)), annotation_entity_ids=annotation_ids, build_annotation=lambda rendered: bbox_set_annotation_artifacts(entity_bboxes_for_ids(rendered.rendered_scene, annotation_ids)))

@register_task
class Games2048MaxTileValueTask:
    """Return the largest tile value after one shown 2048 move."""
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'state_update')
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_2048_lifecycle(task_id=TASK_ID, domain=self.domain, prompt_query_key=PROMPT_QUERY_KEY, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=QUERY_ID, gen_defaults=_GEN_DEFAULTS, instance_seed=int(instance_seed), params=params, max_attempts=int(max_attempts), prepare_objective=_prepare_max_tile_objective)
__all__ = ['Games2048MaxTileValueTask']
