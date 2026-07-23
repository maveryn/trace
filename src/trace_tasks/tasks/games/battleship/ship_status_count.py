"""Count Battleship ships matching a hit-status predicate."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.battleship.shared.annotations import project_ship_status_annotation
from trace_tasks.tasks.games.battleship.shared.output import common_trace_sections, projected_annotation_payload, target_ship_cell_ids, witness_symbolic_payload
from trace_tasks.tasks.games.battleship.shared.rendering import render_battleship_sample
from trace_tasks.tasks.games.battleship.shared.sampling import ResolvedBattleshipSceneAxes, build_battleship_scene_state, build_ship_placements_with_hits, place_fleet, resolve_battleship_scene_axes, sample_miss_coords
from trace_tasks.tasks.games.battleship.shared.state import FLEET_SHAPES, SCENE_ID, BattleshipSample, Coord, sorted_coords
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support
from .shared.prompts import build_battleship_prompt_artifacts
TASK_ID = 'task_games__battleship__ship_status_count'
SHIP_STATUS_COUNT_QUERY_IDS: Tuple[str, ...] = ('sunk_ship_count', 'partial_ship_count')
SUNK_SHIP_COUNT_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
PARTIAL_SHIP_COUNT_SUPPORT: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
MIN_PARTIAL_SHIP_COUNT = 2
MAX_PARTIAL_SHIP_COUNT = 4
_SHIP_STATUS_TARGET_SUPPORTS: dict[str, Tuple[int, ...]] = {'sunk_ship_count_support': SUNK_SHIP_COUNT_SUPPORT, 'partial_ship_count_support': PARTIAL_SHIP_COUNT_SUPPORT}
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults('games', SCENE_ID, task_id=TASK_ID)

@dataclass(frozen=True)
class ShipStatusInstance:
    """Task-owned Battleship ship-status construction result."""
    query_id: str
    axes: ResolvedBattleshipSceneAxes
    sample: BattleshipSample
    answer: int
    annotation_coords: Tuple[Coord, ...]
    annotation_ship_ids: Tuple[str, ...]
    target_answer: int
    target_answer_support: Tuple[int, ...]
    target_answer_probabilities: Dict[str, float]
    query_id_probabilities: Dict[str, float]

def _target_support_key(query_id: str) -> str:
    """Return the target answer support key for this public task."""
    return {'sunk_ship_count': 'sunk_ship_count_support', 'partial_ship_count': 'partial_ship_count_support'}[str(query_id)]

def _select_ship_status_query(instance_seed: int, params: Mapping[str, Any]) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve this task's internal query branch at the public task boundary."""
    return select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SHIP_STATUS_COUNT_QUERY_IDS, default_query_id=SHIP_STATUS_COUNT_QUERY_IDS[0], task_id=TASK_ID, namespace=f'{TASK_ID}.query')

def _partial_ship_count_bounds(params: Mapping[str, Any]) -> Tuple[int, int]:
    """Resolve task-owned partially hit ship count bounds."""
    low = int(params.get('min_partial_ship_count', group_default(_GEN_DEFAULTS, 'min_partial_ship_count', MIN_PARTIAL_SHIP_COUNT)))
    high = int(params.get('max_partial_ship_count', group_default(_GEN_DEFAULTS, 'max_partial_ship_count', MAX_PARTIAL_SHIP_COUNT)))
    if low > high:
        raise ValueError('min_partial_ship_count must be <= max_partial_ship_count')
    return (max(0, int(low)), max(0, int(high)))

def _resolve_target_answer(*, instance_seed: int, params: Mapping[str, Any], query_id: str) -> tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve the target answer for this task-owned query branch."""
    support_key = _target_support_key(str(query_id))
    target_answer, probabilities = resolve_integer_choice(instance_seed=int(instance_seed), params=params, gen_defaults=_GEN_DEFAULTS, support_key=str(support_key), explicit_key='target_answer', fallback_support=_SHIP_STATUS_TARGET_SUPPORTS[support_key], namespace=f'{TASK_ID}.target_answer.{str(query_id)}', balanced_flag_key='balanced_target_answer_sampling', namespace_support_permutation=True)
    support = resolve_integer_support(params, gen_defaults=_GEN_DEFAULTS, key=str(support_key), fallback=_SHIP_STATUS_TARGET_SUPPORTS[support_key])
    return (int(target_answer), tuple((int(value) for value in support)), dict(probabilities))

def _sample_ship_status_instance(*, rng: Any, axes: ResolvedBattleshipSceneAxes, params: Mapping[str, Any], query_id: str, target_answer: int, target_answer_support: Tuple[int, ...], target_answer_probabilities: Dict[str, float], query_id_probabilities: Dict[str, float]) -> ShipStatusInstance:
    """Construct one target-matched ship-status instance without changing query ownership."""
    fleet_size = len(FLEET_SHAPES)
    if int(target_answer) < 0 or int(target_answer) > int(fleet_size):
        raise ValueError(f'unsupported Battleship ship-status target answer: {target_answer}')
    if str(query_id) == 'sunk_ship_count':
        sunk_count = int(target_answer)
        partial_low, partial_high = _partial_ship_count_bounds(params)
        partial_count = min(int(fleet_size) - int(sunk_count), int(rng.randint(int(partial_low), int(partial_high))))
    else:
        partial_count = int(target_answer)
        sunk_support = resolve_integer_support(params, gen_defaults=_GEN_DEFAULTS, key='sunk_ship_count_support', fallback=SUNK_SHIP_COUNT_SUPPORT)
        sunk_candidates = [int(value) for value in sunk_support if 0 < int(value) <= int(fleet_size) - int(partial_count)]
        if not sunk_candidates:
            sunk_candidates = [0]
        sunk_count = int(rng.choice(sunk_candidates))
    base_placements = place_fleet(rng=rng, board_size=int(axes.board_size))
    shuffled_shapes = list(FLEET_SHAPES)
    rng.shuffle(shuffled_shapes)
    sunk_shape_ids = {str(shape.shape_id) for shape in shuffled_shapes[:int(sunk_count)]}
    remaining_shape_ids = [str(shape.shape_id) for shape in shuffled_shapes[int(sunk_count):]]
    partial_shape_ids = set(remaining_shape_ids[:int(partial_count)])
    hit_coords_by_ship_id: dict[str, Tuple[Coord, ...]] = {}
    for ship in base_placements:
        if str(ship.shape_id) in sunk_shape_ids:
            hit_coords_by_ship_id[str(ship.ship_id)] = tuple(ship.coords)
        elif str(ship.shape_id) in partial_shape_ids:
            shuffled = list(ship.coords)
            rng.shuffle(shuffled)
            hit_count = int(rng.randint(1, max(1, len(ship.coords) - 1)))
            hit_coords_by_ship_id[str(ship.ship_id)] = sorted_coords(shuffled[:hit_count])
        else:
            hit_coords_by_ship_id[str(ship.ship_id)] = tuple()
    placements = build_ship_placements_with_hits(base_placements, hit_coords_by_ship_id=hit_coords_by_ship_id)
    ship_cells = {coord for ship in placements for coord in ship.coords}
    miss_coords = sample_miss_coords(rng=rng, board_size=int(axes.board_size), occupied_ship_coords=ship_cells, excluded_coords=tuple(), params=params, gen_defaults=_GEN_DEFAULTS)
    sample = build_battleship_scene_state(board_size=int(axes.board_size), scene_variant=str(axes.scene_variant), placements=placements, miss_coords=miss_coords, construction_mode='placed_fleet_with_full_partial_and_missed_shots')
    if str(query_id) == 'sunk_ship_count':
        annotation_ships = [ship for ship in sample.ship_placements if bool(ship.is_sunk)]
        answer = int(sample.sunk_ship_count)
    else:
        annotation_ships = [ship for ship in sample.ship_placements if bool(ship.hit_coords) and (not bool(ship.is_sunk))]
        answer = int(sample.partial_ship_count)
    if int(answer) != int(target_answer):
        raise ValueError('Battleship ship-status construction did not match target answer')
    annotation_coords = sorted_coords((coord for ship in annotation_ships for coord in ship.hit_coords))
    annotation_ship_ids = tuple((str(ship.ship_id) for ship in annotation_ships))
    return ShipStatusInstance(query_id=str(query_id), axes=axes, sample=sample, answer=int(answer), annotation_coords=annotation_coords, annotation_ship_ids=annotation_ship_ids, target_answer=int(target_answer), target_answer_support=tuple((int(value) for value in target_answer_support)), target_answer_probabilities=dict(target_answer_probabilities), query_id_probabilities=dict(query_id_probabilities))

def _build_task_output(*, instance: ShipStatusInstance, rendered_context: Any, annotation_projection: Any, prompt_defaults: Mapping[str, Any], prompt_artifacts: Any) -> TaskOutput:
    """Build final TaskOutput for this public task."""
    answer_gt = TypedValue(type='integer', value=int(instance.answer))
    annotation_gt = TypedValue(type='bbox_set_map', value=dict(annotation_projection.annotation_bbox_set_map))
    trace_payload = common_trace_sections(axes=instance.axes, sample=instance.sample, rendered_context=rendered_context, annotation_projection=annotation_projection)
    trace_payload['scene_ir']['relations'].update({'query_id': str(instance.query_id), 'target_answer': int(instance.target_answer), 'target_ship_id': '', 'target_ship_display_name': '', 'target_ship_shape_id': '', 'target_cell_status': '', 'target_missing_coord': None})
    trace_payload['query_spec'] = build_prompt_query_spec(prompt_artifacts=prompt_artifacts, query_id=str(instance.query_id), params={'scene_variant': str(instance.axes.scene_variant), 'query_id': str(instance.query_id), 'style_variant': str(instance.axes.style_variant), 'board_size': int(instance.sample.board_size), 'scene_variant_probabilities': dict(instance.axes.scene_variant_probabilities), 'query_id_probabilities': dict(instance.query_id_probabilities), 'style_variant_probabilities': dict(instance.axes.style_variant_probabilities), 'board_size_probabilities': dict(instance.axes.board_size_probabilities), 'target_answer': int(instance.target_answer), 'target_answer_support': [int(value) for value in instance.target_answer_support], 'target_answer_probabilities': dict(instance.target_answer_probabilities), 'target_ship_id': '', 'target_ship_display_name': '', 'target_ship_shape_id': '', 'target_ship_id_probabilities': {}, 'target_cell_status': '', 'target_missing_coord': None, 'candidate_labels': [], 'last_ship_cell_option_count': 0, 'last_ship_cell_option_count_support': [], 'last_ship_cell_option_count_probabilities': {}, 'hit_count': len(instance.sample.hit_coords), 'miss_count': len(instance.sample.miss_coords), 'sunk_ship_count': int(instance.sample.sunk_ship_count), 'partial_ship_count': int(instance.sample.partial_ship_count), 'untouched_ship_count': int(instance.sample.untouched_ship_count)})
    trace_payload['execution_trace'].update({'query_id': str(instance.query_id), 'target_answer': int(instance.target_answer), 'target_answer_support': [int(value) for value in instance.target_answer_support], 'annotation_coords': [[int(row), int(col)] for row, col in instance.annotation_coords], 'annotation_ship_ids': [str(ship_id) for ship_id in instance.annotation_ship_ids], 'target_ship_id': '', 'target_ship_display_name': '', 'target_ship_shape_id': '', 'target_ship_cell_ids': target_ship_cell_ids(instance.sample, target_ship_id=''), 'target_cell_status': '', 'target_missing_coord': None})
    trace_payload['witness_symbolic'] = witness_symbolic_payload(annotation_gt=annotation_gt, annotation_projection=annotation_projection)
    trace_payload['projected_annotation'] = projected_annotation_payload(annotation_gt=annotation_gt, annotation_projection=annotation_projection)
    return TaskOutput(prompt=str(prompt_artifacts.prompt), prompt_variants=dict(prompt_artifacts.prompt_variants), answer_gt=answer_gt, annotation_gt=annotation_gt, image=rendered_context.image, image_id='img0', trace_payload=trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=str(instance.query_id))

@register_task
class GamesBattleshipShipStatusCountTask:
    """Count sunk or partially hit ships in the shown fleet."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SHIP_STATUS_COUNT_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a fixed ship-status query with retryable board construction only."""
        query_id, query_id_probabilities, task_params = _select_ship_status_query(int(instance_seed), params)
        axes = resolve_battleship_scene_axes(int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS)
        target_answer, target_answer_support, target_answer_probabilities = _resolve_target_answer(instance_seed=int(instance_seed), params=task_params, query_id=str(query_id))
        instance = None
        for attempt_index in range(max(1, int(max_attempts))):
            rng = spawn_rng(int(instance_seed), f'{self.task_id}.attempt.{int(attempt_index)}')
            try:
                instance = _sample_ship_status_instance(rng=rng, axes=axes, params=task_params, query_id=str(query_id), target_answer=int(target_answer), target_answer_support=target_answer_support, target_answer_probabilities=target_answer_probabilities, query_id_probabilities=query_id_probabilities)
            except ValueError:
                continue
            break
        if instance is None:
            raise RuntimeError(f'{self.task_id} failed to generate a valid Battleship ship-status scene after {max_attempts} attempts')
        rendered_context = render_battleship_sample(sample=instance.sample, style_variant=str(instance.axes.style_variant), params=task_params, instance_seed=int(instance_seed))
        annotation_projection = project_ship_status_annotation(ship_placements=instance.sample.ship_placements, annotation_ship_ids=instance.annotation_ship_ids, rendered_scene=rendered_context.rendered_scene)
        prompt_defaults, prompt_artifacts = build_battleship_prompt_artifacts(domain=self.domain, instance_seed=int(instance_seed), prompt_query_key=str(instance.query_id), dynamic_slots={})
        return _build_task_output(instance=instance, rendered_context=rendered_context, annotation_projection=annotation_projection, prompt_defaults=prompt_defaults, prompt_artifacts=prompt_artifacts)
__all__ = ['GamesBattleshipShipStatusCountTask', 'SHIP_STATUS_COUNT_QUERY_IDS']
