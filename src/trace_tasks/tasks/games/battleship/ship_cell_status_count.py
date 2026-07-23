"""Count Battleship cells on one named ship matching a hit-status predicate."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.battleship.shared.annotations import project_bbox_set_annotation
from trace_tasks.tasks.games.battleship.shared.rules import cell_status_answer_support_for_ship, fleet_shape_by_id, ship_size_for_shape_id
from trace_tasks.tasks.games.battleship.shared.output import common_trace_sections, projected_annotation_payload, target_ship_cell_ids, witness_symbolic_payload
from trace_tasks.tasks.games.battleship.shared.rendering import render_battleship_sample
from trace_tasks.tasks.games.battleship.shared.sampling import ResolvedBattleshipSceneAxes, build_battleship_scene_state, build_ship_placements_with_hits, place_fleet, resolve_battleship_scene_axes, resolve_battleship_target_ship_id, sample_miss_coords
from trace_tasks.tasks.games.battleship.shared.state import SCENE_ID, SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS, BattleshipSample, Coord, sorted_coords
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice
from .shared.prompts import build_battleship_prompt_artifacts
TASK_ID = 'task_games__battleship__ship_cell_status_count'
SHIP_CELL_STATUS_COUNT_QUERY_IDS: Tuple[str, ...] = ('named_ship_hit_cell_count', 'named_ship_unhit_cell_count')
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults('games', SCENE_ID, task_id=TASK_ID)

@dataclass(frozen=True)
class ShipCellStatusInstance:
    """Task-owned named-ship cell-status construction result."""
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
    target_ship_id: str
    target_ship_display_name: str
    target_ship_shape_id: str
    target_ship_id_probabilities: Dict[str, float]
    target_cell_status: str

def _target_status_for_branch(query_id: str) -> str:
    """Translate this task's public branch into a semantic cell status."""
    return {'named_ship_hit_cell_count': 'hit', 'named_ship_unhit_cell_count': 'unhit'}[str(query_id)]

def _select_ship_cell_status_query(instance_seed: int, params: Mapping[str, Any]) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Resolve this task's internal query branch at the public task boundary."""
    return select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=SHIP_CELL_STATUS_COUNT_QUERY_IDS, default_query_id=SHIP_CELL_STATUS_COUNT_QUERY_IDS[0], task_id=TASK_ID, namespace=f'{TASK_ID}.query')

def _cell_status_pair_options() -> Tuple[Tuple[str, int], ...]:
    """Return feasible `(target_ship_id, answer)` options for this task."""
    pairs: list[Tuple[str, int]] = []
    for shape_id in SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS:
        for answer in cell_status_answer_support_for_ship(str(shape_id)):
            pairs.append((str(shape_id), int(answer)))
    return tuple(pairs)

def _cell_status_pair_index_support() -> Tuple[int, ...]:
    """Return integer support over feasible target-ship/answer pairs."""
    return tuple(range(len(_cell_status_pair_options())))

def _cell_status_target_ship_probabilities(pair_options: Tuple[Tuple[str, int], ...]) -> Dict[str, float]:
    """Return marginal target-ship probabilities for pair sampling."""
    total = max(1, len(pair_options))
    return {str(shape_id): float(sum((1 for item, _answer in pair_options if str(item) == str(shape_id))) / total) for shape_id in SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS}

def _cell_status_answer_probabilities(pair_options: Tuple[Tuple[str, int], ...]) -> Dict[str, float]:
    """Return marginal answer probabilities for pair sampling."""
    answers = sorted({int(answer) for _shape_id, answer in pair_options})
    total = max(1, len(pair_options))
    return {str(answer): float(sum((1 for _shape_id, item in pair_options if int(item) == int(answer))) / total) for answer in answers}

def _resolve_target_ship_and_answer(*, instance_seed: int, params: Mapping[str, Any], target_cell_status: str) -> tuple[str, int, Tuple[int, ...], Dict[str, float], Dict[str, float]]:
    """Resolve the named ship and target answer for this task."""
    explicit_target_ship_id = params.get('target_ship_id')
    explicit_target_answer = params.get('target_answer')
    pair_options = _cell_status_pair_options()
    if explicit_target_ship_id is None and explicit_target_answer is None:
        pair_index, _pair_probabilities = resolve_integer_choice(instance_seed=int(instance_seed), params=params, gen_defaults=_GEN_DEFAULTS, support_key='target_ship_answer_pair_index_support', explicit_key='target_ship_answer_pair_index', fallback_support=_cell_status_pair_index_support(), namespace=f'{TASK_ID}.target_ship_answer_pair', balanced_flag_key='balanced_target_ship_answer_pair_sampling', namespace_support_permutation=True)
        target_ship_id, target_answer = pair_options[int(pair_index)]
        return (str(target_ship_id), int(target_answer), tuple(sorted({int(answer) for _shape_id, answer in pair_options})), _cell_status_answer_probabilities(pair_options), _cell_status_target_ship_probabilities(pair_options))
    if explicit_target_answer is None:
        compatible_target_ship_ids = SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS
    else:
        selected_answer = int(explicit_target_answer)
        compatible_target_ship_ids = tuple((str(shape_id) for shape_id in SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS if int(selected_answer) in set(cell_status_answer_support_for_ship(str(shape_id)))))
        if not compatible_target_ship_ids:
            raise ValueError(f'unsupported Battleship target_answer for named-ship cell status: {selected_answer}')
    target_ship_id, target_ship_id_probabilities = resolve_battleship_target_ship_id(instance_seed=int(instance_seed), params=params, gen_defaults=_GEN_DEFAULTS, namespace='games.battleship.ship_cell_status.target_ship_id', supported_shape_ids=tuple((str(item) for item in compatible_target_ship_ids)))
    cell_count_support = cell_status_answer_support_for_ship(str(target_ship_id))
    target_answer, target_answer_probabilities = resolve_integer_choice(instance_seed=int(instance_seed), params=params, gen_defaults=_GEN_DEFAULTS, support_key=f'{str(target_ship_id)}_cell_count_support', explicit_key='target_answer', fallback_support=cell_count_support, namespace=f'games.battleship.ship_cell_status.{str(target_cell_status)}.{str(target_ship_id)}.target_answer', balanced_flag_key='balanced_target_answer_sampling', namespace_support_permutation=True)
    return (str(target_ship_id), int(target_answer), tuple((int(value) for value in cell_count_support)), dict(target_answer_probabilities), dict(target_ship_id_probabilities))

def _sample_ship_cell_status_instance(*, rng: Any, axes: ResolvedBattleshipSceneAxes, params: Mapping[str, Any], query_id: str, target_cell_status: str, target_ship_id: str, target_answer: int, target_answer_support: Tuple[int, ...], target_answer_probabilities: Dict[str, float], query_id_probabilities: Dict[str, float], target_ship_id_probabilities: Dict[str, float]) -> ShipCellStatusInstance:
    """Construct one target-matched named-ship cell-status instance."""
    target_ship_size = ship_size_for_shape_id(str(target_ship_id))
    if int(target_answer) > int(target_ship_size):
        raise ValueError(f'target answer {target_answer} exceeds target ship size {target_ship_size}')
    target_hit_count = int(target_answer) if str(target_cell_status) == 'hit' else int(target_ship_size) - int(target_answer)
    base_placements = place_fleet(rng=rng, board_size=int(axes.board_size))
    hit_coords_by_ship_id: dict[str, Tuple[Coord, ...]] = {}
    for ship in base_placements:
        shuffled = list(ship.coords)
        rng.shuffle(shuffled)
        if str(ship.ship_id) == str(target_ship_id):
            hit_count = int(target_hit_count)
        else:
            roll = float(rng.random())
            if roll < 0.3:
                hit_count = 0
            elif roll < 0.55:
                hit_count = len(ship.coords)
            else:
                hit_count = int(rng.randint(1, max(1, len(ship.coords) - 1)))
        hit_coords_by_ship_id[str(ship.ship_id)] = sorted_coords(shuffled[:hit_count])
    placements = build_ship_placements_with_hits(base_placements, hit_coords_by_ship_id=hit_coords_by_ship_id)
    ship_cells = {coord for ship in placements for coord in ship.coords}
    miss_coords = sample_miss_coords(rng=rng, board_size=int(axes.board_size), occupied_ship_coords=ship_cells, excluded_coords=tuple(), params=params, gen_defaults=_GEN_DEFAULTS)
    sample = build_battleship_scene_state(board_size=int(axes.board_size), scene_variant=str(axes.scene_variant), placements=placements, miss_coords=miss_coords, construction_mode='placed_fleet_with_named_ship_cell_status')
    target_ships = [ship for ship in sample.ship_placements if str(ship.ship_id) == str(target_ship_id)]
    if len(target_ships) != 1:
        raise ValueError('failed to resolve named Battleship target ship')
    target_ship = target_ships[0]
    if str(target_cell_status) == 'hit':
        annotation_coords = sorted_coords(target_ship.hit_coords)
    else:
        target_hit_coords = set(target_ship.hit_coords)
        annotation_coords = sorted_coords((coord for coord in target_ship.coords if coord not in target_hit_coords))
    answer = len(annotation_coords)
    if int(answer) != int(target_answer):
        raise ValueError('Battleship named-ship cell-status construction did not match target answer')
    shape = fleet_shape_by_id()[str(target_ship_id)]
    return ShipCellStatusInstance(query_id=str(query_id), axes=axes, sample=sample, answer=int(answer), annotation_coords=annotation_coords, annotation_ship_ids=(str(target_ship.ship_id),), target_answer=int(target_answer), target_answer_support=tuple((int(value) for value in target_answer_support)), target_answer_probabilities=dict(target_answer_probabilities), query_id_probabilities=dict(query_id_probabilities), target_ship_id=str(target_ship_id), target_ship_display_name=str(shape.display_name), target_ship_shape_id=str(target_ship_id), target_ship_id_probabilities=dict(target_ship_id_probabilities), target_cell_status=str(target_cell_status))

def _build_task_output(*, instance: ShipCellStatusInstance, rendered_context: Any, annotation_projection: Any, prompt_defaults: Mapping[str, Any], prompt_artifacts: Any) -> TaskOutput:
    """Build final TaskOutput for this public task."""
    answer_gt = TypedValue(type='integer', value=int(instance.answer))
    annotation_gt = TypedValue(type='bbox_set', value=[list(bbox) for bbox in annotation_projection.annotation_bboxes])
    trace_payload = common_trace_sections(axes=instance.axes, sample=instance.sample, rendered_context=rendered_context, annotation_projection=annotation_projection)
    trace_payload['scene_ir']['relations'].update({'query_id': str(instance.query_id), 'target_answer': int(instance.target_answer), 'target_ship_id': str(instance.target_ship_id), 'target_ship_display_name': str(instance.target_ship_display_name), 'target_ship_shape_id': str(instance.target_ship_shape_id), 'target_cell_status': str(instance.target_cell_status), 'target_missing_coord': None})
    trace_payload['query_spec'] = build_prompt_query_spec(prompt_artifacts=prompt_artifacts, query_id=str(instance.query_id), params={'scene_variant': str(instance.axes.scene_variant), 'query_id': str(instance.query_id), 'style_variant': str(instance.axes.style_variant), 'board_size': int(instance.sample.board_size), 'scene_variant_probabilities': dict(instance.axes.scene_variant_probabilities), 'query_id_probabilities': dict(instance.query_id_probabilities), 'style_variant_probabilities': dict(instance.axes.style_variant_probabilities), 'board_size_probabilities': dict(instance.axes.board_size_probabilities), 'target_answer': int(instance.target_answer), 'target_answer_support': [int(value) for value in instance.target_answer_support], 'target_answer_probabilities': dict(instance.target_answer_probabilities), 'target_ship_id': str(instance.target_ship_id), 'target_ship_display_name': str(instance.target_ship_display_name), 'target_ship_shape_id': str(instance.target_ship_shape_id), 'target_ship_id_probabilities': dict(instance.target_ship_id_probabilities), 'target_cell_status': str(instance.target_cell_status), 'target_missing_coord': None, 'candidate_labels': [], 'last_ship_cell_option_count': 0, 'last_ship_cell_option_count_support': [], 'last_ship_cell_option_count_probabilities': {}, 'hit_count': len(instance.sample.hit_coords), 'miss_count': len(instance.sample.miss_coords), 'sunk_ship_count': int(instance.sample.sunk_ship_count), 'partial_ship_count': int(instance.sample.partial_ship_count), 'untouched_ship_count': int(instance.sample.untouched_ship_count)})
    trace_payload['execution_trace'].update({'query_id': str(instance.query_id), 'target_answer': int(instance.target_answer), 'target_answer_support': [int(value) for value in instance.target_answer_support], 'annotation_coords': [[int(row), int(col)] for row, col in instance.annotation_coords], 'annotation_ship_ids': [str(ship_id) for ship_id in instance.annotation_ship_ids], 'target_ship_id': str(instance.target_ship_id), 'target_ship_display_name': str(instance.target_ship_display_name), 'target_ship_shape_id': str(instance.target_ship_shape_id), 'target_ship_cell_ids': target_ship_cell_ids(instance.sample, target_ship_id=str(instance.target_ship_id)), 'target_cell_status': str(instance.target_cell_status), 'target_missing_coord': None})
    trace_payload['witness_symbolic'] = witness_symbolic_payload(annotation_gt=annotation_gt, annotation_projection=annotation_projection)
    trace_payload['projected_annotation'] = projected_annotation_payload(annotation_gt=annotation_gt, annotation_projection=annotation_projection)
    return TaskOutput(prompt=str(prompt_artifacts.prompt), prompt_variants=dict(prompt_artifacts.prompt_variants), answer_gt=answer_gt, annotation_gt=annotation_gt, image=rendered_context.image, image_id='img0', trace_payload=trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=str(instance.query_id))

@register_task
class GamesBattleshipShipCellStatusCountTask:
    """Count hit or unhit cells on the named fleet ship."""
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = SHIP_CELL_STATUS_COUNT_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a fixed named-ship cell-status query with retryable placement construction."""
        query_id, query_id_probabilities, task_params = _select_ship_cell_status_query(int(instance_seed), params)
        target_cell_status = _target_status_for_branch(str(query_id))
        axes = resolve_battleship_scene_axes(int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS)
        target_ship_id, target_answer, target_answer_support, target_answer_probabilities, target_ship_id_probabilities = _resolve_target_ship_and_answer(instance_seed=int(instance_seed), params=task_params, target_cell_status=str(target_cell_status))
        instance = None
        for attempt_index in range(max(1, int(max_attempts))):
            rng = spawn_rng(int(instance_seed), f'{self.task_id}.attempt.{int(attempt_index)}')
            try:
                instance = _sample_ship_cell_status_instance(rng=rng, axes=axes, params=task_params, query_id=str(query_id), target_cell_status=str(target_cell_status), target_ship_id=str(target_ship_id), target_answer=int(target_answer), target_answer_support=target_answer_support, target_answer_probabilities=target_answer_probabilities, query_id_probabilities=query_id_probabilities, target_ship_id_probabilities=target_ship_id_probabilities)
            except ValueError:
                continue
            break
        if instance is None:
            raise RuntimeError(f'{self.task_id} failed to generate a valid Battleship ship-cell-status scene after {max_attempts} attempts')
        rendered_context = render_battleship_sample(sample=instance.sample, style_variant=str(instance.axes.style_variant), params=task_params, instance_seed=int(instance_seed))
        annotation_projection = project_bbox_set_annotation(annotation_coords=instance.annotation_coords, rendered_scene=rendered_context.rendered_scene)
        prompt_defaults, prompt_artifacts = build_battleship_prompt_artifacts(domain=self.domain, instance_seed=int(instance_seed), prompt_query_key=str(instance.query_id), dynamic_slots={'target_ship_name': str(instance.target_ship_display_name)})
        return _build_task_output(instance=instance, rendered_context=rendered_context, annotation_projection=annotation_projection, prompt_defaults=prompt_defaults, prompt_artifacts=prompt_artifacts)
__all__ = ['GamesBattleshipShipCellStatusCountTask', 'SHIP_CELL_STATUS_COUNT_QUERY_IDS']
