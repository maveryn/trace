"""Select the Battleship cell that completes the only not-yet-sunk ship."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.battleship.shared.annotations import project_point_set_annotation
from trace_tasks.tasks.games.battleship.shared.rules import fleet_shape_by_id, matching_fleet_shape_ids
from trace_tasks.tasks.games.battleship.shared.output import common_trace_sections, projected_annotation_payload, target_ship_cell_ids, witness_symbolic_payload
from trace_tasks.tasks.games.battleship.shared.rendering import render_battleship_sample
from trace_tasks.tasks.games.battleship.shared.sampling import ResolvedBattleshipSceneAxes, build_battleship_scene_state, build_ship_placements_with_hits, place_fleet, resolve_battleship_option_count, resolve_battleship_scene_axes, resolve_battleship_target_ship_id, sample_last_cell_candidate_options, sample_miss_coords
from trace_tasks.tasks.games.battleship.shared.state import LAST_CELL_OPTION_LABELS, SCENE_ID, SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS, BattleshipSample, Coord, sorted_coords
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice
from .shared.prompts import build_battleship_prompt_artifacts
TASK_ID = 'task_games__battleship__last_ship_cell_label'
LAST_SHIP_CELL_LABEL_QUERY_IDS: Tuple[str, ...] = ('last_ship_cell_label',)
LAST_SHIP_CELL_OPTION_COUNT_SUPPORT: Tuple[int, ...] = (4, 5, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults('games', SCENE_ID, task_id=TASK_ID)

@dataclass(frozen=True)
class LastShipCellInstance:
    """Task-owned hidden last-cell construction result."""
    query_id: str
    axes: ResolvedBattleshipSceneAxes
    sample: BattleshipSample
    answer: str
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
    target_missing_coord: Coord
    option_count: int
    option_count_support: Tuple[int, ...]
    option_count_probabilities: Dict[str, float]

def _resolve_label_index(*, instance_seed: int, params: Mapping[str, Any], option_count: int) -> tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve which visible option label receives the true missing cell."""
    label_index_support = tuple(range(int(option_count)))
    label_index_params = dict(params)
    label_index_params['last_ship_cell_label_index_support'] = [int(value) for value in label_index_support]
    target_answer, probabilities = resolve_integer_choice(instance_seed=int(instance_seed), params=label_index_params, gen_defaults=_GEN_DEFAULTS, support_key='last_ship_cell_label_index_support', explicit_key='target_answer', fallback_support=label_index_support, namespace='games.battleship.last_ship_cell.label_index', balanced_flag_key='balanced_target_answer_sampling', namespace_support_permutation=True)
    return (int(target_answer), tuple((int(value) for value in label_index_support)), dict(probabilities))

def _sample_last_ship_cell_instance(*, rng: Any, axes: ResolvedBattleshipSceneAxes, params: Mapping[str, Any], query_id: str, target_ship_id: str, label_index: int, label_index_support: Tuple[int, ...], label_index_probabilities: Dict[str, float], query_id_probabilities: Dict[str, float], target_ship_id_probabilities: Dict[str, float], option_count: int, option_count_support: Tuple[int, ...], option_count_probabilities: Dict[str, float]) -> LastShipCellInstance:
    """Construct one hidden last-cell visual-option instance."""
    base_placements = place_fleet(rng=rng, board_size=int(axes.board_size))
    target_base_ships = [ship for ship in base_placements if str(ship.ship_id) == str(target_ship_id)]
    if len(target_base_ships) != 1:
        raise ValueError('failed to resolve last-cell target ship')
    target_base_ship = target_base_ships[0]
    shuffled_target_coords = list(target_base_ship.coords)
    rng.shuffle(shuffled_target_coords)
    missing_coord = (int(shuffled_target_coords[0][0]), int(shuffled_target_coords[0][1]))
    hit_coords_by_ship_id: dict[str, Tuple[Coord, ...]] = {}
    for ship in base_placements:
        if str(ship.ship_id) == str(target_ship_id):
            hit_coords_by_ship_id[str(ship.ship_id)] = sorted_coords((coord for coord in ship.coords if coord != missing_coord))
        else:
            hit_coords_by_ship_id[str(ship.ship_id)] = tuple(ship.coords)
    placements = build_ship_placements_with_hits(base_placements, hit_coords_by_ship_id=hit_coords_by_ship_id)
    target_ships = [ship for ship in placements if str(ship.ship_id) == str(target_ship_id)]
    if len(target_ships) != 1:
        raise ValueError('failed to resolve hit-applied last-cell target ship')
    target_ship = target_ships[0]
    hit_coords = sorted_coords((coord for ship in placements for coord in ship.hit_coords))
    ship_cells = {coord for ship in placements for coord in ship.coords}
    candidate_options = sample_last_cell_candidate_options(rng=rng, board_size=int(axes.board_size), target_ship=target_ship, missing_coord=missing_coord, ship_cells=set(ship_cells), hit_coords=hit_coords, answer_label_index=int(label_index), option_count=int(option_count))
    answer_option = next((option for option in candidate_options if bool(option.is_answer)))
    miss_coords = sample_miss_coords(rng=rng, board_size=int(axes.board_size), occupied_ship_coords=ship_cells, excluded_coords=[option.coord for option in candidate_options], params=params, gen_defaults=_GEN_DEFAULTS)
    sample = build_battleship_scene_state(board_size=int(axes.board_size), scene_variant=str(axes.scene_variant), placements=placements, miss_coords=miss_coords, construction_mode='placed_fleet_with_hidden_last_ship_cell', candidate_options=candidate_options)
    _validate_last_cell_instance(sample=sample, target_ship_id=str(target_ship_id), missing_coord=missing_coord, answer_label=str(answer_option.label))
    shape = fleet_shape_by_id()[str(target_ship_id)]
    return LastShipCellInstance(query_id=str(query_id), axes=axes, sample=sample, answer=str(answer_option.label), annotation_coords=(missing_coord,), annotation_ship_ids=(str(target_ship.ship_id),), target_answer=int(LAST_CELL_OPTION_LABELS.index(str(answer_option.label))), target_answer_support=tuple((int(value) for value in label_index_support)), target_answer_probabilities=dict(label_index_probabilities), query_id_probabilities=dict(query_id_probabilities), target_ship_id=str(target_ship_id), target_ship_display_name=str(shape.display_name), target_ship_shape_id=str(target_ship_id), target_ship_id_probabilities=dict(target_ship_id_probabilities), target_missing_coord=missing_coord, option_count=int(option_count), option_count_support=tuple((int(value) for value in option_count_support)), option_count_probabilities=dict(option_count_probabilities))

def _validate_last_cell_instance(*, sample: BattleshipSample, target_ship_id: str, missing_coord: Coord, answer_label: str) -> None:
    """Validate the task-specific hidden last-cell semantics."""
    target_ships = [ship for ship in sample.ship_placements if str(ship.ship_id) == str(target_ship_id)]
    if len(target_ships) != 1:
        raise ValueError('Battleship last-cell task must name one target ship')
    target_ship = target_ships[0]
    target_coords = set(target_ship.coords)
    if missing_coord not in target_coords:
        raise ValueError('Battleship missing coord must belong to target ship')
    if set(target_ship.hit_coords) != target_coords - {missing_coord}:
        raise ValueError('Battleship last-cell target must have exactly one unhit cell')
    for ship in sample.ship_placements:
        if str(ship.ship_id) != str(target_ship.ship_id) and (not bool(ship.is_sunk)):
            raise ValueError('Battleship last-cell task requires every non-target ship to be sunk')
    answer_options = [option for option in sample.candidate_options if bool(option.is_answer)]
    if len(answer_options) != 1:
        raise ValueError('Battleship last-cell task must have exactly one answer option')
    answer_option = answer_options[0]
    if str(answer_option.label) != str(answer_label):
        raise ValueError('Battleship last-cell answer label mismatch')
    if (int(answer_option.coord[0]), int(answer_option.coord[1])) != missing_coord:
        raise ValueError('Battleship last-cell answer option must mark the missing coord')
    target_hits = set(target_ship.hit_coords)
    valid_candidate_labels = []
    for option in sample.candidate_options:
        completed = target_hits | {(int(option.coord[0]), int(option.coord[1]))}
        if str(target_ship.shape_id) in matching_fleet_shape_ids(tuple(completed)):
            valid_candidate_labels.append(str(option.label))
    if valid_candidate_labels != [str(answer_label)]:
        raise ValueError('Battleship last-cell candidate set must have exactly one shape-valid answer')

def _build_task_output(*, instance: LastShipCellInstance, rendered_context: Any, annotation_projection: Any, prompt_defaults: Mapping[str, Any], prompt_artifacts: Any) -> TaskOutput:
    """Build final TaskOutput for this public task."""
    answer_gt = TypedValue(type='string', value=str(instance.answer))
    if len(annotation_projection.annotation_points) != 1:
        raise RuntimeError('Battleship last-cell scalar annotation must contain exactly one point')
    annotation_gt = TypedValue(type='point', value=list(annotation_projection.annotation_points[0]))
    trace_payload = common_trace_sections(axes=instance.axes, sample=instance.sample, rendered_context=rendered_context, annotation_projection=annotation_projection)
    trace_payload['scene_ir']['relations'].update({'query_id': str(instance.query_id), 'target_answer': int(instance.target_answer), 'target_ship_id': str(instance.target_ship_id), 'target_ship_display_name': str(instance.target_ship_display_name), 'target_ship_shape_id': str(instance.target_ship_shape_id), 'target_cell_status': 'missing', 'target_missing_coord': [int(instance.target_missing_coord[0]), int(instance.target_missing_coord[1])]})
    trace_payload['query_spec'] = build_prompt_query_spec(prompt_artifacts=prompt_artifacts, query_id=str(instance.query_id), params={'scene_variant': str(instance.axes.scene_variant), 'query_id': str(instance.query_id), 'style_variant': str(instance.axes.style_variant), 'board_size': int(instance.sample.board_size), 'scene_variant_probabilities': dict(instance.axes.scene_variant_probabilities), 'query_id_probabilities': dict(instance.query_id_probabilities), 'style_variant_probabilities': dict(instance.axes.style_variant_probabilities), 'board_size_probabilities': dict(instance.axes.board_size_probabilities), 'target_answer': int(instance.target_answer), 'target_answer_support': [int(value) for value in instance.target_answer_support], 'target_answer_probabilities': dict(instance.target_answer_probabilities), 'target_ship_id': str(instance.target_ship_id), 'target_ship_display_name': str(instance.target_ship_display_name), 'target_ship_shape_id': str(instance.target_ship_shape_id), 'target_ship_id_probabilities': dict(instance.target_ship_id_probabilities), 'target_cell_status': 'missing', 'target_missing_coord': [int(instance.target_missing_coord[0]), int(instance.target_missing_coord[1])], 'candidate_labels': [str(option.label) for option in instance.sample.candidate_options], 'last_ship_cell_option_count': int(instance.option_count), 'last_ship_cell_option_count_support': [int(value) for value in instance.option_count_support], 'last_ship_cell_option_count_probabilities': dict(instance.option_count_probabilities), 'hit_count': len(instance.sample.hit_coords), 'miss_count': len(instance.sample.miss_coords), 'sunk_ship_count': int(instance.sample.sunk_ship_count), 'partial_ship_count': int(instance.sample.partial_ship_count), 'untouched_ship_count': int(instance.sample.untouched_ship_count)})
    trace_payload['execution_trace'].update({'query_id': str(instance.query_id), 'target_answer': int(instance.target_answer), 'target_answer_support': [int(value) for value in instance.target_answer_support], 'annotation_coords': [[int(row), int(col)] for row, col in instance.annotation_coords], 'annotation_ship_ids': [str(ship_id) for ship_id in instance.annotation_ship_ids], 'target_ship_id': str(instance.target_ship_id), 'target_ship_display_name': str(instance.target_ship_display_name), 'target_ship_shape_id': str(instance.target_ship_shape_id), 'target_ship_cell_ids': target_ship_cell_ids(instance.sample, target_ship_id=str(instance.target_ship_id)), 'target_cell_status': 'missing', 'target_missing_coord': [int(instance.target_missing_coord[0]), int(instance.target_missing_coord[1])]})
    trace_payload['witness_symbolic'] = witness_symbolic_payload(annotation_gt=annotation_gt, annotation_projection=annotation_projection)
    trace_payload['projected_annotation'] = projected_annotation_payload(annotation_gt=annotation_gt, annotation_projection=annotation_projection)
    return TaskOutput(prompt=str(prompt_artifacts.prompt), prompt_variants=dict(prompt_artifacts.prompt_variants), answer_gt=answer_gt, annotation_gt=annotation_gt, image=rendered_context.image, image_id='img0', trace_payload=trace_payload, task_versions=default_task_versions(), scene_id=SCENE_ID, query_id=str(instance.query_id))

@register_task
class GamesBattleshipLastShipCellLabelTask:
    """Select the labeled candidate cell that completes the remaining ship."""
    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = 'games'
    default_dataset_enabled = True
    supported_query_ids = LAST_SHIP_CELL_LABEL_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate the hidden last-cell option task while preserving one valid answer."""
        query_id, query_id_probabilities, task_params = select_task_query_id(instance_seed=int(instance_seed), params=params, supported_query_ids=self.supported_query_ids, default_query_id=self.supported_query_ids[0], task_id=self.task_id, namespace=f'{self.task_id}.query')
        axes = resolve_battleship_scene_axes(int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS)
        target_ship_id, target_ship_id_probabilities = resolve_battleship_target_ship_id(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS, namespace='games.battleship.last_ship_cell.target_ship_id', supported_shape_ids=SUPPORTED_BATTLESHIP_TARGET_SHIP_IDS)
        option_count, option_count_support, option_count_probabilities = resolve_battleship_option_count(instance_seed=int(instance_seed), params=task_params, gen_defaults=_GEN_DEFAULTS, namespace='games.battleship.last_ship_cell.option_count', support_key='last_ship_cell_option_count_support', explicit_key='last_ship_cell_option_count', fallback_support=LAST_SHIP_CELL_OPTION_COUNT_SUPPORT, balanced_flag_key='balanced_last_ship_cell_option_count_sampling')
        label_index, label_index_support, label_index_probabilities = _resolve_label_index(instance_seed=int(instance_seed), params=task_params, option_count=int(option_count))
        instance = None
        for attempt_index in range(max(1, int(max_attempts))):
            rng = spawn_rng(int(instance_seed), f'{self.task_id}.attempt.{int(attempt_index)}')
            try:
                instance = _sample_last_ship_cell_instance(rng=rng, axes=axes, params=task_params, query_id=str(query_id), target_ship_id=str(target_ship_id), label_index=int(label_index), label_index_support=label_index_support, label_index_probabilities=label_index_probabilities, query_id_probabilities=query_id_probabilities, target_ship_id_probabilities=target_ship_id_probabilities, option_count=int(option_count), option_count_support=option_count_support, option_count_probabilities=option_count_probabilities)
            except ValueError:
                continue
            break
        if instance is None:
            raise RuntimeError(f'{self.task_id} failed to generate a valid Battleship last-cell scene after {max_attempts} attempts')
        rendered_context = render_battleship_sample(sample=instance.sample, style_variant=str(instance.axes.style_variant), params=task_params, instance_seed=int(instance_seed))
        annotation_projection = project_point_set_annotation(annotation_coords=instance.annotation_coords, rendered_scene=rendered_context.rendered_scene)
        prompt_defaults, prompt_artifacts = build_battleship_prompt_artifacts(domain=self.domain, instance_seed=int(instance_seed), prompt_query_key=str(instance.query_id), dynamic_slots={})
        return _build_task_output(instance=instance, rendered_context=rendered_context, annotation_projection=annotation_projection, prompt_defaults=prompt_defaults, prompt_artifacts=prompt_artifacts)
__all__ = ['GamesBattleshipLastShipCellLabelTask', 'LAST_SHIP_CELL_LABEL_QUERY_IDS']
