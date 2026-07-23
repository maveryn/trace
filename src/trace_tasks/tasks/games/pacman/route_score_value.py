"""Compute the Pac-Man score along a highlighted route."""

from __future__ import annotations

import json
from itertools import cycle
from typing import Any, Dict, Mapping

from trace_tasks.core.sampling import shuffled_support
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support

from ._lifecycle import AttemptPacmanResult, ObjectivePacmanPlan, run_pacman_lifecycle
from .shared.annotations import point_set_for_entity_ids
from .shared.defaults import DEFAULTS, PACMAN_ITEM_KINDS, PACMAN_ITEM_LABELS, SCENE_ID
from .shared.sampling import (
    available_open_cells,
    expand_open_cells,
    sample_decorative_ghosts,
    sample_route,
    wall_cells,
)
from .shared.state import PacmanItem, PacmanSceneState, item_entity_id, pellet_entity_id, sorted_coords


TASK_ID = "task_games__pacman__route_score_value"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "route_score_value"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for route-score output."""

    answer_value = 12
    annotation_value = [[315, 209], [369, 209], [424, 264]]
    return (
        json.dumps({"annotation": annotation_value, "answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
    )


def _prepare_route_score_objective(
    _instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
) -> ObjectivePacmanPlan:
    """Resolve route-score supports and bind the score constructor."""

    on_route_pellet_support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="route_score_on_route_pellet_count_support",
        fallback=DEFAULTS.route_score_on_route_pellet_count_support,
    )
    on_route_bonus_support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="route_score_on_route_bonus_count_support",
        fallback=DEFAULTS.route_score_on_route_bonus_count_support,
    )
    off_route_bonus_support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="route_score_off_route_bonus_count_support",
        fallback=DEFAULTS.route_score_off_route_bonus_count_support,
    )
    bonus_value_support = resolve_integer_support(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="route_score_bonus_value_support",
        fallback=DEFAULTS.route_score_bonus_value_support,
    )

    def construct_attempt(rng: Any, axes: Any) -> AttemptPacmanResult:
        return _construct_route_score_attempt(
            rng=rng,
            axes=axes,
            on_route_pellet_support=on_route_pellet_support,
            on_route_bonus_support=on_route_bonus_support,
            off_route_bonus_support=off_route_bonus_support,
            bonus_value_support=bonus_value_support,
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePacmanPlan(
        attempt_namespace="games.pacman.route_score_value",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint='set "answer" to the integer total score from collectibles on the highlighted route',
        annotation_hint='set "annotation" to [x, y] pixel points at the centers of each normal pellet or printed-value bonus item included in the route score',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "route_score_on_route_pellet_count_support": [int(value) for value in on_route_pellet_support],
            "route_score_on_route_bonus_count_support": [int(value) for value in on_route_bonus_support],
            "route_score_off_route_bonus_count_support": [int(value) for value in off_route_bonus_support],
            "route_score_bonus_value_support": [int(value) for value in bonus_value_support],
        },
        construct_attempt=construct_attempt,
    )


def _construct_route_score_attempt(
    *,
    rng: Any,
    axes: Any,
    on_route_pellet_support: tuple[int, ...],
    on_route_bonus_support: tuple[int, ...],
    off_route_bonus_support: tuple[int, ...],
    bonus_value_support: tuple[int, ...],
) -> AttemptPacmanResult:
    """Construct a route with normal pellets and printed bonus scores."""

    rows, cols = int(axes.row_count), int(axes.col_count)
    on_route_pellet_count = int(rng.choice(tuple(on_route_pellet_support)))
    on_route_bonus_count = int(rng.choice(tuple(on_route_bonus_support)))
    max_bonus_count = len(PACMAN_ITEM_LABELS)
    on_route_bonus_count = min(on_route_bonus_count, max_bonus_count)
    off_route_bonus_count = int(rng.choice(tuple(off_route_bonus_support)))
    off_route_bonus_count = max(1, min(off_route_bonus_count, max_bonus_count - on_route_bonus_count))

    route_collectible_count = int(on_route_pellet_count + on_route_bonus_count)
    route_len = min(
        max(route_collectible_count + int(rng.randint(5, 9)), 10),
        max(10, (rows - 2) * (cols - 2) - 2),
    )
    route = sample_route(rng=rng, rows=rows, cols=cols, length=route_len)
    min_open = len(route) + on_route_pellet_count + on_route_bonus_count + off_route_bonus_count + int(rng.randint(12, 20))
    open_cells = expand_open_cells(rng=rng, rows=rows, cols=cols, route_coords=route, min_open_cells=min_open)

    route_candidates = list(route[1:])
    if len(route_candidates) < route_collectible_count:
        raise ValueError("route too short for route score collectibles")
    rng.shuffle(route_candidates)
    on_route_bonus_coords = tuple(tuple(coord) for coord in route_candidates[:on_route_bonus_count])
    on_route_pellet_coords = tuple(tuple(coord) for coord in route_candidates[on_route_bonus_count:route_collectible_count])

    off_route_cells = list(available_open_cells(open_cells, excluded=tuple(route)))
    rng.shuffle(off_route_cells)
    if len(off_route_cells) < off_route_bonus_count:
        raise ValueError("not enough off-route cells for route score bonus distractors")
    off_route_bonus_coords = tuple(tuple(coord) for coord in off_route_cells[:off_route_bonus_count])

    used_item_coords = tuple(on_route_bonus_coords) + tuple(off_route_bonus_coords)
    available_for_pellets = list(available_open_cells(open_cells, excluded=tuple(route) + tuple(used_item_coords)))
    rng.shuffle(available_for_pellets)
    off_route_pellet_count = min(len(available_for_pellets), int(rng.randint(4, 9)))
    pellets = sorted_coords(tuple(on_route_pellet_coords) + tuple(available_for_pellets[:off_route_pellet_count]))

    labels = tuple(PACMAN_ITEM_LABELS[: int(on_route_bonus_count + off_route_bonus_count)])
    bonus_coords = tuple(on_route_bonus_coords) + tuple(off_route_bonus_coords)
    bonus_values = tuple(int(rng.choice(tuple(bonus_value_support))) for _ in labels)
    route_coord_set = {tuple(coord) for coord in route}
    item_kind_cycle = cycle(shuffled_support(rng, PACMAN_ITEM_KINDS))
    items = tuple(
        PacmanItem(
            label=str(label),
            item_id=item_entity_id(str(label)),
            coord=tuple(coord),
            kind=str(next(item_kind_cycle)),
            is_answer=tuple(coord) in route_coord_set,
            score_value=int(bonus_values[index]),
        )
        for index, (label, coord) in enumerate(zip(labels, bonus_coords))
    )
    route_order = {tuple(coord): index for index, coord in enumerate(route)}
    scored_entries = [
        (int(route_order[tuple(coord)]), pellet_entity_id(tuple(coord)))
        for coord in on_route_pellet_coords
    ]
    scored_entries.extend(
        (int(route_order[tuple(item.coord)]), item_entity_id(str(item.label)))
        for item in items
        if tuple(item.coord) in route_coord_set
    )
    annotation_ids = tuple(str(entity_id) for _index, entity_id in sorted(scored_entries, key=lambda pair: pair[0]))
    score_by_item_id = {item_entity_id(str(item.label)): int(item.score_value or 0) for item in items}
    answer = sum(1 for entity_id in annotation_ids if str(entity_id).startswith("pellet_r")) + sum(
        int(score_by_item_id[str(entity_id)])
        for entity_id in annotation_ids
        if str(entity_id).startswith("item_")
    )
    ghosts = sample_decorative_ghosts(
        rng=rng,
        open_cells=open_cells,
        excluded=tuple(route) + tuple(pellets) + tuple(used_item_coords),
        start_index=1,
        min_count=1,
        max_count=2,
    )
    scene = PacmanSceneState(
        row_count=rows,
        col_count=cols,
        scene_variant=str(axes.scene_variant),
        style_variant=str(axes.style_variant),
        open_cells=tuple(open_cells),
        wall_cells=wall_cells(rows=rows, cols=cols, open_cells=open_cells),
        pacman_coord=tuple(route[0]),
        route_coords=tuple(route),
        pellets=tuple(pellets),
        items=items,
        ghosts=ghosts,
        construction_mode="sum_route_collectible_scores",
    )
    return AttemptPacmanResult(
        scene=scene,
        answer_gt=TypedValue(type="integer", value=int(answer)),
        annotation_entity_ids=tuple(annotation_ids),
        build_annotation=lambda rendered: point_set_for_entity_ids(rendered.rendered_scene, annotation_ids),
        execution_extra={"target_answer": int(answer)},
    )


@register_task
class GamesPacmanRouteScoreValueTask:
    """Compute the score of collectibles on the highlighted route."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'topology', 'formula_evaluation')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_pacman_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_route_score_objective,
        )


__all__ = ["GamesPacmanRouteScoreValueTask"]
