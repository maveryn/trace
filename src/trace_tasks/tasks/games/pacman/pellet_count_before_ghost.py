"""Count Pac-Man route pellets before the first ghost."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import AttemptPacmanResult, ObjectivePacmanPlan, run_pacman_lifecycle
from .shared.annotations import point_set_map_for_entity_ids
from .shared.defaults import DEFAULTS, PACMAN_GHOST_COLOR_KEYS, SCENE_ID
from .shared.sampling import (
    available_open_cells,
    expand_open_cells,
    resolve_pacman_integer_target,
    sample_decorative_ghosts,
    sample_route,
    wall_cells,
)
from .shared.state import PacmanGhost, PacmanSceneState, ghost_entity_id, pellet_entity_id, sorted_coords


TASK_ID = "task_games__pacman__pellet_count_before_ghost"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "pellet_count_before_ghost"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for pellet-before-ghost output."""

    answer_value = 4
    annotation_value = {
        "counted_pellets": [[355, 217], [415, 277], [475, 277], [535, 277]],
        "first_ghost": [[548, 278]],
    }
    return (
        json.dumps({"annotation": annotation_value, "answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
    )


def _prepare_pellet_count_before_ghost_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
) -> ObjectivePacmanPlan:
    """Resolve the target prefix pellet count and bind the stop-ghost constructor."""

    target_axis = resolve_pacman_integer_target(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="pellet_count_before_ghost_support",
        fallback_support=DEFAULTS.pellet_count_before_ghost_support,
        namespace=f"{TASK_ID}.target_answer",
    )
    target = int(target_axis.target_answer)

    def construct_attempt(rng: Any, axes: Any) -> AttemptPacmanResult:
        return _construct_pellet_count_before_ghost_attempt(rng=rng, axes=axes, target=target)

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePacmanPlan(
        attempt_namespace="games.pacman.pellet_count_before_ghost",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint='set "answer" to the number of normal pellets reached before the first ghost on the highlighted route',
        annotation_hint='set "annotation" to {"counted_pellets":[[x, y], ...], "first_ghost":[[x, y]]}, using center points for pellets before the ghost and the first route ghost',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "target_answer": int(target_axis.target_answer),
            "target_answer_support": [int(value) for value in target_axis.target_answer_support],
            "target_answer_probabilities": dict(target_axis.target_answer_probabilities),
        },
        construct_attempt=construct_attempt,
    )


def _construct_pellet_count_before_ghost_attempt(*, rng: Any, axes: Any, target: int) -> AttemptPacmanResult:
    """Construct a route with target pellets before the first route ghost."""

    rows, cols = int(axes.row_count), int(axes.col_count)
    route_len = min(max(int(target) + int(rng.randint(5, 8)), int(target) + 3), max(8, (rows - 2) * (cols - 2) - 2))
    route = sample_route(rng=rng, rows=rows, cols=cols, length=route_len)
    min_open = len(route) + int(target) + int(rng.randint(10, 18))
    open_cells = expand_open_cells(rng=rng, rows=rows, cols=cols, route_coords=route, min_open_cells=min_open)
    if len(route) <= int(target) + 1:
        raise ValueError("route too short for ghost stop query")
    counted_pellets = tuple(tuple(coord) for coord in route[1 : int(target) + 1])
    stop_coord = tuple(route[int(target) + 1])
    after_route = list(route[int(target) + 2 :])
    rng.shuffle(after_route)
    after_count = min(len(after_route), int(rng.randint(2, 5)))
    off_route = list(available_open_cells(open_cells, excluded=tuple(route)))
    rng.shuffle(off_route)
    off_route_count = min(len(off_route), int(rng.randint(3, 7)))
    pellets = sorted_coords(tuple(counted_pellets) + tuple(after_route[:after_count]) + tuple(off_route[:off_route_count]))
    stop_ghost = PacmanGhost(
        ghost_id=ghost_entity_id("route_stop"),
        coord=stop_coord,
        color_key=str(PACMAN_GHOST_COLOR_KEYS[0]),
        is_stop_ghost=True,
    )
    decorative_ghosts = sample_decorative_ghosts(
        rng=rng,
        open_cells=open_cells,
        excluded=tuple(route) + tuple(pellets) + (stop_coord,),
        start_index=1,
        min_count=1,
        max_count=3,
    )
    ghosts = (stop_ghost,) + decorative_ghosts
    counted_ids = tuple(pellet_entity_id(coord) for coord in counted_pellets)
    ghost_id = str(stop_ghost.ghost_id)
    annotation_ids = tuple(counted_ids) + (ghost_id,)
    keyed_ids = {"counted_pellets": counted_ids, "first_ghost": (ghost_id,)}
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
        items=tuple(),
        ghosts=ghosts,
        construction_mode="count_route_pellets_before_first_ghost",
    )
    return AttemptPacmanResult(
        scene=scene,
        answer_gt=TypedValue(type="integer", value=int(target)),
        annotation_entity_ids=annotation_ids,
        build_annotation=lambda rendered: point_set_map_for_entity_ids(rendered.rendered_scene, keyed_ids),
        execution_extra={"target_answer": int(target), "annotation_entity_id_map": {key: list(value) for key, value in keyed_ids.items()}},
    )


@register_task
class GamesPacmanPelletCountBeforeGhostTask:
    """Count route pellets before the first ghost is reached."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'spatial_relations', 'topology')
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
            prepare_objective=_prepare_pellet_count_before_ghost_objective,
        )


__all__ = ["GamesPacmanPelletCountBeforeGhostTask"]
