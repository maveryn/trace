"""Identify the first labeled Pac-Man bonus item on the route."""

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

from ._lifecycle import AttemptPacmanResult, ObjectivePacmanPlan, run_pacman_lifecycle
from .shared.annotations import point_for_entity_id
from .shared.defaults import DEFAULTS, PACMAN_ITEM_KINDS, PACMAN_ITEM_LABELS, SCENE_ID
from .shared.sampling import (
    available_open_cells,
    expand_open_cells,
    resolve_pacman_count_axis,
    resolve_pacman_label_target,
    sample_decorative_ghosts,
    sample_route,
    wall_cells,
)
from .shared.state import PacmanItem, PacmanSceneState, item_entity_id, sorted_coords


TASK_ID = "task_games__pacman__next_item_label"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "next_item_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for first-item label output."""

    answer_value = "D"
    annotation_value = [509, 305]
    return (
        json.dumps({"annotation": annotation_value, "answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
    )


def _prepare_next_item_label_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
) -> ObjectivePacmanPlan:
    """Resolve the target item label and bind the route-order constructor."""

    item_count_axis = resolve_pacman_count_axis(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="item_count_support",
        explicit_key="item_count",
        fallback_support=DEFAULTS.item_count_support,
        namespace=f"{TASK_ID}.item_count",
        balanced_flag_key="balanced_item_count_sampling",
    )
    label_axis = resolve_pacman_label_target(
        int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="next_item_label_support",
        explicit_key="target_label",
        fallback_support=DEFAULTS.next_item_label_support,
        namespace=f"{TASK_ID}.target_label",
        balanced_flag_key="balanced_target_label_sampling",
    )
    target_label = str(label_axis.target_label)
    item_count = max(int(item_count_axis.value), PACMAN_ITEM_LABELS.index(target_label) + 1)

    def construct_attempt(rng: Any, axes: Any) -> AttemptPacmanResult:
        return _construct_next_item_label_attempt(
            rng=rng,
            axes=axes,
            target_label=target_label,
            item_count=item_count,
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePacmanPlan(
        attempt_namespace="games.pacman.next_item_label",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint='set "answer" to the single option letter of the first labeled bonus item reached along the highlighted route',
        annotation_hint='set "annotation" to one [x, y] point coordinate at the center of the first labeled bonus item reached along the route',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "target_label": target_label,
            "target_label_support": [str(value) for value in label_axis.target_label_support],
            "target_label_probabilities": dict(label_axis.target_label_probabilities),
            "item_count": int(item_count),
            "item_count_support": [int(value) for value in item_count_axis.support],
            "item_count_probabilities": dict(item_count_axis.probabilities),
        },
        construct_attempt=construct_attempt,
    )


def _construct_next_item_label_attempt(*, rng: Any, axes: Any, target_label: str, item_count: int) -> AttemptPacmanResult:
    """Construct a route where the requested label is the first reached bonus item."""

    rows, cols = int(axes.row_count), int(axes.col_count)
    route_len = min(max(int(item_count) + 5, 10), max(10, (rows - 2) * (cols - 2) - 2))
    route = sample_route(rng=rng, rows=rows, cols=cols, length=route_len)
    min_open = len(route) + int(item_count) + int(rng.randint(10, 18))
    open_cells = expand_open_cells(rng=rng, rows=rows, cols=cols, route_coords=route, min_open_cells=min_open)
    labels = tuple(PACMAN_ITEM_LABELS[: int(item_count)])
    label_to_coord: Dict[str, Any] = {str(target_label): tuple(route[2])}
    item_kind_cycle = cycle(shuffled_support(rng, PACMAN_ITEM_KINDS))
    label_to_kind: Dict[str, str] = {str(target_label): str(next(item_kind_cycle))}

    later_route_cells = list(route[4:])
    off_route_cells = list(available_open_cells(open_cells, excluded=route))
    rng.shuffle(later_route_cells)
    rng.shuffle(off_route_cells)
    used = {tuple(label_to_coord[str(target_label)])}
    later_cursor = 0
    off_cursor = 0
    for label in labels:
        label = str(label)
        if label == str(target_label):
            continue
        use_later_route = later_cursor < len(later_route_cells) and rng.random() < 0.48
        if use_later_route:
            coord = tuple(later_route_cells[later_cursor])
            later_cursor += 1
        else:
            if off_cursor >= len(off_route_cells):
                if later_cursor >= len(later_route_cells):
                    raise ValueError("not enough open cells for labeled items")
                coord = tuple(later_route_cells[later_cursor])
                later_cursor += 1
            else:
                coord = tuple(off_route_cells[off_cursor])
                off_cursor += 1
        if coord in used:
            raise ValueError("duplicate Pac-Man item coordinate")
        used.add(coord)
        label_to_coord[label] = coord
        label_to_kind[label] = str(next(item_kind_cycle))

    available_for_pellets = list(available_open_cells(open_cells, excluded=tuple(route) + tuple(used)))
    rng.shuffle(available_for_pellets)
    pellet_count = min(len(available_for_pellets), int(rng.randint(5, 9)))
    pellets = sorted_coords(available_for_pellets[:pellet_count])
    items = tuple(
        PacmanItem(
            label=str(label),
            item_id=item_entity_id(str(label)),
            coord=tuple(label_to_coord[str(label)]),
            kind=str(label_to_kind[str(label)]),
            is_answer=bool(str(label) == str(target_label)),
        )
        for label in labels
    )
    ghosts = sample_decorative_ghosts(
        rng=rng,
        open_cells=open_cells,
        excluded=tuple(route) + tuple(pellets) + tuple(used),
        start_index=1,
    )
    answer_id = item_entity_id(str(target_label))
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
        construction_mode="first_labeled_bonus_on_highlighted_route",
    )
    return AttemptPacmanResult(
        scene=scene,
        answer_gt=TypedValue(type="string", value=str(target_label)),
        annotation_entity_ids=(answer_id,),
        build_annotation=lambda rendered: point_for_entity_id(rendered.rendered_scene, answer_id),
        execution_extra={"target_label": str(target_label), "target_answer": str(target_label)},
    )


@register_task
class GamesPacmanNextItemLabelTask:
    """Choose the first labeled bonus item reached on the highlighted route."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations', 'topology')
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
            prepare_objective=_prepare_next_item_label_objective,
        )


__all__ = ["GamesPacmanNextItemLabelTask"]
