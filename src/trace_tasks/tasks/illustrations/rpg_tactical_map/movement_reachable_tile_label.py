"""Select the tile reachable by a tactical RPG movement budget."""

from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.option_rendering import sample_visual_label_font_trace

from .shared.output import (
    bbox_projection,
    movement_reachable_render_map,
    rounded_bbox,
    rpg_tactical_map_render_spec,
    rpg_tactical_map_scene_ir,
)
from .shared.prompts import build_rpg_tactical_map_task_prompt_with_default_slots
from .shared.prompts import rpg_tactical_map_terrain_rules_text
from .shared.relations import (
    TERRAIN_MOVEMENT_COSTS,
    TERRAIN_WATER,
    shortest_movement_costs,
)
from .shared.rendering import (
    DEFAULT_CANDIDATE_LABELS,
    DEFAULT_TILE_PX,
    SCENE_ID,
    render_rpg_tactical_map_scene,
    resolve_tactical_map_render_params,
)
from .shared.sampling import select_int_from_support
from .shared.state import RpgTacticalMapScene, RpgTacticalTile


TASK_ID = "task_illustrations__rpg_tactical_map__movement_reachable_tile_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "movement_reachable_tile"
DEFAULT_MOVEMENT_BUDGET_SUPPORT: tuple[int, ...] = (4, 5, 6)
DEFAULT_CANDIDATE_COUNT = 4

_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _tiles_by_coord(scene: RpgTacticalMapScene) -> dict[tuple[int, int], RpgTacticalTile]:
    return {(int(tile.row), int(tile.col)): tile for tile in scene.tiles}


def _tile_by_id(scene: RpgTacticalMapScene) -> dict[str, RpgTacticalTile]:
    return {str(tile.tile_id): tile for tile in scene.tiles}


def _tile_manhattan(first: RpgTacticalTile, second: RpgTacticalTile) -> int:
    return abs(int(first.row) - int(second.row)) + abs(int(first.col) - int(second.col))


def _direction_bucket(origin: RpgTacticalTile, tile: RpgTacticalTile) -> str:
    drow = int(tile.row) - int(origin.row)
    dcol = int(tile.col) - int(origin.col)
    row_part = "s" if drow > 0 else "n" if drow < 0 else ""
    col_part = "e" if dcol > 0 else "w" if dcol < 0 else ""
    return f"{row_part}{col_part}" or "same"


def _select_candidate_tiles(
    *,
    scene: RpgTacticalMapScene,
    movement_budget: int,
    candidate_count: int,
    instance_seed: int,
) -> tuple[dict[str, str], str, dict[str, int]]:
    """Select exactly one reachable candidate tile and unreachable distractors."""

    if not scene.units:
        raise ValueError("movement task requires a blue unit")
    start_tile_id = str(scene.units[0].tile_id)
    tiles_by_id = _tile_by_id(scene)
    start_tile = tiles_by_id[start_tile_id]
    movement_costs = shortest_movement_costs(_tiles_by_coord(scene), start_coord=start_tile.coord)
    reachable = [
        tile
        for tile in scene.tiles
        if str(tile.tile_id) != start_tile_id
        and bool(tile.passable)
        and str(tile.tile_id) in movement_costs
        and int(movement_costs[str(tile.tile_id)]) <= int(movement_budget)
    ]
    unreachable_passable = [
        tile
        for tile in scene.tiles
        if str(tile.tile_id) != start_tile_id
        and bool(tile.passable)
        and (str(tile.tile_id) not in movement_costs or int(movement_costs[str(tile.tile_id)]) > int(movement_budget))
    ]
    blocked = [
        tile
        for tile in scene.tiles
        if str(tile.tile_id) != start_tile_id and not bool(tile.passable)
    ]
    if not reachable:
        raise ValueError("no reachable candidate tile available")
    if len(unreachable_passable) + len(blocked) < int(candidate_count) - 1:
        raise ValueError("not enough unreachable candidate distractors")

    rng = random.Random(f"{int(instance_seed)}:movement_reachable_candidates:{int(movement_budget)}")
    tile_jitter = {str(tile.tile_id): float(rng.random()) for tile in scene.tiles}
    answer_pool = [
        tile
        for tile in reachable
        if int(movement_costs[str(tile.tile_id)]) >= min(2, int(movement_budget))
    ] or reachable

    invalid_tiles = list(unreachable_passable) + list(blocked)
    nearby_radius = max(3, min(5, int(movement_budget) + 1))

    def invalid_neighbor_count(tile: RpgTacticalTile) -> int:
        return sum(1 for invalid in invalid_tiles if _tile_manhattan(tile, invalid) <= nearby_radius)

    def answer_sort_key(tile: RpgTacticalTile) -> tuple[int, int, int, float]:
        tile_cost = int(movement_costs[str(tile.tile_id)])
        return (
            -tile_cost,
            -invalid_neighbor_count(tile),
            _tile_manhattan(start_tile, tile),
            tile_jitter[str(tile.tile_id)],
        )

    answer_tile = sorted(answer_pool, key=answer_sort_key)[0]

    reachable_cost_tiles = [tile for tile in scene.tiles if str(tile.tile_id) in movement_costs]

    def movement_proxy_cost(tile: RpgTacticalTile) -> int | None:
        cost = movement_costs.get(str(tile.tile_id))
        if cost is not None:
            return int(cost)
        if not reachable_cost_tiles:
            return None
        return min(
            int(movement_costs[str(reachable_tile.tile_id)]) + _tile_manhattan(tile, reachable_tile)
            for reachable_tile in reachable_cost_tiles
        )

    proxy_cost_by_tile_id = {
        str(tile.tile_id): movement_proxy_cost(tile)
        for tile in invalid_tiles
    }

    def plausible_invalid_tiles(tiles: Sequence[RpgTacticalTile]) -> list[RpgTacticalTile]:
        plausible: list[RpgTacticalTile] = []
        for tile in tiles:
            proxy_cost = proxy_cost_by_tile_id.get(str(tile.tile_id))
            if proxy_cost is None:
                continue
            if int(movement_budget) < int(proxy_cost) <= int(movement_budget) + 5:
                plausible.append(tile)
        return plausible

    def invalid_sort_key(tile: RpgTacticalTile) -> tuple[int, int, int, float]:
        proxy_cost = proxy_cost_by_tile_id.get(str(tile.tile_id))
        blocked_penalty = 0 if bool(tile.passable) else 1
        return (
            abs(int(proxy_cost if proxy_cost is not None else int(movement_budget) + 99) - (int(movement_budget) + 1)),
            blocked_penalty,
            _tile_manhattan(start_tile, tile),
            tile_jitter[str(tile.tile_id)],
        )

    def spread_score(tile: RpgTacticalTile, selected: Sequence[RpgTacticalTile], used_buckets: set[str]) -> tuple[int, int, int, int, float]:
        min_distance = min(_tile_manhattan(tile, existing) for existing in selected)
        bucket = _direction_bucket(start_tile, tile)
        proxy_cost = proxy_cost_by_tile_id.get(str(tile.tile_id))
        cost_gap = abs(int(proxy_cost if proxy_cost is not None else int(movement_budget) + 99) - (int(movement_budget) + 1))
        blocked_penalty = 0 if bool(tile.passable) else 1
        return (
            1 if bucket not in used_buckets else 0,
            min_distance,
            -cost_gap,
            -blocked_penalty,
            -tile_jitter[str(tile.tile_id)],
        )

    def choose_spread_distractors(pool: Sequence[RpgTacticalTile], *, minimum_distance: int) -> list[RpgTacticalTile]:
        selected = [answer_tile]
        distractor_tiles: list[RpgTacticalTile] = []
        selected_tile_ids = {str(answer_tile.tile_id), start_tile_id}
        used_buckets = {_direction_bucket(start_tile, answer_tile)}
        remaining = sorted(pool, key=invalid_sort_key)
        while len(distractor_tiles) < int(candidate_count) - 1:
            eligible = [
                tile
                for tile in remaining
                if str(tile.tile_id) not in selected_tile_ids
                and min(_tile_manhattan(tile, existing) for existing in selected) >= int(minimum_distance)
            ]
            if not eligible:
                break
            chosen = max(eligible, key=lambda tile: spread_score(tile, selected, used_buckets))
            distractor_tiles.append(chosen)
            selected.append(chosen)
            selected_tile_ids.add(str(chosen.tile_id))
            used_buckets.add(_direction_bucket(start_tile, chosen))
            remaining = [tile for tile in remaining if str(tile.tile_id) != str(chosen.tile_id)]
        return distractor_tiles

    plausible_distractors = plausible_invalid_tiles(invalid_tiles)
    distractors: list[RpgTacticalTile] = []
    for pool in (plausible_distractors, invalid_tiles):
        for minimum_distance in (3, 2, 1):
            distractors = choose_spread_distractors(pool, minimum_distance=minimum_distance)
            if len(distractors) >= int(candidate_count) - 1:
                break
        if len(distractors) >= int(candidate_count) - 1:
            break
    if len(distractors) < int(candidate_count) - 1:
        raise ValueError("could not build enough candidate distractors")

    labels = list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)])
    selected_label = str(labels[int(instance_seed) % int(candidate_count)])
    candidates_by_label: dict[str, str] = {selected_label: str(answer_tile.tile_id)}
    distractor_labels = [str(label) for label in labels if str(label) != selected_label]
    for label, tile in zip(distractor_labels, distractors, strict=True):
        candidates_by_label[str(label)] = str(tile.tile_id)
    candidates_by_label = {str(label): str(candidates_by_label[str(label)]) for label in labels}
    return candidates_by_label, selected_label, movement_costs


@register_task
class IllustrationsRpgTacticalMapMovementReachableTileLabelTask:
    """Choose the lettered tile reachable by the blue unit's movement budget."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one option-selection instance with path-cost metadata."""

        resolved_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}:query",
        )
        movement_budget, movement_budget_probabilities = select_int_from_support(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            support_key="movement_budget_support",
            explicit_key="movement_budget",
            fallback_support=DEFAULT_MOVEMENT_BUDGET_SUPPORT,
            namespace=f"{TASK_ID}:movement_budget",
        )
        candidate_count = int(task_params.get("candidate_count", group_default(_GEN_DEFAULTS, "candidate_count", DEFAULT_CANDIDATE_COUNT)))
        if candidate_count != DEFAULT_CANDIDATE_COUNT:
            raise ValueError("rpg tactical map movement task currently supports exactly four candidate tiles")
        render_params = resolve_tactical_map_render_params(
            task_params,
            _RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:canvas_profile",
        )
        label_font_trace = sample_visual_label_font_trace(
            namespace_prefix=TASK_ID,
            instance_seed=int(instance_seed),
            params=task_params,
            namespace_suffix="candidate_labels",
            explicit_key="candidate_label_font_family",
            weights_key="candidate_label_font_family_weights",
        )
        _prompt_defaults, prompt_artifacts = build_rpg_tactical_map_task_prompt_with_default_slots(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults_source=_PROMPT_DEFAULTS,
            prompt_query_key=PROMPT_QUERY_KEY,
            answer_hint_key="answer_hint_movement_reachable_tile_label",
            annotation_hint_key="annotation_hint_movement_reachable_tile_label",
            json_example_key="json_example_movement_reachable_tile_label",
            json_example_answer_only_key="json_example_answer_only_movement_reachable_tile_label",
            context_label=TASK_ID,
            slots={
                "movement_points": str(int(movement_budget)),
                "terrain_rules": rpg_tactical_map_terrain_rules_text(),
            },
            instance_seed=int(instance_seed),
        )

        selected_scene: RpgTacticalMapScene | None = None
        selected_candidates: dict[str, str] | None = None
        selected_label: str | None = None
        selected_costs: dict[str, int] | None = None
        attempt_errors: list[str] = []
        for attempt in range(max(1, int(max_attempts))):
            scene_seed = int(instance_seed) + int(attempt) * 7919
            base_scene = render_rpg_tactical_map_scene(
                scene_seed,
                width=int(render_params["canvas_width"]),
                height=int(render_params["canvas_height"]),
                grid_cols=int(render_params["grid_cols"]),
                grid_rows=int(render_params["grid_rows"]),
                tile_px=int(render_params.get("tile_px", DEFAULT_TILE_PX)),
                render_metadata=render_params,
            )
            try:
                candidates, answer_label, movement_costs = _select_candidate_tiles(
                    scene=base_scene,
                    movement_budget=int(movement_budget),
                    candidate_count=int(candidate_count),
                    instance_seed=int(instance_seed) + int(attempt) * 104729,
                )
            except ValueError as exc:
                attempt_errors.append(str(exc))
                continue
            selected_scene = render_rpg_tactical_map_scene(
                scene_seed,
                width=int(render_params["canvas_width"]),
                height=int(render_params["canvas_height"]),
                grid_cols=int(render_params["grid_cols"]),
                grid_rows=int(render_params["grid_rows"]),
                tile_px=int(render_params.get("tile_px", DEFAULT_TILE_PX)),
                player_tile_id=str(base_scene.units[0].tile_id),
                candidate_tile_ids_by_label=candidates,
                label_font_family=str(label_font_trace.get("font_family", "")),
                label_font_trace=label_font_trace,
                render_metadata=render_params,
            )
            selected_candidates = candidates
            selected_label = answer_label
            selected_costs = movement_costs
            break
        if selected_scene is None or selected_candidates is None or selected_label is None or selected_costs is None:
            raise ValueError("failed to generate valid movement-reachability candidates: " + "; ".join(attempt_errors[-3:]))

        tiles_by_id = _tile_by_id(selected_scene)
        selected_tile_id = str(selected_candidates[str(selected_label)])
        selected_tile = tiles_by_id[selected_tile_id]
        annotation_value = rounded_bbox(selected_tile.bbox_xyxy)
        render_map = movement_reachable_render_map(
            scene=selected_scene,
            candidate_tile_ids_by_label=selected_candidates,
            selected_label=str(selected_label),
            movement_costs_by_tile_id=selected_costs,
            movement_budget=int(movement_budget),
        )
        query_params = {
            "query_id": str(resolved_query_id),
            "prompt_query_key": PROMPT_QUERY_KEY,
            "query_id_probabilities": dict(query_probabilities),
            "movement_budget": int(movement_budget),
            "movement_budget_probabilities": dict(movement_budget_probabilities),
            "candidate_count": int(candidate_count),
            "candidate_labels": list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)]),
            "candidate_tile_ids_by_label": dict(selected_candidates),
            "selected_label": str(selected_label),
            "selected_tile_id": selected_tile_id,
            "canvas_profile": str(render_params.get("canvas_profile", "")),
            "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
        }
        trace_payload = {
            "scene_ir": rpg_tactical_map_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=selected_scene,
                relations={
                    "operation": "select_reachable_tile_under_movement_budget",
                    "movement_budget": int(movement_budget),
                    "terrain_movement_costs": dict(TERRAIN_MOVEMENT_COSTS),
                    "blocked_terrain": [TERRAIN_WATER],
                    "candidate_tile_ids_by_label": dict(selected_candidates),
                    "selected_label": str(selected_label),
                    "selected_tile_id": selected_tile_id,
                },
            ),
            "query_spec": {
                "task_id": TASK_ID,
                "query_id": str(resolved_query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": query_params,
            },
            "render_spec": rpg_tactical_map_render_spec(selected_scene, scene_id=SCENE_ID),
            "render_map": render_map,
            "execution_trace": {
                "query_id": str(resolved_query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "answer": str(selected_label),
                "movement_budget": int(movement_budget),
                "movement_costs_by_tile_id": {str(key): int(value) for key, value in selected_costs.items()},
                "candidate_tile_ids_by_label": dict(selected_candidates),
                "selected_label": str(selected_label),
                "selected_tile_id": selected_tile_id,
                "selected_tile_cost": int(selected_costs[selected_tile_id]),
                "renderer": dict(selected_scene.trace),
            },
            "witness_symbolic": {
                "answer_label": str(selected_label),
                "selected_tile_id": selected_tile_id,
                "selected_tile_bbox": list(annotation_value),
                "selected_tile_cost": int(selected_costs[selected_tile_id]),
                "movement_budget": int(movement_budget),
            },
            "projected_annotation": bbox_projection(annotation_value),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="option_letter", value=str(selected_label)),
            annotation_gt=TypedValue(type="bbox", value=list(annotation_value)),
            image=selected_scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(resolved_query_id),
        )


__all__ = [
    "IllustrationsRpgTacticalMapMovementReachableTileLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
