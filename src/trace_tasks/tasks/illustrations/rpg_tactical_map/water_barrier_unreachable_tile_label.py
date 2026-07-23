"""Select the lettered tile unreachable across a full water barrier."""

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
    rounded_bbox,
    rpg_tactical_map_render_spec,
    rpg_tactical_map_scene_ir,
    water_barrier_unreachable_render_map,
)
from .shared.prompts import build_rpg_tactical_map_task_prompt_with_default_slots
from .shared.relations import (
    TERRAIN_FOREST,
    TERRAIN_GRASS,
    TERRAIN_ROAD,
    TERRAIN_WATER,
    connected_passable_tile_ids,
)
from .shared.rendering import (
    DEFAULT_CANDIDATE_LABELS,
    DEFAULT_TILE_PX,
    SCENE_ID,
    map_spanning_water_cells,
    render_rpg_tactical_map_scene,
    resolve_tactical_map_render_params,
    resolve_water_feature_style,
)
from .shared.state import RpgTacticalMapScene, RpgTacticalTile


TASK_ID = "task_illustrations__rpg_tactical_map__water_barrier_unreachable_tile_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "water_barrier_unreachable_tile"
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


def _build_water_barrier_grid(
    *,
    cols: int,
    rows: int,
    rng: random.Random,
    explicit_orientation: str | None,
    explicit_style: str | None,
) -> tuple[list[list[str]], str, str, int, int, list[str]]:
    """Build terrain with one full-width/full-height water barrier and no bridges."""

    orientation = str(explicit_orientation or rng.choice(("vertical", "horizontal")))
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("barrier_orientation must be vertical or horizontal")
    style = resolve_water_feature_style(None if explicit_style is None else str(explicit_style), rng=rng)
    max_thickness = 2
    if orientation == "vertical":
        max_thickness = max(1, min(2, int(cols) - 4))
    else:
        max_thickness = max(1, min(2, int(rows) - 4))
    thickness = int(rng.randint(1, max_thickness))
    if orientation == "vertical":
        start_min = 2
        start_max = int(cols) - int(thickness) - 2
    else:
        start_min = 2
        start_max = int(rows) - int(thickness) - 2
    if start_max < start_min:
        raise ValueError("grid is too small for a full water barrier with candidates on both sides")
    start_index = int(rng.randint(start_min, start_max))

    barrier_cells = map_spanning_water_cells(
        orientation=orientation,
        style=style,
        cols=int(cols),
        rows=int(rows),
        start_index=int(start_index),
        thickness=int(thickness),
        rng=rng,
    )
    grid = [[TERRAIN_GRASS for _ in range(int(cols))] for _ in range(int(rows))]
    for row in range(int(rows)):
        for col in range(int(cols)):
            if (int(row), int(col)) in barrier_cells:
                grid[row][col] = TERRAIN_WATER
            elif rng.random() < 0.10:
                grid[row][col] = TERRAIN_ROAD
            elif rng.random() < 0.13:
                grid[row][col] = TERRAIN_FOREST

    barrier_tile_ids = [
        f"r{int(row):02d}_c{int(col):02d}"
        for row, col in sorted(barrier_cells, key=lambda coord: (int(coord[0]), int(coord[1])))
    ]
    return grid, orientation, style, start_index, thickness, barrier_tile_ids


def _passable_components(scene: RpgTacticalMapScene) -> list[set[str]]:
    """Return orthogonally connected passable components for the tactical grid."""

    tiles_by_coord = _tiles_by_coord(scene)
    remaining = {str(tile.tile_id) for tile in scene.tiles if bool(tile.passable)}
    tiles_by_id = _tile_by_id(scene)
    components: list[set[str]] = []
    while remaining:
        start_tile_id = min(remaining)
        component = connected_passable_tile_ids(
            tiles_by_coord,
            start_coord=tiles_by_id[start_tile_id].coord,
        )
        components.append(set(component))
        remaining -= set(component)
    return sorted(components, key=lambda component: (-len(component), sorted(component)[0]))


def _choose_spread_tiles(
    pool: Sequence[RpgTacticalTile],
    *,
    count: int,
    rng: random.Random,
    excluded_tile_ids: set[str],
    anchors: Sequence[RpgTacticalTile],
) -> list[RpgTacticalTile]:
    """Choose visually separated tiles from one side of the barrier."""

    jitter = {str(tile.tile_id): float(rng.random()) for tile in pool}
    for min_distance in (3, 2, 1):
        selected: list[RpgTacticalTile] = []
        unavailable = set(excluded_tile_ids)
        while len(selected) < int(count):
            references = [*anchors, *selected]
            eligible = [
                tile
                for tile in pool
                if str(tile.tile_id) not in unavailable
                and (
                    not references
                    or min(_tile_manhattan(tile, reference) for reference in references) >= int(min_distance)
                )
            ]
            if not eligible:
                break
            chosen = max(
                eligible,
                key=lambda tile: (
                    min(_tile_manhattan(tile, reference) for reference in references) if references else 99,
                    -abs(int(tile.row) - int(pool[0].row)),
                    -jitter[str(tile.tile_id)],
                ),
            )
            selected.append(chosen)
            unavailable.add(str(chosen.tile_id))
        if len(selected) == int(count):
            return selected
    raise ValueError("could not choose enough separated candidate tiles")


def _select_unreachable_candidate_layout(
    *,
    scene: RpgTacticalMapScene,
    candidate_count: int,
    instance_seed: int,
) -> tuple[str, dict[str, str], str, set[str]]:
    """Select the blue-unit tile plus one unreachable and three reachable candidates."""

    rng = random.Random(f"{int(instance_seed)}:water_barrier_unreachable_candidates")
    passable_components = _passable_components(scene)
    player_component_candidates = [
        component for component in passable_components if len(component) >= int(candidate_count)
    ]
    if not player_component_candidates or len(passable_components) < 2:
        raise ValueError("water barrier must create at least two passable regions with enough candidate space")
    player_component = set(rng.choice(player_component_candidates))
    unreachable_component_candidates = [
        component for component in passable_components if set(component) != player_component and component
    ]
    if not unreachable_component_candidates:
        raise ValueError("water barrier did not create an unreachable passable region")
    unreachable_component = set(rng.choice(unreachable_component_candidates))

    passable_tiles = [tile for tile in scene.tiles if bool(tile.passable)]
    same_region_pool = [tile for tile in passable_tiles if str(tile.tile_id) in player_component]
    unreachable_region_pool = [tile for tile in passable_tiles if str(tile.tile_id) in unreachable_component]

    interior_same_side = [
        tile
        for tile in same_region_pool
        if 0 < int(tile.row) < max(int(other.row) for other in scene.tiles)
        and 0 < int(tile.col) < max(int(other.col) for other in scene.tiles)
    ] or same_region_pool
    player_tile = rng.choice(interior_same_side)
    reachable_tile_ids = connected_passable_tile_ids(_tiles_by_coord(scene), start_coord=player_tile.coord)

    same_side_candidates = _choose_spread_tiles(
        [tile for tile in same_region_pool if str(tile.tile_id) in reachable_tile_ids],
        count=int(candidate_count) - 1,
        rng=rng,
        excluded_tile_ids={str(player_tile.tile_id)},
        anchors=[player_tile],
    )
    opposite_candidates = _choose_spread_tiles(
        [tile for tile in unreachable_region_pool if str(tile.tile_id) not in reachable_tile_ids],
        count=1,
        rng=rng,
        excluded_tile_ids=set(),
        anchors=[],
    )
    unreachable_tile = opposite_candidates[0]

    labels = list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)])
    selected_label = str(labels[int(rng.randrange(int(candidate_count)))])
    candidate_tile_ids_by_label: dict[str, str] = {selected_label: str(unreachable_tile.tile_id)}
    reachable_labels = [str(label) for label in labels if str(label) != selected_label]
    for label, tile in zip(reachable_labels, same_side_candidates, strict=True):
        candidate_tile_ids_by_label[str(label)] = str(tile.tile_id)
    ordered_candidates = {str(label): str(candidate_tile_ids_by_label[str(label)]) for label in labels}
    return str(player_tile.tile_id), ordered_candidates, selected_label, reachable_tile_ids


@register_task
class IllustrationsRpgTacticalMapWaterBarrierUnreachableTileLabelTask:
    """Choose the lettered tile unreachable because water cuts across the map."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one full-barrier connectivity instance with one unreachable candidate."""

        resolved_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}:query",
        )
        candidate_count = int(task_params.get("candidate_count", group_default(_GEN_DEFAULTS, "candidate_count", DEFAULT_CANDIDATE_COUNT)))
        if candidate_count != DEFAULT_CANDIDATE_COUNT:
            raise ValueError("water barrier unreachable task currently supports exactly four candidate tiles")
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
            answer_hint_key="answer_hint_water_barrier_unreachable_tile_label",
            annotation_hint_key="annotation_hint_water_barrier_unreachable_tile_label",
            json_example_key="json_example_water_barrier_unreachable_tile_label",
            json_example_answer_only_key="json_example_answer_only_water_barrier_unreachable_tile_label",
            context_label=TASK_ID,
            slots={
                "water_rule": "Water tiles cannot be crossed; all non-water tiles can be crossed. Moves are only up, down, left, or right.",
            },
            instance_seed=int(instance_seed),
        )

        selected_scene: RpgTacticalMapScene | None = None
        selected_candidates: dict[str, str] | None = None
        selected_label: str | None = None
        selected_reachable_tile_ids: set[str] | None = None
        selected_barrier_tile_ids: list[str] | None = None
        selected_barrier_orientation: str | None = None
        selected_barrier_style: str | None = None
        selected_barrier_start_index: int | None = None
        selected_barrier_thickness: int | None = None
        attempt_errors: list[str] = []
        for attempt in range(max(1, int(max_attempts))):
            scene_seed = int(instance_seed) + int(attempt) * 7919
            rng = random.Random(f"{scene_seed}:water_barrier_grid")
            try:
                (
                    terrain_grid,
                    barrier_orientation,
                    barrier_style,
                    barrier_start_index,
                    barrier_thickness,
                    barrier_tile_ids,
                ) = _build_water_barrier_grid(
                    cols=int(render_params["grid_cols"]),
                    rows=int(render_params["grid_rows"]),
                    rng=rng,
                    explicit_orientation=task_params.get("barrier_orientation"),
                    explicit_style=task_params.get("barrier_style"),
                )
                base_scene = render_rpg_tactical_map_scene(
                    scene_seed,
                    width=int(render_params["canvas_width"]),
                    height=int(render_params["canvas_height"]),
                    grid_cols=int(render_params["grid_cols"]),
                    grid_rows=int(render_params["grid_rows"]),
                    tile_px=int(render_params.get("tile_px", DEFAULT_TILE_PX)),
                    terrain_grid_override=terrain_grid,
                    render_metadata=render_params,
                )
                player_tile_id, candidates, answer_label, reachable_tile_ids = _select_unreachable_candidate_layout(
                    scene=base_scene,
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
                player_tile_id=str(player_tile_id),
                candidate_tile_ids_by_label=candidates,
                terrain_grid_override=terrain_grid,
                label_font_family=str(label_font_trace.get("font_family", "")),
                label_font_trace=label_font_trace,
                render_metadata=render_params,
            )
            selected_candidates = candidates
            selected_label = str(answer_label)
            selected_reachable_tile_ids = set(reachable_tile_ids)
            selected_barrier_tile_ids = list(barrier_tile_ids)
            selected_barrier_orientation = str(barrier_orientation)
            selected_barrier_style = str(barrier_style)
            selected_barrier_start_index = int(barrier_start_index)
            selected_barrier_thickness = int(barrier_thickness)
            break
        if (
            selected_scene is None
            or selected_candidates is None
            or selected_label is None
            or selected_reachable_tile_ids is None
            or selected_barrier_tile_ids is None
            or selected_barrier_orientation is None
            or selected_barrier_style is None
            or selected_barrier_start_index is None
            or selected_barrier_thickness is None
        ):
            raise ValueError("failed to generate valid water-barrier unreachable candidates: " + "; ".join(attempt_errors[-3:]))

        tiles_by_id = _tile_by_id(selected_scene)
        selected_tile_id = str(selected_candidates[str(selected_label)])
        selected_tile = tiles_by_id[selected_tile_id]
        candidate_reachable_by_label = {
            str(label): str(tile_id) in selected_reachable_tile_ids
            for label, tile_id in selected_candidates.items()
        }
        unreachable_labels = [label for label, reachable in candidate_reachable_by_label.items() if not bool(reachable)]
        if unreachable_labels != [str(selected_label)]:
            raise ValueError("water barrier task must have exactly one unreachable candidate")
        annotation_value = rounded_bbox(selected_tile.bbox_xyxy)
        render_map = water_barrier_unreachable_render_map(
            scene=selected_scene,
            candidate_tile_ids_by_label=selected_candidates,
            selected_label=str(selected_label),
            reachable_tile_ids=sorted(selected_reachable_tile_ids),
            water_barrier_tile_ids=selected_barrier_tile_ids,
            barrier_orientation=str(selected_barrier_orientation),
            barrier_style=str(selected_barrier_style),
            barrier_start_index=int(selected_barrier_start_index),
            barrier_thickness=int(selected_barrier_thickness),
        )
        query_params = {
            "query_id": str(resolved_query_id),
            "prompt_query_key": PROMPT_QUERY_KEY,
            "query_id_probabilities": dict(query_probabilities),
            "candidate_count": int(candidate_count),
            "candidate_labels": list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)]),
            "candidate_tile_ids_by_label": dict(selected_candidates),
            "selected_label": str(selected_label),
            "selected_tile_id": selected_tile_id,
            "barrier_orientation": str(selected_barrier_orientation),
            "barrier_style": str(selected_barrier_style),
            "barrier_start_index": int(selected_barrier_start_index),
            "barrier_thickness": int(selected_barrier_thickness),
            "water_barrier_tile_ids": list(selected_barrier_tile_ids),
            "canvas_profile": str(render_params.get("canvas_profile", "")),
            "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
        }
        trace_payload = {
            "scene_ir": rpg_tactical_map_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=selected_scene,
                relations={
                    "operation": "select_unreachable_tile_across_full_water_barrier",
                    "water_rule": "water_blocked_all_non_water_crossable",
                    "blocked_terrain": [TERRAIN_WATER],
                    "barrier_orientation": str(selected_barrier_orientation),
                    "barrier_style": str(selected_barrier_style),
                    "barrier_start_index": int(selected_barrier_start_index),
                    "barrier_thickness": int(selected_barrier_thickness),
                    "water_barrier_tile_ids": list(selected_barrier_tile_ids),
                    "reachable_tile_ids": sorted(selected_reachable_tile_ids),
                    "candidate_tile_ids_by_label": dict(selected_candidates),
                    "candidate_reachable_by_label": dict(candidate_reachable_by_label),
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
                "water_rule": "water_blocked_all_non_water_crossable",
                "barrier_orientation": str(selected_barrier_orientation),
                "barrier_style": str(selected_barrier_style),
                "barrier_start_index": int(selected_barrier_start_index),
                "barrier_thickness": int(selected_barrier_thickness),
                "candidate_tile_ids_by_label": dict(selected_candidates),
                "candidate_reachable_by_label": dict(candidate_reachable_by_label),
                "selected_label": str(selected_label),
                "selected_tile_id": selected_tile_id,
                "reachable_tile_ids": sorted(selected_reachable_tile_ids),
                "water_barrier_tile_ids": list(selected_barrier_tile_ids),
                "renderer": dict(selected_scene.trace),
            },
            "witness_symbolic": {
                "answer_label": str(selected_label),
                "selected_tile_id": selected_tile_id,
                "selected_tile_bbox": list(annotation_value),
                "candidate_reachable_by_label": dict(candidate_reachable_by_label),
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
    "IllustrationsRpgTacticalMapWaterBarrierUnreachableTileLabelTask",
    "PROMPT_QUERY_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
