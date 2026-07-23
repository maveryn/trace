"""Select the endpoint tile after a visible move sequence."""

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
    movement_sequence_endpoint_render_map,
    rounded_bbox,
    rpg_tactical_map_render_spec,
    rpg_tactical_map_scene_ir,
)
from .shared.prompts import build_rpg_tactical_map_task_prompt_with_default_slots
from .shared.rendering import (
    DEFAULT_CANDIDATE_LABELS,
    DEFAULT_TILE_PX,
    SCENE_ID,
    render_rpg_tactical_map_scene,
    resolve_tactical_map_render_params,
)
from .shared.sampling import select_int_from_support
from .shared.state import RpgTacticalMapScene, RpgTacticalTile, TileCoord


TASK_ID = "task_illustrations__rpg_tactical_map__movement_sequence_endpoint_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "movement_sequence_endpoint"
DEFAULT_SEQUENCE_LENGTH_SUPPORT: tuple[int, ...] = (4, 5, 6)
DEFAULT_CANDIDATE_COUNT = 4

DIRECTION_UP = "up"
DIRECTION_DOWN = "down"
DIRECTION_LEFT = "left"
DIRECTION_RIGHT = "right"
DIRECTIONS: tuple[str, ...] = (
    DIRECTION_UP,
    DIRECTION_DOWN,
    DIRECTION_LEFT,
    DIRECTION_RIGHT,
)
DIRECTION_DELTAS: Mapping[str, TileCoord] = {
    DIRECTION_UP: (-1, 0),
    DIRECTION_DOWN: (1, 0),
    DIRECTION_LEFT: (0, -1),
    DIRECTION_RIGHT: (0, 1),
}
REVERSE_DIRECTIONS: Mapping[str, str] = {
    DIRECTION_UP: DIRECTION_DOWN,
    DIRECTION_DOWN: DIRECTION_UP,
    DIRECTION_LEFT: DIRECTION_RIGHT,
    DIRECTION_RIGHT: DIRECTION_LEFT,
}

_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _tiles_by_coord(scene: RpgTacticalMapScene) -> dict[TileCoord, RpgTacticalTile]:
    return {(int(tile.row), int(tile.col)): tile for tile in scene.tiles}


def _tile_by_id(scene: RpgTacticalMapScene) -> dict[str, RpgTacticalTile]:
    return {str(tile.tile_id): tile for tile in scene.tiles}


def _tile_manhattan(first: RpgTacticalTile, second: RpgTacticalTile) -> int:
    return abs(int(first.row) - int(second.row)) + abs(int(first.col) - int(second.col))


def _apply_direction(coord: TileCoord, direction: str) -> TileCoord:
    delta = DIRECTION_DELTAS[str(direction)]
    return (int(coord[0]) + int(delta[0]), int(coord[1]) + int(delta[1]))


def _simulate_sequence(
    *,
    tiles_by_coord: Mapping[TileCoord, RpgTacticalTile],
    start_coord: TileCoord,
    sequence: Sequence[str],
) -> list[RpgTacticalTile] | None:
    """Return the tile path for a valid in-bounds, passable cardinal sequence."""

    current_coord = (int(start_coord[0]), int(start_coord[1]))
    start_tile = tiles_by_coord.get(current_coord)
    if start_tile is None or not bool(start_tile.passable):
        return None
    path = [start_tile]
    for direction in sequence:
        next_coord = _apply_direction(current_coord, str(direction))
        next_tile = tiles_by_coord.get(next_coord)
        if next_tile is None or not bool(next_tile.passable):
            return None
        path.append(next_tile)
        current_coord = next_coord
    return path


def _valid_next_steps(
    *,
    tiles_by_coord: Mapping[TileCoord, RpgTacticalTile],
    current: RpgTacticalTile,
) -> list[tuple[str, RpgTacticalTile]]:
    steps: list[tuple[str, RpgTacticalTile]] = []
    for direction in DIRECTIONS:
        tile = tiles_by_coord.get(_apply_direction(current.coord, direction))
        if tile is not None and bool(tile.passable):
            steps.append((direction, tile))
    return steps


def _weighted_direction_choice(
    *,
    rng: random.Random,
    steps: Sequence[tuple[str, RpgTacticalTile]],
    previous_direction: str | None,
    start_tile: RpgTacticalTile,
    visited_tile_ids: set[str],
    current_distance: int,
) -> tuple[str, RpgTacticalTile]:
    """Choose a valid next move while reducing immediate backtracking."""

    weighted: list[tuple[str, RpgTacticalTile, int]] = []
    for direction, tile in steps:
        weight = 1
        if previous_direction is None or str(direction) != REVERSE_DIRECTIONS.get(str(previous_direction)):
            weight += 3
        if str(tile.tile_id) not in visited_tile_ids:
            weight += 2
        if _tile_manhattan(start_tile, tile) >= int(current_distance):
            weight += 1
        weighted.append((str(direction), tile, int(weight)))
    total = sum(weight for _, _, weight in weighted)
    pick = rng.randint(1, int(total))
    running = 0
    for direction, tile, weight in weighted:
        running += int(weight)
        if int(pick) <= int(running):
            return direction, tile
    return weighted[-1][0], weighted[-1][1]


def _sample_valid_sequence(
    *,
    scene: RpgTacticalMapScene,
    sequence_length: int,
    instance_seed: int,
) -> tuple[tuple[str, ...], list[RpgTacticalTile]]:
    """Sample one useful visible sequence from the blue unit's current tile."""

    if not scene.units:
        raise ValueError("movement sequence task requires a blue unit")
    tiles_by_id = _tile_by_id(scene)
    start_tile = tiles_by_id[str(scene.units[0].tile_id)]
    tiles_by_coord = _tiles_by_coord(scene)
    rng = random.Random(f"{int(instance_seed)}:movement_sequence:{int(sequence_length)}")
    fallback: tuple[tuple[str, ...], list[RpgTacticalTile]] | None = None
    min_endpoint_distance = 2 if int(sequence_length) >= 4 else 1
    for _ in range(240):
        current_tile = start_tile
        previous_direction: str | None = None
        sequence: list[str] = []
        path = [start_tile]
        visited_tile_ids = {str(start_tile.tile_id)}
        for _step_index in range(int(sequence_length)):
            steps = _valid_next_steps(tiles_by_coord=tiles_by_coord, current=current_tile)
            if not steps:
                break
            if previous_direction is not None:
                non_reverse_steps = [
                    (direction, tile)
                    for direction, tile in steps
                    if str(direction) != REVERSE_DIRECTIONS.get(str(previous_direction))
                ]
                if non_reverse_steps:
                    steps = non_reverse_steps
            current_distance = _tile_manhattan(start_tile, current_tile)
            direction, next_tile = _weighted_direction_choice(
                rng=rng,
                steps=steps,
                previous_direction=previous_direction,
                start_tile=start_tile,
                visited_tile_ids=visited_tile_ids,
                current_distance=current_distance,
            )
            sequence.append(str(direction))
            path.append(next_tile)
            visited_tile_ids.add(str(next_tile.tile_id))
            previous_direction = str(direction)
            current_tile = next_tile
        if len(sequence) != int(sequence_length):
            continue
        if str(path[-1].tile_id) == str(start_tile.tile_id):
            continue
        candidate = (tuple(sequence), list(path))
        if fallback is None:
            fallback = candidate
        if _tile_manhattan(start_tile, path[-1]) >= int(min_endpoint_distance):
            return candidate
    if fallback is not None:
        return fallback
    raise ValueError("could not sample a valid movement sequence from the blue unit")


def _candidate_mistake_tiles(
    *,
    scene: RpgTacticalMapScene,
    start_tile: RpgTacticalTile,
    sequence: Sequence[str],
    endpoint_tile: RpgTacticalTile,
) -> list[RpgTacticalTile]:
    """Return plausible distractors from common sequence-following mistakes."""

    tiles_by_coord = _tiles_by_coord(scene)
    tiles_by_id = _tile_by_id(scene)
    seen_tile_ids = {str(endpoint_tile.tile_id), str(start_tile.tile_id)}
    mistakes: list[RpgTacticalTile] = []

    def add_path(sequence_candidate: Sequence[str]) -> None:
        path = _simulate_sequence(
            tiles_by_coord=tiles_by_coord,
            start_coord=start_tile.coord,
            sequence=sequence_candidate,
        )
        if not path:
            return
        tile = path[-1]
        tile_id = str(tile.tile_id)
        if tile_id in seen_tile_ids:
            return
        seen_tile_ids.add(tile_id)
        mistakes.append(tile)

    if len(sequence) > 1:
        add_path(sequence[:-1])
    for index, original_direction in enumerate(sequence):
        for replacement in DIRECTIONS:
            if str(replacement) == str(original_direction):
                continue
            candidate = list(sequence)
            candidate[index] = str(replacement)
            add_path(candidate)
    for index in range(len(sequence) - 1):
        candidate = list(sequence)
        candidate[index], candidate[index + 1] = candidate[index + 1], candidate[index]
        add_path(candidate)

    near_endpoint = sorted(
        [
            tile
            for tile in scene.tiles
            if bool(tile.passable)
            and str(tile.tile_id) not in seen_tile_ids
            and 1 <= _tile_manhattan(tile, endpoint_tile) <= 4
        ],
        key=lambda tile: (
            _tile_manhattan(tile, endpoint_tile),
            _tile_manhattan(tile, start_tile),
            str(tile.tile_id),
        ),
    )
    for tile in near_endpoint:
        seen_tile_ids.add(str(tile.tile_id))
        mistakes.append(tiles_by_id[str(tile.tile_id)])
    return mistakes


def _choose_distractors(
    *,
    scene: RpgTacticalMapScene,
    start_tile: RpgTacticalTile,
    sequence: Sequence[str],
    endpoint_tile: RpgTacticalTile,
    distractor_count: int,
    rng: random.Random,
) -> list[RpgTacticalTile]:
    """Choose unique distractor tiles near the endpoint and common mistakes."""

    pool = _candidate_mistake_tiles(
        scene=scene,
        start_tile=start_tile,
        sequence=sequence,
        endpoint_tile=endpoint_tile,
    )
    if len(pool) < int(distractor_count):
        extra = [
            tile
            for tile in scene.tiles
            if bool(tile.passable)
            and str(tile.tile_id) not in {str(endpoint_tile.tile_id), str(start_tile.tile_id)}
            and all(str(tile.tile_id) != str(existing.tile_id) for existing in pool)
        ]
        rng.shuffle(extra)
        pool.extend(extra)
    jitter = {str(tile.tile_id): float(rng.random()) for tile in pool}

    def choose_with_min_distance(min_distance: int) -> list[RpgTacticalTile]:
        selected: list[RpgTacticalTile] = []
        used = {str(endpoint_tile.tile_id), str(start_tile.tile_id)}
        while len(selected) < int(distractor_count):
            anchors = [endpoint_tile, *selected]
            eligible = [
                tile
                for tile in pool
                if str(tile.tile_id) not in used
                and min(_tile_manhattan(tile, anchor) for anchor in anchors) >= int(min_distance)
            ]
            if not eligible:
                break
            chosen = min(
                eligible,
                key=lambda tile: (
                    _tile_manhattan(tile, endpoint_tile),
                    -min(_tile_manhattan(tile, anchor) for anchor in anchors),
                    jitter[str(tile.tile_id)],
                ),
            )
            selected.append(chosen)
            used.add(str(chosen.tile_id))
        return selected

    for min_distance in (2, 1, 0):
        selected = choose_with_min_distance(min_distance)
        if len(selected) == int(distractor_count):
            return selected
    raise ValueError("could not choose enough movement-sequence endpoint distractors")


def _select_sequence_candidates(
    *,
    scene: RpgTacticalMapScene,
    sequence_length: int,
    candidate_count: int,
    instance_seed: int,
) -> tuple[tuple[str, ...], list[RpgTacticalTile], dict[str, str], str]:
    """Select a valid sequence and four candidate endpoint tiles."""

    if int(candidate_count) != DEFAULT_CANDIDATE_COUNT:
        raise ValueError("movement sequence endpoint task supports exactly four candidate tiles")
    sequence, path = _sample_valid_sequence(
        scene=scene,
        sequence_length=int(sequence_length),
        instance_seed=int(instance_seed),
    )
    start_tile = path[0]
    endpoint_tile = path[-1]
    rng = random.Random(f"{int(instance_seed)}:movement_sequence_endpoint_candidates")
    distractors = _choose_distractors(
        scene=scene,
        start_tile=start_tile,
        sequence=sequence,
        endpoint_tile=endpoint_tile,
        distractor_count=int(candidate_count) - 1,
        rng=rng,
    )

    labels = list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)])
    selected_label = str(labels[int(instance_seed) % int(candidate_count)])
    candidates_by_label: dict[str, str] = {selected_label: str(endpoint_tile.tile_id)}
    distractor_labels = [str(label) for label in labels if str(label) != selected_label]
    for label, tile in zip(distractor_labels, distractors, strict=True):
        candidates_by_label[str(label)] = str(tile.tile_id)
    return (
        tuple(sequence),
        list(path),
        {str(label): str(candidates_by_label[str(label)]) for label in labels},
        selected_label,
    )


@register_task
class IllustrationsRpgTacticalMapMovementSequenceEndpointLabelTask:
    """Choose the lettered tile where the blue unit ends after a move sequence."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'state_update')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one endpoint instance with a sequence valid for the sampled map."""

        resolved_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}:query",
        )
        sequence_length, sequence_length_probabilities = select_int_from_support(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=_GEN_DEFAULTS,
            support_key="sequence_length_support",
            explicit_key="sequence_length",
            fallback_support=DEFAULT_SEQUENCE_LENGTH_SUPPORT,
            namespace=f"{TASK_ID}:sequence_length",
        )
        candidate_count = int(task_params.get("candidate_count", group_default(_GEN_DEFAULTS, "candidate_count", DEFAULT_CANDIDATE_COUNT)))
        if candidate_count != DEFAULT_CANDIDATE_COUNT:
            raise ValueError("rpg tactical map movement sequence task currently supports exactly four candidate tiles")
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

        selected_scene: RpgTacticalMapScene | None = None
        selected_sequence: tuple[str, ...] | None = None
        selected_path: list[RpgTacticalTile] | None = None
        selected_candidates: dict[str, str] | None = None
        selected_label: str | None = None
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
                sequence, path, candidates, answer_label = _select_sequence_candidates(
                    scene=base_scene,
                    sequence_length=int(sequence_length),
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
            selected_sequence = sequence
            selected_path = path
            selected_candidates = candidates
            selected_label = answer_label
            break
        if (
            selected_scene is None
            or selected_sequence is None
            or selected_path is None
            or selected_candidates is None
            or selected_label is None
        ):
            raise ValueError("failed to generate valid movement-sequence candidates: " + "; ".join(attempt_errors[-3:]))

        move_sequence_text = ", ".join(str(direction) for direction in selected_sequence)
        _prompt_defaults, prompt_artifacts = build_rpg_tactical_map_task_prompt_with_default_slots(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults_source=_PROMPT_DEFAULTS,
            prompt_query_key=PROMPT_QUERY_KEY,
            answer_hint_key="answer_hint_movement_sequence_endpoint_label",
            annotation_hint_key="annotation_hint_movement_sequence_endpoint_label",
            json_example_key="json_example_movement_sequence_endpoint_label",
            json_example_answer_only_key="json_example_answer_only_movement_sequence_endpoint_label",
            context_label=TASK_ID,
            slots={
                "move_sequence": str(move_sequence_text),
            },
            instance_seed=int(instance_seed),
        )

        tiles_by_id = _tile_by_id(selected_scene)
        selected_tile_id = str(selected_candidates[str(selected_label)])
        selected_tile = tiles_by_id[selected_tile_id]
        path_tile_ids = [str(tile.tile_id) for tile in selected_path]
        annotation_value = rounded_bbox(selected_tile.bbox_xyxy)
        render_map = movement_sequence_endpoint_render_map(
            scene=selected_scene,
            candidate_tile_ids_by_label=selected_candidates,
            selected_label=str(selected_label),
            move_sequence=selected_sequence,
            path_tile_ids=path_tile_ids,
        )
        query_params = {
            "query_id": str(resolved_query_id),
            "prompt_query_key": PROMPT_QUERY_KEY,
            "query_id_probabilities": dict(query_probabilities),
            "sequence_length": int(sequence_length),
            "sequence_length_probabilities": dict(sequence_length_probabilities),
            "move_sequence": [str(direction) for direction in selected_sequence],
            "move_sequence_text": str(move_sequence_text),
            "candidate_count": int(candidate_count),
            "candidate_labels": list(DEFAULT_CANDIDATE_LABELS[: int(candidate_count)]),
            "candidate_tile_ids_by_label": dict(selected_candidates),
            "selected_label": str(selected_label),
            "selected_tile_id": selected_tile_id,
            "start_tile_id": path_tile_ids[0],
            "path_tile_ids": list(path_tile_ids),
            "canvas_profile": str(render_params.get("canvas_profile", "")),
            "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
        }
        trace_payload = {
            "scene_ir": rpg_tactical_map_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=selected_scene,
                relations={
                    "operation": "select_endpoint_after_cardinal_move_sequence",
                    "movement_rule": "one_tile_per_step_orthogonal_sequence",
                    "move_sequence": [str(direction) for direction in selected_sequence],
                    "path_tile_ids": list(path_tile_ids),
                    "start_tile_id": path_tile_ids[0],
                    "endpoint_tile_id": selected_tile_id,
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
                "movement_rule": "one_tile_per_step_orthogonal_sequence",
                "move_sequence": [str(direction) for direction in selected_sequence],
                "path_tile_ids": list(path_tile_ids),
                "path_coords": [[int(tile.row), int(tile.col)] for tile in selected_path],
                "candidate_tile_ids_by_label": dict(selected_candidates),
                "selected_label": str(selected_label),
                "selected_tile_id": selected_tile_id,
                "renderer": dict(selected_scene.trace),
            },
            "witness_symbolic": {
                "answer_label": str(selected_label),
                "selected_tile_id": selected_tile_id,
                "selected_tile_bbox": list(annotation_value),
                "move_sequence": [str(direction) for direction in selected_sequence],
                "path_tile_ids": list(path_tile_ids),
                "start_tile_id": path_tile_ids[0],
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
    "DIRECTIONS",
    "IllustrationsRpgTacticalMapMovementSequenceEndpointLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
