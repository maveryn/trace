"""Count rooms reachable from a visible player through open doors."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    required_group_defaults,
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import resolve_rpg_tile_render_params

from .shared.output import (
    point_set_map_projection,
    player_entity,
    room_point_map,
    rpg_house_reachable_room_count_render_map,
    rpg_house_render_spec,
    rpg_house_scene_ir,
)
from .shared.prompts import build_rpg_house_prompt_artifacts
from .shared.relations import connected_component, door_edges, grow_reachable_subset, reachable_room_ids
from .shared.rendering import (
    DEFAULT_TILE_PX,
    MAX_ROOM_COUNT,
    MIN_ROOM_COUNT,
    SCENE_ID,
    render_rpg_house_profile_scene,
)
from .shared.sampling import select_count_from_support, select_feasible_count_from_support, select_string_from_support


TASK_ID = "task_illustrations__rpg_house__reachable_room_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "reachable_room_count"
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


@dataclass(frozen=True)
class _ReachableCountSample:
    start_room_id: str
    reachable_room_ids: Tuple[str, ...]
    door_states: Mapping[str, str]
    reachable_count_probabilities: Mapping[str, float]
    start_room_probabilities: Mapping[str, float]


def _select_reachable_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_count: int,
) -> tuple[int, Mapping[str, float]]:
    return select_feasible_count_from_support(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="reachable_room_count_support",
        explicit_key="reachable_room_count",
        fallback_support=tuple(range(0, MAX_ROOM_COUNT)),
        feasible=lambda value: 0 <= int(value) <= int(max_count),
        namespace=f"{TASK_ID}:reachable_room_count",
        empty_context=f"max_count={max_count}",
    )


def _select_start_room(
    *,
    params: Mapping[str, Any],
    room_ids: Sequence[str],
    edges: Mapping[str, Sequence[tuple[str, str]]],
    target_count: int,
    instance_seed: int,
) -> tuple[str, Mapping[str, float]]:
    support = tuple(
        room_id
        for room_id in sorted(str(room_id) for room_id in room_ids)
        if len(connected_component(edges, start_room_id=room_id)) >= int(target_count) + 1
    )
    if not support:
        raise ValueError(f"no start room can support reachable count {target_count}")
    return select_string_from_support(
        params=params,
        support=support,
        explicit_key="start_room_id",
        namespace=f"{TASK_ID}:start_room_id:{target_count}",
        instance_seed=int(instance_seed),
    )


def _sample_reachable_room_count(
    *,
    params: Mapping[str, Any],
    rooms: Sequence[Any],
    doors: Sequence[Any],
    instance_seed: int,
) -> _ReachableCountSample:
    room_ids = tuple(str(room.room_id) for room in rooms)
    target_count, target_probabilities = _select_reachable_count(
        instance_seed=int(instance_seed),
        params=params,
        max_count=len(room_ids) - 1,
    )
    edges = door_edges(doors)
    start_room_id, start_probabilities = _select_start_room(
        params=params,
        room_ids=room_ids,
        edges=edges,
        target_count=int(target_count),
        instance_seed=int(instance_seed),
    )
    reachable_ids, open_door_ids = grow_reachable_subset(
        edges=edges,
        start_room_id=start_room_id,
        target_count=int(target_count),
        rng=random.Random(hash64(int(instance_seed), f"{TASK_ID}:{start_room_id}:{target_count}")),
    )
    door_states = {str(door.door_id): ("open" if str(door.door_id) in open_door_ids else "closed") for door in doors}
    return _ReachableCountSample(
        start_room_id=str(start_room_id),
        reachable_room_ids=tuple(reachable_ids),
        door_states=door_states,
        reachable_count_probabilities=dict(target_probabilities),
        start_room_probabilities=dict(start_probabilities),
    )


@register_task
class IllustrationsRpgHouseReachableRoomCountTask:
    """Count rooms reachable from the player's room through open doors."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'topology')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate a player-reachability count while keeping graph construction exact."""

        resolved_query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}:query",
        )
        room_count, room_count_probabilities = select_count_from_support(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            support_key="room_count_support",
            explicit_key="room_count",
            fallback_support=tuple(range(MIN_ROOM_COUNT, MAX_ROOM_COUNT + 1)),
            namespace=f"{TASK_ID}:room_count",
        )
        render_params = resolve_rpg_tile_render_params(
            task_params,
            _RENDER_DEFAULTS,
            tile_px_key="rpg_house_tile_px",
            fallback_tile_px=DEFAULT_TILE_PX,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:canvas_profile",
        )
        tile_px = int(render_params["tile_px"])
        required_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            [
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "answer_hint_rpg_house_reachable_room_count",
                "annotation_hint_rpg_house_reachable_room_count",
                "json_example_rpg_house_reachable_room_count",
                "json_example_answer_only_rpg_house_reachable_room_count",
            ],
            context="prompt defaults for RPG house reachable-room count task",
        )

        scene = None
        sample: _ReachableCountSample | None = None
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            render_seed = hash64(int(instance_seed), f"{TASK_ID}:render", int(attempt))
            try:
                probe_scene = render_rpg_house_profile_scene(
                    render_seed,
                    render_params=render_params,
                    tile_px=tile_px,
                    room_count=int(room_count),
                )
                sample = _sample_reachable_room_count(
                    params=task_params,
                    rooms=probe_scene.rooms,
                    doors=probe_scene.doors,
                    instance_seed=int(hash64(int(instance_seed), "sample", int(attempt))),
                )
                scene = render_rpg_house_profile_scene(
                    render_seed,
                    render_params=render_params,
                    tile_px=tile_px,
                    room_count=int(room_count),
                    player_room_id=sample.start_room_id,
                    door_states=sample.door_states,
                )
                actual_reachable_ids = tuple(
                    room_id
                    for room_id in reachable_room_ids(scene.doors, start_room_id=sample.start_room_id)
                    if room_id != sample.start_room_id
                )
                if tuple(sorted(actual_reachable_ids)) == tuple(sorted(sample.reachable_room_ids)):
                    break
                last_error = RuntimeError(
                    f"reachable-room count mismatch: expected {sample.reachable_room_ids}, got {actual_reachable_ids}"
                )
                scene = None
            except Exception as exc:  # pragma: no cover - retry path
                last_error = exc
                scene = None
        if scene is None:
            raise RuntimeError(f"could not generate RPG house reachable-room count: {last_error}") from last_error
        if sample is None:
            raise RuntimeError("could not sample RPG house reachable-room count")

        room_points = room_point_map(scene)
        player = player_entity(scene)
        if player is None:
            raise RuntimeError("reachable-room count scene did not render a player")
        annotation_value = {
            "player": [[round(float(player.point_xy[0]), 3), round(float(player.point_xy[1]), 3)]],
            "reachable_rooms": [room_points[room_id] for room_id in sample.reachable_room_ids],
        }
        answer = int(len(sample.reachable_room_ids))
        prompt_artifacts = build_rpg_house_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=required_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots={
                "json_output_contract": str(required_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
                "answer_hint": str(required_defaults["answer_hint_rpg_house_reachable_room_count"]),
                "annotation_hint": str(required_defaults["annotation_hint_rpg_house_reachable_room_count"]),
                "json_example": str(required_defaults["json_example_rpg_house_reachable_room_count"]),
                "json_example_answer_only": str(required_defaults["json_example_answer_only_rpg_house_reachable_room_count"]),
            },
            instance_seed=int(instance_seed),
        )
        query_params = {
            "query_id": str(resolved_query_id),
            "prompt_query_key": PROMPT_QUERY_KEY,
            "query_id_probabilities": dict(query_probabilities),
            "room_count": int(room_count),
            "room_count_probabilities": dict(room_count_probabilities),
            "start_room_id": sample.start_room_id,
            "player_room_id": sample.start_room_id,
            "player_room": sample.start_room_id,
            "reachable_room_ids": list(sample.reachable_room_ids),
            "reachable_room_count": int(answer),
            "reachable_room_count_probabilities": dict(sample.reachable_count_probabilities),
            "player_room_probabilities": dict(sample.start_room_probabilities),
            "door_states": dict(sample.door_states),
            "canvas_profile": str(render_params.get("canvas_profile", "")),
            "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
        }
        trace_payload = {
            "scene_ir": rpg_house_scene_ir(
                domain=self.domain,
                scene_id=SCENE_ID,
                scene=scene,
                relations={
                    "query_id": str(resolved_query_id),
                    "prompt_query_key": PROMPT_QUERY_KEY,
                    "player_room_id": sample.start_room_id,
                    "reachable_room_ids": list(sample.reachable_room_ids),
                    "answer": int(answer),
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
            "render_spec": rpg_house_render_spec(scene, scene_id=SCENE_ID),
            "render_map": rpg_house_reachable_room_count_render_map(
                scene=scene,
                player_room_id=sample.start_room_id,
                reachable_room_ids=sample.reachable_room_ids,
            ),
            "execution_trace": {
                "query_id": str(resolved_query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "player_room_id": sample.start_room_id,
                "reachable_room_ids": list(sample.reachable_room_ids),
                "reachable_room_count": int(answer),
                "door_states": dict(sample.door_states),
                "renderer": dict(scene.trace),
            },
            "witness_symbolic": {
                "player_room_id": sample.start_room_id,
                "reachable_room_ids": list(sample.reachable_room_ids),
                "reachable_room_count": int(answer),
            },
            "projected_annotation": point_set_map_projection(annotation_value),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=answer),
            annotation_gt=TypedValue(type="point_set_map", value=annotation_value),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(resolved_query_id),
        )


__all__ = [
    "IllustrationsRpgHouseReachableRoomCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
