"""Generation helper for RPG house adjacent-room counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.illustrations.shared.rpg_tile_profiles import resolve_rpg_tile_render_params

from .output import (
    player_entity,
    point_set_map_projection,
    room_point_map,
    rpg_house_adjacent_room_count_render_map,
    rpg_house_render_spec,
    rpg_house_scene_ir,
)
from .prompts import build_rpg_house_prompt_artifacts
from .relations import adjacent_room_ids
from .rendering import DEFAULT_TILE_PX, MAX_ROOM_COUNT, MIN_ROOM_COUNT, SCENE_ID, render_rpg_house_profile_scene
from .sampling import select_count_from_support, select_feasible_count_from_support, select_string_from_support


@dataclass(frozen=True)
class AdjacentRoomCountSample:
    player_room_id: str
    adjacent_room_ids: Tuple[str, ...]
    adjacent_count_probabilities: Mapping[str, float]
    player_room_probabilities: Mapping[str, float]


def _degree_by_room(rooms: Sequence[Any], doors: Sequence[Any]) -> dict[str, int]:
    return {
        str(room.room_id): len(adjacent_room_ids(doors, start_room_id=str(room.room_id)))
        for room in rooms
    }


def _select_adjacent_count(
    *,
    public_name: str,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    degree_values: Sequence[int],
) -> tuple[int, Mapping[str, float]]:
    feasible_counts = frozenset(int(value) for value in degree_values)
    return select_feasible_count_from_support(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="adjacent_room_count_support",
        explicit_key="adjacent_room_count",
        fallback_support=tuple(range(1, MAX_ROOM_COUNT)),
        feasible=lambda value: int(value) in feasible_counts,
        namespace=f"{public_name}:adjacent_room_count",
        empty_context=f"degree_values={sorted(feasible_counts)}",
    )


def _select_player_room(
    *,
    public_name: str,
    params: Mapping[str, Any],
    degree_by_room: Mapping[str, int],
    target_count: int,
    instance_seed: int,
) -> tuple[str, Mapping[str, float]]:
    support = tuple(
        room_id
        for room_id, degree in sorted((str(room_id), int(degree)) for room_id, degree in degree_by_room.items())
        if int(degree) == int(target_count)
    )
    if not support:
        raise ValueError(f"no player room can support adjacent-room count {target_count}")
    return select_string_from_support(
        params=params,
        support=support,
        explicit_key="player_room_id",
        namespace=f"{public_name}:player_room_id:{target_count}",
        instance_seed=int(instance_seed),
    )


def sample_adjacent_room_count(
    *,
    public_name: str,
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    rooms: Sequence[Any],
    doors: Sequence[Any],
    instance_seed: int,
    target_count: int | None = None,
    target_probabilities: Mapping[str, float] | None = None,
) -> AdjacentRoomCountSample:
    degrees = _degree_by_room(rooms, doors)
    if target_count is None:
        target_count, target_probabilities = _select_adjacent_count(
            public_name=str(public_name),
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            params=params,
            degree_values=tuple(degrees.values()),
        )
    elif int(target_count) not in set(int(value) for value in degrees.values()):
        raise ValueError(f"no room has adjacent-room count {target_count}")
    player_room_id, player_probabilities = _select_player_room(
        public_name=str(public_name),
        params=params,
        degree_by_room=degrees,
        target_count=int(target_count),
        instance_seed=int(instance_seed),
    )
    adjacent_ids = adjacent_room_ids(doors, start_room_id=str(player_room_id))
    if len(adjacent_ids) != int(target_count):
        raise RuntimeError(f"adjacent-room count mismatch for {player_room_id}: {adjacent_ids}")
    return AdjacentRoomCountSample(
        player_room_id=str(player_room_id),
        adjacent_room_ids=tuple(adjacent_ids),
        adjacent_count_probabilities=dict(target_probabilities or {}),
        player_room_probabilities=dict(player_probabilities),
    )


def generate_adjacent_room_count_output(
    *,
    public_name: str,
    domain_name: str,
    branch_names: Sequence[str],
    prompt_query_key: str,
    identity_key_names: Mapping[str, str],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
) -> TaskOutput:
    """Generate one adjacent-room count instance with keyed point witnesses."""

    branch_name, branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in branch_names),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(public_name),
        namespace=f"{public_name}:query",
    )
    room_count, room_count_probabilities = select_count_from_support(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        support_key="room_count_support",
        explicit_key="room_count",
        fallback_support=tuple(range(MIN_ROOM_COUNT, MAX_ROOM_COUNT + 1)),
        namespace=f"{public_name}:room_count",
    )
    target_count, target_probabilities = select_count_from_support(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="adjacent_room_count_support",
        explicit_key="adjacent_room_count",
        fallback_support=(1, 2, 3, 4, 5),
        namespace=f"{public_name}:adjacent_room_count",
    )
    render_params = resolve_rpg_tile_render_params(
        task_params,
        render_defaults,
        tile_px_key="rpg_house_tile_px",
        fallback_tile_px=DEFAULT_TILE_PX,
        instance_seed=int(instance_seed),
        namespace=f"{public_name}:canvas_profile",
    )
    tile_px = int(render_params["tile_px"])
    required_defaults = required_group_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "answer_hint_rpg_house_adjacent_room_count",
            "annotation_hint_rpg_house_adjacent_room_count",
            "json_example_rpg_house_adjacent_room_count",
            "json_example_answer_only_rpg_house_adjacent_room_count",
        ],
        context="prompt defaults for RPG house adjacent-room count task",
    )

    scene = None
    sample: AdjacentRoomCountSample | None = None
    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        render_seed = hash64(int(instance_seed), f"{public_name}:render", int(attempt))
        try:
            probe_scene = render_rpg_house_profile_scene(
                render_seed,
                render_params=render_params,
                tile_px=tile_px,
                room_count=int(room_count),
            )
            sample = sample_adjacent_room_count(
                public_name=str(public_name),
                gen_defaults=gen_defaults,
                params=task_params,
                rooms=probe_scene.rooms,
                doors=probe_scene.doors,
                instance_seed=int(hash64(int(instance_seed), "sample", int(attempt))),
                target_count=int(target_count),
                target_probabilities=target_probabilities,
            )
            scene = render_rpg_house_profile_scene(
                render_seed,
                render_params=render_params,
                tile_px=tile_px,
                room_count=int(room_count),
                player_room_id=sample.player_room_id,
            )
            actual_adjacent_ids = adjacent_room_ids(scene.doors, start_room_id=sample.player_room_id)
            if tuple(sorted(actual_adjacent_ids)) == tuple(sorted(sample.adjacent_room_ids)):
                break
            last_error = RuntimeError(
                f"adjacent-room count mismatch: expected {sample.adjacent_room_ids}, got {actual_adjacent_ids}"
            )
            scene = None
        except Exception as exc:  # pragma: no cover - retry path
            last_error = exc
            scene = None
    if scene is None:
        raise RuntimeError(f"could not generate RPG house adjacent-room count: {last_error}") from last_error
    if sample is None:
        raise RuntimeError("could not sample RPG house adjacent-room count")

    room_points = room_point_map(scene)
    player = player_entity(scene)
    if player is None:
        raise RuntimeError("adjacent-room count scene did not render a player")
    annotation_value = {
        "player": [[round(float(player.point_xy[0]), 3), round(float(player.point_xy[1]), 3)]],
        "adjacent_rooms": [room_points[room_id] for room_id in sample.adjacent_room_ids],
    }
    answer = int(len(sample.adjacent_room_ids))
    prompt_artifacts = build_rpg_house_prompt_artifacts(
        domain=str(domain_name),
        scene_id=SCENE_ID,
        prompt_defaults=required_defaults,
        prompt_query_key=str(prompt_query_key),
        slots={
            "json_output_contract": str(required_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(required_defaults["answer_hint_rpg_house_adjacent_room_count"]),
            "annotation_hint": str(required_defaults["annotation_hint_rpg_house_adjacent_room_count"]),
            "json_example": str(required_defaults["json_example_rpg_house_adjacent_room_count"]),
            "json_example_answer_only": str(required_defaults["json_example_answer_only_rpg_house_adjacent_room_count"]),
        },
        instance_seed=int(instance_seed),
    )
    branch_key = str(identity_key_names["branch"])
    public_key = str(identity_key_names["public"])
    parameters = {
        branch_key: str(branch_name),
        "prompt_query_key": str(prompt_query_key),
        "query_id_probabilities": dict(branch_probabilities),
        "room_count": int(room_count),
        "room_count_probabilities": dict(room_count_probabilities),
        "player_room_id": sample.player_room_id,
        "player_room": sample.player_room_id,
        "adjacent_room_ids": list(sample.adjacent_room_ids),
        "adjacent_room_count": int(answer),
        "adjacent_room_count_probabilities": dict(sample.adjacent_count_probabilities),
        "player_room_probabilities": dict(sample.player_room_probabilities),
        "canvas_profile": str(render_params.get("canvas_profile", "")),
        "canvas_profile_probabilities": dict(render_params.get("canvas_profile_probabilities", {})),
    }
    trace_payload = {
        "scene_ir": rpg_house_scene_ir(
            domain=str(domain_name),
            scene_id=SCENE_ID,
            scene=scene,
            relations={
                branch_key: str(branch_name),
                "prompt_query_key": str(prompt_query_key),
                "player_room_id": sample.player_room_id,
                "adjacent_room_ids": list(sample.adjacent_room_ids),
                "answer": int(answer),
            },
        ),
        "query_spec": {
            public_key: str(public_name),
            branch_key: str(branch_name),
            "prompt_query_key": str(prompt_query_key),
            "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": parameters,
        },
        "render_spec": rpg_house_render_spec(scene, scene_id=SCENE_ID),
        "render_map": rpg_house_adjacent_room_count_render_map(
            scene=scene,
            player_room_id=sample.player_room_id,
            adjacent_room_ids=sample.adjacent_room_ids,
        ),
        "execution_trace": {
            branch_key: str(branch_name),
            "prompt_query_key": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "player_room_id": sample.player_room_id,
            "adjacent_room_ids": list(sample.adjacent_room_ids),
            "adjacent_room_count": int(answer),
            "renderer": dict(scene.trace),
        },
        "witness_symbolic": {
            "player_room_id": sample.player_room_id,
            "adjacent_room_ids": list(sample.adjacent_room_ids),
            "adjacent_room_count": int(answer),
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
        query_id=str(branch_name),
    )


__all__ = [
    "AdjacentRoomCountSample",
    "generate_adjacent_room_count_output",
    "sample_adjacent_room_count",
]
