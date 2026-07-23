"""Count enclosed rooms in a top-down RPG house layout."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
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
    point_set_projection,
    room_point_map,
    rpg_house_render_spec,
    rpg_house_room_count_render_map,
    rpg_house_scene_ir,
)
from .shared.prompts import build_rpg_house_prompt_artifacts
from .shared.rendering import (
    DEFAULT_TILE_PX,
    MAX_ROOM_COUNT,
    MIN_ROOM_COUNT,
    SCENE_ID,
    render_rpg_house_profile_scene,
)
from .shared.sampling import select_count_from_support


TASK_ID = "task_illustrations__rpg_house__room_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "room_count"
_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


@register_task
class IllustrationsRpgHouseRoomCountTask:
    """Count all enclosed rooms in the generated RPG house."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one room-counting instance with point-set room witnesses."""

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
                "answer_hint_rpg_house_room_count",
                "annotation_hint_rpg_house_room_count",
                "json_example_rpg_house_room_count",
                "json_example_answer_only_rpg_house_room_count",
            ],
            context="prompt defaults for RPG house room-count task",
        )

        scene = render_rpg_house_profile_scene(
            int(instance_seed),
            render_params=render_params,
            tile_px=tile_px,
            room_count=int(room_count),
            sample_mixed_door_states=True,
        )
        room_points_by_id = room_point_map(scene)
        annotation_value = [room_points_by_id[str(room.room_id)] for room in scene.rooms]
        answer = int(len(scene.rooms))
        prompt_artifacts = build_rpg_house_prompt_artifacts(
            domain=self.domain,
            scene_id=SCENE_ID,
            prompt_defaults=required_defaults,
            prompt_query_key=PROMPT_QUERY_KEY,
            slots={
                "json_output_contract": str(required_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
                "answer_hint": str(required_defaults["answer_hint_rpg_house_room_count"]),
                "annotation_hint": str(required_defaults["annotation_hint_rpg_house_room_count"]),
                "json_example": str(required_defaults["json_example_rpg_house_room_count"]),
                "json_example_answer_only": str(required_defaults["json_example_answer_only_rpg_house_room_count"]),
            },
            instance_seed=int(instance_seed),
        )
        query_params = {
            "query_id": str(resolved_query_id),
            "prompt_query_key": PROMPT_QUERY_KEY,
            "query_id_probabilities": dict(query_probabilities),
            "room_count": int(room_count),
            "room_count_probabilities": dict(room_count_probabilities),
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
                    "counted_room_ids": [str(room.room_id) for room in scene.rooms],
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
            "render_map": rpg_house_room_count_render_map(scene=scene),
            "execution_trace": {
                "query_id": str(resolved_query_id),
                "prompt_query_key": PROMPT_QUERY_KEY,
                "scene_id": SCENE_ID,
                "room_count": int(answer),
                "counted_room_ids": [str(room.room_id) for room in scene.rooms],
                "renderer": dict(scene.trace),
            },
            "witness_symbolic": {
                "room_count": int(answer),
                "counted_room_ids": [str(room.room_id) for room in scene.rooms],
            },
            "projected_annotation": point_set_projection(annotation_value),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=answer),
            annotation_gt=TypedValue(type="point_set", value=annotation_value),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(resolved_query_id),
        )


__all__ = [
    "IllustrationsRpgHouseRoomCountTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
