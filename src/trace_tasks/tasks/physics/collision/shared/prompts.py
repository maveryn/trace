"""Prompt assembly helpers for the physics collision scene."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


PROMPT_BUNDLE_ID = "physics_collision_v1"
SCENE_PROMPT_KEY = "collision_diagram"
_AFTERMATH_OBJECT_DESCRIPTIONS = {
    "aftermath_table": (
        "a top-down collision table with an impact marker, a target puck shown "
        "after impact with its motion trail, and six labeled incoming path arrows"
    ),
    "aftermath_gridded_table": (
        "a gridded top-down collision table with an impact marker, a target puck "
        "shown after impact with its motion trail, and six labeled incoming path arrows"
    ),
    "aftermath_compact_table": (
        "a compact top-down collision table with an impact marker, a target puck "
        "shown after impact with its motion trail, and six labeled incoming path arrows"
    ),
}
_STICKY_OBJECT_DESCRIPTIONS = {
    "wide_table": (
        "a collision table with puck A moving horizontally, puck B moving vertically, "
        "visible mass and speed labels, a stuck A+B puck, signed axes, and four candidate direction arrows"
    ),
    "compact_table": (
        "a compact collision table with puck A moving horizontally, puck B moving vertically, "
        "visible mass and speed labels, a stuck A+B puck, signed axes, and four candidate direction arrows"
    ),
    "gridded_table": (
        "a gridded collision table with puck A moving horizontally, puck B moving vertically, "
        "visible mass and speed labels, a stuck A+B puck, signed axes, and four candidate direction arrows"
    ),
}
_STICKY_SPEED_OBJECT_DESCRIPTIONS = {
    "wide_table": (
        "a collision table with puck A moving horizontally, puck B moving vertically, "
        "visible mass and speed labels, a stuck A+B puck, and signed axes"
    ),
    "compact_table": (
        "a compact collision table with puck A moving horizontally, puck B moving vertically, "
        "visible mass and speed labels, a stuck A+B puck, and signed axes"
    ),
    "gridded_table": (
        "a gridded collision table with puck A moving horizontally, puck B moving vertically, "
        "visible mass and speed labels, a stuck A+B puck, and signed axes"
    ),
}
_AFTERMATH_ANNOTATION_HINT = (
    'set "annotation" to an object mapping impact_point and target_after_motion '
    "to [x0,y0,x1,y1] pixel boxes around the impact marker and the target puck "
    "with its motion trail"
)
_STICKY_DIRECTION_ANNOTATION_HINT = (
    'set "annotation" to an array of two line segments, each written as '
    "[[x0, y0], [x1, y1]], where each endpoint is a [x, y] pixel point, "
    "for puck A's motion arrow and puck B's motion arrow"
)
_STICKY_SPEED_ANNOTATION_HINT = (
    'set "annotation" to an array of two line segments, each written as '
    "[[x0, y0], [x1, y1]], where each endpoint is a [x, y] pixel point, "
    "for puck A's motion arrow and puck B's motion arrow"
)
_STICKY_DIRECTION_JSON_EXAMPLE = '{"annotation":[[[176,304],[366,304]],[[436,158],[436,254]]],"answer":"D"}'
_STICKY_SPEED_JSON_EXAMPLE = '{"annotation":[[[176,304],[366,304]],[[436,158],[436,254]]],"answer":5.0}'
_STICKY_DIRECTION_JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"D"}'
_STICKY_SPEED_JSON_EXAMPLE_ANSWER_ONLY = '{"answer":5.0}'
_AFTERMATH_ANSWER_HINT = 'set "answer" to the letter of the incoming path that caused the shown target-puck motion'
_STICKY_DIRECTION_ANSWER_HINT = 'set "answer" to the option letter A, B, C, or D of the correct candidate arrow'
_STICKY_SPEED_ANSWER_HINT = 'set "answer" to the final speed in m/s rounded to one decimal place'
_AFTERMATH_JSON_EXAMPLE = '{"annotation":{"impact_point":[120,130,150,160],"target_after_motion":[180,90,260,150]},"answer":"B"}'
_AFTERMATH_JSON_EXAMPLE_ANSWER_ONLY = '{"answer":"B"}'


def aftermath_prompt_slots(scene_variant: str) -> dict[str, Any]:
    """Return prompt slots for one rendered aftermath scene variant."""

    return {
        "object_description": _AFTERMATH_OBJECT_DESCRIPTIONS[str(scene_variant)],
        "annotation_hint": _AFTERMATH_ANNOTATION_HINT,
        "answer_hint": _AFTERMATH_ANSWER_HINT,
        "json_example": _AFTERMATH_JSON_EXAMPLE,
        "json_example_answer_only": _AFTERMATH_JSON_EXAMPLE_ANSWER_ONLY,
    }


def sticky_direction_prompt_slots(scene_variant: str) -> dict[str, Any]:
    """Return prompt slots for the sticky-collision direction objective."""

    return {
        "object_description": _STICKY_OBJECT_DESCRIPTIONS[str(scene_variant)],
        "annotation_hint": _STICKY_DIRECTION_ANNOTATION_HINT,
        "answer_hint": _STICKY_DIRECTION_ANSWER_HINT,
        "json_example": _STICKY_DIRECTION_JSON_EXAMPLE,
        "json_example_answer_only": _STICKY_DIRECTION_JSON_EXAMPLE_ANSWER_ONLY,
    }


def sticky_speed_prompt_slots(scene_variant: str) -> dict[str, Any]:
    """Return prompt slots for the sticky-collision speed objective."""

    return {
        "object_description": _STICKY_SPEED_OBJECT_DESCRIPTIONS[str(scene_variant)],
        "annotation_hint": _STICKY_SPEED_ANNOTATION_HINT,
        "answer_hint": _STICKY_SPEED_ANSWER_HINT,
        "json_example": _STICKY_SPEED_JSON_EXAMPLE,
        "json_example_answer_only": _STICKY_SPEED_JSON_EXAMPLE_ANSWER_ONLY,
    }


def build_collision_prompt_artifacts(
    *,
    domain: str,
    bundle_id: str,
    task_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Build prompt artifacts from collision-scene v1 prompt assets."""

    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
        scene_key=SCENE_PROMPT_KEY,
        task_key=str(task_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={str(key): value for key, value in dynamic_slots.items()},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "PROMPT_BUNDLE_ID",
    "SCENE_PROMPT_KEY",
    "aftermath_prompt_slots",
    "build_collision_prompt_artifacts",
    "sticky_direction_prompt_slots",
    "sticky_speed_prompt_slots",
]
