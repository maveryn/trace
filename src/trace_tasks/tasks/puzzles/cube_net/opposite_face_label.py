"""Public cube-net task for selecting the face opposite a marked face."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import run_face_relation_lifecycle
from .shared.state import DOMAIN, SCENE_ID


TASK_ID = "task_puzzles__cube_net__opposite_face_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "opposite_face_label_query"
PROMPT_QUERY_KEY = "opposite_face_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _build_face_relation_objective() -> dict[str, str]:
    """Return this task's opposite-face objective wiring."""

    return {
        "relation_kind": "opposite",
        "prompt_task_key": PROMPT_TASK_KEY,
        "prompt_query_key": PROMPT_QUERY_KEY,
        "marked_cue_key": "marked_face",
    }


@register_task
class PuzzlesCubeNetOppositeFaceLabelTask:
    """Select the option matching the face opposite the marked cube-net face."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def _build_objective(self) -> dict[str, str]:
        """Return the fixed opposite-face contract for this task."""

        return _build_face_relation_objective()

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one fixed opposite-face option task."""

        objective = self._build_objective()
        return run_face_relation_lifecycle(
            task_identity=TASK_ID,
            relation_kind=objective["relation_kind"],
            prompt_task_key=objective["prompt_task_key"],
            prompt_query_key=objective["prompt_query_key"],
            marked_cue_key=objective["marked_cue_key"],
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["PuzzlesCubeNetOppositeFaceLabelTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
