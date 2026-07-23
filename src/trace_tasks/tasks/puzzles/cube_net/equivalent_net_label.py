"""Public cube-net task for matching equivalent colored nets."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import run_equivalent_net_lifecycle
from .shared.state import DOMAIN, SCENE_ID


TASK_ID = "task_puzzles__cube_net__equivalent_net_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "equivalent_net_label_query"
PROMPT_QUERY_KEY = "equivalent_net_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _build_equivalent_net_objective() -> dict[str, str]:
    """Return prompt metadata for the fixed equivalent-net contract."""

    return {
        "prompt_task_key": PROMPT_TASK_KEY,
        "prompt_query_key": PROMPT_QUERY_KEY,
    }


@register_task
class PuzzlesCubeNetEquivalentNetLabelTask:
    """Select the candidate net that folds to the same colored cube."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def _build_objective(self) -> dict[str, str]:
        """Return this task's fixed colored-net equivalence objective."""

        return _build_equivalent_net_objective()

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one fixed colored-net equivalence option task."""

        objective = self._build_objective()
        return run_equivalent_net_lifecycle(
            task_identity=TASK_ID,
            prompt_task_key=objective["prompt_task_key"],
            prompt_query_key=objective["prompt_query_key"],
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = [
    "PuzzlesCubeNetEquivalentNetLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
