"""Extension-difference public task for spring physics diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import assemble_spring_result
from .shared.mechanics import spring_task_parts
from .shared.state import SCENE_ID, SPRING_MODE_DIFFERENCE, SpringTaskDefaults


TASK_ID = "task_physics__spring__spring_extension_difference"
TASK_NAMESPACE = "physics_spring_extension_difference"
TASK_PROMPT_KEY = "spring_extension_difference_query"
PROMPT_QUERY_KEY = "spring_extension_difference"
INTERNAL_BRANCH = "extension_difference"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)

_DEFAULTS = SpringTaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("physics", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


@register_task
class PhysicsSpringExtensionDifferenceTask:
    """Return the absolute difference between two visible spring extensions."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one extension-difference instance for the no-branch task.

        The public branch is fixed to the no-branch sentinel, while the legacy
        difference label is retained only as internal trace metadata for
        replay/debugging.
        """

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        parts = spring_task_parts(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            domain=self.domain,
            public_name=TASK_ID,
            namespace=TASK_NAMESPACE,
            spring_mode=SPRING_MODE_DIFFERENCE,
            public_branch=str(selected_branch),
            internal_branch=INTERNAL_BRANCH,
            public_branch_probabilities=branch_probabilities,
            solve_for=None,
            prompt_key=TASK_PROMPT_KEY,
            prompt_branch=PROMPT_QUERY_KEY,
            prompt_dynamic_slots={},
            target_support_key="extension_difference_support",
            target_support_fallback=_DEFAULTS.extension_difference_support,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            defaults=_DEFAULTS,
        )
        result = assemble_spring_result(
            parts=parts,
            scene_name=SCENE_ID,
            annotation_value=[list(bbox) for bbox in parts.annotation_value],
        )
        return result


__all__ = ["PhysicsSpringExtensionDifferenceTask"]
