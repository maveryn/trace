"""Missing-value public task for spring physics diagrams."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import assemble_spring_result
from .shared.mechanics import spring_task_parts
from .shared.state import (
    SCENE_ID,
    SPRING_MODE_MISSING_EXTENSION,
    SPRING_MODE_MISSING_WEIGHT,
    SpringTaskDefaults,
)


TASK_ID = "task_physics__spring__spring_missing_value"
TASK_NAMESPACE = "physics_spring_missing_value"
TASK_PROMPT_KEY = "spring_missing_value_query"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "missing_weight_for_extension",
    "missing_extension_for_weight",
)
_BRANCH_TO_MODE = {
    "missing_weight_for_extension": SPRING_MODE_MISSING_WEIGHT,
    "missing_extension_for_weight": SPRING_MODE_MISSING_EXTENSION,
}
_BRANCH_TO_SOLVE_FOR = {
    "missing_weight_for_extension": "weight",
    "missing_extension_for_weight": "extension",
}
_BRANCH_TO_SUPPORT = {
    "missing_weight_for_extension": "missing_weight_support",
    "missing_extension_for_weight": "missing_extension_support",
}

_DEFAULTS = SpringTaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("physics", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


@register_task
class PhysicsSpringMissingValueTask:
    """Return the missing weight or extension from identical spring measurements."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one missing-value instance after binding the public branch.

        The public query id selects which right-side quantity is unknown; the
        shared spring primitives then receive semantic mode and support keys
        rather than public task/query routing metadata.
        """

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        support_key = _BRANCH_TO_SUPPORT[str(selected_branch)]
        fallback_support = getattr(_DEFAULTS, support_key)
        solve_for = _BRANCH_TO_SOLVE_FOR[str(selected_branch)]
        parts = spring_task_parts(
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            domain=self.domain,
            public_name=TASK_ID,
            namespace=TASK_NAMESPACE,
            spring_mode=str(_BRANCH_TO_MODE[str(selected_branch)]),
            public_branch=str(selected_branch),
            internal_branch=str(selected_branch),
            public_branch_probabilities=branch_probabilities,
            solve_for=str(solve_for),
            prompt_key=TASK_PROMPT_KEY,
            prompt_branch=str(selected_branch),
            prompt_dynamic_slots={"solve_for": str(solve_for)},
            target_support_key=str(support_key),
            target_support_fallback=tuple(int(value) for value in fallback_support),
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            defaults=_DEFAULTS,
        )
        result = assemble_spring_result(
            parts=parts,
            scene_name=SCENE_ID,
            annotation_value=dict(parts.annotation_value),
        )
        return result


__all__ = ["PhysicsSpringMissingValueTask"]
