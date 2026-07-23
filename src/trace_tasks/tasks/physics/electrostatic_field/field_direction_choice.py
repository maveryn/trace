"""Choose the field or force direction at a marked electrostatic point."""

from __future__ import annotations

from typing import Any, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import run_direction_lifecycle
from .shared.prompts import PROMPT_BUNDLE_ID
from .shared.state import (
    DIRECTION_MODE_FIELD,
    DIRECTION_MODE_NEGATIVE_FORCE,
    DIRECTION_MODE_POSITIVE_FORCE,
    SCENE_ID,
)


TASK_ID = "task_physics__electrostatic_field__field_direction_choice"
TASK_NAMESPACE = "physics_electrostatic_field_direction_choice"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "electric_field_direction",
    "force_on_positive_charge",
    "force_on_negative_charge",
)
QUERY_DIRECTION_MODES = {
    "electric_field_direction": DIRECTION_MODE_FIELD,
    "force_on_positive_charge": DIRECTION_MODE_POSITIVE_FORCE,
    "force_on_negative_charge": DIRECTION_MODE_NEGATIVE_FORCE,
}
TASK_PROMPT_KEY = "field_direction_choice_query"
DEFAULT_QUERY_ID = "electric_field_direction"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsElectrostaticFieldDirectionChoiceTask:
    """Choose the labeled arrow matching an electric-field or test-charge force direction."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_direction_contract(self) -> dict[str, str]:
        """Return public prompt keys for this direction-choice task."""

        return {"task_key": TASK_PROMPT_KEY}

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one direction-choice diagram and bind answer/annotation."""

        prompt_keys = self._build_direction_contract()
        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params or {},
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.query",
        )
        explicit_mode = task_params.get("direction_mode")
        selected_mode = str(QUERY_DIRECTION_MODES[str(selected_query)])
        if explicit_mode is not None and str(explicit_mode) != selected_mode:
            raise ValueError(
                f"direction_mode must match query_id for {TASK_ID}: "
                f"{explicit_mode!r} != {selected_mode!r}"
            )
        task_params["direction_mode"] = selected_mode
        return run_direction_lifecycle(
            domain=self.domain,
            scene_id=SCENE_ID,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            namespace=TASK_NAMESPACE,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            bundle_fallback=PROMPT_BUNDLE_ID,
            prompt_task_key=prompt_keys["task_key"],
            prompt_query_key=str(selected_query),
            query_id=str(selected_query),
            query_id_probabilities=query_probabilities,
        )


__all__ = ["PhysicsElectrostaticFieldDirectionChoiceTask", "TASK_ID"]
