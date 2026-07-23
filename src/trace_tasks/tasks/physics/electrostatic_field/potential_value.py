"""Compute electric potential at a marked electrostatic point."""

from __future__ import annotations

from typing import Any, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import run_numeric_lifecycle
from .shared.prompts import PROMPT_BUNDLE_ID
from .shared.state import SCENE_ID


TASK_ID = "task_physics__electrostatic_field__potential_value"
TASK_NAMESPACE = "physics_electrostatic_field_potential_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "potential_value_query"
PROMPT_QUERY_KEY = "potential_value"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsElectrostaticFieldPotentialValueTask:
    """Compute signed electric potential at a marked point from shown charges and distances."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_potential_contract(self) -> dict[str, str]:
        """Return public prompt keys and value key for this numeric task."""

        return {
            "task_key": TASK_PROMPT_KEY,
            "query_key": PROMPT_QUERY_KEY,
            "value_key": "potential_value",
        }

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one potential diagram and bind answer/annotation."""

        contract = self._build_potential_contract()
        return run_numeric_lifecycle(
            domain=self.domain,
            scene_id=SCENE_ID,
            instance_seed=int(instance_seed),
            params=params or {},
            max_attempts=int(max_attempts),
            namespace=TASK_NAMESPACE,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            bundle_fallback=PROMPT_BUNDLE_ID,
            prompt_task_key=contract["task_key"],
            prompt_query_key=contract["query_key"],
            value_trace_key=contract["value_key"],
        )


__all__ = ["PhysicsElectrostaticFieldPotentialValueTask", "TASK_ID"]
