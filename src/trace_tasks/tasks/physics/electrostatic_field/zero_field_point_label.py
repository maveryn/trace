"""Choose the labeled point where the electrostatic field is zero."""

from __future__ import annotations

from typing import Any, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import run_zero_lifecycle
from .shared.prompts import PROMPT_BUNDLE_ID
from .shared.state import SCENE_ID


TASK_ID = "task_physics__electrostatic_field__zero_field_point_label"
TASK_NAMESPACE = "physics_electrostatic_field_zero_field_point"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "zero_field_point_label_query"
PROMPT_QUERY_KEY = "zero_field_point_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class PhysicsElectrostaticFieldZeroFieldPointLabelTask:
    """Choose the labeled point where unequal same-sign charges produce zero net field."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _build_zero_field_contract(self) -> dict[str, str]:
        """Return public prompt keys for this zero-field point task."""

        return {"task_key": TASK_PROMPT_KEY, "query_key": PROMPT_QUERY_KEY}

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one zero-field candidate diagram and bind answer/annotation."""

        prompt_keys = self._build_zero_field_contract()
        return run_zero_lifecycle(
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
            prompt_task_key=prompt_keys["task_key"],
            prompt_query_key=prompt_keys["query_key"],
        )


__all__ = ["PhysicsElectrostaticFieldZeroFieldPointLabelTask", "TASK_ID"]
