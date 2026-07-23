"""Compute total equivalent capacitance from a visible capacitor network."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import EquivalentCircuitObjective, run_equivalent_circuit_lifecycle
from .shared.state import CAPACITANCE_SUPPORT_KEY, SCENE_ID


TASK_ID = "task_physics__circuit_equivalent__total_capacitance_value"
TASK_NAMESPACE = "physics_circuit_equivalent_total_capacitance"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "total_capacitance_value_query"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


def _build_capacitance_objective() -> EquivalentCircuitObjective:
    """Return the task-owned capacitance objective contract."""

    return EquivalentCircuitObjective(
        component_kind="capacitor",
        support_key=CAPACITANCE_SUPPORT_KEY,
        task_prompt_key=TASK_PROMPT_KEY,
        scene_kind_suffix="capacitor",
        object_description=(
            "one capacitor circuit between terminals A and B with at least one "
            "labeled parallel-plate capacitor in series with one or two labeled "
            "parallel capacitor blocks"
        ),
        quantity_name="capacitance",
        component_name_plural="capacitor values",
        annotation_hint=(
            "set \"annotation\" to one [x0,y0,x1,y1] pixel box around the full "
            "capacitor network between terminals A and B, including the component "
            "symbols, value labels, and connecting wires"
        ),
        answer_hint=(
            "set \"answer\" to the total equivalent capacitance between terminals "
            "A and B as an integer number of microfarads"
        ),
        annotation_example=[90, 120, 760, 520],
    )


@register_task
class PhysicsCircuitEquivalentTotalCapacitanceValueTask:
    """Compute total equivalent capacitance from a capacitor network."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'topology', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _resolve_objective(self) -> EquivalentCircuitObjective:
        """Bind this public task to the capacitance objective."""

        return _build_capacitance_objective()

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Select the public query branch and run the scene lifecycle."""

        selected_query, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        return run_equivalent_circuit_lifecycle(
            domain=self.domain,
            public_task_id=TASK_ID,
            lifecycle_namespace=TASK_NAMESPACE,
            objective=self._resolve_objective(),
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            task_params=task_params,
            selected_branch=str(selected_query),
            branch_probabilities=branch_probabilities,
            max_attempts=int(max_attempts),
        )


__all__ = ["PhysicsCircuitEquivalentTotalCapacitanceValueTask"]
