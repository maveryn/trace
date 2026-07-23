"""Choose the labeled orbital position with greatest or least speed."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import OrbitLifecyclePlan, run_orbit_lifecycle
from .shared.annotations import speed_annotation_point
from .shared.sampling import make_speed_extremum_spec


TASK_ID = "task_physics__orbital_motion__orbital_speed_extremum_label"
TASK_NAMESPACE = "physics_orbital_motion_speed_extremum_label"
TASK_PROMPT_KEY = "orbital_motion_speed_extremum_label_query"
GREATEST_SPEED_QUERY_ID = "greatest_speed_position_label"
LEAST_SPEED_QUERY_ID = "least_speed_position_label"
SUPPORTED_QUERY_IDS: Tuple[str, str] = (GREATEST_SPEED_QUERY_ID, LEAST_SPEED_QUERY_ID)
_SPEED_DIRECTION_BY_QUERY = {
    GREATEST_SPEED_QUERY_ID: "greatest",
    LEAST_SPEED_QUERY_ID: "least",
}


def _select_branch(instance_seed: int, params: Dict[str, Any]) -> tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select the speed-extremum semantic branch owned by this public task."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=GREATEST_SPEED_QUERY_ID,
        task_id=TASK_ID,
        namespace=f"{TASK_NAMESPACE}.branch",
    )


def _build_speed_spec_builder(speed_direction: str):
    """Return a sampler bound to the chosen greatest/least speed direction."""

    def _build_speed_spec(instance_seed: int, params: Mapping[str, Any], render_defaults: Mapping[str, Any]):
        return make_speed_extremum_spec(
            int(instance_seed),
            params=params,
            render_defaults=render_defaults,
            speed_direction=str(speed_direction),
            namespace=TASK_NAMESPACE,
        )

    return _build_speed_spec


@register_task
class PhysicsOrbitalMotionSpeedExtremumLabelTask:
    """Choose the labeled planet position with greatest or least orbital speed."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations', 'formula_evaluation')
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate an orbit with a visible Sun and bind the speed-extremum answer."""

        branch_id, branch_probabilities, task_params = _select_branch(int(instance_seed), dict(params or {}))
        speed_direction = str(_SPEED_DIRECTION_BY_QUERY[str(branch_id)])
        plan = OrbitLifecyclePlan(
            task_identifier=TASK_ID,
            namespace=TASK_NAMESPACE,
            prompt_task_key=TASK_PROMPT_KEY,
            prompt_query_key=str(branch_id),
            public_query_id=str(branch_id),
            spec_builder=_build_speed_spec_builder(speed_direction),
            annotation_builder=speed_annotation_point,
            query_probabilities=branch_probabilities,
            execution_params={"speed_direction": speed_direction},
        )
        return run_orbit_lifecycle(
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            plan=plan,
        )


__all__ = ["PhysicsOrbitalMotionSpeedExtremumLabelTask", "TASK_ID"]
