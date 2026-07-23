"""Choose the labeled point that can be a focus of an elliptical orbit."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import OrbitLifecyclePlan, run_orbit_lifecycle
from .shared.annotations import focus_annotation_point
from .shared.sampling import make_focus_location_spec


TASK_ID = "task_physics__orbital_motion__focus_location_label"
TASK_NAMESPACE = "physics_orbital_motion_focus_location_label"
TASK_PROMPT_KEY = "orbital_motion_focus_location_label_query"
FOCUS_PROMPT_QUERY_KEY = "focus_location_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)


def _build_focus_spec(instance_seed: int, params: Mapping[str, Any], render_defaults: Mapping[str, Any]):
    """Bind the public focus objective to the scene-local focus geometry sampler."""

    return make_focus_location_spec(
        int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        namespace=TASK_NAMESPACE,
    )


@register_task
class PhysicsOrbitalMotionFocusLocationLabelTask:
    """Choose the labeled point that can be the Sun/focus of an elliptical orbit."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate an elliptical orbit and bind the focus-location answer."""

        plan = OrbitLifecyclePlan(
            task_identifier=TASK_ID,
            namespace=TASK_NAMESPACE,
            prompt_task_key=TASK_PROMPT_KEY,
            prompt_query_key=FOCUS_PROMPT_QUERY_KEY,
            public_query_id=SINGLE_QUERY_ID,
            spec_builder=_build_focus_spec,
            annotation_builder=focus_annotation_point,
            query_probabilities={SINGLE_QUERY_ID: 1.0},
            execution_params={},
        )
        return run_orbit_lifecycle(
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            plan=plan,
        )


__all__ = ["PhysicsOrbitalMotionFocusLocationLabelTask", "TASK_ID"]
