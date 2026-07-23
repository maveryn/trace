"""Count mirror-bounce points in a ray-optics diagram."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import RayOpticsLifecyclePlan, run_ray_optics_lifecycle
from .shared.state import RAY_EVENT_BOUNCE, RayOpticsTaskDefaults, SCENE_ID


TASK_ID = "task_physics__ray_optics__ray_bounce_count"
TASK_NAMESPACE = "physics_ray_optics_ray_bounce_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_BRANCH_KEY = RAY_EVENT_BOUNCE

_DEFAULTS = RayOpticsTaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(
        "physics",
        SCENE_ID,
        task_id=TASK_ID,
    )
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(
    scene_id=SCENE_ID,
    apply_prob=0.5,
)


def _build_bounce_plan(
    *,
    public_branch_id: str,
    public_branch_probabilities: Mapping[str, float],
) -> RayOpticsLifecyclePlan:
    """Bind the mirror-bounce objective to the scene lifecycle."""

    return RayOpticsLifecyclePlan(
        task_identifier=TASK_ID,
        namespace=TASK_NAMESPACE,
        public_branch_id=str(public_branch_id),
        public_branch_probabilities=dict(public_branch_probabilities),
        ray_event_kind=RAY_EVENT_BOUNCE,
        prompt_branch_key=PROMPT_BRANCH_KEY,
        fallback_defaults=_DEFAULTS,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_defaults_group=_PROMPT_DEFAULTS,
        post_noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
    )


@register_task
class PhysicsRayOpticsRayBounceCountTask:
    """Count reflection events along the solved hidden ray path."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'spatial_relations', 'transformation')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one ray-bounce count instance."""

        public_branch_id, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.public_branch",
        )
        plan = _build_bounce_plan(
            public_branch_id=str(public_branch_id),
            public_branch_probabilities=branch_probabilities,
        )
        return run_ray_optics_lifecycle(
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
            plan=plan,
        )


__all__ = ["PhysicsRayOpticsRayBounceCountTask"]
