"""PV process sign-choice task for pressure-volume diagrams."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import SignOptionLifecyclePlan, run_sign_option_lifecycle
from .shared.state import PVDiagramTaskDefaults, SCENE_ID


TASK_ID = "task_physics__pv_diagram__pv_process_sign_choice"
TASK_NAMESPACE = "physics_pv_diagram_pv_process_sign_choice"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)
PROMPT_BRANCH_KEY = "process_sign_choice"

_DEFAULTS = PVDiagramTaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _build_sign_option_plan() -> SignOptionLifecyclePlan:
    """Bind this public objective to the PV sign-option lifecycle."""

    return SignOptionLifecyclePlan(
        task_identifier=TASK_ID,
        namespace=TASK_NAMESPACE,
        public_branch_ids=SUPPORTED_QUERY_IDS,
        default_branch_id=SINGLE_QUERY_ID,
        prompt_branch_key=PROMPT_BRANCH_KEY,
        fallback_defaults=_DEFAULTS,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        prompt_defaults_group=_PROMPT_DEFAULTS,
        post_noise_defaults=POST_IMAGE_NOISE_DEFAULTS,
    )


@register_task
class PhysicsPVDiagramProcessSignChoiceTask:
    """Choose the labeled PV process whose gas work has the requested sign."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
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
        """Sample candidate PV arrows, bind the unique sign match, and render output."""

        return run_sign_option_lifecycle(
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            plan=_build_sign_option_plan(),
        )


__all__ = ["PhysicsPVDiagramProcessSignChoiceTask"]
