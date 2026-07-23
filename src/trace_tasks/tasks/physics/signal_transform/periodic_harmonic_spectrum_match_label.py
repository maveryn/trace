"""Periodic-harmonic spectrum match task for the signal-transform scene."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from ._lifecycle import SignalTransformObjective, run_signal_transform_lifecycle
from .shared.state import PERIODIC_WAVEFORM_FAMILIES, SCENE_ID


TASK_ID = "task_physics__signal_transform__periodic_harmonic_spectrum_match_label"
TASK_NAMESPACE = "physics_signal_transform_periodic_harmonic_spectrum_match_label"
INTERNAL_QUERY_ID = "periodic_wave_harmonic_spectrum"
TASK_PROMPT_KEY = "periodic_harmonic_spectrum_match_query"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "physics",
    SCENE_ID,
    task_id=TASK_ID,
)


def _build_objective() -> SignalTransformObjective:
    """Return the task-owned periodic harmonic objective contract."""

    return SignalTransformObjective(
        public_task_id=TASK_ID,
        lifecycle_namespace=TASK_NAMESPACE,
        internal_query_id=INTERNAL_QUERY_ID,
        task_prompt_key=TASK_PROMPT_KEY,
        supported_waveform_families=PERIODIC_WAVEFORM_FAMILIES,
    )


@register_task
class PhysicsSignalTransformPeriodicHarmonicSpectrumMatchLabelTask:
    """Choose the spectrum option matching periodic-wave harmonic support."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "physics"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def _resolve_objective(self) -> SignalTransformObjective:
        """Bind this public task to the periodic harmonic objective."""

        return _build_objective()

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Select the single public branch and run the signal scene lifecycle."""

        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SINGLE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_NAMESPACE}.branch",
        )
        return run_signal_transform_lifecycle(
            domain=self.domain,
            objective=self._resolve_objective(),
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            task_params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            max_attempts=int(max_attempts),
        )


__all__ = ["PhysicsSignalTransformPeriodicHarmonicSpectrumMatchLabelTask"]
