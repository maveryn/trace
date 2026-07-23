"""Single-tray dice probability for a visible value threshold."""

from __future__ import annotations

from ...registry import register_task
from ._lifecycle import run_dice_probability_lifecycle
from .shared.rules import (
    ThresholdEventSpec,
    build_single_dataset,
    threshold_at_least,
    threshold_at_most,
    value_threshold_candidates,
)


TASK_ID = "task_symbolic__dice__single_threshold_probability"
SUPPORTED_QUERY_IDS = (
    "single_value_at_least_probability",
    "single_value_at_most_probability",
)
_REASONING_LOAD = {
    "single_value_at_least_probability": 0.30,
    "single_value_at_most_probability": 0.30,
}
_QUERY_SPECS = {
    "single_value_at_least_probability": ThresholdEventSpec(
        property_name="at_least",
        description_template="shows a value at least {threshold}",
        thresholds=(3, 4, 5),
        comparator=threshold_at_least,
    ),
    "single_value_at_most_probability": ThresholdEventSpec(
        property_name="at_most",
        description_template="shows a value at most {threshold}",
        thresholds=(2, 3, 4),
        comparator=threshold_at_most,
    ),
}


def _build_dataset(
    *,
    public_query_id: str,
    params,
    gen_defaults,
    instance_seed: int,
):
    def candidate_builder(dice, params, gen_defaults, rng):
        del rng
        try:
            spec = _QUERY_SPECS[str(public_query_id)]
        except KeyError as exc:
            raise ValueError(f"unsupported dice threshold query_id: {public_query_id}") from exc
        return value_threshold_candidates(
            dice,
            params=params,
            gen_defaults=gen_defaults,
            spec=spec,
        )

    return build_single_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        candidate_builder=candidate_builder,
    )


@register_task
class SymbolicProbabilityDiceSingleThresholdProbabilityTask:
    """Compute a single-tray probability from a value threshold."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'formula_evaluation')
    domain = "symbolic"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_dice_probability_lifecycle(
            public_task_id=TASK_ID,
            supported_queries=SUPPORTED_QUERY_IDS,
            prompt_query_key=None,
            dataset_builder=_build_dataset,
            reasoning_load_base=_REASONING_LOAD,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["SUPPORTED_QUERY_IDS", "SymbolicProbabilityDiceSingleThresholdProbabilityTask"]
