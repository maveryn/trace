"""Two-tray dice probability for a pair-sum threshold."""

from __future__ import annotations

from ...registry import register_task
from ._lifecycle import run_dice_probability_lifecycle
from .shared.rules import (
    ThresholdEventSpec,
    build_pair_dataset,
    pair_sum_threshold_candidates,
    threshold_at_least,
    threshold_at_most,
)


TASK_ID = "task_symbolic__dice__pair_sum_threshold_probability"
SUPPORTED_QUERY_IDS = (
    "pair_sum_at_least_probability",
    "pair_sum_at_most_probability",
)
_REASONING_LOAD = {
    "pair_sum_at_least_probability": 0.48,
    "pair_sum_at_most_probability": 0.48,
}
_QUERY_SPECS = {
    "pair_sum_at_least_probability": ThresholdEventSpec(
        property_name="sum_at_least",
        description_template="the selected values sum to at least {threshold}",
        thresholds=(5, 6, 7, 8, 9, 10),
        comparator=threshold_at_least,
    ),
    "pair_sum_at_most_probability": ThresholdEventSpec(
        property_name="sum_at_most",
        description_template="the selected values sum to at most {threshold}",
        thresholds=(4, 5, 6, 7, 8, 9),
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
    def candidate_builder(dice_a, dice_b, params, gen_defaults, rng):
        del rng
        try:
            spec = _QUERY_SPECS[str(public_query_id)]
        except KeyError as exc:
            raise ValueError(f"unsupported dice pair threshold query_id: {public_query_id}") from exc
        return pair_sum_threshold_candidates(
            dice_a,
            dice_b,
            params=params,
            gen_defaults=gen_defaults,
            spec=spec,
        )

    return build_pair_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        candidate_builder=candidate_builder,
    )


@register_task
class SymbolicProbabilityDicePairSumThresholdProbabilityTask:
    """Compute a two-tray probability for a pair-sum threshold."""

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


__all__ = ["SUPPORTED_QUERY_IDS", "SymbolicProbabilityDicePairSumThresholdProbabilityTask"]
