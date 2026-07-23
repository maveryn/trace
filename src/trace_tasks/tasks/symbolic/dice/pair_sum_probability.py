"""Two-tray dice probability for an exact pair sum."""

from __future__ import annotations

from ....core.query_ids import SINGLE_QUERY_ID
from ...registry import register_task
from ..shared.common import get_int_param as _get_int
from ._lifecycle import run_dice_probability_lifecycle
from .shared.rules import build_pair_dataset, pair_ids, valid_favorable_count


TASK_ID = "task_symbolic__dice__pair_sum_probability"
_PROMPT_QUERY_KEY = "pair_sum_probability"
_REASONING_LOAD = 0.45


def _sum_candidates(dice_a, dice_b, params, gen_defaults, _rng):
    total = int(len(dice_a) * len(dice_b))
    min_count = _get_int(params, gen_defaults, "pair_favorable_count_min", 2)
    max_count = _get_int(params, gen_defaults, "pair_favorable_count_max", max(2, int(total) - 2))
    candidates = []
    for target in range(3, 12):
        favorable_pairs = pair_ids(
            dice_a,
            dice_b,
            lambda a, b, target=target: int(a["value"]) + int(b["value"]) == int(target),
        )
        if valid_favorable_count(len(favorable_pairs), total, min_count=int(min_count), max_count=int(max_count)):
            candidates.append(
                {
                    "event_description": f"the selected values sum to {target}",
                    "target_sum": int(target),
                    "favorable_pairs": list(favorable_pairs),
                }
            )
    return candidates


def _build_dataset(
    *,
    public_query_id: str,
    params,
    gen_defaults,
    instance_seed: int,
):
    del public_query_id
    return build_pair_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        candidate_builder=_sum_candidates,
    )


@register_task
class SymbolicProbabilityDicePairSumProbabilityTask:
    """Compute a two-tray probability for an exact pair sum."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
    domain = "symbolic"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_dice_probability_lifecycle(
            public_task_id=TASK_ID,
            supported_queries=(SINGLE_QUERY_ID,),
            prompt_query_key=_PROMPT_QUERY_KEY,
            dataset_builder=_build_dataset,
            reasoning_load_base=float(_REASONING_LOAD),
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["SymbolicProbabilityDicePairSumProbabilityTask"]
