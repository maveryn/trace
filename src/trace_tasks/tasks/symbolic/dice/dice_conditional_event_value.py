"""Single-tray dice conditional probability."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ..shared.common import get_int_param as _get_int
from ._lifecycle import build_dice_probability_output, load_dice_probability_defaults, resolve_dice_query
from .shared.rules import (
    CONDITIONAL_ANSWER_TARGETS,
    build_conditional_dataset,
    color_names,
    conditional_value_properties,
    numeric_properties,
    sample_value_property_given_color_dice,
)


TASK_ID = "task_symbolic__dice__dice_conditional_event_value"
DOMAIN = "symbolic"
SUPPORTED_QUERY_IDS = (
    "conditional_value_property_given_color_probability",
    "conditional_color_given_value_property_probability",
    "conditional_color_given_value_set_probability",
)
_REASONING_LOAD = {
    "conditional_value_property_given_color_probability": 0.48,
    "conditional_color_given_value_property_probability": 0.50,
    "conditional_color_given_value_set_probability": 0.52,
}


def _conditional_count_limits(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> tuple[int, int, int, int]:
    min_denominator = _get_int(params, gen_defaults, "conditional_denominator_count_min", 3)
    max_denominator = _get_int(params, gen_defaults, "conditional_denominator_count_max", 0)
    min_favorable = _get_int(params, gen_defaults, "conditional_favorable_count_min", 1)
    max_favorable = _get_int(params, gen_defaults, "conditional_favorable_count_max", 0)
    return int(min_denominator), int(max_denominator), int(min_favorable), int(max_favorable)


def _candidate_payloads(
    *,
    given_description: str,
    event_description: str,
    denominator_ids: Sequence[str],
    favorable_ids: Sequence[str],
    extra: Mapping[str, Any] | None,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Dict[str, Any] | None:
    min_denominator, max_denominator, min_favorable, max_favorable = _conditional_count_limits(params, gen_defaults)
    if len(denominator_ids) < int(min_denominator):
        return None
    if int(max_denominator) > 0 and len(denominator_ids) > int(max_denominator):
        return None
    if not (int(min_favorable) <= len(favorable_ids) < len(denominator_ids)):
        return None
    if int(max_favorable) > 0 and len(favorable_ids) > int(max_favorable):
        return None
    payload: Dict[str, Any] = {
        "given_description": str(given_description),
        "event_description": str(event_description),
        "denominator_die_ids": [str(item) for item in denominator_ids],
        "favorable_die_ids": [str(item) for item in favorable_ids],
    }
    if extra:
        payload.update(dict(extra))
    return payload


def _value_property_given_color_candidates(
    dice: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for color in color_names(dice):
        denominator = [str(die["die_id"]) for die in dice if str(die["color_name"]) == str(color)]
        color_dice = [die for die in dice if str(die["color_name"]) == str(color)]
        for description, predicate, meta in conditional_value_properties():
            favorable = [str(die["die_id"]) for die in color_dice if predicate(int(die["value"]))]
            payload = _candidate_payloads(
                given_description=f"the selected die is {color}",
                event_description=str(description),
                denominator_ids=denominator,
                favorable_ids=favorable,
                extra={"target_color": str(color), **meta},
                params=params,
                gen_defaults=gen_defaults,
            )
            if payload is not None:
                candidates.append(payload)
    return candidates


def _color_given_value_property_candidates(
    dice: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for description, predicate, meta in numeric_properties():
        denominator_dice = [die for die in dice if predicate(int(die["value"]))]
        denominator = [str(die["die_id"]) for die in denominator_dice]
        for color in color_names(denominator_dice):
            favorable = [str(die["die_id"]) for die in denominator_dice if str(die["color_name"]) == str(color)]
            payload = _candidate_payloads(
                given_description=f"the selected die {description}",
                event_description=f"is {color}",
                denominator_ids=denominator,
                favorable_ids=favorable,
                extra={"target_color": str(color), **meta},
                params=params,
                gen_defaults=gen_defaults,
            )
            if payload is not None:
                candidates.append(payload)
    return candidates


def _color_given_value_set_candidates(
    dice: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    value_sets: List[Tuple[str, set[int], List[int]]] = []
    for first in range(1, 7):
        for second in range(first + 1, 7):
            value_sets.append((f"shows either {first} or {second}", {int(first), int(second)}, [int(first), int(second)]))
            for third in range(second + 1, 7):
                value_sets.append(
                    (
                        f"shows one of {first}, {second}, or {third}",
                        {int(first), int(second), int(third)},
                        [int(first), int(second), int(third)],
                    )
                )
    for given_description, values, target_values in value_sets:
        denominator_dice = [die for die in dice if int(die["value"]) in values]
        denominator = [str(die["die_id"]) for die in denominator_dice]
        for color in color_names(denominator_dice):
            favorable = [str(die["die_id"]) for die in denominator_dice if str(die["color_name"]) == str(color)]
            payload = _candidate_payloads(
                given_description=str(given_description),
                event_description=f"is {color}",
                denominator_ids=denominator,
                favorable_ids=favorable,
                extra={"target_color": str(color), "target_values": list(target_values)},
                params=params,
                gen_defaults=gen_defaults,
            )
            if payload is not None:
                candidates.append(payload)
    return candidates


def _balanced_value_property_given_color_dice(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    count_min: int,
    count_max: int,
    color_pool_size: int,
    fixed_count: int | None,
    rng,
) -> tuple[Sequence[Mapping[str, Any]], Sequence[str] | None]:
    dice, target_answer = sample_value_property_given_color_dice(
        params=params,
        gen_defaults=gen_defaults,
        count_min=int(count_min),
        count_max=int(count_max),
        color_pool_size=int(color_pool_size),
        fixed_count=fixed_count,
        rng=rng,
    )
    return dice, (str(target_answer),)


def _build_dataset(
    *,
    public_query_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    builders = {
        "conditional_value_property_given_color_probability": _value_property_given_color_candidates,
        "conditional_color_given_value_property_probability": _color_given_value_property_candidates,
        "conditional_color_given_value_set_probability": _color_given_value_set_candidates,
    }
    if str(public_query_id) not in builders:
        raise ValueError(f"unsupported dice probability query_id: {public_query_id}")
    dice_builder = (
        _balanced_value_property_given_color_dice
        if str(public_query_id) == "conditional_value_property_given_color_probability"
        else None
    )
    return build_conditional_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        candidate_builder=builders[str(public_query_id)],
        dice_builder=dice_builder,
        answer_targets=CONDITIONAL_ANSWER_TARGETS,
    )


@register_task
class SymbolicProbabilityDiceConditionalEventValueTask:
    """Compute a visible-top conditional probability from one dice tray."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        del max_attempts
        gen_defaults, render_defaults, prompt_defaults = load_dice_probability_defaults(TASK_ID)
        public_query_id, query_probabilities = resolve_dice_query(
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
            public_task_id=TASK_ID,
            supported_queries=SUPPORTED_QUERY_IDS,
        )
        dataset = _build_dataset(
            public_query_id=str(public_query_id),
            params=params,
            gen_defaults=gen_defaults,
            instance_seed=int(instance_seed),
        )
        return build_dice_probability_output(
            public_task_id=TASK_ID,
            public_query_id=str(public_query_id),
            prompt_query_key=str(public_query_id),
            query_probabilities=query_probabilities,
            dataset=dataset,
            params=params,
            gen_defaults=gen_defaults,
            render_defaults=render_defaults,
            prompt_defaults=prompt_defaults,
            instance_seed=int(instance_seed),
            reasoning_load_base=float(_REASONING_LOAD[str(public_query_id)]),
        )


__all__ = ["SymbolicProbabilityDiceConditionalEventValueTask"]
