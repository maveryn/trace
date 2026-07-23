"""Single-tray dice probability for visible die attributes."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ...base import TaskOutput
from ...registry import register_task
from ..shared.common import get_int_param as _get_int
from ._lifecycle import (
    build_dice_probability_output,
    load_dice_probability_defaults,
    resolve_dice_query,
)
from .shared.rules import (
    SINGLE_COLOR_OR_ANSWER_TARGETS,
    build_single_dataset,
    color_names,
    numeric_properties,
    valid_favorable_count,
)


TASK_ID = "task_symbolic__dice__single_attribute_probability"
DOMAIN = "symbolic"
SUPPORTED_QUERY_IDS = (
    "single_parity_probability",
    "single_value_set_probability",
    "single_color_and_value_probability",
    "single_color_or_value_probability",
)
_REASONING_LOAD = {
    "single_parity_probability": 0.25,
    "single_value_set_probability": 0.34,
    "single_color_and_value_probability": 0.43,
    "single_color_or_value_probability": 0.47,
}


def _configured_count_limits(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    total: int,
) -> tuple[int, int]:
    min_count = _get_int(params, gen_defaults, "single_favorable_count_min", 2)
    max_count = _get_int(params, gen_defaults, "single_favorable_count_max", max(2, int(total) - 2))
    return int(min_count), int(max_count)


def _parity_candidates(
    dice: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = len(dice)
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for description, predicate, meta in numeric_properties()[:2]:
        favorable = [str(die["die_id"]) for die in dice if predicate(int(die["value"]))]
        if valid_favorable_count(len(favorable), total, min_count=min_count, max_count=max_count):
            candidates.append({"event_description": str(description), "favorable_die_ids": list(favorable), **meta})
    return candidates


def _value_set_candidates(
    dice: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = len(dice)
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for first in range(1, 7):
        for second in range(first + 1, 7):
            values = {int(first), int(second)}
            favorable = [str(die["die_id"]) for die in dice if int(die["value"]) in values]
            if valid_favorable_count(len(favorable), total, min_count=min_count, max_count=max_count):
                candidates.append(
                    {
                        "event_description": f"shows either {first} or {second}",
                        "target_values": [int(first), int(second)],
                        "favorable_die_ids": list(favorable),
                    }
                )
    return candidates


def _color_and_value_candidates(
    dice: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = len(dice)
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for color in color_names(dice):
        for description, predicate, meta in numeric_properties():
            favorable = [
                str(die["die_id"])
                for die in dice
                if str(die["color_name"]) == str(color) and predicate(int(die["value"]))
            ]
            if valid_favorable_count(len(favorable), total, min_count=min_count, max_count=max_count):
                candidates.append(
                    {
                        "event_description": f"is {color} and {description}",
                        "target_color": str(color),
                        "favorable_die_ids": list(favorable),
                        **meta,
                    }
                )
    return candidates


def _color_or_value_candidates(
    dice: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = len(dice)
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for color in color_names(dice):
        for description, predicate, meta in numeric_properties():
            favorable = [
                str(die["die_id"])
                for die in dice
                if str(die["color_name"]) == str(color) or predicate(int(die["value"]))
            ]
            if valid_favorable_count(len(favorable), total, min_count=min_count, max_count=max_count):
                candidates.append(
                    {
                        "event_description": f"is {color} or {description}",
                        "target_color": str(color),
                        "favorable_die_ids": list(favorable),
                        **meta,
                    }
                )
    return candidates


def _build_dataset(
    *,
    public_query_id: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Dict[str, Any]:
    builders = {
        "single_parity_probability": _parity_candidates,
        "single_value_set_probability": _value_set_candidates,
        "single_color_and_value_probability": _color_and_value_candidates,
        "single_color_or_value_probability": _color_or_value_candidates,
    }
    if str(public_query_id) not in builders:
        raise ValueError(f"unsupported dice probability query_id: {public_query_id}")
    answer_targets = (
        SINGLE_COLOR_OR_ANSWER_TARGETS
        if str(public_query_id) == "single_color_or_value_probability"
        else None
    )
    return build_single_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        candidate_builder=builders[str(public_query_id)],
        answer_targets=answer_targets,
    )


@register_task
class SymbolicProbabilityDiceSingleAttributeProbabilityTask:
    """Compute a single-tray probability from visible die attributes."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'logical_composition', 'formula_evaluation')
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


__all__ = ["SymbolicProbabilityDiceSingleAttributeProbabilityTask"]
