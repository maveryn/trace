"""Two-tray dice probability for paired attribute predicates."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from ...base import TaskOutput
from ...registry import register_task
from ..shared.common import get_int_param as _get_int
from ._lifecycle import build_dice_probability_output, load_dice_probability_defaults, resolve_dice_query
from .shared.rules import build_pair_dataset, color_names, numeric_properties, pair_ids, valid_favorable_count


TASK_ID = "task_symbolic__dice__pair_attribute_combo_probability"
DOMAIN = "symbolic"
SUPPORTED_QUERY_IDS = (
    "pair_parity_combo_probability",
    "pair_color_value_combo_probability",
)
_REASONING_LOAD = {
    "pair_parity_combo_probability": 0.46,
    "pair_color_value_combo_probability": 0.54,
}


def _configured_count_limits(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    total: int,
) -> tuple[int, int]:
    min_count = _get_int(params, gen_defaults, "pair_favorable_count_min", 2)
    max_count = _get_int(params, gen_defaults, "pair_favorable_count_max", max(2, int(total) - 2))
    return int(min_count), int(max_count)


def _parity_combo_candidates(
    dice_a: Sequence[Mapping[str, Any]],
    dice_b: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = int(len(dice_a) * len(dice_b))
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for parity_a, predicate_a in (
        ("even", lambda value: int(value) % 2 == 0),
        ("odd", lambda value: int(value) % 2 == 1),
    ):
        for parity_b, predicate_b in (
            ("even", lambda value: int(value) % 2 == 0),
            ("odd", lambda value: int(value) % 2 == 1),
        ):
            favorable_pairs = pair_ids(
                dice_a,
                dice_b,
                lambda a, b, predicate_a=predicate_a, predicate_b=predicate_b: predicate_a(int(a["value"]))
                and predicate_b(int(b["value"])),
            )
            if not valid_favorable_count(len(favorable_pairs), total, min_count=min_count, max_count=max_count):
                continue
            if str(parity_a) == str(parity_b):
                event_description = f"both selected dice show {parity_a} values"
            else:
                event_description = f"the Tray A die shows an {parity_a} value and the Tray B die shows an {parity_b} value"
            candidates.append(
                {
                    "event_description": str(event_description),
                    "tray_a_parity": str(parity_a),
                    "tray_b_parity": str(parity_b),
                    "favorable_pairs": list(favorable_pairs),
                }
            )
    return candidates


def _color_value_combo_candidates(
    dice_a: Sequence[Mapping[str, Any]],
    dice_b: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    _rng,
) -> List[Dict[str, Any]]:
    total = int(len(dice_a) * len(dice_b))
    min_count, max_count = _configured_count_limits(params, gen_defaults, total)
    candidates: List[Dict[str, Any]] = []
    for color in color_names(dice_a):
        for description, predicate, meta in numeric_properties():
            favorable_pairs = pair_ids(
                dice_a,
                dice_b,
                lambda a, b, color=color, predicate=predicate: str(a["color_name"]) == str(color)
                and predicate(int(b["value"])),
            )
            if valid_favorable_count(len(favorable_pairs), total, min_count=min_count, max_count=max_count):
                candidates.append(
                    {
                        "event_description": f"the Tray A die is {color} and the Tray B die {description}",
                        "target_color": str(color),
                        "favorable_pairs": list(favorable_pairs),
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
        "pair_parity_combo_probability": _parity_combo_candidates,
        "pair_color_value_combo_probability": _color_value_combo_candidates,
    }
    if str(public_query_id) not in builders:
        raise ValueError(f"unsupported dice probability query_id: {public_query_id}")
    return build_pair_dataset(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        rng_namespace=f"{TASK_ID}.dataset",
        candidate_builder=builders[str(public_query_id)],
    )


@register_task
class SymbolicProbabilityDicePairAttributeComboProbabilityTask:
    """Compute a two-tray probability from paired attribute predicates."""

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


__all__ = ["SymbolicProbabilityDicePairAttributeComboProbabilityTask"]
