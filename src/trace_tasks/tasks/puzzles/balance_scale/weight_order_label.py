"""Choose the correct lightest-to-heaviest order from balance-scale options."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id

from ._lifecycle import run_balance_scene_lifecycle
from .shared.constraints import UNKNOWN_LABELS
from .shared.rules import (
    expressions_match,
    make_panel,
    maybe_swap_sides,
    order_signature,
    unique_order_signatures,
)
from .shared.sampling import object_specs_for_labels, resolve_scene_axes
from .shared.state import SCENE_ID, WEIGHT_ORDER_ROW_KIND

TASK_ID = "task_puzzles__balance_scale__weight_order_label"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "weight_order_label"
TASK_PROMPT_KEY = "balance_scale_query"
OPTION_LABELS = ("A", "B", "C", "D")
WEIGHT_SUPPORT = tuple(range(1, 21))

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "puzzles",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _object_term(label: str) -> dict[str, Any]:
    return {"kind": "object", "object_label": str(label), "count": 1}


def _numeric_term(value: int) -> dict[str, Any]:
    return {"kind": "numeric", "value": int(value)}


def _collapse_object_labels(labels: Sequence[str]) -> list[dict[str, Any]]:
    """Collapse repeated visible labels into compact object-count terms."""

    counts = {str(label): 0 for label in UNKNOWN_LABELS}
    for label in labels:
        counts[str(label)] += 1
    return [
        {"kind": "object", "object_label": str(label), "count": int(count)}
        for label, count in counts.items()
        if int(count) > 0
    ]


def _all_order_signatures(labels: Sequence[str]) -> list[str]:
    """Return all weak lightest-to-heaviest orders for the option pool."""

    signatures = {
        order_signature(
            {str(label): int(value) for label, value in zip(labels, values)},
            labels,
        )
        for values in product((1, 2, 3), repeat=len(tuple(labels)))
    }
    return sorted(signatures)


def _sample_order_weights(rng) -> dict[str, int]:
    """Sample three hidden object weights, sometimes with one equality pair."""

    labels = list(UNKNOWN_LABELS)
    if rng.random() < 0.45:
        equal_pair = set(rng.sample(labels, 2))
        other_label = next(label for label in labels if label not in equal_pair)
        base = int(rng.randint(4, 14))
        delta = int(rng.randint(2, 5))
        other_value = base + delta if rng.random() < 0.5 else base - delta
        other_value = max(1, min(20, int(other_value)))
        if other_value == base:
            other_value = min(20, base + 1)
        return {
            label: int(base) if label in equal_pair else int(other_value)
            for label in labels
        }

    values = sorted(rng.sample(range(2, 20), 3))
    rng.shuffle(labels)
    return {label: int(value) for label, value in zip(labels, values)}


def _shared_context_terms(
    *,
    left_label: str,
    right_label: str,
    rng,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Compare two labels inside a shared object context."""

    shared_count = int(rng.randint(1, 2))
    shared_labels = [str(rng.choice(UNKNOWN_LABELS)) for _index in range(shared_count)]
    left_terms = _collapse_object_labels([str(left_label), *shared_labels])
    right_terms = _collapse_object_labels([str(right_label), *shared_labels])
    return left_terms, right_terms, "shared_object_context"


def _offset_terms(
    *,
    left_label: str,
    right_label: str,
    weights: Mapping[str, int],
    rng,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Compare two labels with a visible one-sided numeric offset."""

    left_weight = int(weights[str(left_label)])
    right_weight = int(weights[str(right_label)])
    if left_weight == right_weight:
        return (
            [_object_term(str(left_label))],
            [_object_term(str(right_label))],
            ("direct_equality"),
        )

    if left_weight > right_weight:
        heavier_label, lighter_label = str(left_label), str(right_label)
        difference = int(left_weight) - int(right_weight)
    else:
        heavier_label, lighter_label = str(right_label), str(left_label)
        difference = int(right_weight) - int(left_weight)

    if difference >= 2 and rng.random() < 0.55:
        offset_value = int(rng.randint(1, int(difference) - 1))
        heavier_terms = [_object_term(str(heavier_label))]
        lighter_terms = [_object_term(str(lighter_label)), _numeric_term(offset_value)]
        left_terms, right_terms = maybe_swap_sides(
            heavier_terms, lighter_terms, rng=rng
        )
        return left_terms, right_terms, "offset_inequality"

    balanced_lighter_terms = [
        _object_term(str(lighter_label)),
        _numeric_term(int(difference)),
    ]
    balanced_heavier_terms = [_object_term(str(heavier_label))]
    left_terms, right_terms = maybe_swap_sides(
        balanced_lighter_terms,
        balanced_heavier_terms,
        rng=rng,
    )
    return left_terms, right_terms, "offset_balance"


def _comparison_terms_for_pair(
    *,
    left_label: str,
    right_label: str,
    weights: Mapping[str, int],
    rng,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Sample one comparison grammar family for a label pair."""

    if rng.random() < 0.55:
        return _offset_terms(
            left_label=str(left_label),
            right_label=str(right_label),
            weights=weights,
            rng=rng,
        )
    left_terms, right_terms, _family = _shared_context_terms(
        left_label=str(left_label),
        right_label=str(right_label),
        rng=rng,
    )
    left_terms, right_terms = maybe_swap_sides(left_terms, right_terms, rng=rng)
    family = (
        "aggregate_shared_context"
        if max(
            sum(int(term["count"]) for term in terms if str(term["kind"]) == "object")
            for terms in (left_terms, right_terms)
        )
        >= 3
        else "shared_object_context"
    )
    return left_terms, right_terms, family


def _comparison_panel(
    *,
    panel_index: int,
    left_label: str,
    right_label: str,
    weights: Mapping[str, int],
    rng,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one comparison panel from the task's comparison grammar."""

    left_terms, right_terms, family = _comparison_terms_for_pair(
        left_label=str(left_label),
        right_label=str(right_label),
        weights=weights,
        rng=rng,
    )
    if expressions_match(left_terms, right_terms):
        raise RuntimeError("weight-order comparison cannot use identical expressions")
    panel = make_panel(
        panel_index=int(panel_index),
        left_terms=left_terms,
        right_terms=right_terms,
        weights=weights,
        require_balanced=False,
    )
    comparison = {
        "comparison_id": f"comparison_{int(panel_index)}",
        "panel_id": str(panel["panel_id"]),
        "left_terms": [dict(term) for term in left_terms],
        "right_terms": [dict(term) for term in right_terms],
        "balance_state": str(panel["balance_state"]),
        "heavier_side": str(panel["heavier_side"]),
        "comparison_family": str(family),
    }
    return panel, comparison


def _build_weight_order_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
) -> Dict[str, Any]:
    """Construct three comparison scales and four order options."""

    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.dataset")
    object_type_offset = int(
        spawn_rng(int(instance_seed), f"{TASK_ID}.object_type_offset").randrange(11)
    )
    object_specs = object_specs_for_labels(
        UNKNOWN_LABELS,
        offset=int(object_type_offset),
    )
    possible_orders = _all_order_signatures(UNKNOWN_LABELS)

    for _attempt_index in range(300):
        weights = _sample_order_weights(rng)
        panels: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        for panel_index, (left_label, right_label) in enumerate(
            (
                (UNKNOWN_LABELS[0], UNKNOWN_LABELS[1]),
                (UNKNOWN_LABELS[1], UNKNOWN_LABELS[2]),
                (UNKNOWN_LABELS[0], UNKNOWN_LABELS[2]),
            ),
            start=1,
        ):
            panel, comparison = _comparison_panel(
                panel_index=int(panel_index),
                left_label=str(left_label),
                right_label=str(right_label),
                weights=weights,
                rng=rng,
            )
            panels.append(panel)
            comparisons.append(comparison)

        correct_order = order_signature(weights, UNKNOWN_LABELS)
        unique_orders = unique_order_signatures(
            comparisons=comparisons,
            labels=UNKNOWN_LABELS,
            support=WEIGHT_SUPPORT,
        )
        if unique_orders != [correct_order]:
            continue

        distractors = [order for order in possible_orders if order != correct_order]
        rng.shuffle(distractors)
        correct_option_label = str(uniform_choice(rng, OPTION_LABELS))
        correct_option_index = OPTION_LABELS.index(correct_option_label)
        order_texts = list(distractors[: len(OPTION_LABELS) - 1])
        order_texts.insert(int(correct_option_index), correct_order)
        options = [
            {"option_label": str(label), "order_text": str(order_text)}
            for label, order_text in zip(OPTION_LABELS, order_texts)
        ]
        answer_label = next(
            str(option["option_label"])
            for option in options
            if str(option["order_text"]) == correct_order
        )
        return {
            "prompt_query_key": PROMPT_QUERY_KEY,
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "query_row_kind": WEIGHT_ORDER_ROW_KIND,
            "target_cue_mode": "query_row_only",
            "target_label": answer_label,
            "object_labels": list(UNKNOWN_LABELS),
            "object_specs": object_specs,
            "object_weights": dict(weights),
            "panels": panels,
            "equations": [],
            "comparisons": comparisons,
            "construction_mode": "comparison_grammar",
            "correct_order": correct_order,
            "order_options": options,
            "answer_value": answer_label,
            "answer_labels": list(OPTION_LABELS),
            "answer_range": list(OPTION_LABELS),
            "target_answer_support": list(OPTION_LABELS),
            "annotation_item_id": f"option_{answer_label}",
            "supporting_role_item_ids": {
                "selected_option": f"option_{answer_label}",
            },
        }
    raise RuntimeError("failed to construct a uniquely ordered balance puzzle")


@register_task
class PuzzlesBalanceScaleWeightOrderLabelTask:
    """Choose the MCQ option showing the correct object-weight order."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation', 'matching')
    domain = "puzzles"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        """Generate three comparison scales and a four-option order query."""

        selected_query, branch_probs, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.branch",
        )
        axes = resolve_scene_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            namespace=TASK_ID,
            include_target_cue=False,
        )
        return run_balance_scene_lifecycle(
            task_name=TASK_ID,
            domain=self.domain,
            selected_query=str(selected_query),
            branch_probabilities=branch_probs,
            task_params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            axes=axes,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            dataset_builder=lambda seed: _build_weight_order_dataset(
                params=task_params,
                instance_seed=int(seed),
                scene_variant=str(axes.scene_variant),
            ),
            task_prompt_key=TASK_PROMPT_KEY,
            answer_type="string",
            extra_query_params={
                "answer_labels": list(OPTION_LABELS),
                "answer_range": list(OPTION_LABELS),
            },
        )


__all__ = [
    "OPTION_LABELS",
    "PROMPT_QUERY_KEY",
    "PuzzlesBalanceScaleWeightOrderLabelTask",
    "QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
