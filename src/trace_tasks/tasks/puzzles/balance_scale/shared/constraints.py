"""Semantic balance-equation constructors reused by public balance tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

from .rules import (
    expanded_item_count,
    expressions_match,
    make_panel,
    maybe_swap_sides,
    side_total,
    unique_equivalent_counts,
    unique_target_values,
)
from .sampling import BALANCE_OBJECT_TYPES, object_specs_for_labels
from .state import (
    EQUIVALENT_COUNT_ROW_KIND,
    MISSING_WEIGHT_ROW_KIND,
    OBJECT_LABELS,
    SCENE_ID,
)

UNKNOWN_LABELS = tuple(str(label) for label in OBJECT_LABELS[:3])


def _collapse_object_items(labels: Sequence[str]) -> list[dict[str, Any]]:
    """Collapse rendered object copies into compact symbolic object terms."""

    counts = {str(label): 0 for label in UNKNOWN_LABELS}
    for label in labels:
        counts[str(label)] += 1
    return [
        {"kind": "object", "object_label": str(label), "count": int(count)}
        for label, count in counts.items()
        if int(count) > 0
    ]


def _used_object_labels(equations: Sequence[Mapping[str, Any]]) -> set[str]:
    """Return all unknown labels appearing in a list of symbolic equations."""

    used: set[str] = set()
    for equation in equations:
        for side_name in ("left_terms", "right_terms"):
            for term in equation[side_name]:
                if str(term["kind"]) == "object":
                    used.add(str(term["object_label"]))
    return used


def _random_compound_terms(
    *,
    rng,
    max_numeric_value: int,
    min_items: int = 2,
    max_items: int = 3,
) -> list[dict[str, Any]]:
    """Sample one pan side with at least one object and two or more items."""

    item_count = int(rng.randint(int(min_items), int(max_items)))
    max_numeric_count = min(1, item_count - 1)
    numeric_count = int(rng.randint(0, max_numeric_count))
    object_count = int(item_count - numeric_count)
    object_labels = [str(rng.choice(UNKNOWN_LABELS)) for _ in range(object_count)]
    terms = _collapse_object_items(object_labels)
    for _index in range(numeric_count):
        terms.append(
            {"kind": "numeric", "value": int(rng.randint(1, int(max_numeric_value)))}
        )
    return terms


def _numeric_terms_for_sum(
    value: int,
    *,
    rng,
    max_numeric_value: int,
) -> list[dict[str, Any]]:
    """Represent one visible numeric total as one or two weight tokens."""

    total = int(value)
    if total <= 0 or total > int(max_numeric_value):
        raise RuntimeError("numeric balance token total is outside render support")
    if total >= 4 and rng.random() < 0.45:
        first = int(rng.randint(1, int(total) - 1))
        second = int(total) - int(first)
        return [
            {"kind": "numeric", "value": int(first)},
            {"kind": "numeric", "value": int(second)},
        ]
    return [{"kind": "numeric", "value": int(total)}]


def _direct_value_panel_terms(
    *,
    label: str,
    weights: Mapping[str, int],
    rng,
    max_numeric_value: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build terms where one unknown object's value is directly readable."""

    object_weight = int(weights[str(label)])
    max_offset = min(12, int(max_numeric_value) - int(object_weight))
    if max_offset < 1:
        raise RuntimeError("direct balance panel has no room for numeric offset")
    offset_total = int(rng.randint(1, int(max_offset)))
    object_side = [
        {"kind": "object", "object_label": str(label), "count": 1},
        *_numeric_terms_for_sum(
            int(offset_total),
            rng=rng,
            max_numeric_value=int(max_numeric_value),
        ),
    ]
    numeric_side = _numeric_terms_for_sum(
        int(object_weight) + int(offset_total),
        rng=rng,
        max_numeric_value=int(max_numeric_value),
    )
    return maybe_swap_sides(object_side, numeric_side, rng=rng)


def _make_equation(
    *,
    equation_id: str,
    left_terms: Sequence[Mapping[str, Any]],
    right_terms: Sequence[Mapping[str, Any]],
    equation_family: str,
) -> dict[str, Any]:
    """Create one traceable symbolic equation record."""

    return {
        "equation_id": str(equation_id),
        "equation_family": str(equation_family),
        "left_terms": [dict(term) for term in left_terms],
        "right_terms": [dict(term) for term in right_terms],
    }


def _compound_panel_terms(
    *,
    rng,
    weights: Mapping[str, int],
    max_numeric_value: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct balanced compound left/right pan terms from hidden weights."""

    for _attempt_index in range(400):
        left_terms = _random_compound_terms(
            rng=rng,
            max_numeric_value=int(max_numeric_value),
        )
        right_terms = _random_compound_terms(
            rng=rng,
            max_numeric_value=int(max_numeric_value),
        )
        left_total = side_total(left_terms, weights)
        right_total = side_total(right_terms, weights)
        if int(left_total) == int(right_total):
            if expressions_match(left_terms, right_terms):
                continue
            return left_terms, right_terms

        if int(left_total) > int(right_total):
            numeric_value = int(left_total) - int(right_total)
            if numeric_value > int(max_numeric_value):
                continue
            if expanded_item_count(right_terms) >= 4:
                continue
            right_terms = [dict(term) for term in right_terms]
            right_terms.append({"kind": "numeric", "value": int(numeric_value)})
        else:
            numeric_value = int(right_total) - int(left_total)
            if numeric_value > int(max_numeric_value):
                continue
            if expanded_item_count(left_terms) >= 4:
                continue
            left_terms = [dict(term) for term in left_terms]
            left_terms.append({"kind": "numeric", "value": int(numeric_value)})

        if (
            2 <= expanded_item_count(left_terms) <= 4
            and 2 <= expanded_item_count(right_terms) <= 4
            and not expressions_match(left_terms, right_terms)
        ):
            return left_terms, right_terms
    raise RuntimeError("failed to construct balanced compound pan terms")


def _compound_panel_terms_with_labels(
    *,
    rng,
    weights: Mapping[str, int],
    max_numeric_value: int,
    required_labels: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct a compound panel that includes every requested label."""

    required = {str(label) for label in required_labels}
    for _attempt_index in range(600):
        left_terms, right_terms = _compound_panel_terms(
            rng=rng,
            weights=weights,
            max_numeric_value=int(max_numeric_value),
        )
        used = _used_object_labels(
            [
                {
                    "left_terms": [dict(term) for term in left_terms],
                    "right_terms": [dict(term) for term in right_terms],
                }
            ]
        )
        if required.issubset(used):
            return left_terms, right_terms
    raise RuntimeError("failed to construct required-label compound panel")


def _build_three_balanced_panels(
    *,
    rng,
    weights: Mapping[str, int],
    max_numeric_value: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create exactly three balanced compound scale panels and equations."""

    panels: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    for panel_index in range(1, 4):
        left_terms, right_terms = _compound_panel_terms(
            rng=rng,
            weights=weights,
            max_numeric_value=int(max_numeric_value),
        )
        left_terms, right_terms = maybe_swap_sides(left_terms, right_terms, rng=rng)
        if expressions_match(left_terms, right_terms):
            raise RuntimeError("generated identical balance-scale pan expressions")
        equations.append(
            _make_equation(
                equation_id=f"compound_constraint_{panel_index}",
                left_terms=left_terms,
                right_terms=right_terms,
                equation_family="compound_balance",
            )
        )
        panels.append(
            make_panel(
                panel_index=int(panel_index),
                left_terms=left_terms,
                right_terms=right_terms,
                weights=weights,
            )
        )
    return panels, equations


def _build_guided_balanced_panels(
    *,
    rng,
    weights: Mapping[str, int],
    direct_labels: Sequence[str],
    max_numeric_value: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create three scales with direct-value panels plus one compound panel."""

    labels = [str(label) for label in direct_labels]
    if not labels:
        raise RuntimeError("guided balance construction needs direct labels")

    panels: list[dict[str, Any]] = []
    equations: list[dict[str, Any]] = []
    for panel_index, label in enumerate(labels[:2], start=1):
        left_terms, right_terms = _direct_value_panel_terms(
            label=str(label),
            weights=weights,
            rng=rng,
            max_numeric_value=int(max_numeric_value),
        )
        equations.append(
            _make_equation(
                equation_id=f"direct_value_constraint_{panel_index}",
                left_terms=left_terms,
                right_terms=right_terms,
                equation_family="direct_single_unknown_value",
            )
        )
        panels.append(
            make_panel(
                panel_index=int(panel_index),
                left_terms=left_terms,
                right_terms=right_terms,
                weights=weights,
            )
        )

    required_for_compound = [
        label for label in UNKNOWN_LABELS if label not in set(labels[:2])
    ]
    if not required_for_compound:
        required_for_compound = list(UNKNOWN_LABELS)
    left_terms, right_terms = _compound_panel_terms_with_labels(
        rng=rng,
        weights=weights,
        max_numeric_value=int(max_numeric_value),
        required_labels=required_for_compound,
    )
    left_terms, right_terms = maybe_swap_sides(left_terms, right_terms, rng=rng)
    equations.append(
        _make_equation(
            equation_id="compound_constraint_3",
            left_terms=left_terms,
            right_terms=right_terms,
            equation_family="compound_balance",
        )
    )
    panels.append(
        make_panel(
            panel_index=3,
            left_terms=left_terms,
            right_terms=right_terms,
            weights=weights,
        )
    )
    return panels, equations


def _object_specs(
    instance_seed: int, params: Mapping[str, Any], namespace: str
) -> Dict[str, Dict[str, Any]]:
    """Assign visual object specs to the three unknown labels."""

    _ = params
    rng = spawn_rng(int(instance_seed), f"{namespace}.object_type_offset")
    object_type_offset = int(
        rng.randrange(len(tuple(BALANCE_OBJECT_TYPES)))
    )
    return object_specs_for_labels(UNKNOWN_LABELS, offset=int(object_type_offset))


def build_missing_weight_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    answer_value: int,
    answer_support: Sequence[int],
    scene_variant: str,
    target_cue_mode: str,
    namespace: str,
    prompt_query_key: str,
) -> Dict[str, Any]:
    """Construct guided balanced scales that determine one object value."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    target_label = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.target_label"),
            UNKNOWN_LABELS,
        )
    )
    max_numeric_value = int(gen_defaults.get("numeric_weight_max", 80))
    support_values = [int(value) for value in answer_support]
    object_specs = _object_specs(int(instance_seed), params, str(namespace))

    for _attempt_index in range(600):
        weights = {label: int(rng.choice(support_values)) for label in UNKNOWN_LABELS}
        weights[target_label] = int(answer_value)
        helper_label = next(label for label in UNKNOWN_LABELS if label != target_label)
        panels, equations = _build_guided_balanced_panels(
            rng=rng,
            weights=weights,
            direct_labels=(target_label, helper_label),
            max_numeric_value=int(max_numeric_value),
        )
        if _used_object_labels(equations) != set(UNKNOWN_LABELS):
            continue
        unique_values = unique_target_values(
            equations=equations,
            labels=UNKNOWN_LABELS,
            target_label=target_label,
            support=answer_support,
        )
        if unique_values != [int(answer_value)]:
            continue
        return {
            "prompt_query_key": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "query_row_kind": MISSING_WEIGHT_ROW_KIND,
            "target_cue_mode": str(target_cue_mode),
            "target_label": target_label,
            "object_labels": list(UNKNOWN_LABELS),
            "object_specs": object_specs,
            "object_weights": dict(weights),
            "panels": panels,
            "equations": equations,
            "construction_mode": "direct_target_value_plus_compound",
            "answer_value": int(answer_value),
            "answer_range": [int(min(answer_support)), int(max(answer_support))],
            "target_answer_support": [int(value) for value in answer_support],
            "annotation_item_id": "missing_value_box",
            "supporting_role_item_ids": {"target_box": "missing_value_box"},
        }
    raise RuntimeError("failed to construct a uniquely solvable balance-scale puzzle")


def build_equivalent_count_dataset(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    answer_value: int,
    answer_support: Sequence[int],
    scene_variant: str,
    target_cue_mode: str,
    namespace: str,
    prompt_query_key: str,
) -> Dict[str, Any]:
    """Construct guided balanced scales implying A equals N copies of B."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    source_label = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.source_label"),
            UNKNOWN_LABELS,
        )
    )
    repeated_support = tuple(label for label in UNKNOWN_LABELS if label != source_label)
    repeated_label = str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.repeated_label"),
            repeated_support,
        )
    )
    max_numeric_value = int(gen_defaults.get("numeric_weight_max", 80))
    repeated_weight_max = int(gen_defaults.get("repeated_object_weight_max", 10))
    object_weight_support_max = int(gen_defaults.get("object_weight_support_max", 24))
    object_specs = _object_specs(int(instance_seed), params, str(namespace))
    weight_support = [
        int(value) for value in range(1, int(object_weight_support_max) + 1)
    ]

    for _attempt_index in range(800):
        repeated_limit = max(
            1,
            min(
                int(repeated_weight_max),
                int(object_weight_support_max) // int(answer_value),
            ),
        )
        repeated_weight = int(rng.randint(1, int(repeated_limit)))
        source_weight = int(answer_value) * int(repeated_weight)
        weights = {
            label: int(rng.randint(1, int(object_weight_support_max)))
            for label in UNKNOWN_LABELS
        }
        weights[source_label] = int(source_weight)
        weights[repeated_label] = int(repeated_weight)
        panels, equations = _build_guided_balanced_panels(
            rng=rng,
            weights=weights,
            direct_labels=(source_label, repeated_label),
            max_numeric_value=int(max_numeric_value),
        )
        if _used_object_labels(equations) != set(UNKNOWN_LABELS):
            continue
        unique_counts = unique_equivalent_counts(
            equations=equations,
            labels=UNKNOWN_LABELS,
            source_label=source_label,
            repeated_label=repeated_label,
            count_support=answer_support,
            weight_support=weight_support,
        )
        if unique_counts != [int(answer_value)]:
            continue
        return {
            "prompt_query_key": str(prompt_query_key),
            "scene_id": SCENE_ID,
            "scene_variant": str(scene_variant),
            "query_row_kind": EQUIVALENT_COUNT_ROW_KIND,
            "target_cue_mode": str(target_cue_mode),
            "target_label": source_label,
            "source_label": source_label,
            "repeated_label": repeated_label,
            "object_labels": list(UNKNOWN_LABELS),
            "object_specs": object_specs,
            "object_weights": dict(weights),
            "panels": panels,
            "equations": equations,
            "construction_mode": "direct_ratio_values_plus_compound",
            "answer_value": int(answer_value),
            "answer_range": [int(min(answer_support)), int(max(answer_support))],
            "target_answer_support": [int(value) for value in answer_support],
            "object_weight_support": list(weight_support),
            "annotation_item_id": "missing_count_box",
            "supporting_role_item_ids": {"target_box": "missing_count_box"},
        }
    raise RuntimeError(
        "failed to construct a uniquely solvable equivalent-count puzzle"
    )


__all__ = [
    "UNKNOWN_LABELS",
    "build_equivalent_count_dataset",
    "build_missing_weight_dataset",
]
