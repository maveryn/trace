"""Equation and comparison rules for balance-scale puzzle construction."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping, Sequence, Tuple


def side_total(terms: Sequence[Mapping[str, Any]], weights: Mapping[str, int]) -> int:
    """Return the numeric value represented by one pan side."""

    total = 0
    for term in terms:
        if str(term["kind"]) == "object":
            total += int(term["count"]) * int(weights[str(term["object_label"])])
        else:
            total += int(term["value"])
    return int(total)


def expanded_item_count(terms: Sequence[Mapping[str, Any]]) -> int:
    """Count rendered pan items after object multiplicities are expanded."""

    total = 0
    for term in terms:
        if str(term["kind"]) == "object":
            total += int(term["count"])
        else:
            total += 1
    return int(total)


def expression_signature(
    terms: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[Any, ...], ...]:
    """Return an order-invariant symbolic signature for one visible pan side."""

    object_counts: Dict[str, int] = {}
    numeric_values: list[int] = []
    for term in terms:
        if str(term["kind"]) == "object":
            label = str(term["object_label"])
            object_counts[label] = object_counts.get(label, 0) + int(term["count"])
        else:
            numeric_values.append(int(term["value"]))
    return tuple(
        [
            ("object", label, int(count))
            for label, count in sorted(object_counts.items())
        ]
        + [("numeric", int(value)) for value in sorted(numeric_values)]
    )


def expressions_match(
    left_terms: Sequence[Mapping[str, Any]],
    right_terms: Sequence[Mapping[str, Any]],
) -> bool:
    """Return true when two pan sides show the same symbolic expression."""

    return expression_signature(left_terms) == expression_signature(right_terms)


def expand_terms(
    terms: Sequence[Mapping[str, Any]],
    *,
    panel_name: str,
    side_name: str,
    weights: Mapping[str, int],
) -> List[Dict[str, Any]]:
    """Expand compact symbolic pan terms into renderable object/number items."""

    items: List[Dict[str, Any]] = []
    for term_index, term in enumerate(terms):
        if str(term["kind"]) == "object":
            label = str(term["object_label"])
            for copy_index in range(int(term["count"])):
                items.append(
                    {
                        "item_id": (
                            f"{panel_name}_{side_name}_object_"
                            f"{term_index}_{copy_index}"
                        ),
                        "kind": "object",
                        "object_label": label,
                        "object_weight": int(weights[label]),
                    }
                )
        else:
            items.append(
                {
                    "item_id": f"{panel_name}_{side_name}_weight_{term_index}",
                    "kind": "numeric",
                    "value": int(term["value"]),
                }
            )
    return items


def make_panel(
    *,
    panel_index: int,
    left_terms: Sequence[Mapping[str, Any]],
    right_terms: Sequence[Mapping[str, Any]],
    weights: Mapping[str, int],
    require_balanced: bool = True,
) -> Dict[str, Any]:
    """Create a traceable scale panel from symbolic left/right pan terms."""

    panel_name = f"scale_{int(panel_index)}"
    left_value = side_total(left_terms, weights)
    right_value = side_total(right_terms, weights)
    if bool(require_balanced) and int(left_value) != int(right_value):
        raise RuntimeError(
            f"unbalanced generated scale panel: {left_value} != {right_value}"
        )
    if int(left_value) == int(right_value):
        balance_state = "balanced"
        heavier_side = "none"
    elif int(left_value) > int(right_value):
        balance_state = "left_heavier"
        heavier_side = "left"
    else:
        balance_state = "right_heavier"
        heavier_side = "right"
    return {
        "panel_id": panel_name,
        "panel_label": f"Scale {int(panel_index)}",
        "left_terms": [dict(term) for term in left_terms],
        "right_terms": [dict(term) for term in right_terms],
        "left_items": expand_terms(
            left_terms,
            panel_name=panel_name,
            side_name="left",
            weights=weights,
        ),
        "right_items": expand_terms(
            right_terms,
            panel_name=panel_name,
            side_name="right",
            weights=weights,
        ),
        "left_total": int(left_value),
        "right_total": int(right_value),
        "is_balanced": bool(left_value == right_value),
        "balance_state": str(balance_state),
        "heavier_side": str(heavier_side),
    }


def term_coefficients(
    terms: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
) -> Tuple[Dict[str, int], int]:
    """Return object coefficients and numeric constant for one pan side."""

    coeffs = {str(label): 0 for label in labels}
    constant = 0
    for term in terms:
        if str(term["kind"]) == "object":
            coeffs[str(term["object_label"])] += int(term["count"])
        else:
            constant += int(term["value"])
    return coeffs, int(constant)


def unique_target_values(
    *,
    equations: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    target_label: str,
    support: Sequence[int],
) -> List[int]:
    """Enumerate target values that satisfy every displayed balance equation."""

    target_values: set[int] = set()
    label_list = [str(label) for label in labels]
    for values in product([int(value) for value in support], repeat=len(label_list)):
        assignment = {label: int(value) for label, value in zip(label_list, values)}
        valid = True
        for equation in equations:
            left_coeffs, left_constant = term_coefficients(
                equation["left_terms"],
                label_list,
            )
            right_coeffs, right_constant = term_coefficients(
                equation["right_terms"],
                label_list,
            )
            left_value = int(left_constant) + sum(
                int(left_coeffs[label]) * int(assignment[label]) for label in label_list
            )
            right_value = int(right_constant) + sum(
                int(right_coeffs[label]) * int(assignment[label])
                for label in label_list
            )
            if int(left_value) != int(right_value):
                valid = False
                break
        if valid:
            target_values.add(int(assignment[str(target_label)]))
    return sorted(int(value) for value in target_values)


def unique_equivalent_counts(
    *,
    equations: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    source_label: str,
    repeated_label: str,
    count_support: Sequence[int],
    weight_support: Sequence[int],
) -> List[int]:
    """Enumerate repeated-object counts implied by the displayed equations."""

    counts: set[int] = set()
    label_list = [str(label) for label in labels]
    supported_counts = {int(value) for value in count_support}
    for values in product(
        [int(value) for value in weight_support], repeat=len(label_list)
    ):
        assignment = {label: int(value) for label, value in zip(label_list, values)}
        valid = True
        for equation in equations:
            left_coeffs, left_constant = term_coefficients(
                equation["left_terms"],
                label_list,
            )
            right_coeffs, right_constant = term_coefficients(
                equation["right_terms"],
                label_list,
            )
            left_value = int(left_constant) + sum(
                int(left_coeffs[label]) * int(assignment[label]) for label in label_list
            )
            right_value = int(right_constant) + sum(
                int(right_coeffs[label]) * int(assignment[label])
                for label in label_list
            )
            if int(left_value) != int(right_value):
                valid = False
                break
        if not valid:
            continue
        repeated_weight = int(assignment[str(repeated_label)])
        source_weight = int(assignment[str(source_label)])
        if repeated_weight <= 0 or source_weight % repeated_weight != 0:
            continue
        count = int(source_weight // repeated_weight)
        if count in supported_counts:
            counts.add(int(count))
    return sorted(int(value) for value in counts)


def order_signature(weights: Mapping[str, int], labels: Sequence[str]) -> str:
    """Return a lightest-to-heaviest weak order such as ``A < B = C``."""

    label_list = sorted(str(label) for label in labels)
    grouped: Dict[int, list[str]] = {}
    for label in label_list:
        grouped.setdefault(int(weights[str(label)]), []).append(str(label))
    groups = [
        " = ".join(sorted(items))
        for _weight, items in sorted(grouped.items(), key=lambda item: item[0])
    ]
    return " < ".join(groups)


def unique_order_signatures(
    *,
    comparisons: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    support: Sequence[int],
) -> List[str]:
    """Enumerate lightest-to-heaviest orders satisfying all shown scale states."""

    signatures: set[str] = set()
    label_list = [str(label) for label in labels]
    for values in product([int(value) for value in support], repeat=len(label_list)):
        assignment = {label: int(value) for label, value in zip(label_list, values)}
        valid = True
        for comparison in comparisons:
            left_value = side_total(comparison["left_terms"], assignment)
            right_value = side_total(comparison["right_terms"], assignment)
            balance_state = str(comparison["balance_state"])
            if balance_state == "balanced" and int(left_value) != int(right_value):
                valid = False
                break
            if balance_state == "left_heavier" and int(left_value) <= int(right_value):
                valid = False
                break
            if balance_state == "right_heavier" and int(left_value) >= int(right_value):
                valid = False
                break
        if valid:
            signatures.add(order_signature(assignment, label_list))
    return sorted(signatures)


def maybe_swap_sides(left_terms, right_terms, *, rng):
    """Randomly swap symbolic pan sides while preserving equation meaning."""

    left = [dict(term) for term in left_terms]
    right = [dict(term) for term in right_terms]
    if rng.random() < 0.5:
        return right, left
    return left, right


__all__ = [
    "expanded_item_count",
    "expression_signature",
    "expressions_match",
    "make_panel",
    "maybe_swap_sides",
    "order_signature",
    "side_total",
    "term_coefficients",
    "unique_equivalent_counts",
    "unique_order_signatures",
    "unique_target_values",
]
