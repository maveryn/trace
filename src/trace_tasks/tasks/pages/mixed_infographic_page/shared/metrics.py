"""Target-selection helpers for mixed infographic page tasks."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from ....shared.deterministic_sampling import resolve_selection_index
from .state import (
    ADDITIVE_FIELD_LABELS,
    CATEGORICAL_FIELD_LABELS,
    CONDITION_OPERATORS,
    NUMERIC_FIELD_LABELS,
    RANK_ORDINALS,
    _MixedField,
    _MixedInfographicSpec,
    _MixedItem,
    _MixedModule,
)
from .sampling import (
    resolve_named_variant as _resolve_named_variant,
    resolve_supported_int as _resolve_supported_int,
)


def _select_target(
    *,
    sampling_namespace: str,
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_MixedModule, _MixedItem, _MixedField, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Select one module/item/field lookup target and record uniform axis probabilities."""

    module_count = len(spec.modules)
    module_index = int(params["target_module_index"]) if "target_module_index" in params else int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{sampling_namespace}.target_module",
        )
        % int(module_count)
    )
    if module_index < 0 or module_index >= int(module_count):
        raise ValueError("target_module_index out of range")
    module = spec.modules[int(module_index)]
    item_count = len(module.items)
    field_count = len(module.fields)
    item_index = int(params["target_item_index"]) if "target_item_index" in params else int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{sampling_namespace}.target_item.{module_index}",
        )
        % int(item_count)
    )
    field_index = int(params["target_field_index"]) if "target_field_index" in params else int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{sampling_namespace}.target_field.{module_index}.{item_index}",
        )
        % int(field_count)
    )
    if item_index < 0 or item_index >= int(item_count):
        raise ValueError("target_item_index out of range")
    if field_index < 0 or field_index >= int(field_count):
        raise ValueError("target_field_index out of range")
    module_probs = {str(index): 1.0 / float(module_count) for index in range(int(module_count))}
    item_probs = {str(index): 1.0 / float(item_count) for index in range(int(item_count))}
    field_probs = {str(index): 1.0 / float(field_count) for index in range(int(field_count))}
    if "target_module_index" in params:
        module_probs = {str(int(module_index)): 1.0}
    if "target_item_index" in params:
        item_probs = {str(int(item_index)): 1.0}
    if "target_field_index" in params:
        field_probs = {str(int(field_index)): 1.0}
    return module, module.items[int(item_index)], module.fields[int(field_index)], module_probs, item_probs, field_probs


def _parse_mixed_numeric_value(value: str) -> int:
    text = str(value).strip()
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        raise ValueError(f"mixed infographic value is not numeric: {value!r}")
    return int(digits)


def _field_index(module: _MixedModule, field: _MixedField) -> int:
    for index, candidate in enumerate(module.fields):
        if str(candidate.field_id) == str(field.field_id):
            return int(index)
    raise ValueError("field is not part of module")


def _module_index(spec: _MixedInfographicSpec, module: _MixedModule) -> int:
    for index, candidate in enumerate(spec.modules):
        if str(candidate.module_id) == str(module.module_id):
            return int(index)
    raise ValueError("module is not part of mixed infographic spec")


def _numeric_item_values(module: _MixedModule, field: _MixedField) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for item in module.items:
        visible_value = str(item.values_by_field_id[str(field.field_id)])
        values.append(
            {
                "item_id": str(item.item_id),
                "item_label": str(item.label),
                "field_id": str(field.field_id),
                "field_label": str(field.label),
                "visible_value": str(visible_value),
                "numeric_value": int(_parse_mixed_numeric_value(str(visible_value))),
            }
        )
    return values


def _categorical_item_values(module: _MixedModule, field: _MixedField) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for item in module.items:
        visible_value = str(item.values_by_field_id[str(field.field_id)])
        values.append(
            {
                "item_id": str(item.item_id),
                "item_label": str(item.label),
                "field_id": str(field.field_id),
                "field_label": str(field.label),
                "visible_value": str(visible_value),
            }
        )
    return values


def _condition_phrase_for_operator(operator: str) -> str:
    if str(operator) == "above":
        return "above"
    if str(operator) == "below":
        return "below"
    if str(operator) == "at_least":
        return "at least"
    raise ValueError(f"unsupported condition operator: {operator}")


def _numeric_condition_matches(value: int, *, operator: str, threshold_value: int) -> bool:
    if str(operator) == "above":
        return int(value) > int(threshold_value)
    if str(operator) == "below":
        return int(value) < int(threshold_value)
    if str(operator) == "at_least":
        return int(value) >= int(threshold_value)
    raise ValueError(f"unsupported condition operator: {operator}")


def _threshold_candidate_values(unique_values: Sequence[int], *, operator: str) -> List[int]:
    values = [int(value) for value in sorted({int(value) for value in unique_values})]
    if str(operator) == "above":
        return list(values[:-1])
    if str(operator) in {"below", "at_least"}:
        return list(values[1:])
    raise ValueError(f"unsupported condition operator: {operator}")


def _module_field_pairs(
    *,
    spec: _MixedInfographicSpec,
    allowed_field_labels: Sequence[str],
    predicate: Callable[[_MixedModule, _MixedField], bool] | None = None,
) -> List[Tuple[int, _MixedModule, int, _MixedField]]:
    allowed = {str(label) for label in allowed_field_labels}
    pairs: List[Tuple[int, _MixedModule, int, _MixedField]] = []
    for module_index, module in enumerate(spec.modules):
        for field_index, field in enumerate(module.fields):
            if str(field.label) not in allowed:
                continue
            if predicate is not None and not bool(predicate(module, field)):
                continue
            pairs.append((int(module_index), module, int(field_index), field))
    return pairs


def _select_module_field_pair(
    *,
    sampling_namespace: str,
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
    allowed_field_labels: Sequence[str],
    predicate: Callable[[_MixedModule, _MixedField], bool] | None = None,
) -> Tuple[_MixedModule, _MixedField, Dict[str, float], Dict[str, float]]:
    """Choose an eligible module/field pair; requested params must satisfy predicates."""

    pairs = _module_field_pairs(spec=spec, allowed_field_labels=allowed_field_labels, predicate=predicate)
    if not pairs:
        raise ValueError("no eligible module/field pair in mixed infographic scene")

    requested_module_index = params.get("target_module_index")
    requested_field_index = params.get("target_field_index")
    requested_field_label = params.get("target_field_label")

    filtered_pairs = list(pairs)
    if requested_module_index is not None:
        module_index = int(requested_module_index)
        if module_index < 0 or module_index >= len(spec.modules):
            raise ValueError("target_module_index out of range")
        filtered_pairs = [pair for pair in filtered_pairs if int(pair[0]) == int(module_index)]
    if requested_field_index is not None:
        field_index = int(requested_field_index)
        filtered_pairs = [pair for pair in filtered_pairs if int(pair[2]) == int(field_index)]
    if requested_field_label is not None:
        filtered_pairs = [pair for pair in filtered_pairs if str(pair[3].label) == str(requested_field_label)]
    if not filtered_pairs:
        raise ValueError("requested module/field selection is not eligible for this mixed infographic task")

    if requested_module_index is not None and (requested_field_index is not None or requested_field_label is not None):
        selected_pair = filtered_pairs[0]
    else:
        selected_index = int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{sampling_namespace}.target_module_field",
            )
            % int(len(filtered_pairs))
        )
        selected_pair = filtered_pairs[int(selected_index)]

    selected_module_index, selected_module, selected_field_index, selected_field = selected_pair
    eligible_module_indices = sorted({int(pair[0]) for pair in pairs})
    module_probs = {
        str(index): (1.0 if requested_module_index is not None and int(index) == int(selected_module_index) else 0.0)
        for index in eligible_module_indices
    }
    if requested_module_index is None:
        probability = 1.0 / float(len(eligible_module_indices))
        module_probs = {str(index): float(probability) for index in eligible_module_indices}

    selected_module_fields = [
        pair for pair in pairs if int(pair[0]) == int(selected_module_index)
    ]
    field_probs = {
        str(pair[2]): (
            1.0
            if (requested_field_index is not None or requested_field_label is not None)
            and int(pair[2]) == int(selected_field_index)
            else 0.0
        )
        for pair in selected_module_fields
    }
    if requested_field_index is None and requested_field_label is None:
        probability = 1.0 / float(len(selected_module_fields))
        field_probs = {str(pair[2]): float(probability) for pair in selected_module_fields}
    return selected_module, selected_field, dict(module_probs), dict(field_probs)


def _page_field_values_for_label(
    *,
    spec: _MixedInfographicSpec,
    field_label: str,
) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for module_index, module in enumerate(spec.modules):
        matching_fields = [field for field in module.fields if str(field.label) == str(field_label)]
        if not matching_fields:
            continue
        field = matching_fields[0]
        for value in _numeric_item_values(module, field):
            values.append(
                {
                    "module_index": int(module_index),
                    "module_id": str(module.module_id),
                    "module_title": str(module.title),
                    "module_kind": str(module.kind),
                    "item_id": str(value["item_id"]),
                    "item_label": str(value["item_label"]),
                    "field_id": str(field.field_id),
                    "field_label": str(field.label),
                    "visible_value": str(value["visible_value"]),
                    "numeric_value": int(value["numeric_value"]),
                }
            )
    return values


def _select_page_field_extremum_target(
    *,
    sampling_namespace: str,
    gen_defaults: Mapping[str, Any],
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_MixedModule, _MixedField, Dict[str, Any], Dict[str, float], Dict[str, float]]:
    """Choose a unique page-wide field extremum across modules sharing that field."""

    direction, direction_probs = _resolve_named_variant(
        sampling_namespace=str(sampling_namespace),
        gen_defaults=gen_defaults,
        params=params,
        instance_seed=int(instance_seed),
        supported=("highest", "lowest"),
        explicit_key="rank_direction",
        weights_key="rank_direction_weights",
        balance_flag_key="balanced_rank_direction_sampling",
        namespace="page_field_extremum_direction",
    )
    requested_field_label = params.get("target_field_label")
    candidate_by_label: Dict[str, Dict[str, Any]] = {}
    for field_label in sorted(set(NUMERIC_FIELD_LABELS)):
        if requested_field_label is not None and str(field_label) != str(requested_field_label):
            continue
        values = _page_field_values_for_label(spec=spec, field_label=str(field_label))
        module_ids = sorted({str(value["module_id"]) for value in values})
        if len(module_ids) < 3 or len(values) < 3:
            continue
        numeric_values = [int(value["numeric_value"]) for value in values]
        target_value = max(numeric_values) if str(direction) == "highest" else min(numeric_values)
        winners = [dict(value) for value in values if int(value["numeric_value"]) == int(target_value)]
        if len(winners) != 1:
            continue
        candidate_by_label[str(field_label)] = {
            "field_label": str(field_label),
            "candidate_values": [dict(value) for value in values],
            "winner": dict(winners[0]),
        }
    if not candidate_by_label:
        raise ValueError("no unique page-wide field extremum target in mixed infographic scene")
    labels = sorted(candidate_by_label)
    if requested_field_label is not None:
        selected_label = str(requested_field_label)
    else:
        selected_label = labels[
            int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{sampling_namespace}.page_field_extremum_field.{direction}",
                )
            )
            % int(len(labels))
        ]
    selected = dict(candidate_by_label[str(selected_label)])
    winner = dict(selected["winner"])
    target_module = next(module for module in spec.modules if str(module.module_id) == str(winner["module_id"]))
    target_field = next(field for field in target_module.fields if str(field.field_id) == str(winner["field_id"]))
    candidate_values = sorted(
        [dict(value) for value in selected["candidate_values"]],
        key=lambda value: (str(value["module_id"]), str(value["item_id"]), str(value["field_id"])),
    )
    target = {
        "module_id": str(winner["module_id"]),
        "module_title": str(winner["module_title"]),
        "module_kind": str(winner["module_kind"]),
        "module_index": int(winner["module_index"]),
        "item_id": str(winner["item_id"]),
        "item_label": str(winner["item_label"]),
        "field_id": str(winner["field_id"]),
        "field_label": str(winner["field_label"]),
        "rank_direction": str(direction),
        "rank_order_phrase": "highest to lowest" if str(direction) == "highest" else "lowest to highest",
        "visible_value": str(winner["visible_value"]),
        "numeric_value": int(winner["numeric_value"]),
        "candidate_values": [dict(value) for value in candidate_values],
        "candidate_module_count": len({str(value["module_id"]) for value in candidate_values}),
        "answer_value": str(winner["module_title"]),
    }
    field_label_probs = _uniform_axis_probabilities(
        labels,
        str(selected_label),
        locked=requested_field_label is not None,
    )
    return target_module, target_field, target, dict(field_label_probs), dict(direction_probs)


def _ranked_numeric_values(module: _MixedModule, field: _MixedField, *, direction: str) -> List[Dict[str, Any]]:
    values = _numeric_item_values(module, field)
    numeric_values = [int(value["numeric_value"]) for value in values]
    if len(values) < 2 or len(set(numeric_values)) != len(values):
        return []
    reverse = str(direction) == "highest"
    return sorted(
        [dict(value) for value in values],
        key=lambda value: (int(value["numeric_value"]), str(value["item_label"])),
        reverse=bool(reverse),
    )


def _select_ranked_target(
    *,
    sampling_namespace: str,
    gen_defaults: Mapping[str, Any],
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
    module_predicate: Callable[[_MixedModule], bool] | None = None,
) -> Tuple[
    _MixedModule,
    _MixedField,
    Dict[str, Any],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
]:
    """Choose a unique ranked module-local value with enough ordered candidates."""

    direction, direction_probs = _resolve_named_variant(
        sampling_namespace=str(sampling_namespace),
        gen_defaults=gen_defaults,
        params=params,
        instance_seed=int(instance_seed),
        supported=("highest", "lowest"),
        explicit_key="rank_direction",
        weights_key="rank_direction_weights",
        balance_flag_key="balanced_rank_direction_sampling",
        namespace="rank_direction",
    )
    rank_position, rank_position_support, rank_position_probs = _resolve_supported_int(
        sampling_namespace=str(sampling_namespace),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="rank_position",
        support_key="rank_position_support",
        fallback=(2, 3),
        instance_seed=int(instance_seed),
        namespace=f"rank_position.{direction}",
    )
    module, field, module_probs, field_probs = _select_module_field_pair(
        sampling_namespace=str(sampling_namespace),
        spec=spec,
        params=params,
        instance_seed=int(instance_seed),
        allowed_field_labels=NUMERIC_FIELD_LABELS,
        predicate=lambda candidate_module, candidate_field: (
            module_predicate is None or bool(module_predicate(candidate_module))
        )
        and len(_ranked_numeric_values(candidate_module, candidate_field, direction=str(direction)))
        >= int(rank_position),
    )
    ranked_values = _ranked_numeric_values(module, field, direction=str(direction))
    if len(ranked_values) < int(rank_position):
        raise ValueError("mixed infographic ranked target requires enough unique numeric values")
    target = dict(ranked_values[int(rank_position) - 1])
    target.update(
        {
            "module_id": str(module.module_id),
            "module_title": str(module.title),
            "module_kind": str(module.kind),
            "rank_direction": str(direction),
            "rank_position": int(rank_position),
            "rank_ordinal": str(RANK_ORDINALS.get(int(rank_position), f"{int(rank_position)}th")),
            "rank_order_phrase": "highest to lowest" if str(direction) == "highest" else "lowest to highest",
            "rank_position_support": [int(value) for value in rank_position_support],
            "ranked_values": [dict(value) for value in ranked_values],
            "candidate_values": [dict(value) for value in ranked_values],
        }
    )
    return (
        module,
        field,
        target,
        dict(module_probs),
        dict(field_probs),
        dict(direction_probs),
        dict(rank_position_probs),
    )


def _uniform_axis_probabilities(values: Sequence[Any], selected: Any, *, locked: bool = False) -> Dict[str, float]:
    keys = sorted({str(value) for value in values})
    if not keys:
        return {}
    if bool(locked):
        return {str(selected): 1.0}
    probability = 1.0 / float(len(keys))
    return {str(key): float(probability) for key in keys}


def _select_two_field_condition_target(
    *,
    sampling_namespace: str,
    gen_defaults: Mapping[str, Any],
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[
    _MixedModule,
    _MixedField,
    _MixedField,
    Dict[str, Any],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
]:
    """Find one item uniquely satisfying both numeric and categorical predicates."""

    operator, operator_probs = _resolve_named_variant(
        sampling_namespace=str(sampling_namespace),
        gen_defaults=gen_defaults,
        params=params,
        instance_seed=int(instance_seed),
        supported=CONDITION_OPERATORS,
        explicit_key="condition_operator",
        weights_key="condition_operator_weights",
        balance_flag_key="balanced_condition_operator_sampling",
        namespace="two_field_condition_operator",
    )
    requested_module_index = params.get("target_module_index")
    requested_numeric_field_index = params.get("target_numeric_field_index")
    requested_numeric_field_label = params.get("target_numeric_field_label")
    requested_category_field_index = params.get("target_category_field_index")
    requested_category_field_label = params.get("target_category_field_label")
    requested_category_value = params.get("category_value")

    candidates: List[Dict[str, Any]] = []
    for module_index, module in enumerate(spec.modules):
        if requested_module_index is not None and int(module_index) != int(requested_module_index):
            continue
        numeric_fields = [
            (field_index, field)
            for field_index, field in enumerate(module.fields)
            if str(field.label) in set(NUMERIC_FIELD_LABELS)
        ]
        category_fields = [
            (field_index, field)
            for field_index, field in enumerate(module.fields)
            if str(field.label) in set(CATEGORICAL_FIELD_LABELS)
        ]
        if requested_numeric_field_index is not None:
            numeric_fields = [
                (field_index, field)
                for field_index, field in numeric_fields
                if int(field_index) == int(requested_numeric_field_index)
            ]
        if requested_numeric_field_label is not None:
            numeric_fields = [
                (field_index, field)
                for field_index, field in numeric_fields
                if str(field.label) == str(requested_numeric_field_label)
            ]
        if requested_category_field_index is not None:
            category_fields = [
                (field_index, field)
                for field_index, field in category_fields
                if int(field_index) == int(requested_category_field_index)
            ]
        if requested_category_field_label is not None:
            category_fields = [
                (field_index, field)
                for field_index, field in category_fields
                if str(field.label) == str(requested_category_field_label)
            ]
        for numeric_field_index, numeric_field in numeric_fields:
            numeric_values = _numeric_item_values(module, numeric_field)
            threshold_values = _threshold_candidate_values(
                [int(value["numeric_value"]) for value in numeric_values],
                operator=str(operator),
            )
            if not threshold_values:
                continue
            numeric_by_item = {str(value["item_id"]): dict(value) for value in numeric_values}
            for category_field_index, category_field in category_fields:
                category_values = _categorical_item_values(module, category_field)
                category_by_item = {str(value["item_id"]): dict(value) for value in category_values}
                visible_category_values = sorted({str(value["visible_value"]) for value in category_values})
                for category_value in visible_category_values:
                    if requested_category_value is not None and str(category_value) != str(requested_category_value):
                        continue
                    category_matches = [
                        dict(value) for value in category_values if str(value["visible_value"]) == str(category_value)
                    ]
                    if len(category_matches) < 2:
                        continue
                    category_item_ids = {str(value["item_id"]) for value in category_matches}
                    for threshold_index, threshold_value in enumerate(threshold_values):
                        threshold_visible = next(
                            str(value["visible_value"])
                            for value in numeric_values
                            if int(value["numeric_value"]) == int(threshold_value)
                        )
                        numeric_matches = [
                            dict(value)
                            for value in numeric_values
                            if _numeric_condition_matches(
                                int(value["numeric_value"]),
                                operator=str(operator),
                                threshold_value=int(threshold_value),
                            )
                        ]
                        if len(numeric_matches) < 2:
                            continue
                        numeric_item_ids = {str(value["item_id"]) for value in numeric_matches}
                        matching_ids = sorted(category_item_ids.intersection(numeric_item_ids))
                        if len(matching_ids) != 1:
                            continue
                        matching_item_id = str(matching_ids[0])
                        candidates.append(
                            {
                                "module_index": int(module_index),
                                "module": module,
                                "numeric_field_index": int(numeric_field_index),
                                "numeric_field": numeric_field,
                                "category_field_index": int(category_field_index),
                                "category_field": category_field,
                                "category_value": str(category_value),
                                "threshold_index": int(threshold_index),
                                "threshold_value": int(threshold_value),
                                "threshold_visible": str(threshold_visible),
                                "numeric_matches": [dict(value) for value in numeric_matches],
                                "category_matches": [dict(value) for value in category_matches],
                                "matching_item_id": str(matching_item_id),
                                "matching_numeric_value": dict(numeric_by_item[str(matching_item_id)]),
                                "matching_category_value": dict(category_by_item[str(matching_item_id)]),
                            }
                        )
    if not candidates:
        raise ValueError("no unique two-field condition target in mixed infographic scene")
    selection_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{sampling_namespace}.two_field_condition_target.{operator}",
        )
        % int(len(candidates))
    )
    selected = dict(candidates[int(selection_index)])
    module = selected["module"]
    numeric_field = selected["numeric_field"]
    category_field = selected["category_field"]
    matching_item = next(item for item in module.items if str(item.item_id) == str(selected["matching_item_id"]))
    module_indices = [int(candidate["module_index"]) for candidate in candidates]
    numeric_field_indices = [
        int(candidate["numeric_field_index"])
        for candidate in candidates
        if int(candidate["module_index"]) == int(selected["module_index"])
    ]
    category_field_indices = [
        int(candidate["category_field_index"])
        for candidate in candidates
        if int(candidate["module_index"]) == int(selected["module_index"])
    ]
    category_values = [
        str(candidate["category_value"])
        for candidate in candidates
        if int(candidate["module_index"]) == int(selected["module_index"])
        and int(candidate["category_field_index"]) == int(selected["category_field_index"])
    ]
    threshold_indices = [
        int(candidate["threshold_index"])
        for candidate in candidates
        if int(candidate["module_index"]) == int(selected["module_index"])
        and int(candidate["numeric_field_index"]) == int(selected["numeric_field_index"])
        and int(candidate["category_field_index"]) == int(selected["category_field_index"])
        and str(candidate["category_value"]) == str(selected["category_value"])
    ]
    target = {
        "module_id": str(module.module_id),
        "module_title": str(module.title),
        "module_kind": str(module.kind),
        "item_id": str(matching_item.item_id),
        "item_label": str(matching_item.label),
        "numeric_field_id": str(numeric_field.field_id),
        "numeric_field_label": str(numeric_field.label),
        "category_field_id": str(category_field.field_id),
        "category_field_label": str(category_field.label),
        "category_value": str(selected["category_value"]),
        "condition_operator": str(operator),
        "condition_phrase": str(_condition_phrase_for_operator(str(operator))),
        "threshold_rank_index": int(selected["threshold_index"]),
        "threshold_value": int(selected["threshold_value"]),
        "threshold_visible": str(selected["threshold_visible"]),
        "numeric_matches": [dict(value) for value in selected["numeric_matches"]],
        "category_matches": [dict(value) for value in selected["category_matches"]],
        "matching_numeric_value": dict(selected["matching_numeric_value"]),
        "matching_category_value": dict(selected["matching_category_value"]),
        "answer_value": str(matching_item.label),
    }
    return (
        module,
        numeric_field,
        category_field,
        target,
        _uniform_axis_probabilities(module_indices, int(selected["module_index"]), locked=requested_module_index is not None),
        _uniform_axis_probabilities(
            numeric_field_indices,
            int(selected["numeric_field_index"]),
            locked=requested_numeric_field_index is not None or requested_numeric_field_label is not None,
        ),
        _uniform_axis_probabilities(
            category_field_indices,
            int(selected["category_field_index"]),
            locked=requested_category_field_index is not None or requested_category_field_label is not None,
        ),
        dict(operator_probs),
        _uniform_axis_probabilities(
            category_values,
            str(selected["category_value"]),
            locked=requested_category_value is not None,
        ),
        _uniform_axis_probabilities(threshold_indices, int(selected["threshold_index"])),
    )


def _select_condition_target(
    *,
    sampling_namespace: str,
    gen_defaults: Mapping[str, Any],
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_MixedModule, _MixedField, Dict[str, Any], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Select a numeric threshold whose matching item count is nonzero and nontrivial."""

    module, field, module_probs, field_probs = _select_module_field_pair(
        sampling_namespace=str(sampling_namespace),
        spec=spec,
        params=params,
        instance_seed=int(instance_seed),
        allowed_field_labels=NUMERIC_FIELD_LABELS,
        predicate=lambda candidate_module, candidate_field: len(
            {int(value["numeric_value"]) for value in _numeric_item_values(candidate_module, candidate_field)}
        )
        >= 6,
    )
    operator, operator_probs = _resolve_named_variant(
        sampling_namespace=str(sampling_namespace),
        gen_defaults=gen_defaults,
        params=params,
        instance_seed=int(instance_seed),
        supported=CONDITION_OPERATORS,
        explicit_key="condition_operator",
        weights_key="condition_operator_weights",
        balance_flag_key="balanced_condition_operator_sampling",
        namespace=f"condition_operator.{module.module_id}.{field.field_id}",
    )
    values = _numeric_item_values(module, field)
    unique_values = sorted({int(value["numeric_value"]) for value in values})
    if len(unique_values) < 2:
        raise ValueError("condition count target needs at least two unique numeric values")
    if str(operator) == "above":
        threshold_candidate_values = unique_values[:-1]
        operator_phrase = "above"
    elif str(operator) == "below":
        threshold_candidate_values = unique_values[1:]
        operator_phrase = "below"
    else:
        threshold_candidate_values = unique_values[1:]
        operator_phrase = "at least"
    if not threshold_candidate_values:
        raise ValueError("condition count target has no useful threshold")
    threshold_candidates: List[Dict[str, Any]] = []
    for threshold_index, candidate_value in enumerate(threshold_candidate_values):
        threshold_value = int(candidate_value)
        threshold_visible = next(
            str(value["visible_value"]) for value in values if int(value["numeric_value"]) == int(threshold_value)
        )
        if str(operator) == "above":
            matches = [value for value in values if int(value["numeric_value"]) > int(threshold_value)]
        elif str(operator) == "below":
            matches = [value for value in values if int(value["numeric_value"]) < int(threshold_value)]
        else:
            matches = [value for value in values if int(value["numeric_value"]) >= int(threshold_value)]
        if not matches or len(matches) == len(values):
            continue
        threshold_candidates.append(
            {
                "threshold_index": int(threshold_index),
                "threshold_value": int(threshold_value),
                "threshold_visible": str(threshold_visible),
                "matching_values": [dict(value) for value in matches],
                "answer_value": int(len(matches)),
            }
        )
    if not threshold_candidates:
        raise ValueError("condition count answer must be nonzero and not all visible items")
    if params.get("threshold_rank_index") is not None:
        threshold_index = int(params["threshold_rank_index"])
        if threshold_index < 0 or threshold_index >= len(threshold_candidate_values):
            raise ValueError("threshold_rank_index out of range")
        matching_candidates = [candidate for candidate in threshold_candidates if int(candidate["threshold_index"]) == int(threshold_index)]
        if not matching_candidates:
            raise ValueError("threshold_rank_index selects an invalid condition count")
        selected_candidate = dict(matching_candidates[0])
        answer_count_probs = {str(int(selected_candidate["answer_value"])): 1.0}
    else:
        candidates_by_answer_count: Dict[int, List[Dict[str, Any]]] = {}
        for candidate in threshold_candidates:
            candidates_by_answer_count.setdefault(int(candidate["answer_value"]), []).append(dict(candidate))
        answer_counts = sorted(candidates_by_answer_count)
        requested_answer_count = params.get("condition_answer_count")
        if requested_answer_count is not None:
            selected_answer_count = int(requested_answer_count)
            if int(selected_answer_count) not in candidates_by_answer_count:
                raise ValueError("condition_answer_count cannot be produced by selected module field")
            answer_count_probs = {str(int(selected_answer_count)): 1.0}
        else:
            answer_count_index = int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{sampling_namespace}.condition_answer_count.{module.module_id}.{field.field_id}.{operator}",
                )
                % int(len(answer_counts))
            )
            selected_answer_count = int(answer_counts[int(answer_count_index)])
            answer_count_probs = {str(int(answer_count)): 1.0 / float(len(answer_counts)) for answer_count in answer_counts}
        answer_group = candidates_by_answer_count[int(selected_answer_count)]
        threshold_group_index = int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{sampling_namespace}.threshold.{module.module_id}.{field.field_id}.{operator}.{selected_answer_count}",
            )
            % int(len(answer_group))
        )
        selected_candidate = dict(answer_group[int(threshold_group_index)])
    threshold_probs = {str(index): 0.0 for index in range(len(threshold_candidate_values))}
    if params.get("threshold_rank_index") is not None:
        threshold_probs[str(int(selected_candidate["threshold_index"]))] = 1.0
    else:
        candidates_by_answer_count = {}
        for candidate in threshold_candidates:
            candidates_by_answer_count.setdefault(int(candidate["answer_value"]), []).append(dict(candidate))
        for answer_count, candidates in candidates_by_answer_count.items():
            answer_probability = float(answer_count_probs.get(str(int(answer_count)), 0.0))
            if answer_probability <= 0.0:
                continue
            threshold_probability = answer_probability / float(len(candidates))
            for candidate in candidates:
                threshold_probs[str(int(candidate["threshold_index"]))] = float(threshold_probability)
    target = {
        "module_id": str(module.module_id),
        "module_title": str(module.title),
        "module_kind": str(module.kind),
        "field_id": str(field.field_id),
        "field_label": str(field.label),
        "condition_operator": str(operator),
        "condition_phrase": str(operator_phrase),
        "threshold_rank_index": int(selected_candidate["threshold_index"]),
        "threshold_value": int(selected_candidate["threshold_value"]),
        "threshold_visible": str(selected_candidate["threshold_visible"]),
        "candidate_values": [dict(value) for value in values],
        "matching_values": [dict(value) for value in selected_candidate["matching_values"]],
        "answer_value": int(selected_candidate["answer_value"]),
    }
    return module, field, target, dict(module_probs), dict(field_probs), dict(operator_probs), dict(threshold_probs), dict(answer_count_probs)


def _select_total_target(
    *,
    sampling_namespace: str,
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_MixedModule, _MixedField, Dict[str, Any], Dict[str, float], Dict[str, float]]:
    module, field, module_probs, field_probs = _select_module_field_pair(
        sampling_namespace=str(sampling_namespace),
        spec=spec,
        params=params,
        instance_seed=int(instance_seed),
        allowed_field_labels=ADDITIVE_FIELD_LABELS,
        predicate=lambda candidate_module, candidate_field: len(candidate_module.items) >= 2
        and str(candidate_field.label) in set(ADDITIVE_FIELD_LABELS),
    )
    values = _numeric_item_values(module, field)
    total_value = int(sum(int(value["numeric_value"]) for value in values))
    target = {
        "module_id": str(module.module_id),
        "module_title": str(module.title),
        "module_kind": str(module.kind),
        "field_id": str(field.field_id),
        "field_label": str(field.label),
        "summed_values": [dict(value) for value in values],
        "answer_value": int(total_value),
    }
    return module, field, target, dict(module_probs), dict(field_probs)


def _module_field_total_payload(
    *,
    module_index: int,
    module: _MixedModule,
    field_label: str,
) -> Dict[str, Any] | None:
    matching_fields = [field for field in module.fields if str(field.label) == str(field_label)]
    if not matching_fields or len(module.items) < 2:
        return None
    field = matching_fields[0]
    values = _numeric_item_values(module, field)
    return {
        "module_index": int(module_index),
        "module_id": str(module.module_id),
        "module_title": str(module.title),
        "module_kind": str(module.kind),
        "field_id": str(field.field_id),
        "field_label": str(field.label),
        "summed_values": [dict(value) for value in values],
        "total_value": int(sum(int(value["numeric_value"]) for value in values)),
    }


def _select_two_module_total_comparison_target(
    *,
    sampling_namespace: str,
    spec: _MixedInfographicSpec,
    params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[_MixedModule, _MixedField, _MixedModule, _MixedField, Dict[str, Any], Dict[str, float], Dict[str, float]]:
    """Select two modules with unequal additive-field totals for a visible comparison."""

    requested_field_label = params.get("target_field_label")
    requested_module_a_index = params.get("target_module_a_index")
    requested_module_b_index = params.get("target_module_b_index")
    candidates_by_label: Dict[str, List[Dict[str, Any]]] = {}
    for field_label in sorted(set(ADDITIVE_FIELD_LABELS)):
        if requested_field_label is not None and str(field_label) != str(requested_field_label):
            continue
        module_totals = [
            payload
            for module_index, module in enumerate(spec.modules)
            for payload in [_module_field_total_payload(module_index=module_index, module=module, field_label=str(field_label))]
            if payload is not None
        ]
        if len(module_totals) < 2:
            continue
        pairs: List[Dict[str, Any]] = []
        for module_a in module_totals:
            if requested_module_a_index is not None and int(module_a["module_index"]) != int(requested_module_a_index):
                continue
            for module_b in module_totals:
                if int(module_a["module_index"]) == int(module_b["module_index"]):
                    continue
                if requested_module_b_index is not None and int(module_b["module_index"]) != int(requested_module_b_index):
                    continue
                if int(module_a["total_value"]) == int(module_b["total_value"]):
                    continue
                winning_side = "module_a" if int(module_a["total_value"]) > int(module_b["total_value"]) else "module_b"
                winning_module = module_a if str(winning_side) == "module_a" else module_b
                pairs.append(
                    {
                        "field_label": str(field_label),
                        "module_a": dict(module_a),
                        "module_b": dict(module_b),
                        "winning_side": str(winning_side),
                        "answer_value": str(winning_module["module_title"]),
                    }
                )
        if pairs:
            candidates_by_label[str(field_label)] = [dict(pair) for pair in pairs]
    if not candidates_by_label:
        raise ValueError("no unequal two-module total comparison target in mixed infographic scene")
    labels = sorted(candidates_by_label)
    if requested_field_label is not None:
        selected_label = str(requested_field_label)
    else:
        selected_label = labels[
            int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{sampling_namespace}.two_module_total_field",
                )
            )
            % int(len(labels))
        ]
    pair_candidates = [dict(pair) for pair in candidates_by_label[str(selected_label)]]
    selected_pair = pair_candidates[
        int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{sampling_namespace}.two_module_total_pair.{selected_label}",
            )
        )
        % int(len(pair_candidates))
    ]
    module_a_payload = dict(selected_pair["module_a"])
    module_b_payload = dict(selected_pair["module_b"])
    module_a = spec.modules[int(module_a_payload["module_index"])]
    module_b = spec.modules[int(module_b_payload["module_index"])]
    field_a = next(field for field in module_a.fields if str(field.field_id) == str(module_a_payload["field_id"]))
    field_b = next(field for field in module_b.fields if str(field.field_id) == str(module_b_payload["field_id"]))
    pair_keys = [
        f'{int(pair["module_a"]["module_index"])}:{int(pair["module_b"]["module_index"])}'
        for pair in pair_candidates
    ]
    selected_pair_key = f'{int(module_a_payload["module_index"])}:{int(module_b_payload["module_index"])}'
    target = {
        "field_label": str(selected_label),
        "module_a": dict(module_a_payload),
        "module_b": dict(module_b_payload),
        "module_a_total": int(module_a_payload["total_value"]),
        "module_b_total": int(module_b_payload["total_value"]),
        "winning_side": str(selected_pair["winning_side"]),
        "answer_value": str(selected_pair["answer_value"]),
    }
    return (
        module_a,
        field_a,
        module_b,
        field_b,
        dict(target),
        _uniform_axis_probabilities(labels, str(selected_label), locked=requested_field_label is not None),
        _uniform_axis_probabilities(
            pair_keys,
            str(selected_pair_key),
            locked=requested_module_a_index is not None and requested_module_b_index is not None,
        ),
    )

__all__ = [
    "_categorical_item_values",
    "_condition_phrase_for_operator",
    "_numeric_item_values",
    "_parse_mixed_numeric_value",
    "_select_condition_target",
    "_select_page_field_extremum_target",
    "_select_ranked_target",
    "_select_target",
    "_select_total_target",
    "_select_two_field_condition_target",
    "_select_two_module_total_comparison_target",
]
