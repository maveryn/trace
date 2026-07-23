"""Predicate and count primitives for named-field icon tasks."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ...shared.procedural_named_icons import procedural_named_icon_display_name


BOOLEAN_PREDICATE_AND = "shape_and_attribute"
BOOLEAN_PREDICATE_OR = "shape_or_attribute"
BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE = "shape_without_attribute"
BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE = "attribute_without_shape"
BOOLEAN_PREDICATE_NEITHER = "neither_shape_nor_attribute"
BOOLEAN_PREDICATE_XOR = "exactly_one_shape_or_attribute"
BOOLEAN_PREDICATES: Tuple[str, ...] = (
    BOOLEAN_PREDICATE_AND,
    BOOLEAN_PREDICATE_OR,
    BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE,
    BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE,
    BOOLEAN_PREDICATE_NEITHER,
    BOOLEAN_PREDICATE_XOR,
)

COUNTERFACTUAL_SHAPE_REPLACEMENT = "shape_replacement"
COUNTERFACTUAL_SHAPE_REMOVAL = "shape_removal"
COUNTERFACTUAL_EDIT_KINDS: Tuple[str, ...] = (
    COUNTERFACTUAL_SHAPE_REPLACEMENT,
    COUNTERFACTUAL_SHAPE_REMOVAL,
)


def trace_key(left: str, right: str) -> str:
    """Build identity-sensitive trace field names without shared identity literals."""

    return f"{left}_{right}"


def boolean_predicate_matches(predicate_kind: str, *, is_shape: bool, is_attribute: bool) -> bool:
    """Evaluate one task-owned Boolean predicate on symbolic memberships."""

    if predicate_kind == BOOLEAN_PREDICATE_AND:
        return bool(is_shape and is_attribute)
    if predicate_kind == BOOLEAN_PREDICATE_OR:
        return bool(is_shape or is_attribute)
    if predicate_kind == BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE:
        return bool(is_shape and not is_attribute)
    if predicate_kind == BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE:
        return bool(is_attribute and not is_shape)
    if predicate_kind == BOOLEAN_PREDICATE_NEITHER:
        return bool((not is_shape) and (not is_attribute))
    if predicate_kind == BOOLEAN_PREDICATE_XOR:
        return bool(is_shape) ^ bool(is_attribute)
    raise ValueError(f"unsupported Boolean predicate kind: {predicate_kind}")


def boolean_attribute_matches(sample: Any, instance: Any) -> bool:
    """Return whether a rendered icon matches the sample's target attribute."""

    if str(sample.target_attribute_axis) == "color":
        return str(instance.color_name) == str(sample.target_attribute_value)
    raise ValueError(f"unsupported attribute axis: {sample.target_attribute_axis}")


def boolean_counted_instance_ids(sample: Any, instances: Sequence[Any]) -> Tuple[str, ...]:
    """Return instance ids counted by one Boolean named-field query."""

    return tuple(
        str(instance.instance_id)
        for instance in instances
        if boolean_predicate_matches(
            str(sample.predicate_kind),
            is_shape=str(instance.shape_id) == str(sample.target_shape_id),
            is_attribute=boolean_attribute_matches(sample, instance),
        )
    )


def boolean_entity_partition(sample: Any, instances: Sequence[Any]) -> Dict[str, str]:
    """Return each rendered entity's symbolic Boolean partition."""

    return {
        str(instance.instance_id): (
            "both"
            if str(instance.shape_id) == str(sample.target_shape_id) and boolean_attribute_matches(sample, instance)
            else "shape_only"
            if str(instance.shape_id) == str(sample.target_shape_id)
            else "attribute_only"
            if boolean_attribute_matches(sample, instance)
            else "neither"
        )
        for instance in instances
    }


def boolean_attribute_phrase(sample: Any) -> str:
    """Return the prompt phrase for the sampled target attribute."""

    if str(sample.target_attribute_axis) == "color":
        return f"the color {sample.target_attribute_label}"
    raise ValueError(f"unsupported attribute axis: {sample.target_attribute_axis}")


def boolean_attribute_expression(sample: Any) -> str:
    """Return the trace expression for the sampled attribute match."""

    if str(sample.target_attribute_axis) == "color":
        return "color_name == target_color_name"
    raise ValueError(f"unsupported attribute axis: {sample.target_attribute_axis}")


def boolean_query_expression(sample: Any) -> str:
    """Return a symbolic expression for the sampled Boolean predicate."""

    attribute_expression = boolean_attribute_expression(sample)
    if sample.predicate_kind == BOOLEAN_PREDICATE_AND:
        return f"shape_id == target_shape_id AND {attribute_expression}"
    if sample.predicate_kind == BOOLEAN_PREDICATE_OR:
        return f"shape_id == target_shape_id OR {attribute_expression}"
    if sample.predicate_kind == BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE:
        return f"shape_id == target_shape_id AND NOT ({attribute_expression})"
    if sample.predicate_kind == BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE:
        return f"{attribute_expression} AND shape_id != target_shape_id"
    if sample.predicate_kind == BOOLEAN_PREDICATE_NEITHER:
        return f"shape_id != target_shape_id AND NOT ({attribute_expression})"
    if sample.predicate_kind == BOOLEAN_PREDICATE_XOR:
        return f"(shape_id == target_shape_id) XOR ({attribute_expression})"
    raise ValueError(f"unsupported Boolean predicate kind: {sample.predicate_kind}")


def counterfactual_counted_instance_ids(sample: Any) -> Tuple[str, ...]:
    """Return pre-edit instance ids counted after the hypothetical edit."""

    return tuple(
        f"named_icon_{int(index):02d}"
        for index, spec in enumerate(sample.semantic_specs)
        if bool(spec.counted_after_edit)
    )


def counterfactual_role_by_instance_id(sample: Any) -> Dict[str, Dict[str, Any]]:
    """Return symbolic pre-edit role metadata keyed by rendered instance id."""

    return {
        f"named_icon_{int(index):02d}": {
            "shape_id": str(spec.shape_id),
            "shape_name": procedural_named_icon_display_name(str(spec.shape_id)),
            "counterfactual_role": str(spec.counterfactual_role),
            "counted_after_edit": bool(spec.counted_after_edit),
        }
        for index, spec in enumerate(sample.semantic_specs)
    }


def counterfactual_counted_shape_ids_after_edit(sample: Any) -> Tuple[str, ...]:
    """Return the final counted target-shape ids after applying the edit."""

    counted_shape_ids: list[str] = []
    for spec in sample.semantic_specs:
        if not bool(spec.counted_after_edit):
            continue
        if str(spec.counterfactual_role) == "source_shape_changed_to_target":
            counted_shape_ids.append(str(sample.target_shape_id))
        else:
            counted_shape_ids.append(str(spec.shape_id))
    return tuple(counted_shape_ids)


def pair_arithmetic_matches_operand(spec: Any, operand: Any, *, uses_color_binding: bool) -> bool:
    """Return whether one rendered/semantic icon matches a pair-arithmetic operand."""

    if str(spec.shape_id) != str(operand.shape_id):
        return False
    if bool(uses_color_binding):
        return str(spec.color_name) == str(operand.color_name)
    return True


def pair_arithmetic_counted_instance_ids(sample: Any, instances: Sequence[Any]) -> Tuple[str, ...]:
    """Return rendered instance ids counted by the two arithmetic operands."""

    left_ids = []
    right_ids = []
    for instance in instances:
        if pair_arithmetic_matches_operand(instance, sample.left_operand, uses_color_binding=bool(sample.uses_color_binding)):
            left_ids.append(str(instance.instance_id))
        elif pair_arithmetic_matches_operand(instance, sample.right_operand, uses_color_binding=bool(sample.uses_color_binding)):
            right_ids.append(str(instance.instance_id))
    return tuple(left_ids + right_ids)


def pair_arithmetic_role_by_instance_id(sample: Any) -> Dict[str, str]:
    """Map synthetic instance ids to sampled arithmetic operand roles."""

    return {
        f"named_icon_{int(index):02d}": str(semantic_spec.role)
        for index, semantic_spec in enumerate(sample.semantic_specs)
    }


__all__ = [
    "BOOLEAN_PREDICATES",
    "BOOLEAN_PREDICATE_AND",
    "BOOLEAN_PREDICATE_ATTRIBUTE_WITHOUT_SHAPE",
    "BOOLEAN_PREDICATE_NEITHER",
    "BOOLEAN_PREDICATE_OR",
    "BOOLEAN_PREDICATE_SHAPE_WITHOUT_ATTRIBUTE",
    "BOOLEAN_PREDICATE_XOR",
    "COUNTERFACTUAL_EDIT_KINDS",
    "COUNTERFACTUAL_SHAPE_REMOVAL",
    "COUNTERFACTUAL_SHAPE_REPLACEMENT",
    "boolean_attribute_matches",
    "boolean_attribute_phrase",
    "boolean_counted_instance_ids",
    "boolean_entity_partition",
    "boolean_predicate_matches",
    "boolean_query_expression",
    "counterfactual_counted_instance_ids",
    "counterfactual_counted_shape_ids_after_edit",
    "counterfactual_role_by_instance_id",
    "pair_arithmetic_counted_instance_ids",
    "pair_arithmetic_matches_operand",
    "pair_arithmetic_role_by_instance_id",
    "trace_key",
]
