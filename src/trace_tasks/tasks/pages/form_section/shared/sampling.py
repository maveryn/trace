"""Sampling primitives for section-local arithmetic on form-section pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.pages.shared.common import (
    build_document_field_specs,
    build_document_section_specs,
    resolve_pages_axis_variant,
)
from trace_tasks.tasks.pages.shared.text_generation import format_currency_from_cents

from .fields import (
    SECTIONED_DOCUMENT_AMOUNT_SECTION_BY_SCENE,
    SECTIONED_DOCUMENT_FIELD_COUNT_RANGE,
    SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE,
    SUPPORTED_SECTIONED_DOCUMENT_SCENE_VARIANTS,
    build_sectioned_document_values,
)
from .forms import DOCUMENT_SCENE_TITLES


SCENE_VARIANTS: Tuple[str, ...] = SUPPORTED_SECTIONED_DOCUMENT_SCENE_VARIANTS


@dataclass(frozen=True)
class ExpressionPlan:
    """Task-owned arithmetic operation passed into the shared sampler."""

    operation_name: str
    operand_count: int
    operators: Tuple[str, ...]
    sort_operands_descending: bool = False


@dataclass(frozen=True)
class RankPlan:
    """Task-owned rank selection passed into the shared sampler."""

    operation_name: str
    rank_from: str
    rank_position: int
    rank_phrase: str


def _apply_expression(*, start_value: int, operators: Sequence[str], remaining_values: Sequence[int]) -> int:
    """Return the integer-cent result of the configured left-to-right expression."""

    result = int(start_value)
    for operator, value in zip(operators, remaining_values):
        if str(operator) == "+":
            result += int(value)
        elif str(operator) == "-":
            result -= int(value)
        else:
            raise ValueError(f"unsupported arithmetic operator '{operator}'")
    return int(result)


def _sample_operand_specs(
    *,
    candidate_specs: Sequence[Mapping[str, Any]],
    amount_cents: Mapping[str, int],
    expression: ExpressionPlan,
    instance_seed: int,
    sampling_namespace: str,
    attempt: int,
) -> list[Dict[str, Any]]:
    """Sample expression operands from amount candidates in the queried section."""

    operand_count = int(expression.operand_count)
    if len(candidate_specs) < operand_count:
        raise ValueError(
            f"scene_variant candidate pool has {len(candidate_specs)} amount fields; "
            f"expression requires {operand_count}"
        )
    operand_rng = spawn_rng(
        int(instance_seed),
        f"{sampling_namespace}.operands.{expression.operation_name}",
        index=int(attempt),
    )
    sampled = [dict(spec) for spec in operand_rng.sample(list(candidate_specs), operand_count)]
    if bool(expression.sort_operands_descending):
        sampled = sorted(
            sampled,
            key=lambda spec: int(amount_cents[str(spec["field_id"])]),
            reverse=True,
        )
    return sampled


def resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    sampling_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the visual document variant for one form-section page."""

    return resolve_pages_axis_variant(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=SCENE_VARIANTS,
        task_id=str(sampling_namespace),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def build_section_expression_case(
    *,
    expression: ExpressionPlan,
    scene_variant: str,
    instance_seed: int,
    sampling_namespace: str,
) -> Dict[str, Any]:
    """Build a section-local arithmetic case with unique visible operands and result."""

    templates = list(SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE[str(scene_variant)])
    target_section_id, target_section_label = SECTIONED_DOCUMENT_AMOUNT_SECTION_BY_SCENE[str(scene_variant)]
    operators = tuple(str(operator) for operator in expression.operators)

    for attempt in range(96):
        value_rng = spawn_rng(int(instance_seed), f"{sampling_namespace}.values", index=int(attempt))
        visible_values, amount_cents = build_sectioned_document_values(str(scene_variant), value_rng)
        field_specs = build_document_field_specs(templates, visible_values=visible_values)
        if field_specs is None or len(field_specs) != len(templates):
            continue
        seen_values = {str(spec["field_value"]) for spec in field_specs}

        candidate_specs = [
            spec
            for spec in field_specs
            if str(spec["section_id"]) == str(target_section_id)
            and str(spec["comparison_kind"]) == "amount"
        ]
        if len(candidate_specs) <= len(operators):
            raise ValueError(f"scene_variant='{scene_variant}' has too few amount fields in the target section")
        candidate_values = [int(amount_cents[str(spec["field_id"])]) for spec in candidate_specs]
        if len(candidate_values) != len(set(candidate_values)):
            continue

        operand_specs = _sample_operand_specs(
            candidate_specs=candidate_specs,
            amount_cents=amount_cents,
            expression=expression,
            instance_seed=int(instance_seed),
            sampling_namespace=str(sampling_namespace),
            attempt=int(attempt),
        )
        operand_cents = [int(amount_cents[str(spec["field_id"])]) for spec in operand_specs]
        result_cents = _apply_expression(
            start_value=int(operand_cents[0]),
            operators=operators,
            remaining_values=list(operand_cents[1:]),
        )
        if int(result_cents) <= 0:
            continue
        result_value = format_currency_from_cents(int(result_cents))
        if str(result_value) in seen_values:
            continue

        section_specs = build_document_section_specs(field_specs)
        prompt_slots = {
            "section_label": str(target_section_label),
            "first_label": str(operand_specs[0]["field_label"]),
            "second_label": str(operand_specs[1]["field_label"]),
            "third_label": str(operand_specs[2]["field_label"]) if len(operand_specs) >= 3 else "",
        }
        return {
            "scene_variant": str(scene_variant),
            "operation_name": str(expression.operation_name),
            "scene_title": str(DOCUMENT_SCENE_TITLES[str(scene_variant)]),
            "prompt_slots": dict(prompt_slots),
            "question_format": "form_section_expression_value",
            "view_family": "structured_document",
            "field_specs": list(field_specs),
            "section_specs": list(section_specs),
            "field_count": int(len(field_specs)),
            "field_count_range": list(SECTIONED_DOCUMENT_FIELD_COUNT_RANGE),
            "query_section_id": str(target_section_id),
            "query_section_label": str(target_section_label),
            "target_amount_candidate_count": int(len(candidate_specs)),
            "target_amount_candidate_field_ids": [str(spec["field_id"]) for spec in candidate_specs],
            "operand_field_ids": [str(spec["field_id"]) for spec in operand_specs],
            "operand_field_labels": [str(spec["field_label"]) for spec in operand_specs],
            "operand_field_values": [str(spec["field_value"]) for spec in operand_specs],
            "operand_value_bbox_ids": [str(spec["value_bbox_id"]) for spec in operand_specs],
            "operator_sequence": list(operators),
            "expression_operand_cents": list(operand_cents),
            "result_cents": int(result_cents),
            "result_value": str(result_value),
        }
    raise ValueError("failed to build a section-expression document scene with unique visible values")


def build_section_rank_case(
    *,
    rank_plan: RankPlan,
    scene_variant: str,
    instance_seed: int,
    sampling_namespace: str,
) -> Dict[str, Any]:
    """Build a section-local amount ranking case with a unique selected field."""

    templates = list(SECTIONED_DOCUMENT_FIELD_TEMPLATES_BY_SCENE[str(scene_variant)])
    target_section_id, target_section_label = SECTIONED_DOCUMENT_AMOUNT_SECTION_BY_SCENE[str(scene_variant)]
    rank_from = str(rank_plan.rank_from)
    if rank_from not in {"highest", "lowest"}:
        raise ValueError(f"unsupported rank_from={rank_from!r}")
    rank_position = int(rank_plan.rank_position)
    if rank_position <= 0:
        raise ValueError("rank_position must be positive")

    # Values are already deterministic for the scene attempt through the caller's seed.
    # Iterate only to preserve the same uniqueness contract as arithmetic cases.
    for attempt in range(96):
        value_rng = spawn_rng(int(instance_seed), f"{sampling_namespace}.values", index=int(attempt))
        visible_values, amount_cents = build_sectioned_document_values(str(scene_variant), value_rng)
        field_specs = build_document_field_specs(templates, visible_values=visible_values)
        if field_specs is None or len(field_specs) != len(templates):
            continue
        candidate_specs = [
            spec
            for spec in field_specs
            if str(spec["section_id"]) == str(target_section_id)
            and str(spec["comparison_kind"]) == "amount"
        ]
        if len(candidate_specs) < rank_position:
            raise ValueError(
                f"scene_variant='{scene_variant}' has too few amount fields for rank_position={rank_position}"
            )
        candidate_values = [int(amount_cents[str(spec["field_id"])]) for spec in candidate_specs]
        if len(candidate_values) != len(set(candidate_values)):
            continue

        ranked_specs = sorted(
            [dict(spec) for spec in candidate_specs],
            key=lambda spec: int(amount_cents[str(spec["field_id"])]),
            reverse=(rank_from == "highest"),
        )
        selected_spec = dict(ranked_specs[rank_position - 1])
        selected_field_id = str(selected_spec["field_id"])
        selected_amount_cents = int(amount_cents[selected_field_id])
        section_specs = build_document_section_specs(field_specs)
        prompt_slots = {
            "section_label": str(target_section_label),
            "rank_phrase": str(rank_plan.rank_phrase),
        }
        return {
            "scene_variant": str(scene_variant),
            "operation_name": str(rank_plan.operation_name),
            "scene_title": str(DOCUMENT_SCENE_TITLES[str(scene_variant)]),
            "prompt_slots": dict(prompt_slots),
            "question_format": "form_section_ranked_amount_field_label",
            "view_family": "structured_document",
            "field_specs": list(field_specs),
            "section_specs": list(section_specs),
            "field_count": int(len(field_specs)),
            "field_count_range": list(SECTIONED_DOCUMENT_FIELD_COUNT_RANGE),
            "query_section_id": str(target_section_id),
            "query_section_label": str(target_section_label),
            "target_amount_candidate_count": int(len(candidate_specs)),
            "target_amount_candidate_field_ids": [str(spec["field_id"]) for spec in candidate_specs],
            "target_amount_candidate_field_labels": [str(spec["field_label"]) for spec in candidate_specs],
            "target_amount_candidate_field_values": [str(spec["field_value"]) for spec in candidate_specs],
            "rank_from": str(rank_from),
            "rank_position": int(rank_position),
            "rank_phrase": str(rank_plan.rank_phrase),
            "selected_field_id": str(selected_field_id),
            "selected_field_label": str(selected_spec["field_label"]),
            "selected_field_value": str(selected_spec["field_value"]),
            "selected_amount_cents": int(selected_amount_cents),
            "answer_value": str(selected_spec["field_label"]),
        }
    raise ValueError("failed to build a section-rank document scene with unique visible values")


__all__ = [
    "ExpressionPlan",
    "RankPlan",
    "SCENE_VARIANTS",
    "build_section_rank_case",
    "build_section_expression_case",
    "resolve_scene_variant",
]
