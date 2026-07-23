"""Shared pages-domain helpers reused across multiple structured-page task families."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant


def resolve_pages_axis_variant(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    supported_variants: Sequence[str],
    task_id: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    axis_namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one semantic or visual page axis with deterministic balancing."""

    rng = spawn_rng(int(instance_seed), f"{task_id}.{axis_namespace}")
    selected_variant, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=[str(item) for item in supported_variants],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_variant),
        variant_probabilities=probabilities,
        supported_variants=[str(item) for item in supported_variants],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{task_id}:{axis_namespace}",
    )
    return str(balanced), {str(key): float(value) for key, value in probabilities.items()}


def projected_document_bbox_annotation(
    bbox_map: Mapping[str, Sequence[float]],
    item_ids: Sequence[str],
) -> Dict[str, Any]:
    """Project ordered document item ids into prompt-facing `bbox_set` annotation."""

    return {
        "bbox_set": [
            list(bbox_map[str(item_id)])
            for item_id in [str(item) for item in item_ids]
            if str(item_id) in bbox_map
        ]
    }


def projected_document_keyed_bbox_annotation(
    bbox_map: Mapping[str, Sequence[float]],
    role_to_item_id: Mapping[str, str],
) -> Dict[str, Any]:
    """Project document item ids into role-bound `bbox_map` annotation."""

    keyed_bboxes = {
        str(role): list(bbox_map[str(item_id)])
        for role, item_id in role_to_item_id.items()
    }
    return {
        "type": "bbox_map",
        "bbox_map": dict(keyed_bboxes),
        "pixel_bbox_map": dict(keyed_bboxes),
    }


def build_document_field_specs(
    templates: Sequence[Mapping[str, str]],
    *,
    visible_values: Mapping[str, str],
) -> List[Dict[str, str]] | None:
    """Build canonical field specs from visible values or return `None` on duplicates/empties."""

    field_specs: List[Dict[str, str]] = []
    seen_values = set()
    for template in templates:
        field_id = str(template["field_id"])
        field_value = str(visible_values[field_id]).strip()
        if not field_value or field_value in seen_values:
            return None
        seen_values.add(field_value)
        field_specs.append(
            {
                "field_id": field_id,
                "field_label": str(template["field_label"]),
                "field_value": field_value,
                "section_id": str(template["section_id"]),
                "section_label": str(template["section_label"]),
                "comparison_kind": str(template.get("comparison_kind", "other")),
                "label_bbox_id": f"{field_id}:label",
                "value_bbox_id": f"{field_id}:value",
            }
        )
    return list(field_specs)


def build_document_section_specs(field_specs: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    """Build ordered unique section specs from document field specs."""

    section_specs: List[Dict[str, Any]] = []
    seen_sections = set()
    for spec in field_specs:
        section_id = str(spec["section_id"])
        if section_id in seen_sections:
            continue
        seen_sections.add(section_id)
        section_specs.append(
            {
                "section_id": section_id,
                "section_label": str(spec["section_label"]),
                "field_ids": [
                    str(other_spec["field_id"])
                    for other_spec in field_specs
                    if str(other_spec["section_id"]) == section_id
                ],
            }
        )
    return list(section_specs)


__all__ = [
    "build_document_field_specs",
    "build_document_section_specs",
    "projected_document_bbox_annotation",
    "projected_document_keyed_bbox_annotation",
    "resolve_pages_axis_variant",
]
