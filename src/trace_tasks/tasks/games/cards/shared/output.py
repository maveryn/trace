"""Trace payload helpers for cards scene tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .rendering import RenderedCardsTaskContext
from .state import SampledHand, RuleSample


def card_specs_trace(rendered_context: RenderedCardsTaskContext, *, include_labels: bool) -> list[Dict[str, Any]]:
    """Serialize rendered card specs for trace payloads."""

    out: list[Dict[str, Any]] = []
    for spec in rendered_context.rendered_scene.card_specs:
        item: Dict[str, Any] = {
            "card_id": str(spec.card_id),
            "rank_label": str(spec.rank_label),
            "rank_value": int(spec.rank_value),
            "suit_name": str(spec.suit_name),
            "is_reference": bool(spec.is_reference),
            "order_index": int(spec.order_index),
        }
        if include_labels:
            item.update(
                {
                    "badge_text": None if spec.badge_text is None else str(spec.badge_text),
                    "group_label": None if spec.group_label is None else str(spec.group_label),
                    "row_index": int(spec.row_index),
                }
            )
        out.append(item)
    return out


def cards_hand_count_trace_params(
    *,
    sample: SampledHand,
    rendered_context: RenderedCardsTaskContext,
    hand_kind: str,
    scene_variant: str,
    style_variant: str,
    target_answer: int,
    target_answer_support: Sequence[int],
    target_answer_probabilities: Mapping[str, float],
    card_count: int,
    card_count_support: Sequence[int],
    card_count_probabilities: Mapping[str, float],
    style_variant_probabilities: Mapping[str, float],
    card_ordering: str,
    query_id_probabilities: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    """Return shared cards hand-count trace params plus task-owned params."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_variant": str(scene_variant),
        "hand_kind": str(hand_kind),
        "style_variant": str(style_variant),
        "style_variant_probabilities": dict(style_variant_probabilities),
        "query_id_probabilities": dict(query_id_probabilities or {}),
        "target_answer": int(target_answer),
        "target_answer_support": [int(value) for value in target_answer_support],
        "target_answer_probabilities": dict(target_answer_probabilities),
        "card_count": int(card_count),
        "card_count_support": [int(value) for value in card_count_support],
        "card_count_probabilities": dict(card_count_probabilities),
        "row_count": int(rendered_scene.render_map["row_count"]),
        "max_cards_per_row": int(rendered_scene.render_map["max_cards_per_row"]),
        "card_ordering": str(card_ordering),
    }


def build_cards_hand_count_trace_payload(
    *,
    annotation_gt: Any,
    projected_annotation: Mapping[str, Any],
    sample: SampledHand,
    rendered_context: RenderedCardsTaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    hand_kind: str,
    scene_variant: str,
    style_variant: str,
    target_answer: int,
    target_answer_support: Sequence[int],
    target_answer_probabilities: Mapping[str, float],
    card_count: int,
    card_count_support: Sequence[int],
    card_count_probabilities: Mapping[str, float],
    style_variant_probabilities: Mapping[str, float],
    card_ordering: str,
) -> Dict[str, Any]:
    """Assemble trace payload for card hand-count tasks."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_ir": {
            "scene_kind": f"games_cards_hand_{str(scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(scene_variant),
                "hand_kind": str(hand_kind),
                "style_variant": str(style_variant),
                "card_count": int(card_count),
                "row_count": int(rendered_scene.render_map["row_count"]),
                "max_cards_per_row": int(rendered_scene.render_map["max_cards_per_row"]),
                "center_label_mode": str(rendered_scene.render_map["center_label_mode"]),
                "target_answer": int(target_answer),
                "annotation_entity_ids": list(sample.annotation_card_ids),
                "annotation_rank_card_ids": {
                    str(rank_label): [str(card_id) for card_id in card_ids]
                    for rank_label, card_ids in sample.keyed_annotation_card_ids
                },
                "reference_card_id": None if sample.reference_card_id is None else str(sample.reference_card_id),
                "card_ordering": str(card_ordering),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(scene_variant),
            "style_variant": str(style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "row_count": int(rendered_scene.render_map["row_count"]),
            "max_cards_per_row": int(rendered_scene.render_map["max_cards_per_row"]),
            "center_label_mode": str(rendered_scene.render_map["center_label_mode"]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style") or {}),
            "font_family": str(rendered_context.render_params.font_family),
            "suit_symbol_font_family": str(rendered_scene.render_map.get("suit_symbol_font_family", "")),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(scene_variant),
            "hand_kind": str(hand_kind),
            "style_variant": str(style_variant),
            "target_answer": int(target_answer),
            "target_answer_support": [int(value) for value in target_answer_support],
            "card_count": int(card_count),
            "card_count_support": [int(value) for value in card_count_support],
            "card_ordering": str(card_ordering),
            "row_count": int(rendered_scene.render_map["row_count"]),
            "max_cards_per_row": int(rendered_scene.render_map["max_cards_per_row"]),
            "rank_sequence": [int(value) for value in sample.rank_sequence],
            "reference_card_id": None if sample.reference_card_id is None else str(sample.reference_card_id),
            "reference_rank_value": None if sample.reference_rank_value is None else int(sample.reference_rank_value),
            "reference_rank_label": None if sample.reference_rank_label is None else str(sample.reference_rank_label),
            "reference_suit_name": None if sample.reference_suit_name is None else str(sample.reference_suit_name),
            "card_specs": card_specs_trace(rendered_context, include_labels=False),
            "annotation_entity_ids": [str(card_id) for card_id in sample.annotation_card_ids],
            "annotation_rank_card_ids": {
                str(rank_label): [str(card_id) for card_id in card_ids]
                for rank_label, card_ids in sample.keyed_annotation_card_ids
            },
            "answer": int(target_answer),
        },
        "witness_symbolic": {"type": "object_set", "ids": [str(card_id) for card_id in sample.annotation_card_ids]},
        "projected_annotation": dict(projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
}


def cards_rule_trace_params(
    *,
    sample: RuleSample,
    rendered_context: RenderedCardsTaskContext,
    style_variant: str,
    style_variant_probabilities: Mapping[str, float],
    extra_trace_params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return shared cards rule-task trace params plus task-owned params."""

    rendered_scene = rendered_context.rendered_scene
    return {
        "scene_variant": str(sample.scene_variant),
        "pattern_kind": str(sample.pattern_kind),
        "style_variant": str(style_variant),
        "style_variant_probabilities": dict(style_variant_probabilities),
        "option_count": int(sample.option_count),
        "cards_per_row": int(sample.cards_per_row),
        **dict(extra_trace_params),
        **dict(sample.metadata),
    }


def build_cards_rule_trace_payload(
    *,
    annotation_gt: Any,
    sample: RuleSample,
    rendered_context: RenderedCardsTaskContext,
    prompt_defaults: Mapping[str, Any],
    prompt_artifacts: Any,
    query_spec: Mapping[str, Any],
    style_variant: str,
    style_variant_probabilities: Mapping[str, float],
) -> Dict[str, Any]:
    """Assemble trace payload for labelled card-rule tasks."""

    rendered_scene = rendered_context.rendered_scene
    if str(annotation_gt.type) == "bbox":
        projected_annotation = {
            "type": "bbox",
            "bbox": list(annotation_gt.value),
            "pixel_bbox": list(annotation_gt.value),
        }
    else:
        projected_annotation = {
            "type": "bbox_set",
            "bbox_set": [list(bbox) for bbox in annotation_gt.value],
            "pixel_bbox_set": [list(bbox) for bbox in annotation_gt.value],
        }
    return {
        "scene_ir": {
            "scene_kind": f"games_cards_hand_{str(sample.scene_variant)}",
            "entities": [dict(entity) for entity in rendered_scene.scene_entities],
            "relations": {
                "scene_variant": str(sample.scene_variant),
                "pattern_kind": str(sample.pattern_kind),
                "style_variant": str(style_variant),
                "card_count": len(sample.cards),
                "option_count": int(sample.option_count),
                "answer_label": str(sample.answer),
                "annotation_entity_ids": list(sample.annotation_card_ids),
            },
        },
        "query_spec": dict(query_spec),
        "render_spec": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(style_variant),
            "canvas_width": int(rendered_context.image.size[0]),
            "canvas_height": int(rendered_context.image.size[1]),
            "row_count": int(rendered_scene.render_map["row_count"]),
            "max_cards_per_row": int(rendered_scene.render_map["max_cards_per_row"]),
            "center_label_mode": str(rendered_scene.render_map["center_label_mode"]),
            "layout_jitter": dict(rendered_scene.render_map.get("layout_jitter", {})),
            "panel_scene_style": dict(rendered_scene.render_map.get("panel_scene_style") or {}),
            "font_family": str(rendered_context.render_params.font_family),
            "suit_symbol_font_family": str(rendered_scene.render_map.get("suit_symbol_font_family", "")),
        },
        "render_map": dict(rendered_scene.render_map),
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "pattern_kind": str(sample.pattern_kind),
            "style_variant": str(style_variant),
            "answer_label": str(sample.answer),
            "card_count": len(sample.cards),
            "option_count": int(sample.option_count),
            "card_specs": card_specs_trace(rendered_context, include_labels=True),
            "annotation_entity_ids": [str(card_id) for card_id in sample.annotation_card_ids],
            **dict(sample.metadata),
        },
        "witness_symbolic": {"type": "object_set", "ids": [str(card_id) for card_id in sample.annotation_card_ids]},
        "projected_annotation": dict(projected_annotation),
        "background": dict(rendered_context.background_meta),
        "post_image_noise": dict(rendered_context.post_noise_meta),
    }



__all__ = [
    "build_cards_hand_count_trace_payload",
    "build_cards_rule_trace_payload",
    "cards_hand_count_trace_params",
    "cards_rule_trace_params",
    "card_specs_trace",
]
