"""Objective-neutral trace serialization for solitaire tableau scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts

from .state import SCENE_ID, RenderedSolitaireScene, SolitaireSample


def card_specs(sample: SolitaireSample) -> list[dict[str, Any]]:
    """Serialize tableau cards with their logical positions."""

    return [
        {
            "card_id": str(card.card_id),
            "rank_value": int(card.rank_value),
            "rank_label": str(card.rank_label),
            "suit_name": str(card.suit_name),
            "suit_short": str(card.suit_short),
            "badge_text": None if card.badge_text is None else str(card.badge_text),
            "column_index": int(col_index),
            "row_index": int(row_index),
            "is_exposed": bool(row_index == len(column) - 1),
        }
        for col_index, column in enumerate(sample.columns)
        for row_index, card in enumerate(column)
    ]


def foundation_specs(sample: SolitaireSample) -> list[dict[str, Any]]:
    """Serialize foundation piles with suit and top-rank state."""

    return [
        {
            "foundation_id": str(foundation.foundation_id),
            "label": str(foundation.label),
            "suit_name": str(foundation.suit_name),
            "top_rank_value": int(foundation.top_rank_value),
        }
        for foundation in sample.foundations
    ]


def solitaire_trace_params(
    *,
    sample: SolitaireSample,
    prompt_query_key: str,
    scene_variant_probabilities: Mapping[str, float],
    style_variant: str,
    style_variant_probabilities: Mapping[str, float],
    public_query_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Return common query params plus task-owned sampling metadata."""

    return {
        "scene_variant": str(sample.scene_variant),
        "scene_variant_probabilities": dict(scene_variant_probabilities),
        "prompt_query_key": str(prompt_query_key),
        "style_variant": str(style_variant),
        "style_variant_probabilities": dict(style_variant_probabilities),
        "public_query_probabilities": dict(public_query_probabilities),
        **dict(sample.metadata),
    }


def build_solitaire_trace_payload(
    *,
    sample: SolitaireSample,
    rendered: RenderedSolitaireScene,
    prompt_query_key: str,
    style_variant: str,
    annotation_artifacts: AnnotationArtifacts,
    background_meta: Mapping[str, Any],
    post_noise_meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble solitaire trace sections after task-specific binding."""

    if str(annotation_artifacts.annotation_type) == "bbox_map":
        witness_symbolic = {
            "type": "object_map",
            "ids": {
                "source_card": str(sample.metadata["legal_source_id"]),
                "target": str(sample.metadata["legal_target_id"]),
            },
        }
    elif str(annotation_artifacts.annotation_type) in {"bbox", "point"}:
        witness_symbolic = {
            "type": "object",
            "ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        }
    else:
        witness_symbolic = {
            "type": "object_set",
            "ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
        }
    return {
        "scene_ir": {
            "scene_kind": "games_solitaire_tableau",
            "entities": [dict(entity) for entity in rendered.entities],
            "relations": {
                "scene_variant": str(sample.scene_variant),
                "prompt_query_key": str(prompt_query_key),
                "style_variant": str(style_variant),
                "answer": sample.answer,
                "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            },
        },
        "render_spec": {
            "scene_variant": str(sample.scene_variant),
            "style_variant": str(style_variant),
            "canvas_width": int(rendered.image.size[0]),
            "canvas_height": int(rendered.image.size[1]),
            "column_count": int(len(sample.columns)),
            "style": dict(rendered.style_meta),
            "panel_scene_style": dict(rendered.style_meta.get("panel_scene_style", {})),
            "solitaire_tableau_style": dict(rendered.style_meta.get("solitaire_tableau_style", {})),
            "card_face_style": dict(rendered.style_meta.get("card_face_style", {})),
            "text_style": dict(rendered.style_meta.get("text_style", {})),
            "layout_jitter": dict(rendered.render_map.get("layout_jitter", {})),
        },
        "render_map": dict(rendered.render_map),
        "execution_trace": {
            "scene_variant": str(sample.scene_variant),
            "prompt_query_key": str(prompt_query_key),
            "style_variant": str(style_variant),
            "answer": sample.answer,
            "card_specs": card_specs(sample),
            "foundation_specs": foundation_specs(sample),
            "annotation_entity_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            **dict(sample.metadata),
        },
        "witness_symbolic": dict(witness_symbolic),
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "background": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }
