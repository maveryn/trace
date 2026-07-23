"""Output packaging helpers for music-staff task-owned datasets."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from .components import MusicSceneSpec
from .sampling import build_text_option_cards
from .state import MusicStaffDataset


def build_text_option_dataset(
    rng,
    *,
    branch_key: str,
    correct_text: str,
    candidate_texts: Iterable[str],
    annotation_item_ids: Sequence[str],
    spec: MusicSceneSpec,
    scene_variant: str,
    prompt_slots: Mapping[str, str],
    metadata: Mapping[str, Any],
) -> MusicStaffDataset:
    """Attach visible text options and package an option-letter dataset."""

    options, answer_label, option_texts = build_text_option_cards(
        rng,
        correct_text=str(correct_text),
        candidate_texts=candidate_texts,
    )
    return MusicStaffDataset(
        branch_key=str(branch_key),
        answer_type="string",
        answer_value=str(answer_label),
        annotation_item_ids=tuple(str(item) for item in annotation_item_ids),
        spec=replace(spec, option_cards=options),
        scene_variant=str(scene_variant),
        prompt_slots={str(key): str(value) for key, value in prompt_slots.items()},
        metadata={
            **dict(metadata),
            "correct_option_text": str(correct_text),
            "option_texts": list(option_texts),
        },
        target_answer_support=tuple(str(option.label) for option in options),
    )


__all__ = ["build_text_option_dataset"]
