"""Scene-local sampling helpers for icon-cutout tasks."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ....shared.labeling import LABEL_POOL_A_L
from ...shared.icon_assets import icon_transform_signature

from .defaults import IconCutoutDefaults


def resolve_icon_cutout_object_count(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: IconCutoutDefaults,
) -> int:
    """Resolve the number of labeled full-icon options."""

    raw = params.get("object_count", group_default(gen_defaults, "object_count_max", defaults.object_count_max))
    count = int(raw)
    if count < 4 or count > len(LABEL_POOL_A_L):
        raise ValueError("object_count must be between 4 and the supported label-pool size")
    return int(count)


def icon_cutout_labels(object_count: int) -> Tuple[str, ...]:
    """Return the visible option labels for an icon-cutout scene."""

    return tuple(str(value) for value in LABEL_POOL_A_L[: int(object_count)])


def resolve_icon_cutout_answer_index(rng, *, params: Mapping[str, Any], labels: Sequence[str]) -> int:
    """Resolve which labeled option is correct."""

    if params.get("answer_index") is not None:
        index = int(params["answer_index"])
        if not 0 <= int(index) < len(labels):
            raise ValueError("answer_index out of range")
        return int(index)
    if params.get("answer_label") is not None:
        label = str(params["answer_label"]).strip().upper()
        normalized_labels = [str(value) for value in labels]
        if label not in set(normalized_labels):
            raise ValueError("answer_label is not supported by the sampled option count")
        return int(normalized_labels.index(label))
    return int(rng.randrange(len(labels)))


def sample_icon_cutout_option_icon_ids(
    rng,
    *,
    pool: Sequence[str],
    correct_icon_id: str,
    correct_index: int,
    object_count: int,
    signature_size_px: int,
) -> Tuple[str, ...]:
    """Sample option icon ids with the correct id at the resolved index."""

    correct_signature = icon_transform_signature(
        str(correct_icon_id),
        int(signature_size_px),
        transform_id="identity",
    )
    candidates = [str(icon_id) for icon_id in pool if str(icon_id) != str(correct_icon_id)]
    rng.shuffle(candidates)
    distractors: List[str] = []
    seen_signatures = {correct_signature}
    for icon_id in candidates:
        signature = icon_transform_signature(str(icon_id), int(signature_size_px), transform_id="identity")
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        distractors.append(str(icon_id))
        if len(distractors) >= int(object_count) - 1:
            break
    if len(distractors) < int(object_count) - 1:
        raise ValueError("icon-cutout task resolved too few visually distinct distractors")

    option_ids: List[str] = []
    cursor = 0
    for index in range(int(object_count)):
        if int(index) == int(correct_index):
            option_ids.append(str(correct_icon_id))
        else:
            option_ids.append(str(distractors[int(cursor)]))
            cursor += 1
    return tuple(str(value) for value in option_ids)


__all__ = [
    "icon_cutout_labels",
    "resolve_icon_cutout_answer_index",
    "resolve_icon_cutout_object_count",
    "sample_icon_cutout_option_icon_ids",
]
