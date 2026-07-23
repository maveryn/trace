"""Deterministic non-semantic color variation for synthetic 3D scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

COLOR_VARIANT_ACCENTS: Tuple[Tuple[int, int, int], ...] = (
    (245, 126, 78),
    (74, 154, 232),
    (79, 190, 130),
    (192, 104, 209),
    (237, 188, 68),
    (61, 193, 196),
    (222, 98, 154),
    (137, 166, 70),
)


def _shade(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(float(channel) * float(factor))))) for channel in rgb)


def _tint(rgb: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    return tuple(
        max(0, min(255, int(round(float(channel) + (255.0 - float(channel)) * float(factor)))))
        for channel in rgb
    )


def _stable_color_index(parts: Sequence[Any], modulo: int) -> int:
    if int(modulo) <= 0:
        return 0
    text = "|".join(str(part) for part in parts)
    value = 2166136261
    for index, char in enumerate(text):
        value ^= ord(char) + index
        value = (value * 16777619) % (2**32)
    return int(value % int(modulo))


def _mix_rgb(a: Tuple[int, int, int], b: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    mix = max(0.0, min(1.0, float(amount)))
    return tuple(
        max(0, min(255, int(round(float(a_channel) * (1.0 - mix) + float(b_channel) * mix))))
        for a_channel, b_channel in zip(a, b)
    )


def resolve_three_d_object_fill_rgb(
    spec: Mapping[str, Any],
    *,
    base_rgb: Tuple[int, int, int] | None = None,
    palette: Sequence[Tuple[int, int, int]] | None = None,
    candidate_palette: Sequence[Tuple[int, int, int]] | None = None,
    candidate_labels: Sequence[str] | None = None,
    salt: str = "",
    variation_strength: float = 0.14,
) -> Tuple[int, int, int]:
    """Resolve a stable, non-semantic render color for one 3D object."""

    object_id = str(spec.get("object_id", ""))
    point_label = str(spec.get("point_label", ""))
    shape_type = str(spec.get("shape_type", spec.get("object_type", "")))
    object_type = str(spec.get("object_type", shape_type))
    if base_rgb is None:
        labels = tuple(str(label) for label in (candidate_labels or ()))
        candidate_colors = tuple(candidate_palette or ())
        if bool(spec.get("is_answer_candidate", False)) and point_label in labels and candidate_colors:
            candidate_index = labels.index(point_label)
            if candidate_index < len(candidate_colors):
                base_rgb = candidate_colors[candidate_index]
            else:
                base_rgb = candidate_colors[
                    _stable_color_index((salt, point_label, "candidate"), len(candidate_colors))
                ]
        else:
            active_palette = tuple(palette or candidate_colors or ((128, 128, 128),))
            base_rgb = active_palette[
                _stable_color_index((salt, object_id, shape_type, object_type), len(active_palette))
            ]
    base = tuple(max(0, min(255, int(channel))) for channel in base_rgb)
    strength = max(0.0, min(0.40, float(variation_strength)))
    if strength <= 0.0:
        return base
    key = (salt, object_id, point_label, shape_type, object_type)
    accent = COLOR_VARIANT_ACCENTS[_stable_color_index((*key, "accent"), len(COLOR_VARIANT_ACCENTS))]
    mix_amount = strength * (0.45 + 0.55 * (_stable_color_index((*key, "mix"), 1000) / 999.0))
    mixed = _mix_rgb(base, accent, mix_amount)
    tone = (0.92, 0.97, 1.0, 1.05, 1.10)[_stable_color_index((*key, "tone"), 5)]
    if tone < 1.0:
        return _shade(mixed, tone)
    if tone > 1.0:
        return _tint(mixed, tone - 1.0)
    return mixed
