"""Neutral sampling helpers for Venn-field icon scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.color_format import format_named_color_with_hex
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map
from ....shared.named_colors import available_named_colors, named_color
from ....shared.weighted_sampling import sample_weighted_value, weighted_probability_map
from ...shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_FILL_STYLES,
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
    procedural_named_icon_fill_style_probability_map,
    sample_procedural_named_icon_fill_style,
    validate_procedural_named_icon_fill_style_support,
)

from .defaults import VennFieldDefaults
from .state import (
    NamedColorEntry,
    TargetPredicateSample,
    VennCountInputs,
    VennCountSample,
)


def int_bounds(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    low_key: str,
    high_key: str,
    fallback_low: int,
    fallback_high: int,
) -> Tuple[int, int]:
    """Resolve inclusive integer bounds from params/defaults."""

    low = int(params.get(low_key, group_default(defaults, low_key, fallback_low)))
    high = int(params.get(high_key, group_default(defaults, high_key, fallback_high)))
    if low < 0 or high < low:
        raise ValueError(f"invalid {low_key}/{high_key} bounds")
    return int(low), int(high)


def shape_support(
    params: Mapping[str, Any], defaults: Mapping[str, Any]
) -> Tuple[str, ...]:
    """Resolve procedural named-icon shape support."""

    raw = params.get(
        "shape_id_support",
        group_default(defaults, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(str(value) for value in raw)
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    support = tuple(dict.fromkeys(values))
    if len(support) < 5:
        raise ValueError("shape_id_support must include at least five shapes")
    return support


def color_support(
    params: Mapping[str, Any], defaults: Mapping[str, Any]
) -> Tuple[NamedColorEntry, ...]:
    """Resolve semantic named-color support."""

    color_by_name = {
        str(name): tuple(int(channel) for channel in rgb)
        for name, rgb in available_named_colors()
    }
    raw = params.get(
        "named_color_support",
        group_default(defaults, "named_color_support", tuple(color_by_name)),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("named_color_support must be a sequence")
    names = tuple(
        dict.fromkeys(str(value).strip().lower() for value in raw if str(value).strip())
    )
    unsupported = sorted(set(names) - set(color_by_name))
    if unsupported:
        raise ValueError(f"unsupported named colors: {unsupported}")
    if len(names) < 2:
        raise ValueError("named_color_support must include at least two colors")
    return tuple(
        NamedColorEntry(
            name=str(name),
            rgb=tuple(int(channel) for channel in named_color(str(name))),
            label=format_named_color_with_hex(str(name), named_color(str(name))),
        )
        for name in names
    )


def fill_style_support(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    fallback: Sequence[str] = PROCEDURAL_NAMED_ICON_FILL_STYLES,
) -> Tuple[str, ...]:
    """Resolve named-icon fill-style support used as render variation."""

    key = "named_icon_fill_style_support"
    raw = params.get(key, group_default(defaults, key, tuple(fallback)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = tuple(fallback)
    return validate_procedural_named_icon_fill_style_support(
        tuple(str(value) for value in raw)
    )


def fill_style_probabilities(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    support: Sequence[str],
) -> Dict[str, float]:
    """Resolve fill-style sampling probabilities."""

    raw = params.get(
        "named_icon_fill_style_weights",
        group_default(defaults, "named_icon_fill_style_weights", None),
    )
    if not isinstance(raw, Mapping):
        raw = None
    return procedural_named_icon_fill_style_probability_map(
        tuple(str(value) for value in support), dict(raw) if raw is not None else None
    )


def uniform_string_probability_map(
    values: Sequence[str], *, selected: str | None = None
) -> Dict[str, float]:
    """Return a uniform probability map, or a point mass for explicit values."""

    support = tuple(str(value) for value in values)
    if selected is not None:
        return {str(selected): 1.0}
    probability = 1.0 / float(len(support))
    return {str(value): probability for value in support}


def target_mode_probabilities(
    params: Mapping[str, Any], defaults: Mapping[str, Any]
) -> Dict[str, float]:
    """Resolve target-predicate mode probabilities for task-owned selection."""

    raw = params.get(
        "target_attribute_mode_weights",
        group_default(defaults, "target_attribute_mode_weights", None),
    )
    if not isinstance(raw, Mapping):
        raw = {"shape_only": 0.5, "color_shape": 0.5}
    return weighted_probability_map(("shape_only", "color_shape"), raw)


def target_description(
    *, mode: str, shape_id: str, target_color: NamedColorEntry | None
) -> str:
    """Return prompt-facing target phrase for one sampled predicate."""

    shape_name = procedural_named_icon_display_name(str(shape_id))
    quoted_shape = f'"{shape_name}"'
    if str(mode) == "shape_only":
        return f"{quoted_shape} icons"
    if str(mode) == "color_shape":
        if target_color is None:
            raise ValueError("color_shape target is missing target_color")
        return f"{target_color.label} {quoted_shape} icons"
    raise ValueError(f"unsupported target mode: {mode}")


def default_target_mode_support() -> Tuple[str, ...]:
    """Return supported semantic target predicate modes."""

    return ("shape_only", "color_shape")


def sample_target_predicate(
    rng: Any,
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    shape_ids: Sequence[str],
    colors: Sequence[NamedColorEntry],
    mode_support: Sequence[str] | None = None,
) -> TargetPredicateSample:
    """Sample or resolve one shape/color target predicate."""

    modes = tuple(
        str(value) for value in (mode_support or default_target_mode_support())
    )
    probabilities = target_mode_probabilities(params, defaults)
    explicit_mode = params.get("target_attribute_mode")
    if explicit_mode is not None:
        mode = str(explicit_mode)
        if mode not in set(modes):
            raise ValueError(f"target_attribute_mode must be one of {modes}")
        mode_probabilities = {value: (1.0 if value == mode else 0.0) for value in modes}
    else:
        mode = str(sample_weighted_value(rng, modes, probabilities))
        mode_probabilities = dict(probabilities)

    target_shape_id, shape_probabilities = sample_target_shape(
        rng,
        params=params,
        shape_ids=shape_ids,
    )
    target_color, color_probabilities = sample_target_color(
        rng,
        params=params,
        mode=str(mode),
        colors=colors,
    )
    return TargetPredicateSample(
        mode=str(mode),
        mode_probabilities=dict(mode_probabilities),
        shape_id=str(target_shape_id),
        shape_probabilities=dict(shape_probabilities),
        color=target_color,
        color_probabilities=dict(color_probabilities),
    )


def sample_target_shape(
    rng: Any,
    *,
    params: Mapping[str, Any],
    shape_ids: Sequence[str],
) -> tuple[str, Dict[str, float]]:
    """Sample or resolve one target shape id."""

    explicit_shape = params.get("shape_id", params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(shape_ids):
            raise ValueError(f"target shape must be one of {tuple(shape_ids)}")
        return target_shape_id, uniform_string_probability_map(
            tuple(shape_ids), selected=target_shape_id
        )
    target_shape_id = str(rng.choice(tuple(shape_ids)))
    return target_shape_id, uniform_string_probability_map(tuple(shape_ids))


def sample_target_color(
    rng: Any,
    *,
    params: Mapping[str, Any],
    mode: str,
    colors: Sequence[NamedColorEntry],
) -> tuple[NamedColorEntry | None, Dict[str, float]]:
    """Sample or resolve one target color when the target mode uses color."""

    names = tuple(str(entry.name) for entry in colors)
    if str(mode) != "color_shape":
        return None, uniform_string_probability_map(names)
    color_by_name = {str(entry.name): entry for entry in colors}
    explicit_color = params.get("color_name", params.get("target_color_name"))
    if explicit_color is not None:
        color_name = str(explicit_color).strip().lower()
        if color_name not in color_by_name:
            raise ValueError(f"target color must be one of {names}")
        return color_by_name[str(color_name)], uniform_string_probability_map(
            names, selected=str(color_name)
        )
    color = rng.choice(tuple(colors))
    return color_by_name[str(color.name)], uniform_string_probability_map(names)


def sample_target_and_object_counts(
    rng: Any,
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    fallback_defaults: VennFieldDefaults,
    minimum_extra_icons: int,
) -> VennCountSample:
    """Sample target/object counts with task-specified non-target headroom."""

    answer_min, answer_max = int_bounds(
        params,
        defaults,
        low_key="target_count_min",
        high_key="target_count_max",
        fallback_low=fallback_defaults.target_count_min,
        fallback_high=fallback_defaults.target_count_max,
    )
    object_min, object_max = int_bounds(
        params,
        defaults,
        low_key="object_count_min",
        high_key="object_count_max",
        fallback_low=fallback_defaults.object_count_min,
        fallback_high=fallback_defaults.object_count_max,
    )
    opposite_min, opposite_max = int_bounds(
        params,
        defaults,
        low_key="target_opposite_count_min",
        high_key="target_opposite_count_max",
        fallback_low=fallback_defaults.target_opposite_count_min,
        fallback_high=fallback_defaults.target_opposite_count_max,
    )
    answer_support = tuple(range(int(answer_min), int(answer_max) + 1))
    target_count_probabilities = weighted_probability_map(
        answer_support,
        params.get(
            "target_count_weights",
            group_default(defaults, "target_count_weights", None),
        ),
    )
    explicit_target = params.get("target_count", params.get("target_answer"))
    if explicit_target is not None:
        target_count = int(explicit_target)
        if target_count not in set(answer_support):
            raise ValueError(f"target_count must be in {answer_support}")
        target_count_probability_map = dict(
            uniform_probability_map(answer_support, selected=int(target_count))
        )
    else:
        target_count = int(
            sample_weighted_value(rng, answer_support, target_count_probabilities)
        )
        target_count_probability_map = dict(target_count_probabilities)

    target_opposite_count = int(rng.randint(int(opposite_min), int(opposite_max)))
    min_object_count = max(
        int(object_min),
        int(target_count) + int(target_opposite_count) + int(minimum_extra_icons),
    )
    if min_object_count > int(object_max):
        raise ValueError("object_count range cannot support requested Venn icon counts")
    object_support = tuple(range(int(min_object_count), int(object_max) + 1))
    explicit_object = params.get("object_count")
    if explicit_object is not None:
        object_count = int(explicit_object)
        if object_count not in set(object_support):
            raise ValueError(f"object_count must be in {object_support}")
        object_count_probability_map = dict(
            uniform_probability_map(object_support, selected=int(object_count))
        )
    else:
        object_count = int(rng.choice(object_support))
        object_count_probability_map = dict(uniform_probability_map(object_support))
    return VennCountSample(
        target_count=int(target_count),
        target_count_probabilities=dict(target_count_probability_map),
        object_count=int(object_count),
        object_count_probabilities=dict(object_count_probability_map),
        target_opposite_count=int(target_opposite_count),
    )


def sample_nonmatching_icon(
    rng: Any,
    *,
    mode: str,
    target_shape_id: str,
    target_color: NamedColorEntry | None,
    shape_ids: Sequence[str],
    colors: Sequence[NamedColorEntry],
    fill_styles: Sequence[str],
    fill_style_weights: Mapping[str, float],
) -> tuple[str, NamedColorEntry, str]:
    """Sample one icon that cannot satisfy a target shape/color predicate."""

    other_shapes = [
        str(value) for value in shape_ids if str(value) != str(target_shape_id)
    ]
    if not other_shapes:
        raise ValueError("shape support resolved no distractor shapes")
    fill_style = str(
        sample_procedural_named_icon_fill_style(
            rng,
            support=fill_styles,
            probabilities=fill_style_weights,
        )
    )
    if str(mode) == "shape_only":
        return str(rng.choice(other_shapes)), rng.choice(tuple(colors)), fill_style

    if str(mode) == "color_shape":
        if target_color is None:
            raise ValueError("color_shape distractor is missing target_color")
        other_colors = [
            entry for entry in colors if str(entry.name) != str(target_color.name)
        ]
        draw = float(rng.random())
        if draw < 0.40 and other_colors:
            return str(target_shape_id), rng.choice(tuple(other_colors)), fill_style
        if draw < 0.78:
            return str(rng.choice(other_shapes)), target_color, fill_style
        return (
            str(rng.choice(other_shapes)),
            rng.choice(tuple(other_colors or list(colors))),
            fill_style,
        )

    return str(rng.choice(other_shapes)), rng.choice(tuple(colors)), fill_style


def sample_venn_count_inputs(
    rng: Any,
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    fallback_defaults: VennFieldDefaults,
    minimum_extra_icons: int,
    mode_support: Sequence[str] | None = None,
) -> VennCountInputs:
    """Resolve target, support, fill-style, and count inputs for a Venn count task."""

    shape_ids = shape_support(params, defaults)
    colors = color_support(params, defaults)
    fill_styles = fill_style_support(
        params,
        defaults,
        fallback=fallback_defaults.named_icon_fill_style_support,
    )
    fill_weights = fill_style_probabilities(params, defaults, fill_styles)
    target = sample_target_predicate(
        rng,
        params=params,
        defaults=defaults,
        shape_ids=shape_ids,
        colors=colors,
        mode_support=mode_support,
    )
    counts = sample_target_and_object_counts(
        rng,
        params=params,
        defaults=defaults,
        fallback_defaults=fallback_defaults,
        minimum_extra_icons=int(minimum_extra_icons),
    )
    return VennCountInputs(
        shape_ids=tuple(shape_ids),
        colors=tuple(colors),
        fill_styles=tuple(fill_styles),
        fill_style_probabilities=dict(fill_weights),
        target=target,
        counts=counts,
    )


__all__ = [
    "color_support",
    "default_target_mode_support",
    "fill_style_probabilities",
    "fill_style_support",
    "int_bounds",
    "sample_nonmatching_icon",
    "sample_target_and_object_counts",
    "sample_target_color",
    "sample_target_predicate",
    "sample_target_shape",
    "sample_venn_count_inputs",
    "shape_support",
    "target_description",
    "target_mode_probabilities",
    "uniform_string_probability_map",
]
