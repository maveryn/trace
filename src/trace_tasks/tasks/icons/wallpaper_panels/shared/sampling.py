"""Identity-free sampling helpers for wallpaper-panel icon scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map
from ....shared.fixed_query import resolve_task_query_id_param, strip_query_id_params

from .defaults import OPTION_LABELS, WallpaperPanelDefaults
from .rendering import resolve_wallpaper_group_support, uniform_str_probability_map


def int_tuple_param(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve a non-empty integer support from params or generation defaults."""

    raw = params.get(str(key), group_default(generation_defaults, str(key), list(fallback)))
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{key} must be a sequence")
    values = tuple(int(value) for value in raw)
    if not values:
        raise ValueError(f"{key} must contain at least one value")
    return values


def option_count_support(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    fallback_defaults: WallpaperPanelDefaults,
    context: str,
) -> Tuple[int, ...]:
    """Resolve visible option-count support; wallpaper panels currently use four labels."""

    support = tuple(
        int(value)
        for value in int_tuple_param(
            params,
            generation_defaults,
            key="option_count_choices",
            fallback=fallback_defaults.option_count_choices,
        )
    )
    if any(value != len(OPTION_LABELS) for value in support):
        raise ValueError(f"{context} currently supports four option panels")
    return support


def wallpaper_group_support(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    fallback_defaults: WallpaperPanelDefaults,
) -> Tuple[str, ...]:
    """Resolve supported wallpaper-group ids from params or generation defaults."""

    raw = params.get("wallpaper_group_ids")
    if raw is None:
        raw = group_default(generation_defaults, "wallpaper_group_ids", None)
    if raw is None:
        raw = params.get(
            "motif_ids",
            group_default(generation_defaults, "motif_ids", list(fallback_defaults.wallpaper_group_ids)),
        )
    return resolve_wallpaper_group_support(raw, fallback=fallback_defaults.wallpaper_group_ids)


def active_option_labels(option_count: int) -> Tuple[str, ...]:
    """Return visible option labels for a supported option count."""

    count = int(option_count)
    if count != len(OPTION_LABELS):
        raise ValueError("wallpaper panels currently support exactly four option labels")
    return tuple(str(label) for label in OPTION_LABELS[:count])


def normalize_single_branch(
    params: Mapping[str, Any],
    *,
    accepted: Sequence[str],
    selected: str,
    owner: str,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Validate that callers did not request an unsupported single branch."""

    resolve_task_query_id_param(
        params=params,
        supported_query_ids=tuple(str(value) for value in accepted),
        default_query_id=str(selected),
        task_id=str(owner),
    )
    return {str(selected): 1.0}, strip_query_id_params(params)


def choose_option_count(
    rng: Any,
    params: Mapping[str, Any],
    support: Sequence[int],
) -> tuple[int, Tuple[str, ...], Dict[int, float]]:
    """Resolve the active option count and corresponding labels."""

    choices = tuple(int(value) for value in support)
    explicit = params.get("option_count")
    if explicit is None:
        count = int(rng.choice(choices))
        probabilities = dict(uniform_probability_map(choices))
    else:
        count = int(explicit)
        if count not in set(choices):
            raise ValueError("option_count is outside configured support")
        probabilities = dict(uniform_probability_map(choices, selected=count))
    return int(count), active_option_labels(int(count)), probabilities


def choose_panel_label(
    rng: Any,
    params: Mapping[str, Any],
    labels: Sequence[str],
    *,
    explicit_keys: Sequence[str] = ("answer_label", "correct_option_label"),
) -> tuple[str, int, Dict[str, float]]:
    """Resolve one visible option-panel label."""

    label_support = tuple(str(label) for label in labels)
    explicit_label = ""
    for key in explicit_keys:
        explicit_label = str(params.get(str(key), "")).strip().upper()
        if explicit_label:
            break
    if explicit_label:
        if explicit_label not in set(label_support):
            raise ValueError("selected panel label must be one of the active option panel labels")
        label = str(explicit_label)
        probabilities = uniform_str_probability_map(label_support, selected=label)
    else:
        label = str(rng.choice(label_support))
        probabilities = uniform_str_probability_map(label_support)
    return str(label), int(label_support.index(str(label))), dict(probabilities)


def choose_wallpaper_group(
    rng: Any,
    params: Mapping[str, Any],
    support: Sequence[str],
    *,
    explicit_keys: Sequence[str],
    context: str,
) -> tuple[str, Dict[str, float]]:
    """Resolve one wallpaper group from a named support."""

    group_support = tuple(str(group_id) for group_id in support)
    explicit = None
    for key in explicit_keys:
        if key in params and params.get(str(key)) is not None:
            explicit = str(params.get(str(key))).strip()
            break
    if explicit is None:
        selected = str(rng.choice(group_support))
        probabilities = uniform_str_probability_map(group_support)
    else:
        selected = str(explicit)
        if selected not in set(group_support):
            raise ValueError(f"{context} is outside configured support")
        probabilities = uniform_str_probability_map(group_support, selected=selected)
    return str(selected), dict(probabilities)


def shuffled_remaining_groups(
    *,
    instance_seed: int,
    owner: str,
    support: Sequence[str],
    excluded: str,
) -> list[str]:
    """Return deterministic shuffled groups excluding one selected group."""

    groups = [str(group_id) for group_id in support if str(group_id) != str(excluded)]
    rng = spawn_rng(int(instance_seed), f"{owner}.distractor_groups")
    rng.shuffle(groups)
    return list(groups)


__all__ = [
    "active_option_labels",
    "choose_option_count",
    "choose_panel_label",
    "choose_wallpaper_group",
    "int_tuple_param",
    "normalize_single_branch",
    "option_count_support",
    "shuffled_remaining_groups",
    "wallpaper_group_support",
]
