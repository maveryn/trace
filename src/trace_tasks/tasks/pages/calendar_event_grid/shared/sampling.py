"""Sampling and event-chip construction for calendar event-grid scenes."""

from __future__ import annotations

import calendar
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import hash64, spawn_rng
from trace_tasks.tasks.pages.shared.calendar_scene import CalendarEventChipSpec
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.time_artifact_task_support import resolve_time_artifact_named_variant
from trace_tasks.tasks.shared.time_format import month_name

from .annotations import chip_key
from .defaults import (
    CHIP_FILL_PALETTE,
    DEFAULTS,
    EVENT_CATEGORY_LABELS,
    EVENT_SLOT_SPECS,
    GENERATION_DEFAULTS,
    SCENE,
    SUPPORTED_EVENT_GRID_LAYOUT_MODES,
    SUPPORTED_EVENT_GRID_SURFACE_MODES,
    SUPPORTED_EVENT_GRID_TEXT_COLOR_MODES,
    SUPPORTED_EVENT_GRID_TITLE_MODES,
    SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS,
    SUPPORTED_TIME_ARTIFACT_COLOR_NAMES,
    SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS,
)
from .state import EventGridCase


NAMESPACE_ROOT = "pages.calendar_event_grid"


def slot_label(slot_id: str) -> str:
    """Return the visible label for one event slot id."""

    labels = {str(slot): str(label) for slot, label in EVENT_SLOT_SPECS}
    if str(slot_id) not in labels:
        raise ValueError(f"unsupported event slot id: {slot_id}")
    return labels[str(slot_id)]


def uniform_probability(values: Sequence[str | int]) -> Dict[str, float]:
    """Return a uniform probability map over the supplied values."""

    resolved = tuple(str(value) for value in values)
    if not resolved:
        return {}
    probability = 1.0 / float(len(resolved))
    return {str(value): float(probability) for value in resolved}


def resolve_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported: Tuple[str, ...],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named event-grid axis."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    return resolve_time_artifact_named_variant(
        rng,
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported=tuple(str(value) for value in supported),
        instance_seed=int(instance_seed),
        task_id=NAMESPACE_ROOT,
        namespace=str(namespace),
    )


def resolve_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Resolve one integer support list from scene config or explicit params."""

    raw = params.get(str(key), group_default(GENERATION_DEFAULTS, str(key), tuple(int(value) for value in fallback)))
    values = tuple(int(value) for value in raw)
    if not values:
        raise ValueError(f"{key} must contain at least one value")
    return tuple(values)


def resolve_category_labels(params: Mapping[str, Any]) -> Tuple[str, ...]:
    """Resolve the visible category label pool."""

    raw = params.get("event_category_labels", group_default(GENERATION_DEFAULTS, "event_category_labels", EVENT_CATEGORY_LABELS))
    labels: List[str] = []
    for value in raw:
        label = str(value).strip()
        if label and label not in labels:
            labels.append(label)
    if len(labels) < 4:
        raise ValueError("event_category_labels must contain at least four distinct visible labels")
    return tuple(labels)


def resolve_choice(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    explicit_key: str,
    values: Sequence[str],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one string choice from an explicit value or deterministic index."""

    explicit = params.get(str(explicit_key))
    value_set = {str(value) for value in values}
    if explicit is not None:
        if str(explicit) not in value_set:
            raise ValueError(f"unsupported {explicit_key}: {explicit}")
        return str(explicit), {str(explicit): 1.0}
    probabilities = uniform_probability(tuple(str(value) for value in values))
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.{namespace}",
    )
    return str(tuple(values)[int(index) % len(values)]), dict(probabilities)


def resolve_target_count(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the requested count support for count-style event-grid cases."""

    support = resolve_int_support(params, "target_count_support", DEFAULTS.target_count_support)
    explicit = params.get("target_count")
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"target_count must be in {list(support)}")
        return int(value), {str(value): 1.0}
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.target_count",
    )
    return int(support[int(index) % len(support)]), uniform_probability(support)


def resolve_date_filled_slot_count(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve how many event slots should be filled on the target date."""

    support = resolve_int_support(
        params,
        "filled_event_slot_count_support",
        DEFAULTS.date_filled_slot_count_support,
    )
    max_slots = len(EVENT_SLOT_SPECS)
    if any(int(value) < 0 or int(value) > int(max_slots) for value in support):
        raise ValueError(f"date_filled_slot_count_support values must be between 0 and {max_slots}")
    explicit = params.get("filled_event_slot_count")
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"filled_event_slot_count must be in {list(support)}")
        return int(value), {str(value): 1.0}
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.filled_event_slot_count",
    )
    return int(support[int(index) % len(support)]), uniform_probability(support)


def resolve_weekday_index(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the requested weekday index where Monday is 0."""

    support = resolve_int_support(params, "weekday_column_index_support", tuple(range(7)))
    if any(int(value) < 0 or int(value) > 6 for value in support):
        raise ValueError("weekday_column_index_support values must be between 0 and 6")
    explicit = params.get("weekday_index", params.get("target_weekday_index"))
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"weekday_index must be in {list(support)}")
        return int(value), {str(value): 1.0}
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.weekday_column_count.weekday_index",
    )
    return int(support[int(index) % len(support)]), uniform_probability(support)


def resolve_weekday_event_count(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    """Resolve the target number of chips in one weekday column."""

    support = resolve_int_support(params, "weekday_column_count_support", DEFAULTS.weekday_column_count_support)
    if any(int(value) < 0 for value in support):
        raise ValueError("weekday_column_count_support values must be non-negative")
    explicit = params.get("weekday_column_count")
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"weekday_column_count must be in {list(support)}")
        return int(value), {str(value): 1.0}
    sampling_index = params.get("_sample_cursor")
    if sampling_index is not None:
        return int(support[abs(int(sampling_index)) % len(support)]), uniform_probability(support)
    index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.weekday_column_count.target_count",
    )
    return int(support[int(index) % len(support)]), uniform_probability(support)


def resolve_year_month(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    """Resolve a calendar month and return year, month, days, and row count."""

    explicit_year = params.get("year")
    explicit_month = params.get("month")
    if (explicit_year is None) != (explicit_month is None):
        raise ValueError("year and month must be provided together for calendar event-grid tasks")
    if explicit_year is not None and explicit_month is not None:
        year = int(explicit_year)
        month = int(explicit_month)
    else:
        year_min = int(params.get("year_min", GENERATION_DEFAULTS.get("year_min", DEFAULTS.year_min)))
        year_max = int(params.get("year_max", GENERATION_DEFAULTS.get("year_max", DEFAULTS.year_max)))
        if year_min > year_max:
            raise ValueError("year_min must be <= year_max")
        year_span = int(year_max - year_min + 1)
        year_index = resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.year",
        )
        month_index = resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}.month",
        )
        year = int(year_min + (int(year_index) % int(year_span)))
        month = int(1 + (int(month_index) % 12))
    _start_weekday_index, days_in_month = calendar.monthrange(int(year), int(month))
    row_count = len(calendar.Calendar(firstweekday=0).monthdayscalendar(int(year), int(month)))
    return int(year), int(month), int(days_in_month), int(row_count)


def chip_fill(category_label: str, *, instance_seed: int) -> Tuple[int, int, int]:
    """Return a stable fill color for one category label."""

    index = abs(int(hash64(int(instance_seed), str(category_label), 50921)))
    return tuple(int(value) for value in CHIP_FILL_PALETTE[int(index) % len(CHIP_FILL_PALETTE)])


def make_chip(
    *,
    day: int,
    slot_id: str,
    category_label: str,
    instance_seed: int,
) -> CalendarEventChipSpec:
    """Create one visible event chip spec."""

    return CalendarEventChipSpec(
        day=int(day),
        slot_id=str(slot_id),
        slot_label=str(slot_label(str(slot_id))),
        category_label=str(category_label),
        fill_rgb=chip_fill(str(category_label), instance_seed=int(instance_seed)),
        text_rgb=(20, 34, 48),
    )


def build_event_chips(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    days_in_month: int,
    slot_id: str,
    category_label: str,
    category_labels: Sequence[str],
    target_count: int,
    target_date: int | None,
    unique_category_slot: bool,
) -> Tuple[Tuple[CalendarEventChipSpec, ...], int | None, Tuple[str, ...]]:
    """Build all event chips plus the task-relevant matching chip keys."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.event_chips")
    slots = tuple(str(slot) for slot, _ in EVENT_SLOT_SPECS)
    categories = tuple(str(value) for value in category_labels)
    chip_by_day_slot: Dict[Tuple[int, str], str] = {}
    matching_keys: List[str] = []
    selected_date = int(target_date) if target_date is not None else None

    if selected_date is not None:
        if int(selected_date) < 1 or int(selected_date) > int(days_in_month):
            raise ValueError("target_date must be within the sampled month")
        chip_by_day_slot[(int(selected_date), str(slot_id))] = str(category_label)
        matching_keys.append(chip_key(int(selected_date), str(slot_id)))
    else:
        day_pool = list(range(1, int(days_in_month) + 1))
        rng.shuffle(day_pool)
        selected_days = sorted(day_pool[: int(target_count)])
        for day in selected_days:
            chip_by_day_slot[(int(day), str(slot_id))] = str(category_label)
            matching_keys.append(chip_key(int(day), str(slot_id)))

    same_category_other_slot_count = min(4, max(2, int(days_in_month) // 9))
    same_slot_other_category_count = min(6, max(3, int(days_in_month) // 6))
    random_extra_min = int(GENERATION_DEFAULTS.get("min_random_extra_chips", DEFAULTS.min_random_extra_chips))
    random_extra_max = int(GENERATION_DEFAULTS.get("max_random_extra_chips", DEFAULTS.max_random_extra_chips))
    if random_extra_min > random_extra_max:
        raise ValueError("min_random_extra_chips must be <= max_random_extra_chips")
    random_extra_count = int(rng.randint(int(random_extra_min), int(random_extra_max)))

    other_slots = [slot for slot in slots if str(slot) != str(slot_id)]
    other_categories = [label for label in categories if str(label) != str(category_label)]
    for _ in range(same_category_other_slot_count):
        day = int(rng.randint(1, int(days_in_month)))
        slot = str(rng.choice(other_slots))
        chip_by_day_slot.setdefault((day, slot), str(category_label))
    for _ in range(same_slot_other_category_count):
        day = int(rng.randint(1, int(days_in_month)))
        category = str(rng.choice(other_categories))
        chip_by_day_slot.setdefault((day, str(slot_id)), str(category))
    for _ in range(random_extra_count):
        day = int(rng.randint(1, int(days_in_month)))
        slot = str(rng.choice(slots))
        category = str(rng.choice(categories))
        if bool(unique_category_slot) and str(slot) == str(slot_id) and str(category) == str(category_label):
            category = str(rng.choice(other_categories))
        chip_by_day_slot.setdefault((day, slot), category)

    chips = tuple(
        make_chip(
            day=int(day),
            slot_id=str(slot),
            category_label=str(category),
            instance_seed=int(instance_seed),
        )
        for (day, slot), category in sorted(chip_by_day_slot.items(), key=lambda item: (int(item[0][0]), str(item[0][1])))
    )
    return chips, selected_date, tuple(str(key) for key in matching_keys)


def build_date_filled_slot_count_chips(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    days_in_month: int,
    target_date: int,
    filled_slot_count: int,
    category_labels: Sequence[str],
) -> Tuple[Tuple[CalendarEventChipSpec, ...], Tuple[str, ...]]:
    """Build chips where the target date has exactly the requested number of filled slots."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.filled_event_slot_count_chips")
    slots = tuple(str(slot) for slot, _label in EVENT_SLOT_SPECS)
    categories = tuple(str(value) for value in category_labels)
    if not categories:
        raise ValueError("category_labels must not be empty")
    if int(filled_slot_count) < 0 or int(filled_slot_count) > len(slots):
        raise ValueError("filled_slot_count must fit within the visible event slots")

    slot_order = {str(slot): index for index, slot in enumerate(slots)}
    shuffled_slots = list(slots)
    rng.shuffle(shuffled_slots)
    selected_slots = tuple(
        sorted(
            shuffled_slots[: int(filled_slot_count)],
            key=lambda value: int(slot_order[str(value)]),
        )
    )

    chip_by_day_slot: Dict[Tuple[int, str], str] = {}
    for slot in selected_slots:
        chip_by_day_slot[(int(target_date), str(slot))] = str(rng.choice(categories))

    random_extra_min = int(GENERATION_DEFAULTS.get("min_random_extra_chips", DEFAULTS.min_random_extra_chips))
    random_extra_max = int(GENERATION_DEFAULTS.get("max_random_extra_chips", DEFAULTS.max_random_extra_chips))
    if random_extra_min > random_extra_max:
        raise ValueError("min_random_extra_chips must be <= max_random_extra_chips")
    random_extra_count = int(rng.randint(int(random_extra_min), int(random_extra_max)))
    available_pairs = [
        (int(day), str(slot))
        for day in range(1, int(days_in_month) + 1)
        if int(day) != int(target_date)
        for slot in slots
    ]
    rng.shuffle(available_pairs)
    for day, slot in available_pairs[: min(int(random_extra_count), len(available_pairs))]:
        chip_by_day_slot[(int(day), str(slot))] = str(rng.choice(categories))

    chips = tuple(
        make_chip(
            day=int(day),
            slot_id=str(slot),
            category_label=str(category),
            instance_seed=int(instance_seed),
        )
        for (day, slot), category in sorted(
            chip_by_day_slot.items(),
            key=lambda item: (int(item[0][0]), int(slot_order[str(item[0][1])])),
        )
    )
    matching_keys = tuple(chip_key(int(target_date), str(slot)) for slot in selected_slots)
    return chips, matching_keys


def build_weekday_event_count_chips(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    year: int,
    month: int,
    days_in_month: int,
    weekday_index: int,
    target_count: int,
    category_labels: Sequence[str],
) -> Tuple[Tuple[CalendarEventChipSpec, ...], Tuple[str, ...]]:
    """Build chips with exactly `target_count` chips in one weekday column."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.weekday_column_count_chips")
    slots = tuple(str(slot) for slot, _label in EVENT_SLOT_SPECS)
    categories = tuple(str(value) for value in category_labels)
    if not categories:
        raise ValueError("category_labels must not be empty")
    first_weekday_index, _days = calendar.monthrange(int(year), int(month))
    target_dates = tuple(
        int(day)
        for day in range(1, int(days_in_month) + 1)
        if int((int(first_weekday_index) + int(day) - 1) % 7) == int(weekday_index)
    )
    target_pairs = [(int(day), str(slot)) for day in target_dates for slot in slots]
    if int(target_count) > len(target_pairs):
        raise ValueError("weekday column count exceeds available weekday event slots")

    rng.shuffle(target_pairs)
    selected_pairs = set(target_pairs[: int(target_count)])
    chip_by_day_slot: Dict[Tuple[int, str], str] = {
        (int(day), str(slot)): str(rng.choice(categories))
        for day, slot in selected_pairs
    }

    distractor_support = resolve_int_support(
        params,
        "weekday_column_distractor_support",
        DEFAULTS.weekday_column_distractor_support,
    )
    distractor_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.weekday_column_count.distractor_count",
    )
    distractor_count = int(distractor_support[int(distractor_index) % len(distractor_support)])
    outside_pairs = [
        (int(day), str(slot))
        for day in range(1, int(days_in_month) + 1)
        if int((int(first_weekday_index) + int(day) - 1) % 7) != int(weekday_index)
        for slot in slots
    ]
    rng.shuffle(outside_pairs)
    for day, slot in outside_pairs[: min(int(distractor_count), len(outside_pairs))]:
        chip_by_day_slot[(int(day), str(slot))] = str(rng.choice(categories))

    slot_order = {str(slot): index for index, slot in enumerate(slots)}
    chips = tuple(
        make_chip(
            day=int(day),
            slot_id=str(slot),
            category_label=str(category),
            instance_seed=int(instance_seed),
        )
        for (day, slot), category in sorted(
            chip_by_day_slot.items(),
            key=lambda item: (int(item[0][0]), int(slot_order[str(item[0][1])])),
        )
    )
    matching_keys = tuple(
        chip_key(int(day), str(slot))
        for day, slot in sorted(selected_pairs, key=lambda item: (int(item[0]), int(slot_order[str(item[1])])))
    )
    return chips, matching_keys


def build_busiest_date_label_chips(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    days_in_month: int,
    target_date: int,
    category_labels: Sequence[str],
) -> Tuple[Tuple[CalendarEventChipSpec, ...], Tuple[str, ...], Dict[str, int]]:
    """Build chips where one date uniquely has all visible event slots filled."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.busiest_date_label_chips")
    slots = tuple(str(slot) for slot, _label in EVENT_SLOT_SPECS)
    categories = tuple(str(value) for value in category_labels)
    if not categories:
        raise ValueError("category_labels must not be empty")
    if int(target_date) < 1 or int(target_date) > int(days_in_month):
        raise ValueError("target_date must be within the sampled month")

    slot_order = {str(slot): index for index, slot in enumerate(slots)}
    chip_by_day_slot: Dict[Tuple[int, str], str] = {}
    for slot in slots:
        chip_by_day_slot[(int(target_date), str(slot))] = str(rng.choice(categories))

    two_chip_support = resolve_int_support(
        params,
        "busiest_two_chip_distractor_date_count_support",
        DEFAULTS.busiest_two_chip_distractor_date_count_support,
    )
    single_chip_support = resolve_int_support(
        params,
        "busiest_single_chip_distractor_date_count_support",
        DEFAULTS.busiest_single_chip_distractor_date_count_support,
    )
    two_chip_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.busiest_date_label.two_chip_distractor_dates",
    )
    single_chip_index = resolve_selection_index(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{NAMESPACE_ROOT}.busiest_date_label.single_chip_distractor_dates",
    )
    two_chip_count = int(two_chip_support[int(two_chip_index) % len(two_chip_support)])
    single_chip_count = int(single_chip_support[int(single_chip_index) % len(single_chip_support)])

    day_pool = [int(day) for day in range(1, int(days_in_month) + 1) if int(day) != int(target_date)]
    rng.shuffle(day_pool)
    two_chip_days = tuple(day_pool[: min(int(two_chip_count), len(day_pool))])
    remaining_days = day_pool[len(two_chip_days) :]
    single_chip_days = tuple(remaining_days[: min(int(single_chip_count), len(remaining_days))])

    for day in two_chip_days:
        selected_slots = list(slots)
        rng.shuffle(selected_slots)
        for slot in selected_slots[:2]:
            chip_by_day_slot[(int(day), str(slot))] = str(rng.choice(categories))
    for day in single_chip_days:
        chip_by_day_slot[(int(day), str(rng.choice(slots)))] = str(rng.choice(categories))

    chips = tuple(
        make_chip(
            day=int(day),
            slot_id=str(slot),
            category_label=str(category),
            instance_seed=int(instance_seed),
        )
        for (day, slot), category in sorted(
            chip_by_day_slot.items(),
            key=lambda item: (int(item[0][0]), int(slot_order[str(item[0][1])])),
        )
    )
    matching_keys = tuple(chip_key(int(target_date), str(slot)) for slot in slots)
    date_chip_counts: Dict[str, int] = {}
    for day, _slot in chip_by_day_slot:
        date_key = str(int(day))
        date_chip_counts[date_key] = int(date_chip_counts.get(date_key, 0)) + 1
    return chips, matching_keys, dict(sorted(date_chip_counts.items(), key=lambda item: int(item[0])))


def build_common_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    target_date: int | None,
    unique_category_slot: bool,
) -> EventGridCase:
    """Build one event-grid case from shared axes and task-provided operands."""

    scene_variant, scene_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS,
        namespace="scene_variant",
    )
    style_variant, style_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="style_variant",
        weights_key="style_variant_weights",
        balance_flag_key="balanced_style_variant_sampling",
        supported=SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS,
        namespace="style_variant",
    )
    accent_color_name, accent_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="accent_color_name",
        weights_key="accent_color_name_weights",
        balance_flag_key="balanced_accent_color_name_sampling",
        supported=SUPPORTED_TIME_ARTIFACT_COLOR_NAMES,
        namespace="accent_color_name",
    )
    layout_mode, layout_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="layout_mode",
        weights_key="layout_mode_weights",
        balance_flag_key="balanced_layout_mode_sampling",
        supported=SUPPORTED_EVENT_GRID_LAYOUT_MODES,
        namespace="layout_mode",
    )
    title_mode, title_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="title_mode",
        weights_key="title_mode_weights",
        balance_flag_key="balanced_title_mode_sampling",
        supported=SUPPORTED_EVENT_GRID_TITLE_MODES,
        namespace="title_mode",
    )
    surface_mode, surface_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="surface_mode",
        weights_key="surface_mode_weights",
        balance_flag_key="balanced_surface_mode_sampling",
        supported=SUPPORTED_EVENT_GRID_SURFACE_MODES,
        namespace="surface_mode",
    )
    text_color_mode, text_color_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="text_color_mode",
        weights_key="text_color_mode_weights",
        balance_flag_key="balanced_text_color_mode_sampling",
        supported=SUPPORTED_EVENT_GRID_TEXT_COLOR_MODES,
        namespace="text_color_mode",
    )
    category_labels = resolve_category_labels(params)
    supported_slots = tuple(str(slot) for slot, _ in EVENT_SLOT_SPECS)
    if not supported_slots:
        raise ValueError("slot_support must contain at least one slot id")
    slot_id, slot_probs = resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="slot_id",
        weights_key="slot_id_weights",
        balance_flag_key="balanced_slot_id_sampling",
        supported=tuple(str(slot) for slot in supported_slots),
        namespace="slot_id",
    )
    category_label, category_probs = resolve_choice(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="category_label",
        values=category_labels,
        namespace="category_label",
    )
    target_count, target_count_probs = resolve_target_count(int(instance_seed), params)
    year, month, days_in_month, row_count = resolve_year_month(int(instance_seed), params)
    event_chips, selected_date, matching_keys = build_event_chips(
        instance_seed=int(instance_seed),
        params=params,
        days_in_month=int(days_in_month),
        slot_id=str(slot_id),
        category_label=str(category_label),
        category_labels=tuple(str(value) for value in category_labels),
        target_count=int(target_count),
        target_date=target_date,
        unique_category_slot=bool(unique_category_slot),
    )
    return EventGridCase(
        scene_variant=str(scene_variant),
        style_variant=str(style_variant),
        accent_color_name=str(accent_color_name),
        layout_mode=str(layout_mode),
        title_mode=str(title_mode),
        surface_mode=str(surface_mode),
        text_color_mode=str(text_color_mode),
        year=int(year),
        month=int(month),
        month_name=str(month_name(int(month))),
        days_in_month=int(days_in_month),
        row_count=int(row_count),
        slot_id=str(slot_id),
        slot_label=str(slot_label(str(slot_id))),
        category_label=str(category_label),
        event_category_labels=tuple(str(value) for value in category_labels),
        target_date=(int(selected_date) if selected_date is not None else None),
        target_count=(int(target_count) if target_date is None else None),
        event_chips=tuple(event_chips),
        matching_chip_keys=tuple(str(key) for key in matching_keys),
        weekday_index=None,
        weekday_label=None,
        category_probabilities=dict(category_probs),
        slot_probabilities=dict(slot_probs),
        weekday_probabilities={},
        target_count_probabilities=dict(target_count_probs),
        scene_variant_probabilities=dict(scene_probs),
        style_variant_probabilities=dict(style_probs),
        accent_color_name_probabilities=dict(accent_probs),
        layout_mode_probabilities=dict(layout_probs),
        title_mode_probabilities=dict(title_probs),
        surface_mode_probabilities=dict(surface_probs),
        text_color_mode_probabilities=dict(text_color_probs),
    )


def resolve_target_date(instance_seed: int, params: Mapping[str, Any], days_in_month: int) -> int:
    """Resolve one target date within the sampled month."""

    explicit = params.get("target_date")
    if explicit is not None:
        value = int(explicit)
    else:
        value = 1 + (
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{NAMESPACE_ROOT}.target_date",
            )
            % int(days_in_month)
        )
    if int(value) < 1 or int(value) > int(days_in_month):
        raise ValueError("target_date must be within the sampled month")
    return int(value)


def build_category_slot_day_count_case(instance_seed: int, *, params: Mapping[str, Any]) -> EventGridCase:
    """Build a case for counting dates with a requested category in one slot."""

    return build_common_case(
        instance_seed=int(instance_seed),
        params=params,
        target_date=None,
        unique_category_slot=True,
    )


def build_weekday_event_count_case(instance_seed: int, *, params: Mapping[str, Any]) -> EventGridCase:
    """Build a case for counting all event chips in one weekday column."""

    base_params = dict(params)
    base_params.pop("target_count", None)
    base_case = build_common_case(
        instance_seed=int(instance_seed),
        params=base_params,
        target_date=None,
        unique_category_slot=False,
    )
    weekday_index, weekday_probs = resolve_weekday_index(int(instance_seed), params)
    target_count, count_probs = resolve_weekday_event_count(int(instance_seed), params)
    chips, matching_keys = build_weekday_event_count_chips(
        instance_seed=int(instance_seed),
        params=params,
        year=int(base_case.year),
        month=int(base_case.month),
        days_in_month=int(base_case.days_in_month),
        weekday_index=int(weekday_index),
        target_count=int(target_count),
        category_labels=tuple(base_case.event_category_labels),
    )
    return replace(
        base_case,
        target_count=int(target_count),
        event_chips=tuple(chips),
        matching_chip_keys=tuple(str(key) for key in matching_keys),
        weekday_index=int(weekday_index),
        weekday_label=str(calendar.day_name[int(weekday_index)]),
        weekday_probabilities=dict(weekday_probs),
        target_count_probabilities=dict(count_probs),
    )


def build_date_for_category_slot_case(instance_seed: int, *, params: Mapping[str, Any]) -> EventGridCase:
    """Build a case for finding the unique date containing a category in one slot."""

    year, month, days_in_month, _row_count = resolve_year_month(int(instance_seed), params)
    del year, month
    target_date = resolve_target_date(int(instance_seed), params, int(days_in_month))
    return build_common_case(
        instance_seed=int(instance_seed),
        params=params,
        target_date=int(target_date),
        unique_category_slot=True,
    )


def build_date_filled_slot_count_case(instance_seed: int, *, params: Mapping[str, Any]) -> EventGridCase:
    """Build a case for counting filled event slots on one target date."""

    year, month, days_in_month, _row_count = resolve_year_month(int(instance_seed), params)
    del year, month
    target_date = resolve_target_date(int(instance_seed), params, int(days_in_month))
    filled_slot_count, count_probs = resolve_date_filled_slot_count(int(instance_seed), params)
    base_case = build_common_case(
        instance_seed=int(instance_seed),
        params=params,
        target_date=int(target_date),
        unique_category_slot=False,
    )
    chips, matching_keys = build_date_filled_slot_count_chips(
        instance_seed=int(instance_seed),
        params=params,
        days_in_month=int(base_case.days_in_month),
        target_date=int(target_date),
        filled_slot_count=int(filled_slot_count),
        category_labels=tuple(base_case.event_category_labels),
    )
    return replace(
        base_case,
        event_chips=tuple(chips),
        matching_chip_keys=tuple(str(key) for key in matching_keys),
        target_count_probabilities=dict(count_probs),
    )


def build_busiest_date_label_case(instance_seed: int, *, params: Mapping[str, Any]) -> EventGridCase:
    """Build a case whose answer is the unique date with the most event chips."""

    year, month, days_in_month, _row_count = resolve_year_month(int(instance_seed), params)
    del year, month
    target_date = resolve_target_date(int(instance_seed), params, int(days_in_month))
    base_case = build_common_case(
        instance_seed=int(instance_seed),
        params=params,
        target_date=int(target_date),
        unique_category_slot=False,
    )
    chips, matching_keys, _date_chip_counts = build_busiest_date_label_chips(
        instance_seed=int(instance_seed),
        params=params,
        days_in_month=int(base_case.days_in_month),
        target_date=int(target_date),
        category_labels=tuple(base_case.event_category_labels),
    )
    return replace(
        base_case,
        event_chips=tuple(chips),
        matching_chip_keys=tuple(str(key) for key in matching_keys),
        target_count=3,
        target_count_probabilities={"3": 1.0},
    )
