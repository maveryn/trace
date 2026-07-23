"""Calendar scene sampling and date arithmetic."""

from __future__ import annotations

import calendar
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.deterministic_sampling import resolve_selection_index
from trace_tasks.tasks.shared.time_artifact_task_support import resolve_time_artifact_named_variant
from trace_tasks.tasks.shared.time_format import month_name

from .defaults import (
    DEFAULTS,
    GENERATION_DEFAULTS,
    SUPPORTED_MARKED_DAY_CLASSES,
    SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES,
    SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS,
    SUPPORTED_PAGE_CALENDAR_TITLE_MODES,
    SUPPORTED_PAGE_CALENDAR_WEEK_STARTS,
)
from .state import CalendarCase


NAMESPACE_ROOT = "pages.calendar"
WEEK_START_TO_FIRST_WEEKDAY_INDEX = {
    "monday": 0,
    "sunday": 6,
}


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
    """Resolve one balanced named calendar axis."""

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.{namespace}")
    return resolve_time_artifact_named_variant(
        rng,
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported=supported,
        instance_seed=int(instance_seed),
        task_id=NAMESPACE_ROOT,
        namespace=str(namespace),
    )


def resolve_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
    """Resolve one integer support list from scene config or explicit params."""

    raw_values = params.get(key, group_default(GENERATION_DEFAULTS, key, fallback))
    resolved: List[int] = []
    for raw_value in raw_values:
        value = int(raw_value)
        if value not in resolved:
            resolved.append(value)
    if not resolved:
        raise ValueError(f"{key} must not be empty for calendar")
    return tuple(int(value) for value in resolved)


def normalize_marked_day_class(value: Any) -> str:
    """Normalize one marked-day class label."""

    normalized = str(value).strip().lower()
    if normalized in {"weekend", "weekends"}:
        return "weekend"
    if normalized in {"weekday", "weekdays"}:
        return "weekday"
    raise ValueError(f"unsupported marked_day_class: {value}")


def resolve_marked_day_class(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Resolve the weekday/weekend class for marked-date count tasks."""

    explicit = params.get("marked_day_class", params.get("day_class"))
    if explicit is not None:
        selected = normalize_marked_day_class(explicit)
        return selected, {
            key: (1.0 if key == selected else 0.0)
            for key in SUPPORTED_MARKED_DAY_CLASSES
        }

    return resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="marked_day_class",
        weights_key="marked_day_class_weights",
        balance_flag_key="balanced_marked_day_class_sampling",
        supported=SUPPORTED_MARKED_DAY_CLASSES,
        namespace="marked_day_class",
    )


def marked_day_class_phrase(marked_day_class: str) -> str:
    """Return a prompt phrase for one marked-day class."""

    return "Saturday or Sunday" if str(marked_day_class) == "weekend" else "Monday through Friday"


def normalize_week_start(value: Any) -> str:
    """Normalize one supported displayed week-start label."""

    normalized = str(value).strip().lower()
    if normalized in {"monday", "mon", "0"}:
        return "monday"
    if normalized in {"sunday", "sun", "6"}:
        return "sunday"
    raise ValueError(f"unsupported calendar week_start: {value}")


def first_weekday_index_for_week_start(week_start: str) -> int:
    """Return the Python-calendar first weekday index for a week-start label."""

    normalized = normalize_week_start(week_start)
    return int(WEEK_START_TO_FIRST_WEEKDAY_INDEX[str(normalized)])


def resolve_week_start(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Resolve the displayed calendar week-start axis."""

    explicit = params.get("week_start", params.get("calendar_week_start"))
    if explicit is None and params.get("first_weekday_index") is not None:
        explicit = params.get("first_weekday_index")
    if explicit is not None:
        selected = normalize_week_start(explicit)
        return selected, {
            key: (1.0 if key == selected else 0.0)
            for key in SUPPORTED_PAGE_CALENDAR_WEEK_STARTS
        }

    return resolve_named_axis(
        instance_seed=int(instance_seed),
        params=params,
        explicit_key="week_start",
        weights_key="week_start_weights",
        balance_flag_key="balanced_week_start_sampling",
        supported=SUPPORTED_PAGE_CALENDAR_WEEK_STARTS,
        namespace="week_start",
    )


def sample_month(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    """Sample one Gregorian month and return `(year, month, start_weekday, days_in_month)`."""

    explicit_year = params.get("year")
    explicit_month = params.get("month")
    if explicit_year is not None and explicit_month is None:
        raise ValueError("month must be provided when year is explicit for page calendars")
    if explicit_month is not None and explicit_year is None:
        raise ValueError("year must be provided when month is explicit for page calendars")

    if explicit_year is not None and explicit_month is not None:
        year = int(explicit_year)
        month = int(explicit_month)
        start_weekday_index, days_in_month = calendar.monthrange(int(year), int(month))
        return int(year), int(month), int(start_weekday_index), int(days_in_month)

    year_min = int(params.get("year_min", group_default(GENERATION_DEFAULTS, "year_min", DEFAULTS.year_min)))
    year_max = int(params.get("year_max", group_default(GENERATION_DEFAULTS, "year_max", DEFAULTS.year_max)))
    if int(year_max) < int(year_min):
        raise ValueError("year_max must be >= year_min for page calendars")

    year_support = tuple(range(int(year_min), int(year_max) + 1))
    year = int(
        year_support[
            int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{NAMESPACE_ROOT}:year",
                )
                % len(year_support)
            )
        ]
    )
    month = 1 + int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}:month",
        )
        % 12
    )
    start_weekday_index, days_in_month = calendar.monthrange(int(year), int(month))
    return int(year), int(month), int(start_weekday_index), int(days_in_month)


def resolve_visual_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Dict[str, Tuple[str, Dict[str, float]]]:
    """Resolve the shared scene style/layout axes."""

    return {
        "scene_variant": resolve_named_axis(
            instance_seed=int(instance_seed),
            params=params,
            explicit_key="scene_variant",
            weights_key="scene_variant_weights",
            balance_flag_key="balanced_scene_variant_sampling",
            supported=SUPPORTED_PAGE_CALENDAR_SCENE_VARIANTS,
            namespace="scene_variant",
        ),
        "layout_mode": resolve_named_axis(
            instance_seed=int(instance_seed),
            params=params,
            explicit_key="layout_mode",
            weights_key="layout_mode_weights",
            balance_flag_key="balanced_layout_mode_sampling",
            supported=SUPPORTED_PAGE_CALENDAR_LAYOUT_MODES,
            namespace="layout_mode",
        ),
        "title_mode": resolve_named_axis(
            instance_seed=int(instance_seed),
            params=params,
            explicit_key="title_mode",
            weights_key="title_mode_weights",
            balance_flag_key="balanced_title_mode_sampling",
            supported=SUPPORTED_PAGE_CALENDAR_TITLE_MODES,
            namespace="title_mode",
        ),
        "week_start": resolve_week_start(instance_seed=int(instance_seed), params=params),
    }


def resolve_common_case_fields(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve shared visual/month fields used by every calendar task."""

    axes = resolve_visual_axes(instance_seed=int(instance_seed), params=params)
    year, month, start_weekday_index, days_in_month = sample_month(int(instance_seed), params)
    week_start = str(axes["week_start"][0])
    first_weekday_index = first_weekday_index_for_week_start(str(week_start))
    month_weeks = calendar.Calendar(firstweekday=int(first_weekday_index)).monthdayscalendar(int(year), int(month))
    row_count = len(month_weeks)
    return {
        "scene_variant": str(axes["scene_variant"][0]),
        "layout_mode": str(axes["layout_mode"][0]),
        "title_mode": str(axes["title_mode"][0]),
        "week_start": str(week_start),
        "first_weekday_index": int(first_weekday_index),
        "year": int(year),
        "month": int(month),
        "month_name": str(month_name(int(month))),
        "row_count": int(row_count),
        "start_weekday_index": int(start_weekday_index),
        "days_in_month": int(days_in_month),
        "weekend_weekday_indices": resolve_int_support(
            params,
            "weekend_weekday_indices",
            DEFAULTS.weekend_weekday_indices,
        ),
        "date_occurrence_support": resolve_int_support(
            params,
            "date_occurrence_support",
            DEFAULTS.date_occurrence_support,
        ),
        "marked_weekend_count_support": resolve_int_support(
            params,
            "marked_weekend_count_support",
            DEFAULTS.marked_weekend_count_support,
        ),
        "marked_weekday_count_support": resolve_int_support(
            params,
            "marked_weekday_count_support",
            DEFAULTS.marked_weekday_count_support,
        ),
        "marked_weekday_distractor_support": resolve_int_support(
            params,
            "marked_weekday_distractor_support",
            DEFAULTS.marked_weekday_distractor_support,
        ),
        "marked_weekend_distractor_support": resolve_int_support(
            params,
            "marked_weekend_distractor_support",
            DEFAULTS.marked_weekend_distractor_support,
        ),
        "workday_offset_support": resolve_int_support(
            params,
            "workday_offset_support",
            DEFAULTS.workday_offset_support,
        ),
        "date_range_day_count_support": resolve_int_support(
            params,
            "date_range_day_count_support",
            DEFAULTS.date_range_day_count_support,
        ),
        "scene_variant_probabilities": dict(axes["scene_variant"][1]),
        "layout_mode_probabilities": dict(axes["layout_mode"][1]),
        "title_mode_probabilities": dict(axes["title_mode"][1]),
        "week_start_probabilities": dict(axes["week_start"][1]),
    }


def resolve_marked_count_query(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    days_in_month: int,
    start_weekday_index: int,
    weekend_weekday_indices: Tuple[int, ...],
    target_weekend: bool,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
    """Resolve marked-date placement for one weekday/weekend count case."""

    weekend_dates = sorted(
        int(day)
        for day in range(1, int(days_in_month) + 1)
        if int((int(start_weekday_index) + int(day) - 1) % 7)
        in set(int(value) for value in weekend_weekday_indices)
    )
    weekday_dates = [int(day) for day in range(1, int(days_in_month) + 1) if int(day) not in set(weekend_dates)]
    target_dates = weekend_dates if bool(target_weekend) else weekday_dates
    distractor_dates = weekday_dates if bool(target_weekend) else weekend_dates
    target_key = "marked_weekend_count_support" if bool(target_weekend) else "marked_weekday_count_support"
    target_fallback = (
        DEFAULTS.marked_weekend_count_support if bool(target_weekend) else DEFAULTS.marked_weekday_count_support
    )
    distractor_key = (
        "marked_weekday_distractor_support" if bool(target_weekend) else "marked_weekend_distractor_support"
    )
    distractor_fallback = (
        DEFAULTS.marked_weekday_distractor_support
        if bool(target_weekend)
        else DEFAULTS.marked_weekend_distractor_support
    )

    target_support = [
        int(value)
        for value in resolve_int_support(params, target_key, target_fallback)
        if 0 <= int(value) <= len(target_dates)
    ]
    if not target_support:
        raise ValueError("no feasible marked date-count support exists for this month")
    target_count = int(
        target_support[
            int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{NAMESPACE_ROOT}:{target_key}",
                )
                % len(target_support)
            )
        ]
    )

    distractor_support = [
        int(value)
        for value in resolve_int_support(params, distractor_key, distractor_fallback)
        if 0 <= int(value) <= len(distractor_dates)
    ]
    if not distractor_support:
        raise ValueError("no feasible marked date distractor support exists for this month")
    distractor_count = int(
        distractor_support[
            int(
                resolve_selection_index(
                    params=params,
                    instance_seed=int(instance_seed),
                    namespace=f"{NAMESPACE_ROOT}:{distractor_key}",
                )
                % len(distractor_support)
            )
        ]
    )

    rng = spawn_rng(int(instance_seed), f"{NAMESPACE_ROOT}.marked_dates")
    target_marked_dates = tuple(sorted(int(day) for day in rng.sample(target_dates, k=int(target_count))))
    distractor_marked_dates = tuple(sorted(int(day) for day in rng.sample(distractor_dates, k=int(distractor_count))))
    marked_dates = tuple(sorted(int(day) for day in (*target_marked_dates, *distractor_marked_dates)))
    return tuple(int(day) for day in marked_dates), tuple(int(day) for day in target_marked_dates), int(target_count)


def is_workday_date(day: int, *, start_weekday_index: int, weekend_weekday_indices: Tuple[int, ...]) -> bool:
    """Return whether one date number is a weekday under configured weekend indices."""

    weekday_index = int((int(start_weekday_index) + int(day) - 1) % 7)
    return int(weekday_index) not in set(int(value) for value in weekend_weekday_indices)


def is_weekend_date(day: int, *, start_weekday_index: int, weekend_weekday_indices: Tuple[int, ...]) -> bool:
    """Return whether one date number is in the configured weekend set."""

    return not is_workday_date(
        int(day),
        start_weekday_index=int(start_weekday_index),
        weekend_weekday_indices=tuple(int(value) for value in weekend_weekday_indices),
    )


def resolve_date_range_day_class_query(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    days_in_month: int,
    start_weekday_index: int,
    weekend_weekday_indices: Tuple[int, ...],
    target_weekend: bool,
    count_support: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int]:
    """Resolve one inclusive date-range count query bounded by two marked dates."""

    min_span = int(params.get("date_range_min_span_days", DEFAULTS.date_range_min_span_days))
    max_span = int(params.get("date_range_max_span_days", DEFAULTS.date_range_max_span_days))
    if min_span < 2:
        raise ValueError("date_range_min_span_days must be at least 2")
    if max_span < min_span:
        raise ValueError("date_range_max_span_days must be >= date_range_min_span_days")

    support_values = tuple(int(value) for value in count_support if int(value) >= 1)
    if not support_values:
        raise ValueError("date_range_day_count_support must include at least one positive value")

    feasible_by_count: Dict[int, List[Tuple[int, int, Tuple[int, ...]]]] = {}
    weekend_indices = tuple(int(value) for value in weekend_weekday_indices)
    for start_day in range(1, int(days_in_month) + 1):
        max_end = min(int(days_in_month), int(start_day) + int(max_span) - 1)
        for end_day in range(int(start_day) + int(min_span) - 1, int(max_end) + 1):
            interval_days = tuple(range(int(start_day), int(end_day) + 1))
            annotation_dates = tuple(
                int(day)
                for day in interval_days
                if is_weekend_date(
                    int(day),
                    start_weekday_index=int(start_weekday_index),
                    weekend_weekday_indices=weekend_indices,
                )
                == bool(target_weekend)
            )
            count = int(len(annotation_dates))
            if count in support_values:
                feasible_by_count.setdefault(int(count), []).append(
                    (int(start_day), int(end_day), tuple(int(day) for day in annotation_dates))
                )

    feasible_counts = tuple(sorted(feasible_by_count))
    if not feasible_counts:
        raise ValueError("no feasible inclusive date-range day-class query exists for the sampled month")

    class_key = "weekend" if bool(target_weekend) else "weekday"
    count_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}:date_range_{class_key}_count",
        )
        % len(feasible_counts)
    )
    answer_value = int(feasible_counts[int(count_index)])
    interval_options = feasible_by_count[int(answer_value)]
    interval_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}:date_range_{class_key}_interval",
        )
        % len(interval_options)
    )
    range_start_date, range_end_date, annotation_dates = interval_options[int(interval_index)]
    return (
        (int(range_start_date), int(range_end_date)),
        tuple(int(day) for day in annotation_dates),
        int(answer_value),
        int(range_start_date),
        int(range_end_date),
    )


def resolve_workday_offset_query(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    direction: str,
    days_in_month: int,
    start_weekday_index: int,
    weekend_weekday_indices: Tuple[int, ...],
    workday_offset_support: Tuple[int, ...],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], int, int, int, int]:
    """Resolve one before/after weekday traversal inside a visible calendar month."""

    direction_text = str(direction)
    if direction_text not in {"after", "before"}:
        raise ValueError(f"unsupported workday direction: {direction}")
    workday_dates = [
        int(day)
        for day in range(1, int(days_in_month) + 1)
        if is_workday_date(
            int(day),
            start_weekday_index=int(start_weekday_index),
            weekend_weekday_indices=tuple(int(value) for value in weekend_weekday_indices),
        )
    ]
    if not workday_dates:
        raise ValueError("sampled month has no feasible workday dates")
    feasible: List[Tuple[int, int, int]] = []
    for offset in workday_offset_support:
        offset_value = int(offset)
        if offset_value < 1:
            continue
        for index, reference_date in enumerate(workday_dates):
            target_index = int(index + offset_value) if direction_text == "after" else int(index - offset_value)
            if 0 <= int(target_index) < len(workday_dates):
                feasible.append((int(offset_value), int(reference_date), int(workday_dates[int(target_index)])))
    if not feasible:
        raise ValueError("no feasible workday-offset query exists for the sampled month")
    selected_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}:workday_{direction_text}",
        )
        % len(feasible)
    )
    workday_offset, reference_date, target_date = feasible[int(selected_index)]
    return (
        (int(reference_date),),
        (int(reference_date), int(target_date)),
        int(target_date),
        int(workday_offset),
        int(reference_date),
        int(target_date),
    )


def build_weekday_occurrence_case(instance_seed: int, *, params: Mapping[str, Any]) -> CalendarCase:
    """Build one nth-weekday lookup case from calendar rows.

    This function owns only the operand sampling for weekday and occurrence;
    scene visual axes and month geometry are resolved by the shared common
    field helper before the final case is assembled.
    """

    common = resolve_common_case_fields(instance_seed=int(instance_seed), params=params)
    feasible_pairs: List[Tuple[int, int, int]] = []
    for weekday_index in range(7):
        dates_for_weekday = [
            int(day)
            for day in range(1, int(common["days_in_month"]) + 1)
            if int((int(common["start_weekday_index"]) + int(day) - 1) % 7) == int(weekday_index)
        ]
        for occurrence in common["date_occurrence_support"]:
            if int(occurrence) <= len(dates_for_weekday):
                feasible_pairs.append((int(weekday_index), int(occurrence), int(dates_for_weekday[int(occurrence) - 1])))
    if not feasible_pairs:
        raise ValueError("no feasible weekday-occurrence query exists for the sampled month")
    pair_index = int(
        resolve_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{NAMESPACE_ROOT}:weekday_occurrence_pair",
        )
        % len(feasible_pairs)
    )
    weekday_index, occurrence, answer_value = feasible_pairs[int(pair_index)]
    return _case_from_common(
        common,
        marked_day_class=None,
        marked_day_class_probabilities={},
        marked_dates=(),
        annotation_dates=(int(answer_value),),
        answer_value=int(answer_value),
        weekday_index=int(weekday_index),
        occurrence=int(occurrence),
        workday_direction=None,
        workday_offset=None,
        reference_date=None,
        target_date=None,
    )


def build_date_weekday_label_case(instance_seed: int, *, params: Mapping[str, Any]) -> CalendarCase:
    """Build one date-to-visible-weekday-header lookup case."""

    common = resolve_common_case_fields(instance_seed=int(instance_seed), params=params)
    explicit_date = params.get("target_date", params.get("date_number"))
    if explicit_date is not None:
        answer_value = int(explicit_date)
        if not 1 <= int(answer_value) <= int(common["days_in_month"]):
            raise ValueError("target_date must be within the sampled month")
    else:
        answer_value = 1 + int(
            resolve_selection_index(
                params=params,
                instance_seed=int(instance_seed),
                namespace=f"{NAMESPACE_ROOT}:date_weekday_target_date",
            )
            % int(common["days_in_month"])
        )
    weekday_index = int((int(common["start_weekday_index"]) + int(answer_value) - 1) % 7)
    return _case_from_common(
        common,
        marked_day_class=None,
        marked_day_class_probabilities={},
        marked_dates=(),
        annotation_dates=(int(answer_value),),
        answer_value=int(answer_value),
        weekday_index=int(weekday_index),
        occurrence=None,
        workday_direction=None,
        workday_offset=None,
        reference_date=None,
        target_date=int(answer_value),
    )


def build_marked_day_class_count_case(instance_seed: int, *, params: Mapping[str, Any]) -> CalendarCase:
    """Build one marked weekday/weekend count case."""

    common = resolve_common_case_fields(instance_seed=int(instance_seed), params=params)
    marked_day_class, class_probabilities = resolve_marked_day_class(instance_seed=int(instance_seed), params=params)
    marked_dates, annotation_dates, answer_value = resolve_marked_count_query(
        instance_seed=int(instance_seed),
        params=params,
        days_in_month=int(common["days_in_month"]),
        start_weekday_index=int(common["start_weekday_index"]),
        weekend_weekday_indices=tuple(int(value) for value in common["weekend_weekday_indices"]),
        target_weekend=str(marked_day_class) == "weekend",
    )
    return _case_from_common(
        common,
        marked_day_class=str(marked_day_class),
        marked_day_class_probabilities=dict(class_probabilities),
        marked_dates=tuple(int(day) for day in marked_dates),
        annotation_dates=tuple(int(day) for day in annotation_dates),
        answer_value=int(answer_value),
        weekday_index=None,
        occurrence=None,
        workday_direction=None,
        workday_offset=None,
        reference_date=None,
        target_date=None,
    )


def build_date_range_day_class_count_case(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    day_class: str,
) -> CalendarCase:
    """Build one inclusive date-range day-class count case."""

    common = resolve_common_case_fields(instance_seed=int(instance_seed), params=params)
    marked_day_class = normalize_marked_day_class(day_class)
    marked_dates, annotation_dates, answer_value, range_start_date, range_end_date = resolve_date_range_day_class_query(
        instance_seed=int(instance_seed),
        params=params,
        days_in_month=int(common["days_in_month"]),
        start_weekday_index=int(common["start_weekday_index"]),
        weekend_weekday_indices=tuple(int(value) for value in common["weekend_weekday_indices"]),
        target_weekend=str(marked_day_class) == "weekend",
        count_support=tuple(int(value) for value in common["date_range_day_count_support"]),
    )
    return _case_from_common(
        common,
        marked_day_class=str(marked_day_class),
        marked_day_class_probabilities={
            key: (1.0 if str(key) == str(marked_day_class) else 0.0)
            for key in SUPPORTED_MARKED_DAY_CLASSES
        },
        marked_dates=tuple(int(day) for day in marked_dates),
        annotation_dates=tuple(int(day) for day in annotation_dates),
        answer_value=int(answer_value),
        weekday_index=None,
        occurrence=None,
        workday_direction=None,
        workday_offset=None,
        reference_date=int(range_start_date),
        target_date=int(range_end_date),
    )


def build_workday_offset_case(instance_seed: int, *, params: Mapping[str, Any], direction: str) -> CalendarCase:
    """Build one workday offset lookup case."""

    common = resolve_common_case_fields(instance_seed=int(instance_seed), params=params)
    marked_dates, annotation_dates, answer_value, workday_offset, reference_date, target_date = resolve_workday_offset_query(
        instance_seed=int(instance_seed),
        params=params,
        direction=str(direction),
        days_in_month=int(common["days_in_month"]),
        start_weekday_index=int(common["start_weekday_index"]),
        weekend_weekday_indices=tuple(int(value) for value in common["weekend_weekday_indices"]),
        workday_offset_support=tuple(int(value) for value in common["workday_offset_support"]),
    )
    return _case_from_common(
        common,
        marked_day_class=None,
        marked_day_class_probabilities={},
        marked_dates=tuple(int(day) for day in marked_dates),
        annotation_dates=tuple(int(day) for day in annotation_dates),
        answer_value=int(answer_value),
        weekday_index=None,
        occurrence=None,
        workday_direction=str(direction),
        workday_offset=int(workday_offset),
        reference_date=int(reference_date),
        target_date=int(target_date),
    )


def _case_from_common(
    common: Mapping[str, Any],
    *,
    marked_day_class: str | None,
    marked_day_class_probabilities: Dict[str, float],
    marked_dates: Tuple[int, ...],
    annotation_dates: Tuple[int, ...],
    answer_value: int,
    weekday_index: int | None,
    occurrence: int | None,
    workday_direction: str | None,
    workday_offset: int | None,
    reference_date: int | None,
    target_date: int | None,
) -> CalendarCase:
    """Normalize common calendar fields into one immutable case record.

    The public task files decide which semantic operands and annotation dates
    matter; this helper only copies shared visual/month state and objective
    outputs into a stable replay structure.
    """
    return CalendarCase(
        marked_day_class=marked_day_class,
        scene_variant=str(common["scene_variant"]),
        layout_mode=str(common["layout_mode"]),
        title_mode=str(common["title_mode"]),
        week_start=str(common["week_start"]),
        first_weekday_index=int(common["first_weekday_index"]),
        year=int(common["year"]),
        month=int(common["month"]),
        month_name=str(common["month_name"]),
        row_count=int(common["row_count"]),
        start_weekday_index=int(common["start_weekday_index"]),
        days_in_month=int(common["days_in_month"]),
        marked_dates=tuple(int(day) for day in marked_dates),
        annotation_dates=tuple(int(day) for day in annotation_dates),
        answer_value=int(answer_value),
        weekday_index=(int(weekday_index) if weekday_index is not None else None),
        occurrence=(int(occurrence) if occurrence is not None else None),
        workday_direction=(str(workday_direction) if workday_direction is not None else None),
        workday_offset=(int(workday_offset) if workday_offset is not None else None),
        reference_date=(int(reference_date) if reference_date is not None else None),
        target_date=(int(target_date) if target_date is not None else None),
        weekend_weekday_indices=tuple(int(value) for value in common["weekend_weekday_indices"]),
        date_occurrence_support=tuple(int(value) for value in common["date_occurrence_support"]),
        marked_weekend_count_support=tuple(int(value) for value in common["marked_weekend_count_support"]),
        marked_weekday_count_support=tuple(int(value) for value in common["marked_weekday_count_support"]),
        marked_weekday_distractor_support=tuple(int(value) for value in common["marked_weekday_distractor_support"]),
        marked_weekend_distractor_support=tuple(int(value) for value in common["marked_weekend_distractor_support"]),
        workday_offset_support=tuple(int(value) for value in common["workday_offset_support"]),
        marked_day_class_probabilities=dict(marked_day_class_probabilities),
        scene_variant_probabilities=dict(common["scene_variant_probabilities"]),
        layout_mode_probabilities=dict(common["layout_mode_probabilities"]),
        title_mode_probabilities=dict(common["title_mode_probabilities"]),
        week_start_probabilities=dict(common["week_start_probabilities"]),
    )
