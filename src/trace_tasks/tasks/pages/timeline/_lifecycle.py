"""Scene-private lifecycle for milestone-timeline page tasks."""

from __future__ import annotations

import calendar
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...pages.shared.information_style import (
  PagesInformationStyle,
  make_pages_information_background,
  resolve_pages_information_style,
)
from ...shared.config_defaults import group_default, split_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
  PROMPT_OUTPUT_MODES,
  build_prompt_query_spec,
  build_prompt_trace_artifacts,
  render_task_prompt_variants,
)
from ...shared.support_sampling import resolve_integer_choice
from ...shared.time_artifact_style import (
  SUPPORTED_TIME_ARTIFACT_COLOR_NAMES,
  SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS,
  TimeArtifactTimelineTheme,
)
from ...shared.time_artifact_task_support import resolve_time_artifact_named_variant, resolve_time_artifact_selection_index
from ...shared.time_format import format_month_day_label, month_name
from ...shared.text_legibility import READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO, contrast_ratio
from .shared.rendering import (
  SUPPORTED_PAGE_TIMELINE_SCENE_VARIANTS,
  TimelineEventSpec,
  render_timeline_scene,
  resolve_timeline_render_params,
)
from ..shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults


DOMAIN = "pages"
SCENE = "timeline"
TASK_NAMESPACE = "pages.timeline"
PROMPT_BUNDLE = "pages_timeline_v1"
PROMPT_SCENE_KEY = "milestone_timeline"
PROMPT_TASK_KEY = "timeline_milestone_query"
INTERVAL_EVENT_MODE = "events_inside_or_outside_reference_span"
THRESHOLD_EVENT_COUNT_MODE = "events_before_or_after_threshold_date"
RELATIVE_POSITION_EVENT_LABEL_MODE = "event_label_at_relative_timeline_position"
_SUPPORTED_INTERVAL_RELATIONS: Tuple[str, ...] = ("between", "outside")
_SUPPORTED_THRESHOLD_RELATIONS: Tuple[str, ...] = ("before", "after")
_SUPPORTED_RELATIVE_POSITION_RELATIONS: Tuple[str, ...] = ("before", "after")

_VISUAL_SCAN_BASE_BY_SCENE = {
  "classic": 0.40,
  "roadmap": 0.48,
  "minimal": 0.30,
}
_VISUAL_SCAN_STYLE_BONUS = {
  "studio": 0.00,
  "accented": 0.04,
  "marker": 0.06,
}
_CLUTTER_BASE_BY_SCENE = {
  "classic": 0.28,
  "roadmap": 0.34,
  "minimal": 0.20,
}
_CLUTTER_STYLE_BONUS = {
  "studio": 0.00,
  "accented": 0.04,
  "marker": 0.07,
}
_BALANCE_SALT = 73961


@dataclass(frozen=True)
class _TaskDefaults:
  """Stable fallback defaults for milestone-timeline scenes."""

  year_min: int = 2024
  year_max: int = 2030
  event_count_support: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)
  between_count_support: Tuple[int, ...] = (1, 2, 3, 4)
  outside_count_support: Tuple[int, ...] = (1, 2, 3, 4)
  threshold_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
  relative_offset_support: Tuple[int, ...] = (1, 2, 3, 4)
  event_label_pool: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")
  canvas_width: int = 1120
  canvas_height: int = 700
  outer_margin_px: int = 34
  title_height_px: int = 86
  title_gap_px: int = 18
  panel_corner_radius_px: int = 20
  panel_outline_width_px: int = 3
  axis_width_px: int = 4
  axis_tick_height_px: int = 12
  marker_radius_px: int = 10
  marker_outline_width_px: int = 3
  connector_width_px: int = 3
  card_width_px: int = 106
  card_height_px: int = 70
  card_corner_radius_px: int = 14
  card_outline_width_px: int = 3
  label_font_size_px: int = 22
  date_font_size_px: int = 16
  title_font_size_px: int = 30
  subtitle_font_size_px: int = 18
  event_vertical_gap_px: int = 16
  event_stem_length_px: int = 44


@dataclass(frozen=True)
class _RawTimelineEvent:
  """Task-internal milestone record before rendering."""

  event_id: str
  label: str
  day_of_month: int
  order_index: int
  reference_kind: str = "none"

  @property
  def date_text(self) -> str:
    """Return one placeholder date label for debugging before month binding."""

    return str(self.day_of_month)


@dataclass(frozen=True)
class _ResolvedQuery:
  """Resolved semantic and visual support for one milestone-timeline query."""

  query_id: str
  interval_relation: str
  scene_variant: str
  style_variant: str
  accent_color_name: str
  year: int
  month: int
  month_name: str
  title_text: str
  subtitle_text: str
  raw_events: Tuple[_RawTimelineEvent, ...]
  answer_value: int | None
  answer_label: str
  answer_event_ids: Tuple[str, ...]
  reference_event_ids: Tuple[str, ...]
  endpoint_event_ids: Tuple[str, ...]
  prompt_endpoint_event_ids: Tuple[str, ...]
  prompt_reference_event_ids: Tuple[str, ...]
  threshold_day: int | None
  threshold_date_text: str
  reference_date_text: str
  relative_offset: int | None
  relative_offset_phrase: str
  event_count_support: Tuple[int, ...]
  between_count_support: Tuple[int, ...]
  outside_count_support: Tuple[int, ...]
  threshold_count_support: Tuple[int, ...]
  relative_offset_support: Tuple[int, ...]
  query_id_probabilities: Dict[str, float]
  interval_relation_probabilities: Dict[str, float]
  scene_variant_probabilities: Dict[str, float]
  style_variant_probabilities: Dict[str, float]
  accent_color_name_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
  _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)


def _readable_on(surface_rgb: Sequence[int], *, preferred_rgb: Sequence[int]) -> Tuple[int, int, int]:
  surface = tuple(int(value) for value in surface_rgb[:3])
  candidates = (
    tuple(int(value) for value in preferred_rgb[:3]),
    (255, 255, 255),
    (250, 252, 255),
    (10, 14, 22),
    (0, 0, 0),
  )
  passing = [
    candidate
    for candidate in candidates
    if float(contrast_ratio(candidate, surface)) >= float(READ_REQUIRED_TEXT_MIN_CONTRAST_RATIO)
  ]
  if passing:
    return tuple(int(value) for value in passing[0])
  return max(candidates, key=lambda candidate: float(contrast_ratio(candidate, surface)))


def _timeline_theme_from_information_style(style: PagesInformationStyle) -> TimeArtifactTimelineTheme:
  reference_fill = tuple(int(value) for value in style.highlight_rgb)
  return TimeArtifactTimelineTheme(
    accent_color_name=str(style.palette_id),
    style_variant=str(style.treatment),
    panel_fill_rgb=tuple(int(value) for value in style.panel_fill_rgb),
    panel_outline_rgb=tuple(int(value) for value in style.panel_border_rgb),
    title_text_rgb=tuple(int(value) for value in style.text_rgb),
    subtitle_text_rgb=tuple(int(value) for value in style.muted_text_rgb),
    axis_line_rgb=tuple(int(value) for value in style.axis_rgb),
    tick_line_rgb=tuple(int(value) for value in style.guide_rgb),
    connector_line_rgb=tuple(int(value) for value in style.connector_rgb),
    marker_fill_rgb=tuple(int(value) for value in style.callout_fill_rgb),
    marker_outline_rgb=tuple(int(value) for value in style.accent_rgb),
    event_fill_rgb=tuple(int(value) for value in style.surface_alt_rgb),
    event_outline_rgb=tuple(int(value) for value in style.panel_border_rgb),
    event_text_rgb=tuple(int(value) for value in style.text_rgb),
    event_subtext_rgb=tuple(int(value) for value in style.muted_text_rgb),
    primary_reference_fill_rgb=reference_fill,
    primary_reference_outline_rgb=tuple(int(value) for value in style.accent_rgb),
    primary_reference_text_rgb=_readable_on(reference_fill, preferred_rgb=style.text_rgb),
    secondary_reference_fill_rgb=reference_fill,
    secondary_reference_outline_rgb=tuple(int(value) for value in style.secondary_accent_rgb),
    secondary_reference_text_rgb=_readable_on(reference_fill, preferred_rgb=style.text_rgb),
  )


def _resolve_named_variant(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
  explicit_key: str,
  weights_key: str,
  balance_flag_key: str,
  supported: Tuple[str, ...],
  namespace: str,
) -> Tuple[str, Dict[str, float]]:
  """Resolve one balanced named timeline axis."""

  rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.{namespace}")
  return resolve_time_artifact_named_variant(
    rng,
    params=params,
    gen_defaults=_GEN_DEFAULTS,
    explicit_key=str(explicit_key),
    weights_key=str(weights_key),
    balance_flag_key=str(balance_flag_key),
    supported=supported,
    instance_seed=int(instance_seed),
    task_id=TASK_NAMESPACE,
    namespace=str(namespace),
  )


def _resolve_int_support(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, ...]:
  """Resolve one integer support list from config or explicit params."""

  raw_values = params.get(key, group_default(_GEN_DEFAULTS, key, fallback))
  resolved: List[int] = []
  for raw_value in raw_values:
    value = int(raw_value)
    if value not in resolved:
      resolved.append(value)
  if not resolved:
    raise ValueError(f"{key} must not be empty for {TASK_NAMESPACE}")
  return tuple(int(value) for value in resolved)


def _resolve_str_support(params: Mapping[str, Any], key: str, fallback: Sequence[str]) -> Tuple[str, ...]:
  """Resolve one string support list from config or explicit params."""

  raw_values = params.get(key, group_default(_GEN_DEFAULTS, key, fallback))
  resolved: List[str] = []
  for raw_value in raw_values:
    value = str(raw_value).strip()
    if value and value not in resolved:
      resolved.append(value)
  if not resolved:
    raise ValueError(f"{key} must not be empty for {TASK_NAMESPACE}")
  return tuple(str(value) for value in resolved)


def _normalize_interval_relation(value: Any) -> str:
  """Normalize one timeline interval relation."""

  normalized = str(value).strip().lower()
  if normalized in _SUPPORTED_INTERVAL_RELATIONS:
    return normalized
  raise ValueError(f"unsupported interval_relation: {value}")


def _normalize_threshold_relation(value: Any) -> str:
  """Normalize one one-sided threshold-count relation."""

  normalized = str(value).strip().lower()
  if normalized in _SUPPORTED_THRESHOLD_RELATIONS:
    return normalized
  raise ValueError(f"unsupported threshold_relation: {value}")


def _normalize_relative_position_relation(value: Any) -> str:
  """Normalize one relative-position label relation."""

  normalized = str(value).strip().lower()
  if normalized in _SUPPORTED_RELATIVE_POSITION_RELATIONS:
    return normalized
  raise ValueError(f"unsupported relative_position_relation: {value}")


def _resolve_support_selection_index(
  *,
  params: Mapping[str, Any],
  instance_seed: int,
  namespace: str,
) -> int:
  """Return a support index decoupled from query-id cycling."""

  return int(resolve_time_artifact_selection_index(params=params, instance_seed=int(instance_seed), namespace=str(namespace)))


def _decoupled_named_axis_params(
  *,
  params: Mapping[str, Any],
  axis_key: str,
  namespace: str,
) -> Mapping[str, Any]:
  """Return params with one balanced visual axis decoupled from query ids."""

  _ = axis_key, namespace
  return params


def _sample_month(instance_seed: int, params: Mapping[str, Any]) -> Tuple[int, int, int]:
  """Sample one Gregorian month and return `(year, month, days_in_month)`."""

  explicit_year = params.get("year")
  explicit_month = params.get("month")
  if explicit_year is not None and explicit_month is None:
    raise ValueError("month must be provided when year is explicit for page timelines")
  if explicit_month is not None and explicit_year is None:
    raise ValueError("year must be provided when month is explicit for page timelines")

  if explicit_year is not None and explicit_month is not None:
    _, days_in_month = calendar.monthrange(int(explicit_year), int(explicit_month))
    return int(explicit_year), int(explicit_month), int(days_in_month)

  year_min = int(params.get("year_min", group_default(_GEN_DEFAULTS, "year_min", _DEFAULTS.year_min)))
  year_max = int(params.get("year_max", group_default(_GEN_DEFAULTS, "year_max", _DEFAULTS.year_max)))
  if int(year_max) < int(year_min):
    raise ValueError("year_max must be >= year_min for page timelines")
  year_support = tuple(range(int(year_min), int(year_max) + 1))
  year = int(
    year_support[
      int(
        _resolve_support_selection_index(
          params=params,
          instance_seed=int(instance_seed),
          namespace=f"{TASK_NAMESPACE}:year",
        )
        % len(year_support)
      )
    ]
  )
  month = 1 + int(
    _resolve_support_selection_index(
      params=params,
      instance_seed=int(instance_seed),
      namespace=f"{TASK_NAMESPACE}:month",
    )
    % 12
  )
  _, days_in_month = calendar.monthrange(int(year), int(month))
  return int(year), int(month), int(days_in_month)


def _build_raw_events(
  *,
  month: int,
  day_values: Sequence[int],
  labels: Sequence[str],
  primary_reference_index: int | None,
  secondary_reference_index: int | None,
) -> Tuple[_RawTimelineEvent, ...]:
  """Build one ordered milestone list from sampled labels and dates."""

  del month
  events: List[_RawTimelineEvent] = []
  for index, (label, day_of_month) in enumerate(zip(labels, day_values)):
    reference_kind = "none"
    if primary_reference_index is not None and int(index) == int(primary_reference_index):
      reference_kind = "primary"
    elif secondary_reference_index is not None and int(index) == int(secondary_reference_index):
      reference_kind = "secondary"
    events.append(
      _RawTimelineEvent(
        event_id=f"event_{str(label).lower()}",
        label=str(label),
        day_of_month=int(day_of_month),
        order_index=int(index),
        reference_kind=str(reference_kind),
      )
    )
  return tuple(events)


def _sample_interval_query(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
  interval_relation: str,
  days_in_month: int,
  event_count_support: Tuple[int, ...],
  between_count_support: Tuple[int, ...],
  outside_count_support: Tuple[int, ...],
  event_label_pool: Tuple[str, ...],
) -> Tuple[int, int, int, Tuple[int, ...], Tuple[int, ...]]:
  """Sample the existing interval-membership query branch."""

  max_event_count = min(int(days_in_month), len(event_label_pool), max(int(value) for value in event_count_support))
  if int(max_event_count) < min(int(value) for value in event_count_support):
    raise ValueError("timeline month does not have enough days for the configured event_count_support")

  if str(interval_relation) == "between":
    feasible_answers = [int(value) for value in between_count_support if 0 <= int(value) <= int(max_event_count - 2)]
    if not feasible_answers:
      raise ValueError("between_count_support has no feasible values for page timelines")
    answer_value, _answer_probabilities = resolve_integer_choice(
      instance_seed=int(instance_seed),
      params={**dict(params), "between_count_support": tuple(feasible_answers)},
      gen_defaults=_GEN_DEFAULTS,
      support_key="between_count_support",
      explicit_key="answer_value",
      fallback_support=feasible_answers,
      namespace=f"{TASK_NAMESPACE}:between_answer",
      balanced_flag_key="balanced_between_count_sampling",
    )
    answer_value = int(answer_value)
    feasible_event_counts = [int(value) for value in event_count_support if int(answer_value + 2) <= int(value) <= int(max_event_count)]
    event_count = int(
      feasible_event_counts[
        int(
          _resolve_support_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}:between_event_count",
          )
          % len(feasible_event_counts)
        )
      ]
    )
    left_index_support = list(range(0, int(event_count - answer_value - 1)))
    primary_reference_index = int(
      left_index_support[
        int(
          _resolve_support_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}:between_left_index",
          )
          % len(left_index_support)
        )
      ]
    )
    secondary_reference_index = int(primary_reference_index + answer_value + 1)
    answer_indices = tuple(range(int(primary_reference_index + 1), int(secondary_reference_index)))
  else:
    feasible_answers = [int(value) for value in outside_count_support if 1 <= int(value) <= int(max_event_count - 2)]
    if not feasible_answers:
      raise ValueError("outside_count_support has no feasible values for page timelines")
    answer_value, _answer_probabilities = resolve_integer_choice(
      instance_seed=int(instance_seed),
      params={**dict(params), "outside_count_support": tuple(feasible_answers)},
      gen_defaults=_GEN_DEFAULTS,
      support_key="outside_count_support",
      explicit_key="answer_value",
      fallback_support=feasible_answers,
      namespace=f"{TASK_NAMESPACE}:outside_answer",
      balanced_flag_key="balanced_outside_count_sampling",
    )
    answer_value = int(answer_value)
    feasible_event_counts = [int(value) for value in event_count_support if int(answer_value + 2) <= int(value) <= int(max_event_count)]
    event_count = int(
      feasible_event_counts[
        int(
          _resolve_support_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}:outside_event_count",
          )
          % len(feasible_event_counts)
        )
      ]
    )
    left_outside_support = tuple(range(0, int(answer_value) + 1))
    left_outside_count = int(
      left_outside_support[
        int(
          _resolve_support_selection_index(
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_NAMESPACE}:outside_left_count",
          )
          % len(left_outside_support)
        )
      ]
    )
    right_outside_count = int(answer_value - left_outside_count)
    primary_reference_index = int(left_outside_count)
    secondary_reference_index = int(event_count - right_outside_count - 1)
    answer_indices = tuple(range(0, int(primary_reference_index))) + tuple(
      range(int(secondary_reference_index + 1), int(event_count))
    )
  return (
    int(event_count),
    int(primary_reference_index),
    int(secondary_reference_index),
    tuple(int(index) for index in answer_indices),
    tuple(),
  )


def _sample_threshold_query(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
  threshold_relation: str,
  days_in_month: int,
  event_count_support: Tuple[int, ...],
  threshold_count_support: Tuple[int, ...],
  event_label_pool: Tuple[str, ...],
) -> Tuple[int, int, int, Tuple[int, ...], Tuple[int, ...]]:
  """Sample a before/after threshold-date count query.

  The threshold date is a prompt-only operand, not a highlighted event. It is
  chosen so it is not equal to any event-card date and exactly ``answer_value``
  event cards lie on the requested side.
  """

  max_event_count = min(int(days_in_month), len(event_label_pool), max(int(value) for value in event_count_support))
  feasible_event_counts = [int(value) for value in event_count_support if 2 <= int(value) <= int(max_event_count)]
  if not feasible_event_counts:
    raise ValueError("event_count_support has no feasible values for timeline threshold-count tasks")
  threshold_relation = _normalize_threshold_relation(threshold_relation)

  event_count, _event_count_probabilities = resolve_integer_choice(
    instance_seed=int(instance_seed),
    params=params,
    gen_defaults=_GEN_DEFAULTS,
    support_key="event_count_support",
    explicit_key="event_count",
    fallback_support=feasible_event_counts,
    namespace=f"{TASK_NAMESPACE}:threshold_event_count",
    balanced_flag_key="balanced_event_count_sampling",
  )
  event_count = int(event_count)
  if int(event_count) not in set(feasible_event_counts):
    feasible_event_counts = [int(value) for value in feasible_event_counts if int(value) >= 2]
    if int(event_count) not in set(feasible_event_counts):
      raise ValueError("selected event_count is not feasible for timeline threshold-count tasks")

  feasible_answers = [
    int(value)
    for value in threshold_count_support
    if 1 <= int(value) <= int(event_count - 1)
  ]
  if not feasible_answers:
    raise ValueError("threshold_count_support has no feasible values for page timelines")
  answer_value, _answer_probabilities = resolve_integer_choice(
    instance_seed=int(instance_seed),
    params={**dict(params), "threshold_count_support": tuple(feasible_answers)},
    gen_defaults=_GEN_DEFAULTS,
    support_key="threshold_count_support",
    explicit_key="answer_value",
    fallback_support=feasible_answers,
    namespace=f"{TASK_NAMESPACE}:threshold_answer:{threshold_relation}",
    balanced_flag_key="balanced_threshold_count_sampling",
  )
  answer_value = int(answer_value)

  before_count = int(answer_value) if threshold_relation == "before" else int(event_count - answer_value)
  after_count = int(event_count - answer_value) if threshold_relation == "before" else int(answer_value)
  threshold_support = tuple(
    int(day)
    for day in range(1, int(days_in_month) + 1)
    if int(day - 1) >= int(before_count) and int(days_in_month - day) >= int(after_count)
  )
  if not threshold_support:
    raise ValueError("timeline threshold-count query cannot place a feasible threshold date")
  rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.threshold_days:{threshold_relation}")
  threshold_day = int(rng.choice(list(threshold_support)))
  before_pool = list(range(1, int(threshold_day)))
  after_pool = list(range(int(threshold_day + 1), int(days_in_month) + 1))
  day_values = tuple(
    sorted(
      [
        *[int(value) for value in rng.sample(before_pool, k=int(before_count))],
        *[int(value) for value in rng.sample(after_pool, k=int(after_count))],
      ]
    )
  )
  if len(day_values) != int(event_count):
    raise ValueError("timeline threshold-count day construction produced the wrong event count")
  if int(threshold_day) in set(int(value) for value in day_values):
    raise ValueError("threshold date must not be reused as an event date")
  if threshold_relation == "before":
    answer_indices = tuple(index for index, day in enumerate(day_values) if int(day) < int(threshold_day))
  else:
    answer_indices = tuple(index for index, day in enumerate(day_values) if int(day) > int(threshold_day))
  if len(answer_indices) != int(answer_value):
    raise ValueError("timeline threshold-count construction did not match the target answer")
  return int(event_count), int(answer_value), int(threshold_day), tuple(int(value) for value in day_values), tuple(int(index) for index in answer_indices)


def _sample_relative_position_query(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
  relative_relation: str,
  days_in_month: int,
  event_count_support: Tuple[int, ...],
  relative_offset_support: Tuple[int, ...],
  event_label_pool: Tuple[str, ...],
) -> Tuple[int, int, int, int, Tuple[int, ...]]:
  """Sample a label target at a fixed offset from a prompt-dated event."""

  max_event_count = min(int(days_in_month), len(event_label_pool), max(int(value) for value in event_count_support))
  relative_relation = _normalize_relative_position_relation(relative_relation)
  feasible_offsets = [
    int(value)
    for value in relative_offset_support
    if 1 <= int(value) <= int(max_event_count - 1)
  ]
  if not feasible_offsets:
    raise ValueError("relative_offset_support has no feasible values for page timelines")
  relative_offset, _offset_probabilities = resolve_integer_choice(
    instance_seed=int(instance_seed),
    params={**dict(params), "relative_offset_support": tuple(feasible_offsets)},
    gen_defaults=_GEN_DEFAULTS,
    support_key="relative_offset_support",
    explicit_key="relative_offset",
    fallback_support=feasible_offsets,
    namespace=f"{TASK_NAMESPACE}:relative_offset:{relative_relation}",
    balanced_flag_key="balanced_relative_offset_sampling",
  )
  relative_offset = int(relative_offset)

  feasible_event_counts = [
    int(value)
    for value in event_count_support
    if int(value) >= int(relative_offset + 1) and int(value) <= int(max_event_count)
  ]
  if not feasible_event_counts:
    raise ValueError("event_count_support cannot support the selected relative offset")
  event_count, _event_count_probabilities = resolve_integer_choice(
    instance_seed=int(instance_seed),
    params={**dict(params), "event_count_support": tuple(feasible_event_counts)},
    gen_defaults=_GEN_DEFAULTS,
    support_key="event_count_support",
    explicit_key="event_count",
    fallback_support=feasible_event_counts,
    namespace=f"{TASK_NAMESPACE}:relative_event_count",
    balanced_flag_key="balanced_event_count_sampling",
  )
  event_count = int(event_count)

  if relative_relation == "after":
    reference_index_support = tuple(range(0, int(event_count - relative_offset)))
    target_index_delta = int(relative_offset)
  else:
    reference_index_support = tuple(range(int(relative_offset), int(event_count)))
    target_index_delta = -int(relative_offset)
  rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.relative_position:{relative_relation}")
  reference_index = int(rng.choice(list(reference_index_support)))
  target_index = int(reference_index + target_index_delta)
  if not (0 <= target_index < event_count):
    raise ValueError("relative-position timeline target index is out of range")

  day_values = tuple(
    sorted(
      int(value)
      for value in rng.sample(list(range(1, int(days_in_month) + 1)), k=int(event_count))
    )
  )
  return (
    int(event_count),
    int(reference_index),
    int(target_index),
    int(relative_offset),
    tuple(int(value) for value in day_values),
  )


def _relative_offset_phrase(relative_offset: int) -> str:
  """Return a natural prompt phrase for one card offset."""

  offset = int(relative_offset)
  return "1 card" if offset == 1 else f"{offset} cards"


def _resolve_query(
  instance_seed: int,
  *,
  params: Mapping[str, Any],
  branch_name: str,
  branch_probabilities: Mapping[str, float],
  program_mode: str,
  interval_relation: str,
) -> _ResolvedQuery:
  """Resolve one concrete milestone timeline from task-owned arguments."""

  branch_probabilities_map = {
    str(key): float(value)
    for key, value in branch_probabilities.items()
  }
  if str(program_mode) == THRESHOLD_EVENT_COUNT_MODE:
    threshold_relation = _normalize_threshold_relation(interval_relation)
    interval_relation = str(threshold_relation)
    interval_relation_probabilities = {
      key: (1.0 if key == str(threshold_relation) else 0.0)
      for key in _SUPPORTED_THRESHOLD_RELATIONS
    }
  elif str(program_mode) == RELATIVE_POSITION_EVENT_LABEL_MODE:
    relative_position_relation = _normalize_relative_position_relation(interval_relation)
    interval_relation = str(relative_position_relation)
    threshold_relation = ""
    interval_relation_probabilities = {
      key: (1.0 if key == str(relative_position_relation) else 0.0)
      for key in _SUPPORTED_RELATIVE_POSITION_RELATIONS
    }
  else:
    threshold_relation = ""
    relative_position_relation = ""
    interval_relation = _normalize_interval_relation(interval_relation)
    interval_relation_probabilities = {
      key: (1.0 if key == str(interval_relation) else 0.0)
      for key in _SUPPORTED_INTERVAL_RELATIONS
    }
  scene_variant, scene_variant_probabilities = _resolve_named_variant(
    instance_seed=int(instance_seed),
    params=_decoupled_named_axis_params(
      params=params,
      axis_key="scene_variant",
      namespace=f"{TASK_NAMESPACE}:scene_variant",
    ),
    explicit_key="scene_variant",
    weights_key="scene_variant_weights",
    balance_flag_key="balanced_scene_variant_sampling",
    supported=SUPPORTED_PAGE_TIMELINE_SCENE_VARIANTS,
    namespace="scene_variant",
  )
  style_variant, style_variant_probabilities = _resolve_named_variant(
    instance_seed=int(instance_seed),
    params=_decoupled_named_axis_params(
      params=params,
      axis_key="style_variant",
      namespace=f"{TASK_NAMESPACE}:style_variant",
    ),
    explicit_key="style_variant",
    weights_key="style_variant_weights",
    balance_flag_key="balanced_style_variant_sampling",
    supported=SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS,
    namespace="style_variant",
  )
  accent_color_name, accent_color_name_probabilities = _resolve_named_variant(
    instance_seed=int(instance_seed),
    params=_decoupled_named_axis_params(
      params=params,
      axis_key="accent_color_name",
      namespace=f"{TASK_NAMESPACE}:accent_color_name",
    ),
    explicit_key="accent_color_name",
    weights_key="accent_color_name_weights",
    balance_flag_key="balanced_accent_color_name_sampling",
    supported=SUPPORTED_TIME_ARTIFACT_COLOR_NAMES,
    namespace="accent_color_name",
  )

  year, month, days_in_month = _sample_month(int(instance_seed), params)
  event_count_support = _resolve_int_support(params, "event_count_support", _DEFAULTS.event_count_support)
  between_count_support = _resolve_int_support(params, "between_count_support", _DEFAULTS.between_count_support)
  outside_count_support = _resolve_int_support(params, "outside_count_support", _DEFAULTS.outside_count_support)
  threshold_count_support = _resolve_int_support(params, "threshold_count_support", _DEFAULTS.threshold_count_support)
  relative_offset_support = _resolve_int_support(params, "relative_offset_support", _DEFAULTS.relative_offset_support)
  event_label_pool = _resolve_str_support(params, "event_label_pool", _DEFAULTS.event_label_pool)

  threshold_day: int | None = None
  prompt_reference_index: int | None = None
  relative_offset: int | None = None
  answer_value: int | None
  if str(program_mode) == THRESHOLD_EVENT_COUNT_MODE:
    event_count, answer_value, threshold_day, day_values, answer_indices = _sample_threshold_query(
      instance_seed=int(instance_seed),
      params=params,
      threshold_relation=str(threshold_relation),
      days_in_month=int(days_in_month),
      event_count_support=tuple(event_count_support),
      threshold_count_support=tuple(threshold_count_support),
      event_label_pool=tuple(event_label_pool),
    )
    primary_reference_index = None
    secondary_reference_index = None
    endpoint_indices: Tuple[int, ...] = tuple()
  elif str(program_mode) == RELATIVE_POSITION_EVENT_LABEL_MODE:
    event_count, prompt_reference_index, target_index, relative_offset, day_values = _sample_relative_position_query(
      instance_seed=int(instance_seed),
      params=params,
      relative_relation=str(relative_position_relation),
      days_in_month=int(days_in_month),
      event_count_support=tuple(event_count_support),
      relative_offset_support=tuple(relative_offset_support),
      event_label_pool=tuple(event_label_pool),
    )
    primary_reference_index = None
    secondary_reference_index = None
    endpoint_indices = tuple()
    answer_indices = (int(target_index),)
    answer_value = None
  else:
    event_count, primary_reference_index, secondary_reference_index, answer_indices, endpoint_indices = _sample_interval_query(
      instance_seed=int(instance_seed),
      params=params,
      interval_relation=str(interval_relation),
      days_in_month=int(days_in_month),
      event_count_support=tuple(event_count_support),
      between_count_support=tuple(between_count_support),
      outside_count_support=tuple(outside_count_support),
      event_label_pool=tuple(event_label_pool),
    )
    if str(interval_relation) == "between":
      answer_value = int(len(answer_indices))
    else:
      answer_value = int(len(answer_indices))
    rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.days")
    day_values = tuple(sorted(int(value) for value in rng.sample(list(range(1, int(days_in_month) + 1)), k=int(event_count))))

  labels = tuple(str(label) for label in event_label_pool[:event_count])
  raw_events = _build_raw_events(
    month=int(month),
    day_values=tuple(day_values),
    labels=labels,
    primary_reference_index=(int(primary_reference_index) if primary_reference_index is not None else None),
    secondary_reference_index=(int(secondary_reference_index) if secondary_reference_index is not None else None),
  )
  answer_event_ids = tuple(str(raw_events[index].event_id) for index in answer_indices)
  answer_label = str(raw_events[answer_indices[0]].label) if len(answer_indices) == 1 else ""
  reference_event_ids = tuple(
    str(raw_events[index].event_id)
    for index in (primary_reference_index, secondary_reference_index)
    if index is not None
  )
  endpoint_event_ids = tuple(str(raw_events[index].event_id) for index in endpoint_indices)
  prompt_endpoint_event_ids = tuple(endpoint_event_ids)
  prompt_reference_event_ids = (
    (str(raw_events[int(prompt_reference_index)].event_id),)
    if prompt_reference_index is not None
    else tuple()
  )
  reference_date_text = (
    str(format_month_day_label(int(month), int(raw_events[int(prompt_reference_index)].day_of_month)))
    if prompt_reference_index is not None
    else ""
  )

  return _ResolvedQuery(
    query_id=str(branch_name),
    interval_relation=str(interval_relation),
    scene_variant=str(scene_variant),
    style_variant=str(style_variant),
    accent_color_name=str(accent_color_name),
    year=int(year),
    month=int(month),
    month_name=str(month_name(int(month))),
    title_text="Milestone timeline",
    subtitle_text=f"{month_name(int(month))} {int(year)}",
    raw_events=tuple(raw_events),
    answer_value=(int(answer_value) if answer_value is not None else None),
    answer_label=str(answer_label),
    answer_event_ids=tuple(answer_event_ids),
    reference_event_ids=tuple(reference_event_ids),
    endpoint_event_ids=tuple(endpoint_event_ids),
    prompt_endpoint_event_ids=tuple(prompt_endpoint_event_ids),
    prompt_reference_event_ids=tuple(prompt_reference_event_ids),
    threshold_day=(int(threshold_day) if threshold_day is not None else None),
    threshold_date_text=(
      str(format_month_day_label(int(month), int(threshold_day)))
      if threshold_day is not None
      else ""
    ),
    reference_date_text=str(reference_date_text),
    relative_offset=(int(relative_offset) if relative_offset is not None else None),
    relative_offset_phrase=(
      _relative_offset_phrase(int(relative_offset))
      if relative_offset is not None
      else ""
    ),
    event_count_support=tuple(int(value) for value in event_count_support),
    between_count_support=tuple(int(value) for value in between_count_support),
    outside_count_support=tuple(int(value) for value in outside_count_support),
    threshold_count_support=tuple(int(value) for value in threshold_count_support),
    relative_offset_support=tuple(int(value) for value in relative_offset_support),
    query_id_probabilities=dict(branch_probabilities_map),
    interval_relation_probabilities=dict(interval_relation_probabilities),
    scene_variant_probabilities=dict(scene_variant_probabilities),
    style_variant_probabilities=dict(style_variant_probabilities),
    accent_color_name_probabilities=dict(accent_color_name_probabilities),
  )


def _resolve_card_side(*, scene_variant: str, order_index: int) -> str:
  """Resolve whether one event card sits above or below the timeline axis."""

  if str(scene_variant) == "minimal":
    return "above"
  if str(scene_variant) == "roadmap":
    return "above" if (int(order_index) % 3) != 1 else "below"
  return "above" if (int(order_index) % 2) == 0 else "below"



def build_timeline_response(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
  selected_branch: str,
  branch_probabilities: Mapping[str, float],
  program_mode: str,
  interval_relation: str,
  prompt_query_key: str,
  source_query_name: str,
) -> TaskOutput:
  """Build one complete milestone-timeline task response."""

  query = _resolve_query(
    int(instance_seed),
    params=params,
    branch_name=str(selected_branch),
    branch_probabilities=branch_probabilities,
    program_mode=str(program_mode),
    interval_relation=str(interval_relation),
  )
  render_params = resolve_timeline_render_params(
    params,
    render_defaults=_RENDER_DEFAULTS,
    fallback_values=asdict(_DEFAULTS),
    instance_seed=int(instance_seed),
  )
  information_style, information_style_meta = resolve_pages_information_style(
    instance_seed=int(instance_seed),
    params=params,
    scene_id=SCENE,
  )
  timeline_theme = _timeline_theme_from_information_style(information_style)

  background, background_meta = make_pages_information_background(
    canvas_width=int(render_params.canvas_width),
    canvas_height=int(render_params.canvas_height),
    style=information_style,
    instance_seed=int(instance_seed),
    namespace="pages.timeline.information_scene_background",
  )
  background_meta = dict(background_meta)
  background_meta["information_scene_style"] = dict(information_style_meta)
  image = background.copy().convert("RGB")
  rendered_events = tuple(
    TimelineEventSpec(
      event_id=str(event.event_id),
      label=str(event.label),
      date_text=str(format_month_day_label(int(query.month), int(event.day_of_month))),
      order_index=int(event.order_index),
      anchor_x_px=0.0,
      card_side=str(
        _resolve_card_side(
          scene_variant=str(query.scene_variant),
          order_index=int(event.order_index),
        )
      ),
      reference_kind=str(event.reference_kind),
    )
    for event in query.raw_events
  )
  rendered_scene = render_timeline_scene(
    image,
    title_text=str(query.title_text),
    subtitle_text=str(query.subtitle_text),
    events=rendered_events,
    scene_variant=str(query.scene_variant),
    render_params=render_params,
    visual_theme=timeline_theme,
  )
  image, post_noise_meta = apply_post_image_noise(
    image,
    instance_seed=int(instance_seed),
    params=params,
    default_config=POST_IMAGE_NOISE_DEFAULTS,
  )

  dynamic_slots: Dict[str, str] = {}
  if str(program_mode) == THRESHOLD_EVENT_COUNT_MODE:
    dynamic_slots["threshold_date_text"] = str(query.threshold_date_text)
  elif str(program_mode) == RELATIVE_POSITION_EVENT_LABEL_MODE:
    dynamic_slots["reference_date_text"] = str(query.reference_date_text)
    dynamic_slots["relative_offset_phrase"] = str(query.relative_offset_phrase)

  prompt_selection = render_task_prompt_variants(
    domain=DOMAIN,
    scene_id=SCENE,
    bundle_id=PROMPT_BUNDLE,
    scene_key=PROMPT_SCENE_KEY,
    task_key=PROMPT_TASK_KEY,
    query_key=str(prompt_query_key),
    answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
    dynamic_slots=dynamic_slots,
    instance_seed=int(instance_seed),
  )
  prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

  annotation_bboxes = [
    [round(float(value), 3) for value in rendered_scene.event_bboxes_by_id[str(event_id)]]
    for event_id in query.answer_event_ids
  ]
  if str(program_mode) == RELATIVE_POSITION_EVENT_LABEL_MODE:
    if len(annotation_bboxes) != 1:
      raise ValueError("relative-position timeline task must bind exactly one target event bbox")
    answer_gt = TypedValue(type="string", value=str(query.answer_label))
    annotation_gt = TypedValue(type="bbox", value=list(annotation_bboxes[0]))
  else:
    if query.answer_value is None:
      raise ValueError("timeline count tasks must bind an integer answer")
    answer_gt = TypedValue(type="integer", value=int(query.answer_value))
    annotation_gt = TypedValue(type="bbox_set", value=[list(box) for box in annotation_bboxes])

  event_records = [
    {
      "event_id": str(event.event_id),
      "label": str(event.label),
      "day_of_month": int(event.day_of_month),
      "date_text": str(format_month_day_label(int(query.month), int(event.day_of_month))),
      "order_index": int(event.order_index),
      "reference_kind": str(event.reference_kind),
      "card_side": str(
        _resolve_card_side(
          scene_variant=str(query.scene_variant),
          order_index=int(event.order_index),
        )
      ),
    }
    for event in query.raw_events
  ]
  probabilities = {
    str(key): float(value)
    for key, value in branch_probabilities.items()
  }
  common_params = {
    "query_id": str(selected_branch),
    "source_query_id": str(source_query_name),
    "prompt_query_key": str(prompt_query_key),
    "program_mode": str(program_mode),
    "interval_relation": str(query.interval_relation),
    "scene_variant": str(query.scene_variant),
    "style_variant": str(query.style_variant),
    "accent_color_name": str(query.accent_color_name),
    "information_scene_treatment": str(information_style_meta.get("treatment", "")),
    "information_scene_palette_id": str(information_style_meta.get("palette_id", "")),
    "information_scene_style_pack": str(information_style_meta.get("style_pack", "")),
    "year": int(query.year),
    "month": int(query.month),
    "month_name": str(query.month_name),
    "event_count_support": [int(value) for value in query.event_count_support],
    "between_count_support": [int(value) for value in query.between_count_support],
    "outside_count_support": [int(value) for value in query.outside_count_support],
    "threshold_count_support": [int(value) for value in query.threshold_count_support],
    "relative_offset_support": [int(value) for value in query.relative_offset_support],
    "threshold_day": int(query.threshold_day) if query.threshold_day is not None else None,
    "threshold_date_text": str(query.threshold_date_text),
    "reference_date_text": str(query.reference_date_text),
    "relative_offset": int(query.relative_offset) if query.relative_offset is not None else None,
    "relative_offset_phrase": str(query.relative_offset_phrase),
    "query_id_probabilities": dict(probabilities),
    "interval_relation_probabilities": dict(query.interval_relation_probabilities),
    "scene_variant_probabilities": dict(query.scene_variant_probabilities),
    "style_variant_probabilities": dict(query.style_variant_probabilities),
    "accent_color_name_probabilities": dict(query.accent_color_name_probabilities),
  }
  query_spec = build_prompt_query_spec(
    prompt_artifacts=prompt_artifacts,
    query_id=str(selected_branch),
    params=common_params,
  )
  query_spec["scene_id"] = SCENE

  trace_payload = {
    "scene_ir": {
      "scene_id": SCENE,
      "scene_kind": "pages_milestone_timeline",
      "entities": [dict(entity) for entity in rendered_scene.entities],
      "relations": {
        "query_id": str(selected_branch),
        "source_query_id": str(source_query_name),
        "prompt_query_key": str(prompt_query_key),
        "program_mode": str(program_mode),
        "interval_relation": str(query.interval_relation),
        "scene_variant": str(query.scene_variant),
        "style_variant": str(query.style_variant),
        "accent_color_name": str(query.accent_color_name),
        "year": int(query.year),
        "month": int(query.month),
        "month_name": str(query.month_name),
        "reference_event_ids": [str(value) for value in query.reference_event_ids],
        "prompt_reference_event_ids": [str(value) for value in query.prompt_reference_event_ids],
        "threshold_day": int(query.threshold_day) if query.threshold_day is not None else None,
        "threshold_date_text": str(query.threshold_date_text),
        "reference_date_text": str(query.reference_date_text),
        "relative_offset": int(query.relative_offset) if query.relative_offset is not None else None,
      },
    },
    "query_spec": query_spec,
    "render_spec": {
      "canvas_width": int(render_params.canvas_width),
      "canvas_height": int(render_params.canvas_height),
      "coord_space": "pixel",
      "scene_id": SCENE,
      "scene_variant": str(query.scene_variant),
      "background_style": dict(background_meta),
      "information_scene_style": dict(information_style_meta),
      "post_image_noise": dict(post_noise_meta),
      "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
      "context_text_policy": {
        "container_bboxes_are_background": True,
      },
      "timeline_style": {
        "accent_color_name": str(query.accent_color_name),
        "style_variant": str(query.style_variant),
        "information_scene_treatment": str(information_style_meta.get("treatment", "")),
        "information_scene_palette_id": str(information_style_meta.get("palette_id", "")),
        "information_scene_style_pack": str(information_style_meta.get("style_pack", "")),
        "title_text": str(query.title_text),
        "subtitle_text": str(query.subtitle_text),
        "resolved_colors_rgb": {
          "panel_fill": [int(value) for value in timeline_theme.panel_fill_rgb],
          "panel_outline": [int(value) for value in timeline_theme.panel_outline_rgb],
          "axis_line": [int(value) for value in timeline_theme.axis_line_rgb],
          "event_fill": [int(value) for value in timeline_theme.event_fill_rgb],
          "event_outline": [int(value) for value in timeline_theme.event_outline_rgb],
          "primary_reference_fill": [int(value) for value in timeline_theme.primary_reference_fill_rgb],
          "secondary_reference_fill": [int(value) for value in timeline_theme.secondary_reference_fill_rgb],
        },
      },
    },
    "render_map": {
      "image_id": "img0",
      "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
      "panel_bbox_px": [round(float(value), 3) for value in rendered_scene.panel_bbox_px],
      "axis_bbox_px": [round(float(value), 3) for value in rendered_scene.axis_bbox_px],
      "title_text": str(rendered_scene.title_text),
      "subtitle_text": str(rendered_scene.subtitle_text),
      "event_bboxes_by_id": {
        str(event_id): [round(float(value), 3) for value in bbox]
        for event_id, bbox in rendered_scene.event_bboxes_by_id.items()
      },
      "reference_event_ids": [str(value) for value in query.reference_event_ids],
      "prompt_reference_event_ids": [str(value) for value in query.prompt_reference_event_ids],
      "answer_event_ids": [str(value) for value in query.answer_event_ids],
      "endpoint_event_ids": [str(value) for value in query.endpoint_event_ids],
      "threshold_day": int(query.threshold_day) if query.threshold_day is not None else None,
      "threshold_date_text": str(query.threshold_date_text),
      "reference_date_text": str(query.reference_date_text),
      "relative_offset": int(query.relative_offset) if query.relative_offset is not None else None,
    },
    "execution_trace": {
      **dict(common_params),
      "event_count": len(query.raw_events),
      "answer_value": int(query.answer_value) if query.answer_value is not None else None,
      "answer_label": str(query.answer_label),
      "answer_event_ids": [str(value) for value in query.answer_event_ids],
      "reference_event_ids": [str(value) for value in query.reference_event_ids],
      "prompt_reference_event_ids": [str(value) for value in query.prompt_reference_event_ids],
      "endpoint_event_ids": [str(value) for value in query.endpoint_event_ids],
      "prompt_endpoint_event_ids": [str(value) for value in query.prompt_endpoint_event_ids],
      "threshold_day": int(query.threshold_day) if query.threshold_day is not None else None,
      "threshold_date_text": str(query.threshold_date_text),
      "reference_date_text": str(query.reference_date_text),
      "relative_offset": int(query.relative_offset) if query.relative_offset is not None else None,
      "events": event_records,
    },
    "witness_symbolic": {
      "type": str(annotation_gt.type),
      "value": annotation_gt.value,
    },
    "projected_annotation": {
      str(annotation_gt.type): annotation_gt.value,
    },
  }

  return TaskOutput(
    prompt=str(prompt_artifacts.prompt),
    answer_gt=answer_gt,
    annotation_gt=annotation_gt,
    image=image,
    image_id="img0",
    trace_payload=trace_payload,
    task_versions=default_task_versions(),
    query_id=str(selected_branch),
    prompt_variants=dict(prompt_artifacts.prompt_variants),
  )


__all__ = [
  "DOMAIN",
  "INTERVAL_EVENT_MODE",
  "RELATIVE_POSITION_EVENT_LABEL_MODE",
  "SCENE",
  "THRESHOLD_EVENT_COUNT_MODE",
  "build_timeline_response",
]
