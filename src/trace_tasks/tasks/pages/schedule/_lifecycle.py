"""Single-day schedule page task with overlap, duration, and optimization queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ....core.visual.background import make_background_canvas
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.config_defaults import group_default, split_generation_rendering_prompt_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
  PROMPT_OUTPUT_MODES,
  build_prompt_query_spec,
  build_prompt_trace_artifacts,
  render_task_prompt_variants,
)
from ...shared.time_artifact_style import (
  SUPPORTED_TIME_ARTIFACT_COLOR_NAMES,
  SUPPORTED_TIME_ARTIFACT_STYLE_VARIANTS,
  build_time_artifact_schedule_theme,
)
from ...shared.time_artifact_task_support import resolve_time_artifact_named_variant, resolve_time_artifact_selection_index
from ...shared.time_format import format_day_time_hhmm
from ..shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults
from .shared.rendering import (
  SUPPORTED_PAGE_SCHEDULE_SCENE_VARIANTS,
  RenderedScheduleScene,
  ScheduledEventSpec,
  render_day_schedule_scene,
  resolve_schedule_render_params,
)


DOMAIN = "pages"
SCENE = "schedule"
TASK_NAMESPACE = "pages.schedule"
PROMPT_BUNDLE = "pages_schedule_v1"
PROMPT_SCENE_KEY = "day_schedule"
PROMPT_TASK_KEY = "schedule_day_query"
OVERLAP_COUNT_MODE = "events_overlapping_reference_interval"
LONGER_THAN_REFERENCE_MODE = "events_longer_than_reference"
MAXIMUM_NON_OVERLAPPING_MODE = "maximum_non_overlapping_event_set"
SUPPORTED_PROGRAM_MODES: Tuple[str, ...] = (
  OVERLAP_COUNT_MODE,
  LONGER_THAN_REFERENCE_MODE,
  MAXIMUM_NON_OVERLAPPING_MODE,
)


@dataclass(frozen=True)
class _TaskDefaults:
  """Stable fallback defaults for single-day planner scenes."""

  start_hour: int = 8
  end_hour: int = 18
  slot_minutes: int = 30
  max_lane_count: int = 5
  overlap_max_lane_count: int = 4
  overlap_nonoverlap_min_gap_slots: int = 1
  show_reference_time_band: bool = False
  event_count_support: Tuple[int, ...] = (7, 8, 9, 10)
  overlap_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
  longer_than_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
  maximum_non_overlapping_support: Tuple[int, ...] = (2, 3, 4, 5)
  reference_duration_slots_support: Tuple[int, ...] = (2, 3, 4)
  duration_slots_support: Tuple[int, ...] = (1, 2, 3, 4, 5)
  day_label_support: Tuple[str, ...] = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
  event_label_pool: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M")
  canvas_width: int = 920
  canvas_height: int = 820
  outer_margin_px: int = 36
  header_height_px: int = 92
  panel_corner_radius_px: int = 20
  panel_outline_width_px: int = 3
  time_axis_width_px: int = 94
  planner_top_gap_px: int = 18
  planner_bottom_gap_px: int = 24
  lane_gap_px: int = 8
  grid_line_width_px: int = 2
  minor_grid_line_width_px: int = 1
  hour_label_font_size_px: int = 16
  title_font_size_px: int = 28
  event_label_font_size_px: int = 18
  event_corner_radius_px: int = 12
  event_text_padding_px: int = 8


@dataclass(frozen=True)
class _RawEvent:
  """Task-internal event interval before lane assignment and rendering."""

  event_id: str
  label: str
  start_slot: int
  end_slot: int
  is_reference: bool = False

  @property
  def duration_slots(self) -> int:
    """Return the interval duration in discrete schedule slots."""

    return int(self.end_slot) - int(self.start_slot)


@dataclass(frozen=True)
class _ResolvedQuery:
  """Resolved semantic and visual support for one single-day schedule query."""

  program_mode: str
  scene_variant: str
  style_variant: str
  accent_color_name: str
  day_label: str
  title_text: str
  start_hour: int
  end_hour: int
  slot_minutes: int
  max_lane_count: int
  overlap_max_lane_count: int
  overlap_nonoverlap_min_gap_slots: int
  show_reference_time_band: bool
  event_count_support: Tuple[int, ...]
  overlap_count_support: Tuple[int, ...]
  longer_than_count_support: Tuple[int, ...]
  maximum_non_overlapping_support: Tuple[int, ...]
  raw_events: Tuple[_RawEvent, ...]
  rendered_events: Tuple[ScheduledEventSpec, ...]
  event_count: int
  lane_count: int
  answer_value: int
  answer_event_ids: Tuple[str, ...]
  reference_event_id: str | None
  branch_probabilities: Dict[str, float]
  scene_variant_probabilities: Dict[str, float]
  style_variant_probabilities: Dict[str, float]
  accent_color_name_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults(DOMAIN, SCENE)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = split_generation_rendering_prompt_defaults(
  _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id=SCENE)
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id=SCENE, apply_prob=0.0)


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
  """Resolve one balanced named schedule-task axis."""

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
    value = str(raw_value)
    if value and value not in resolved:
      resolved.append(value)
  if not resolved:
    raise ValueError(f"{key} must not be empty for {TASK_NAMESPACE}")
  return tuple(str(value) for value in resolved)


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


def _intervals_overlap(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
  """Return whether two half-open slot intervals overlap."""

  return bool(int(left[0]) < int(right[1]) and int(right[0]) < int(left[1]))


def _interval_gap_slots(left: Tuple[int, int], right: Tuple[int, int]) -> int:
  """Return the non-overlap gap between two intervals, or zero when they overlap/touch."""

  if _intervals_overlap(left, right):
    return 0
  if int(left[1]) <= int(right[0]):
    return max(0, int(right[0]) - int(left[1]))
  return max(0, int(left[0]) - int(right[1]))


def _slots_to_total_minutes(slot_index: int, *, start_hour: int, slot_minutes: int) -> int:
  """Convert one schedule slot index into absolute day minutes."""

  return int((int(start_hour) * 60) + (int(slot_index) * int(slot_minutes)))


def _build_raw_event(label: str, start_slot: int, end_slot: int, *, is_reference: bool = False) -> _RawEvent:
  """Build one task-internal raw event record."""

  return _RawEvent(
    event_id=f"event_{str(label).lower()}",
    label=str(label),
    start_slot=int(start_slot),
    end_slot=int(end_slot),
    is_reference=bool(is_reference),
  )


def _assign_lanes(raw_events: Sequence[_RawEvent]) -> Tuple[Tuple[ScheduledEventSpec, ...], int]:
  """Assign non-overlapping lane indices to one set of schedule events."""

  lane_end_slots: List[int] = []
  scheduled: List[ScheduledEventSpec] = []
  for raw_event in sorted(raw_events, key=lambda event: (int(event.start_slot), int(event.end_slot), str(event.label))):
    lane_index: int | None = None
    for candidate_lane, lane_end in enumerate(lane_end_slots):
      if int(lane_end) <= int(raw_event.start_slot):
        lane_index = int(candidate_lane)
        lane_end_slots[int(candidate_lane)] = int(raw_event.end_slot)
        break
    if lane_index is None:
      lane_index = len(lane_end_slots)
      lane_end_slots.append(int(raw_event.end_slot))
    scheduled.append(
      ScheduledEventSpec(
        event_id=str(raw_event.event_id),
        label=str(raw_event.label),
        start_total_minutes=0,
        end_total_minutes=0,
        lane_index=int(lane_index),
        is_reference=bool(raw_event.is_reference),
      )
    )
  lane_count = max(1, len(lane_end_slots))
  return tuple(scheduled), int(lane_count)


def _maximum_non_overlapping_subsets(raw_events: Sequence[_RawEvent]) -> Tuple[int, Tuple[Tuple[str, ...], ...]]:
  """Return the maximum compatible subset size and all optimum event-id subsets."""

  event_ids = tuple(str(event.event_id) for event in raw_events)
  intervals = {
    str(event.event_id): (int(event.start_slot), int(event.end_slot))
    for event in raw_events
  }
  best_size = 0
  best_subsets: set[Tuple[str, ...]] = set()
  for subset_size in range(1, len(event_ids) + 1):
    for subset in combinations(event_ids, int(subset_size)):
      if any(
        _intervals_overlap(intervals[str(left)], intervals[str(right)])
        for left, right in combinations(subset, 2)
      ):
        continue
      sorted_subset = tuple(sorted(str(event_id) for event_id in subset))
      if int(subset_size) > int(best_size):
        best_size = int(subset_size)
        best_subsets = {sorted_subset}
      elif int(subset_size) == int(best_size):
        best_subsets.add(sorted_subset)
  return int(best_size), tuple(sorted(best_subsets))


def _assign_lanes_spreading_answer(
  raw_events: Sequence[_RawEvent],
  *,
  answer_event_ids: Sequence[str],
  max_lane_count: int,
  rng,
) -> Tuple[Tuple[ScheduledEventSpec, ...], int] | None:
  """Assign lanes while preventing the answer set from becoming one visual column."""

  answer_set = {str(event_id) for event_id in answer_event_ids}
  if len(answer_set) <= 1 or len(answer_set) > int(max_lane_count):
    return None

  event_by_id = {str(event.event_id): event for event in raw_events}
  answer_events = [
    event_by_id[str(event_id)]
    for event_id in answer_event_ids
    if str(event_id) in event_by_id
  ]
  if len(answer_events) != len(answer_set):
    return None
  answer_events = sorted(
    answer_events,
    key=lambda event: (int(event.start_slot), int(event.end_slot), str(event.label)),
  )

  target_answer_lane_count = min(
    int(max_lane_count),
    len(answer_events),
    2 + (int(rng.randrange(2)) if len(answer_events) >= 4 else 0),
  )
  target_answer_lane_count = max(2, int(target_answer_lane_count))
  lanes: List[List[_RawEvent]] = [[] for _ in range(int(max_lane_count))]
  lane_index_by_event_id: Dict[str, int] = {}
  answer_lane_indices = list(range(int(target_answer_lane_count)))
  rng.shuffle(answer_lane_indices)
  for index, event in enumerate(answer_events):
    lane_index = int(answer_lane_indices[int(index) % len(answer_lane_indices)])
    lanes[lane_index].append(event)
    lane_index_by_event_id[str(event.event_id)] = int(lane_index)

  distractors = [
    event
    for event in raw_events
    if str(event.event_id) not in answer_set
  ]
  distractors.sort(
    key=lambda event: (
      -int(event.duration_slots),
      int(event.start_slot),
      int(event.end_slot),
      str(event.label),
    )
  )
  for event in distractors:
    feasible_lane_indices = [
      lane_index
      for lane_index, lane_events in enumerate(lanes)
      if all(
        not _intervals_overlap(
          (int(event.start_slot), int(event.end_slot)),
          (int(other.start_slot), int(other.end_slot)),
        )
        for other in lane_events
      )
    ]
    if not feasible_lane_indices:
      return None
    min_lane_size = min(len(lanes[int(lane_index)]) for lane_index in feasible_lane_indices)
    smallest_lanes = [
      int(lane_index)
      for lane_index in feasible_lane_indices
      if len(lanes[int(lane_index)]) == int(min_lane_size)
    ]
    lane_index = int(smallest_lanes[int(rng.randrange(len(smallest_lanes)))])
    lanes[lane_index].append(event)
    lane_index_by_event_id[str(event.event_id)] = int(lane_index)

  used_lane_indices = [
    lane_index
    for lane_index, lane_events in enumerate(lanes)
    if lane_events
  ]
  if not used_lane_indices:
    return None
  lane_remap = {
    int(lane_index): int(new_index)
    for new_index, lane_index in enumerate(sorted(used_lane_indices))
  }
  answer_lanes = {
    int(lane_remap[lane_index_by_event_id[str(event_id)]])
    for event_id in answer_set
  }
  if len(answer_lanes) <= 1:
    return None

  lane_sets: Dict[int, set[str]] = {}
  for event_id, old_lane_index in lane_index_by_event_id.items():
    lane_sets.setdefault(int(lane_remap[int(old_lane_index)]), set()).add(str(event_id))
  if any(set(event_ids) == set(answer_set) for event_ids in lane_sets.values()):
    return None

  scheduled = [
    ScheduledEventSpec(
      event_id=str(event.event_id),
      label=str(event.label),
      start_total_minutes=0,
      end_total_minutes=0,
      lane_index=int(lane_remap[lane_index_by_event_id[str(event.event_id)]]),
      is_reference=bool(event.is_reference),
    )
    for event in sorted(
      raw_events,
      key=lambda item: (
        int(item.start_slot),
        int(item.end_slot),
        int(lane_remap[lane_index_by_event_id[str(item.event_id)]]),
        str(item.label),
      ),
    )
  ]
  return tuple(scheduled), int(len(used_lane_indices))


def _hydrate_event_minutes(
  scheduled_events: Sequence[ScheduledEventSpec],
  raw_events_by_id: Mapping[str, _RawEvent],
  *,
  start_hour: int,
  slot_minutes: int,
) -> Tuple[ScheduledEventSpec, ...]:
  """Fill rendered schedule events with day-minute coordinates."""

  hydrated: List[ScheduledEventSpec] = []
  for scheduled_event in scheduled_events:
    raw_event = raw_events_by_id[str(scheduled_event.event_id)]
    hydrated.append(
      ScheduledEventSpec(
        event_id=str(raw_event.event_id),
        label=str(raw_event.label),
        start_total_minutes=_slots_to_total_minutes(
          int(raw_event.start_slot),
          start_hour=int(start_hour),
          slot_minutes=int(slot_minutes),
        ),
        end_total_minutes=_slots_to_total_minutes(
          int(raw_event.end_slot),
          start_hour=int(start_hour),
          slot_minutes=int(slot_minutes),
        ),
        lane_index=int(scheduled_event.lane_index),
        is_reference=bool(raw_event.is_reference),
      )
    )
  return tuple(hydrated)


def _random_interval(
  rng,
  *,
  total_slots: int,
  duration_support: Sequence[int],
) -> Tuple[int, int]:
  """Sample one random slot interval from the provided duration support."""

  duration = int(rng.choice([int(value) for value in duration_support]))
  duration = max(1, min(int(duration), int(total_slots)))
  start_slot = int(rng.randint(0, int(total_slots - duration)))
  return int(start_slot), int(start_slot + duration)


def _sample_overlap_variant(
  *,
  instance_seed: int,
  total_slots: int,
  params: Mapping[str, Any],
  event_labels: Sequence[str],
  event_count_support: Sequence[int],
  overlap_count_support: Sequence[int],
  reference_duration_slots_support: Sequence[int],
  duration_slots_support: Sequence[int],
  max_lane_count: int,
  nonoverlap_min_gap_slots: int,
) -> Tuple[Tuple[_RawEvent, ...], Tuple[str, ...], str]:
  """Sample one overlap-count schedule with a highlighted reference event."""

  rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.overlap_mode")
  feasible_event_counts = [int(value) for value in event_count_support if 2 <= int(value) <= len(event_labels)]
  if not feasible_event_counts:
    raise ValueError("no feasible event_count_support exists for overlap schedule mode")
  event_count = int(
    feasible_event_counts[
      int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:overlap_event_count") % len(feasible_event_counts))
    ]
  )
  feasible_answer_support = [int(value) for value in overlap_count_support if 0 <= int(value) <= int(event_count - 1)]
  if not feasible_answer_support:
    raise ValueError("no feasible overlap_count_support exists for overlap schedule mode")
  total_support = [int(value) for value in duration_slots_support if 1 <= int(value) < int(total_slots)]
  if not total_support:
    raise ValueError("duration_slots_support must contain values inside the schedule horizon")

  for _ in range(600):
    answer_count = int(
      feasible_answer_support[
        int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:overlap_answer_count") % len(feasible_answer_support))
      ]
    )
    reference_support = [int(value) for value in reference_duration_slots_support if 1 <= int(value) < int(total_slots)]
    ref_duration = int(
      reference_support[
        int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:overlap_reference_duration") % len(reference_support))
      ]
    )
    ref_start = int(rng.randint(0, int(total_slots - ref_duration)))
    ref_interval = (int(ref_start), int(ref_start + ref_duration))

    used_intervals = {tuple(ref_interval)}
    overlap_intervals: List[Tuple[int, int]] = []
    distractor_intervals: List[Tuple[int, int]] = []
    min_gap_slots = max(0, int(nonoverlap_min_gap_slots))

    inner_guard = 0
    while len(overlap_intervals) < int(answer_count) and inner_guard < 600:
      inner_guard += 1
      candidate = _random_interval(rng, total_slots=int(total_slots), duration_support=total_support)
      if candidate in used_intervals or candidate == tuple(ref_interval):
        continue
      if not _intervals_overlap(candidate, ref_interval):
        continue
      overlap_intervals.append(tuple(candidate))
      used_intervals.add(tuple(candidate))

    inner_guard = 0
    while len(distractor_intervals) < int(event_count - 1 - answer_count) and inner_guard < 800:
      inner_guard += 1
      if float(rng.random()) < 0.35:
        if float(rng.random()) < 0.5 and int(ref_start) > int(min_gap_slots):
          end_slot = int(ref_start) - int(min_gap_slots)
          duration = int(rng.choice(total_support))
          start_slot = max(0, int(end_slot - duration))
          candidate = (int(start_slot), int(end_slot))
        else:
          start_slot = int(ref_interval[1]) + int(min_gap_slots)
          if int(start_slot) >= int(total_slots):
            continue
          duration = int(rng.choice(total_support))
          candidate = (int(start_slot), min(int(total_slots), int(start_slot + duration)))
        if int(candidate[1]) <= int(candidate[0]):
          continue
      else:
        candidate = _random_interval(rng, total_slots=int(total_slots), duration_support=total_support)
      if candidate in used_intervals or candidate == tuple(ref_interval):
        continue
      if _intervals_overlap(candidate, ref_interval):
        continue
      if _interval_gap_slots(candidate, ref_interval) < int(min_gap_slots):
        continue
      distractor_intervals.append(tuple(candidate))
      used_intervals.add(tuple(candidate))

    if len(overlap_intervals) != int(answer_count) or len(distractor_intervals) != int(event_count - 1 - answer_count):
      continue

    raw_events = [
      _build_raw_event(str(event_labels[0]), int(ref_interval[0]), int(ref_interval[1]), is_reference=True),
    ]
    raw_events.extend(
      _build_raw_event(str(event_labels[index + 1]), int(interval[0]), int(interval[1]))
      for index, interval in enumerate(overlap_intervals)
    )
    raw_events.extend(
      _build_raw_event(str(event_labels[1 + len(overlap_intervals) + index]), int(interval[0]), int(interval[1]))
      for index, interval in enumerate(distractor_intervals)
    )
    _, lane_count = _assign_lanes(raw_events)
    if int(lane_count) > int(max_lane_count):
      continue
    answer_event_ids = tuple(event.event_id for event in raw_events if not event.is_reference and _intervals_overlap((event.start_slot, event.end_slot), ref_interval))
    return tuple(raw_events), tuple(answer_event_ids), str(raw_events[0].event_id)

  raise ValueError("failed to sample a readable overlap-count schedule")


def _sample_longer_variant(
  *,
  instance_seed: int,
  total_slots: int,
  params: Mapping[str, Any],
  event_labels: Sequence[str],
  event_count_support: Sequence[int],
  longer_than_count_support: Sequence[int],
  reference_duration_slots_support: Sequence[int],
  duration_slots_support: Sequence[int],
  max_lane_count: int,
) -> Tuple[Tuple[_RawEvent, ...], Tuple[str, ...], str]:
  """Sample one duration-comparison schedule with a highlighted reference event."""

  rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.duration_comparison_mode")
  feasible_event_counts = [int(value) for value in event_count_support if 2 <= int(value) <= len(event_labels)]
  if not feasible_event_counts:
    raise ValueError("no feasible event_count_support exists for duration comparison schedule mode")
  durations = [int(value) for value in duration_slots_support if 1 <= int(value) < int(total_slots)]
  if not durations:
    raise ValueError("duration_slots_support must contain values inside the schedule horizon")

  for _ in range(600):
    event_count = int(
      feasible_event_counts[
        int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:longer_event_count") % len(feasible_event_counts))
      ]
    )
    reference_support = [int(value) for value in reference_duration_slots_support if 1 <= int(value) < int(total_slots)]
    ref_duration = int(
      reference_support[
        int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:longer_reference_duration") % len(reference_support))
      ]
    )
    reference_interval = _random_interval(rng, total_slots=int(total_slots), duration_support=(int(ref_duration),))
    feasible_answer_support = [int(value) for value in longer_than_count_support if 0 <= int(value) <= int(event_count - 1)]
    if not feasible_answer_support:
      raise ValueError("no feasible longer_than_count_support exists for duration comparison schedule mode")
    answer_count = int(
      feasible_answer_support[
        int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:longer_answer_count") % len(feasible_answer_support))
      ]
    )

    longer_durations = [int(value) for value in durations if int(value) > int(ref_duration)]
    not_longer_durations = [int(value) for value in durations if int(value) <= int(ref_duration)]
    if answer_count > 0 and not longer_durations:
      continue
    if int(event_count - 1 - answer_count) > 0 and not not_longer_durations:
      continue

    used_intervals = {tuple(reference_interval)}
    longer_intervals: List[Tuple[int, int]] = []
    other_intervals: List[Tuple[int, int]] = []

    inner_guard = 0
    while len(longer_intervals) < int(answer_count) and inner_guard < 600:
      inner_guard += 1
      candidate = _random_interval(rng, total_slots=int(total_slots), duration_support=tuple(longer_durations))
      if candidate in used_intervals or candidate == tuple(reference_interval):
        continue
      longer_intervals.append(tuple(candidate))
      used_intervals.add(tuple(candidate))

    inner_guard = 0
    while len(other_intervals) < int(event_count - 1 - answer_count) and inner_guard < 800:
      inner_guard += 1
      candidate = _random_interval(rng, total_slots=int(total_slots), duration_support=tuple(not_longer_durations))
      if candidate in used_intervals or candidate == tuple(reference_interval):
        continue
      other_intervals.append(tuple(candidate))
      used_intervals.add(tuple(candidate))

    if len(longer_intervals) != int(answer_count) or len(other_intervals) != int(event_count - 1 - answer_count):
      continue

    raw_events = [
      _build_raw_event(str(event_labels[0]), int(reference_interval[0]), int(reference_interval[1]), is_reference=True),
    ]
    raw_events.extend(
      _build_raw_event(str(event_labels[index + 1]), int(interval[0]), int(interval[1]))
      for index, interval in enumerate(longer_intervals)
    )
    raw_events.extend(
      _build_raw_event(str(event_labels[1 + len(longer_intervals) + index]), int(interval[0]), int(interval[1]))
      for index, interval in enumerate(other_intervals)
    )
    _, lane_count = _assign_lanes(raw_events)
    if int(lane_count) > int(max_lane_count):
      continue
    answer_event_ids = tuple(
      event.event_id
      for event in raw_events
      if not event.is_reference and int(event.duration_slots) > int(ref_duration)
    )
    return tuple(raw_events), tuple(answer_event_ids), str(raw_events[0].event_id)

  raise ValueError("failed to sample a readable longer-than-reference schedule")


def _sample_unique_optimal_variant(
  *,
  instance_seed: int,
  total_slots: int,
  params: Mapping[str, Any],
  event_labels: Sequence[str],
  maximum_non_overlapping_support: Sequence[int],
  max_lane_count: int,
) -> Tuple[Tuple[_RawEvent, ...], Tuple[str, ...], Tuple[ScheduledEventSpec, ...], int]:
  """Sample one schedule with a unique maximum-size non-overlapping subset."""

  rng = spawn_rng(int(instance_seed), f"{TASK_NAMESPACE}.maximum_compatible_set_mode")
  feasible_support = [
    int(value)
    for value in maximum_non_overlapping_support
    if 2 <= int(value) <= 5 and int(value) < len(event_labels)
  ]
  if not feasible_support:
    raise ValueError("maximum_non_overlapping_support must contain feasible values in 2..5 whose witness set fits the label pool")
  answer_value = int(
    feasible_support[
      int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:maximum_compatible_set") % len(feasible_support))
    ]
  )

  for _ in range(1400):
    answer_durations = [int(rng.choice((1, 1, 2, 2, 3))) for _ in range(int(answer_value))]
    base_gap_total = int(answer_value - 1)
    slack = int(total_slots) - int(sum(answer_durations)) - int(base_gap_total)
    if int(slack) < 0:
      continue

    gaps = [0] + [1 for _ in range(int(answer_value - 1))] + [0]
    for _slack_index in range(int(slack)):
      gaps[int(rng.randrange(len(gaps)))] += 1

    answer_intervals: List[Tuple[int, int]] = []
    cursor = int(gaps[0])
    for index, duration in enumerate(answer_durations):
      answer_intervals.append((int(cursor), int(cursor + duration)))
      cursor += int(duration)
      if index < int(answer_value - 1):
        cursor += int(gaps[int(index + 1)])

    distractor_intervals: List[Tuple[int, int]] = []
    seen_intervals = set(tuple(interval) for interval in answer_intervals)

    def _add_distractor_interval(interval: Tuple[int, int]) -> None:
      start, end = int(interval[0]), int(interval[1])
      if not (0 <= int(start) < int(end) <= int(total_slots)):
        return
      candidate = (int(start), int(end))
      if candidate in seen_intervals:
        return
      seen_intervals.add(candidate)
      distractor_intervals.append(candidate)

    for boundary_index in range(int(answer_value - 1)):
      left = answer_intervals[int(boundary_index)]
      right = answer_intervals[int(boundary_index + 1)]
      _add_distractor_interval(
        (
          int(rng.randint(int(left[0]), int(left[1] - 1))),
          int(rng.randint(int(right[0] + 1), int(right[1]))),
        )
      )

    max_distractors = max(0, int(len(event_labels)) - int(answer_value))
    target_distractor_count = min(
      int(max_distractors),
      int(answer_value - 1) + int(rng.randint(2, 4)),
    )
    inner_guard = 0
    while len(distractor_intervals) < int(target_distractor_count) and inner_guard < 500:
      inner_guard += 1
      span_width = int(rng.randint(2, int(answer_value)))
      left_index = int(rng.randint(0, int(answer_value - span_width)))
      right_index = int(left_index + span_width - 1)
      left = answer_intervals[int(left_index)]
      right = answer_intervals[int(right_index)]
      _add_distractor_interval(
        (
          int(rng.randint(int(left[0]), int(left[1] - 1))),
          int(rng.randint(int(right[0] + 1), int(right[1]))),
        )
      )

    if len(distractor_intervals) < int(answer_value):
      continue

    labels = [str(label) for label in event_labels]
    rng.shuffle(labels)
    needed_labels = int(answer_value) + int(len(distractor_intervals))
    if int(needed_labels) > len(labels):
      continue

    raw_events: List[_RawEvent] = []
    answer_event_ids: List[str] = []
    for index, interval in enumerate(answer_intervals):
      event = _build_raw_event(str(labels[int(index)]), int(interval[0]), int(interval[1]))
      raw_events.append(event)
      answer_event_ids.append(str(event.event_id))
    label_offset = int(answer_value)
    for index, interval in enumerate(distractor_intervals):
      raw_events.append(
        _build_raw_event(
          str(labels[int(label_offset + index)]),
          int(interval[0]),
          int(interval[1]),
        )
      )

    best_size, best_subsets = _maximum_non_overlapping_subsets(raw_events)
    expected_subset = tuple(sorted(str(event_id) for event_id in answer_event_ids))
    if int(best_size) != int(answer_value) or tuple(best_subsets) != (expected_subset,):
      continue

    assigned = _assign_lanes_spreading_answer(
      tuple(raw_events),
      answer_event_ids=tuple(answer_event_ids),
      max_lane_count=int(max_lane_count),
      rng=rng,
    )
    if assigned is None:
      continue
    scheduled_events, lane_count = assigned
    return tuple(raw_events), tuple(answer_event_ids), tuple(scheduled_events), int(lane_count)

  raise ValueError("failed to sample a unique-optimal non-overlapping schedule")


def _resolve_query(
  instance_seed: int,
  *,
  params: Mapping[str, Any],
  program_mode: str,
  branch_probabilities: Mapping[str, float],
) -> _ResolvedQuery:
  """Resolve one concrete single-day planner query from balanced supports."""

  resolved_program_mode = str(program_mode)
  if resolved_program_mode not in set(SUPPORTED_PROGRAM_MODES):
    raise ValueError(f"unsupported program_mode for {TASK_NAMESPACE}: {resolved_program_mode}")
  resolved_branch_probabilities = {str(key): float(value) for key, value in dict(branch_probabilities).items()}
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
    supported=SUPPORTED_PAGE_SCHEDULE_SCENE_VARIANTS,
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

  start_hour = int(params.get("start_hour", group_default(_GEN_DEFAULTS, "start_hour", _DEFAULTS.start_hour)))
  end_hour = int(params.get("end_hour", group_default(_GEN_DEFAULTS, "end_hour", _DEFAULTS.end_hour)))
  slot_minutes = int(params.get("slot_minutes", group_default(_GEN_DEFAULTS, "slot_minutes", _DEFAULTS.slot_minutes)))
  if int(end_hour) <= int(start_hour):
    raise ValueError("end_hour must be greater than start_hour for schedule tasks")
  if int(slot_minutes) <= 0 or (60 % int(slot_minutes)) != 0:
    raise ValueError("slot_minutes must be a positive divisor of 60 for schedule tasks")
  total_slots = int(((int(end_hour) - int(start_hour)) * 60) // int(slot_minutes))
  max_lane_count = int(params.get("max_lane_count", group_default(_GEN_DEFAULTS, "max_lane_count", _DEFAULTS.max_lane_count)))
  if int(max_lane_count) <= 0:
    raise ValueError("max_lane_count must be positive for schedule tasks")
  overlap_max_lane_count = int(
    params.get(
      "overlap_max_lane_count",
      group_default(_GEN_DEFAULTS, "overlap_max_lane_count", _DEFAULTS.overlap_max_lane_count),
    )
  )
  if int(overlap_max_lane_count) <= 0:
    raise ValueError("overlap_max_lane_count must be positive for schedule tasks")
  overlap_nonoverlap_min_gap_slots = int(
    params.get(
      "overlap_nonoverlap_min_gap_slots",
      group_default(
        _GEN_DEFAULTS,
        "overlap_nonoverlap_min_gap_slots",
        _DEFAULTS.overlap_nonoverlap_min_gap_slots,
      ),
    )
  )
  if int(overlap_nonoverlap_min_gap_slots) < 0:
    raise ValueError("overlap_nonoverlap_min_gap_slots must be non-negative for schedule tasks")
  show_reference_time_band = bool(
    params.get(
      "show_reference_time_band",
      group_default(_RENDER_DEFAULTS, "show_reference_time_band", _DEFAULTS.show_reference_time_band),
    )
  )

  event_count_support = _resolve_int_support(params, "event_count_support", _DEFAULTS.event_count_support)
  overlap_count_support = _resolve_int_support(params, "overlap_count_support", _DEFAULTS.overlap_count_support)
  longer_than_count_support = _resolve_int_support(params, "longer_than_count_support", _DEFAULTS.longer_than_count_support)
  maximum_non_overlapping_support = _resolve_int_support(
    params,
    "maximum_non_overlapping_support",
    _DEFAULTS.maximum_non_overlapping_support,
  )
  reference_duration_slots_support = _resolve_int_support(
    params,
    "reference_duration_slots_support",
    _DEFAULTS.reference_duration_slots_support,
  )
  duration_slots_support = _resolve_int_support(params, "duration_slots_support", _DEFAULTS.duration_slots_support)
  day_label_support = _resolve_str_support(params, "day_label_support", _DEFAULTS.day_label_support)
  event_label_pool = _resolve_str_support(params, "event_label_pool", _DEFAULTS.event_label_pool)
  day_label = str(
    day_label_support[
      int(_resolve_support_selection_index(params=params, instance_seed=int(instance_seed), namespace=f"{TASK_NAMESPACE}:day_label") % len(day_label_support))
    ]
  )

  if str(resolved_program_mode) == OVERLAP_COUNT_MODE:
    raw_events, answer_event_ids, reference_event_id = _sample_overlap_variant(
      instance_seed=int(instance_seed),
      total_slots=int(total_slots),
      params=params,
      event_labels=tuple(event_label_pool),
      event_count_support=tuple(event_count_support),
      overlap_count_support=tuple(overlap_count_support),
      reference_duration_slots_support=tuple(reference_duration_slots_support),
      duration_slots_support=tuple(duration_slots_support),
      max_lane_count=min(int(max_lane_count), int(overlap_max_lane_count)),
      nonoverlap_min_gap_slots=int(overlap_nonoverlap_min_gap_slots),
    )
  elif str(resolved_program_mode) == LONGER_THAN_REFERENCE_MODE:
    raw_events, answer_event_ids, reference_event_id = _sample_longer_variant(
      instance_seed=int(instance_seed),
      total_slots=int(total_slots),
      params=params,
      event_labels=tuple(event_label_pool),
      event_count_support=tuple(event_count_support),
      longer_than_count_support=tuple(longer_than_count_support),
      reference_duration_slots_support=tuple(reference_duration_slots_support),
      duration_slots_support=tuple(duration_slots_support),
      max_lane_count=int(max_lane_count),
    )
  else:
    raw_events, answer_event_ids, scheduled_placeholder, lane_count = _sample_unique_optimal_variant(
      instance_seed=int(instance_seed),
      total_slots=int(total_slots),
      params=params,
      event_labels=tuple(event_label_pool),
      maximum_non_overlapping_support=tuple(maximum_non_overlapping_support),
      max_lane_count=int(max_lane_count),
    )
    reference_event_id = None

  if str(resolved_program_mode) != MAXIMUM_NON_OVERLAPPING_MODE:
    scheduled_placeholder, lane_count = _assign_lanes(raw_events)
    if int(lane_count) > int(max_lane_count):
      raise ValueError("sampled schedule exceeds configured max_lane_count")
  rendered_events = _hydrate_event_minutes(
    scheduled_placeholder,
    {str(event.event_id): event for event in raw_events},
    start_hour=int(start_hour),
    slot_minutes=int(slot_minutes),
  )

  return _ResolvedQuery(
    program_mode=str(resolved_program_mode),
    scene_variant=str(scene_variant),
    style_variant=str(style_variant),
    accent_color_name=str(accent_color_name),
    day_label=str(day_label),
    title_text="Day Planner",
    start_hour=int(start_hour),
    end_hour=int(end_hour),
    slot_minutes=int(slot_minutes),
    max_lane_count=int(max_lane_count),
    overlap_max_lane_count=int(overlap_max_lane_count),
    overlap_nonoverlap_min_gap_slots=int(overlap_nonoverlap_min_gap_slots),
    show_reference_time_band=bool(show_reference_time_band),
    event_count_support=tuple(int(value) for value in event_count_support),
    overlap_count_support=tuple(int(value) for value in overlap_count_support),
    longer_than_count_support=tuple(int(value) for value in longer_than_count_support),
    maximum_non_overlapping_support=tuple(int(value) for value in maximum_non_overlapping_support),
    raw_events=tuple(raw_events),
    rendered_events=tuple(rendered_events),
    event_count=len(raw_events),
    lane_count=int(lane_count),
    answer_value=len(answer_event_ids),
    answer_event_ids=tuple(str(value) for value in answer_event_ids),
    reference_event_id=(str(reference_event_id) if reference_event_id is not None else None),
    branch_probabilities=dict(resolved_branch_probabilities),
    scene_variant_probabilities=dict(scene_variant_probabilities),
    style_variant_probabilities=dict(style_variant_probabilities),
    accent_color_name_probabilities=dict(accent_color_name_probabilities),
  )



def build_schedule_response(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
  selected_branch: str,
  branch_probabilities: Mapping[str, float],
  program_mode: str,
  prompt_query_key: str,
  source_query_name: str,
) -> TaskOutput:
  """Build one complete single-day schedule task response."""

  query = _resolve_query(
    int(instance_seed),
    params=params,
    program_mode=str(program_mode),
    branch_probabilities=branch_probabilities,
  )
  render_params = resolve_schedule_render_params(
    params,
    render_defaults=_RENDER_DEFAULTS,
    fallback_values=asdict(_DEFAULTS),
    instance_seed=int(instance_seed),
  )
  schedule_theme = build_time_artifact_schedule_theme(
    accent_color_name=str(query.accent_color_name),
    style_variant=str(query.style_variant),
  )

  background, background_meta = make_background_canvas(
    canvas_width=int(render_params.canvas_width),
    canvas_height=int(render_params.canvas_height),
    instance_seed=int(instance_seed),
    params=params,
    default_config=POST_IMAGE_BACKGROUND_DEFAULTS,
  )
  image = background.copy().convert("RGB")
  rendered_scene: RenderedScheduleScene = render_day_schedule_scene(
    image,
    title_text=str(query.title_text),
    day_label_text=str(query.day_label),
    start_total_minutes=int(query.start_hour * 60),
    end_total_minutes=int(query.end_hour * 60),
    slot_minutes=int(query.slot_minutes),
    lane_count=int(query.lane_count),
    events=tuple(query.rendered_events),
    scene_variant=str(query.scene_variant),
    render_params=render_params,
    visual_theme=schedule_theme,
    show_reference_time_band=False,
  )
  image, post_noise_meta = apply_post_image_noise(
    image,
    instance_seed=int(instance_seed),
    params=params,
    default_config=POST_IMAGE_NOISE_DEFAULTS,
  )

  annotation_bboxes = [
    [round(float(value), 3) for value in rendered_scene.event_bboxes_by_id[str(event_id)]]
    for event_id in query.answer_event_ids
  ]
  rendered_events_by_id = {str(event.event_id): event for event in query.rendered_events}

  prompt_selection = render_task_prompt_variants(
    domain=DOMAIN,
    scene_id=SCENE,
    bundle_id=PROMPT_BUNDLE,
    scene_key=PROMPT_SCENE_KEY,
    task_key=PROMPT_TASK_KEY,
    query_key=str(prompt_query_key),
    answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
    dynamic_slots={},
    instance_seed=int(instance_seed),
  )
  prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

  answer_gt = TypedValue(type="integer", value=int(query.answer_value))
  annotation_gt = TypedValue(type="bbox_set", value=[list(box) for box in annotation_bboxes])
  probabilities = {str(key): float(value) for key, value in dict(branch_probabilities).items()}

  common_params: Dict[str, Any] = {
    "query_id": str(selected_branch),
    "source_query_id": str(source_query_name),
    "prompt_query_key": str(prompt_query_key),
    "program_mode": str(program_mode),
    "scene_variant": str(query.scene_variant),
    "style_variant": str(query.style_variant),
    "accent_color_name": str(query.accent_color_name),
    "day_label": str(query.day_label),
    "start_hour": int(query.start_hour),
    "end_hour": int(query.end_hour),
    "slot_minutes": int(query.slot_minutes),
    "event_count": int(query.event_count),
    "lane_count": int(query.lane_count),
    "max_lane_count": int(query.max_lane_count),
    "overlap_max_lane_count": int(query.overlap_max_lane_count),
    "overlap_nonoverlap_min_gap_slots": int(query.overlap_nonoverlap_min_gap_slots),
    "show_reference_time_band": bool(query.show_reference_time_band),
    "event_count_support": [int(value) for value in query.event_count_support],
    "overlap_count_support": [int(value) for value in query.overlap_count_support],
    "longer_than_count_support": [int(value) for value in query.longer_than_count_support],
    "maximum_non_overlapping_support": [int(value) for value in query.maximum_non_overlapping_support],
    "answer_event_ids": [str(value) for value in query.answer_event_ids],
    "reference_event_id": str(query.reference_event_id) if query.reference_event_id is not None else None,
    "query_id_probabilities": dict(probabilities),
    "program_mode_probabilities": dict(query.branch_probabilities),
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

  event_records = [
    {
      "event_id": str(raw_event.event_id),
      "label": str(raw_event.label),
      "start_slot": int(raw_event.start_slot),
      "end_slot": int(raw_event.end_slot),
      "duration_slots": int(raw_event.duration_slots),
      "start_time": str(format_day_time_hhmm(int(rendered_events_by_id[str(raw_event.event_id)].start_total_minutes))),
      "end_time": str(format_day_time_hhmm(int(rendered_events_by_id[str(raw_event.event_id)].end_total_minutes))),
      "lane_index": int(rendered_events_by_id[str(raw_event.event_id)].lane_index),
      "is_reference": bool(raw_event.is_reference),
    }
    for raw_event in query.raw_events
  ]

  trace_payload = {
    "scene_ir": {
      "scene_id": SCENE,
      "scene_kind": "pages_day_schedule",
      "entities": [dict(entity) for entity in rendered_scene.entities],
      "relations": {
        "query_id": str(selected_branch),
        "source_query_id": str(source_query_name),
        "prompt_query_key": str(prompt_query_key),
        "program_mode": str(program_mode),
        "scene_variant": str(query.scene_variant),
        "style_variant": str(query.style_variant),
        "accent_color_name": str(query.accent_color_name),
        "day_label": str(query.day_label),
        "reference_event_id": str(query.reference_event_id) if query.reference_event_id is not None else None,
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
      "post_image_noise": dict(post_noise_meta),
      "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
      "schedule_style": {
        "accent_color_name": str(query.accent_color_name),
        "style_variant": str(query.style_variant),
        "resolved_colors_rgb": {
          "panel_fill": [int(value) for value in schedule_theme.panel_fill_rgb],
          "panel_outline": [int(value) for value in schedule_theme.panel_outline_rgb],
          "header_fill": [int(value) for value in schedule_theme.header_fill_rgb],
          "header_text": [int(value) for value in schedule_theme.header_text_rgb],
          "grid_line": [int(value) for value in schedule_theme.grid_line_rgb],
          "minor_grid_line": [int(value) for value in schedule_theme.minor_grid_line_rgb],
          "event_fill": [int(value) for value in schedule_theme.event_fill_rgb],
          "event_outline": [int(value) for value in schedule_theme.event_outline_rgb],
          "reference_fill": [int(value) for value in schedule_theme.reference_fill_rgb],
          "reference_outline": [int(value) for value in schedule_theme.reference_outline_rgb],
        },
        "header_kind": str(schedule_theme.header_kind),
      },
    },
    "render_map": {
      "image_id": "img0",
      "scene_bbox_px": [round(float(value), 3) for value in rendered_scene.scene_bbox_px],
      "panel_bbox_px": [round(float(value), 3) for value in rendered_scene.panel_bbox_px],
      "title_text": str(rendered_scene.title_text),
      "day_label": str(query.day_label),
      "event_bboxes_by_id": {
        str(event_id): [round(float(value), 3) for value in bbox]
        for event_id, bbox in rendered_scene.event_bboxes_by_id.items()
      },
      "answer_event_ids": [str(value) for value in query.answer_event_ids],
      "reference_event_id": str(query.reference_event_id) if query.reference_event_id is not None else None,
    },
    "execution_trace": {
      **dict(common_params),
      "answer_value": int(query.answer_value),
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
  "SCENE",
  "OVERLAP_COUNT_MODE",
  "LONGER_THAN_REFERENCE_MODE",
  "MAXIMUM_NON_OVERLAPPING_MODE",
  "build_schedule_response",
]
