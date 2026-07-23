"""Multi-clock comparison task with earliest/latest label queries."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Tuple

from PIL import ImageDraw

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice, uniform_choice_with_probabilities
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import bbox_annotation_artifacts
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
  PROMPT_OUTPUT_MODES,
  build_prompt_trace_artifacts,
  render_task_prompt_variants,
)
from ...shared.text_rendering import draw_text_centered, load_font, temporary_default_font_family
from .shared.state import (
  SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
  ClockRenderParams,
)
from .shared.rendering import (
  draw_clock_geometry,
)
from .shared.styles import (
  resolve_clock_render_params,
)
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style
from ...shared.time_artifact_task_support import resolve_time_artifact_named_variant
from ...shared.time_artifact_style import (
  SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
  SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
  build_time_artifact_clock_theme,
)
from ...shared.time_format import clock_hand_angle_gap_deg, clock_total_minutes, format_clock_hhmm, split_clock_total_minutes
from ..shared.visual_defaults import load_symbolic_background_defaults, load_symbolic_noise_defaults


DOMAIN = "symbolic"
SCENE_ID = "clock"
TASK_ID = "task_symbolic__clock__time_extremum_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
  "earliest_time_label",
  "latest_time_label",
)
_EXTREMUM_DIRECTION_BY_QUERY = {
  "earliest_time_label": "earliest",
  "latest_time_label": "latest",
}
_SUPPORTED_EXTREMUM_DIRECTIONS: Tuple[str, ...] = ("earliest", "latest")

_TIME_READING_BASE_BY_DIRECTION = {
  "earliest": 0.44,
  "latest": 0.44,
}
_VISUAL_SCAN_BASE_BY_SCENE = {
  "classic": 0.58,
  "minimal": 0.46,
  "outline": 0.50,
}
_VISUAL_SCAN_STYLE_BONUS = {
  "studio": 0.00,
  "accented": 0.04,
  "marker": 0.06,
}
_CLUTTER_BASE_BY_SCENE = {
  "classic": 0.42,
  "minimal": 0.24,
  "outline": 0.30,
}
_CLUTTER_STYLE_BONUS = {
  "studio": 0.00,
  "accented": 0.04,
  "marker": 0.07,
}


@dataclass(frozen=True)
class _TaskDefaults:
  """Stable fallback defaults for multi-clock comparison scenes."""

  hour_min: int = 1
  hour_max: int = 12
  minute_min: int = 0
  minute_max: int = 55
  minute_step: int = 5
  min_hand_angle_gap_deg: float = 10.0
  clock_label_support: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
  clock_count_support: Tuple[int, ...] = (6,)
  min_compare_gap_minutes: int = 15
  canvas_width: int = 960
  canvas_height: int = 760
  outer_margin_px: int = 40
  face_radius_px: int = 84
  bezel_width_px: int = 8
  numeral_font_size_px: int = 18
  major_tick_length_px: int = 14
  minor_tick_length_px: int = 6
  major_tick_width_px: int = 3
  minor_tick_width_px: int = 2
  minor_tick_dot_radius_px: int = 2
  hour_hand_width_px: int = 8
  minute_hand_width_px: int = 6
  second_hand_width_px: int = 2
  hand_bbox_padding_px: int = 5
  center_dot_radius_px: int = 6
  inner_ring_inset_px: int = 14
  inner_ring_width_px: int = 3
  grid_col_gap_px: int = 30
  grid_row_gap_px: int = 34
  label_font_size_px: int = 28
  label_gap_px: int = 20


@dataclass(frozen=True)
class _ResolvedQuery:
  """Resolved semantic and visual support for one multi-clock comparison instance."""

  query_id: str
  extremum_direction: str
  scene_variant: str
  style_variant: str
  accent_color_name: str
  clock_count: int
  clock_labels: Tuple[str, ...]
  clock_label_pool: Tuple[str, ...]
  shown_total_minutes_by_label: Dict[str, int]
  winner_label: str
  winner_total_minutes: int
  min_compare_gap_minutes: int
  hour_support: Tuple[int, int]
  minute_support: Tuple[int, int, int]
  clock_count_support: Tuple[int, ...]
  query_id_probabilities: Dict[str, float]
  extremum_direction_probabilities: Dict[str, float]
  scene_variant_probabilities: Dict[str, float]
  style_variant_probabilities: Dict[str, float]
  accent_color_name_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
  DOMAIN,
  SCENE_ID,
  task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_symbolic_background_defaults(scene_id="clock")
POST_IMAGE_NOISE_DEFAULTS = {
  **load_symbolic_noise_defaults(scene_id="clock", apply_prob=0.5),
  "apply_prob": 0.5,
}


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
  """Resolve one balanced named symbolic-clock axis."""

  rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{namespace}")
  return resolve_time_artifact_named_variant(
    rng,
    params=params,
    gen_defaults=_GEN_DEFAULTS,
    explicit_key=str(explicit_key),
    weights_key=str(weights_key),
    balance_flag_key=str(balance_flag_key),
    supported=supported,
    instance_seed=int(instance_seed),
    task_id=TASK_ID,
    namespace=str(namespace),
  )


def _canonical_compare_example_bbox() -> list[int]:
  """Return one stable winning-clock bbox example for prompt examples."""

  return [368, 94, 552, 278]


def _build_prompt_json_examples() -> tuple[str, str]:
  """Return prompt JSON examples for one compare variant."""

  answer_and_annotation = {
    "annotation": _canonical_compare_example_bbox(),
    "answer": "B",
  }
  answer_only = {"answer": "B"}
  return (
    json.dumps(answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    json.dumps(answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
  )


def _resolve_clock_labels(params: Mapping[str, Any]) -> Tuple[str, ...]:
  """Resolve the global visible-label pool for this task."""

  raw_labels = params.get("clock_label_support", group_default(_GEN_DEFAULTS, "clock_label_support", _DEFAULTS.clock_label_support))
  labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
  if len(labels) < 2:
    raise ValueError("symbolic clock compare requires at least two candidate clock labels")
  if len(set(labels)) != len(labels):
    raise ValueError("symbolic clock compare labels must be unique")
  return labels


def _resolve_clock_count_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
  """Resolve the supported visible clock counts for compare scenes."""

  raw_support = params.get(
    "clock_count_support",
    group_default(_GEN_DEFAULTS, "clock_count_support", _DEFAULTS.clock_count_support),
  )
  support: List[int] = []
  for value in raw_support:
    clock_count = int(value)
    if clock_count < 2:
      raise ValueError("symbolic clock compare clock_count_support must contain counts of at least two")
    if clock_count not in support:
      support.append(clock_count)
  if not support:
    raise ValueError("symbolic clock compare clock_count_support is empty")
  return tuple(int(value) for value in support)


def _resolve_query_id(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
  """Resolve the public compare query id."""

  return select_task_query_id(
    instance_seed=int(instance_seed),
    params=params,
    supported_query_ids=SUPPORTED_QUERY_IDS,
    default_query_id="earliest_time_label",
    task_id=TASK_ID,
    namespace=f"{TASK_ID}.query",
  )


def _normalize_extremum_direction(value: Any) -> str:
  """Normalize one clock-comparison extremum direction."""

  normalized = str(value).strip().lower()
  if normalized in _SUPPORTED_EXTREMUM_DIRECTIONS:
    return normalized
  raise ValueError(f"unsupported extremum_direction: {value}")


def _resolve_extremum_direction(
  *,
  query_id: str,
) -> Tuple[str, Dict[str, float]]:
  """Resolve whether the query asks for the earliest or latest clock."""

  try:
    selected = str(_EXTREMUM_DIRECTION_BY_QUERY[str(query_id)])
  except KeyError as exc:
    raise ValueError(f"unsupported query_id for {TASK_ID}: {query_id}") from exc
  return selected, {
    key: (1.0 if key == selected else 0.0)
    for key in _SUPPORTED_EXTREMUM_DIRECTIONS
  }


def _feasible_clock_support(
  *,
  hour_support: Tuple[int, ...],
  minute_support: Tuple[int, ...],
  min_hand_angle_gap_deg: float,
) -> Tuple[int, ...]:
  """Return all feasible shown times after the hand-gap filter."""

  return tuple(
    clock_total_minutes(int(hour), int(minute))
    for hour in hour_support
    for minute in minute_support
    if float(clock_hand_angle_gap_deg(clock_total_minutes(int(hour), int(minute)))) >= float(min_hand_angle_gap_deg)
  )


def _choose_clock_times(
  *,
  rng,
  extremum_direction: str,
  shown_total_support: Tuple[int, ...],
  labels: Tuple[str, ...],
  winner_index: int,
  min_compare_gap_minutes: int,
) -> Tuple[Dict[str, int], str, int]:
  """Choose one winning label plus unique visible times for all labeled clocks."""

  clock_count = len(labels)
  gap = int(min_compare_gap_minutes)
  if clock_count <= 1:
    raise ValueError("symbolic clock compare requires at least two clocks")

  feasible_winners: List[int] = []
  for shown_total in shown_total_support:
    if str(extremum_direction) == "earliest":
      candidate_pool = [value for value in shown_total_support if int(value) - int(shown_total) >= int(gap)]
    else:
      candidate_pool = [value for value in shown_total_support if int(shown_total) - int(value) >= int(gap)]
    if len(candidate_pool) >= int(clock_count - 1):
      feasible_winners.append(int(shown_total))
  if not feasible_winners:
    raise ValueError("symbolic clock compare has no feasible winning times under the current support")

  winning_total = int(uniform_choice(rng, tuple(feasible_winners), sort_keys=True))
  if str(extremum_direction) == "earliest":
    candidate_pool = [value for value in shown_total_support if int(value) - int(winning_total) >= int(gap)]
  else:
    candidate_pool = [value for value in shown_total_support if int(winning_total) - int(value) >= int(gap)]
  other_totals = [int(value) for value in rng.sample(candidate_pool, k=int(clock_count - 1))]

  winner_label = str(labels[int(winner_index)])
  shown_total_minutes_by_label: Dict[str, int] = {}
  remaining_labels = [str(label) for offset, label in enumerate(labels) if int(offset) != int(winner_index)]
  rng.shuffle(other_totals)
  for label, shown_total in zip(remaining_labels, other_totals):
    shown_total_minutes_by_label[str(label)] = int(shown_total)
  shown_total_minutes_by_label[str(winner_label)] = int(winning_total)
  return shown_total_minutes_by_label, str(winner_label), int(winning_total)


def _sample_clock_labels(
  *,
  instance_seed: int,
  params: Mapping[str, Any],
  label_pool: Tuple[str, ...],
  clock_count: int,
) -> Tuple[Tuple[str, ...], str]:
  """Return row-major visible labels and the sampled winning label."""

  active_labels = tuple(str(label) for label in label_pool[: int(clock_count)])

  winner_index_explicit = params.get("winner_index")
  if winner_index_explicit is not None:
    winner_index = int(winner_index_explicit)
    if winner_index < 0 or winner_index >= int(clock_count):
      raise ValueError("winner_index is outside the active visible clock range")
  else:
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.winner_index")
    winner_index = int(
      uniform_choice(
        rng,
        tuple(range(int(clock_count))),
        sort_keys=True,
      )
    )

  winner_label_explicit = params.get("winner_label")
  if winner_label_explicit is not None:
    winner_label = str(winner_label_explicit).strip()
    if winner_label not in active_labels:
      raise ValueError("winner_label must be one of the visible row-major clock labels")
    winner_index = int(active_labels.index(str(winner_label)))
  else:
    winner_label = str(active_labels[int(winner_index)])
  return active_labels, str(winner_label)


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
  """Resolve one multi-clock comparison query from balanced supports."""

  query_id, query_id_probabilities, task_params = _resolve_query_id(
    instance_seed=int(instance_seed),
    params=params,
  )
  extremum_direction, extremum_direction_probabilities = _resolve_extremum_direction(
    query_id=str(query_id),
  )
  scene_variant, scene_variant_probabilities = _resolve_named_variant(
    instance_seed=int(instance_seed),
    params=task_params,
    explicit_key="scene_variant",
    weights_key="scene_variant_weights",
    balance_flag_key="balanced_scene_variant_sampling",
    supported=SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
    namespace="scene_variant",
  )
  style_variant, style_variant_probabilities = _resolve_named_variant(
    instance_seed=int(instance_seed),
    params=task_params,
    explicit_key="style_variant",
    weights_key="style_variant_weights",
    balance_flag_key="balanced_style_variant_sampling",
    supported=SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
    namespace="style_variant",
  )
  accent_color_name, accent_color_name_probabilities = _resolve_named_variant(
    instance_seed=int(instance_seed),
    params=task_params,
    explicit_key="accent_color_name",
    weights_key="accent_color_name_weights",
    balance_flag_key="balanced_accent_color_name_sampling",
    supported=SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
    namespace="accent_color_name",
  )

  hour_min = int(task_params.get("hour_min", group_default(_GEN_DEFAULTS, "hour_min", _DEFAULTS.hour_min)))
  hour_max = int(task_params.get("hour_max", group_default(_GEN_DEFAULTS, "hour_max", _DEFAULTS.hour_max)))
  minute_min = int(task_params.get("minute_min", group_default(_GEN_DEFAULTS, "minute_min", _DEFAULTS.minute_min)))
  minute_max = int(task_params.get("minute_max", group_default(_GEN_DEFAULTS, "minute_max", _DEFAULTS.minute_max)))
  minute_step = int(task_params.get("minute_step", group_default(_GEN_DEFAULTS, "minute_step", _DEFAULTS.minute_step)))
  min_hand_angle_gap_deg = float(
      task_params.get(
      "min_hand_angle_gap_deg",
      group_default(_GEN_DEFAULTS, "min_hand_angle_gap_deg", _DEFAULTS.min_hand_angle_gap_deg),
    )
  )
  min_compare_gap_minutes = int(
      task_params.get(
      "min_compare_gap_minutes",
      group_default(_GEN_DEFAULTS, "min_compare_gap_minutes", _DEFAULTS.min_compare_gap_minutes),
    )
  )
  if minute_step <= 0:
    raise ValueError("minute_step must be positive for symbolic clock compare")
  if int(min_compare_gap_minutes) <= 0:
    raise ValueError("min_compare_gap_minutes must be positive for symbolic clock compare")

  hour_support = tuple(range(int(hour_min), int(hour_max) + 1))
  minute_support = tuple(range(int(minute_min), int(minute_max) + 1, int(minute_step)))
  if not hour_support or not minute_support:
    raise ValueError("symbolic clock compare support is empty")
  label_pool = _resolve_clock_labels(task_params)
  clock_count_support = _resolve_clock_count_support(task_params)
  if len(label_pool) < max(clock_count_support):
    raise ValueError("symbolic clock compare requires enough clock labels to cover the maximum visible clock count")
  explicit_clock_count = task_params.get("clock_count")
  if explicit_clock_count is not None:
    clock_count = int(explicit_clock_count)
    if clock_count not in clock_count_support:
      raise ValueError("clock_count is outside the configured support")
  else:
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.clock_count")
    clock_count, _clock_count_probs = uniform_choice_with_probabilities(
      rng,
      clock_count_support,
      sort_keys=True,
    )
    clock_count = int(clock_count)
  labels, winner_label = _sample_clock_labels(
    instance_seed=int(instance_seed),
    params=task_params,
    label_pool=tuple(str(label) for label in label_pool),
    clock_count=int(clock_count),
  )
  shown_total_support = _feasible_clock_support(
    hour_support=hour_support,
    minute_support=minute_support,
    min_hand_angle_gap_deg=float(min_hand_angle_gap_deg),
  )
  if not shown_total_support:
    raise ValueError("shown time support is empty after clock-hand filtering for symbolic clock compare")

  winner_index = int(labels.index(str(winner_label)))

  rng = spawn_rng(int(instance_seed), f"{TASK_ID}.clock_times")
  shown_total_minutes_by_label, winner_label, winner_total_minutes = _choose_clock_times(
    rng=rng,
    extremum_direction=str(extremum_direction),
    shown_total_support=tuple(int(value) for value in shown_total_support),
    labels=tuple(str(label) for label in labels),
    winner_index=int(winner_index),
    min_compare_gap_minutes=int(min_compare_gap_minutes),
  )

  return _ResolvedQuery(
    query_id=str(query_id),
    extremum_direction=str(extremum_direction),
    scene_variant=str(scene_variant),
    style_variant=str(style_variant),
    accent_color_name=str(accent_color_name),
    clock_count=int(clock_count),
    clock_labels=tuple(str(label) for label in labels),
    clock_label_pool=tuple(str(label) for label in label_pool),
    shown_total_minutes_by_label={str(key): int(value) for key, value in shown_total_minutes_by_label.items()},
    winner_label=str(winner_label),
    winner_total_minutes=int(winner_total_minutes),
    min_compare_gap_minutes=int(min_compare_gap_minutes),
    hour_support=(int(hour_support[0]), int(hour_support[-1])),
    minute_support=(int(minute_support[0]), int(minute_support[-1]), int(minute_step)),
    clock_count_support=tuple(int(value) for value in clock_count_support),
    query_id_probabilities=dict(query_id_probabilities),
    extremum_direction_probabilities=dict(extremum_direction_probabilities),
    scene_variant_probabilities=dict(scene_variant_probabilities),
    style_variant_probabilities=dict(style_variant_probabilities),
    accent_color_name_probabilities=dict(accent_color_name_probabilities),
  )


def _resolve_row_lengths(clock_count: int) -> Tuple[int, ...]:
  """Return centered multi-row layouts that preserve the compare-task grid feel."""

  row_lengths_by_count = {
    6: (3, 3),
    7: (4, 3),
    8: (4, 4),
    9: (3, 3, 3),
    10: (4, 3, 3),
    11: (4, 4, 3),
    12: (4, 4, 4),
  }
  if int(clock_count) not in row_lengths_by_count:
    raise ValueError("symbolic clock compare currently supports visible clock counts in 6..12")
  return tuple(int(value) for value in row_lengths_by_count[int(clock_count)])


def _clock_centers(
  *,
  render_params: ClockRenderParams,
  clock_count: int,
  grid_col_gap_px: int,
  grid_row_gap_px: int,
  label_gap_px: int,
  label_font_size_px: int,
) -> List[Tuple[float, float]]:
  """Return centered clock centers for the active compare-scene layout."""

  row_lengths = _resolve_row_lengths(int(clock_count))
  face_diameter = 2.0 * float(render_params.face_radius_px)
  row_band_height = float(label_font_size_px) + float(label_gap_px) + float(face_diameter)
  grid_height = (float(len(row_lengths)) * float(row_band_height)) + (
    float(max(0, len(row_lengths) - 1)) * float(grid_row_gap_px)
  )
  origin_y = 0.5 * float(render_params.canvas_height - grid_height)

  centers: List[Tuple[float, float]] = []
  row_top = float(origin_y)
  for row_length in row_lengths:
    row_width = (float(row_length) * float(face_diameter)) + (float(max(0, row_length - 1)) * float(grid_col_gap_px))
    row_origin_x = 0.5 * float(render_params.canvas_width - row_width)
    for col_index in range(int(row_length)):
      center_x = float(
        row_origin_x + (col_index * (face_diameter + float(grid_col_gap_px))) + float(render_params.face_radius_px)
      )
      center_y = float(row_top + float(label_font_size_px) + float(label_gap_px) + float(render_params.face_radius_px))
      centers.append((float(center_x), float(center_y)))
    row_top += float(row_band_height + float(grid_row_gap_px))
  return centers


@register_task
class SymbolicClockCompareTask:
  """Compare multiple labeled analog clocks and identify the earliest or latest one."""

  task_id = TASK_ID
  reasoning_operations = ('ranking',)
  domain = DOMAIN
  supported_query_ids = SUPPORTED_QUERY_IDS
  default_dataset_enabled = True

  def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
    """Generate one labeled clock-grid instance with exactly one extremum clock.

    This public task owns the earliest/latest objective, including the winning
    label, scalar clock-face bbox annotation, prompt slots, and final output.
    The scene-shared clock renderer is used only after the unique winner and
    all visible clock times are fixed.
    """

    del max_attempts
    query = _resolve_query(int(instance_seed), params=params)
    render_params = resolve_clock_render_params(
      params,
      render_defaults=_RENDER_DEFAULTS,
      fallback_values=asdict(_DEFAULTS),
      instance_seed=int(instance_seed),
    )
    clock_theme = build_time_artifact_clock_theme(
      accent_color_name=str(query.accent_color_name),
      style_variant=str(query.style_variant),
    )
    font_family = sample_font_family(
      role="readout",
      instance_seed=int(instance_seed),
      namespace=f"{TASK_ID}.font",
      params={**dict(_RENDER_DEFAULTS), **dict(params)},
    )

    label_font_size_px = int(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", _DEFAULTS.label_font_size_px)))
    label_gap_px = int(params.get("label_gap_px", group_default(_RENDER_DEFAULTS, "label_gap_px", _DEFAULTS.label_gap_px)))
    grid_col_gap_px = int(params.get("grid_col_gap_px", group_default(_RENDER_DEFAULTS, "grid_col_gap_px", _DEFAULTS.grid_col_gap_px)))
    grid_row_gap_px = int(params.get("grid_row_gap_px", group_default(_RENDER_DEFAULTS, "grid_row_gap_px", _DEFAULTS.grid_row_gap_px)))

    scene_style, scene_style_meta = resolve_symbolic_scene_style(
      instance_seed=int(instance_seed),
      namespace=f"{TASK_ID}.background",
    )
    background, background_meta = make_symbolic_scene_background(
      canvas_width=int(render_params.canvas_width),
      canvas_height=int(render_params.canvas_height),
      style=scene_style,
    )
    image = background.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    centers = _clock_centers(
      render_params=render_params,
      clock_count=len(query.clock_labels),
      grid_col_gap_px=int(grid_col_gap_px),
      grid_row_gap_px=int(grid_row_gap_px),
      label_gap_px=int(label_gap_px),
      label_font_size_px=int(label_font_size_px),
    )

    scene_entities: List[Dict[str, Any]] = []
    clocks_by_label: Dict[str, Dict[str, Any]] = {}
    scene_bbox_values: List[Tuple[float, float, float, float]] = []
    with temporary_default_font_family(str(font_family)):
      label_font = load_font(int(label_font_size_px), bold=True)
      for label, center in zip(query.clock_labels, centers):
        geometry = draw_clock_geometry(
          image,
          center_px=center,
          face_radius_px=float(render_params.face_radius_px),
          scene_variant=str(query.scene_variant),
          shown_total_minutes=int(query.shown_total_minutes_by_label[str(label)]),
          render_params=render_params,
          visual_theme=clock_theme,
          entity_prefix=f"clock_{str(label).lower()}",
          extra_face_attrs={
            "clock_label": str(label),
            "shown_time_text": str(format_clock_hhmm(int(query.shown_total_minutes_by_label[str(label)]))),
          },
        )
        label_center = (
          float(center[0]),
          float(center[1] - float(render_params.face_radius_px) - float(label_gap_px)),
        )
        draw_text_centered(
          draw,
          text=str(label),
          center=label_center,
          font=label_font,
          fill=tuple(int(value) for value in clock_theme.numeral_color_rgb),
        )
        scene_entities.extend([dict(entity) for entity in geometry.entities])
        clocks_by_label[str(label)] = {
          "face_bbox_px": [round(float(value), 3) for value in geometry.face_bbox_px],
          "center_px": [round(float(value), 3) for value in geometry.center_px],
          "hour_hand_bbox_px": [round(float(value), 3) for value in geometry.hour_hand_bbox_px],
          "minute_hand_bbox_px": [round(float(value), 3) for value in geometry.minute_hand_bbox_px],
          "hour_hand_tip_px": [round(float(value), 3) for value in geometry.hour_hand_tip_px],
          "minute_hand_tip_px": [round(float(value), 3) for value in geometry.minute_hand_tip_px],
          "shown_total_minutes": int(query.shown_total_minutes_by_label[str(label)]),
          "shown_time_text": str(format_clock_hhmm(int(query.shown_total_minutes_by_label[str(label)]))),
          "clock_label": str(label),
        }
        scene_bbox_values.append(tuple(float(value) for value in geometry.face_bbox_px))

    image, post_noise_meta = apply_post_image_noise(
      image,
      instance_seed=int(instance_seed),
      params=params,
      default_config=POST_IMAGE_NOISE_DEFAULTS,
    )

    prompt_defaults = required_group_defaults(
      _PROMPT_DEFAULTS,
      (
        "bundle_id",
        "scene_key",
        "task_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        "object_description_time_extremum_label_classic",
        "object_description_time_extremum_label_minimal",
        "object_description_time_extremum_label_outline",
        "annotation_hint_time_extremum_label",
        "answer_hint_time_extremum_label",
      ),
      context=f"prompt defaults for {self.task_id}",
    )
    object_description = str(prompt_defaults[f"object_description_time_extremum_label_{str(query.scene_variant)}"])
    json_example, json_example_answer_only = _build_prompt_json_examples()
    prompt_selection = render_task_prompt_variants(
      domain=DOMAIN,
      scene_id=SCENE_ID,
      bundle_id=str(prompt_defaults["bundle_id"]),
      scene_key=str(prompt_defaults["scene_key"]),
      task_key=str(prompt_defaults["task_key"]),
      query_key=str(query.query_id),
      answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
      dynamic_slots={
        "object_description": str(object_description),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "annotation_hint": str(prompt_defaults["annotation_hint_time_extremum_label"]),
        "answer_hint": str(prompt_defaults["answer_hint_time_extremum_label"]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
      },
      instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    winning_bbox = list(clocks_by_label[str(query.winner_label)]["face_bbox_px"])
    answer_gt = TypedValue(type="string", value=str(query.winner_label))
    annotation_payload = bbox_annotation_artifacts(winning_bbox)
    annotation_gt = annotation_payload.annotation_gt

    scene_bbox = (
      min(box[0] for box in scene_bbox_values),
      min(box[1] for box in scene_bbox_values),
      max(box[2] for box in scene_bbox_values),
      max(box[3] for box in scene_bbox_values),
    )
    ordered_times = [int(query.shown_total_minutes_by_label[str(label)]) for label in query.clock_labels]
    sorted_times = sorted(int(value) for value in ordered_times)
    boundary_gap = int(sorted_times[1] - sorted_times[0]) if str(query.extremum_direction) == "earliest" else int(sorted_times[-1] - sorted_times[-2])

    trace_payload = {
      "scene_ir": {
        "scene_kind": "symbolic_clock_grid",
        "entities": list(scene_entities),
        "relations": {
          "query_id": str(query.query_id),
          "extremum_direction": str(query.extremum_direction),
          "scene_variant": str(query.scene_variant),
          "style_variant": str(query.style_variant),
          "accent_color_name": str(query.accent_color_name),
          "winner_label": str(query.winner_label),
          "winner_time_text": str(format_clock_hhmm(int(query.winner_total_minutes))),
          "clock_count": int(query.clock_count),
          "clock_labels": [str(label) for label in query.clock_labels],
        },
      },
      "query_spec": {
        "query_id": str(query.query_id),
        "template_id": str(prompt_defaults["bundle_id"]),
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        "params": {
          "query_id": str(query.query_id),
          "extremum_direction": str(query.extremum_direction),
          "scene_variant": str(query.scene_variant),
          "style_variant": str(query.style_variant),
          "accent_color_name": str(query.accent_color_name),
          "query_id_probabilities": dict(query.query_id_probabilities),
          "extremum_direction_probabilities": dict(query.extremum_direction_probabilities),
          "scene_variant_probabilities": dict(query.scene_variant_probabilities),
          "style_variant_probabilities": dict(query.style_variant_probabilities),
          "accent_color_name_probabilities": dict(query.accent_color_name_probabilities),
          "clock_count": int(query.clock_count),
          "clock_count_support": [int(value) for value in query.clock_count_support],
          "clock_label_pool": [str(label) for label in query.clock_label_pool],
          "clock_labels": [str(label) for label in query.clock_labels],
          "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
          "minute_support": [int(value) for value in query.minute_support],
          "min_compare_gap_minutes": int(query.min_compare_gap_minutes),
        },
      },
      "render_spec": {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(query.scene_variant),
        "background_style": dict(background_meta),
        "scene_style": dict(scene_style_meta),
        "post_image_noise": dict(post_noise_meta),
        "scene_bbox_px": [round(float(value), 3) for value in scene_bbox],
        "clock_style": {
          "accent_color_name": str(query.accent_color_name),
          "style_variant": str(query.style_variant),
          "face_radius_px": int(render_params.face_radius_px),
          "bezel_width_px": int(render_params.bezel_width_px),
          "numeral_font_size_px": int(render_params.numeral_font_size_px),
          "hour_hand_width_px": int(render_params.hour_hand_width_px),
          "minute_hand_width_px": int(render_params.minute_hand_width_px),
          "label_font_size_px": int(label_font_size_px),
          "label_gap_px": int(label_gap_px),
          "font": {
            "source": "global_font_pool",
            "font_family": str(font_family),
            "font_asset_version": font_asset_version(),
            "scope": "multi_clock_faces_and_labels",
          },
          "clock_count": int(query.clock_count),
          "row_lengths": [int(value) for value in _resolve_row_lengths(int(query.clock_count))],
          "grid_col_gap_px": int(grid_col_gap_px),
          "grid_row_gap_px": int(grid_row_gap_px),
          "resolved_colors_rgb": {
            "face_fill": [int(value) for value in clock_theme.face_fill_rgb],
            "face_outline": [int(value) for value in clock_theme.face_outline_rgb],
            "numerals": [int(value) for value in clock_theme.numeral_color_rgb],
            "ticks": [int(value) for value in clock_theme.tick_color_rgb],
            "hour_hand": [int(value) for value in clock_theme.hour_hand_color_rgb],
            "minute_hand": [int(value) for value in clock_theme.minute_hand_color_rgb],
            "center_dot": [int(value) for value in clock_theme.center_dot_color_rgb],
          },
          "minor_tick_mode": str(clock_theme.minor_tick_mode),
        },
      },
      "render_map": {
        "image_id": "img0",
        "scene_bbox_px": [round(float(value), 3) for value in scene_bbox],
        "clocks_by_label": dict(clocks_by_label),
        "winning_label": str(query.winner_label),
        "winning_clock_bbox_px": list(winning_bbox),
      },
      "execution_trace": {
        "query_id": str(query.query_id),
        "extremum_direction": str(query.extremum_direction),
        "scene_variant": str(query.scene_variant),
        "style_variant": str(query.style_variant),
        "accent_color_name": str(query.accent_color_name),
        "clock_count": int(query.clock_count),
        "clock_label_pool": [str(label) for label in query.clock_label_pool],
        "clock_labels": [str(label) for label in query.clock_labels],
        "shown_total_minutes_by_label": {str(key): int(value) for key, value in query.shown_total_minutes_by_label.items()},
        "shown_time_text_by_label": {
          str(label): str(format_clock_hhmm(int(query.shown_total_minutes_by_label[str(label)])))
          for label in query.clock_labels
        },
        "winner_label": str(query.winner_label),
        "winner_total_minutes": int(query.winner_total_minutes),
        "winner_time_text": str(format_clock_hhmm(int(query.winner_total_minutes))),
        "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
        "minute_support": [int(value) for value in query.minute_support],
        "min_compare_gap_minutes": int(query.min_compare_gap_minutes),
        "query_id_probabilities": dict(query.query_id_probabilities),
        "extremum_direction_probabilities": dict(query.extremum_direction_probabilities),
        "scene_variant_probabilities": dict(query.scene_variant_probabilities),
        "style_variant_probabilities": dict(query.style_variant_probabilities),
        "accent_color_name_probabilities": dict(query.accent_color_name_probabilities),
        "question_format": str(query.query_id),
        "supporting_parts": ["winning_clock_face"],
      },
      "witness_symbolic": {
        "type": "bbox",
        "value": list(annotation_payload.value),
      },
      "projected_annotation": dict(annotation_payload.projected_annotation),
    }


    output = TaskOutput(
      prompt=str(prompt_artifacts.prompt),
      answer_gt=answer_gt,
      annotation_gt=annotation_gt,
      image=image,
      image_id="img0",
      trace_payload=trace_payload,
      task_versions=default_task_versions(),
      scene_id=SCENE_ID,
      query_id=str(query.query_id),
      prompt_variants=dict(prompt_artifacts.prompt_variants),
    )
    return output


__all__ = ["SymbolicClockCompareTask"]
