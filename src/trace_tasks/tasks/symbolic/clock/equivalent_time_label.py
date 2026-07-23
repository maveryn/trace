"""Analog/digital clock matching panel task."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults, required_group_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.font_assets import font_asset_version, sample_font_family
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
  PROMPT_OUTPUT_MODES,
  build_prompt_trace_artifacts,
  render_task_prompt_variants,
)
from ...shared.text_legibility import draw_traced_text
from ...shared.text_rendering import draw_text_centered, load_font, temporary_default_font_family
from ...shared.time_artifact_style import (
  SUPPORTED_TIME_ARTIFACT_CLOCK_COLOR_NAMES,
  SUPPORTED_TIME_ARTIFACT_CLOCK_STYLE_VARIANTS,
  build_time_artifact_clock_theme,
)
from ...shared.time_artifact_task_support import resolve_time_artifact_named_variant
from ...shared.time_format import clock_hand_angle_gap_deg, clock_total_minutes, format_clock_hhmm, split_clock_total_minutes
from .shared.state import (
  SUPPORTED_SYMBOLIC_CLOCK_SCENE_VARIANTS,
)
from .shared.rendering import (
  draw_clock_geometry,
)
from .shared.styles import (
  resolve_clock_render_params,
  scale_clock_render_params_for_radius,
)
from ..shared.scene_style import make_symbolic_scene_background, resolve_symbolic_scene_style
from ..shared.visual_defaults import load_symbolic_background_defaults, load_symbolic_noise_defaults


DOMAIN = "symbolic"
SCENE_ID = "clock"
TASK_ID = "task_symbolic__clock__equivalent_time_label"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
  "analog_reference_digital_options",
  "digital_reference_analog_options",
)
SUPPORTED_DIGITAL_DISPLAY_PALETTES: Tuple[str, ...] = (
  "charcoal_mint",
  "navy_cyan",
  "graphite_amber",
  "cream_ink",
  "blue_lcd",
  "wine_rose",
  "forest_lime",
  "plum_ice",
)
_REPRESENTATION_BY_QUERY_ID = {
  "analog_reference_digital_options": ("analog", "digital"),
  "digital_reference_analog_options": ("digital", "analog"),
}

_TIME_READING_BASE_BY_QUERY = {
  "analog_reference_digital_options": 0.48,
  "digital_reference_analog_options": 0.42,
}
_VISUAL_SCAN_BASE_BY_SCENE = {
  "classic": 0.40,
  "minimal": 0.32,
  "outline": 0.35,
}
_VISUAL_SCAN_STYLE_BONUS = {
  "studio": 0.00,
  "accented": 0.04,
  "marker": 0.06,
}
_CLUTTER_BASE_BY_QUERY = {
  "analog_reference_digital_options": 0.34,
  "digital_reference_analog_options": 0.46,
}


@dataclass(frozen=True)
class _DigitalDisplayPalette:
  """Resolved non-semantic digital display colors."""

  case_fill_rgb: Tuple[int, int, int]
  case_outline_rgb: Tuple[int, int, int]
  screen_fill_rgb: Tuple[int, int, int]
  screen_outline_rgb: Tuple[int, int, int]
  text_rgb: Tuple[int, int, int]
  glow_rgb: Tuple[int, int, int]
  shadow_rgb: Tuple[int, int, int]


_DIGITAL_DISPLAY_PALETTES: Dict[str, _DigitalDisplayPalette] = {
  "charcoal_mint": _DigitalDisplayPalette(
    case_fill_rgb=(30, 35, 42),
    case_outline_rgb=(46, 132, 102),
    screen_fill_rgb=(14, 18, 22),
    screen_outline_rgb=(58, 65, 72),
    text_rgb=(235, 248, 238),
    glow_rgb=(86, 214, 155),
    shadow_rgb=(210, 214, 220),
  ),
  "navy_cyan": _DigitalDisplayPalette(
    case_fill_rgb=(18, 35, 58),
    case_outline_rgb=(62, 154, 198),
    screen_fill_rgb=(9, 24, 40),
    screen_outline_rgb=(67, 111, 144),
    text_rgb=(214, 247, 255),
    glow_rgb=(69, 192, 223),
    shadow_rgb=(201, 210, 222),
  ),
  "graphite_amber": _DigitalDisplayPalette(
    case_fill_rgb=(42, 40, 36),
    case_outline_rgb=(183, 126, 42),
    screen_fill_rgb=(20, 18, 16),
    screen_outline_rgb=(94, 75, 46),
    text_rgb=(255, 218, 127),
    glow_rgb=(232, 145, 40),
    shadow_rgb=(214, 209, 199),
  ),
  "cream_ink": _DigitalDisplayPalette(
    case_fill_rgb=(242, 231, 204),
    case_outline_rgb=(91, 101, 112),
    screen_fill_rgb=(251, 246, 226),
    screen_outline_rgb=(154, 146, 124),
    text_rgb=(31, 43, 54),
    glow_rgb=(194, 203, 211),
    shadow_rgb=(209, 205, 194),
  ),
  "blue_lcd": _DigitalDisplayPalette(
    case_fill_rgb=(196, 214, 224),
    case_outline_rgb=(68, 101, 124),
    screen_fill_rgb=(220, 241, 245),
    screen_outline_rgb=(128, 163, 174),
    text_rgb=(19, 54, 76),
    glow_rgb=(160, 212, 224),
    shadow_rgb=(196, 203, 207),
  ),
  "wine_rose": _DigitalDisplayPalette(
    case_fill_rgb=(58, 22, 44),
    case_outline_rgb=(164, 84, 122),
    screen_fill_rgb=(31, 10, 25),
    screen_outline_rgb=(104, 57, 82),
    text_rgb=(255, 225, 237),
    glow_rgb=(224, 113, 158),
    shadow_rgb=(213, 201, 208),
  ),
  "forest_lime": _DigitalDisplayPalette(
    case_fill_rgb=(29, 50, 42),
    case_outline_rgb=(103, 154, 74),
    screen_fill_rgb=(15, 32, 25),
    screen_outline_rgb=(70, 105, 84),
    text_rgb=(222, 255, 179),
    glow_rgb=(154, 220, 88),
    shadow_rgb=(203, 214, 204),
  ),
  "plum_ice": _DigitalDisplayPalette(
    case_fill_rgb=(45, 36, 70),
    case_outline_rgb=(129, 114, 184),
    screen_fill_rgb=(25, 20, 44),
    screen_outline_rgb=(85, 79, 125),
    text_rgb=(232, 236, 255),
    glow_rgb=(154, 172, 246),
    shadow_rgb=(205, 205, 218),
  ),
}


@dataclass(frozen=True)
class _TaskDefaults:
  """Stable fallback defaults for clock-match panels."""

  hour_min: int = 1
  hour_max: int = 12
  minute_min: int = 0
  minute_max: int = 55
  minute_step: int = 5
  min_hand_angle_gap_deg: float = 10.0
  min_option_gap_minutes: int = 10
  option_label_support: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
  option_count: int = 6
  canvas_width: int = 980
  canvas_height: int = 760
  outer_margin_px: int = 40
  face_radius_px: int = 90
  bezel_width_px: int = 8
  numeral_font_size_px: int = 17
  major_tick_length_px: int = 13
  minor_tick_length_px: int = 6
  major_tick_width_px: int = 3
  minor_tick_width_px: int = 2
  minor_tick_dot_radius_px: int = 2
  hour_hand_width_px: int = 8
  minute_hand_width_px: int = 6
  second_hand_width_px: int = 2
  hand_bbox_padding_px: int = 5
  center_dot_radius_px: int = 6
  inner_ring_inset_px: int = 13
  inner_ring_width_px: int = 3
  reference_clock_radius_px: int = 136
  option_clock_radius_px: int = 76
  label_font_size_px: int = 28
  panel_title_font_size_px: int = 23
  digital_font_size_px: int = 58
  option_digital_font_size_px: int = 42
  digital_corner_radius_px: int = 18


@dataclass(frozen=True)
class _ResolvedQuery:
  """Resolved semantic and visual support for one clock-match instance."""

  query_id: str
  reference_representation: str
  option_representation: str
  scene_variant: str
  style_variant: str
  accent_color_name: str
  digital_display_palette: str
  target_total_minutes: int
  target_time_text: str
  option_labels: Tuple[str, ...]
  correct_label: str
  option_total_minutes_by_label: Dict[str, int]
  hour_support: Tuple[int, int]
  minute_support: Tuple[int, int, int]
  min_hand_angle_gap_deg: float
  min_option_gap_minutes: int
  query_id_probabilities: Dict[str, float]
  scene_variant_probabilities: Dict[str, float]
  style_variant_probabilities: Dict[str, float]
  accent_color_name_probabilities: Dict[str, float]
  digital_display_palette_probabilities: Dict[str, float]


_DEFAULTS = _TaskDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
  DOMAIN,
  SCENE_ID,
  task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_symbolic_background_defaults(scene_id="clock")
POST_IMAGE_NOISE_DEFAULTS = {
  **load_symbolic_noise_defaults(scene_id="clock", apply_prob=0.4),
  "apply_prob": 0.4,
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


def _resolve_option_labels(params: Mapping[str, Any]) -> Tuple[str, ...]:
  """Resolve the fixed visible option labels."""

  raw_labels = params.get("option_label_support", group_default(_GEN_DEFAULTS, "option_label_support", _DEFAULTS.option_label_support))
  labels = tuple(str(label).strip() for label in raw_labels if str(label).strip())
  option_count = int(params.get("option_count", group_default(_GEN_DEFAULTS, "option_count", _DEFAULTS.option_count)))
  if option_count != 6:
    raise ValueError("clock match panel currently requires exactly 6 visual options")
  if len(labels) != int(option_count):
    raise ValueError("clock match panel requires exactly six option labels")
  if len(set(labels)) != len(labels):
    raise ValueError("clock match panel option labels must be unique")
  return tuple(str(label) for label in labels)


def _cyclic_minute_gap(first_total: int, second_total: int) -> int:
  """Return the smaller gap between two 12-hour clock times."""

  raw = abs(int(first_total) - int(second_total)) % (12 * 60)
  return int(min(raw, (12 * 60) - raw))


def _feasible_time_support(
  *,
  hour_support: Tuple[int, ...],
  minute_support: Tuple[int, ...],
  min_hand_angle_gap_deg: float,
) -> Tuple[int, ...]:
  """Return all feasible times after hand-angle filtering."""

  return tuple(
    clock_total_minutes(int(hour), int(minute))
    for hour in hour_support
    for minute in minute_support
    if float(clock_hand_angle_gap_deg(clock_total_minutes(int(hour), int(minute)))) >= float(min_hand_angle_gap_deg)
  )


def _choose_option_times(
  *,
  rng,
  support: Tuple[int, ...],
  target_total: int,
  labels: Tuple[str, ...],
  correct_label: str,
  min_gap_minutes: int,
) -> Dict[str, int]:
  """Return distinct option times with the target placed under `correct_label`."""

  distractor_pool = [
    int(value)
    for value in support
    if int(value) != int(target_total) and _cyclic_minute_gap(int(value), int(target_total)) >= int(min_gap_minutes)
  ]
  if len(distractor_pool) < len(labels) - 1:
    raise ValueError("clock match panel has too few distractor times under the configured gap")
  sampled_distractors = [int(value) for value in rng.sample(distractor_pool, k=len(labels) - 1)]
  option_times: Dict[str, int] = {}
  distractor_index = 0
  for label in labels:
    if str(label) == str(correct_label):
      option_times[str(label)] = int(target_total)
      continue
    option_times[str(label)] = int(sampled_distractors[distractor_index])
    distractor_index += 1
  return option_times


def _resolve_query(instance_seed: int, *, params: Mapping[str, Any]) -> _ResolvedQuery:
  """Resolve one concrete clock-match panel query."""

  query_id, query_id_probabilities, task_params = select_task_query_id(
    instance_seed=int(instance_seed),
    params=params,
    supported_query_ids=SUPPORTED_QUERY_IDS,
    default_query_id="analog_reference_digital_options",
    task_id=TASK_ID,
    namespace=f"{TASK_ID}.query",
  )
  reference_representation, option_representation = _REPRESENTATION_BY_QUERY_ID[str(query_id)]
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
  digital_display_palette, digital_display_palette_probabilities = _resolve_named_variant(
    instance_seed=int(instance_seed),
    params=task_params,
    explicit_key="digital_display_palette",
    weights_key="digital_display_palette_weights",
    balance_flag_key="balanced_digital_display_palette_sampling",
    supported=SUPPORTED_DIGITAL_DISPLAY_PALETTES,
    namespace="digital_display_palette",
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
  min_option_gap_minutes = int(
    task_params.get(
      "min_option_gap_minutes",
      group_default(_GEN_DEFAULTS, "min_option_gap_minutes", _DEFAULTS.min_option_gap_minutes),
    )
  )
  if minute_step <= 0:
    raise ValueError("minute_step must be positive for clock match panel")
  if float(min_hand_angle_gap_deg) < 0.0:
    raise ValueError("min_hand_angle_gap_deg must be non-negative for clock match panel")
  if int(min_option_gap_minutes) < int(minute_step):
    raise ValueError("min_option_gap_minutes must be at least one minute step")

  hour_support = tuple(range(int(hour_min), int(hour_max) + 1))
  minute_support = tuple(range(int(minute_min), int(minute_max) + 1, int(minute_step)))
  if not hour_support or not minute_support:
    raise ValueError("clock match panel time support is empty")
  if hour_support[0] < 1 or hour_support[-1] > 12:
    raise ValueError("hour support must stay within 1..12 for clock match panel")
  if minute_support[0] < 0 or minute_support[-1] > 59:
    raise ValueError("minute support must stay within 0..59 for clock match panel")

  time_support = _feasible_time_support(
    hour_support=hour_support,
    minute_support=minute_support,
    min_hand_angle_gap_deg=float(min_hand_angle_gap_deg),
  )
  if len(time_support) < 6:
    raise ValueError("clock match panel needs at least six feasible times")

  explicit_total = task_params.get("target_total_minutes")
  explicit_hour = task_params.get("target_hour")
  explicit_minute = task_params.get("target_minute")
  if explicit_total is not None:
    target_total = int(explicit_total)
  elif explicit_hour is not None or explicit_minute is not None:
    if explicit_hour is None or explicit_minute is None:
      raise ValueError("target_hour and target_minute must be provided together")
    target_total = clock_total_minutes(int(explicit_hour), int(explicit_minute))
  else:
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.target_total")
    target_total, _target_probs = uniform_choice_with_probabilities(
      rng,
      time_support,
      sort_keys=True,
    )
    target_total = int(target_total)
  if int(target_total) not in time_support:
    raise ValueError("target time is outside configured feasible support for clock match panel")

  option_labels = _resolve_option_labels(task_params)
  explicit_answer_label = task_params.get("answer_label", task_params.get("correct_label"))
  if explicit_answer_label is not None:
    correct_label = str(explicit_answer_label).strip()
    if correct_label not in option_labels:
      raise ValueError("answer_label is outside clock match panel option labels")
  else:
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}.answer_label")
    correct_label, _label_probs = uniform_choice_with_probabilities(
      rng,
      option_labels,
      sort_keys=False,
    )
    correct_label = str(correct_label)

  rng = spawn_rng(int(instance_seed), f"{TASK_ID}.option_times")
  option_total_minutes_by_label = _choose_option_times(
    rng=rng,
    support=tuple(int(value) for value in time_support),
    target_total=int(target_total),
    labels=tuple(str(label) for label in option_labels),
    correct_label=str(correct_label),
    min_gap_minutes=int(min_option_gap_minutes),
  )

  return _ResolvedQuery(
    query_id=str(query_id),
    reference_representation=str(reference_representation),
    option_representation=str(option_representation),
    scene_variant=str(scene_variant),
    style_variant=str(style_variant),
    accent_color_name=str(accent_color_name),
    digital_display_palette=str(digital_display_palette),
    target_total_minutes=int(target_total),
    target_time_text=str(format_clock_hhmm(int(target_total))),
    option_labels=tuple(str(label) for label in option_labels),
    correct_label=str(correct_label),
    option_total_minutes_by_label={str(key): int(value) for key, value in option_total_minutes_by_label.items()},
    hour_support=(int(hour_support[0]), int(hour_support[-1])),
    minute_support=(int(minute_support[0]), int(minute_support[-1]), int(minute_step)),
    min_hand_angle_gap_deg=float(min_hand_angle_gap_deg),
    min_option_gap_minutes=int(min_option_gap_minutes),
    query_id_probabilities=dict(query_id_probabilities),
    scene_variant_probabilities=dict(scene_variant_probabilities),
    style_variant_probabilities=dict(style_variant_probabilities),
    accent_color_name_probabilities=dict(accent_color_name_probabilities),
    digital_display_palette_probabilities=dict(digital_display_palette_probabilities),
  )


def _label_bbox(
  *,
  draw: ImageDraw.ImageDraw,
  text: str,
  center: Tuple[float, float],
  font,
) -> Tuple[float, float, float, float]:
  """Draw centered text and return its bbox."""

  bbox = draw.textbbox((0, 0), str(text), font=font)
  width = float(bbox[2] - bbox[0])
  height = float(bbox[3] - bbox[1])
  xy = (float(center[0]) - (0.5 * width), float(center[1]) - (0.5 * height))
  draw_traced_text(
    draw,
    xy=xy,
    text=str(text),
    font=font,
    fill_rgb=(35, 40, 48),
    stroke_width=0,
    role="option_label",
    required=False,
  )
  return (float(xy[0]), float(xy[1]), float(xy[0] + width), float(xy[1] + height))


def _digital_display_palette(name: str) -> _DigitalDisplayPalette:
  """Return one supported digital display palette."""

  try:
    return _DIGITAL_DISPLAY_PALETTES[str(name)]
  except KeyError as exc:
    raise ValueError(f"unsupported digital display palette for {TASK_ID}: {name!r}") from exc


def _digital_display_palette_trace(palette: _DigitalDisplayPalette) -> Dict[str, List[int]]:
  """Return palette colors in a JSON-friendly trace shape."""

  return {
    "case_fill": [int(value) for value in palette.case_fill_rgb],
    "case_outline": [int(value) for value in palette.case_outline_rgb],
    "screen_fill": [int(value) for value in palette.screen_fill_rgb],
    "screen_outline": [int(value) for value in palette.screen_outline_rgb],
    "text": [int(value) for value in palette.text_rgb],
    "glow": [int(value) for value in palette.glow_rgb],
    "shadow": [int(value) for value in palette.shadow_rgb],
  }


def _draw_digital_display(
  draw: ImageDraw.ImageDraw,
  *,
  bbox: Tuple[float, float, float, float],
  time_text: str,
  font_size_px: int,
  corner_radius_px: int,
  palette: _DigitalDisplayPalette,
  font_family_scope: str,
) -> Tuple[float, float, float, float]:
  """Draw a digital display and return the display bbox."""

  del font_family_scope
  x0, y0, x1, y1 = (float(value) for value in bbox)
  shadow = (float(x0 + 3), float(y0 + 4), float(x1 + 3), float(y1 + 4))
  draw.rounded_rectangle(shadow, radius=int(corner_radius_px), fill=tuple(int(v) for v in palette.shadow_rgb))
  draw.rounded_rectangle(
    (x0, y0, x1, y1),
    radius=int(corner_radius_px),
    fill=tuple(int(v) for v in palette.case_fill_rgb),
    outline=tuple(int(v) for v in palette.case_outline_rgb),
    width=3,
  )
  inner = (float(x0 + 12), float(y0 + 10), float(x1 - 12), float(y1 - 10))
  draw.rounded_rectangle(
    inner,
    radius=max(4, int(corner_radius_px) - 8),
    fill=tuple(int(v) for v in palette.screen_fill_rgb),
    outline=tuple(int(v) for v in palette.screen_outline_rgb),
    width=1,
  )
  font = load_font(int(font_size_px), bold=False)
  center = ((float(x0 + x1) / 2.0), (float(y0 + y1) / 2.0) + 1.0)
  for dx, dy in ((-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)):
    draw_text_centered(
      draw,
      text=str(time_text),
      center=(float(center[0] + dx), float(center[1] + dy)),
      font=font,
      fill=tuple(int(v) for v in palette.glow_rgb),
      stroke_width=1,
    )
  draw_text_centered(
    draw,
    text=str(time_text),
    center=center,
    font=font,
    fill=tuple(int(v) for v in palette.text_rgb),
    stroke_width=1,
  )
  return (float(x0), float(y0), float(x1), float(y1))


def _option_card_bboxes() -> Dict[str, Tuple[float, float, float, float]]:
  """Return fixed A-F option-card bboxes for the match panel."""

  labels = tuple(_DEFAULTS.option_label_support)
  card_w = 270.0
  card_h = 160.0
  gap_x = 36.0
  gap_y = 26.0
  start_x = 0.5 * (float(_DEFAULTS.canvas_width) - ((3.0 * card_w) + (2.0 * gap_x)))
  start_y = 390.0
  bboxes: Dict[str, Tuple[float, float, float, float]] = {}
  for index, label in enumerate(labels):
    row = int(index // 3)
    col = int(index % 3)
    x0 = float(start_x + (float(col) * (card_w + gap_x)))
    y0 = float(start_y + (float(row) * (card_h + gap_y)))
    bboxes[str(label)] = (float(x0), float(y0), float(x0 + card_w), float(y0 + card_h))
  return bboxes


def _reference_bbox_for(representation: str) -> Tuple[float, float, float, float]:
  """Return the top reference visual bbox for one representation."""

  if str(representation) == "analog":
    radius = float(_DEFAULTS.reference_clock_radius_px)
    cx = 0.5 * float(_DEFAULTS.canvas_width)
    cy = 210.0
    return (float(cx - radius), float(cy - radius), float(cx + radius), float(cy + radius))
  return (float(0.5 * _DEFAULTS.canvas_width - 175.0), 145.0, float(0.5 * _DEFAULTS.canvas_width + 175.0), 255.0)


def _visual_bbox_inside_card(card_bbox: Tuple[float, float, float, float], *, representation: str) -> Tuple[float, float, float, float]:
  """Return the option visual bbox inside one option card, excluding the label."""

  x0, y0, x1, y1 = (float(value) for value in card_bbox)
  if str(representation) == "analog":
    cx = 0.5 * (float(x0) + float(x1))
    cy = float(y0 + 84.0)
    radius = float(_DEFAULTS.option_clock_radius_px)
    return (float(cx - radius), float(cy - radius), float(cx + radius), float(cy + radius))
  return (float(x0 + 58.0), float(y0 + 52.0), float(x1 - 20.0), float(y1 - 30.0))


def _canonical_match_example_bbox() -> tuple[str, str]:
  """Return prompt JSON examples for the match-panel task."""

  answer_and_annotation = {
    "annotation": {
      "reference": [315, 74, 587, 346],
      "correct_option": [382, 442, 652, 602],
    },
    "answer": "B",
  }
  answer_only = {"answer": "B"}
  return (
    json.dumps(answer_and_annotation, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
    json.dumps(answer_only, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
  )


@register_task
class SymbolicClockMatchPanelTask:
  """Match one analog/digital reference clock to the equivalent visual option."""

  task_id = TASK_ID
  reasoning_operations = ('formula_evaluation', 'matching')
  domain = DOMAIN
  supported_query_ids = SUPPORTED_QUERY_IDS
  default_dataset_enabled = True

  def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
    """Generate one reference-to-option clock matching panel.

    This public task owns the modality branch, target time, correct option
    label, role-bound bbox annotation, prompt slots, and final output. Rendering
    happens only after the reference, options, and unique matching option are
    fixed.
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
    digital_palette = _digital_display_palette(str(query.digital_display_palette))
    font_family = sample_font_family(
      role="readout",
      instance_seed=int(instance_seed),
      namespace=f"{TASK_ID}.font",
      params={**dict(_RENDER_DEFAULTS), **dict(params)},
    )

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

    label_font_size_px = int(params.get("label_font_size_px", group_default(_RENDER_DEFAULTS, "label_font_size_px", _DEFAULTS.label_font_size_px)))
    panel_title_font_size_px = int(params.get("panel_title_font_size_px", group_default(_RENDER_DEFAULTS, "panel_title_font_size_px", _DEFAULTS.panel_title_font_size_px)))
    digital_font_size_px = int(params.get("digital_font_size_px", group_default(_RENDER_DEFAULTS, "digital_font_size_px", _DEFAULTS.digital_font_size_px)))
    option_digital_font_size_px = int(params.get("option_digital_font_size_px", group_default(_RENDER_DEFAULTS, "option_digital_font_size_px", _DEFAULTS.option_digital_font_size_px)))
    digital_corner_radius_px = int(params.get("digital_corner_radius_px", group_default(_RENDER_DEFAULTS, "digital_corner_radius_px", _DEFAULTS.digital_corner_radius_px)))
    reference_clock_radius_px = int(params.get("reference_clock_radius_px", group_default(_RENDER_DEFAULTS, "reference_clock_radius_px", _DEFAULTS.reference_clock_radius_px)))
    option_clock_radius_px = int(params.get("option_clock_radius_px", group_default(_RENDER_DEFAULTS, "option_clock_radius_px", _DEFAULTS.option_clock_radius_px)))

    scene_entities: List[Dict[str, Any]] = []
    reference_visual_bbox = _reference_bbox_for(str(query.reference_representation))
    option_card_bboxes = _option_card_bboxes()
    option_visual_bboxes: Dict[str, List[float]] = {}
    option_card_bboxes_out: Dict[str, List[float]] = {}

    with temporary_default_font_family(str(font_family)):
      title_font = load_font(int(panel_title_font_size_px), bold=True)
      option_label_font = load_font(int(label_font_size_px), bold=True)
      _label_bbox(
        draw=draw,
        text="Reference",
        center=(float(render_params.canvas_width) / 2.0, 48.0),
        font=title_font,
      )

      if str(query.reference_representation) == "analog":
        reference_params = scale_clock_render_params_for_radius(render_params, radius_px=int(reference_clock_radius_px))
        ref_center = (float(render_params.canvas_width) / 2.0, 210.0)
        ref_geometry = draw_clock_geometry(
          image,
          center_px=ref_center,
          face_radius_px=float(reference_clock_radius_px),
          scene_variant=str(query.scene_variant),
          shown_total_minutes=int(query.target_total_minutes),
          render_params=reference_params,
          visual_theme=clock_theme,
          entity_prefix="reference_",
          extra_face_attrs={
            "role": "reference",
            "representation": "analog",
            "shown_time_text": str(query.target_time_text),
          },
        )
        reference_visual_bbox = tuple(float(value) for value in ref_geometry.face_bbox_px)
        scene_entities.extend([dict(entity) for entity in ref_geometry.entities])
      else:
        reference_visual_bbox = _draw_digital_display(
          draw,
          bbox=_reference_bbox_for("digital"),
          time_text=str(query.target_time_text),
          font_size_px=int(digital_font_size_px),
          corner_radius_px=int(digital_corner_radius_px),
          palette=digital_palette,
          font_family_scope="reference_digital",
        )
        scene_entities.append(
          {
            "entity_id": "reference_digital_display",
            "entity_kind": "digital_clock_display",
            "bbox_px": [round(float(value), 3) for value in reference_visual_bbox],
            "role": "reference",
            "shown_time_text": str(query.target_time_text),
          }
        )

      for label in query.option_labels:
        card_bbox = option_card_bboxes[str(label)]
        option_card_bboxes_out[str(label)] = [round(float(value), 3) for value in card_bbox]
        draw.rounded_rectangle(card_bbox, radius=18, fill=(252, 252, 250), outline=(172, 178, 188), width=2)
        draw.rounded_rectangle((card_bbox[0] + 8, card_bbox[1] + 8, card_bbox[0] + 42, card_bbox[1] + 42), radius=10, fill=(42, 48, 56))
        draw_text_centered(
          draw,
          text=str(label),
          center=(float(card_bbox[0] + 25), float(card_bbox[1] + 25)),
          font=option_label_font,
          fill=(248, 250, 252),
        )
        option_time = int(query.option_total_minutes_by_label[str(label)])
        option_text = str(format_clock_hhmm(int(option_time)))
        visual_bbox = _visual_bbox_inside_card(card_bbox, representation=str(query.option_representation))
        if str(query.option_representation) == "analog":
          option_params = scale_clock_render_params_for_radius(render_params, radius_px=int(option_clock_radius_px))
          center = ((float(visual_bbox[0]) + float(visual_bbox[2])) / 2.0, (float(visual_bbox[1]) + float(visual_bbox[3])) / 2.0)
          geometry = draw_clock_geometry(
            image,
            center_px=center,
            face_radius_px=float(option_clock_radius_px),
            scene_variant=str(query.scene_variant),
            shown_total_minutes=int(option_time),
            render_params=option_params,
            visual_theme=clock_theme,
            entity_prefix=f"option_{str(label).lower()}_",
            extra_face_attrs={
              "role": "option",
              "option_label": str(label),
              "representation": "analog",
              "shown_time_text": str(option_text),
              "is_correct": bool(str(label) == str(query.correct_label)),
            },
          )
          visual_bbox = tuple(float(value) for value in geometry.face_bbox_px)
          scene_entities.extend([dict(entity) for entity in geometry.entities])
        else:
          visual_bbox = _draw_digital_display(
            draw,
            bbox=visual_bbox,
            time_text=str(option_text),
            font_size_px=int(option_digital_font_size_px),
            corner_radius_px=max(8, int(digital_corner_radius_px) - 6),
            palette=digital_palette,
            font_family_scope=f"option_{str(label).lower()}_digital",
          )
          scene_entities.append(
            {
              "entity_id": f"option_{str(label).lower()}_digital_display",
              "entity_kind": "digital_clock_display",
              "bbox_px": [round(float(value), 3) for value in visual_bbox],
              "role": "option",
              "option_label": str(label),
              "shown_time_text": str(option_text),
              "is_correct": bool(str(label) == str(query.correct_label)),
            }
          )
        option_visual_bboxes[str(label)] = [round(float(value), 3) for value in visual_bbox]

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
        "object_description_equivalent_time_label_analog_reference_digital_options",
        "object_description_equivalent_time_label_digital_reference_analog_options",
        "annotation_hint_equivalent_time_label",
        "answer_hint_equivalent_time_label",
      ),
      context=f"prompt defaults for {self.task_id}",
    )
    object_description = str(prompt_defaults[f"object_description_equivalent_time_label_{str(query.query_id)}"])
    json_example, json_example_answer_only = _canonical_match_example_bbox()
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
        "annotation_hint": str(prompt_defaults["annotation_hint_equivalent_time_label"]),
        "answer_hint": str(prompt_defaults["answer_hint_equivalent_time_label"]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
      },
      instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

    annotation_bboxes = {
      "reference": [round(float(value), 3) for value in reference_visual_bbox],
      "correct_option": list(option_visual_bboxes[str(query.correct_label)]),
    }
    answer_gt = TypedValue(type="string", value=str(query.correct_label))
    annotation_gt = TypedValue(type="bbox_map", value=dict(annotation_bboxes))
    scene_bbox = (
      min(float(reference_visual_bbox[0]), *(float(box[0]) for box in option_card_bboxes.values())),
      min(float(reference_visual_bbox[1]), *(float(box[1]) for box in option_card_bboxes.values())),
      max(float(reference_visual_bbox[2]), *(float(box[2]) for box in option_card_bboxes.values())),
      max(float(reference_visual_bbox[3]), *(float(box[3]) for box in option_card_bboxes.values())),
    )
    target_hour, target_minute = split_clock_total_minutes(int(query.target_total_minutes))

    trace_payload = {
      "scene_ir": {
        "scene_kind": "symbolic_clock_equivalent_time_panel",
        "entities": [dict(entity) for entity in scene_entities],
        "relations": {
          "query_id": str(query.query_id),
          "reference_representation": str(query.reference_representation),
          "option_representation": str(query.option_representation),
          "digital_display_palette": str(query.digital_display_palette),
          "correct_label": str(query.correct_label),
          "target_time_text": str(query.target_time_text),
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
          "reference_representation": str(query.reference_representation),
          "option_representation": str(query.option_representation),
          "scene_variant": str(query.scene_variant),
          "style_variant": str(query.style_variant),
          "accent_color_name": str(query.accent_color_name),
          "digital_display_palette": str(query.digital_display_palette),
          "query_id_probabilities": dict(query.query_id_probabilities),
          "scene_variant_probabilities": dict(query.scene_variant_probabilities),
          "style_variant_probabilities": dict(query.style_variant_probabilities),
          "accent_color_name_probabilities": dict(query.accent_color_name_probabilities),
          "digital_display_palette_probabilities": dict(query.digital_display_palette_probabilities),
          "option_labels": [str(label) for label in query.option_labels],
          "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
          "minute_support": [int(value) for value in query.minute_support],
          "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
          "min_option_gap_minutes": int(query.min_option_gap_minutes),
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
          "digital_display_palette": str(query.digital_display_palette),
          "reference_clock_radius_px": int(reference_clock_radius_px),
          "option_clock_radius_px": int(option_clock_radius_px),
          "digital_font_size_px": int(digital_font_size_px),
          "option_digital_font_size_px": int(option_digital_font_size_px),
          "font": {
            "source": "global_font_pool",
            "font_family": str(font_family),
            "font_asset_version": font_asset_version(),
            "scope": "clock_equivalent_time_reference_and_options",
          },
          "resolved_colors_rgb": {
            "face_fill": [int(value) for value in clock_theme.face_fill_rgb],
            "face_outline": [int(value) for value in clock_theme.face_outline_rgb],
            "numerals": [int(value) for value in clock_theme.numeral_color_rgb],
            "ticks": [int(value) for value in clock_theme.tick_color_rgb],
            "hour_hand": [int(value) for value in clock_theme.hour_hand_color_rgb],
            "minute_hand": [int(value) for value in clock_theme.minute_hand_color_rgb],
            "center_dot": [int(value) for value in clock_theme.center_dot_color_rgb],
          },
          "digital_display_colors_rgb": _digital_display_palette_trace(digital_palette),
          "minor_tick_mode": str(clock_theme.minor_tick_mode),
        },
      },
      "render_map": {
        "image_id": "img0",
        "scene_bbox_px": [round(float(value), 3) for value in scene_bbox],
        "reference_bbox_px": [round(float(value), 3) for value in reference_visual_bbox],
        "option_visual_bboxes_px": dict(option_visual_bboxes),
        "option_card_bboxes_px": dict(option_card_bboxes_out),
        "correct_option_bbox_px": list(option_visual_bboxes[str(query.correct_label)]),
        "correct_label": str(query.correct_label),
      },
      "execution_trace": {
        "query_id": str(query.query_id),
        "reference_representation": str(query.reference_representation),
        "option_representation": str(query.option_representation),
        "scene_variant": str(query.scene_variant),
        "style_variant": str(query.style_variant),
        "accent_color_name": str(query.accent_color_name),
        "digital_display_palette": str(query.digital_display_palette),
        "target_total_minutes": int(query.target_total_minutes),
        "target_hour": int(target_hour),
        "target_minute": int(target_minute),
        "target_time_text": str(query.target_time_text),
        "option_labels": [str(label) for label in query.option_labels],
        "correct_label": str(query.correct_label),
        "option_total_minutes_by_label": {str(key): int(value) for key, value in query.option_total_minutes_by_label.items()},
        "option_time_text_by_label": {
          str(label): str(format_clock_hhmm(int(query.option_total_minutes_by_label[str(label)])))
          for label in query.option_labels
        },
        "hour_support": [int(query.hour_support[0]), int(query.hour_support[1])],
        "minute_support": [int(value) for value in query.minute_support],
        "min_hand_angle_gap_deg": float(query.min_hand_angle_gap_deg),
        "min_option_gap_minutes": int(query.min_option_gap_minutes),
        "query_id_probabilities": dict(query.query_id_probabilities),
        "scene_variant_probabilities": dict(query.scene_variant_probabilities),
        "style_variant_probabilities": dict(query.style_variant_probabilities),
        "accent_color_name_probabilities": dict(query.accent_color_name_probabilities),
        "digital_display_palette_probabilities": dict(query.digital_display_palette_probabilities),
        "question_format": str(query.query_id),
        "supporting_bbox_roles": ["reference", "correct_option"],
      },
      "witness_symbolic": {
        "type": "bbox_map",
        "value": dict(annotation_bboxes),
      },
      "projected_annotation": {
        "type": "bbox_map",
        "bbox_map": dict(annotation_bboxes),
        "pixel_bbox_map": dict(annotation_bboxes),
        "value": dict(annotation_bboxes),
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
      scene_id=SCENE_ID,
      query_id=str(query.query_id),
      prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = ["SymbolicClockMatchPanelTask", "SUPPORTED_DIGITAL_DISPLAY_PALETTES"]
