"""Shared constants and data containers for infographic metric-card page tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults
from trace_tasks.tasks.pages.shared.page_semantic_assets import page_semantic_asset_ids
from trace_tasks.tasks.pages.shared.visual_defaults import load_pages_background_defaults, load_pages_noise_defaults

TASK_ID = "pages_infographic_metric_arithmetic_value_base"
SCENE_ID = "infographic"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "sum_named_metrics",
    "section_extrema_arithmetic",
    "section_total_extrema_difference",
    "section_total_except_named",
)
SECTION_RANKED_TOTAL_VARIANTS: Tuple[str, ...] = (
    "section_ranked_total_label",
)
FILTERED_METRIC_TOTAL_VARIANTS: Tuple[str, ...] = (
    "section_icon_total_value",
)
COLUMN_PROFILE_COMPARISON_VARIANTS: Tuple[str, ...] = (
    "section_icon_total_difference_value",
)
FILTERED_SECTION_EXTREMUM_VARIANTS: Tuple[str, ...] = (
    "section_icon_extremum_label",
)
GLOBAL_METRIC_RANKED_ITEM_VARIANTS: Tuple[str, ...] = (
    "nth_highest_metric_label",
    "nth_lowest_metric_label",
)
SECTION_METRIC_RANKED_ITEM_VARIANTS: Tuple[str, ...] = (
    "nth_highest_metric_in_section_label",
    "nth_lowest_metric_in_section_label",
)
METRIC_RANKED_ITEM_VARIANTS: Tuple[str, ...] = (
    *GLOBAL_METRIC_RANKED_ITEM_VARIANTS,
    *SECTION_METRIC_RANKED_ITEM_VARIANTS,
)
ALL_QUERY_IDS: Tuple[str, ...] = (
    *SUPPORTED_QUERY_IDS,
    *SECTION_RANKED_TOTAL_VARIANTS,
    *FILTERED_METRIC_TOTAL_VARIANTS,
    *COLUMN_PROFILE_COMPARISON_VARIANTS,
    *FILTERED_SECTION_EXTREMUM_VARIANTS,
    *METRIC_RANKED_ITEM_VARIANTS,
)
_TASK_GROUP_DEFAULTS = get_scene_defaults("pages", "infographic")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)
POST_IMAGE_BACKGROUND_DEFAULTS = load_pages_background_defaults(scene_id="infographic")
POST_IMAGE_NOISE_DEFAULTS = load_pages_noise_defaults(scene_id="infographic", apply_prob=0.0)

_ICON_KINDS: Tuple[str, ...] = page_semantic_asset_ids(semantic_role="metric_icon", allowed_use="filter")
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (52, 116, 170),
    (39, 139, 112),
    (196, 89, 73),
    (132, 101, 184),
    (202, 143, 56),
    (74, 132, 92),
    (60, 105, 142),
    (172, 85, 128),
)
_REASONING_LOAD_BY_VARIANT: Dict[str, float] = {
    "sum_named_metrics": 0.72,
    "section_extrema_arithmetic": 0.88,
    "section_total_extrema_difference": 1.00,
    "section_total_except_named": 1.00,
    "section_ranked_total_label": 0.92,
    "section_icon_total_value": 0.78,
    "section_icon_total_difference_value": 0.86,
    "section_icon_extremum_label": 0.94,
    "nth_highest_metric_label": 0.70,
    "nth_lowest_metric_label": 0.70,
    "nth_highest_metric_in_section_label": 0.66,
    "nth_lowest_metric_in_section_label": 0.66,
}
_EXTREMUM_KINDS: Tuple[str, ...] = ("maximum", "minimum")
_EXTREMA_OPERATIONS: Tuple[str, ...] = ("sum", "absolute_difference")
_RANK_DIRECTIONS: Tuple[str, ...] = ("highest", "lowest")
_RANK_POSITION_SUPPORT: Tuple[int, ...] = (2, 3)
_RANK_ORDINALS: Dict[int, str] = {
    1: "highest",
    2: "second",
    3: "third",
}
_INFOGRAPHIC_STYLE_VARIANTS: Tuple[str, ...] = (
    "card_wall",
    "kpi_dashboard",
    "staggered_mosaic",
    "column_fact_sheet",
    "radial_spokes",
    "circular_sections",
)
_STYLE_TITLE_COPY: Dict[str, Tuple[str, str]] = {
    "card_wall": ("Operational Metrics", "Arithmetic from printed values"),
    "kpi_dashboard": ("Program Snapshot", "KPI cards grouped by section"),
    "staggered_mosaic": ("Metrics Bulletin", "Grouped figures with section totals"),
    "column_fact_sheet": ("Field Report", "Column-style metric summary"),
    "radial_spokes": ("Signal Wheel", "Metric groups arranged by section"),
    "circular_sections": ("Metric Orbit", "Section groups arranged in rings"),
}


@dataclass(frozen=True)
class _MetricCard:
    card_id: str
    label: str
    value: int
    display_text: str
    unit: str
    section: str
    icon_kind: str
    color_rgb: Tuple[int, int, int]
    caption_number: int
    caption_text: str


@dataclass(frozen=True)
class _RenderedInfographic:
    image: Image.Image
    entities: List[Dict[str, Any]]
    card_traces: List[Dict[str, Any]]
    page_bbox: List[float]
    document_title_bbox: List[float]
    document_subtitle_bbox: List[float]
    section_bboxes: Dict[str, List[float]]
    section_title_bboxes: Dict[str, List[float]]
    layout_jitter_meta: Dict[str, Any]


def _quote_label(label: str) -> str:
    return f'"{str(label)}"'


def _quoted_label_list(labels: Sequence[str]) -> str:
    quoted = [_quote_label(str(label)) for label in labels]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return f"{', '.join(quoted[:-1])}, and {quoted[-1]}"


def _labels_by_section(
    *,
    labels: Sequence[str],
    section_titles: Sequence[str],
    section_card_counts: Sequence[int],
) -> Dict[str, List[str]]:
    """Return metric labels grouped by their rendered section."""

    grouped: Dict[str, List[str]] = {}
    label_index = 0
    for section_index, section_title in enumerate(section_titles):
        section_labels: List[str] = []
        for _ in range(int(section_card_counts[section_index])):
            section_labels.append(str(labels[label_index]))
            label_index += 1
        grouped[str(section_title)] = section_labels
    return grouped


def _section_totals(
    labels_by_section: Mapping[str, Sequence[str]],
    values_by_label: Mapping[str, int],
) -> Dict[str, int]:
    """Compute section totals from the same label-value map used for answers."""

    return {
        str(section): int(sum(int(values_by_label[str(label)]) for label in labels))
        for section, labels in labels_by_section.items()
    }


def _adjust_value_to_break_tie(
    values_by_label: Dict[str, int],
    *,
    label: str,
    value_min: int,
    value_max: int,
) -> None:
    """Adjust one metric by one step while staying inside the configured value range."""

    current = int(values_by_label[str(label)])
    if current < int(value_max):
        values_by_label[str(label)] = current + 1
    elif current > int(value_min):
        values_by_label[str(label)] = current - 1
    else:
        raise ValueError(f"cannot adjust tied value for label {label!r} inside {value_min}..{value_max}")


def _adjust_section_total(
    values_by_label: Dict[str, int],
    labels: Sequence[str],
    *,
    delta: int,
    value_min: int,
    value_max: int,
) -> bool:
    """Move one section total by a small integer delta while preserving value bounds."""

    direction = 1 if int(delta) > 0 else -1
    for _ in range(abs(int(delta))):
        adjusted = False
        for label in labels:
            current = int(values_by_label[str(label)])
            if direction > 0 and current < int(value_max):
                values_by_label[str(label)] = current + 1
                adjusted = True
                break
            if direction < 0 and current > int(value_min):
                values_by_label[str(label)] = current - 1
                adjusted = True
                break
        if not adjusted:
            return False
    return True


def _partition_cards(card_count: int, section_count: int) -> List[int]:
    base = int(card_count) // int(section_count)
    remainder = int(card_count) % int(section_count)
    return [int(base + (1 if index < remainder else 0)) for index in range(int(section_count))]
