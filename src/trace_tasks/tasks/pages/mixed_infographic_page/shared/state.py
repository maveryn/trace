"""Shared model types and semantic constants for mixed infographic pages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ...shared.page_visual_assets import PageVisualAssetSelection


@dataclass(frozen=True)
class _MixedField:
    field_id: str
    label: str


@dataclass(frozen=True)
class _MixedItem:
    item_id: str
    label: str
    visual_asset_selection: PageVisualAssetSelection
    values_by_field_id: Dict[str, str]


@dataclass(frozen=True)
class _MixedModule:
    module_id: str
    kind: str
    title: str
    accent_rgb: Tuple[int, int, int]
    section_asset_selection: PageVisualAssetSelection
    fields: Tuple[_MixedField, ...]
    items: Tuple[_MixedItem, ...]


@dataclass(frozen=True)
class _InfographicTextBlock:
    block_id: str
    kind: str
    text: str
    placement_region: str
    font_role: str


@dataclass(frozen=True)
class _MixedInfographicSpec:
    title: str
    subtitle: str
    hero_asset_selection: PageVisualAssetSelection
    modules: Tuple[_MixedModule, ...]
    text_blocks: Tuple[_InfographicTextBlock, ...]
    text_resource_metadata: Dict[str, Any]


_TEXT_BLOCK_KIND_ORDER: Tuple[str, ...] = (
    "paragraph_note",
    "summary_note",
    "caption_strip",
    "source_line",
    "callout_quote",
    "badge_note",
)
MODULE_KINDS: Tuple[str, ...] = (
    "fact_grid",
    "profile_cards",
    "icon_metric_list",
    "comparison_strip",
    "ranked_list",
    "timeline_snippet",
    "callout_stats",
    "mini_table",
    "radial_bubbles",
    "ring_summary",
)
_FIELD_VALUE_BANKS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Reach", ("18k", "24k", "31k", "39k", "46k", "52k", "68k", "74k", "83k", "91k")),
    ("Rate", ("12%", "18%", "23%", "27%", "34%", "41%", "48%", "56%", "63%", "72%")),
    ("Score", ("54", "61", "67", "73", "79", "84", "88", "92", "96", "101")),
    ("Cost", ("$18", "$24", "$31", "$46", "$58", "$64", "$72", "$86", "$97", "$113")),
    ("Index", ("A14", "B27", "C35", "D48", "E52", "F69", "G74", "H83", "J91", "K06")),
    ("Status", ("Open", "Ready", "Pilot", "Active", "Paused", "Review", "Queued", "Live")),
    ("Zone", ("North", "South", "East", "West", "Central", "Harbor", "Uptown", "Riverside")),
    ("Level", ("Basic", "Core", "Plus", "Prime", "Select", "Elite", "Gold", "Platinum")),
    ("Window", ("Q1", "Q2", "Q3", "Q4", "Early", "Mid", "Late", "Night")),
    ("Rank", ("#1", "#2", "#3", "#4", "#5", "#6", "#7", "#8")),
    ("Count", ("14", "19", "22", "28", "33", "37", "42", "49", "55", "62")),
    ("Trend", ("Up", "Flat", "Down", "Mixed", "Rising", "Stable", "Cooling", "Shifting")),
)
NUMERIC_FIELD_LABELS: Tuple[str, ...] = ("Reach", "Rate", "Score", "Cost", "Count", "Rank")
CATEGORICAL_FIELD_LABELS: Tuple[str, ...] = ("Status", "Zone", "Level", "Window", "Trend")
ADDITIVE_FIELD_LABELS: Tuple[str, ...] = ("Score", "Count")
CONDITION_OPERATORS: Tuple[str, ...] = ("above", "below", "at_least")
RANK_ORDINALS: Dict[int, str] = {1: "first", 2: "second", 3: "third", 4: "fourth"}
_ACCENTS: Tuple[Tuple[int, int, int], ...] = (
    (59, 122, 177),
    (46, 142, 112),
    (190, 93, 78),
    (126, 102, 183),
    (196, 142, 61),
    (72, 132, 151),
    (166, 83, 116),
    (86, 139, 76),
)


__all__ = [
    "ADDITIVE_FIELD_LABELS",
    "CATEGORICAL_FIELD_LABELS",
    "CONDITION_OPERATORS",
    "MODULE_KINDS",
    "NUMERIC_FIELD_LABELS",
    "RANK_ORDINALS",
    "_ACCENTS",
    "_FIELD_VALUE_BANKS",
    "_InfographicTextBlock",
    "_MixedField",
    "_MixedInfographicSpec",
    "_MixedItem",
    "_MixedModule",
    "_TEXT_BLOCK_KIND_ORDER",
]
