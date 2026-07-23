"""Passive state and constants for rule-override board scene-package tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


SCENE_ID = "rule_override_board"
SCENE_NAMESPACE = "games.rule_override_board"

LINE_BOARD_FAMILY = "line"
PIECE_BOARD_FAMILY = "piece"
WIN_RESULT = "win"
LOSS_RESULT = "loss"

SUPPORTED_BOARD_STYLES: Tuple[str, ...] = ("classic", "paper", "chalkboard", "arcade", "wood")
LINE_PLAYERS: Tuple[str, ...] = ("X", "O")
PIECE_PLAYERS: Tuple[str, ...] = ("Black", "White")


@dataclass(frozen=True)
class RuleOverrideAxes:
    """Resolved visual and construction axes shared by public tasks."""

    board_family: str
    board_style: str
    target_player: str
    board_count: int
    board_size: int
    target_answer: int
    board_style_probabilities: Dict[str, float]
    target_player_probabilities: Dict[str, float]
    board_count_probabilities: Dict[str, float]
    board_size_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RuleOverrideRenderParams:
    """Resolved render dimensions and text settings for one sample."""

    canvas_width: int
    canvas_height: int
    cell_size_px: int
    board_gap_px: int
    board_padding_px: int
    board_label_height_px: int
    content_margin_px: int
    panel_radius_px: int
    panel_border_width_px: int
    board_border_width_px: int
    grid_width_px: int
    board_label_font_size_px: int
    mark_font_size_px: int
    font_family: str
    layout_jitter_meta: Dict[str, Any]
    unit_size_meta: Dict[str, Any]


@dataclass(frozen=True)
class BoardPanel:
    """One rendered mini-board and its symbolic outcome."""

    board_id: str
    label: str
    cells: Tuple[Tuple[str, ...], ...]
    counted: bool
    result: str
    target_player: str
    target_stat: int
    opponent_stat: int


@dataclass(frozen=True)
class RuleOverrideSceneSample:
    """One sampled panel scene plus task-bound symbolic witnesses."""

    board_family: str
    board_style: str
    target_player: str
    board_size: int
    answer: int
    rule_text: str
    boards: Tuple[BoardPanel, ...]
    annotation_entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RenderedRuleOverrideScene:
    """Rendered scene plus projection metadata."""

    image: Image.Image
    render_map: Dict[str, Any]
    scene_entities: Tuple[Dict[str, Any], ...]


__all__ = [
    "BoardPanel",
    "LINE_BOARD_FAMILY",
    "LINE_PLAYERS",
    "LOSS_RESULT",
    "PIECE_BOARD_FAMILY",
    "PIECE_PLAYERS",
    "RenderedRuleOverrideScene",
    "RuleOverrideAxes",
    "RuleOverrideRenderParams",
    "RuleOverrideSceneSample",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_BOARD_STYLES",
    "WIN_RESULT",
]
