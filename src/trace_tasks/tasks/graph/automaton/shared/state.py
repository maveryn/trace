"""Passive state contracts for the graph automaton scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image

from ...shared.graph_sample_types import GraphTopologySample
from ...shared.graph_scene import GraphRenderParams, RenderedGraphScene


SCENE_ID = "automaton"
AUTOMATON_SYMBOLS: Tuple[str, ...] = ("0", "1")
SUPPORTED_AUTOMATON_LAYOUT_VARIANTS: Tuple[str, ...] = (
    "circular",
    "shell",
    "layered",
    "path_spine",
    "spring",
)
OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEF")


@dataclass(frozen=True)
class AcceptanceAxes:
    """Resolved non-query axes for one automaton string-acceptance instance."""

    automaton_kind: str
    state_count: int
    input_length: int
    input_length_min: int
    input_length_max: int
    candidate_count: int
    candidate_count_support: Tuple[int, ...]
    answer_option_index: int
    layout_variant: str
    layout_transform_variant: str
    edge_routing_variant: str
    node_color_name: str
    state_count_probabilities: Dict[str, float]
    input_length_probabilities: Dict[str, float]
    candidate_count_probabilities: Dict[str, float]
    answer_option_probabilities: Dict[str, float]
    layout_variant_probabilities: Dict[str, float]
    layout_transform_variant_probabilities: Dict[str, float]
    edge_routing_variant_probabilities: Dict[str, float]
    node_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class AcceptanceSample:
    """One sampled automaton with exactly one accepted displayed option."""

    graph_sample: GraphTopologySample
    start_label: str
    accepting_labels: Tuple[str, ...]
    answer_option_label: str
    answer_input_string: str
    accepting_path_labels: Tuple[str, ...]
    candidate_strings_by_option: Dict[str, str]
    accepted_option_labels: Tuple[str, ...]
    transition_labels_by_edge: Dict[Tuple[str, str], str]
    transition_function: Dict[str, Dict[str, Tuple[str, ...]]]


@dataclass(frozen=True)
class AcceptanceRender:
    """Rendered acceptance scene before public task output binding."""

    render_params: GraphRenderParams
    rendered_scene: RenderedGraphScene
    image: Image.Image
    background_meta: Mapping[str, Any]
    post_noise_meta: Mapping[str, Any]
    option_bboxes: Mapping[str, list[int]]
    option_panel_meta: Mapping[str, Any]


__all__ = [
    "AUTOMATON_SYMBOLS",
    "OPTION_LABELS",
    "SCENE_ID",
    "SUPPORTED_AUTOMATON_LAYOUT_VARIANTS",
    "AcceptanceAxes",
    "AcceptanceRender",
    "AcceptanceSample",
]
