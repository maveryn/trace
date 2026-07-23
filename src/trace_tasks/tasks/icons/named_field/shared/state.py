"""Passive state objects for named-field icon tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image

@dataclass(frozen=True)
class NamedColorEntry:
    """One semantic color available to named-field queries."""

    name: str
    rgb: Tuple[int, int, int]
    label: str


@dataclass(frozen=True)
class ShapeCountSampleSpec:
    """Fully sampled symbolic plan for one direct named-shape count."""

    arrangement_mode: str
    target_shape_id: str
    target_shape_name: str
    target_count: int
    object_count: int
    shape_ids: Tuple[str, ...]
    placement_groups: Tuple[str, ...]
    arrangement_details: Dict[str, Any]
    arrangement_mode_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    object_count_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PairArithmeticOperandSpec:
    """One named-icon operand used by pair arithmetic."""

    shape_id: str
    shape_name: str
    color_name: str
    color_label: str
    label: str


@dataclass(frozen=True)
class PairArithmeticIconSemanticSpec:
    """One renderable icon in a pair-arithmetic field."""

    shape_id: str
    color_name: str
    fill_style: str
    role: str


@dataclass(frozen=True)
class PairArithmeticSampleSpec:
    """Sampled symbolic plan for named-field pair arithmetic."""

    query_key: str
    operation: str
    uses_color_binding: bool
    left_operand: PairArithmeticOperandSpec
    right_operand: PairArithmeticOperandSpec
    left_count: int
    right_count: int
    target_answer: int
    distractor_count: int
    object_count: int
    arrangement_mode: str
    semantic_specs: Tuple[PairArithmeticIconSemanticSpec, ...]
    query_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    color_probabilities: Dict[str, float]
    answer_probabilities: Dict[str, float]
    operand_count_probabilities: Dict[str, float]
    distractor_count_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]
    arrangement_mode_probabilities: Dict[str, float]


@dataclass(frozen=True)
class BooleanIconSemanticSpec:
    """One target-relative icon before projection to pixels."""

    shape_id: str
    color_name: str
    fill_style: str
    partition: str


@dataclass(frozen=True)
class BooleanSampleSpec:
    """Fully sampled symbolic plan for one Boolean named-field count."""

    prompt_query_key: str
    predicate_kind: str
    target_shape_id: str
    target_shape_name: str
    target_attribute_axis: str
    target_attribute_value: str
    target_attribute_label: str
    target_color: NamedColorEntry | None
    target_answer: int
    object_count: int
    object_count_max_answer_offset: int
    arrangement_mode: str
    partition_counts: Dict[str, int]
    semantic_specs: Tuple[BooleanIconSemanticSpec, ...]
    shape_probabilities: Dict[str, float]
    color_probabilities: Dict[str, float]
    fill_style_probabilities: Dict[str, float]
    attribute_axis_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    object_count_probabilities: Dict[str, float]
    arrangement_mode_probabilities: Dict[str, float]


@dataclass(frozen=True)
class CounterfactualIconSemanticSpec:
    """One icon role for a hypothetical named-field edit."""

    shape_id: str
    counterfactual_role: str
    counted_after_edit: bool


@dataclass(frozen=True)
class CounterfactualSampleSpec:
    """Fully sampled symbolic plan for one counterfactual count."""

    prompt_query_key: str
    edit_kind: str
    target_answer: int
    object_count: int
    target_shape_id: str
    target_shape_name: str
    source_shape_id: str
    source_shape_name: str
    remove_shape_id: str
    remove_shape_name: str
    source_count: int
    existing_target_count: int
    removal_count: int
    distractor_count: int
    arrangement_mode: str
    semantic_specs: Tuple[CounterfactualIconSemanticSpec, ...]
    shape_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    removal_count_probabilities: Dict[str, float]
    distractor_count_probabilities: Dict[str, float]
    arrangement_mode_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


@dataclass(frozen=True)
class CloserReferenceIconPlan:
    """Semantic plan for one closer-to-reference icon."""

    role: str
    label: str
    shape_id: str
    color_name: str
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    nominal_size_px: int
    rotation_degrees: int
    desired_closer_label: str
    noise_edits: Tuple[Any, ...]
    noise_seed: int | None


@dataclass(frozen=True)
class CloserReferenceRenderedIcon:
    """Rendered icon metadata for a closer-to-reference field."""

    instance_id: str
    role: str
    label: str
    shape_id: str
    shape_name: str
    color_name: str
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]
    nominal_size_px: int
    rotation_degrees: int
    distance_to_reference_a_px: float | None
    distance_to_reference_b_px: float | None
    closer_reference_label: str
    counted: bool
    label_bbox_xyxy: Tuple[int, int, int, int] | None
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None


@dataclass(frozen=True)
class CloserReferenceSampleSpec:
    """Sampled symbolic plan for closer-to-reference counting."""

    query_key: str
    queried_reference_label: str
    target_shape_id: str
    target_shape_name: str
    reference_a_shape_name: str
    reference_b_shape_name: str
    target_answer: int
    target_icon_count: int
    closer_count_by_reference: Dict[str, int]
    plans: Tuple[CloserReferenceIconPlan, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    query_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    color_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    target_icon_count_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]
    reference_axis_probabilities: Dict[str, float]
    queried_reference_label_probabilities: Dict[str, float]


@dataclass(frozen=True)
class CloserReferenceScenePayload:
    """Trace-ready rendered payload for closer-to-reference counting."""

    image: Image.Image
    icons: Tuple[CloserReferenceRenderedIcon, ...]
    panel_geometry: Dict[str, Any]
    reference_axis_degrees: int


@dataclass(frozen=True)
class DistanceRankIconPlan:
    """Semantic plan for one distance-rank icon."""

    role: str
    label: str
    shape_id: str
    color_name: str
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    nominal_size_px: int
    rotation_degrees: int
    noise_edits: Tuple[Any, ...]
    noise_seed: int | None


@dataclass(frozen=True)
class DistanceRankRenderedIcon:
    """Rendered icon metadata for the distance-rank scene."""

    instance_id: str
    role: str
    label: str
    shape_id: str
    shape_name: str
    color_name: str
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]
    nominal_size_px: int
    rotation_degrees: int
    distance_to_reference_px: float | None
    distance_rank: int | None
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None


@dataclass(frozen=True)
class DistanceRankScenePayload:
    """Trace-ready payload for one named-reference distance-rank scene."""

    query_key: str
    answer_label: str
    answer_rank: int
    reference_description: str
    reference_icon: DistanceRankRenderedIcon
    candidate_icons: Tuple[DistanceRankRenderedIcon, ...]
    distractor_icons: Tuple[DistanceRankRenderedIcon, ...]
    distance_by_label: Dict[str, float]
    sorted_candidate_labels_by_distance: Tuple[str, ...]
    panel_geometry: Dict[str, Any]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    distractor_count: int


@dataclass(frozen=True)
class RegionSpec:
    """Visible region specification for scoped named-field counting."""

    query_key: str
    region_kind: str
    counts_inside: bool
    shape_kind: str = ""
    band_kind: str = ""
    quadrant_id: str = ""
    shelf_index: int = -1
    shelf_count: int = 0
    bbox_xyxy: Tuple[int, int, int, int] | None = None
    ellipse_center_xy: Tuple[float, float] | None = None
    ellipse_radii_xy: Tuple[float, float] | None = None
    band_normal_xy: Tuple[float, float] | None = None
    band_center_distance: float | None = None
    band_half_width_px: float | None = None
    band_polygon_xy: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class RegionIconPlan:
    """Semantic plan for one scoped-region icon."""

    shape_id: str
    desired_inside_region: bool
    is_target_shape: bool


@dataclass(frozen=True)
class RenderedRegionIcon:
    """Rendered icon metadata for scoped-region counting."""

    instance_id: str
    shape_id: str
    shape_name: str
    bbox_xyxy: Tuple[int, int, int, int]
    center_xy: Tuple[float, float]
    nominal_size_px: int
    rotation_degrees: int
    tint_rgb: Tuple[int, int, int]
    fill_style: str
    inside_region: bool
    counted: bool
    noise_edits: Tuple[Dict[str, Any], ...]
    noise_seed: int | None


@dataclass(frozen=True)
class ScopedRegionScenePayload:
    """Rendered payload for scoped-region named-field counting."""

    image: Image.Image
    panel_geometry: Dict[str, Any]
    region: RegionSpec
    target_shape_id: str
    target_shape_name: str
    target_count: int
    object_count: int
    instances: Tuple[RenderedRegionIcon, ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    query_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    target_count_probabilities: Dict[str, float]
    object_count_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


__all__ = [
    "BooleanIconSemanticSpec",
    "BooleanSampleSpec",
    "CloserReferenceIconPlan",
    "CloserReferenceRenderedIcon",
    "CloserReferenceSampleSpec",
    "CloserReferenceScenePayload",
    "CounterfactualIconSemanticSpec",
    "CounterfactualSampleSpec",
    "DistanceRankIconPlan",
    "DistanceRankRenderedIcon",
    "DistanceRankScenePayload",
    "NamedColorEntry",
    "PairArithmeticIconSemanticSpec",
    "PairArithmeticOperandSpec",
    "PairArithmeticSampleSpec",
    "RegionIconPlan",
    "RegionSpec",
    "RenderedRegionIcon",
    "ScopedRegionScenePayload",
    "ShapeCountSampleSpec",
]
