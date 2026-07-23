"""Count reference-matching icons on one requested side of an anchored scene icon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image

from ....core.scene_config import get_scene_defaults
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    required_group_defaults,
    split_generation_rendering_prompt_defaults,
)
from ...shared.counting_sampling import resolve_counting_target_and_distractor_triplet
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from ..shared.anchor_marking import draw_anchor_marker
from ..shared.icon_assets import render_icon_rgba, resolve_icon_pool
from ..shared.defaults import ICON_SHARED_DEFAULTS
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.icon_noise import serialize_icon_noise_edits
from ..shared.icon_scene import (
    draw_two_panel_panels,
    max_overlap_with_existing,
    overlap_fraction_smaller,
    panel_geometry_to_trace,
    random_paste_bbox,
    resolve_two_panel_layout,
    sort_bboxes_reading_order,
)
from ..shared.icon_style import icon_palette_meets_distance_constraints, sample_icon_palette, sample_icon_tints
from ..shared.icon_task_rendering import (
    icon_render_style_trace,
    resolve_icon_render_params,
    resolve_icon_rgb_param,
    sample_icon_instance_noise,
)
TASK_ID = "task_icons__reference_canvas__anchor_position_count"
SCENE_ID = "reference_canvas"
_RELATION_VARIANTS: Tuple[str, ...] = (
    "left_of_anchor",
    "right_of_anchor",
    "above_anchor",
    "below_anchor",
)
SUPPORTED_QUERY_IDS = _RELATION_VARIANTS
_DIRECTIONS: Tuple[str, ...] = ("left", "right", "above", "below")
_OPPOSITE_RELATION = {
    "left_of_anchor": "right_of_anchor",
    "right_of_anchor": "left_of_anchor",
    "above_anchor": "below_anchor",
    "below_anchor": "above_anchor",
}


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable fallback defaults for anchored icon relation counting."""

    object_count_min: int = 1
    object_count_max: int = 15
    target_count_min: int = 0
    target_count_max: int = 5
    distractor_count_min: int = 1
    distractor_count_max: int = 10
    distractor_margin_over_target: int = 1
    canvas_width: int = ICON_SHARED_DEFAULTS.canvas_width
    canvas_height: int = ICON_SHARED_DEFAULTS.canvas_height
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = ICON_SHARED_DEFAULTS.scene_icon_size_min_px
    scene_icon_size_max_px: int = ICON_SHARED_DEFAULTS.scene_icon_size_max_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    scene_max_overlap_fraction: float = 0.05
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    pool_manifest: str = "all_icons.txt"
    palette_size_min: int = 8
    palette_size_max: int = 12
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    anchor_gap_px_directional: int = 8
    anchor_target_area_ratio_min: float = 0.35
    anchor_target_area_ratio_max: float = 0.65
    anchor_opposite_area_ratio_min: float = 0.25
    same_type_distractor_opposite_fraction_min: float = 0.75
    anchor_highlight_padding_px: int = 10
    anchor_highlight_radius_px: int = 14
    anchor_outline_rgb: Tuple[int, int, int] = (74, 113, 188)
    anchor_label_color_rgb: Tuple[int, int, int] = (57, 87, 145)
    anchor_label_font_size_px: int = 20


@dataclass(frozen=True)
class _ScenePayload:
    """Trace-ready payload for one anchored icon-relation scene."""

    object_count: int
    target_count: int
    distractor_count: int
    same_type_distractor_count: int
    different_type_distractor_count: int
    same_type_nonspatial_distractor_count: int
    different_type_spatial_distractor_count: int
    different_type_nonspatial_distractor_count: int
    query_id: str
    reference_icon_id: str
    anchor_icon_id: str
    scene_icon_ids: Tuple[str, ...]
    matching_scene_indices: Tuple[int, ...]
    matching_bboxes: Tuple[Tuple[int, int, int, int], ...]
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...]
    panel_geometry: Dict[str, Any]
    scene_instances: Tuple[Dict[str, Any], ...]
    reference_instance: Dict[str, Any]
    anchor_instance: Dict[str, Any]


_DEFAULTS = _TaskDefaults()
_SCENE_DEFAULTS = get_scene_defaults("icons", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _direction_to_variant(direction: str) -> str:
    """Map public direction to the construction variant."""

    if str(direction) == "left":
        return "left_of_anchor"
    if str(direction) == "right":
        return "right_of_anchor"
    if str(direction) == "above":
        return "above_anchor"
    if str(direction) == "below":
        return "below_anchor"
    raise ValueError(f"unsupported direction: {direction}")


def _variant_to_direction(query_id: str) -> str:
    """Map source construction variant to public direction."""

    if str(query_id) == "left_of_anchor":
        return "left"
    if str(query_id) == "right_of_anchor":
        return "right"
    if str(query_id) == "above_anchor":
        return "above"
    if str(query_id) == "below_anchor":
        return "below"
    raise ValueError(f"unsupported relation variant: {query_id}")


def _resolve_direction(scene_rng, *, params: Mapping[str, Any], instance_seed: int) -> Tuple[str, Dict[str, float]]:
    """Resolve directional relation around the anchor."""

    direction_params = dict(params)
    if direction_params.get("direction") is None and direction_params.get("spatial_direction") is not None:
        direction_params["direction"] = direction_params["spatial_direction"]
    explicit_variant = direction_params.get("query_id")
    if direction_params.get("direction") is None and explicit_variant is not None:
        variant = str(explicit_variant).strip()
        if variant in set(_RELATION_VARIANTS):
            direction_params["direction"] = _variant_to_direction(variant)
        elif variant == str(_PUBLIC_QUERY_ID):
            direction_params.pop("query_id", None)
    selected_direction, direction_probabilities = resolve_variant(
        scene_rng,
        params=direction_params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=_DIRECTIONS,
        explicit_key="direction",
        weights_key="direction_weights",
    )
    selected_direction = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=direction_params,
        gen_defaults=_GEN_DEFAULTS,
        selected_variant=str(selected_direction),
        variant_probabilities=direction_probabilities,
        supported_variants=_DIRECTIONS,
        balance_flag_key="balanced_direction_sampling",
        explicit_key="direction",
        weights_key="direction_weights",
    )
    return str(selected_direction), dict(direction_probabilities)

def _bbox_area(box: Sequence[int | float]) -> float:
    """Return one `xyxy` box area."""

    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _direction_region_bbox(
    content_bbox: Sequence[int | float],
    anchor_bbox: Sequence[int | float],
    *,
    relation_id: str,
    gap_px: int,
) -> Tuple[int, int, int, int] | None:
    """Return the usable region for one directional relation relative to the anchor bbox."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in content_bbox]
    ax0, ay0, ax1, ay1 = [int(round(float(value))) for value in anchor_bbox]
    gap = max(0, int(gap_px))
    if str(relation_id) == "left_of_anchor":
        region = (x0, y0, ax0 - gap, y1)
    elif str(relation_id) == "right_of_anchor":
        region = (ax1 + gap, y0, x1, y1)
    elif str(relation_id) == "above_anchor":
        region = (x0, y0, x1, ay0 - gap)
    elif str(relation_id) == "below_anchor":
        region = (x0, ay1 + gap, x1, y1)
    else:
        raise ValueError(f"unsupported relation_id: {relation_id}")
    if region[2] - region[0] <= 0 or region[3] - region[1] <= 0:
        return None
    return region


def _bbox_satisfies_relation(
    candidate_bbox: Sequence[int | float],
    anchor_bbox: Sequence[int | float],
    *,
    relation_id: str,
    gap_px: int,
) -> bool:
    """Return whether one placed candidate box strictly satisfies the anchor relation."""

    x0, y0, x1, y1 = [int(round(float(value))) for value in candidate_bbox]
    ax0, ay0, ax1, ay1 = [int(round(float(value))) for value in anchor_bbox]
    gap = max(0, int(gap_px))
    if str(relation_id) == "left_of_anchor":
        return int(x1) <= int(ax0) - int(gap)
    if str(relation_id) == "right_of_anchor":
        return int(x0) >= int(ax1) + int(gap)
    if str(relation_id) == "above_anchor":
        return int(y1) <= int(ay0) - int(gap)
    if str(relation_id) == "below_anchor":
        return int(y0) >= int(ay1) + int(gap)
    raise ValueError(f"unsupported relation_id: {relation_id}")


def _bbox_fraction_in_relation_region(
    candidate_bbox: Sequence[int | float],
    anchor_bbox: Sequence[int | float],
    *,
    relation_id: str,
    gap_px: int,
) -> float:
    """Return the exact fraction of one candidate bbox that lies in the relation region."""

    x0, y0, x1, y1 = [float(value) for value in candidate_bbox]
    ax0, ay0, ax1, ay1 = [float(value) for value in anchor_bbox]
    gap = float(max(0, int(gap_px)))
    width = max(1e-6, float(x1 - x0))
    height = max(1e-6, float(y1 - y0))
    candidate_area = float(width * height)
    if str(relation_id) == "left_of_anchor":
        overlap_w = max(0.0, min(float(x1), float(ax0) - gap) - float(x0))
        return float(overlap_w * height) / float(candidate_area)
    if str(relation_id) == "right_of_anchor":
        overlap_w = max(0.0, float(x1) - max(float(x0), float(ax1) + gap))
        return float(overlap_w * height) / float(candidate_area)
    if str(relation_id) == "above_anchor":
        overlap_h = max(0.0, min(float(y1), float(ay0) - gap) - float(y0))
        return float(width * overlap_h) / float(candidate_area)
    if str(relation_id) == "below_anchor":
        overlap_h = max(0.0, float(y1) - max(float(y0), float(ay1) + gap))
        return float(width * overlap_h) / float(candidate_area)
    raise ValueError(f"unsupported relation_id: {relation_id}")


def _passes_relaxed_same_type_distractor_rule(
    candidate_bbox: Sequence[int | float],
    anchor_bbox: Sequence[int | float],
    *,
    relation_id: str,
    gap_px: int,
    opposite_fraction_min: float,
) -> bool:
    """Return true when a same-type distractor lies mostly outside the queried region."""

    frac_target = _bbox_fraction_in_relation_region(
        candidate_bbox,
        anchor_bbox,
        relation_id=str(relation_id),
        gap_px=int(gap_px),
    )
    frac_opposite = 1.0 - float(frac_target)
    return float(frac_opposite) >= max(0.0, min(1.0, float(opposite_fraction_min)))


def _sample_anchor_bbox(
    rng,
    *,
    sprite_size: Tuple[int, int],
    content_bbox: Sequence[int | float],
    relation_id: str,
    gap_px: int,
    target_area_ratio_min: float,
    target_area_ratio_max: float,
    opposite_area_ratio_min: float,
    max_attempts: int,
) -> Tuple[int, int, int, int]:
    """Sample one anchor placement that preserves room on both directional sides."""

    content_area = max(1e-6, _bbox_area(content_bbox))
    opposite_relation = _OPPOSITE_RELATION[str(relation_id)]
    for _ in range(max(1, int(max_attempts))):
        candidate = random_paste_bbox(sprite_size=sprite_size, content_bbox=tuple(int(v) for v in content_bbox), rng=rng)
        target_region = _direction_region_bbox(content_bbox, candidate, relation_id=str(relation_id), gap_px=int(gap_px))
        opposite_region = _direction_region_bbox(
            content_bbox,
            candidate,
            relation_id=str(opposite_relation),
            gap_px=int(gap_px),
        )
        if target_region is None or opposite_region is None:
            continue
        target_ratio = float(_bbox_area(target_region)) / float(content_area)
        opposite_ratio = float(_bbox_area(opposite_region)) / float(content_area)
        if not (float(target_area_ratio_min) <= target_ratio <= float(target_area_ratio_max)):
            continue
        if float(opposite_ratio) < float(opposite_area_ratio_min):
            continue
        return tuple(int(v) for v in candidate)
    raise ValueError("failed to place anchor with sufficient directional room")


def _render_reference_icon(
    *,
    image: Image.Image,
    layout,
    icon_id: str,
    tint_rgb: Sequence[int],
    noise_edits,
    noise_seed: int,
    reference_icon_size_px: int,
) -> Dict[str, Any]:
    """Render one centered reference icon and return trace metadata."""

    reference_rgba = render_icon_rgba(
        icon_id=str(icon_id),
        size_px=int(reference_icon_size_px),
        tint_rgb=tuple(int(value) for value in tint_rgb),
        rotation_degrees=0,
        mirror_x=False,
        noise_edits=tuple(noise_edits),
        noise_seed=int(noise_seed),
    )
    ref_x0, ref_y0, ref_x1, ref_y1 = layout.reference_content_xyxy
    ref_w = int(reference_rgba.size[0])
    ref_h = int(reference_rgba.size[1])
    paste_x = int(ref_x0 + max(0, (ref_x1 - ref_x0 - ref_w) // 2))
    paste_y = int(ref_y0 + max(0, (ref_y1 - ref_y0 - ref_h) // 2))
    image.alpha_composite(reference_rgba, (paste_x, paste_y))
    return {
        "instance_id": "reference_icon",
        "icon_id": str(icon_id),
        "panel": "reference",
        "bbox_xyxy": [paste_x, paste_y, paste_x + ref_w, paste_y + ref_h],
        "rotation_degrees": 0,
        "mirror_x": False,
        "tint_rgb": [int(value) for value in tint_rgb],
        "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(noise_edits))],
        "noise_seed": int(noise_seed),
    }


def _sample_scene(
    rng,
    *,
    instance_seed: int,
    query_id: str,
    object_count: int,
    target_count: int,
    distractor_count: int,
    pool_manifest: str,
    render_params: Mapping[str, Any],
) -> Tuple[_ScenePayload, Image.Image]:
    """Sample and render one anchored relation scene."""

    pool = list(resolve_icon_pool(str(pool_manifest)))
    if len(pool) < 3:
        raise ValueError("icon pool is too small for relation scene")

    reference_icon_id = str(rng.choice(pool))
    anchor_pool = [str(icon_id) for icon_id in pool if str(icon_id) != str(reference_icon_id)]
    if not anchor_pool:
        raise ValueError("icon pool did not resolve an anchor icon distinct from the reference")
    anchor_icon_id = str(rng.choice(anchor_pool))

    same_type_nonspatial_distractor_count = 0
    different_type_spatial_distractor_count = 0
    different_type_nonspatial_distractor_count = 0
    if int(distractor_count) == 1:
        if bool(rng.randint(0, 1)):
            same_type_nonspatial_distractor_count = 1
        else:
            different_type_spatial_distractor_count = 1
    elif int(distractor_count) >= 2:
        same_type_nonspatial_distractor_count = 1
        different_type_spatial_distractor_count = 1
        remaining = int(distractor_count) - 2
        for _ in range(int(remaining)):
            bucket = int(rng.randint(0, 2))
            if int(bucket) == 0:
                same_type_nonspatial_distractor_count += 1
            elif int(bucket) == 1:
                different_type_spatial_distractor_count += 1
            else:
                different_type_nonspatial_distractor_count += 1
    same_type_distractor_count = int(same_type_nonspatial_distractor_count)
    different_type_distractor_count = int(different_type_spatial_distractor_count) + int(different_type_nonspatial_distractor_count)
    other_type_pool = [
        str(icon_id)
        for icon_id in pool
        if str(icon_id) not in {str(reference_icon_id), str(anchor_icon_id)}
    ]
    if len(other_type_pool) < int(different_type_distractor_count):
        raise ValueError("insufficient non-reference icon types for relation distractors")

    candidate_specs: List[Dict[str, Any]] = []
    candidate_specs.extend(
        {"icon_id": str(reference_icon_id), "is_match": True, "is_same_type": True, "spatial_match": True}
        for _ in range(int(target_count))
    )
    candidate_specs.extend(
        {"icon_id": str(reference_icon_id), "is_match": False, "is_same_type": True, "spatial_match": False}
        for _ in range(int(same_type_nonspatial_distractor_count))
    )
    different_type_icon_ids = list(rng.sample(other_type_pool, int(different_type_distractor_count)))
    spatial_type_ids = different_type_icon_ids[: int(different_type_spatial_distractor_count)]
    nonspatial_type_ids = different_type_icon_ids[int(different_type_spatial_distractor_count) :]
    for icon_id in spatial_type_ids:
        candidate_specs.append(
            {"icon_id": str(icon_id), "is_match": False, "is_same_type": False, "spatial_match": True}
        )
    for icon_id in nonspatial_type_ids:
        candidate_specs.append(
            {"icon_id": str(icon_id), "is_match": False, "is_same_type": False, "spatial_match": False}
        )
    rng.shuffle(candidate_specs)

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    palette = sample_icon_palette(
        rng,
        palette_size=int(palette_size),
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    if not icon_palette_meets_distance_constraints(
        palette=palette,
        anchor_colors=(
            tuple(int(v) for v in render_params["background_color_rgb"]),
            tuple(int(v) for v in render_params["panel_fill_rgb"]),
            tuple(int(v) for v in render_params["panel_border_rgb"]),
            tuple(int(v) for v in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    ):
        raise ValueError("sampled icon palette did not satisfy strict distance constraints")
    sampled_tints = list(sample_icon_tints(rng, palette=palette, count=int(object_count) + 2))
    reference_tint = tuple(int(v) for v in sampled_tints.pop(0))
    anchor_tint = tuple(int(v) for v in sampled_tints.pop(0))

    reference_noise_edits, reference_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{IconsRelationRelativePositionTypeTask.task_id}:reference_icon",
        render_params=render_params,
    )
    anchor_noise_edits, anchor_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{IconsRelationRelativePositionTypeTask.task_id}:anchor_icon",
        render_params=render_params,
    )
    for index, spec in enumerate(candidate_specs):
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{IconsRelationRelativePositionTypeTask.task_id}:scene_icon_{int(index)}",
            render_params=render_params,
        )
        spec["tint_rgb"] = tuple(int(v) for v in sampled_tints.pop(0))
        spec["noise_edits"] = tuple(noise_edits)
        spec["noise_seed"] = int(noise_seed)

    layout = resolve_two_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        reference_panel_width_px=int(render_params["reference_panel_width_px"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_gap_px=int(render_params["panel_gap_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
    )
    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_two_panel_panels(
        image=image,
        layout=layout,
        background_rgb=tuple(int(v) for v in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(v) for v in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(v) for v in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )

    reference_instance = _render_reference_icon(
        image=image,
        layout=layout,
        icon_id=str(reference_icon_id),
        tint_rgb=reference_tint,
        noise_edits=tuple(reference_noise_edits),
        noise_seed=int(reference_noise_seed),
        reference_icon_size_px=int(render_params["reference_icon_size_px"]),
    )

    scene_content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    anchor_size = int(rng.randint(int(render_params["scene_icon_size_min_px"]), int(render_params["scene_icon_size_max_px"])))
    anchor_rgba = render_icon_rgba(
        icon_id=str(anchor_icon_id),
        size_px=int(anchor_size),
        tint_rgb=tuple(int(value) for value in anchor_tint),
        rotation_degrees=0,
        mirror_x=False,
        noise_edits=tuple(anchor_noise_edits),
        noise_seed=int(anchor_noise_seed),
    )
    anchor_bbox = _sample_anchor_bbox(
        rng,
        sprite_size=anchor_rgba.size,
        content_bbox=scene_content_bbox,
        relation_id=str(query_id),
        gap_px=int(render_params["anchor_gap_px_directional"]),
        target_area_ratio_min=float(render_params["anchor_target_area_ratio_min"]),
        target_area_ratio_max=float(render_params["anchor_target_area_ratio_max"]),
        opposite_area_ratio_min=float(render_params["anchor_opposite_area_ratio_min"]),
        max_attempts=int(render_params["scene_placement_max_attempts"]),
    )
    image.alpha_composite(anchor_rgba, (int(anchor_bbox[0]), int(anchor_bbox[1])))

    anchor_highlight_box = draw_anchor_marker(
        image=image,
        anchor_bbox=anchor_bbox,
        content_bbox=scene_content_bbox,
        highlight_padding_px=int(render_params["anchor_highlight_padding_px"]),
        highlight_radius_px=int(render_params["anchor_highlight_radius_px"]),
        outline_rgb=tuple(int(v) for v in render_params["anchor_outline_rgb"]),
        label_color_rgb=tuple(int(v) for v in render_params["anchor_label_color_rgb"]),
        panel_fill_rgb=tuple(int(v) for v in render_params["panel_fill_rgb"]),
        label_font_size_px=int(render_params["anchor_label_font_size_px"]),
        label_text="Anchor",
    )

    target_region = _direction_region_bbox(
        scene_content_bbox,
        anchor_bbox,
        relation_id=str(query_id),
        gap_px=int(render_params["anchor_gap_px_directional"]),
    )
    if target_region is None:
        raise ValueError("anchor placement resolved no usable target region")

    placed_bboxes: List[Tuple[int, int, int, int]] = [tuple(int(value) for value in anchor_bbox)]
    scene_instances: List[Dict[str, Any]] = []
    matching_scene_indices: List[int] = []
    scene_icon_ids: List[str] = []
    for index, spec in enumerate(candidate_specs):
        min_size = max(16, int(render_params["scene_icon_size_min_px"]))
        max_size = max(min_size, int(render_params["scene_icon_size_max_px"]))
        content_region = target_region if bool(spec["spatial_match"]) else scene_content_bbox
        region_w = max(1, int(content_region[2] - content_region[0]))
        region_h = max(1, int(content_region[3] - content_region[1]))
        current_max_size = min(int(max_size), int(region_w), int(region_h))
        sprite = None
        paste_bbox = None
        for shrink_round in range(max(0, int(render_params["scene_size_shrink_rounds"])) + 1):
            round_max_size = max(
                int(min_size),
                int(round(float(current_max_size) * (float(render_params["scene_size_shrink_factor"]) ** shrink_round))),
            )
            if int(round_max_size) < int(min_size):
                round_max_size = int(min_size)
            for _ in range(max(1, int(render_params["scene_placement_max_attempts"]))):
                nominal_size = int(rng.randint(int(min_size), int(round_max_size)))
                candidate_sprite = render_icon_rgba(
                    icon_id=str(spec["icon_id"]),
                    size_px=int(nominal_size),
                    tint_rgb=tuple(int(v) for v in spec["tint_rgb"]),
                    rotation_degrees=0,
                    mirror_x=False,
                    noise_edits=tuple(spec["noise_edits"]),
                    noise_seed=int(spec["noise_seed"]),
                )
                try:
                    candidate_bbox = random_paste_bbox(
                        sprite_size=candidate_sprite.size,
                        content_bbox=tuple(int(value) for value in content_region),
                        rng=rng,
                    )
                except ValueError:
                    continue
                spatial_match = _bbox_satisfies_relation(
                    candidate_bbox,
                    anchor_bbox,
                    relation_id=str(query_id),
                    gap_px=int(render_params["anchor_gap_px_directional"]),
                )
                if bool(spatial_match) != bool(spec["spatial_match"]):
                    continue
                if (
                    (not bool(spec["is_match"]))
                    and bool(spec["is_same_type"])
                    and (not bool(spec["spatial_match"]))
                ):
                    if not _passes_relaxed_same_type_distractor_rule(
                        candidate_bbox,
                        anchor_bbox,
                        relation_id=str(query_id),
                        gap_px=int(render_params["anchor_gap_px_directional"]),
                        opposite_fraction_min=float(render_params["same_type_distractor_opposite_fraction_min"]),
                    ):
                        continue
                if float(max_overlap_with_existing(candidate_bbox, placed_bboxes)) > float(render_params["scene_max_overlap_fraction"]):
                    continue
                sprite = candidate_sprite
                paste_bbox = candidate_bbox
                break
            if sprite is not None and paste_bbox is not None:
                break
        if sprite is None or paste_bbox is None:
            raise ValueError("failed to place relation icon under directional constraints")

        image.alpha_composite(sprite, (int(paste_bbox[0]), int(paste_bbox[1])))
        placed_bboxes.append(tuple(int(value) for value in paste_bbox))
        if bool(spec["is_match"]):
            matching_scene_indices.append(int(index))
        scene_icon_ids.append(str(spec["icon_id"]))
        scene_instances.append(
            {
                "instance_id": f"scene_icon_{int(index)}",
                "icon_id": str(spec["icon_id"]),
                "panel": "scene",
                "role": "candidate",
                "bbox_xyxy": [int(value) for value in paste_bbox],
                "rotation_degrees": 0,
                "mirror_x": False,
                "tint_rgb": [int(value) for value in spec["tint_rgb"]],
                "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(spec["noise_edits"]))],
                "noise_seed": int(spec["noise_seed"]),
                "is_match": bool(spec["is_match"]),
                "is_same_type": bool(spec["is_same_type"]),
                "spatial_match": bool(spec["spatial_match"]),
                "index": int(index),
            }
        )

    anchor_instance = {
        "instance_id": "anchor_icon",
        "icon_id": str(anchor_icon_id),
        "panel": "scene",
        "role": "anchor",
        "bbox_xyxy": [int(value) for value in anchor_bbox],
        "rotation_degrees": 0,
        "mirror_x": False,
        "tint_rgb": [int(value) for value in anchor_tint],
        "noise_edits": [dict(edit) for edit in serialize_icon_noise_edits(tuple(anchor_noise_edits))],
        "noise_seed": int(anchor_noise_seed),
        "highlight_bbox_xyxy": [int(value) for value in anchor_highlight_box],
    }
    matching_bboxes = tuple(
        tuple(int(value) for value in scene_instances[int(index)]["bbox_xyxy"])
        for index in sorted(int(value) for value in matching_scene_indices)
    )
    return _ScenePayload(
        object_count=int(object_count),
        target_count=int(target_count),
        distractor_count=int(distractor_count),
        same_type_distractor_count=int(same_type_distractor_count),
        different_type_distractor_count=int(different_type_distractor_count),
        same_type_nonspatial_distractor_count=int(same_type_nonspatial_distractor_count),
        different_type_spatial_distractor_count=int(different_type_spatial_distractor_count),
        different_type_nonspatial_distractor_count=int(different_type_nonspatial_distractor_count),
        query_id=str(query_id),
        reference_icon_id=str(reference_icon_id),
        anchor_icon_id=str(anchor_icon_id),
        scene_icon_ids=tuple(str(icon_id) for icon_id in scene_icon_ids),
        matching_scene_indices=tuple(sorted(int(value) for value in matching_scene_indices)),
        matching_bboxes=matching_bboxes,
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in palette),
        panel_geometry=panel_geometry_to_trace(layout),
        scene_instances=tuple(scene_instances),
        reference_instance=reference_instance,
        anchor_instance=anchor_instance,
    ), image.convert("RGB")


@register_task
class IconsRelationRelativePositionTypeTask:
    """Count scene icons that match a reference type on one side of the anchor."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "icons"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic anchored icon relation-counting instance."""

        scene_rng = spawn_rng(int(instance_seed), "scene")
        direction, direction_probabilities = _resolve_direction(
            scene_rng,
            params=params,
            instance_seed=int(instance_seed),
        )
        query_id = _direction_to_variant(str(direction))
        counting_params = dict(params)
        (
            object_count,
            object_count_probabilities,
            target_count,
            target_count_probabilities,
            distractor_count,
            distractor_count_probabilities,
        ) = resolve_counting_target_and_distractor_triplet(
            scene_rng,
            instance_seed=int(instance_seed),
            params=counting_params,
            gen_defaults=_GEN_DEFAULTS,
            fallback_total_min=_DEFAULTS.object_count_min,
            fallback_total_max=_DEFAULTS.object_count_max,
            fallback_target_min=_DEFAULTS.target_count_min,
            fallback_target_max=_DEFAULTS.target_count_max,
            fallback_distractor_min=_DEFAULTS.distractor_count_min,
            fallback_distractor_max=_DEFAULTS.distractor_count_max,
        )
        render_params = resolve_icon_render_params(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        for extra_key, fallback_value in (
            ("anchor_gap_px_directional", _DEFAULTS.anchor_gap_px_directional),
            ("anchor_target_area_ratio_min", _DEFAULTS.anchor_target_area_ratio_min),
            ("anchor_target_area_ratio_max", _DEFAULTS.anchor_target_area_ratio_max),
            ("anchor_opposite_area_ratio_min", _DEFAULTS.anchor_opposite_area_ratio_min),
            ("same_type_distractor_opposite_fraction_min", _DEFAULTS.same_type_distractor_opposite_fraction_min),
            ("anchor_highlight_padding_px", _DEFAULTS.anchor_highlight_padding_px),
            ("anchor_highlight_radius_px", _DEFAULTS.anchor_highlight_radius_px),
            ("anchor_label_font_size_px", _DEFAULTS.anchor_label_font_size_px),
        ):
            render_params[str(extra_key)] = params.get(
                str(extra_key),
                group_default(_RENDER_DEFAULTS, str(extra_key), fallback_value),
            )
        render_params["anchor_outline_rgb"] = resolve_icon_rgb_param(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            key="anchor_outline_rgb",
            fallback=_DEFAULTS.anchor_outline_rgb,
            instance_seed=int(instance_seed),
        )
        render_params["anchor_label_color_rgb"] = resolve_icon_rgb_param(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            key="anchor_label_color_rgb",
            fallback=_DEFAULTS.anchor_label_color_rgb,
            instance_seed=int(instance_seed),
        )
        pool_manifest = str(params.get("pool_manifest", group_default(_GEN_DEFAULTS, "pool_manifest", _DEFAULTS.pool_manifest)))

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = _sample_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    query_id=str(query_id),
                    object_count=int(object_count),
                    target_count=int(target_count),
                    distractor_count=int(distractor_count),
                    pool_manifest=str(pool_manifest),
                    render_params=render_params,
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError("failed to generate task_icons__reference_canvas__anchor_position_count instance") from last_error

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            query_key=str(query_id),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
        annotation_value = sort_bboxes_reading_order(scene_payload.matching_bboxes)
        annotation_payload = icon_bbox_set_annotation(
            annotation_value,
            clip_bbox=scene_payload.panel_geometry["scene_content_xyxy"],
        )
        answer_value = int(scene_payload.target_count)

        scene_entities = [
            dict(scene_payload.reference_instance),
            dict(scene_payload.anchor_instance),
            *[dict(entity) for entity in scene_payload.scene_instances],
        ]

        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_reference_anchor_relation_type",
                "entities": scene_entities,
                "relations": {
                    "counting_target": "same_icon_type_and_directional_relation_to_anchor",
                    "query_id": str(query_id),
                    "reference_icon_id": str(scene_payload.reference_icon_id),
                    "anchor_icon_id": str(scene_payload.anchor_icon_id),
                    "spatial_relation": str(query_id),
                    "direction": str(direction),
                    "matching_scene_indices": [int(value) for value in scene_payload.matching_scene_indices],
                    "same_type_nonspatial_distractor_count": int(scene_payload.same_type_nonspatial_distractor_count),
                    "different_type_spatial_distractor_count": int(scene_payload.different_type_spatial_distractor_count),
                    "different_type_nonspatial_distractor_count": int(scene_payload.different_type_nonspatial_distractor_count),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": {
                "query_id": str(query_id),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "object_count": int(object_count),
                    "object_count_probabilities": dict(object_count_probabilities),
                    "target_count": int(target_count),
                    "target_count_probabilities": dict(target_count_probabilities),
                    "distractor_count": int(distractor_count),
                    "distractor_count_probabilities": dict(distractor_count_probabilities),
                    "distractor_margin_over_target": int(
                        params.get(
                            "distractor_margin_over_target",
                            group_default(_GEN_DEFAULTS, "distractor_margin_over_target", _DEFAULTS.distractor_margin_over_target),
                        )
                    ),
                    "pool_manifest": str(pool_manifest),
                    "direction": str(direction),
                    "direction_probabilities": dict(direction_probabilities),
                    "anchor_gap_px_directional": int(render_params["anchor_gap_px_directional"]),
                    "same_type_distractor_opposite_fraction_min": float(
                        render_params["same_type_distractor_opposite_fraction_min"]
                    ),
                },
            },
            "render_spec": {
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": {
                    **icon_render_style_trace(
                        render_params=render_params,
                        sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                    ),
                    "anchor_gap_px_directional": int(render_params["anchor_gap_px_directional"]),
                    "anchor_target_area_ratio_min": float(render_params["anchor_target_area_ratio_min"]),
                    "anchor_target_area_ratio_max": float(render_params["anchor_target_area_ratio_max"]),
                    "anchor_opposite_area_ratio_min": float(render_params["anchor_opposite_area_ratio_min"]),
                    "same_type_distractor_opposite_fraction_min": float(
                        render_params["same_type_distractor_opposite_fraction_min"]
                    ),
                    "anchor_highlight_padding_px": int(render_params["anchor_highlight_padding_px"]),
                    "anchor_highlight_radius_px": int(render_params["anchor_highlight_radius_px"]),
                    "anchor_outline_rgb": [int(v) for v in render_params["anchor_outline_rgb"]],
                    "anchor_label_color_rgb": [int(v) for v in render_params["anchor_label_color_rgb"]],
                    "anchor_label_font_size_px": int(render_params["anchor_label_font_size_px"]),
                },
            },
            "render_map": {
                "image_id": "img0",
                "anchors": {
                    "reference_icon": dict(scene_payload.reference_instance),
                    "anchor_icon": dict(scene_payload.anchor_instance),
                    "matching_scene_boxes": list(annotation_payload["annotation_value"]),
                },
            },
            "execution_trace": {
                "scene_variant": "reference_scene_anchor",
                "query_id": str(query_id),
                "direction": str(direction),
                "object_count": int(scene_payload.object_count),
                "object_count_probabilities": dict(object_count_probabilities),
                "target_count": int(scene_payload.target_count),
                "target_count_probabilities": dict(target_count_probabilities),
                "distractor_count": int(scene_payload.distractor_count),
                "distractor_count_probabilities": dict(distractor_count_probabilities),
                "distractor_margin_over_target": int(
                    params.get(
                        "distractor_margin_over_target",
                        group_default(_GEN_DEFAULTS, "distractor_margin_over_target", _DEFAULTS.distractor_margin_over_target),
                    )
                ),
                "same_type_distractor_count": int(scene_payload.same_type_distractor_count),
                "different_type_distractor_count": int(scene_payload.different_type_distractor_count),
                "same_type_nonspatial_distractor_count": int(scene_payload.same_type_nonspatial_distractor_count),
                "different_type_spatial_distractor_count": int(scene_payload.different_type_spatial_distractor_count),
                "different_type_nonspatial_distractor_count": int(scene_payload.different_type_nonspatial_distractor_count),
                "reference_icon_id": str(scene_payload.reference_icon_id),
                "anchor_icon_id": str(scene_payload.anchor_icon_id),
                "scene_icon_ids": list(scene_payload.scene_icon_ids),
                "matching_scene_indices": [int(value) for value in scene_payload.matching_scene_indices],
                "spatial_relation": str(query_id),
                "question_format": "count_matching_scene_icons_by_reference_and_anchor_relation",
                "direction_probabilities": dict(direction_probabilities),
            },
            "witness_symbolic": {
                "reference_icon_id": str(scene_payload.reference_icon_id),
                "anchor_icon_id": str(scene_payload.anchor_icon_id),
                "query_id": str(query_id),
                "direction": str(direction),
                "spatial_relation": str(query_id),
                "matching_scene_indices": [int(value) for value in scene_payload.matching_scene_indices],
            },
            "projected_annotation": dict(annotation_payload["projected_annotation"]),
        }
        scene_content_bbox = scene_payload.panel_geometry["scene_content_xyxy"]
        anchor_bbox = scene_payload.anchor_instance["bbox_xyxy"]
        relation_region_bbox = _direction_region_bbox(
            scene_content_bbox,
            anchor_bbox,
            relation_id=str(query_id),
            gap_px=int(render_params["anchor_gap_px_directional"]),
        )
        scene_area = _bbox_area(scene_content_bbox)
        target_region_area_ratio = (
            float(_bbox_area(relation_region_bbox)) / float(scene_area)
            if relation_region_bbox is not None and scene_area > 0.0
            else 0.0
        )
        max_allowed_target_fraction = max(
            1e-6,
            1.0 - float(render_params["same_type_distractor_opposite_fraction_min"]),
        )
        boundary_proximity_samples = []
        for entity in scene_payload.scene_instances:
            if str(entity["icon_id"]) != str(scene_payload.reference_icon_id) or bool(entity["spatial_match"]):
                continue
            target_fraction = _bbox_fraction_in_relation_region(
                entity["bbox_xyxy"],
                anchor_bbox,
                relation_id=str(query_id),
                gap_px=int(render_params["anchor_gap_px_directional"]),
            )
            boundary_proximity_samples.append(min(1.0, float(target_fraction) / float(max_allowed_target_fraction)))
        same_type_boundary_proximity = (
            float(sum(boundary_proximity_samples)) / float(len(boundary_proximity_samples))
            if boundary_proximity_samples
            else 0.0
        )
        output = TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
            answer_gt=TypedValue(type="integer", value=int(answer_value)),
            annotation_gt=TypedValue(
                type=str(annotation_payload["annotation_type"]),
                value=list(annotation_payload["annotation_value"]),
            ),
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
        )
        return output


__all__ = ["IconsRelationRelativePositionTypeTask", "SUPPORTED_QUERY_IDS"]
