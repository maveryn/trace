"""Select the labeled named icon at a requested distance rank from a named reference."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.color_format import format_named_color_with_hex
from ...shared.config_defaults import group_default, required_group_defaults, split_generation_rendering_prompt_defaults
from ...shared.labeling import LABEL_POOL_A_L
from ...shared.named_colors import available_named_colors, named_color
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_task_prompt_variants
from ...shared.text_legibility import resolve_readable_text_style, text_legibility_summary_from_records
from ...shared.text_rendering import draw_text_centered, load_font
from ...shared.variant_sampling import resolve_variant
from ..shared.defaults import ICON_SHARED_DEFAULTS
from ..shared.icon_noise import serialize_icon_noise_edits
from ..shared.annotation import icon_bbox_map_annotation
from ..shared.icon_scene import BBox, draw_single_panel, resolve_single_panel_layout, single_panel_geometry_to_trace
from ..shared.icon_task_rendering import resolve_icon_render_params, resolve_icon_rgb_param, sample_icon_instance_noise
from ..shared.procedural_named_icon_field_scene import (
    bbox_from_center_dimensions,
    bbox_inside,
    boxes_overlap,
    label_bbox_for_icon,
    render_planned_named_icon_sprite,
    rotation_for_named_shape,
    union_bbox,
    resolve_named_icon_fill_style_probabilities,
)
from ..shared.procedural_named_icons import (
    DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS,
    PROCEDURAL_NAMED_ICON_FILL_STYLES,
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
    procedural_named_icon_fill_style_probability_map,
    render_procedural_named_icon_rgba,
    sample_procedural_named_icon_fill_style,
    validate_procedural_named_icon_fill_style_support,
)
from .shared.output import build_distance_rank_trace_payload
from .shared.rendering import (
    render_distance_rank_scene as _render_placed_scene,
)
from .shared.state import (
    DistanceRankIconPlan as _IconPlan,
    DistanceRankScenePayload as _ScenePayload,
)


TASK_ID = "task_icons__named_field__reference_distance_rank_label"
SCENE_ID = "named_field"

QUERY_IDS: Tuple[str, ...] = (
    "closest_to_named_reference_label",
    "second_closest_to_named_reference_label",
    "farthest_from_named_reference_label",
)
RANK_BY_QUERY_ID: Dict[str, int] = {
    "closest_to_named_reference_label": 0,
    "second_closest_to_named_reference_label": 1,
    "farthest_from_named_reference_label": 5,
}
OPTION_LABELS: Tuple[str, ...] = tuple(str(label) for label in LABEL_POOL_A_L[:6])
_ANGLE_POOL_DEGREES: Tuple[int, ...] = (
    -62,
    -46,
    -30,
    -14,
    4,
    20,
    38,
    56,
    124,
    140,
    158,
    176,
    194,
    212,
    230,
    248,
)


@dataclass(frozen=True)
class _TaskDefaults:
    """Stable defaults for the named-reference distance-rank task."""

    candidate_count: int = 6
    distractor_count_min: int = 4
    distractor_count_max: int = 8
    canvas_width: int = 960
    canvas_height: int = 560
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 72
    reference_icon_size_min_px: int = 58
    reference_icon_size_max_px: int = 76
    reference_icon_size_px: int = 68
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = 260
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
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
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=lambda: {
            "blur": {"radius": (0.2, 0.6)},
            "downsample": {"scale": (0.85, 0.95)},
            "jpeg": {"quality": (70.0, 90.0)},
            "noise": {"alpha": (0.03, 0.08)},
        }
    )
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES
    distance_rank_margin_px: int = 24
    center_distance_min_px: int = 92
    center_distance_gap_jitter_px: int = 5
    icon_collision_gap_px: int = 8
    candidate_label_font_size_px: int = 24
    candidate_label_padding_px: int = 5
    candidate_label_gap_px: int = 4
    candidate_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    candidate_label_background_rgb: Tuple[int, int, int] = (255, 255, 255)
    candidate_label_border_rgb: Tuple[int, int, int] = (172, 183, 204)


_DEFAULTS = _TaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("icons", "named_field")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _shape_support(params: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get("shape_id_support", group_default(_GEN_DEFAULTS, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(dict.fromkeys(str(value) for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    if len(values) < 8:
        raise ValueError("named-reference distance rank needs at least eight supported icon shapes")
    return values


def _color_support(params: Mapping[str, Any]) -> Tuple[str, ...]:
    available = {str(name): tuple(int(channel) for channel in rgb) for name, rgb in available_named_colors()}
    raw = params.get("named_color_support", group_default(_GEN_DEFAULTS, "named_color_support", tuple(available)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("named_color_support must be a sequence")
    values = tuple(dict.fromkeys(str(value).strip().lower() for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(available))
    if unsupported:
        raise ValueError(f"unsupported named colors: {unsupported}")
    if len(values) < 4:
        raise ValueError("named-reference distance rank needs at least four named colors")
    return values


def _fill_style_support(params: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get(
        "named_icon_fill_style_support",
        group_default(_GEN_DEFAULTS, "named_icon_fill_style_support", _DEFAULTS.named_icon_fill_style_support),
    )
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = _DEFAULTS.named_icon_fill_style_support
    return validate_procedural_named_icon_fill_style_support(tuple(str(value) for value in raw))



def uniform_string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def _uniform_int_probability_map(values: Sequence[int], *, selected: int | None = None) -> Dict[str, float]:
    support = tuple(int(value) for value in values)
    if not support:
        return {}
    if selected is not None:
        return {str(value): (1.0 if int(value) == int(selected) else 0.0) for value in support}
    probability = 1.0 / float(len(support))
    return {str(value): float(probability) for value in support}


def _weighted_choice(rng, probabilities: Mapping[str, float]) -> str:
    threshold = float(rng.random())
    cumulative = 0.0
    items = [(str(key), float(value)) for key, value in sorted(probabilities.items()) if float(value) > 0.0]
    if not items:
        raise ValueError("probability map has no positive values")
    total = sum(float(value) for _key, value in items)
    for key, value in items:
        cumulative += float(value) / float(total)
        if threshold <= cumulative:
            return str(key)
    return str(items[-1][0])


def _resolve_query(rng, *, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    query_params = dict(params)
    explicit_variant = str(query_params.get("query_id", "") or "").strip()
    if query_params.get("distance_rank_query") is None and explicit_variant in set(QUERY_IDS):
        query_params["distance_rank_query"] = explicit_variant
    query_id, probabilities = resolve_variant(
        rng,
        params=query_params,
        gen_defaults=_GEN_DEFAULTS,
        supported_variants=QUERY_IDS,
        explicit_key="distance_rank_query",
        weights_key="distance_rank_query_weights",
    )
    return str(query_id), dict(probabilities)


def _resolve_answer_label(rng, *, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    explicit_label = params.get("answer_label")
    if explicit_label is not None:
        value = str(explicit_label).strip().upper()
        if value not in set(OPTION_LABELS):
            raise ValueError(f"answer_label must be one of {OPTION_LABELS}")
        return value, uniform_string_probability_map(OPTION_LABELS, selected=value)
    explicit_index = params.get("answer_index")
    if explicit_index is not None:
        index = int(explicit_index)
        if index < 0 or index >= len(OPTION_LABELS):
            raise ValueError("answer_index must be in 0..5")
        value = str(OPTION_LABELS[index])
        return value, uniform_string_probability_map(OPTION_LABELS, selected=value)
    value = str(rng.choice(OPTION_LABELS))
    return value, uniform_string_probability_map(OPTION_LABELS)


def _resolve_distractor_count(rng, *, params: Mapping[str, Any]) -> Tuple[int, Dict[str, float]]:
    low = int(params.get("distractor_count_min", group_default(_GEN_DEFAULTS, "distractor_count_min", _DEFAULTS.distractor_count_min)))
    high = int(params.get("distractor_count_max", group_default(_GEN_DEFAULTS, "distractor_count_max", _DEFAULTS.distractor_count_max)))
    if low < 0 or high < low:
        raise ValueError("invalid distractor_count_min/distractor_count_max")
    explicit = params.get("distractor_count")
    support = tuple(range(int(low), int(high) + 1))
    if explicit is not None:
        value = int(explicit)
        if value not in support:
            raise ValueError("distractor_count is outside configured support")
        return value, _uniform_int_probability_map(support, selected=value)
    value = int(rng.randint(int(low), int(high)))
    return value, _uniform_int_probability_map(support)


def _resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> Dict[str, Any]:
    """Resolve distance-rank rendering parameters and label readability settings."""
    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=_RENDER_DEFAULTS,
        fallback_defaults=_DEFAULTS,
        instance_seed=int(instance_seed),
    )
    for key in (
        "distance_rank_margin_px",
        "center_distance_min_px",
        "center_distance_gap_jitter_px",
        "icon_collision_gap_px",
        "candidate_label_font_size_px",
        "candidate_label_padding_px",
        "candidate_label_gap_px",
    ):
        render_params[key] = int(params.get(key, group_default(_RENDER_DEFAULTS, key, getattr(_DEFAULTS, key))))
    for key in ("candidate_label_color_rgb", "candidate_label_background_rgb", "candidate_label_border_rgb"):
        render_params[key] = resolve_icon_rgb_param(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            key=key,
            fallback=getattr(_DEFAULTS, key),
            instance_seed=int(instance_seed),
        )
    candidate_label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:candidate_label_text",
        role="named_field_candidate_label_text",
        surface_rgbs=(tuple(int(value) for value in render_params["candidate_label_background_rgb"]),),
        preferred_rgbs=(tuple(int(value) for value in render_params["candidate_label_color_rgb"]),),
    )
    render_params["candidate_label_color_rgb"] = tuple(int(value) for value in candidate_label_style.fill_rgb)
    render_params["candidate_label_stroke_rgb"] = tuple(int(value) for value in candidate_label_style.stroke_rgb)
    candidate_label_record = candidate_label_style.metadata()
    candidate_label_record["stroke_rgb"] = list(render_params["candidate_label_stroke_rgb"])
    previous_legibility = render_params.get("text_legibility")
    previous_records = []
    if isinstance(previous_legibility, Mapping) and isinstance(previous_legibility.get("records"), list):
        previous_records = [dict(record) for record in previous_legibility["records"] if isinstance(record, Mapping)]
    render_params["text_legibility"] = text_legibility_summary_from_records(
        [*previous_records, candidate_label_record]
    )
    return render_params


def _sample_size(rng, *, low: int, high: int) -> int:
    return int(rng.randint(min(int(low), int(high)), max(int(low), int(high))))


def _sample_non_reference_combo(
    rng,
    *,
    shape_support: Sequence[str],
    color_support: Sequence[str],
    reference_shape_id: str,
    reference_color_name: str,
) -> Tuple[str, str]:
    for _ in range(200):
        shape_id = str(rng.choice(tuple(str(value) for value in shape_support)))
        color_name = str(rng.choice(tuple(str(value) for value in color_support)))
        if shape_id != str(reference_shape_id) or color_name != str(reference_color_name):
            return shape_id, color_name
    raise RuntimeError("failed to sample non-reference icon color/shape combo")


def _sample_icon_plans(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_params: Mapping[str, Any],
    answer_label: str,
    answer_rank: int,
    distractor_count: int,
) -> Tuple[Tuple[_IconPlan, ...], str, str, str, Tuple[Tuple[int, int, int], ...]]:
    """Sample the reference, labeled candidates, and distractors for distance ranking."""
    shape_support = _shape_support(params)
    color_support = _color_support(params)
    fill_support = _fill_style_support(params)
    fill_probs = resolve_named_icon_fill_style_probabilities(params, _GEN_DEFAULTS, fill_support, default_weights=DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS)

    reference_shape_id = str(params.get("reference_shape_id", rng.choice(shape_support)))
    if reference_shape_id not in set(shape_support):
        raise ValueError(f"unsupported reference_shape_id: {reference_shape_id}")
    reference_color_name = str(params.get("reference_color_name", rng.choice(color_support))).strip().lower()
    if reference_color_name not in set(color_support):
        raise ValueError(f"unsupported reference_color_name: {reference_color_name}")
    reference_rgb = tuple(int(channel) for channel in named_color(reference_color_name))

    remaining_labels = [str(label) for label in OPTION_LABELS if str(label) != str(answer_label)]
    rng.shuffle(remaining_labels)
    labels_by_rank: List[str] = []
    for rank in range(len(OPTION_LABELS)):
        labels_by_rank.append(str(answer_label) if int(rank) == int(answer_rank) else str(remaining_labels.pop()))

    reference_size = _sample_size(
        rng,
        low=int(render_params.get("reference_icon_size_min_px", _DEFAULTS.reference_icon_size_min_px)),
        high=int(render_params.get("reference_icon_size_max_px", _DEFAULTS.reference_icon_size_max_px)),
    )
    reference_noise, reference_noise_seed = sample_icon_instance_noise(
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:reference",
        render_params=render_params,
    )
    plans: List[_IconPlan] = [
        _IconPlan(
            role="reference",
            label="",
            shape_id=str(reference_shape_id),
            color_name=str(reference_color_name),
            tint_rgb=reference_rgb,
            fill_style="solid",
            nominal_size_px=int(reference_size),
            rotation_degrees=rotation_for_named_shape(rng, str(reference_shape_id)),
            noise_edits=tuple(reference_noise),
            noise_seed=int(reference_noise_seed),
        )
    ]

    for rank, label in enumerate(labels_by_rank):
        shape_id, color_name = _sample_non_reference_combo(
            rng,
            shape_support=shape_support,
            color_support=color_support,
            reference_shape_id=str(reference_shape_id),
            reference_color_name=str(reference_color_name),
        )
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:candidate:{label}",
            render_params=render_params,
        )
        plans.append(
            _IconPlan(
                role="candidate",
                label=str(label),
                shape_id=str(shape_id),
                color_name=str(color_name),
                tint_rgb=tuple(int(channel) for channel in named_color(str(color_name))),
                fill_style=sample_procedural_named_icon_fill_style(rng, support=fill_support, probabilities=fill_probs),
                nominal_size_px=_sample_size(
                    rng,
                    low=int(render_params["scene_icon_size_min_px"]),
                    high=int(render_params["scene_icon_size_max_px"]),
                ),
                rotation_degrees=rotation_for_named_shape(rng, str(shape_id)),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )

    for index in range(int(distractor_count)):
        shape_id, color_name = _sample_non_reference_combo(
            rng,
            shape_support=shape_support,
            color_support=color_support,
            reference_shape_id=str(reference_shape_id),
            reference_color_name=str(reference_color_name),
        )
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:distractor:{index}",
            render_params=render_params,
        )
        plans.append(
            _IconPlan(
                role="distractor",
                label="",
                shape_id=str(shape_id),
                color_name=str(color_name),
                tint_rgb=tuple(int(channel) for channel in named_color(str(color_name))),
                fill_style=sample_procedural_named_icon_fill_style(rng, support=fill_support, probabilities=fill_probs),
                nominal_size_px=_sample_size(
                    rng,
                    low=int(render_params["scene_icon_size_min_px"]),
                    high=int(render_params["scene_icon_size_max_px"]),
                ),
                rotation_degrees=rotation_for_named_shape(rng, str(shape_id)),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )

    sampled_palette_rgb = tuple(tuple(int(channel) for channel in named_color(color_name)) for color_name in color_support)
    reference_description = (
        f"{format_named_color_with_hex(reference_color_name, reference_rgb)} "
        f'"{procedural_named_icon_display_name(reference_shape_id)}" icon'
    )
    return tuple(plans), str(reference_shape_id), str(reference_color_name), str(reference_description), sampled_palette_rgb










@register_task
class IconsRelationNamedReferenceDistanceRankLabelTask:
    """Select the labeled named icon nearest/farthest/second-nearest to a unique reference."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations')
    domain = "icons"
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic named-reference distance-rank instance."""

        sample_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
        query_id, query_probabilities = _resolve_query(sample_rng, params=params)
        answer_rank = int(RANK_BY_QUERY_ID[str(query_id)])
        answer_label, answer_label_probabilities = _resolve_answer_label(sample_rng, params=params)
        candidate_count = int(params.get("candidate_count", group_default(_GEN_DEFAULTS, "candidate_count", _DEFAULTS.candidate_count)))
        if int(candidate_count) != len(OPTION_LABELS):
            raise ValueError("task_icons__named_field__reference_distance_rank_label requires candidate_count=6")
        distractor_count, distractor_count_probabilities = _resolve_distractor_count(sample_rng, params=params)
        render_params = _resolve_render_params(params, instance_seed=int(instance_seed))

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                plans, _reference_shape_id, _reference_color_name, reference_description, sampled_palette_rgb = _sample_icon_plans(
                    sample_rng,
                    instance_seed=int(instance_seed),
                    params=params,
                    render_params=render_params,
                    answer_label=str(answer_label),
                    answer_rank=int(answer_rank),
                    distractor_count=int(distractor_count),
                )
                scene_payload, image = _render_placed_scene(
                    rng=sample_rng,
                    query_name=str(query_id),
                    answer_label=str(answer_label),
                    answer_rank=int(answer_rank),
                    plans=plans,
                    reference_description=str(reference_description),
                    sampled_palette_rgb=tuple(sampled_palette_rgb),
                    distractor_count=int(distractor_count),
                    render_params=render_params,
                    option_labels=OPTION_LABELS,
                    angle_pool_degrees=_ANGLE_POOL_DEGREES,
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError("failed to generate task_icons__named_field__reference_distance_rank_label instance") from last_error

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                f"question_text_{scene_payload.query_key}",
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        question_text = str(prompt_defaults[f"question_text_{scene_payload.query_key}"]).format(
            reference_description=str(scene_payload.reference_description)
        )
        prompt_selection = render_task_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            slots={
                "object_description": str(prompt_defaults["object_description"]),
                "question_text": str(question_text),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint"]),
                "answer_hint": str(prompt_defaults["answer_hint"]),
                "json_example": str(prompt_defaults["json_example"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        candidate_by_label = {str(icon.label): icon for icon in scene_payload.candidate_icons}
        answer_icon = candidate_by_label[str(scene_payload.answer_label)]
        annotation_artifacts = icon_bbox_map_annotation(
            {
                "reference_icon": scene_payload.reference_icon.bbox_xyxy,
                "selected_candidate": answer_icon.bbox_xyxy,
            }
        )
        answer_gt = TypedValue(type="option_letter", value=str(scene_payload.answer_label))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=dict(annotation_artifacts["annotation_value"]),
        )
        trace_payload = build_distance_rank_trace_payload(
            scene_payload=scene_payload,
            render_params=render_params,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            annotation_artifacts=annotation_artifacts,
            answer_icon=answer_icon,
            query_probabilities=query_probabilities,
            answer_label_probabilities=answer_label_probabilities,
            distractor_count_probabilities=distractor_count_probabilities,
            option_labels=OPTION_LABELS,
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(scene_payload.query_key),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )
