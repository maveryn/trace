"""Count named-shape icons closer to one of two prompt-named references."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.color_format import format_named_color_with_hex
from ...shared.config_defaults import group_default, required_group_defaults, split_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.named_colors import available_named_colors, named_color
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_task_prompt_variants
from ...shared.weighted_sampling import sample_weighted_value, weighted_probability_map
from ..shared.defaults import ICON_SHARED_DEFAULTS
from ..shared.icon_noise import serialize_icon_noise_edits
from ..shared.icon_scene import BBox, draw_single_panel, resolve_single_panel_layout, single_panel_geometry_to_trace
from ..shared.icon_task_rendering import icon_render_style_trace, resolve_icon_render_params, sample_icon_instance_noise
from ..shared.procedural_named_icon_field_scene import (
    SCENE_ID,
    bbox_center_float,
    bbox_from_center_dimensions,
    bbox_inside,
    boxes_overlap,
    render_planned_named_icon_sprite,
    resolve_named_icon_fill_style_probabilities,
    resolve_named_icon_fill_style_support,
    resolve_named_icon_int_bounds,
    rotation_for_named_shape,
    uniform_string_probability_map,
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
from .shared.output import serialize_closer_reference_icon as _serialize_icon
from .shared.annotations import bbox_set_from_bboxes
from .shared.rendering import (
    render_closer_reference_scene as _render_scene,
)
from .shared.state import (
    CloserReferenceIconPlan as _IconPlan,
    CloserReferenceSampleSpec as _SampleSpec,
    CloserReferenceScenePayload as _ScenePayload,
)


TASK_ID = "task_icons__named_field__closer_to_reference_count"

QUERY_ID = "closer_to_reference_count"
QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
QUERIED_REFERENCE_LABELS: Tuple[str, ...] = ("A", "B")


@dataclass(frozen=True)
class _TaskDefaults:
    target_icon_count_min: int = 4
    target_icon_count_max: int = 8
    target_answer_min: int = 0
    target_answer_max: int = 4
    canvas_width: int = 800
    canvas_height: int = 480
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 72
    reference_icon_size_min_px: int = 56
    reference_icon_size_max_px: int = 72
    reference_icon_size_px: int = 64
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = 260
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    palette_size_min: int = 8
    palette_size_max: int = 10
    color_channel_min: int = 24
    color_channel_max: int = 230
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
    reference_axis_degrees: Tuple[int, ...] = (0, 35, 90, 145)
    distance_margin_px: int = 42
    icon_collision_gap_px: int = 8


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
    if len(values) < 3:
        raise ValueError("shape_id_support must include at least three shapes")
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
    if len(values) < 3:
        raise ValueError("named_color_support must include at least three named colors")
    return values




def _resolve_public_query(params: Mapping[str, Any]) -> Tuple[str, Dict[str, float]]:
    explicit_query = params.get("query_id")
    if explicit_query is not None and str(explicit_query) != QUERY_ID:
        raise ValueError(f"query_id must be {QUERY_ID}")
    return QUERY_ID, {QUERY_ID: 1.0}


def _resolve_queried_reference_label(params: Mapping[str, Any], rng) -> Tuple[str, Dict[str, float]]:
    explicit = params.get("queried_reference_label", params.get("reference_label"))
    if explicit is not None:
        label = str(explicit).strip().upper()
        if label not in set(QUERIED_REFERENCE_LABELS):
            raise ValueError(f"queried_reference_label must be one of {QUERIED_REFERENCE_LABELS}")
        return label, {label: 1.0}
    probabilities = weighted_probability_map(
        QUERIED_REFERENCE_LABELS,
        params.get("queried_reference_label_weights", group_default(_GEN_DEFAULTS, "queried_reference_label_weights", None)),
    )
    label = str(sample_weighted_value(rng, QUERIED_REFERENCE_LABELS, probabilities))
    return label, dict(probabilities)


def _axis_support(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("reference_axis_degrees", group_default(_GEN_DEFAULTS, "reference_axis_degrees", _DEFAULTS.reference_axis_degrees))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = _DEFAULTS.reference_axis_degrees
    values = tuple(dict.fromkeys(int(value) for value in raw))
    if not values:
        raise ValueError("reference_axis_degrees resolved no axis values")
    return values



def _uniform_axis_probability_map(values: Sequence[int], *, selected: int | None = None) -> Dict[str, float]:
    support = tuple(int(value) for value in values)
    if selected is not None:
        return {str(int(selected)): 1.0}
    probability = 1.0 / float(len(support))
    return {str(int(value)): probability for value in support}


def _choose_other(rng, values: Sequence[str], excluded: Sequence[str]) -> str:
    excluded_set = {str(value) for value in excluded}
    candidates = [str(value) for value in values if str(value) not in excluded_set]
    if not candidates:
        raise ValueError("no alternate value available")
    return str(rng.choice(candidates))



def _sample_counts(params: Mapping[str, Any], rng) -> Tuple[int, int, int, Dict[str, float], Dict[str, float]]:
    """Resolve target/reference counts while preserving the requested distance comparison answer."""
    answer_min, answer_max = resolve_named_icon_int_bounds(params, _GEN_DEFAULTS, "target_answer_min", "target_answer_max", _DEFAULTS.target_answer_min, _DEFAULTS.target_answer_max)
    target_min, target_max = resolve_named_icon_int_bounds(params, _GEN_DEFAULTS,
        "target_icon_count_min",
        "target_icon_count_max",
        _DEFAULTS.target_icon_count_min,
        _DEFAULTS.target_icon_count_max,
    )
    if int(target_min) < 1:
        raise ValueError("target_icon_count_min must be positive")
    answer_support = tuple(range(int(answer_min), int(answer_max) + 1))
    feasible_answers = tuple(int(value) for value in answer_support if int(value) <= int(target_max))
    if not feasible_answers:
        raise ValueError("target answer support has no feasible target icon counts")
    answer_probabilities = weighted_probability_map(
        feasible_answers,
        params.get("target_answer_weights", group_default(_GEN_DEFAULTS, "target_answer_weights", None)),
    )
    explicit_answer = params.get("target_answer", params.get("answer"))
    if explicit_answer is not None:
        target_answer = int(explicit_answer)
        if target_answer not in set(feasible_answers):
            raise ValueError(f"target_answer must be in feasible support {feasible_answers}")
    else:
        target_answer = int(sample_weighted_value(rng, feasible_answers, answer_probabilities))

    target_count_min = max(int(target_min), int(target_answer))
    target_count_support = tuple(range(int(target_count_min), int(target_max) + 1))
    explicit_target_count = params.get("target_icon_count", params.get("object_count"))
    if explicit_target_count is not None:
        target_icon_count = int(explicit_target_count)
        if target_icon_count not in set(target_count_support):
            raise ValueError(f"target_icon_count must be in {target_count_support}")
    else:
        target_icon_count = int(rng.choice(target_count_support))
    other_side_count = int(target_icon_count) - int(target_answer)
    return (
        int(target_answer),
        int(other_side_count),
        int(target_icon_count),
        dict(uniform_probability_map(feasible_answers, selected=int(target_answer)) if explicit_answer is not None else answer_probabilities),
        dict(uniform_probability_map(target_count_support, selected=int(target_icon_count) if explicit_target_count is not None else None)),
    )


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Resolve task sampling axes and construct a feasible named-field semantic scene."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    shape_support = _shape_support(params)
    color_support = _color_support(params)
    fill_support = resolve_named_icon_fill_style_support(params, _GEN_DEFAULTS, fallback_support=_DEFAULTS.named_icon_fill_style_support)
    fill_probs = resolve_named_icon_fill_style_probabilities(params, _GEN_DEFAULTS, fill_support, default_weights=DEFAULT_PROCEDURAL_NAMED_ICON_FILL_STYLE_WEIGHTS)
    query_id, query_probabilities = _resolve_public_query(params)
    queried_reference_label, queried_reference_label_probabilities = _resolve_queried_reference_label(params, rng)
    opposite_reference_label = "B" if queried_reference_label == "A" else "A"
    target_answer, opposite_count, target_icon_count, answer_probs, target_count_probs = _sample_counts(params, rng)

    explicit_target_shape = params.get("target_shape_id", params.get("shape_id"))
    if explicit_target_shape is not None:
        target_shape_id = str(explicit_target_shape)
        if target_shape_id not in set(shape_support):
            raise ValueError(f"target_shape_id must be one of {shape_support}")
    else:
        target_shape_id = str(rng.choice(shape_support))
    reference_a_shape = str(params.get("reference_a_shape_id", _choose_other(rng, shape_support, (target_shape_id,))))
    if reference_a_shape not in set(shape_support) or reference_a_shape == target_shape_id:
        raise ValueError("reference_a_shape_id must be supported and distinct from target_shape_id")
    reference_b_shape = str(params.get("reference_b_shape_id", _choose_other(rng, shape_support, (target_shape_id, reference_a_shape))))
    if reference_b_shape not in set(shape_support) or reference_b_shape in {target_shape_id, reference_a_shape}:
        raise ValueError("reference_b_shape_id must be supported and distinct from target/reference A")

    reference_a_color = str(params.get("reference_a_color_name", rng.choice(color_support))).strip().lower()
    if reference_a_color not in set(color_support):
        raise ValueError(f"reference_a_color_name must be one of {color_support}")
    reference_b_color = str(params.get("reference_b_color_name", _choose_other(rng, color_support, (reference_a_color,)))).strip().lower()
    if reference_b_color not in set(color_support) or reference_b_color == reference_a_color:
        raise ValueError("reference_b_color_name must be supported and distinct from reference A color")

    axis_support = _axis_support(params)
    explicit_axis = params.get("reference_axis_degrees_value", params.get("reference_axis_degrees_selected"))
    if explicit_axis is not None:
        axis_degrees = int(explicit_axis)
        if axis_degrees not in set(axis_support):
            raise ValueError(f"reference axis must be one of {axis_support}")
    else:
        axis_degrees = int(rng.choice(axis_support))

    def make_plan(
        *,
        role: str,
        label: str,
        shape_id: str,
        color_name: str,
        desired_closer_label: str,
        index_namespace: str,
    ) -> _IconPlan:
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}:{index_namespace}",
            render_params=_resolve_render_params(params, instance_seed=instance_seed),
        )
        if role == "reference":
            low = int(params.get("reference_icon_size_min_px", group_default(_RENDER_DEFAULTS, "reference_icon_size_min_px", _DEFAULTS.reference_icon_size_min_px)))
            high = int(params.get("reference_icon_size_max_px", group_default(_RENDER_DEFAULTS, "reference_icon_size_max_px", _DEFAULTS.reference_icon_size_max_px)))
            fill_style = "solid"
        else:
            low = int(params.get("scene_icon_size_min_px", group_default(_RENDER_DEFAULTS, "scene_icon_size_min_px", _DEFAULTS.scene_icon_size_min_px)))
            high = int(params.get("scene_icon_size_max_px", group_default(_RENDER_DEFAULTS, "scene_icon_size_max_px", _DEFAULTS.scene_icon_size_max_px)))
            fill_style = sample_procedural_named_icon_fill_style(rng, support=fill_support, probabilities=fill_probs)
        return _IconPlan(
            role=str(role),
            label=str(label),
            shape_id=str(shape_id),
            color_name=str(color_name),
            tint_rgb=tuple(int(channel) for channel in named_color(str(color_name))),
            fill_style=str(fill_style),
            nominal_size_px=int(rng.randint(min(low, high), max(low, high))),
            rotation_degrees=rotation_for_named_shape(rng, str(shape_id)),
            desired_closer_label=str(desired_closer_label),
            noise_edits=tuple(noise_edits),
            noise_seed=int(noise_seed),
        )

    target_color_values = tuple(str(value) for value in color_support)
    plans = [
        make_plan(role="reference", label="A", shape_id=reference_a_shape, color_name=reference_a_color, desired_closer_label="", index_namespace="reference_a"),
        make_plan(role="reference", label="B", shape_id=reference_b_shape, color_name=reference_b_color, desired_closer_label="", index_namespace="reference_b"),
    ]
    target_roles = [queried_reference_label] * int(target_answer) + [opposite_reference_label] * int(opposite_count)
    rng.shuffle(target_roles)
    for index, closer_label in enumerate(target_roles):
        plans.append(
            make_plan(
                role="target",
                label="",
                shape_id=target_shape_id,
                color_name=str(rng.choice(target_color_values)),
                desired_closer_label=str(closer_label),
                index_namespace=f"target_{int(index)}",
            )
        )

    return _SampleSpec(
        query_key=str(query_id),
        queried_reference_label=str(queried_reference_label),
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        reference_a_shape_name=procedural_named_icon_display_name(str(reference_a_shape)),
        reference_b_shape_name=procedural_named_icon_display_name(str(reference_b_shape)),
        target_answer=int(target_answer),
        target_icon_count=int(target_icon_count),
        closer_count_by_reference={
            str(queried_reference_label): int(target_answer),
            str(opposite_reference_label): int(opposite_count),
        },
        plans=tuple(plans),
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in named_color(color_name)) for color_name in color_support),
        query_probabilities=dict(query_probabilities),
        shape_probabilities=uniform_string_probability_map(shape_support, selected=str(target_shape_id) if explicit_target_shape is not None else None),
        color_probabilities=uniform_string_probability_map(color_support),
        target_answer_probabilities=dict(answer_probs),
        target_icon_count_probabilities=dict(target_count_probs),
        fill_style_support=tuple(fill_support),
        fill_style_probabilities=dict(fill_probs),
        reference_axis_probabilities=_uniform_axis_probability_map(axis_support, selected=int(axis_degrees) if explicit_axis is not None else None),
        queried_reference_label_probabilities=dict(queried_reference_label_probabilities),
    )


def _resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> Dict[str, Any]:
    """Resolve distance-rank rendering parameters and label readability settings."""
    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=_RENDER_DEFAULTS,
        fallback_defaults=_DEFAULTS,
        instance_seed=int(instance_seed),
    )
    for key in (
        "distance_margin_px",
        "icon_collision_gap_px",
    ):
        render_params[key] = int(params.get(key, group_default(_RENDER_DEFAULTS, key, getattr(_DEFAULTS, key))))
    return render_params


@register_task
class IconsCountingNamedShapeCloserToReferenceCountTask:
    """Count target-shape icons closer to one of two prompt-named references."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "icons"
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance by binding sampling, rendering, prompt, answer, and annotation."""
        render_params = _resolve_render_params(params, instance_seed=int(instance_seed))
        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene: _ScenePayload | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params)
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                scene = _render_scene(
                    rng=scene_rng,
                    sample=sample,
                    render_params=render_params,
                )
                break
            except Exception as exc:  # pragma: no cover - exercised by smoke tests.
                last_error = exc
                sample = None
                scene = None
        if sample is None or scene is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        target_icons = tuple(icon for icon in scene.icons if str(icon.role) == "target")
        reference_icons = tuple(icon for icon in scene.icons if str(icon.role) == "reference")
        counted_icons = tuple(icon for icon in target_icons if bool(icon.counted))
        if len(reference_icons) != 2 or len(target_icons) != int(sample.target_icon_count):
            raise RuntimeError("closer-reference scene has inconsistent icon counts")
        if len(counted_icons) != int(sample.target_answer):
            raise RuntimeError("closer-reference rendered answer does not match sampled answer")

        question_key = f"question_text_{sample.query_key}"
        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                question_key,
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
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
                "question_text": str(prompt_defaults[question_key]).format(
                    target_shape_name=str(sample.target_shape_name),
                    queried_reference_shape_name=(
                        str(sample.reference_a_shape_name)
                        if str(sample.queried_reference_label) == "A"
                        else str(sample.reference_b_shape_name)
                    ),
                    other_reference_shape_name=(
                        str(sample.reference_b_shape_name)
                        if str(sample.queried_reference_label) == "A"
                        else str(sample.reference_a_shape_name)
                    ),
                ),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint"]).format(target_shape_name=str(sample.target_shape_name)),
                "answer_hint": str(prompt_defaults["answer_hint"]),
                "json_example": str(prompt_defaults["json_example"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        annotation_bboxes = tuple(icon.bbox_xyxy for icon in counted_icons)
        annotation_artifacts = bbox_set_from_bboxes(annotation_bboxes)
        counted_instance_ids = tuple(str(icon.instance_id) for icon in counted_icons)
        reference_by_label = {str(icon.label): icon for icon in reference_icons}
        closer_counts = Counter(str(icon.closer_reference_label) for icon in target_icons)
        closer_count_by_reference = {
            "A": int(closer_counts.get("A", 0)),
            "B": int(closer_counts.get("B", 0)),
        }
        shape_counts = Counter(str(icon.shape_id) for icon in scene.icons)
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_named_shape_closer_to_reference_field",
                "scene_id": SCENE_ID,
                "entities": [_serialize_icon(icon) for icon in scene.icons],
                "relations": {
                    "counting_rule": "target_shape_icons_closer_to_queried_reference",
                    "target_shape_id": str(sample.target_shape_id),
                    "target_shape_name": str(sample.target_shape_name),
                    "reference_a_shape_name": str(sample.reference_a_shape_name),
                    "reference_b_shape_name": str(sample.reference_b_shape_name),
                    "queried_reference_label": str(sample.queried_reference_label),
                    "closer_count_by_reference": dict(closer_count_by_reference),
                    "target_answer": int(sample.target_answer),
                    "target_icon_count": int(sample.target_icon_count),
                    "reference_axis_degrees": int(scene.reference_axis_degrees),
                    "distance_margin_px": int(render_params["distance_margin_px"]),
                    "reference_ids": {
                        "A": str(reference_by_label["A"].instance_id),
                        "B": str(reference_by_label["B"].instance_id),
                    },
                    "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene.panel_geometry),
                },
            },
            "query_spec": {
                "query_id": str(sample.query_key),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "query_id": str(sample.query_key),
                    "target_shape_id": str(sample.target_shape_id),
                    "target_shape_name": str(sample.target_shape_name),
                    "target_answer": int(sample.target_answer),
                    "target_icon_count": int(sample.target_icon_count),
                    "queried_reference_label": str(sample.queried_reference_label),
                    "queried_reference_label_probabilities": dict(sample.queried_reference_label_probabilities),
                    "closer_count_by_reference": dict(closer_count_by_reference),
                    "reference_axis_degrees": int(scene.reference_axis_degrees),
                    "query_probabilities": dict(sample.query_probabilities),
                    "shape_id_support": list(_shape_support(params)),
                    "named_color_support": list(_color_support(params)),
                    "shape_probabilities": dict(sample.shape_probabilities),
                    "color_probabilities": dict(sample.color_probabilities),
                    "target_answer_probabilities": dict(sample.target_answer_probabilities),
                    "target_icon_count_probabilities": dict(sample.target_icon_count_probabilities),
                    "named_icon_fill_style_support": list(sample.fill_style_support),
                    "fill_style_probabilities": dict(sample.fill_style_probabilities),
                    "reference_axis_probabilities": dict(sample.reference_axis_probabilities),
                },
            },
            "render_spec": {
                "canvas_size": list(scene.panel_geometry["canvas_size"]),
                "coord_space": "pixel",
                "scene_id": SCENE_ID,
                "panel_geometry": dict(scene.panel_geometry),
                "style": {
                    **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=sample.sampled_palette_rgb),
                    "reference_axis_degrees": int(scene.reference_axis_degrees),
                    "distance_margin_px": int(render_params["distance_margin_px"]),
                    "semantic_color_palette": [
                        {
                            "name": str(name),
                            "rgb": [int(channel) for channel in rgb],
                            "label": format_named_color_with_hex(str(name), rgb),
                        }
                        for name, rgb in available_named_colors()
                    ],
                    "semantic_fill_style_support": list(resolve_named_icon_fill_style_support(params, _GEN_DEFAULTS, fallback_support=_DEFAULTS.named_icon_fill_style_support)),
                },
            },
            "render_map": {
                "image_id": "img0",
                "object_bboxes_px": {str(icon.instance_id): [int(value) for value in icon.bbox_xyxy] for icon in scene.icons},
                "counted_instance_ids": list(counted_instance_ids),
                "reference_instance_ids": {
                    "A": str(reference_by_label["A"].instance_id),
                    "B": str(reference_by_label["B"].instance_id),
                },
            },
            "execution_trace": {
                "scene_variant": "single_panel_named_shape_closer_to_reference_field",
                "query_id": str(sample.query_key),
                "question_format": "count_named_shape_icons_closer_to_named_reference",
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "reference_a_shape_name": str(sample.reference_a_shape_name),
                "reference_b_shape_name": str(sample.reference_b_shape_name),
                "queried_reference_label": str(sample.queried_reference_label),
                "queried_reference_label_probabilities": dict(sample.queried_reference_label_probabilities),
                "target_answer": int(sample.target_answer),
                "target_icon_count": int(sample.target_icon_count),
                "closer_count_by_reference": dict(closer_count_by_reference),
                "reference_axis_degrees": int(scene.reference_axis_degrees),
                "target_icon_ids": [str(icon.instance_id) for icon in target_icons],
                "counted_instance_ids": list(counted_instance_ids),
            },
            "witness_symbolic": {
                "target_shape_id": str(sample.target_shape_id),
                "target_shape_name": str(sample.target_shape_name),
                "reference_a_shape_name": str(sample.reference_a_shape_name),
                "reference_b_shape_name": str(sample.reference_b_shape_name),
                "queried_reference_label": str(sample.queried_reference_label),
                "answer": int(sample.target_answer),
                "counted_instance_ids": list(counted_instance_ids),
                "closer_count_by_reference": dict(closer_count_by_reference),
            },
            "projected_annotation": {
                **dict(annotation_artifacts["projected_annotation"]),
            },
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(sample.target_answer)),
            annotation_gt=TypedValue(
                type=str(annotation_artifacts["annotation_type"]),
                value=list(annotation_artifacts["annotation_value"]),
            ),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(sample.query_key),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        )


__all__ = ["IconsCountingNamedShapeCloserToReferenceCountTask", "QUERY_ID", "QUERY_IDS", "QUERIED_REFERENCE_LABELS"]
