"""Count prompt-named procedural icons inside or outside visible regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.scene_config import get_scene_defaults
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, required_group_defaults, split_generation_rendering_prompt_defaults
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_task_prompt_variants
from ...shared.weighted_sampling import sample_weighted_value, weighted_probability_map
from ..shared.defaults import ICON_SHARED_DEFAULTS
from ..shared.icon_scene import (
    BBox,
    resolve_single_panel_layout,
)
from ..shared.icon_style import sample_icon_palette
from ..shared.icon_task_rendering import resolve_icon_render_params
from ..shared.procedural_named_icon_field_scene import (
    SCENE_ID,
    resolve_named_icon_fill_style_probabilities,
    resolve_named_icon_fill_style_support,
    resolve_named_icon_int_bounds,
    uniform_string_probability_map,
)
from ..shared.procedural_named_icons import (
    PROCEDURAL_NAMED_ICON_FILL_STYLES,
    PROCEDURAL_NAMED_ICON_SHAPES,
    procedural_named_icon_display_name,
)
from .shared.layout import (
    sample_band_region as _sample_band_region,
    sample_box_region as _sample_box_region,
    sample_quadrant_region as _sample_quadrant_region,
    sample_shelf_region as _sample_shelf_region,
)
from .shared.output import build_scoped_region_trace_payload
from .shared.annotations import bbox_set_from_bboxes
from .shared.rendering import (
    render_scoped_region_scene as _render_scoped_region_scene,
)
from .shared.state import (
    RegionIconPlan as _IconPlan,
    RegionSpec as _RegionSpec,
    ScopedRegionScenePayload as _ScenePayload,
)


TASK_ID = "task_icons__named_field__scoped_attribute_count"

QUERY_IDS: Tuple[str, ...] = (
    "inside_shape_count",
    "outside_shape_count",
    "inside_band_count",
    "outside_band_count",
    "inside_quadrant_count",
    "inside_shelf_count",
)

_INSIDE_QUERY_IDS = {
    "inside_shape_count",
    "inside_band_count",
    "inside_quadrant_count",
    "inside_shelf_count",
}


@dataclass(frozen=True)
class _TaskDefaults:
    object_count_min: int = 10
    object_count_max: int = 18
    target_count_min: int = 1
    target_count_max: int = 6
    target_opposite_count_min: int = 1
    target_opposite_count_max: int = 3
    canvas_width: int = 800
    canvas_height: int = 480
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 96
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = 160
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
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
    region_shape_kinds: Tuple[str, ...] = ("rectangle", "ellipse")
    band_kinds: Tuple[str, ...] = ("vertical", "horizontal", "slanted_positive", "slanted_negative")
    quadrant_ids: Tuple[str, ...] = ("top_left", "top_right", "bottom_left", "bottom_right")
    shelf_count_min: int = 3
    shelf_count_max: int = 4
    region_boundary_margin_px: int = 18
    region_fill_rgb: Tuple[int, int, int] = (255, 226, 105)
    region_outline_rgb: Tuple[int, int, int] = (49, 103, 191)
    region_guide_rgb: Tuple[int, int, int] = (154, 164, 180)
    region_fill_alpha: int = 54
    region_outline_width_px: int = 3
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES


_DEFAULTS = _TaskDefaults()
_TASK_GROUP_DEFAULTS = get_scene_defaults("icons", "named_field")
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_generation_rendering_prompt_defaults(
    _TASK_GROUP_DEFAULTS if isinstance(_TASK_GROUP_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)



def _int_default(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(params.get(key, group_default(defaults, key, fallback)))


def _rgb_default(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, int, int]:
    raw = params.get(key, group_default(defaults, key, tuple(fallback)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) < 3:
        raw = tuple(fallback)
    return tuple(int(value) for value in raw[:3])


def _shape_support(params: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get("shape_id_support", group_default(_GEN_DEFAULTS, "shape_id_support", PROCEDURAL_NAMED_ICON_SHAPES))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("shape_id_support must be a sequence")
    values = tuple(str(value) for value in raw)
    unsupported = sorted(set(values) - set(PROCEDURAL_NAMED_ICON_SHAPES))
    if unsupported:
        raise ValueError(f"unsupported procedural named icon shapes: {unsupported}")
    support = tuple(dict.fromkeys(values))
    if len(support) < 5:
        raise ValueError("shape_id_support must include at least five shapes")
    return support


def _query_support(params: Mapping[str, Any]) -> Tuple[str, ...]:
    raw = params.get("named_region_query_ids", group_default(_GEN_DEFAULTS, "named_region_query_ids", QUERY_IDS))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("named_region_query_ids must be a sequence")
    values = tuple(dict.fromkeys(str(value) for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(QUERY_IDS))
    if unsupported:
        raise ValueError(f"unsupported named-region query ids: {unsupported}")
    if not values:
        raise ValueError("named_region_query_ids resolved no query ids")
    return values




def _string_support(
    params: Mapping[str, Any],
    *,
    key: str,
    fallback: Sequence[str],
    allowed: Sequence[str],
) -> Tuple[str, ...]:
    raw = params.get(key, group_default(_GEN_DEFAULTS, key, tuple(fallback)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = tuple(fallback)
    values = tuple(dict.fromkeys(str(value) for value in raw if str(value).strip()))
    unsupported = sorted(set(values) - set(allowed))
    if unsupported:
        raise ValueError(f"unsupported {key}: {unsupported}")
    if not values:
        raise ValueError(f"{key} resolved no values")
    return values







def _sample_region(
    *,
    rng,
    query_id: str,
    content_bbox: BBox,
    params: Mapping[str, Any],
) -> _RegionSpec:
    """Sample a visible region compatible with the selected scoped-count query."""
    if str(query_id) in {"inside_shape_count", "outside_shape_count"}:
        shape_support = _string_support(
            params,
            key="region_shape_kinds",
            fallback=_DEFAULTS.region_shape_kinds,
            allowed=("rectangle", "ellipse"),
        )
        explicit = params.get("region_shape_kind")
        shape_kind = str(explicit) if explicit is not None else str(rng.choice(shape_support))
        if shape_kind not in set(shape_support):
            raise ValueError(f"region_shape_kind must be one of {shape_support}")
        return _sample_box_region(
            rng,
            query_key=str(query_id),
            counts_inside=str(query_id) in _INSIDE_QUERY_IDS,
            content_bbox=content_bbox,
            shape_kind=str(shape_kind),
        )
    if str(query_id) in {"inside_band_count", "outside_band_count"}:
        band_support = _string_support(
            params,
            key="band_kinds",
            fallback=_DEFAULTS.band_kinds,
            allowed=("vertical", "horizontal", "slanted_positive", "slanted_negative"),
        )
        explicit = params.get("band_kind")
        band_kind = str(explicit) if explicit is not None else str(rng.choice(band_support))
        if band_kind not in set(band_support):
            raise ValueError(f"band_kind must be one of {band_support}")
        return _sample_band_region(
            rng,
            query_key=str(query_id),
            counts_inside=str(query_id) in _INSIDE_QUERY_IDS,
            content_bbox=content_bbox,
            band_kind=str(band_kind),
        )
    if str(query_id) == "inside_quadrant_count":
        quadrant_support = _string_support(
            params,
            key="quadrant_ids",
            fallback=_DEFAULTS.quadrant_ids,
            allowed=("top_left", "top_right", "bottom_left", "bottom_right"),
        )
        explicit = params.get("quadrant_id")
        quadrant_id = str(explicit) if explicit is not None else str(rng.choice(quadrant_support))
        if quadrant_id not in set(quadrant_support):
            raise ValueError(f"quadrant_id must be one of {quadrant_support}")
        return _sample_quadrant_region(rng, query_key=str(query_id), content_bbox=content_bbox, quadrant_id=str(quadrant_id))
    if str(query_id) == "inside_shelf_count":
        shelf_min = _int_default(params, _GEN_DEFAULTS, "shelf_count_min", _DEFAULTS.shelf_count_min)
        shelf_max = _int_default(params, _GEN_DEFAULTS, "shelf_count_max", _DEFAULTS.shelf_count_max)
        if shelf_min <= 0 or shelf_max < shelf_min:
            raise ValueError("invalid shelf_count_min/shelf_count_max")
        return _sample_shelf_region(
            rng,
            query_key=str(query_id),
            content_bbox=content_bbox,
            shelf_count_min=int(shelf_min),
            shelf_count_max=int(shelf_max),
        )
    raise ValueError(f"unsupported query_id: {query_id}")



def _make_scene(*, instance_seed: int, params: Mapping[str, Any], render_params: Mapping[str, Any], attempt: int) -> _ScenePayload:
    """Render a scoped named-icon field while enforcing inside/outside separation."""
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    query_support = _query_support(params)
    fill_style_support = resolve_named_icon_fill_style_support(params, _GEN_DEFAULTS, fallback_support=_DEFAULTS.named_icon_fill_style_support)
    fill_style_probabilities = resolve_named_icon_fill_style_probabilities(params, _GEN_DEFAULTS, fill_style_support)
    explicit_query = params.get("query_id", params.get("named_region_query_id"))
    if explicit_query is not None:
        query_id = str(explicit_query)
        if query_id not in set(query_support):
            raise ValueError(f"query_id must be one of {query_support}")
    else:
        query_id = str(rng.choice(query_support))
    region = _sample_region(rng=rng, query_id=str(query_id), content_bbox=content_bbox, params=params)
    shape_support = _shape_support(params)
    explicit_shape = params.get("shape_id", params.get("target_shape_id"))
    if explicit_shape is not None:
        target_shape_id = str(explicit_shape)
        if target_shape_id not in set(shape_support):
            raise ValueError(f"target shape must be one of {shape_support}")
    else:
        target_shape_id = str(rng.choice(shape_support))

    answer_min, answer_max = resolve_named_icon_int_bounds(params, _GEN_DEFAULTS, "target_count_min", "target_count_max", _DEFAULTS.target_count_min, _DEFAULTS.target_count_max)
    object_min, object_max = resolve_named_icon_int_bounds(params, _GEN_DEFAULTS, "object_count_min", "object_count_max", _DEFAULTS.object_count_min, _DEFAULTS.object_count_max)
    opposite_min, opposite_max = resolve_named_icon_int_bounds(params, _GEN_DEFAULTS,
        "target_opposite_count_min",
        "target_opposite_count_max",
        _DEFAULTS.target_opposite_count_min,
        _DEFAULTS.target_opposite_count_max,
    )
    if answer_min < 1:
        raise ValueError("named-shape region count uses target_count_min >= 1")
    target_support = tuple(range(int(answer_min), int(answer_max) + 1))
    target_count_probabilities = weighted_probability_map(
        target_support,
        params.get("target_count_weights", group_default(_GEN_DEFAULTS, "target_count_weights", None)),
    )
    explicit_target = params.get("target_count", params.get("target_answer"))
    if explicit_target is not None:
        target_count = int(explicit_target)
        if target_count not in set(target_support):
            raise ValueError(f"target_count must be in {target_support}")
    else:
        target_count = int(sample_weighted_value(rng, target_support, target_count_probabilities))
    target_opposite_count = int(rng.randint(int(opposite_min), int(opposite_max)))
    min_object_count = max(int(object_min), int(target_count) + int(target_opposite_count) + 4)
    if min_object_count > int(object_max):
        raise ValueError("object_count range cannot support requested target/opposite counts")
    object_support = tuple(range(int(min_object_count), int(object_max) + 1))
    explicit_object = params.get("object_count")
    if explicit_object is not None:
        object_count = int(explicit_object)
        if object_count not in set(object_support):
            raise ValueError(f"object_count must be in {object_support}")
    else:
        object_count = int(rng.choice(object_support))

    counts_inside = bool(region.counts_inside)
    plans: list[_IconPlan] = []
    for _ in range(int(target_count)):
        plans.append(_IconPlan(shape_id=str(target_shape_id), desired_inside_region=bool(counts_inside), is_target_shape=True))
    for _ in range(int(target_opposite_count)):
        plans.append(_IconPlan(shape_id=str(target_shape_id), desired_inside_region=not bool(counts_inside), is_target_shape=True))
    distractor_shapes = [str(value) for value in shape_support if str(value) != str(target_shape_id)]
    rng.shuffle(distractor_shapes)
    non_target_count = int(object_count) - len(plans)
    for index in range(int(non_target_count)):
        plans.append(
            _IconPlan(
                shape_id=str(rng.choice(distractor_shapes)),
                desired_inside_region=bool(index % 2 == 0),
                is_target_shape=False,
            )
        )
    rng.shuffle(plans)

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    palette = sample_icon_palette(
        rng,
        palette_size=int(palette_size),
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(value) for value in render_params["background_color_rgb"]),
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["panel_border_rgb"]),
            tuple(int(value) for value in render_params["header_text_rgb"]),
            tuple(int(value) for value in render_params["region_fill_rgb"]),
            tuple(int(value) for value in render_params["region_outline_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )

    return _render_scoped_region_scene(
        rng=rng,
        instance_seed=int(instance_seed),
        namespace=TASK_ID,
        region=region,
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        target_count=int(target_count),
        object_count=int(object_count),
        plans=tuple(plans),
        render_params=render_params,
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in palette),
        query_probabilities=uniform_string_probability_map(query_support, selected=str(query_id) if explicit_query is not None else None),
        shape_probabilities=uniform_string_probability_map(shape_support, selected=str(target_shape_id) if explicit_shape is not None else None),
        target_count_probabilities=dict(uniform_probability_map(target_support, selected=int(target_count)) if explicit_target is not None else target_count_probabilities),
        object_count_probabilities=dict(uniform_probability_map(object_support, selected=int(object_count) if explicit_object is not None else None)),
        fill_style_support=tuple(fill_style_support),
        fill_style_probabilities=dict(fill_style_probabilities),
    )




@register_task
class IconsCountingNamedShapeRegionCountTask:
    """Count named procedural icon shapes satisfying a visible region filter."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "icons"
    supported_query_ids = QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one task instance by binding sampling, rendering, prompt, answer, and annotation."""
        render_params = resolve_icon_render_params(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        render_params["region_shape_kinds"] = tuple(
            str(value)
            for value in params.get("region_shape_kinds", group_default(_GEN_DEFAULTS, "region_shape_kinds", _DEFAULTS.region_shape_kinds))
        )
        render_params["band_kinds"] = tuple(str(value) for value in params.get("band_kinds", group_default(_GEN_DEFAULTS, "band_kinds", _DEFAULTS.band_kinds)))
        render_params["quadrant_ids"] = tuple(
            str(value)
            for value in params.get("quadrant_ids", group_default(_GEN_DEFAULTS, "quadrant_ids", _DEFAULTS.quadrant_ids))
        )
        render_params["region_boundary_margin_px"] = _int_default(
            params,
            _RENDER_DEFAULTS,
            "region_boundary_margin_px",
            _DEFAULTS.region_boundary_margin_px,
        )
        render_params["region_fill_rgb"] = _rgb_default(params, _RENDER_DEFAULTS, "region_fill_rgb", _DEFAULTS.region_fill_rgb)
        render_params["region_outline_rgb"] = _rgb_default(params, _RENDER_DEFAULTS, "region_outline_rgb", _DEFAULTS.region_outline_rgb)
        render_params["region_guide_rgb"] = _rgb_default(params, _RENDER_DEFAULTS, "region_guide_rgb", _DEFAULTS.region_guide_rgb)
        render_params["region_fill_alpha"] = _int_default(params, _RENDER_DEFAULTS, "region_fill_alpha", _DEFAULTS.region_fill_alpha)
        render_params["region_outline_width_px"] = _int_default(
            params,
            _RENDER_DEFAULTS,
            "region_outline_width_px",
            _DEFAULTS.region_outline_width_px,
        )

        last_error: Exception | None = None
        scene: _ScenePayload | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                scene = _make_scene(
                    instance_seed=int(instance_seed),
                    params=params,
                    render_params=render_params,
                    attempt=int(attempt),
                )
                break
            except Exception as exc:  # pragma: no cover - exercised by generation smoke tests.
                last_error = exc
                scene = None
        if scene is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        annotation_bboxes = tuple(instance.bbox_xyxy for instance in scene.instances if instance.counted)
        if len(annotation_bboxes) != int(scene.target_count):
            raise RuntimeError("projected region annotation did not match target answer")
        annotation_artifacts = bbox_set_from_bboxes(annotation_bboxes)

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                f"question_text_{scene.region.query_key}",
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        question_key = f"question_text_{scene.region.query_key}"
        prompt_selection = render_task_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            slots={
                "object_description": str(prompt_defaults["object_description"]),
                "question_text": str(prompt_defaults[question_key]).format(shape_name=str(scene.target_shape_name)),
                "json_output_contract": str(prompt_defaults["json_output_contract"]),
                "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
                "annotation_hint": str(prompt_defaults["annotation_hint"]).format(shape_name=str(scene.target_shape_name)),
                "answer_hint": str(prompt_defaults["answer_hint"]),
                "json_example": str(prompt_defaults["json_example"]),
                "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
            },
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
        counted_instance_ids = tuple(str(instance.instance_id) for instance in scene.instances if instance.counted)
        trace_payload = build_scoped_region_trace_payload(
            scene=scene,
            render_params=render_params,
            prompt_defaults=prompt_defaults,
            prompt_artifacts=prompt_artifacts,
            annotation_artifacts=annotation_artifacts,
            counted_instance_ids=counted_instance_ids,
            shape_support=_shape_support(params),
        )
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=TypedValue(type="integer", value=int(scene.target_count)),
            annotation_gt=TypedValue(
                type=str(annotation_artifacts["annotation_type"]),
                value=list(annotation_artifacts["annotation_value"]),
            ),
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(scene.region.query_key),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        )


__all__ = ["IconsCountingNamedShapeRegionCountTask"]
