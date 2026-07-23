"""Count stops strictly between two named icon types on a path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.fixed_query import resolve_task_query_id_param, strip_query_id_params
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.icon_task_rendering import icon_render_style_trace
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.named_colors import named_color
from ..shared.procedural_named_icon_field_scene import rotation_for_named_shape
from ..shared.procedural_named_icons import sample_procedural_named_icon_fill_style

from .shared.defaults import NamedPathDefaults, SCENE_ID
from .shared.rendering import render_named_path_scene, serialize_path_icon
from .shared.sampling import (
    display_shape_name,
    fill_style_probability_map,
    fill_style_support,
    shape_support,
    color_support,
    string_probability_map,
)
from .shared.state import IconPlan, PathScenePayload
from .shared.styles import named_path_style_trace, resolve_named_path_render_params
from ..shared.icon_task_rendering import sample_icon_instance_noise


TASK_ID = "task_icons__named_path__path_distance_value"
DOMAIN = "icons"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
NOISE_NAMESPACE = "icons.named_path.path_distance"


@dataclass(frozen=True)
class _SampleSpec:
    """Task-owned symbolic state for one path-distance question."""

    answer_count: int
    first_shape_id: str
    first_shape_name: str
    second_shape_id: str
    second_shape_name: str
    first_position_index: int
    second_position_index: int
    stop_count: int
    extra_stop_count: int
    answer_probabilities: Dict[str, float]
    extra_stop_count_probabilities: Dict[str, float]
    shape_probabilities: Dict[str, float]
    fill_style_support: Tuple[str, ...]
    fill_style_probabilities: Dict[str, float]


_DEFAULTS = NamedPathDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _normalize_query(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate the public single-query contract."""

    resolve_task_query_id_param(
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=QUERY_ID,
        task_id=TASK_ID,
    )
    return strip_query_id_params(params)


def _sample_int_support(
    rng,
    *,
    params: Mapping[str, Any],
    low_key: str,
    high_key: str,
    explicit_keys: Sequence[str],
    fallback_low: int,
    fallback_high: int,
) -> Tuple[int, Dict[str, float]]:
    """Sample one integer from a configured inclusive support."""

    low = int(params.get(low_key, group_default(_GEN_DEFAULTS, low_key, int(fallback_low))))
    high = int(params.get(high_key, group_default(_GEN_DEFAULTS, high_key, int(fallback_high))))
    if int(low) < 0 or int(high) < int(low):
        raise ValueError(f"invalid {low_key}/{high_key}")
    support = tuple(range(int(low), int(high) + 1))
    explicit = None
    for key in explicit_keys:
        if key in params:
            explicit = params[key]
            break
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError(f"explicit value for {explicit_keys[0]} is outside configured support")
        return int(value), dict(uniform_probability_map(support, selected=int(value)))
    value = int(rng.choice(support))
    return int(value), dict(uniform_probability_map(support))


def _sample_shape_pair(rng, *, params: Mapping[str, Any]) -> Tuple[str, str, Dict[str, float]]:
    """Choose two distinct endpoint shape ids."""

    shapes = shape_support(params, _GEN_DEFAULTS)
    first = params.get("first_shape_id", params.get("shape_id_a"))
    second = params.get("second_shape_id", params.get("shape_id_b"))
    if first is not None:
        first_shape = str(first)
        if first_shape not in set(shapes):
            raise ValueError(f"unsupported first_shape_id: {first_shape}")
    else:
        first_shape = str(rng.choice(shapes))
    remaining = tuple(str(value) for value in shapes if str(value) != str(first_shape))
    if not remaining:
        raise ValueError("named-path distance task needs at least two supported shapes")
    if second is not None:
        second_shape = str(second)
        if second_shape not in set(remaining):
            raise ValueError("second_shape_id must be supported and distinct from first_shape_id")
    else:
        second_shape = str(rng.choice(remaining))
    if str(first_shape) == str(second_shape):
        raise ValueError("path-distance endpoint shapes must be distinct")
    if first is not None and second is not None:
        probabilities = {str(value): (1.0 if str(value) in {str(first_shape), str(second_shape)} else 0.0) for value in shapes}
    else:
        probabilities = string_probability_map(shapes)
    return str(first_shape), str(second_shape), dict(probabilities)


def _sample_endpoint_positions(
    rng,
    *,
    answer_count: int,
    extra_stop_count: int,
) -> Tuple[int, int, int]:
    """Place two endpoint icons with `answer_count` stops strictly between them."""

    extra = max(2, int(extra_stop_count))
    left_extra = int(rng.randint(1, int(extra) - 1))
    first_position = int(left_extra)
    second_position = int(first_position + int(answer_count) + 1)
    stop_count = int(answer_count) + 2 + int(extra)
    if second_position >= stop_count - 1:
        raise RuntimeError("sampled endpoint positions did not leave a right-side path stop")
    return int(first_position), int(second_position), int(stop_count)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample symbolic state for one path-distance task instance."""

    task_params = _normalize_query(params)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    answer_count, answer_probabilities = _sample_int_support(
        rng,
        params=task_params,
        low_key="distance_answer_min",
        high_key="distance_answer_max",
        explicit_keys=("answer_count", "distance_value", "answer"),
        fallback_low=1,
        fallback_high=8,
    )
    extra_stop_count, extra_probabilities = _sample_int_support(
        rng,
        params=task_params,
        low_key="extra_stop_count_min",
        high_key="extra_stop_count_max",
        explicit_keys=("extra_stop_count",),
        fallback_low=2,
        fallback_high=6,
    )
    first_shape, second_shape, shape_probabilities = _sample_shape_pair(rng, params=task_params)
    first_position, second_position, stop_count = _sample_endpoint_positions(
        rng,
        answer_count=int(answer_count),
        extra_stop_count=int(extra_stop_count),
    )
    if bool(rng.randint(0, 1)):
        first_position, second_position = int(second_position), int(first_position)

    fill_values = fill_style_support(task_params, _GEN_DEFAULTS)
    fill_probabilities = fill_style_probability_map(task_params, _GEN_DEFAULTS, fill_values)
    return _SampleSpec(
        answer_count=int(answer_count),
        first_shape_id=str(first_shape),
        first_shape_name=display_shape_name(str(first_shape)),
        second_shape_id=str(second_shape),
        second_shape_name=display_shape_name(str(second_shape)),
        first_position_index=int(first_position),
        second_position_index=int(second_position),
        stop_count=int(stop_count),
        extra_stop_count=int(extra_stop_count),
        answer_probabilities=dict(answer_probabilities),
        extra_stop_count_probabilities=dict(extra_probabilities),
        shape_probabilities=dict(shape_probabilities),
        fill_style_support=tuple(fill_values),
        fill_style_probabilities=dict(fill_probabilities),
    )


def _prompt_artifacts(*, sample: _SampleSpec, prompt_defaults: Mapping[str, Any], instance_seed: int):
    """Render prompt variants for the path-distance task."""

    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(prompt_defaults["question_text"]).format(
                first_shape_name=str(sample.first_shape_name),
                second_shape_name=str(sample.second_shape_name),
            ),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["annotation_hint"]).format(
                first_shape_name=str(sample.first_shape_name),
                second_shape_name=str(sample.second_shape_name),
            ),
            "answer_hint": str(prompt_defaults["answer_hint"]),
            "json_example": str(prompt_defaults["json_example"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def _icon_plans(
    *,
    rng,
    sample: _SampleSpec,
    params: Mapping[str, Any],
    render_params: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[Tuple[IconPlan, ...], Tuple[Tuple[int, int, int], ...]]:
    """Sample path-stop icon semantics for the two-endpoint distance task."""

    shapes = shape_support(params, _GEN_DEFAULTS)
    distractor_shapes = tuple(
        str(value)
        for value in shapes
        if str(value) not in {str(sample.first_shape_id), str(sample.second_shape_id)}
    )
    if len(distractor_shapes) < 8:
        raise ValueError("named-path distance task needs at least eight non-endpoint shapes")
    colors = color_support(params, _GEN_DEFAULTS)
    plans = []
    for position in range(int(sample.stop_count)):
        if int(position) == int(sample.first_position_index):
            shape_id = str(sample.first_shape_id)
            role = "first_named_endpoint"
        elif int(position) == int(sample.second_position_index):
            shape_id = str(sample.second_shape_id)
            role = "second_named_endpoint"
        elif min(int(sample.first_position_index), int(sample.second_position_index)) < int(position) < max(
            int(sample.first_position_index),
            int(sample.second_position_index),
        ):
            shape_id = str(rng.choice(distractor_shapes))
            role = "between_stop"
        else:
            shape_id = str(rng.choice(distractor_shapes))
            role = "outside_stop"
        color_name = str(rng.choice(colors))
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{NOISE_NAMESPACE}:path_stop:{int(position)}",
            render_params=render_params,
        )
        low = int(render_params["scene_icon_size_min_px"])
        high = int(render_params["scene_icon_size_max_px"])
        nominal_size = int(rng.randint(min(low, high), max(low, high)))
        plans.append(
            IconPlan(
                position_index=int(position),
                role=str(role),
                label="",
                shape_id=str(shape_id),
                color_name=str(color_name),
                tint_rgb=tuple(int(channel) for channel in named_color(str(color_name))),
                fill_style=sample_procedural_named_icon_fill_style(
                    rng,
                    support=tuple(str(value) for value in sample.fill_style_support),
                    probabilities=dict(sample.fill_style_probabilities),
                ),
                nominal_size_px=int(nominal_size),
                rotation_degrees=rotation_for_named_shape(rng, str(shape_id)),
                noise_edits=tuple(noise_edits),
                noise_seed=int(noise_seed),
            )
        )
    sampled_palette_rgb = tuple(tuple(int(channel) for channel in named_color(color_name)) for color_name in colors)
    return tuple(plans), sampled_palette_rgb


def _render_scene(
    *,
    sample: _SampleSpec,
    params: Mapping[str, Any],
    render_params: Mapping[str, Any],
    instance_seed: int,
    scene_rng,
) -> PathScenePayload:
    """Render a path-distance scene through the neutral path renderer."""

    plans, sampled_palette_rgb = _icon_plans(
        rng=scene_rng,
        sample=sample,
        params=params,
        render_params=render_params,
        instance_seed=int(instance_seed),
    )
    return render_named_path_scene(
        rng=scene_rng,
        plans=tuple(plans),
        answer_label="",
        target_shape_id=str(sample.first_shape_id),
        target_shape_name=str(sample.first_shape_name),
        target_occurrence_count=1,
        stop_count=int(sample.stop_count),
        distractor_count=int(sample.stop_count) - 2,
        selected_position=int(sample.first_position_index),
        answer_position=int(sample.second_position_index),
        neighbor_direction="between",
        target_positions=(int(sample.first_position_index),),
        option_positions=(),
        labels_by_position={},
        sampled_palette_rgb=tuple(sampled_palette_rgb),
        render_params=render_params,
    )


@register_task
class IconsNamedPathPathDistanceValueTask:
    """Count path stops strictly between two named icon types."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic named-path distance instance."""

        render_params = resolve_named_path_render_params(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        sample: _SampleSpec | None = None
        scene: PathScenePayload | None = None
        last_error: Exception | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params)
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                scene = _render_scene(
                    sample=sample,
                    params=dict(_normalize_query(params)),
                    render_params=render_params,
                    instance_seed=int(instance_seed),
                    scene_rng=scene_rng,
                )
                break
            except Exception as exc:
                last_error = exc
                sample = None
                scene = None
        if sample is None or scene is None:
            raise RuntimeError(f"failed to generate {TASK_ID} instance") from last_error

        prompt_defaults = required_group_defaults(
            _PROMPT_DEFAULTS,
            (
                "bundle_id",
                "scene_key",
                "task_key",
                "json_output_contract",
                "json_output_contract_answer_only",
                "object_description",
                "question_text",
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = _prompt_artifacts(
            sample=sample,
            prompt_defaults=prompt_defaults,
            instance_seed=int(instance_seed),
        )

        icons_by_position = {int(icon.position_index): icon for icon in scene.icons}
        first_icon = icons_by_position[int(sample.first_position_index)]
        second_icon = icons_by_position[int(sample.second_position_index)]
        between_positions = tuple(
            range(
                min(int(sample.first_position_index), int(sample.second_position_index)) + 1,
                max(int(sample.first_position_index), int(sample.second_position_index)),
            )
        )
        counted_icons = tuple(icons_by_position[int(position)] for position in between_positions)
        annotation_payload = icon_bbox_set_annotation(
            [icon.bbox_xyxy for icon in counted_icons]
        )
        serialized_icons = [serialize_path_icon(icon) for icon in scene.icons]
        answer_gt = TypedValue(type="integer", value=int(sample.answer_count))
        annotation_gt = TypedValue(
            type=str(annotation_payload["annotation_type"]),
            value=[list(bbox) for bbox in annotation_payload["annotation_value"]],
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=QUERY_ID,
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": {QUERY_ID: 1.0},
                "first_shape_id": str(sample.first_shape_id),
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_id": str(sample.second_shape_id),
                "second_shape_name": str(sample.second_shape_name),
                "shape_probabilities": dict(sample.shape_probabilities),
                "answer_count": int(sample.answer_count),
                "answer_probabilities": dict(sample.answer_probabilities),
                "extra_stop_count": int(sample.extra_stop_count),
                "extra_stop_count_probabilities": dict(sample.extra_stop_count_probabilities),
                "stop_count": int(sample.stop_count),
                "named_icon_fill_style_support": list(sample.fill_style_support),
                "fill_style_probabilities": dict(sample.fill_style_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_named_path_distance",
                "scene_id": SCENE_ID,
                "query_id": QUERY_ID,
                "entities": list(serialized_icons),
                "relations": {
                    "target": "count_path_stops_strictly_between_two_named_icon_types",
                    "first_shape_id": str(sample.first_shape_id),
                    "first_shape_name": str(sample.first_shape_name),
                    "second_shape_id": str(sample.second_shape_id),
                    "second_shape_name": str(sample.second_shape_name),
                    "first_position_index": int(sample.first_position_index),
                    "second_position_index": int(sample.second_position_index),
                    "between_positions": [int(value) for value in between_positions],
                    "answer_count": int(sample.answer_count),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene.panel_geometry),
                    "path": {
                        "order": "start_to_end",
                        "points_xy": [[float(x), float(y)] for x, y in scene.path_points_xy],
                    },
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": QUERY_ID,
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene.panel_geometry),
                "style": {
                    **icon_render_style_trace(
                        render_params=render_params,
                        sampled_palette_rgb=tuple(scene.sampled_palette_rgb),
                    ),
                    **named_path_style_trace(render_params),
                },
            },
            "render_map": {
                "image_id": "img0",
                "object_bboxes_px": {
                    str(icon.instance_id): [int(value) for value in icon.bbox_xyxy]
                    for icon in scene.icons
                },
                "path_points_xy": [[float(x), float(y)] for x, y in scene.path_points_xy],
                "first_named_icon": serialize_path_icon(first_icon),
                "second_named_icon": serialize_path_icon(second_icon),
                "between_instance_ids": [
                    str(icons_by_position[int(position)].instance_id)
                    for position in between_positions
                ],
            },
            "execution_trace": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "scene_variant": "single_panel_named_path",
                "query_id": QUERY_ID,
                "query_id_probabilities": {QUERY_ID: 1.0},
                "question_format": "count_stops_between_two_named_icon_types_along_start_to_end_path",
                "first_shape_id": str(sample.first_shape_id),
                "first_shape_name": str(sample.first_shape_name),
                "second_shape_id": str(sample.second_shape_id),
                "second_shape_name": str(sample.second_shape_name),
                "first_position_index": int(sample.first_position_index),
                "second_position_index": int(sample.second_position_index),
                "between_positions": [int(value) for value in between_positions],
                "answer": int(sample.answer_count),
                "stop_count": int(sample.stop_count),
                "extra_stop_count": int(sample.extra_stop_count),
            },
            "witness_symbolic": {
                "query_id": QUERY_ID,
                "answer": int(sample.answer_count),
                "first_shape_id": str(sample.first_shape_id),
                "second_shape_id": str(sample.second_shape_id),
                "first_instance_id": str(first_icon.instance_id),
                "second_instance_id": str(second_icon.instance_id),
                "between_positions": [int(value) for value in between_positions],
            },
            "projected_annotation": dict(annotation_payload["projected_annotation"]),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=scene.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=QUERY_ID,
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        )


__all__ = ["IconsNamedPathPathDistanceValueTask", "TASK_ID"]
