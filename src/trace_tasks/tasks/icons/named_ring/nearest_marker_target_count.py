"""Count target icons closer to marker A than marker B on a named ring."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.deterministic_sampling import uniform_probability_map
from ...shared.fixed_query import resolve_task_query_id_param, strip_query_id_params
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_query_spec,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import icon_bbox_set_annotation
from ..shared.icon_scene import sort_bboxes_reading_order
from ..shared.icon_task_rendering import icon_render_style_trace
from ..shared.procedural_named_icons import procedural_named_icon_display_name

from .shared.defaults import NamedRingDefaults, SCENE_ID
from .shared.rendering import render_named_ring_scene, serialize_ring_icon
from .shared.sampling import (
    choose_answer_count,
    choose_off_arc_target_count,
    fill_style_probability_map,
    fill_style_support,
    int_bounds,
    resolve_target_shape,
    shape_support,
)
from .shared.state import RingArcPlan, RingScenePayload
from .shared.styles import named_ring_style_trace, resolve_named_ring_render_params


TASK_ID = "task_icons__named_ring__nearest_marker_target_count"
DOMAIN = "icons"
QUERY_ID = SINGLE_QUERY_ID
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
NOISE_NAMESPACE = "icons.named_ring.nearest_marker"


@dataclass(frozen=True)
class _SampleSpec:
    """Task-owned symbolic state for one nearest-marker count question."""

    query_probabilities: Dict[str, float]
    plan: RingArcPlan
    close_to_a_indices: Tuple[int, ...]
    close_to_b_indices: Tuple[int, ...]
    tie_indices: Tuple[int, ...]
    distance_to_a_by_index: Dict[int, int]
    distance_to_b_by_index: Dict[int, int]


_DEFAULTS = NamedRingDefaults()
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


def _ring_distance(index_a: int, index_b: int, *, count: int) -> int:
    """Return shortest ring-step distance between two ring indices."""

    delta = abs(int(index_a) - int(index_b)) % int(count)
    return int(min(delta, int(count) - delta))


def _marker_distance_partitions(
    *,
    ring_icon_count: int,
    start_index: int,
    end_index: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Dict[int, int], Dict[int, int]]:
    """Partition non-marker indices by proximity to marker A or marker B."""

    close_a = []
    close_b = []
    ties = []
    distance_a: Dict[int, int] = {}
    distance_b: Dict[int, int] = {}
    markers = {int(start_index), int(end_index)}
    for index in range(int(ring_icon_count)):
        if int(index) in markers:
            continue
        da = _ring_distance(index, int(start_index), count=int(ring_icon_count))
        db = _ring_distance(index, int(end_index), count=int(ring_icon_count))
        distance_a[int(index)] = int(da)
        distance_b[int(index)] = int(db)
        if int(da) < int(db):
            close_a.append(int(index))
        elif int(db) < int(da):
            close_b.append(int(index))
        else:
            ties.append(int(index))
    return (
        tuple(close_a),
        tuple(close_b),
        tuple(ties),
        dict(distance_a),
        dict(distance_b),
    )


def _choose_ring_icon_count_for_nearest_marker(
    rng,
    *,
    params: Mapping[str, Any],
    answer_count: int,
) -> Tuple[int, Dict[str, float]]:
    """Choose a ring size with enough positions closer to marker A."""

    low, high = int_bounds(
        params,
        _GEN_DEFAULTS,
        low_key="ring_icon_count_min",
        high_key="ring_icon_count_max",
        fallback_low=_DEFAULTS.ring_icon_count_min,
        fallback_high=_DEFAULTS.ring_icon_count_max,
    )
    low = max(int(low), int(answer_count) + 6, (2 * int(answer_count)) + 2)
    explicit = params.get("ring_icon_count", params.get("object_count"))
    support = tuple(range(int(low), int(high) + 1))
    if not support:
        raise ValueError("ring_icon_count support is empty for nearest-marker answer support")
    if explicit is not None:
        value = int(explicit)
        if value not in set(support):
            raise ValueError("ring_icon_count is outside configured nearest-marker support")
        return int(value), dict(uniform_probability_map(support, selected=int(value)))
    value = int(rng.choice(support))
    return int(value), dict(uniform_probability_map(support))


def _sample_marker_indices(
    rng,
    *,
    ring_icon_count: int,
    answer_count: int,
) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Dict[int, int], Dict[int, int]]:
    """Sample marker locations with enough positions closer to marker A."""

    for _ in range(800):
        start_index = int(rng.randrange(int(ring_icon_count)))
        end_index = int(rng.randrange(int(ring_icon_count)))
        if int(end_index) == int(start_index):
            continue
        gap = _ring_distance(int(start_index), int(end_index), count=int(ring_icon_count))
        if int(gap) < 4 or int(gap) > int(ring_icon_count) - 4:
            continue
        close_a, close_b, ties, distance_a, distance_b = _marker_distance_partitions(
            ring_icon_count=int(ring_icon_count),
            start_index=int(start_index),
            end_index=int(end_index),
        )
        if len(close_a) < int(answer_count):
            continue
        if len(close_b) + len(ties) < 1:
            continue
        return (
            int(start_index),
            int(end_index),
            tuple(close_a),
            tuple(close_b),
            tuple(ties),
            dict(distance_a),
            dict(distance_b),
        )
    raise RuntimeError("failed to sample named-ring marker positions with feasible nearest-marker support")


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample task-owned state and a neutral ring render plan."""

    task_params = _normalize_query(params)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    answer_count, answer_probabilities = choose_answer_count(
        rng,
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        fallback_defaults=_DEFAULTS,
    )
    ring_icon_count, ring_icon_count_probabilities = _choose_ring_icon_count_for_nearest_marker(
        rng,
        params=task_params,
        answer_count=int(answer_count),
    )
    start_index, end_index, close_a, close_b, ties, distance_a, distance_b = _sample_marker_indices(
        rng,
        ring_icon_count=int(ring_icon_count),
        answer_count=int(answer_count),
    )
    shape_values = shape_support(task_params, _GEN_DEFAULTS)
    target_shape_id, shape_probabilities = resolve_target_shape(rng, params=task_params, support=shape_values)

    close_a_pool = list(int(value) for value in close_a)
    rng.shuffle(close_a_pool)
    counted_indices = tuple(sorted(int(value) for value in close_a_pool[: int(answer_count)]))
    off_target_candidates = [int(value) for value in (*close_b, *ties)]
    off_target_count, off_target_probabilities = choose_off_arc_target_count(
        rng,
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        fallback_defaults=_DEFAULTS,
        feasible_count=len(off_target_candidates),
    )
    rng.shuffle(off_target_candidates)
    off_arc_target_indices = tuple(sorted(int(value) for value in off_target_candidates[: int(off_target_count)]))
    target_indices = set(counted_indices) | set(off_arc_target_indices)
    distractor_support = tuple(str(value) for value in shape_values if str(value) != str(target_shape_id))
    if not distractor_support:
        raise ValueError("named-ring nearest-marker task needs non-target distractor shapes")
    shape_ids = []
    for index in range(int(ring_icon_count)):
        if int(index) in target_indices:
            shape_ids.append(str(target_shape_id))
        else:
            shape_ids.append(str(rng.choice(distractor_support)))
    for index in (int(start_index), int(end_index)):
        if str(shape_ids[int(index)]) == str(target_shape_id):
            shape_ids[int(index)] = str(rng.choice(distractor_support))

    realized = sum(1 for index in close_a if str(shape_ids[int(index)]) == str(target_shape_id))
    if int(realized) != int(answer_count):
        raise RuntimeError("constructed nearest-marker ring did not realize requested answer")

    fill_values = fill_style_support(task_params, _GEN_DEFAULTS, _DEFAULTS)
    fill_probabilities = fill_style_probability_map(task_params, _GEN_DEFAULTS, fill_values)
    plan = RingArcPlan(
        direction="closer_to_marker_a",
        target_shape_id=str(target_shape_id),
        target_shape_name=procedural_named_icon_display_name(str(target_shape_id)),
        answer_count=int(answer_count),
        ring_icon_count=int(ring_icon_count),
        arc_span_count=len(close_a),
        start_index=int(start_index),
        end_index=int(end_index),
        arc_indices=tuple(int(value) for value in close_a),
        counted_indices=tuple(int(value) for value in counted_indices),
        off_arc_target_indices=tuple(int(value) for value in off_arc_target_indices),
        shape_ids_by_index=tuple(str(value) for value in shape_ids),
        answer_probabilities=dict(answer_probabilities),
        ring_icon_count_probabilities=dict(ring_icon_count_probabilities),
        arc_span_probabilities=dict(uniform_probability_map((len(close_a),), selected=len(close_a))),
        off_arc_target_count_probabilities=dict(off_target_probabilities),
        shape_probabilities=dict(shape_probabilities),
        fill_style_support=tuple(fill_values),
        fill_style_probabilities=dict(fill_probabilities),
    )
    return _SampleSpec(
        query_probabilities={QUERY_ID: 1.0},
        plan=plan,
        close_to_a_indices=tuple(int(value) for value in close_a),
        close_to_b_indices=tuple(int(value) for value in close_b),
        tie_indices=tuple(int(value) for value in ties),
        distance_to_a_by_index=dict(distance_a),
        distance_to_b_by_index=dict(distance_b),
    )


def _prompt_artifacts(*, sample: _SampleSpec, prompt_defaults: Mapping[str, Any], instance_seed: int):
    """Render both prompt output variants."""

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
                target_shape_name=str(sample.plan.target_shape_name)
            ),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["nearest_marker_annotation_hint"]).format(
                target_shape_name=str(sample.plan.target_shape_name)
            ),
            "answer_hint": str(prompt_defaults["nearest_marker_answer_hint"]),
            "json_example": str(prompt_defaults["nearest_marker_json_example"]),
            "json_example_answer_only": str(prompt_defaults["nearest_marker_json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


@register_task
class IconsNamedRingNearestMarkerTargetCountTask:
    """Count target named icons closer to marker A than marker B."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'logical_composition', 'spatial_relations')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic nearest-marker ring-count instance."""

        render_params = resolve_named_ring_render_params(
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        last_error: Exception | None = None
        sample: _SampleSpec | None = None
        scene: RingScenePayload | None = None
        for attempt in range(max(1, int(max_attempts))):
            try:
                sample = _sample_spec(instance_seed=int(instance_seed), params=params)
                scene_rng = spawn_rng(int(instance_seed), f"{TASK_ID}:scene", int(attempt))
                scene = render_named_ring_scene(
                    rng=scene_rng,
                    plan=sample.plan,
                    instance_seed=int(instance_seed),
                    render_params=render_params,
                    noise_namespace=NOISE_NAMESPACE,
                )
                break
            except Exception as exc:
                last_error = exc
                sample = None
                scene = None
        if sample is None or scene is None:
            raise RuntimeError(f"could not generate {TASK_ID}: {last_error}") from last_error

        counted_icons = tuple(icon for icon in scene.icons if bool(icon.is_counted))
        annotation_bboxes = sort_bboxes_reading_order(icon.bbox_xyxy for icon in counted_icons)
        if len(annotation_bboxes) != int(sample.plan.answer_count):
            raise RuntimeError("rendered named-ring annotation count does not match answer")
        annotation_payload = icon_bbox_set_annotation(annotation_bboxes)
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
                "nearest_marker_annotation_hint",
                "nearest_marker_answer_hint",
                "nearest_marker_json_example",
                "nearest_marker_json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_artifacts = _prompt_artifacts(
            sample=sample,
            prompt_defaults=prompt_defaults,
            instance_seed=int(instance_seed),
        )

        serialized_icons = [serialize_ring_icon(icon) for icon in scene.icons]
        counted_instance_ids = tuple(str(icon.instance_id) for icon in counted_icons)
        shape_counts = dict(Counter(str(icon.shape_id) for icon in scene.icons))
        answer_gt = TypedValue(type="integer", value=int(sample.plan.answer_count))
        annotation_gt = TypedValue(
            type=str(annotation_payload["annotation_type"]),
            value=list(annotation_payload["annotation_value"]),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=QUERY_ID,
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id_probabilities": dict(sample.query_probabilities),
                "target_shape_id": str(sample.plan.target_shape_id),
                "target_shape_name": str(sample.plan.target_shape_name),
                "answer_count": int(sample.plan.answer_count),
                "ring_icon_count": int(sample.plan.ring_icon_count),
                "answer_probabilities": dict(sample.plan.answer_probabilities),
                "ring_icon_count_probabilities": dict(sample.plan.ring_icon_count_probabilities),
                "off_marker_target_count_probabilities": dict(sample.plan.off_arc_target_count_probabilities),
                "shape_id_support": list(shape_support(params, _GEN_DEFAULTS)),
                "shape_probabilities": dict(sample.plan.shape_probabilities),
                "named_icon_fill_style_support": list(sample.plan.fill_style_support),
                "fill_style_probabilities": dict(sample.plan.fill_style_probabilities),
            },
        )
        marker_indices = {"A": int(sample.plan.start_index), "B": int(sample.plan.end_index)}
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_named_ring",
                "scene_id": SCENE_ID,
                "query_id": QUERY_ID,
                "entities": list(serialized_icons),
                "relations": {
                    "counting_rule": "target_shape_closer_to_marker_a_than_marker_b",
                    "target_shape_id": str(sample.plan.target_shape_id),
                    "target_shape_name": str(sample.plan.target_shape_name),
                    "start_marker": "A",
                    "end_marker": "B",
                    "marker_indices": dict(marker_indices),
                    "answer_count": int(sample.plan.answer_count),
                    "ring_icon_count": int(sample.plan.ring_icon_count),
                    "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
                    "clockwise_order_shape_ids": [str(value) for value in sample.plan.shape_ids_by_index],
                    "close_to_a_indices": [int(value) for value in sample.close_to_a_indices],
                    "close_to_b_indices": [int(value) for value in sample.close_to_b_indices],
                    "tie_indices": [int(value) for value in sample.tie_indices],
                    "counted_indices": [int(value) for value in sample.plan.counted_indices],
                    "non_counted_target_indices": [int(value) for value in sample.plan.off_arc_target_indices],
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene.panel_geometry),
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": QUERY_ID,
                "canvas_size": list(scene.panel_geometry["canvas_size"]),
                "coord_space": "pixel",
                "panel_geometry": dict(scene.panel_geometry),
                "ring_bbox_xyxy": [int(value) for value in scene.ring_bbox_xyxy],
                "ring_center_xy": [float(scene.ring_center_xy[0]), float(scene.ring_center_xy[1])],
                "ring_radius_xy": [float(scene.ring_radius_xy[0]), float(scene.ring_radius_xy[1])],
                "style": {
                    **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=scene.sampled_palette_rgb),
                    **named_ring_style_trace(render_params),
                },
            },
            "render_map": {
                "image_id": "img0",
                "object_bboxes_px": {
                    str(icon.instance_id): [int(value) for value in icon.bbox_xyxy]
                    for icon in scene.icons
                },
                "marker_label_bboxes_px": {
                    str(label): [int(value) for value in bbox]
                    for label, bbox in scene.marker_label_bboxes.items()
                },
                "counted_instance_ids": list(counted_instance_ids),
            },
            "execution_trace": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "scene_variant": "single_panel_named_ring",
                "query_id": QUERY_ID,
                "query_id_probabilities": dict(sample.query_probabilities),
                "question_format": "count_target_shape_icons_closer_to_marker_a_than_marker_b",
                "target_shape_id": str(sample.plan.target_shape_id),
                "target_shape_name": str(sample.plan.target_shape_name),
                "answer": int(sample.plan.answer_count),
                "start_marker": "A",
                "end_marker": "B",
                "start_index": int(sample.plan.start_index),
                "end_index": int(sample.plan.end_index),
                "ring_icon_count": int(sample.plan.ring_icon_count),
                "clockwise_order_shape_ids": [str(value) for value in sample.plan.shape_ids_by_index],
                "close_to_a_indices": [int(value) for value in sample.close_to_a_indices],
                "close_to_b_indices": [int(value) for value in sample.close_to_b_indices],
                "tie_indices": [int(value) for value in sample.tie_indices],
                "distance_to_a_by_index": {str(key): int(value) for key, value in sample.distance_to_a_by_index.items()},
                "distance_to_b_by_index": {str(key): int(value) for key, value in sample.distance_to_b_by_index.items()},
                "counted_indices": [int(value) for value in sample.plan.counted_indices],
                "non_counted_target_indices": [int(value) for value in sample.plan.off_arc_target_indices],
                "counted_instance_ids": list(counted_instance_ids),
            },
            "witness_symbolic": {
                "query_id": QUERY_ID,
                "target_shape_id": str(sample.plan.target_shape_id),
                "target_shape_name": str(sample.plan.target_shape_name),
                "answer": int(sample.plan.answer_count),
                "marker_indices": dict(marker_indices),
                "counted_indices": [int(value) for value in sample.plan.counted_indices],
                "counted_instance_ids": list(counted_instance_ids),
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


__all__ = ["IconsNamedRingNearestMarkerTargetCountTask", "TASK_ID"]
