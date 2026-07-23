"""Count named icons on a directed arc between ring markers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.fixed_query import select_task_query_id
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

from .shared.defaults import NamedRingDefaults, SCENE_ID
from .shared.rendering import render_named_ring_scene, serialize_ring_icon
from .shared.sampling import sample_ring_arc_plan, shape_support
from .shared.state import RingArcPlan, RingScenePayload
from .shared.styles import named_ring_style_trace, resolve_named_ring_render_params


TASK_ID = "task_icons__named_ring__scoped_attribute_count"
DOMAIN = "icons"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("clockwise_arc_shape_count", "counterclockwise_arc_shape_count")
NOISE_NAMESPACE = "icons.named_ring.ring_icon"


@dataclass(frozen=True)
class _SampleSpec:
    """Task-owned symbolic state for one named-ring count question."""

    query_id: str
    query_probabilities: Dict[str, float]
    plan: RingArcPlan


_DEFAULTS = NamedRingDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _direction_for_query(query_id: str) -> str:
    """Return the traversal direction named by a public query id."""

    query = str(query_id)
    if query == "clockwise_arc_shape_count":
        return "clockwise"
    if query == "counterclockwise_arc_shape_count":
        return "counterclockwise"
    raise ValueError(f"unsupported named-ring query_id: {query_id}")


def _select_query(instance_seed: int, params: Mapping[str, Any]) -> Tuple[str, Dict[str, float], Dict[str, Any]]:
    """Select and validate one public directed-arc query branch."""

    return select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id=SUPPORTED_QUERY_IDS[0],
        task_id=TASK_ID,
        namespace=f"{TASK_ID}.query",
    )


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Sample task-owned query state and neutral ring-scene plan."""

    query_id, query_probabilities, task_params = _select_query(int(instance_seed), params)
    rng = spawn_rng(int(instance_seed), f"{TASK_ID}:sample")
    plan = sample_ring_arc_plan(
        rng=rng,
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        fallback_defaults=_DEFAULTS,
        direction=_direction_for_query(str(query_id)),
    )
    return _SampleSpec(
        query_id=str(query_id),
        query_probabilities=dict(query_probabilities),
        plan=plan,
    )


def _prompt_question_key(query_id: str) -> str:
    return f"question_text_{query_id}"


def _prompt_artifacts(*, sample: _SampleSpec, prompt_defaults: Mapping[str, Any], instance_seed: int):
    """Render both prompt output variants."""

    question_key = _prompt_question_key(str(sample.query_id))
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(prompt_defaults[question_key]).format(
                target_shape_name=str(sample.plan.target_shape_name)
            ),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["annotation_hint"]).format(
                target_shape_name=str(sample.plan.target_shape_name),
                direction=str(sample.plan.direction),
            ),
            "answer_hint": str(prompt_defaults["answer_hint"]),
            "json_example": str(prompt_defaults["json_example"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


@register_task
class IconsNamedRingScopedAttributeCountTask:
    """Count target named icons strictly between ring markers A and B."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic named-ring directed-arc count instance."""

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
                "question_text_clockwise_arc_shape_count",
                "question_text_counterclockwise_arc_shape_count",
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

        serialized_icons = [serialize_ring_icon(icon) for icon in scene.icons]
        counted_instance_ids = tuple(str(icon.instance_id) for icon in counted_icons)
        shape_counts = dict(Counter(str(icon.shape_id) for icon in scene.icons))
        marker_indices = {"A": int(sample.plan.start_index), "B": int(sample.plan.end_index)}
        answer_gt = TypedValue(type="integer", value=int(sample.plan.answer_count))
        annotation_gt = TypedValue(
            type=str(annotation_payload["annotation_type"]),
            value=list(annotation_payload["annotation_value"]),
        )
        query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_artifacts,
            query_id=str(sample.query_id),
            params={
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "target_shape_id": str(sample.plan.target_shape_id),
                "target_shape_name": str(sample.plan.target_shape_name),
                "answer_count": int(sample.plan.answer_count),
                "ring_icon_count": int(sample.plan.ring_icon_count),
                "arc_span_count": int(sample.plan.arc_span_count),
                "direction": str(sample.plan.direction),
                "query_id_probabilities": dict(sample.query_probabilities),
                "answer_probabilities": dict(sample.plan.answer_probabilities),
                "ring_icon_count_probabilities": dict(sample.plan.ring_icon_count_probabilities),
                "arc_span_probabilities": dict(sample.plan.arc_span_probabilities),
                "off_arc_target_count_probabilities": dict(sample.plan.off_arc_target_count_probabilities),
                "shape_id_support": list(shape_support(params, _GEN_DEFAULTS)),
                "shape_probabilities": dict(sample.plan.shape_probabilities),
                "named_icon_fill_style_support": list(sample.plan.fill_style_support),
                "fill_style_probabilities": dict(sample.plan.fill_style_probabilities),
            },
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_named_ring",
                "scene_id": SCENE_ID,
                "query_id": str(sample.query_id),
                "entities": list(serialized_icons),
                "relations": {
                    "counting_rule": "named_shape_on_directed_arc_between_markers",
                    "target_shape_id": str(sample.plan.target_shape_id),
                    "target_shape_name": str(sample.plan.target_shape_name),
                    "direction": str(sample.plan.direction),
                    "start_marker": "A",
                    "end_marker": "B",
                    "marker_indices": dict(marker_indices),
                    "answer_count": int(sample.plan.answer_count),
                    "ring_icon_count": int(sample.plan.ring_icon_count),
                    "arc_span_count": int(sample.plan.arc_span_count),
                    "shape_counts": {str(key): int(value) for key, value in shape_counts.items()},
                    "clockwise_order_shape_ids": [str(value) for value in sample.plan.shape_ids_by_index],
                    "arc_indices": [int(value) for value in sample.plan.arc_indices],
                    "counted_indices": [int(value) for value in sample.plan.counted_indices],
                    "off_arc_target_indices": [int(value) for value in sample.plan.off_arc_target_indices],
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
                "query_id": str(sample.query_id),
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
                "query_id": str(sample.query_id),
                "query_id_probabilities": dict(sample.query_probabilities),
                "question_format": "count_named_shape_icons_strictly_between_ring_markers",
                "target_shape_id": str(sample.plan.target_shape_id),
                "target_shape_name": str(sample.plan.target_shape_name),
                "answer": int(sample.plan.answer_count),
                "direction": str(sample.plan.direction),
                "start_marker": "A",
                "end_marker": "B",
                "start_index": int(sample.plan.start_index),
                "end_index": int(sample.plan.end_index),
                "ring_icon_count": int(sample.plan.ring_icon_count),
                "arc_span_count": int(sample.plan.arc_span_count),
                "clockwise_order_shape_ids": [str(value) for value in sample.plan.shape_ids_by_index],
                "arc_indices": [int(value) for value in sample.plan.arc_indices],
                "counted_indices": [int(value) for value in sample.plan.counted_indices],
                "off_arc_target_indices": [int(value) for value in sample.plan.off_arc_target_indices],
                "counted_instance_ids": list(counted_instance_ids),
            },
            "witness_symbolic": {
                "query_id": str(sample.query_id),
                "target_shape_id": str(sample.plan.target_shape_id),
                "target_shape_name": str(sample.plan.target_shape_name),
                "answer": int(sample.plan.answer_count),
                "direction": str(sample.plan.direction),
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
            query_id=str(sample.query_id),
            prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        )


__all__ = ["IconsNamedRingScopedAttributeCountTask"]
