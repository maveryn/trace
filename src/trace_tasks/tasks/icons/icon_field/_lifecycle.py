"""Scene-private lifecycle helpers for icon-field type-frequency tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.config_defaults import group_default, required_group_defaults
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import icon_bbox_set_annotation

from .shared.defaults import IconFieldDefaults
from .shared.rendering import sample_and_render_icon_field_scene
from .shared.state import IconFieldScenePayload, TypeFrequencySpec
from .shared.styles import icon_field_style_trace, resolve_icon_field_render_params


DOMAIN = "icons"
SCENE_ID = "icon_field"
NOISE_NAMESPACE = "icon_field_type_frequency"


FrequencySpecResolver = Callable[[int, Mapping[str, Any]], TypeFrequencySpec]
AnswerAnnotationBinder = Callable[[IconFieldScenePayload, TypeFrequencySpec], tuple[int, Mapping[str, Any]]]


@dataclass(frozen=True)
class IconFieldFrequencyObjective:
    """Task-owned hooks and metadata for one icon-field frequency objective."""

    params: Mapping[str, Any]
    generation_defaults: Mapping[str, Any]
    render_defaults: Mapping[str, Any]
    prompt_defaults: Mapping[str, Any]
    defaults: IconFieldDefaults
    resolve_frequency_spec: FrequencySpecResolver
    bind_answer_annotation: AnswerAnnotationBinder


def build_frequency_binding_payload(
    *,
    scene_payload: IconFieldScenePayload,
    frequency_spec: TypeFrequencySpec,
    counting_rule: str,
    question_format: str,
    counted_icon_ids: tuple[str, ...],
    annotation_indices: tuple[int, ...],
    annotation_bboxes: tuple[list[int], ...],
    target_count: int,
    winner_icon_id: str | None,
    winner_frequency: int | None,
    counted_color_keys: tuple[str, ...] = (),
    representative_rule: str = "",
) -> tuple[int, Mapping[str, Any]]:
    """Build common trace binding fields for icon-field frequency counts."""

    return int(target_count), {
        "counting_rule": str(counting_rule),
        "question_format": str(question_format),
        "counted_icon_ids": list(counted_icon_ids),
        "counted_color_keys": list(counted_color_keys),
        "annotation_indices": list(annotation_indices),
        "annotation_bboxes": list(annotation_bboxes),
        "target_count": int(target_count),
        "singleton_count": int(scene_payload.singleton_count),
        "representative_rule": str(representative_rule),
        "winner_icon_id": winner_icon_id,
        "winner_frequency": winner_frequency,
        "object_count_probabilities": dict(frequency_spec.object_count_probabilities),
        "target_count_probabilities": dict(frequency_spec.target_count_probabilities),
    }


class IconFieldFrequencyTaskLifecycle:
    """Common lifecycle methods for one-concept icon-field frequency tasks."""

    prompt_query_key: str
    scene_kind: str

    def _render_frequency_scene(
        self,
        *,
        instance_seed: int,
        params: Mapping[str, Any],
        max_attempts: int,
        frequency_spec: TypeFrequencySpec,
        generation_defaults: Mapping[str, Any],
        render_defaults: Mapping[str, Any],
        defaults: IconFieldDefaults,
    ) -> Tuple[IconFieldScenePayload, Any, Dict[str, Any], str]:
        """Render one icon-field frequency scene from a task-resolved spec."""

        render_params = resolve_icon_field_render_params(
            params,
            render_defaults=render_defaults,
            defaults=defaults,
            instance_seed=int(instance_seed),
        )
        pool_manifest = str(
            params.get(
                "pool_manifest",
                group_default(generation_defaults, "pool_manifest", defaults.pool_manifest),
            )
        )
        scene_rng = spawn_rng(int(instance_seed), "scene")
        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = sample_and_render_icon_field_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    frequency_spec=frequency_spec,
                    pool_manifest=str(pool_manifest),
                    render_params=render_params,
                    noise_namespace=f"{NOISE_NAMESPACE}:{self.task_id}",
                )
                break
            except Exception as exc:  # pragma: no cover - exercised through retry loop
                last_error = exc
                continue
        if scene_payload is None or image is None:
            raise RuntimeError(f"failed to generate {self.task_id} instance") from last_error
        return scene_payload, image, render_params, pool_manifest

    def _build_frequency_task_output(
        self,
        *,
        instance_seed: int,
        scene_payload: IconFieldScenePayload,
        image: Any,
        render_params: Mapping[str, Any],
        pool_manifest: str,
        prompt_defaults: Mapping[str, Any],
        question_format: str,
        answer_value: int,
        binding: Mapping[str, Any],
    ) -> TaskOutput:
        """Build prompt, trace, answer, and annotation payloads."""

        resolved_prompt_defaults = required_group_defaults(
            prompt_defaults,
            ("bundle_id", "scene_key", "task_key"),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=DOMAIN,
            scene_id=SCENE_ID,
            bundle_id=str(resolved_prompt_defaults["bundle_id"]),
            scene_key=str(resolved_prompt_defaults["scene_key"]),
            task_key=str(resolved_prompt_defaults["task_key"]),
            query_key=str(self.prompt_query_key),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={},
            instance_seed=int(instance_seed),
        )
        prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)

        annotation_bboxes = [list(bbox) for bbox in binding["annotation_bboxes"]]
        annotation_indices = [int(index) for index in binding["annotation_indices"]]
        annotation_payload = icon_bbox_set_annotation(
            annotation_bboxes,
            clip_bbox=scene_payload.panel_geometry["scene_content_xyxy"],
        )
        answer_gt = TypedValue(type="integer", value=int(answer_value))
        annotation_gt = TypedValue(
            type=str(annotation_payload["annotation_type"]),
            value=list(annotation_payload["annotation_value"]),
        )

        trace_payload = {
            "scene_ir": {
                "scene_kind": str(self.scene_kind),
                "scene_id": SCENE_ID,
                "query_id": SINGLE_QUERY_ID,
                "entities": [dict(entity) for entity in scene_payload.scene_instances],
                "relations": {
                    "counting_rule": str(binding["counting_rule"]),
                    "counted_icon_ids": list(binding["counted_icon_ids"]),
                    "singleton_icon_ids": list(scene_payload.singleton_icon_ids),
                    "repeated_icon_ids": list(scene_payload.repeated_icon_ids),
                    "type_frequencies": dict(scene_payload.type_frequencies),
                    "color_frequencies": dict(scene_payload.color_frequencies),
                    "singleton_indices": list(scene_payload.singleton_indices),
                    "repeated_indices": list(scene_payload.repeated_indices),
                    "counted_color_keys": list(binding.get("counted_color_keys", [])),
                    "representative_rule": str(binding.get("representative_rule", "")),
                    "winner_icon_id": binding["winner_icon_id"],
                    "winner_frequency": binding["winner_frequency"],
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": SINGLE_QUERY_ID,
                "template_id": str(resolved_prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "task_id": str(self.task_id),
                    "scene_id": SCENE_ID,
                    "query_id": SINGLE_QUERY_ID,
                    "prompt_query_key": str(self.prompt_query_key),
                    "object_count": int(scene_payload.object_count),
                    "target_count": int(answer_value),
                    "singleton_count": int(scene_payload.singleton_count),
                    "repeated_icon_count": int(scene_payload.object_count) - int(scene_payload.singleton_count),
                    "repeated_type_count": int(scene_payload.repeated_type_count),
                    "repeated_type_multiplicities": list(scene_payload.repeated_type_multiplicities),
                    "unique_type_total": int(scene_payload.distinct_type_count),
                    "unique_color_total": int(scene_payload.distinct_color_count),
                    "placement_mode": str(scene_payload.placement_mode),
                    "object_count_probabilities": dict(binding["object_count_probabilities"]),
                    "target_count_probabilities": dict(binding["target_count_probabilities"]),
                    "pool_manifest": str(pool_manifest),
                    "rotation_candidates_degrees": [
                        int(value) for value in render_params["rotation_candidates_degrees"]
                    ],
                    "winner_icon_id": binding["winner_icon_id"],
                    "winner_frequency": binding["winner_frequency"],
                },
            },
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": SINGLE_QUERY_ID,
                "canvas_size": list(scene_payload.panel_geometry["canvas_size"]),
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": icon_field_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                ),
            },
            "render_map": {
                "image_id": "img0",
                "anchors": {},
            },
            "execution_trace": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "scene_variant": "single_panel_scene",
                "query_id": SINGLE_QUERY_ID,
                "prompt_query_key": str(self.prompt_query_key),
                "question_format": str(question_format),
                "object_count": int(scene_payload.object_count),
                "target_count": int(answer_value),
                "singleton_count": int(scene_payload.singleton_count),
                "repeated_icon_count": int(scene_payload.object_count) - int(scene_payload.singleton_count),
                "repeated_type_count": int(scene_payload.repeated_type_count),
                "repeated_type_multiplicities": list(scene_payload.repeated_type_multiplicities),
                "unique_type_total": int(scene_payload.distinct_type_count),
                "unique_color_total": int(scene_payload.distinct_color_count),
                "placement_mode": str(scene_payload.placement_mode),
                "scene_icon_ids": list(scene_payload.scene_icon_ids),
                "scene_rotations_degrees": list(scene_payload.scene_rotations_degrees),
                "scene_color_keys": list(scene_payload.scene_color_keys),
                "scene_tint_rgbs": [list(color) for color in scene_payload.scene_tint_rgbs],
                "color_group_indices": list(scene_payload.color_group_indices),
                "type_frequencies": dict(scene_payload.type_frequencies),
                "color_frequencies": dict(scene_payload.color_frequencies),
                "singleton_indices": list(scene_payload.singleton_indices),
                "repeated_indices": list(scene_payload.repeated_indices),
                "annotation_indices": list(annotation_indices),
                "annotation_bboxes": list(annotation_bboxes),
                "counted_color_keys": list(binding.get("counted_color_keys", [])),
                "representative_rule": str(binding.get("representative_rule", "")),
                "winner_icon_id": binding["winner_icon_id"],
                "winner_frequency": binding["winner_frequency"],
            },
            "witness_symbolic": {
                "counted_icon_ids": list(binding["counted_icon_ids"]),
                "counted_color_keys": list(binding.get("counted_color_keys", [])),
                "singleton_icon_ids": list(scene_payload.singleton_icon_ids),
                "repeated_icon_ids": list(scene_payload.repeated_icon_ids),
                "type_frequencies": dict(scene_payload.type_frequencies),
                "color_frequencies": dict(scene_payload.color_frequencies),
                "singleton_indices": list(scene_payload.singleton_indices),
                "repeated_indices": list(scene_payload.repeated_indices),
                "annotation_indices": list(annotation_indices),
                "annotation_bboxes": list(annotation_bboxes),
                "representative_rule": str(binding.get("representative_rule", "")),
                "winner_icon_id": binding["winner_icon_id"],
                "winner_frequency": binding["winner_frequency"],
            },
            "projected_annotation": dict(annotation_payload["projected_annotation"]),
        }
        return TaskOutput(
            prompt=str(prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=SINGLE_QUERY_ID,
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


def run_icon_field_frequency_lifecycle(
    task: IconFieldFrequencyTaskLifecycle,
    *,
    instance_seed: int,
    max_attempts: int,
    objective: IconFieldFrequencyObjective,
) -> TaskOutput:
    """Run the common icon-field render/output lifecycle for task-owned hooks."""

    frequency_spec = objective.resolve_frequency_spec(int(instance_seed), objective.params)
    scene_payload, image, render_params, pool_manifest = task._render_frequency_scene(
        instance_seed=int(instance_seed),
        params=objective.params,
        max_attempts=int(max_attempts),
        frequency_spec=frequency_spec,
        generation_defaults=objective.generation_defaults,
        render_defaults=objective.render_defaults,
        defaults=objective.defaults,
    )
    answer_value, binding = objective.bind_answer_annotation(scene_payload, frequency_spec)
    return task._build_frequency_task_output(
        instance_seed=int(instance_seed),
        scene_payload=scene_payload,
        image=image,
        render_params=render_params,
        pool_manifest=str(pool_manifest),
        prompt_defaults=objective.prompt_defaults,
        question_format=str(binding["question_format"]),
        answer_value=int(answer_value),
        binding=binding,
    )


__all__ = [
    "DOMAIN",
    "SCENE_ID",
    "IconFieldFrequencyObjective",
    "IconFieldFrequencyTaskLifecycle",
    "build_frequency_binding_payload",
    "run_icon_field_frequency_lifecycle",
]
