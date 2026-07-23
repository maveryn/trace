"""Scene-private lifecycle helpers for icon-grid category-count tasks."""

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
from ..shared.annotation import bbox_set_annotation

from .shared.defaults import IconGridDefaults
from .shared.rendering import sample_and_render_icon_grid_scene
from .shared.state import IconGridFrequencySpec, IconGridScenePayload
from .shared.styles import icon_grid_style_trace, resolve_icon_grid_render_params


DOMAIN = "icons"
SCENE_ID = "icon_grid"
NOISE_NAMESPACE = "icon_grid_category_count"

FrequencySpecResolver = Callable[[int, Mapping[str, Any]], IconGridFrequencySpec]
AnswerAnnotationBinder = Callable[[IconGridScenePayload, IconGridFrequencySpec], tuple[int, Mapping[str, Any]]]


@dataclass(frozen=True)
class IconGridCategoryObjective:
    """Task-owned hooks and metadata for one icon-grid category objective."""

    params: Mapping[str, Any]
    generation_defaults: Mapping[str, Any]
    render_defaults: Mapping[str, Any]
    prompt_defaults: Mapping[str, Any]
    defaults: IconGridDefaults
    resolve_frequency_spec: FrequencySpecResolver
    bind_answer_annotation: AnswerAnnotationBinder


def build_category_binding_payload(
    *,
    scene_payload: IconGridScenePayload,
    frequency_spec: IconGridFrequencySpec,
    counting_rule: str,
    question_format: str,
    counted_icon_ids: tuple[str, ...],
    annotation_indices: tuple[int, ...],
    annotation_bboxes: tuple[list[int], ...],
    target_count: int,
    counted_color_keys: tuple[str, ...] = (),
) -> tuple[int, Mapping[str, Any]]:
    """Build common trace binding fields for icon-grid category counts."""

    return int(target_count), {
        "counting_rule": str(counting_rule),
        "question_format": str(question_format),
        "counted_icon_ids": list(counted_icon_ids),
        "counted_color_keys": list(counted_color_keys),
        "annotation_indices": list(annotation_indices),
        "annotation_bboxes": list(annotation_bboxes),
        "target_count": int(target_count),
        "representative_rule": "topmost_row_then_left_to_right_cell_per_category",
        "object_count_probabilities": dict(frequency_spec.object_count_probabilities),
        "target_count_probabilities": dict(frequency_spec.target_count_probabilities),
        "scene_unique_type_total": int(scene_payload.distinct_type_count),
        "scene_unique_color_total": int(scene_payload.distinct_color_count),
    }


class IconGridCategoryTaskLifecycle:
    """Common lifecycle methods for visible icon-grid category tasks."""

    prompt_query_key: str
    scene_kind: str

    def _render_category_scene(
        self,
        *,
        instance_seed: int,
        params: Mapping[str, Any],
        max_attempts: int,
        frequency_spec: IconGridFrequencySpec,
        generation_defaults: Mapping[str, Any],
        render_defaults: Mapping[str, Any],
        defaults: IconGridDefaults,
    ) -> Tuple[IconGridScenePayload, Any, Dict[str, Any], str]:
        """Render one icon-grid category scene from a task-resolved spec."""

        render_params = resolve_icon_grid_render_params(
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
                scene_payload, image = sample_and_render_icon_grid_scene(
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

    def _build_category_task_output(
        self,
        *,
        instance_seed: int,
        scene_payload: IconGridScenePayload,
        image: Any,
        render_params: Mapping[str, Any],
        pool_manifest: str,
        prompt_defaults: Mapping[str, Any],
        question_format: str,
        answer_value: int,
        binding: Mapping[str, Any],
    ) -> TaskOutput:
        """Build prompt, trace, answer, and cell-bbox annotation payloads."""

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
        annotation_payload = bbox_set_annotation(annotation_bboxes)
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
                    "counted_color_keys": list(binding["counted_color_keys"]),
                    "type_frequencies": dict(scene_payload.type_frequencies),
                    "color_frequencies": dict(scene_payload.color_frequencies),
                    "representative_rule": str(binding["representative_rule"]),
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
                    "grid_rows": int(scene_payload.grid_rows),
                    "grid_cols": int(scene_payload.grid_cols),
                    "unique_type_total": int(scene_payload.distinct_type_count),
                    "unique_color_total": int(scene_payload.distinct_color_count),
                    "pool_manifest": str(pool_manifest),
                    "object_count_probabilities": dict(binding["object_count_probabilities"]),
                    "target_count_probabilities": dict(binding["target_count_probabilities"]),
                },
            },
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": SINGLE_QUERY_ID,
                "canvas_size": list(scene_payload.panel_geometry["canvas_size"]),
                "coord_space": "pixel",
                "panel_geometry": {
                    **dict(scene_payload.panel_geometry),
                    "grid_bbox_xyxy": [int(value) for value in scene_payload.grid_bbox_xyxy],
                    "cell_bboxes_xyxy": [
                        [[int(value) for value in bbox] for bbox in row]
                        for row in scene_payload.cell_bboxes_xyxy
                    ],
                },
                "style": icon_grid_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                ),
            },
            "render_map": {"image_id": "img0", "anchors": {}},
            "execution_trace": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "scene_variant": "visible_icon_grid",
                "query_id": SINGLE_QUERY_ID,
                "prompt_query_key": str(self.prompt_query_key),
                "question_format": str(question_format),
                "object_count": int(scene_payload.object_count),
                "target_count": int(answer_value),
                "grid_rows": int(scene_payload.grid_rows),
                "grid_cols": int(scene_payload.grid_cols),
                "unique_type_total": int(scene_payload.distinct_type_count),
                "unique_color_total": int(scene_payload.distinct_color_count),
                "scene_icon_ids": list(scene_payload.scene_icon_ids),
                "scene_color_keys": list(scene_payload.scene_color_keys),
                "type_frequencies": dict(scene_payload.type_frequencies),
                "color_frequencies": dict(scene_payload.color_frequencies),
                "annotation_indices": list(annotation_indices),
                "annotation_bboxes": list(annotation_bboxes),
                "representative_rule": str(binding["representative_rule"]),
                "counted_icon_ids": list(binding["counted_icon_ids"]),
                "counted_color_keys": list(binding["counted_color_keys"]),
            },
            "witness_symbolic": {
                "counted_icon_ids": list(binding["counted_icon_ids"]),
                "counted_color_keys": list(binding["counted_color_keys"]),
                "annotation_indices": list(annotation_indices),
                "annotation_bboxes": list(annotation_bboxes),
                "representative_rule": str(binding["representative_rule"]),
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


def run_icon_grid_category_lifecycle(
    task: IconGridCategoryTaskLifecycle,
    *,
    instance_seed: int,
    max_attempts: int,
    objective: IconGridCategoryObjective,
) -> TaskOutput:
    """Run the common icon-grid render/output lifecycle for task-owned hooks."""

    frequency_spec = objective.resolve_frequency_spec(int(instance_seed), objective.params)
    scene_payload, image, render_params, pool_manifest = task._render_category_scene(
        instance_seed=int(instance_seed),
        params=objective.params,
        max_attempts=int(max_attempts),
        frequency_spec=frequency_spec,
        generation_defaults=objective.generation_defaults,
        render_defaults=objective.render_defaults,
        defaults=objective.defaults,
    )
    answer_value, binding = objective.bind_answer_annotation(scene_payload, frequency_spec)
    return task._build_category_task_output(
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
    "IconGridCategoryObjective",
    "IconGridCategoryTaskLifecycle",
    "build_category_binding_payload",
    "run_icon_grid_category_lifecycle",
]
