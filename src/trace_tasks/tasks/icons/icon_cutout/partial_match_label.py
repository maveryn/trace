"""Match a partial curated-icon fragment to the full labeled icon option."""

from __future__ import annotations

from typing import Any, Dict

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import (
    group_default,
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from ..shared.annotation import bbox_map_annotation
from .shared.annotations import fragment_option_annotation_boxes, matching_option_cell
from .shared.defaults import IconCutoutDefaults
from .shared.rendering import sample_and_render_icon_cutout_scene
from .shared.sampling import (
    icon_cutout_labels,
    resolve_icon_cutout_answer_index,
    resolve_icon_cutout_object_count,
)
from .shared.styles import icon_cutout_style_trace, resolve_icon_cutout_render_params


TASK_ID = "task_icons__icon_cutout__partial_match_label"
DOMAIN = "icons"
SCENE_ID = "icon_cutout"
QUERY_ID = "partial_icon_match_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
NOISE_NAMESPACE = "icon_cutout_partial_match"


_DEFAULTS = IconCutoutDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class IconsIconCutoutPartialMatchLabelTask:
    """Select which full curated icon generated the partial fragment."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic partial-icon matching instance."""

        scene_rng = spawn_rng(int(instance_seed), "scene")
        render_params = resolve_icon_cutout_render_params(
            params,
            render_defaults=_RENDER_DEFAULTS,
            defaults=_DEFAULTS,
            instance_seed=int(instance_seed),
        )
        pool_manifest = str(
            params.get("pool_manifest", group_default(_GEN_DEFAULTS, "pool_manifest", _DEFAULTS.pool_manifest))
        )
        object_count = resolve_icon_cutout_object_count(params, gen_defaults=_GEN_DEFAULTS, defaults=_DEFAULTS)
        labels = icon_cutout_labels(int(object_count))
        matching_index = resolve_icon_cutout_answer_index(scene_rng, params=params, labels=labels)

        scene_payload = None
        image = None
        last_error: Exception | None = None
        for _ in range(max(1, int(max_attempts))):
            try:
                scene_payload, image = sample_and_render_icon_cutout_scene(
                    scene_rng,
                    instance_seed=int(instance_seed),
                    render_params=render_params,
                    pool_manifest=str(pool_manifest),
                    labels=labels,
                    matching_index=int(matching_index),
                    noise_namespace=NOISE_NAMESPACE,
                )
                break
            except Exception as exc:
                last_error = exc
                continue
        if scene_payload is None or image is None:
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
                "question_text_partial_icon_match_label",
                "annotation_hint",
                "answer_hint",
                "json_example",
                "json_example_answer_only",
            ),
            context=f"prompt defaults for {self.task_id}",
        )
        prompt_selection = render_scene_prompt_variants(
            domain=self.domain,
            scene_id=SCENE_ID,
            bundle_id=str(prompt_defaults["bundle_id"]),
            scene_key=str(prompt_defaults["scene_key"]),
            task_key=str(prompt_defaults["task_key"]),
            answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
            dynamic_slots={
                "object_description": str(prompt_defaults["object_description"]),
                "question_text": str(prompt_defaults["question_text_partial_icon_match_label"]),
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

        matching_cell = matching_option_cell(scene_payload)
        annotation_artifacts = bbox_map_annotation(
            fragment_option_annotation_boxes(scene_payload, matching_cell)
        )
        answer_gt = TypedValue(type="option_letter", value=str(scene_payload.answer_label))
        annotation_gt = TypedValue(
            type=str(annotation_artifacts["annotation_type"]),
            value=dict(annotation_artifacts["annotation_value"]),
        )
        trace_payload = {
            "scene_ir": {
                "scene_kind": "icons_icon_cutout_partial_match_label",
                "scene_id": SCENE_ID,
                "query_id": str(QUERY_ID),
                "entities": [dict(scene_payload.reference_cell), *[dict(item) for item in scene_payload.scene_cells]],
                "relations": {
                    "target": "full_icon_option_matches_partial_source_fragment",
                    "query_id": str(QUERY_ID),
                    "answer_label": str(scene_payload.answer_label),
                    "answer_icon_id": str(scene_payload.correct_icon_id),
                    "matching_cell_label": str(scene_payload.answer_label),
                },
                "frames": {
                    "pixel": {"origin": [0.0, 0.0], "x_positive": "right", "y_positive": "down"},
                    "panels": dict(scene_payload.panel_geometry),
                },
            },
            "query_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(QUERY_ID),
                "template_id": str(prompt_defaults["bundle_id"]),
                "prompt_variant": dict(prompt_artifacts.prompt_variant),
                "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
                "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
                "params": {
                    "task_id": str(self.task_id),
                    "scene_id": SCENE_ID,
                    "query_id": str(QUERY_ID),
                    "query_id_probabilities": {str(QUERY_ID): 1.0},
                    "object_count": int(scene_payload.object_count),
                    "pool_manifest": str(pool_manifest),
                    "answer_label": str(scene_payload.answer_label),
                    "fragment_window_style": str(scene_payload.fragment_window_style),
                    "fragment_visible_alpha_ratio": float(scene_payload.fragment_visible_alpha_ratio),
                    "fragment_alpha_density": float(scene_payload.fragment_alpha_density),
                },
            },
            "render_spec": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(QUERY_ID),
                "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
                "coord_space": "pixel",
                "panel_geometry": dict(scene_payload.panel_geometry),
                "style": icon_cutout_style_trace(
                    render_params=render_params,
                    sampled_palette_rgb=scene_payload.sampled_palette_rgb,
                ),
            },
            "render_map": {
                "image_id": "img0",
                "anchors": {
                    "source_fragment": dict(scene_payload.reference_cell),
                    "answer_label": str(scene_payload.answer_label),
                    "selected_option": dict(matching_cell),
                    "scene_cells": [dict(item) for item in scene_payload.scene_cells],
                },
            },
            "execution_trace": {
                "task_id": str(self.task_id),
                "scene_id": SCENE_ID,
                "query_id": str(QUERY_ID),
                "query_id_probabilities": {str(QUERY_ID): 1.0},
                "scene_variant": "partial_fragment_with_full_icon_options",
                "question_format": "select_full_icon_option_matching_partial_source_fragment",
                "object_count": int(scene_payload.object_count),
                "cell_labels": list(scene_payload.cell_labels),
                "answer": str(scene_payload.answer_label),
                "answer_label": str(scene_payload.answer_label),
                "correct_icon_id": str(scene_payload.correct_icon_id),
                "option_icon_ids_by_label": {
                    str(cell["label"]): str(cell["icon_id"])
                    for cell in scene_payload.scene_cells
                },
                "fragment_window_style": str(scene_payload.fragment_window_style),
                "fragment_visible_alpha_ratio": float(scene_payload.fragment_visible_alpha_ratio),
                "fragment_alpha_density": float(scene_payload.fragment_alpha_density),
                "annotation_roles": ["source_fragment", "selected_option"],
            },
            "witness_symbolic": {
                "source_fragment_icon_id": str(scene_payload.correct_icon_id),
                "selected_option_label": str(scene_payload.answer_label),
                "selected_option_icon_id": str(matching_cell["icon_id"]),
                "source_fragment_bbox": list(scene_payload.reference_cell["fragment_bbox_xyxy"]),
                "selected_option_bbox": list(matching_cell["cell_bbox_xyxy"]),
            },
            "projected_annotation": dict(annotation_artifacts["projected_annotation"]),
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
            query_id=str(QUERY_ID),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["IconsIconCutoutPartialMatchLabelTask"]
