from __future__ import annotations

from typing import Any, Dict

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.annotation_artifacts import point_annotation_artifacts
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.prompt_variants import build_prompt_query_spec

from .shared.output import compose_radial_trace_payload, draw_radial_scene_artifacts, radial_entity_trace_records
from .shared.prompts import render_radial_prompt
from .shared.rendering import render_radial_missing_symbol_scene
from .shared.sampling import build_missing_code_symbol_choice, build_with_retries, resolve_radial_scene_variant


TASK_ID = "task_symbolic__radial_code_wheel__missing_code_symbol_label"
INTERNAL_QUERY_KEY = "missing_code_symbol_label"
TASK_PROMPT_KEY = "missing_code_symbol_label"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "symbolic",
    "radial_code_wheel",
    task_id=TASK_ID,
)


@register_task
class SymbolicRadialMissingCodeSymbolLabelTask:
    """Infer the hidden symbol in an incomplete radial code."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
    domain = "symbolic"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one target-output plus partial-code missing-symbol task."""

        scene_variant, scene_variant_probabilities = resolve_radial_scene_variant(
            params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{TASK_ID}.scene_variant",
        )
        dataset = build_with_retries(
            lambda retry_seed: build_missing_code_symbol_choice(
                rng=spawn_rng(int(retry_seed), f"{TASK_ID}.dataset"),
                instance_seed=int(retry_seed),
                namespace=TASK_ID,
                scene_variant=str(scene_variant),
                scene_variant_probabilities=scene_variant_probabilities,
                params=params,
                gen_defaults=_GEN_DEFAULTS,
            ),
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            failure_message=f"failed to generate radial missing-code-symbol instance for {TASK_ID}",
        )
        render_artifacts = draw_radial_scene_artifacts(
            instance_seed=int(instance_seed),
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            namespace=f"{TASK_ID}.background",
            render_scene=render_radial_missing_symbol_scene,
            render_kwargs={
                "target_output_label": str(dataset.target_output_label),
                "partial_code": str(dataset.partial_code),
                "terminal_specs": tuple(dataset.terminal_specs),
                "target_code": str(dataset.target_code),
            },
            annotation_source="item_points_px",
        )
        prompt_runtime = render_radial_prompt(
            _PROMPT_DEFAULTS,
            domain="symbolic",
            scene_id="radial_code_wheel",
            scene_variant=str(dataset.scene_variant),
            task_key=TASK_PROMPT_KEY,
            object_description_key=f"object_description_missing_code_symbol_label_{dataset.scene_variant}",
            annotation_hint_key="annotation_hint_missing_code_symbol_label",
            answer_hint_key="answer_hint_missing_code_symbol_label",
            json_example_key="json_example_missing_code_symbol_label",
            json_example_answer_only_key="json_example_answer_only_missing_code_symbol_label",
            instance_seed=int(instance_seed),
            context=f"prompt defaults for {TASK_ID}",
        )

        projection = render_artifacts.projection
        missing_point = projection.item_points[str(dataset.annotation_item_id)]
        annotation_artifacts = point_annotation_artifacts(missing_point)
        answer_gt = TypedValue(type="string", value=str(dataset.answer_value))
        query_params = {
            "query_id": INTERNAL_QUERY_KEY,
            "internal_query_id": INTERNAL_QUERY_KEY,
            "internal_query_id_probabilities": {INTERNAL_QUERY_KEY: 1.0},
            "scene_id": "radial_code_wheel",
            "scene_variant": str(dataset.scene_variant),
            "scene_variant_probabilities": dict(dataset.scene_variant_probabilities),
            "target_answer_support": [str(symbol) for symbol in dataset.target_answer_support],
            "question_format": INTERNAL_QUERY_KEY,
            "target_output_label": str(dataset.target_output_label),
            "partial_code": str(dataset.partial_code),
            "missing_position_index": int(dataset.missing_position_index),
            "missing_ring_role": str(dataset.missing_ring_role),
        }
        prompt_query_spec = build_prompt_query_spec(
            prompt_artifacts=prompt_runtime.artifacts,
            query_id=INTERNAL_QUERY_KEY,
            params=query_params,
        )
        trace_payload = compose_radial_trace_payload(
            scene_id="radial_code_wheel",
            scene_variant=str(dataset.scene_variant),
            projection=projection,
            prompt_query_spec=prompt_query_spec,
            prompt_bundle_id=str(prompt_runtime.metadata["bundle_id"]),
            relations={
                "query_id": INTERNAL_QUERY_KEY,
                "internal_query_id": INTERNAL_QUERY_KEY,
                "scene_id": "radial_code_wheel",
                "scene_variant": str(dataset.scene_variant),
                "answer_value": str(dataset.answer_value),
                "target_output_label": str(dataset.target_output_label),
                "partial_code": str(dataset.partial_code),
                "missing_ring_role": str(dataset.missing_ring_role),
            },
            execution_trace={
                **dict(query_params),
                "answer_value": str(dataset.answer_value),
                "answer_type": "string",
                "annotation_item_ids": [str(dataset.annotation_item_id)],
                "radial_code_wheel_metadata": dict(dataset.metadata),
                "radial_code_wheel_entities": radial_entity_trace_records(projection.entity_records),
            },
            annotation_artifacts=annotation_artifacts,
            answer_gt=answer_gt,
        )
        trace_payload["render_map"].update(
            {
                "missing_symbol_point_px": list(annotation_artifacts.value),
                "target_output_bbox_px": list(projection.item_bboxes["target_output"]),
                "partial_code_bbox_px": list(projection.item_bboxes["partial_code"]),
            }
        )

        return TaskOutput(
            prompt=str(prompt_runtime.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_artifacts.annotation_gt,
            image=render_artifacts.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=dict(projection.task_versions),
            scene_id="radial_code_wheel",
            query_id=INTERNAL_QUERY_KEY,
            prompt_variants=dict(prompt_runtime.prompt_variants),
        )


__all__ = ["SymbolicRadialMissingCodeSymbolLabelTask"]
