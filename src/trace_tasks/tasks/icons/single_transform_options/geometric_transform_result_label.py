"""Select the result of applying one geometric transform to a curated icon."""

from __future__ import annotations

from typing import Any, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ..shared.icon_transform import IDENTITY_TRANSFORM_ID

from .shared.defaults import SingleTransformOptionsDefaults
from .shared.output import (
    build_transform_trace_payload,
    common_selected_option_execution_payload,
    cue_for_operation,
    prepare_selected_option_prompt_binding,
    prepare_transform_scene_context,
    render_transform_scene_with_retries,
    task_output_fields,
    transform_for_operation,
)


TASK_ID = "task_icons__single_transform_options__geometric_transform_result_label"
DOMAIN = "icons"
SCENE_ID = "single_transform_options"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "rotate_90_clockwise_result_label",
    "rotate_90_counterclockwise_result_label",
    "rotate_180_result_label",
    "flip_horizontal_result_label",
    "flip_vertical_result_label",
)
_DEFAULTS = SingleTransformOptionsDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _operation_key(query_id: str) -> str:
    """Return the semantic operation branch represented by a result query."""

    return str(query_id).removesuffix("_result_label")


@register_task
class IconsSingleTransformOptionsGeometricTransformResultLabelTask:
    """Select the labeled result of applying one geometric transform."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic transform-result option instance."""

        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        context = prepare_transform_scene_context(
            instance_seed=int(instance_seed),
            params=task_params,
            generation_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            fallback_defaults=_DEFAULTS,
        )
        operation_key = _operation_key(str(query_id))
        target_transform_id = transform_for_operation(operation_key)
        operation_cue = cue_for_operation(operation_key)
        scene_payload, image = render_transform_scene_with_retries(
            context["scene_rng"],
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            option_count=int(context["option_count"]),
            answer_index=int(context["answer_index"]),
            target_transform_id=str(target_transform_id),
            operation_cue=str(operation_cue),
            pool_manifest=str(context["pool_manifest"]),
            transform_check_size_px=int(context["transform_check_size_px"]),
            render_params=context["render_params"],
            reference_transform_id=IDENTITY_TRANSFORM_ID,
        )

        binding = prepare_selected_option_prompt_binding(
            instance_seed=int(instance_seed),
            prompt_defaults=_PROMPT_DEFAULTS,
            operation_key=str(query_id),
            public_id=str(self.task_id),
            scene_id=SCENE_ID,
            branch_key=str(query_id),
            query_probabilities=query_probabilities,
            scene_context=context,
            scene_payload=scene_payload,
        )
        selected_cell = binding["selected_cell"]

        trace_payload = build_transform_trace_payload(
            public_id=str(self.task_id),
            scene_id=SCENE_ID,
            branch_key=str(query_id),
            scene_kind="icons_single_transform_options_result_label",
            scene_payload=scene_payload,
            selected_cell=selected_cell,
            query_spec=binding["query_spec"],
            render_params=context["render_params"],
            relation_payload={
                "target": "option_transform_equals_reference_after_operation",
                "operation_cue": str(scene_payload.operation_cue),
                "target_transform_id": str(scene_payload.target_transform_id),
                "answer_label": str(scene_payload.answer_label),
                "answer_transform_id": str(selected_cell["transform_id"]),
            },
            execution_payload=common_selected_option_execution_payload(
                public_id=str(self.task_id),
                scene_id=SCENE_ID,
                branch_key=str(query_id),
                query_probabilities=query_probabilities,
                scene_variant="reference_icon_with_transform_result_options",
                question_format="select_transformed_reference_icon_option",
                scene_payload=scene_payload,
                answer_label_probabilities=context["answer_label_probabilities"],
                option_transform_map_key="option_transform_ids_by_label",
            ),
            witness_symbolic={
                "reference_icon_id": str(scene_payload.icon_id),
                "reference_transform_id": IDENTITY_TRANSFORM_ID,
                "target_transform_id": str(scene_payload.target_transform_id),
                "selected_option_label": str(scene_payload.answer_label),
                "selected_option_transform_id": str(selected_cell["transform_id"]),
                "reference_icon_bbox": list(scene_payload.reference_cell["icon_bbox_xyxy"]),
                "selected_option_bbox": list(selected_cell["cell_bbox_xyxy"]),
            },
            projected_annotation=dict(binding["annotation_artifacts"]["projected_annotation"]),
        )
        return TaskOutput(
            **task_output_fields(
                prompt_artifacts=binding["prompt_artifacts"],
                answer_gt=binding["answer_gt"],
                annotation_gt=binding["annotation_gt"],
                image=image,
                trace_payload=trace_payload,
                scene_id=SCENE_ID,
                branch_key=str(query_id),
            )
        )


__all__ = ["IconsSingleTransformOptionsGeometricTransformResultLabelTask", "TASK_ID", "SUPPORTED_QUERY_IDS"]
