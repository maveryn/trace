"""Select the source option that becomes the Reference after one transform."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id

from .shared.defaults import SingleTransformOptionsDefaults
from .shared.sampling import (
    ALL_OPTION_TRANSFORMS,
    compose_transform_ids,
)
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


TASK_ID = "task_icons__single_transform_options__inverse_geometric_transform_source_label"
DOMAIN = "icons"
SCENE_ID = "single_transform_options"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (
    "rotate_90_clockwise_source_label",
    "rotate_90_counterclockwise_source_label",
    "rotate_180_source_label",
    "flip_horizontal_source_label",
    "flip_vertical_source_label",
)
_DEFAULTS = SingleTransformOptionsDefaults()
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


def _operation_key(query_id: str) -> str:
    """Return the semantic operation branch represented by an inverse query."""

    return str(query_id).removesuffix("_source_label")


def _source_transform_probabilities(valid_sources: Sequence[str]) -> Dict[str, float]:
    """Return a support map for the sampled hidden source transform."""

    source_set = {str(value) for value in valid_sources}
    return {
        str(value): (1.0 / float(len(source_set)) if str(value) in source_set else 0.0)
        for value in ALL_OPTION_TRANSFORMS
    }


def _sample_inverse_option_transforms(
    rng,
    *,
    target_transform_id: str,
    answer_index: int,
    option_count: int,
) -> Tuple[str, str, Tuple[str, ...], Dict[str, float]]:
    """Sample source-option transforms for an inverse transform task."""

    target = str(target_transform_id)
    valid_sources = tuple(
        str(transform_id)
        for transform_id in ALL_OPTION_TRANSFORMS
        if compose_transform_ids(after_transform_id=target, before_transform_id=str(transform_id)) in set(ALL_OPTION_TRANSFORMS)
    )
    if not valid_sources:
        raise ValueError("inverse transform task found no valid source transforms")
    correct_source = str(rng.choice(valid_sources))
    reference_transform = compose_transform_ids(after_transform_id=target, before_transform_id=correct_source)

    distractors = [
        str(transform_id)
        for transform_id in ALL_OPTION_TRANSFORMS
        if str(transform_id) not in {str(correct_source), str(reference_transform)}
    ]
    rng.shuffle(distractors)
    if len(distractors) < int(option_count) - 1:
        raise ValueError("not enough inverse-transform source distractors")

    option_transform_ids: list[str] = []
    cursor = 0
    for index in range(int(option_count)):
        if int(index) == int(answer_index):
            option_transform_ids.append(str(correct_source))
        else:
            option_transform_ids.append(str(distractors[int(cursor)]))
            cursor += 1
    probabilities = _source_transform_probabilities(valid_sources)
    return str(correct_source), str(reference_transform), tuple(option_transform_ids), probabilities


@register_task
class IconsSingleTransformOptionsInverseGeometricTransformSourceLabelTask:
    """Select the option that would become the Reference after the shown operation."""

    task_id = TASK_ID
    reasoning_operations = ('transformation',)
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate one deterministic inverse-transform source-option instance."""

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
        (
            correct_source_transform_id,
            reference_transform_id,
            option_transform_ids,
            source_transform_probabilities,
        ) = _sample_inverse_option_transforms(
            context["scene_rng"],
            target_transform_id=str(target_transform_id),
            answer_index=int(context["answer_index"]),
            option_count=int(context["option_count"]),
        )
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
            reference_transform_id=str(reference_transform_id),
            option_transform_ids=tuple(option_transform_ids),
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
        binding["query_spec"]["params"].update(
            {
                "reference_transform_id": str(reference_transform_id),
                "correct_source_transform_id": str(correct_source_transform_id),
                "source_transform_probabilities": dict(source_transform_probabilities),
            }
        )
        selected_cell = binding["selected_cell"]

        trace_payload = build_transform_trace_payload(
            public_id=str(self.task_id),
            scene_id=SCENE_ID,
            branch_key=str(query_id),
            scene_kind="icons_single_transform_options_inverse_source_label",
            scene_payload=scene_payload,
            selected_cell=selected_cell,
            query_spec=binding["query_spec"],
            render_params=context["render_params"],
            relation_payload={
                "target": "option_after_operation_equals_reference",
                "operation_cue": str(scene_payload.operation_cue),
                "target_transform_id": str(scene_payload.target_transform_id),
                "reference_transform_id": str(reference_transform_id),
                "answer_label": str(scene_payload.answer_label),
                "answer_source_transform_id": str(selected_cell["transform_id"]),
            },
            execution_payload={
                **common_selected_option_execution_payload(
                    public_id=str(self.task_id),
                    scene_id=SCENE_ID,
                    branch_key=str(query_id),
                    query_probabilities=query_probabilities,
                    scene_variant="transformed_reference_icon_with_source_options",
                    question_format="select_source_option_that_becomes_reference_after_operation",
                    scene_payload=scene_payload,
                    answer_label_probabilities=context["answer_label_probabilities"],
                    option_transform_map_key="option_source_transform_ids_by_label",
                ),
                "reference_transform_id": str(reference_transform_id),
                "correct_source_transform_id": str(correct_source_transform_id),
            },
            witness_symbolic={
                "reference_icon_id": str(scene_payload.icon_id),
                "reference_transform_id": str(reference_transform_id),
                "target_transform_id": str(scene_payload.target_transform_id),
                "selected_option_label": str(scene_payload.answer_label),
                "selected_option_source_transform_id": str(selected_cell["transform_id"]),
                "selected_option_after_operation_transform_id": compose_transform_ids(
                    after_transform_id=str(scene_payload.target_transform_id),
                    before_transform_id=str(selected_cell["transform_id"]),
                ),
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


__all__ = ["IconsSingleTransformOptionsInverseGeometricTransformSourceLabelTask", "TASK_ID", "SUPPORTED_QUERY_IDS"]
