"""Infer total area from one shaded area-partition diagram."""

from __future__ import annotations
from typing import Any, Dict, Mapping, Tuple
from trace_tasks.core.types import TypedValue
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from .shared.annotations import (
    keyed_bbox_annotation,
    keyed_region_bboxes,
    keyed_region_points,
)
from .shared.construction import AREA_PARTITION_CASES, resolve_area_partition_problem
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.prompts import build_area_partition_prompt_artifacts
from .shared.rendering import (
    create_area_partition_render_context,
    render_area_partition_scene,
)
from .shared.state import SCENE_ID

TASK_ID = "task_geometry__area_partition__total_area_value"
AREA_PARTITION_SCENE_ID = SCENE_ID
PROMPT_QUERY_KEY = "total_area_from_shaded_partition"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (PROMPT_QUERY_KEY,)
_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "geometry", SCENE_ID, task_id=TASK_ID
    )
)


def _area_partition_query_spec(
    *, prompt_artifacts: Any, selected_query: str, params: Mapping[str, Any]
) -> dict[str, Any]:
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts, query_id=str(selected_query), params=params
    )
    query_spec["scene_id"] = SCENE_ID
    return query_spec


@register_task
class GeometryAreaPartitionTotalAreaValueTask:
    """Infer the full outer area from one shaded equal-area partition."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Own total-area solving while delegating neutral partition rendering.

        This task selects the valid partition case, binds the integer answer,
        and exports keyed region annotation for the outer and shaded regions.
        """
        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=PROMPT_QUERY_KEY,
            task_id=TASK_ID,
        )
        problem = resolve_area_partition_problem(
            instance_seed=int(instance_seed),
            params=task_params,
            partition_cases=AREA_PARTITION_CASES,
            sampling_namespace=f"{TASK_ID}.{selected_query}.case",
        )
        last_error: Exception | None = None
        rendered = None
        render_meta: Dict[str, Any] | None = None
        for _attempt_index in range(max(1, int(max_attempts))):
            try:
                ctx, render_meta_attempt = create_area_partition_render_context(
                    instance_seed=int(instance_seed),
                    params=task_params,
                    render_defaults=_RENDER_DEFAULTS,
                )
                rendered = render_area_partition_scene(ctx, problem)
                render_meta = dict(render_meta_attempt)
                if ctx.scene_transform is not None:
                    render_meta["single_object_scene_rotation"] = (
                        ctx.scene_transform.metadata()
                    )
                break
            except Exception as exc:
                last_error = exc
                continue
        if rendered is None or render_meta is None:
            raise RuntimeError(f"failed to generate {TASK_ID}") from last_error
        image, noise_meta = apply_post_image_noise(
            rendered.image,
            instance_seed=int(instance_seed),
            params=task_params,
            default_config=POST_IMAGE_NOISE_DEFAULTS,
        )
        _prompt_defaults, prompt_artifacts = build_area_partition_prompt_artifacts(
            domain=self.domain,
            prompt_query_key=str(selected_query),
            dynamic_slots={"shape_type": str(rendered.witness.shape_type)},
            instance_seed=int(instance_seed),
        )
        annotation_gt = keyed_bbox_annotation(rendered)
        annotation_keyed_bboxes = keyed_region_bboxes(rendered)
        annotation_keyed_points = keyed_region_points(annotation_keyed_bboxes)
        answer_value = int(round(float(rendered.answer)))
        answer_gt = TypedValue(type="integer", value=int(answer_value))
        witness_payload = {
            "formula_family": str(selected_query),
            **dict(rendered.witness.to_trace()),
        }
        witness_payload["answer_value"] = int(answer_value)
        query_params = {
            "scene_id": SCENE_ID,
            "scene_variant": str(problem.scene_variant),
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "target_support_probabilities": dict(problem.support_probabilities),
            **dict(witness_payload),
        }
        query_spec = _area_partition_query_spec(
            prompt_artifacts=prompt_artifacts,
            selected_query=str(selected_query),
            params=query_params,
        )
        trace_payload: Dict[str, Any] = {
            "scene_ir": {
                "scene_kind": "geometry_area_partition",
                "scene_id": SCENE_ID,
                "entities": [dict(entity) for entity in rendered.scene_entities],
                "relations": {
                    "query_id": str(selected_query),
                    "scene_variant": str(problem.scene_variant),
                    "answer_value": int(answer_value),
                    "annotation_roles": list(rendered.annotation_roles),
                },
            },
            "query_spec": query_spec,
            "render_spec": {
                "canvas_size": [int(image.size[0]), int(image.size[1])],
                "coord_space": "pixel",
                "post_image_noise": dict(noise_meta),
                **dict(render_meta),
            },
            "render_map": {
                "coord_space": "pixel",
                **dict(rendered.render_map),
            },
            "execution_trace": {
                "scene_id": SCENE_ID,
                "scene_variant": str(problem.scene_variant),
                "query_id": str(selected_query),
                "query_id_probabilities": dict(query_probabilities),
                "answer_type": "integer",
                "answer_value": int(answer_value),
                "annotation_roles": list(rendered.annotation_roles),
                "reasoning_steps": 1,
                **dict(witness_payload),
            },
            "witness_symbolic": {
                "type": "area_partition_formula",
                "scene_id": SCENE_ID,
                "query_id": str(selected_query),
                "answer_value": int(answer_value),
                "source_witness_type": "bbox_map",
                "original_annotation_value": dict(annotation_keyed_bboxes),
                **dict(witness_payload),
            },
            "projected_annotation": {
                "type": "bbox_map",
                "bbox_map": dict(annotation_keyed_bboxes),
                "pixel_bbox_map": dict(annotation_keyed_bboxes),
                "point_map": dict(annotation_keyed_points),
                "pixel_point_map": dict(annotation_keyed_points),
            },
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
            query_id=str(selected_query),
            prompt_variants=dict(prompt_artifacts.prompt_variants),
        )


__all__ = ["AREA_PARTITION_SCENE_ID", "GeometryAreaPartitionTotalAreaValueTask"]
