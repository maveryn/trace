"""Neutral lifecycle runner for migrated pixel-village count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ....core.query_ids import SINGLE_QUERY_ID
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from .shared.output import (
    pixel_village_render_map,
    pixel_village_render_spec,
    pixel_village_scene_ir,
)
from .shared.prompts import build_pixel_village_prompt_artifacts
from .shared.sampling import SCENE_ID
from .shared.state import PixelVillageCountBinding


@dataclass(frozen=True)
class PixelVillageCountPlan:
    """Public-task plan describing one count objective's local hooks."""

    public_id: str
    prompt_query_key: str
    bind_result: Callable[..., PixelVillageCountBinding]
    supported_query_ids: Tuple[str, ...] = (SINGLE_QUERY_ID,)


def run_pixel_village_count_lifecycle(
    *,
    task: Any,
    plan: PixelVillageCountPlan,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Dict[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Select the query branch, call public-owned binding hooks, and emit output."""

    resolved_query_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in plan.supported_query_ids),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(plan.public_id),
        namespace=f"{plan.public_id}:query",
    )
    binding = plan.bind_result(
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        namespace=str(plan.public_id),
    )
    prompt_artifacts = build_pixel_village_prompt_artifacts(
        domain=str(task.domain),
        scene_id=SCENE_ID,
        prompt_defaults=binding.prompt_defaults,
        prompt_query_key=str(plan.prompt_query_key),
        slots=dict(binding.slots),
        instance_seed=int(instance_seed),
    )
    query_params = {
        "query_id": str(resolved_query_id),
        "prompt_query_key": str(plan.prompt_query_key),
        "query_id_probabilities": dict(query_probabilities),
        **dict(binding.branch_params),
    }
    trace_payload = {
        "scene_ir": pixel_village_scene_ir(
            domain=str(task.domain),
            scene_id=SCENE_ID,
            scene=binding.scene,
            relations={
                "query_id": str(resolved_query_id),
                "prompt_query_key": str(plan.prompt_query_key),
                **dict(binding.scene_relations),
            },
        ),
        "query_spec": {
            "task_id": str(task.task_id),
            "query_id": str(resolved_query_id),
            "prompt_query_key": str(plan.prompt_query_key),
            "prompt_variant_active_key": prompt_artifacts.prompt_variant_active_key,
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": query_params,
        },
        "render_spec": pixel_village_render_spec(binding.scene, scene_id=SCENE_ID),
        "render_map": pixel_village_render_map(
            entity_bboxes=binding.entity_bboxes,
            counted_entity_ids=binding.counted_entity_ids,
            extra=binding.render_map_extra,
        ),
        "execution_trace": {
            "query_id": str(resolved_query_id),
            "prompt_query_key": str(plan.prompt_query_key),
            "scene_id": SCENE_ID,
            **dict(binding.execution_trace),
        },
        "witness_symbolic": dict(binding.witness_symbolic),
        "projected_annotation": dict(binding.projected_annotation),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants={str(key): str(value) for key, value in prompt_artifacts.prompt_variants.items()},
        answer_gt=TypedValue(type="integer", value=int(binding.answer)),
        annotation_gt=TypedValue(type="bbox_set", value=list(binding.annotation_value)),
        image=binding.scene.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(resolved_query_id),
    )


__all__ = ["PixelVillageCountPlan", "run_pixel_village_count_lifecycle"]
