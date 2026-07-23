"""Compute the direct final bearing after following a two-leg route."""

from __future__ import annotations
from dataclasses import replace
from typing import Any, Dict, Tuple
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    split_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from ._lifecycle import (
    build_bearing_route_trace_payload,
    prepare_bearing_route_rendering,
)
from .shared.construction import resolve_final_bearing_route_case
from .shared.rendering import render_final_bearing_scene
from .shared.state import SCENE_ID, RenderedBearingScene

TASK_ID = "task_geometry__bearing_route__final_bearing_value"
QUERY_ID = "final_bearing_value"
SCENE_VARIANT = "drawn_route_bearing"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (QUERY_ID,)
_OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
_DISTRACTOR_OFFSETS: Tuple[int, ...] = (-180, -135, -90, -45, -30, -15, 15, 30, 45, 90, 135, 180)
_SCENE_DEFAULTS = get_scene_defaults("geometry", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    split_scene_generation_rendering_prompt_defaults(_SCENE_DEFAULTS, task_id=TASK_ID)
)


def _resolve_bearing_options(*, instance_seed: int, correct_bearing: int) -> tuple[Tuple[int, ...], int]:
    """Return six visible MCQ bearing values and the correct option index."""

    target_index_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{QUERY_ID}.option_index")
    target_index = int(uniform_choice(target_index_rng, tuple(range(len(_OPTION_LABELS)))))
    distractor_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.{QUERY_ID}.option_values")
    shuffled_offsets = list(_DISTRACTOR_OFFSETS)
    distractor_rng.shuffle(shuffled_offsets)
    distractors: list[int] = []
    seen = {int(correct_bearing) % 360}
    for offset in shuffled_offsets:
        value = (int(correct_bearing) + int(offset)) % 360
        if value in seen:
            continue
        seen.add(value)
        distractors.append(value)
        if len(distractors) >= len(_OPTION_LABELS) - 1:
            break
    if len(distractors) < len(_OPTION_LABELS) - 1:
        raise ValueError("failed to construct six unique bearing options")

    option_values = list(distractors)
    option_values.insert(target_index, int(correct_bearing) % 360)
    return tuple(int(value) for value in option_values), int(target_index)


@register_task
class GeometryBearingRouteFinalBearingValueTask:
    """Compute the compass bearing from the start point to the final point."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations', 'formula_evaluation')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        """Own direct-bearing answer binding after route rendering and projection."""
        selected_query, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            task_id=TASK_ID,
        )
        route_case, bearing_probabilities = resolve_final_bearing_route_case(
            params=task_params,
            instance_seed=int(instance_seed),
            bearing_namespace=f"{TASK_ID}.{selected_query}.bearing",
            leg_length_namespace=f"{TASK_ID}.{selected_query}.leg_length",
            leg_order_namespace=f"{TASK_ID}.{selected_query}.leg_order",
        )
        if route_case.final_bearing is None:
            raise ValueError("final bearing task requires a resolved final bearing")
        option_values, target_index = _resolve_bearing_options(
            instance_seed=int(instance_seed),
            correct_bearing=int(route_case.final_bearing),
        )
        route_case = replace(
            route_case,
            option_count=len(_OPTION_LABELS),
            target_index=int(target_index),
            option_labels=_OPTION_LABELS,
            option_values=option_values,
        )
        prepared = prepare_bearing_route_rendering(
            route_case=route_case,
            scene_renderer=render_final_bearing_scene,
            domain=self.domain,
            prompt_defaults=_PROMPT_DEFAULTS,
            prompt_key=str(selected_query),
            instance_seed=int(instance_seed),
            params=task_params,
            render_defaults=_RENDER_DEFAULTS,
            max_attempts=int(max_attempts),
            style_namespace=f"{TASK_ID}.style",
        )
        rendered = prepared.runtime.rendered
        answer_value = str(_OPTION_LABELS[int(target_index)])
        answer_gt = TypedValue(type="option_letter", value=answer_value)
        annotation_gt = TypedValue(
            type="point_map", value=dict(prepared.annotation_keyed_points)
        )
        query_params = {
            "scene_id": SCENE_ID,
            "scene_variant": SCENE_VARIANT,
            "query_id": str(selected_query),
            "query_id_probabilities": dict(query_probabilities),
            "target_bearing_probabilities": dict(bearing_probabilities),
            "option_count": int(route_case.option_count),
            "target_index": int(target_index),
            "option_labels": list(route_case.option_labels),
            "option_values": list(route_case.option_values),
            **dict(rendered.witness),
        }
        trace_payload = build_bearing_route_trace_payload(
            runtime=prepared.runtime,
            prompt_artifacts=prepared.prompt_artifacts,
            noise_meta=prepared.noise_meta,
            branch_name=str(selected_query),
            branch_params=query_params,
            scene_variant=SCENE_VARIANT,
            answer_type="option_letter",
            answer_value=answer_value,
            witness_kind="bearing_route_final_bearing",
            annotation_bboxes=prepared.annotation_bboxes,
            annotation_points=prepared.annotation_points,
            annotation_keyed_bboxes=prepared.annotation_keyed_bboxes,
            annotation_keyed_points=prepared.annotation_keyed_points,
        )
        return TaskOutput(
            prompt=str(prepared.prompt_artifacts.prompt),
            answer_gt=answer_gt,
            annotation_gt=annotation_gt,
            image=prepared.image,
            image_id="img0",
            trace_payload=trace_payload,
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(selected_query),
            prompt_variants=dict(prepared.prompt_artifacts.prompt_variants),
        )


__all__ = ["GeometryBearingRouteFinalBearingValueTask"]
