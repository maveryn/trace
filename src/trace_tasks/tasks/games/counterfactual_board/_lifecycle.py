"""Scene-private lifecycle for counterfactual-board public tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.visual_defaults import (
    default_noise_fallback,
    load_scene_noise_defaults,
)

from .shared.annotations import (
    bbox_set_for_counted_elements,
    counted_elements_for_axis,
    segment_set_for_counted_elements,
)
from .shared.output import build_counterfactual_board_trace_payload, json_ready
from .shared.prompts import build_counterfactual_board_prompt_artifacts
from .shared.rendering import render_counterfactual_board
from .shared.state import (
    CounterfactualBoardCase,
    HORIZONTAL_LINE_AXIS,
    SCENE_ID,
    VERTICAL_LINE_AXIS,
)

CaseBuilder = Callable[[int], CounterfactualBoardCase]


@dataclass(frozen=True)
class CounterfactualBoardObjectivePlan:
    """Task-owned case builder consumed by neutral scene lifecycle plumbing."""

    construct_case: CaseBuilder


def run_counterfactual_board_lifecycle(
    *,
    task_id: str,
    domain: str,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    params: Mapping[str, object],
    render_defaults: Mapping[str, object],
    instance_seed: int,
    max_attempts: int,
    objective: CounterfactualBoardObjectivePlan,
) -> TaskOutput:
    """Run rendering, prompt assembly, annotation projection, and output wiring."""

    visual_defaults = load_scene_noise_defaults(
        domain=str(domain),
        scene_id=SCENE_ID,
        fallback=default_noise_fallback(apply_prob=0.5),
        merge_with_fallback=True,
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            case = objective.construct_case(attempt_seed)
            rendered = render_counterfactual_board(
                style=str(case.style),
                rows=int(case.rows),
                cols=int(case.cols),
                board_kind=str(case.board_kind),
                instance_seed=int(attempt_seed),
                params=params,
                rendering_defaults=render_defaults,
                visual_defaults=visual_defaults,
            )
            counted_elements = counted_elements_for_axis(
                counted_axis=str(case.counted_axis),
                rows=int(case.rows),
                cols=int(case.cols),
                board_bbox=rendered.layout.board_bbox_px,
                line_thickness_px=int(rendered.layout.line_annotation_thickness_px),
            )
            if str(case.counted_axis) in {HORIZONTAL_LINE_AXIS, VERTICAL_LINE_AXIS}:
                annotation_artifacts = segment_set_for_counted_elements(
                    counted_elements
                )
            else:
                annotation_artifacts = bbox_set_for_counted_elements(counted_elements)
            if len(annotation_artifacts.value) != int(case.answer_value):
                raise ValueError("annotation cardinality does not match answer")
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"{task_id} failed to generate counterfactual-board sample") from last_error

    prompt_defaults, prompt_artifacts = build_counterfactual_board_prompt_artifacts(
        domain=str(domain),
        prompt_query_key=str(case.prompt_query_key),
        dynamic_slots={},
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params={
            "scene_id": SCENE_ID,
            "query_id": str(selected_query_id),
            "prompt_query_key": str(case.prompt_query_key),
            "query_id_probabilities": {
                str(key): float(value)
                for key, value in dict(query_probabilities).items()
            },
            "answer_type": "integer",
            **json_ready(dict(case.case_params)),
        },
    )
    trace_payload = build_counterfactual_board_trace_payload(
        case=case,
        rendered=rendered,
        counted_elements=counted_elements,
        annotation_artifacts=annotation_artifacts,
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
        query_spec=query_spec,
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=TypedValue(type="integer", value=int(case.answer_value)),
        annotation_gt=annotation_artifacts.annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(selected_query_id),
    )


__all__ = [
    "CounterfactualBoardObjectivePlan",
    "run_counterfactual_board_lifecycle",
]
