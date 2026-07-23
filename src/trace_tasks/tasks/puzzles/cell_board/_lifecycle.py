"""Scene-private lifecycle for cell-board public task files."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID, select_task_query_id
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec
from trace_tasks.tasks.shared.visual_defaults import (
    default_noise_fallback,
    load_scene_noise_defaults,
)

from .shared.annotations import (
    coord_bbox_set_annotation,
    coord_pair_segment_set_annotation,
    coord_path_segment_set_annotation,
)
from .shared.output import (
    build_cell_board_trace_payload,
    json_ready,
)
from .shared.prompts import build_cell_board_prompt_artifacts
from .shared.rendering import render_cell_board
from .shared.state import CellBoardCase, SCENE_ID

CaseBuilder = Callable[[int], CellBoardCase]


@dataclass(frozen=True)
class CellBoardObjectivePlan:
    """Task-owned case builder consumed by neutral scene lifecycle plumbing."""

    prompt_query_key: str
    construct_case: CaseBuilder
    query_params: Mapping[str, Any]


def _annotation_for_case(*, case: CellBoardCase, bbox_map):
    """Project the case's declared board witnesses into public annotation."""

    if str(case.annotation_kind) == "bbox_set":
        return coord_bbox_set_annotation(
            coords=case.annotation_coords,
            bbox_map=bbox_map,
        )
    if str(case.annotation_kind) == "segment_set":
        return coord_path_segment_set_annotation(
            path=case.annotation_path,
            bbox_map=bbox_map,
        )
    if str(case.annotation_kind) == "cell_pair_segment_set":
        return coord_pair_segment_set_annotation(
            coord_pairs=case.annotation_coord_pairs,
            bbox_map=bbox_map,
        )
    raise ValueError(f"unsupported cell-board annotation kind: {case.annotation_kind}")


def run_cell_board_lifecycle(
    *,
    task_id: str,
    domain: str,
    selected_query_id: str,
    query_probabilities: Mapping[str, float],
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    objective: CellBoardObjectivePlan,
) -> TaskOutput:
    """Run rendering, prompt assembly, annotation projection, and output wiring."""

    visual_defaults = load_scene_noise_defaults(
        domain=str(domain),
        scene_id=SCENE_ID,
        fallback=default_noise_fallback(apply_prob=0.18),
        merge_with_fallback=True,
    )
    last_error: Exception | None = None
    for attempt_index in range(max(1, int(max_attempts))):
        attempt_seed = int(instance_seed) + int(attempt_index)
        try:
            case = objective.construct_case(attempt_seed)
            rendered = render_cell_board(
                rows=int(case.rows),
                cols=int(case.cols),
                board_colors=case.board_colors,
                instance_seed=int(attempt_seed),
                params=params,
                rendering_defaults=render_defaults,
                visual_defaults=visual_defaults,
                coordinate_labels=bool(case.coordinate_labels),
                cell_text=case.cell_text,
            )
            annotation_artifacts = _annotation_for_case(
                case=case,
                bbox_map=rendered.bbox_map,
            )
            break
        except ValueError as exc:
            last_error = exc
    else:
        raise RuntimeError(f"{task_id} failed to generate cell-board sample") from last_error

    prompt_defaults, prompt_artifacts = build_cell_board_prompt_artifacts(
        domain=str(domain),
        prompt_task_key=str(case.prompt_task_key),
        prompt_query_key=str(case.prompt_query_key),
        dynamic_slots=case.prompt_slots,
        instance_seed=int(instance_seed),
    )
    query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=str(selected_query_id),
        params={
            "scene_id": SCENE_ID,
            "query_id": str(selected_query_id),
            "prompt_task_key": str(case.prompt_task_key),
            "prompt_query_key": str(case.prompt_query_key),
            "query_id_probabilities": {
                str(key): float(value)
                for key, value in dict(query_probabilities).items()
            },
            "answer_type": "integer",
            **json_ready(dict(objective.query_params)),
        },
    )
    trace_payload = build_cell_board_trace_payload(
        case=case,
        rendered=rendered,
        annotation_artifacts=annotation_artifacts,
        prompt_defaults=prompt_defaults,
        prompt_artifacts=prompt_artifacts,
        query_spec=query_spec,
        execution_extra={"answer": int(case.answer_value)},
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


def run_single_query_cell_board_task(
    *,
    task_id: str,
    domain: str,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    max_attempts: int,
    prompt_query_key: str,
    construct_case: CaseBuilder,
    query_params: Mapping[str, Any] | None = None,
    namespace: str,
) -> TaskOutput:
    """Run the common fixed-query shell around a task-owned case builder."""

    selected_query, branch_probs, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=(DEFAULT_QUERY_ID,),
        default_query_id=DEFAULT_QUERY_ID,
        task_id=str(task_id),
        namespace=str(namespace),
    )
    return run_cell_board_lifecycle(
        task_id=str(task_id),
        domain=str(domain),
        selected_query_id=str(selected_query),
        query_probabilities=branch_probs,
        params=task_params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        objective=CellBoardObjectivePlan(
            prompt_query_key=str(prompt_query_key),
            construct_case=construct_case,
            query_params=dict(query_params or {"prompt_query_key": str(prompt_query_key)}),
        ),
    )


__all__ = [
    "CellBoardObjectivePlan",
    "run_cell_board_lifecycle",
    "run_single_query_cell_board_task",
]
