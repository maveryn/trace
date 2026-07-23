"""Count the maximum complete unit squares after adding multiple matchsticks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BoundMatchstickOutput,
    MatchstickRenderContext,
    build_bound_output,
    run_matchstick_public_task,
)
from .shared.annotations import bbox_set_artifacts
from .shared.rendering import render_square_lattice_scene
from .shared.rules import lattice_square_item_id, optimal_lattice_square_additions
from .shared.sampling import build_square_completion_dataset
from .shared.state import DOMAIN, SCENE_ID, SquareCompletionDataset


TASK_ID = "task_puzzles__matchstick__max_square_count_after_additions_value"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
TASK_PROMPT_KEY = "max_square_count_after_additions_value_query"
PROMPT_QUERY_KEY = "max_square_count_after_additions"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _build_dataset_for_query(
    *,
    query_id: str,
    scene_variant: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> SquareCompletionDataset:
    """Build the single-query square-completion lattice instance."""

    if str(query_id) != SINGLE_QUERY_ID:
        raise ValueError(f"unsupported matchstick square-completion query: {query_id}")
    return build_square_completion_dataset(
        scene_variant=str(scene_variant),
        params=params,
        generation_defaults=generation_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
    )


def _counted_square_item_ids(dataset: SquareCompletionDataset) -> list[str]:
    """Return render item ids for the final complete squares being counted."""

    return [
        lattice_square_item_id(str(square_id))
        for square_id in dataset.completed_square_ids
    ]


def _validate_unique_optimal_square_result(dataset: SquareCompletionDataset) -> None:
    """Assert the sampled lattice has one final counted-square set."""

    optimal = optimal_lattice_square_additions(
        frozenset(str(edge) for edge in dataset.present_edges),
        rows=int(dataset.rows),
        cols=int(dataset.cols),
        add_count=int(dataset.add_count),
    )
    best_square_sets = tuple(optimal["best_square_sets"])  # type: ignore[arg-type]
    if int(optimal["best_count"]) != int(dataset.answer_value):
        raise RuntimeError("square-completion answer is not the optimal count")
    if len(best_square_sets) != 1:
        raise RuntimeError("square-completion final annotation is ambiguous")
    if tuple(str(square) for square in best_square_sets[0]) != tuple(
        str(square) for square in dataset.completed_square_ids
    ):
        raise RuntimeError("square-completion counted squares do not match optimum")


def _scene_extra(dataset: SquareCompletionDataset) -> dict[str, int]:
    """Build task-owned scene relation fields for the square lattice."""

    return {
        "rows": int(dataset.rows),
        "cols": int(dataset.cols),
        "add_count": int(dataset.add_count),
        "answer_value": int(dataset.answer_value),
    }


def _execution_extra(
    dataset: SquareCompletionDataset,
    counted_item_ids: list[str],
) -> dict[str, object]:
    """Build task-owned execution trace fields for this objective."""

    return {
        "rows": int(dataset.rows),
        "cols": int(dataset.cols),
        "add_count": int(dataset.add_count),
        "answer_value": int(dataset.answer_value),
        "present_edges": list(dataset.present_edges),
        "missing_edges": list(dataset.missing_edges),
        "initial_completed_square_ids": list(dataset.initial_completed_square_ids),
        "completed_square_ids": list(dataset.completed_square_ids),
        "optimal_added_edges": list(dataset.optimal_added_edges),
        "optimal_added_edge_sets": [
            list(edge_set) for edge_set in dataset.optimal_added_edge_sets
        ],
        "supporting_item_ids": list(counted_item_ids),
    }


def _bind_output(
    *,
    dataset: SquareCompletionDataset,
    context: MatchstickRenderContext,
    query_id: str,
    query_probabilities: Mapping[str, float],
    scene_variant_probabilities: Mapping[str, float],
) -> BoundMatchstickOutput:
    """Bind final completed unit-square bboxes as the count annotation."""

    _validate_unique_optimal_square_result(dataset)
    counted_item_ids = _counted_square_item_ids(dataset)
    annotation = bbox_set_artifacts(
        context.rendered_scene.item_bbox_map,
        counted_item_ids,
    )
    return build_bound_output(
        dataset=dataset,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        scene_variant_probabilities=scene_variant_probabilities,
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_dynamic_slots={"stick_count": str(dataset.add_count)},
        answer_gt=TypedValue(type="integer", value=int(dataset.answer_value)),
        annotation_artifacts=annotation,
        annotation_source="unit_square_bboxes_px",
        scene_extra=_scene_extra(dataset),
        execution_extra=_execution_extra(dataset, counted_item_ids),
        witness_symbolic={
            "type": "bbox_set",
            "value": [list(bbox) for bbox in annotation.value],
            "completed_square_ids": list(dataset.completed_square_ids),
        },
    )


@register_task
class PuzzlesMatchstickMaxSquareCountAfterAdditionsValueTask:
    """Count final unit squares under an optimal multi-stick addition."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'state_update')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Run square-completion callbacks through shared scene plumbing."""

        return run_matchstick_public_task(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            task_prompt_key=TASK_PROMPT_KEY,
            params=params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            dataset_builder=_build_dataset_for_query,
            render_scene=render_square_lattice_scene,
            output_binder=_bind_output,
        )


__all__ = [
    "PuzzlesMatchstickMaxSquareCountAfterAdditionsValueTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
