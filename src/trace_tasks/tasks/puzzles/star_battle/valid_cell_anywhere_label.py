"""Choose a labeled Star Battle cell where a star can be placed anywhere."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import bind_valid_cell_label_output, run_star_battle_public_task
from .shared.sampling import build_valid_cell_dataset
from .shared.state import DOMAIN, SCENE_ID, StarBattleDataset


TASK_ID = "task_puzzles__star_battle__valid_cell_anywhere_label"
SUPPORTED_QUERY_IDS = ("valid_cell_anywhere_label",)
PROMPT_TASK_KEY = "star_battle_valid_cell_query"
SCOPE_KIND_BY_QUERY = {"valid_cell_anywhere_label": "whole_board"}
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


def _build_dataset_for_query(
    *,
    query_id: str,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
) -> StarBattleDataset:
    """Translate the task query into an unscoped valid-cell sampling program."""

    return build_valid_cell_dataset(
        scope_kind=str(SCOPE_KIND_BY_QUERY[str(query_id)]),
        params=params,
        instance_seed=int(instance_seed),
        generation_defaults=generation_defaults,
        namespace=str(namespace),
    )


def _bind_output(
    *,
    dataset: StarBattleDataset,
    context: Any,
    query_id: str,
    query_probabilities: Mapping[str, float],
) -> Any:
    """Bind scalar selected-cell annotation for the legal candidate label."""

    return bind_valid_cell_label_output(
        dataset=dataset,
        context=context,
        query_id=str(query_id),
        query_probabilities=query_probabilities,
        scope_kind=str(SCOPE_KIND_BY_QUERY[str(query_id)]),
    )


@register_task
class PuzzlesStarBattleValidCellAnywhereLabelTask:
    """Choose the one labeled cell where another Star Battle star can be placed."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
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
        """Generate one unscoped Star Battle legal-cell option task."""

        return run_star_battle_public_task(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            prompt_task_key=PROMPT_TASK_KEY,
            params=params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            dataset_builder=_build_dataset_for_query,
            output_binder=_bind_output,
        )


__all__ = [
    "PROMPT_TASK_KEY",
    "PuzzlesStarBattleValidCellAnywhereLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
