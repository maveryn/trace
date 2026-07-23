"""Select the composite shape that can be formed from two source pieces."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import build_composition_dataset_payload
from ._lifecycle import run_polyomino_assembly_public_task
from .shared.state import DOMAIN, SCENE_ID


TASK_ID = "task_puzzles__polyomino_assembly__composition_result_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "composition_result_label_query"
PROMPT_QUERY_KEY = "composition_result_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.composition_result_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesPolyominoAssemblyCompositionResultLabelTask:
    """Choose which option shape can be composed from the two source pieces."""

    task_id = TASK_ID
    reasoning_operations = ('transformation', 'matching')
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
        """Generate one source-pieces-to-composite-shape puzzle."""

        return run_polyomino_assembly_public_task(
            instance_seed=int(instance_seed),
            params=params,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            task_id=TASK_ID,
            namespace_base=_NAMESPACE_BASE,
            prompt_task_key=PROMPT_TASK_KEY,
            prompt_query_key=PROMPT_QUERY_KEY,
            dataset_builder=_build_composition_dataset,
            max_attempts=int(max_attempts),
            question_format="composition_result_label",
            view_family="polyomino_piece_pair_with_result_options",
        )


def _build_composition_dataset(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_variant: str,
    total_cell_count: int,
    total_cell_count_range: tuple[int, int],
    total_cell_count_probabilities: Mapping[str, float],
    answer_label: str,
    answer_label_probabilities: Mapping[str, float],
) -> dict[str, Any]:
    """Build source pieces and four candidate composite shapes."""

    return build_composition_dataset_payload(
        instance_seed=int(instance_seed),
        params=params,
        generation_defaults=_GEN_DEFAULTS,
        namespace_base=_NAMESPACE_BASE,
        scene_variant=str(scene_variant),
        total_cell_count=int(total_cell_count),
        total_cell_count_range=total_cell_count_range,
        total_cell_count_probabilities=total_cell_count_probabilities,
        answer_label=str(answer_label),
        answer_label_probabilities=answer_label_probabilities,
    )


__all__ = [
    "PuzzlesPolyominoAssemblyCompositionResultLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
