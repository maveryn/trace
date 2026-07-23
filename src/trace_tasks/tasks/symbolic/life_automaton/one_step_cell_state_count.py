"""Count future alive/dead cells after one Life update."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ....core.seed import spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import group_default, load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ..shared.common import get_int_range as _get_range
from ._lifecycle import life_query_params, render_life_task_scene
from .shared.annotations import annotation_trace_payload, bbox_set
from .shared.output import build_life_trace_payload, life_grid_execution_fields
from .shared.prompts import build_life_prompt
from .shared.rules import SCENE_ID, sample_life_grid, sample_square_grid_size, simulate_life
from .shared.state import LifeSceneSpec
from .shared.styles import resolve_scene_variant


DOMAIN = "symbolic"
TASK_ID = "task_symbolic__life_automaton__one_step_cell_state_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("one_step_alive_cell_count", "one_step_dead_cell_count")
QUESTION_FORMAT = "one_step_cell_state_count"
TASK_PROMPT_KEY = "life_one_step_cell_state_count_query"
_TARGET_STATE = {
    "one_step_alive_cell_count": (1, "alive"),
    "one_step_dead_cell_count": (0, "dead"),
}

_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class SymbolicLifeAutomatonOneStepCellStateCountTask:
    """Count cells that will be alive or dead after one Life update."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        """Build a source-only Life scene, then bind count and bbox_set to one-step future cells."""

        query_id, query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=SUPPORTED_QUERY_IDS[0],
            task_id=TASK_ID,
            namespace=f"{QUESTION_FORMAT}.query",
        )
        scene_variant, scene_variant_probabilities = resolve_scene_variant(
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            instance_seed=int(instance_seed),
            sampling_scope=QUESTION_FORMAT,
        )
        rng = spawn_rng(int(instance_seed), f"{QUESTION_FORMAT}.life")
        grid_size = _sample_grid_size(rng, task_params)
        density = float(task_params.get("live_density", group_default(_GEN_DEFAULTS, "live_density", 0.36)))
        initial_grid = sample_life_grid(rng, rows=grid_size, cols=grid_size, live_prob=density)
        future_grid = simulate_life(initial_grid, steps=1)
        target_value, target_name = _TARGET_STATE[str(query_id)]
        target_cells = tuple(
            (row, col)
            for row, values in enumerate(future_grid)
            for col, value in enumerate(values)
            if int(value) == int(target_value)
        )
        annotation_item_ids = tuple(f"source_cell_{row}_{col}" for row, col in target_cells)

        render_bundle = render_life_task_scene(
            scene=LifeSceneSpec(
                rows=int(grid_size),
                cols=int(grid_size),
                initial_grid=tuple(initial_grid),
                future_grid=tuple(future_grid),
                source_marker_label="START",
            ),
            params=task_params,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            scene_variant=str(scene_variant),
            instance_seed=int(instance_seed),
            sampling_scope=QUESTION_FORMAT,
        )
        annotation_value = bbox_set(render_bundle.rendered.item_bboxes, annotation_item_ids)
        witness_symbolic, projected_annotation = annotation_trace_payload(
            annotation_type="bbox_set",
            annotation_value=annotation_value,
        )
        prompt, prompt_variants, _prompt_meta, prompt_artifacts = build_life_prompt(
            domain=DOMAIN,
            prompt_defaults=_PROMPT_DEFAULTS,
            scene_variant=str(scene_variant),
            query_key=str(query_id),
            steps=1,
            instance_seed=int(instance_seed),
            task_key=TASK_PROMPT_KEY,
            query_instruction_key=f"query_instruction_{query_id}",
            annotation_hint_key="annotation_hint_one_step_cell_state_count",
            answer_hint_key="answer_hint_one_step_cell_state_count",
            json_example_key="json_example_one_step_cell_state_count",
            json_example_answer_only_key="json_example_answer_only_one_step_cell_state_count",
            object_description_prefix="object_description_one_step_cell_state_count",
        )
        query_params = life_query_params(
            query_id=str(query_id),
            query_probabilities=query_probabilities,
            question_format=QUESTION_FORMAT,
            scene_variant=str(scene_variant),
            scene_variant_probabilities=scene_variant_probabilities,
            render_bundle=render_bundle,
            extra={"target_state": str(target_name)},
        )
        execution_trace = {
            **dict(query_params),
            "task_id": TASK_ID,
            "answer_value": len(target_cells),
            **life_grid_execution_fields(
                rows=grid_size,
                cols=grid_size,
                steps=1,
                initial_grid=initial_grid,
                future_grid=future_grid,
            ),
            "target_state_value": int(target_value),
            "target_state_name": str(target_name),
            "target_cells_after_update": [[int(row), int(col)] for row, col in target_cells],
            "supporting_item_ids": list(annotation_item_ids),
        }
        return TaskOutput(
            prompt=prompt,
            answer_gt=TypedValue(type="integer", value=len(target_cells)),
            annotation_gt=TypedValue(type="bbox_set", value=list(annotation_value)),
            image=render_bundle.image,
            image_id="img0",
            trace_payload=build_life_trace_payload(
                scene_name=SCENE_ID,
                prompt_artifacts=prompt_artifacts,
                branch_name=str(query_id),
                params_payload=query_params,
                render_bundle=render_bundle,
                execution_record=execution_trace,
                witness_symbolic=witness_symbolic,
                projected_annotation=projected_annotation,
                render_map_extra={"annotation_item_ids": list(annotation_item_ids)},
            ),
            task_versions=default_task_versions(),
            scene_id=SCENE_ID,
            query_id=str(query_id),
            prompt_variants=prompt_variants,
        )


def _sample_grid_size(rng: Any, params: Dict[str, Any]) -> int:
    grid_min, grid_max = _get_range(params, _GEN_DEFAULTS, min_key="life_grid_size_min", max_key="life_grid_size_max", fallback_min=3, fallback_max=5)
    rows_min, rows_max = _get_range(params, _GEN_DEFAULTS, min_key="life_rows_min", max_key="life_rows_max", fallback_min=grid_min, fallback_max=grid_max)
    cols_min, cols_max = _get_range(params, _GEN_DEFAULTS, min_key="life_cols_min", max_key="life_cols_max", fallback_min=grid_min, fallback_max=grid_max)
    return sample_square_grid_size(
        rng,
        grid_size_min=grid_min,
        grid_size_max=grid_max,
        rows_min=rows_min,
        rows_max=rows_max,
        cols_min=cols_min,
        cols_max=cols_max,
    )
