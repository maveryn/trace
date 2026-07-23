"""Public cyclic-order swap-repair option task."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import CyclicOrderTaskObjective, run_cyclic_order_objective
from .shared.rendering import render_swap_repair_scene
from .shared.sampling import build_swap_repair_dataset
from .shared.state import DOMAIN, SCENE_ID


TASK_ID = "task_puzzles__cyclic_order__swap_repair_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "swap_repair_label_query"
PROMPT_QUERY_KEY = "swap_repair"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.swap_repair"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)
_OBJECTIVE = CyclicOrderTaskObjective(
    prompt_task_key=PROMPT_TASK_KEY,
    prompt_query_key=PROMPT_QUERY_KEY,
    namespace_base=_NAMESPACE_BASE,
    generation_defaults=_GEN_DEFAULTS,
    rendering_defaults=_RENDER_DEFAULTS,
    dataset_builder=build_swap_repair_dataset,
    scene_renderer=render_swap_repair_scene,
    render_field_map={
        "reference_token_specs": "reference_bead_specs",
        "reference_loop_shape_variant": "reference_loop_shape_variant",
        "reference_loop_path_style": "reference_loop_path_style",
        "reference_start_angle_deg": "reference_start_angle_deg",
        "broken_token_specs": "broken_bead_specs",
        "broken_loop_shape_variant": "broken_loop_shape_variant",
        "broken_loop_path_style": "broken_loop_path_style",
        "broken_start_angle_deg": "broken_start_angle_deg",
        "option_specs": "option_specs",
    },
    trace_field_keys=(
        "reference_token_sequence",
        "solved_token_sequence",
        "broken_token_sequence",
        "reference_loop_shape_variant",
        "broken_loop_shape_variant",
        "reference_loop_path_style",
        "broken_loop_path_style",
    ),
    description_by_variant={
        "necklace_board": "a reference token loop, a broken numbered loop, and six swap options",
        "charm_card_grid": "a reference charm loop, a broken numbered loop, and six swap options",
        "route_loop_diagram": "a reference route loop, a broken numbered loop, and six swap options",
        "token_ring_outline": "a reference token ring, a broken numbered ring, and six swap options",
        "default": "a reference loop, a broken numbered loop, and six swap options",
    },
    option_sequence_key="repaired_token_sequence",
)


@register_task
class PuzzlesCyclicOrderSwapRepairLabelTask:
    """Choose the swap that repairs a broken cyclic order."""

    task_id = TASK_ID
    reasoning_operations = ('topology', 'state_update', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def _build_objective(self) -> CyclicOrderTaskObjective:
        """Return this task's objective-owned lifecycle spec."""

        return _OBJECTIVE

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one swap-repair cyclic-order puzzle."""

        objective = self._build_objective()
        return run_cyclic_order_objective(
            task_identity=TASK_ID,
            objective=objective,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = ["PuzzlesCyclicOrderSwapRepairLabelTask", "SUPPORTED_QUERY_IDS", "TASK_ID"]
