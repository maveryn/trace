"""Public maze task for selecting one reachable or unreachable exit."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    build_maze_task_output,
    prepare_maze_visual_case,
    resolve_maze_public_branch,
    resolve_maze_scene_variant,
    retry_maze_generation,
)
from .shared.annotations import single_item_point
from .shared.sampling import sample_exit_label_maze
from .shared.state import (
    DOMAIN,
    SCENE_ID,
    TARGET_REACHABILITY_DESCRIPTIONS,
    TARGET_REACHABILITY_VALUES,
)

TASK_ID = "task_puzzles__maze__exit_reachability_label"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_TASK_KEY = "exit_reachability_label_query"
PROMPT_QUERY_KEY = "exit_reachability_label"
_NAMESPACE_BASE = f"{DOMAIN}.{SCENE_ID}.exit_reachability_label"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(DOMAIN, SCENE_ID, task_id=TASK_ID)
)


@register_task
class PuzzlesMazeExitReachabilityLabelTask:
    """Return the only exit matching a sampled reachability target."""

    task_id = TASK_ID
    reasoning_operations = ('topology',)
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
        """Generate one maze exit-label task with scalar point annotation."""

        return retry_maze_generation(
            build_case=_build_exit_reachability_label_case,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=max_attempts,
        )


def _build_exit_reachability_label_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Resolve target reachability, sample a maze, and bind one exit label."""

    selected_branch, branch_probabilities, task_params = resolve_maze_public_branch(
        instance_seed=int(instance_seed),
        params=params,
        task_id=TASK_ID,
        namespace=_NAMESPACE_BASE,
    )
    target_reachability, target_reachability_probabilities = _resolve_target_reachability(
        params=task_params,
        instance_seed=int(instance_seed),
    )
    scene_variant, scene_variant_probabilities = resolve_maze_scene_variant(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=_NAMESPACE_BASE,
    )
    dataset = sample_exit_label_maze(
        target_reachability=str(target_reachability),
        scene_variant=str(scene_variant),
        params=task_params,
        instance_seed=int(instance_seed),
        generation_defaults=_GEN_DEFAULTS,
        max_attempts=int(max_attempts),
    )
    visual = prepare_maze_visual_case(
        dataset=dataset,
        params=task_params,
        generation_defaults=_GEN_DEFAULTS,
        rendering_defaults=_RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        scene_variant=str(scene_variant),
        prompt_task_key=PROMPT_TASK_KEY,
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_dynamic_slots={
            "target_reachability_description": TARGET_REACHABILITY_DESCRIPTIONS[
                str(target_reachability)
            ],
        },
        namespace=_NAMESPACE_BASE,
    )
    supporting_item_ids = [str(value) for value in dataset["supporting_item_ids"]]
    if len(supporting_item_ids) != 1:
        raise ValueError("exit reachability label task must bind exactly one exit")
    annotation_gt, projected_annotation, witness_symbolic = single_item_point(
        visual["rendered_scene"].item_point_map,
        supporting_item_ids[0],
    )
    answer_gt = TypedValue(type="string", value=str(dataset["answer_value"]))
    _validate_label_answer(dataset, str(target_reachability), str(answer_gt.value))
    return build_maze_task_output(
        dataset=dataset,
        visual=visual,
        public_query_id=str(selected_branch),
        branch_probabilities=branch_probabilities,
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        projected_annotation=projected_annotation,
        witness_symbolic=witness_symbolic,
        prompt_query_key=PROMPT_QUERY_KEY,
        semantic_params={
            "target_reachability_probabilities": dict(target_reachability_probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
        relation_fields={
            "target_reachability_description": TARGET_REACHABILITY_DESCRIPTIONS[
                str(target_reachability)
            ],
        },
        execution_fields={
            "target_reachability_probabilities": dict(target_reachability_probabilities),
            "scene_variant_probabilities": dict(scene_variant_probabilities),
        },
    )


def _resolve_target_reachability(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[str, dict[str, float]]:
    """Sample reachable or unreachable as a semantic operand, not a public query."""

    explicit = params.get("target_reachability")
    if explicit is not None:
        selected = str(explicit).strip().lower()
        if selected not in set(TARGET_REACHABILITY_VALUES):
            raise ValueError(f"unsupported target_reachability: {explicit}")
        return (
            str(selected),
            support_probability_map(TARGET_REACHABILITY_VALUES, selected=selected, sort_keys=True),
        )
    rng = spawn_rng(int(instance_seed), f"{_NAMESPACE_BASE}.target_reachability")
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        TARGET_REACHABILITY_VALUES,
        sort_keys=True,
    )
    return str(selected), dict(probabilities)


def _validate_label_answer(
    dataset: Mapping[str, Any],
    target_reachability: str,
    answer_value: str,
) -> None:
    """Check the selected label belongs to the requested reachable/unreachable set."""

    if str(target_reachability) == "reachable":
        valid = {str(value) for value in dataset["reachable_exit_labels"]}
    else:
        valid = {str(value) for value in dataset["unreachable_exit_labels"]}
    if str(answer_value) not in valid:
        raise ValueError("maze exit-label answer drifted from sampled reachability set")


__all__ = [
    "PROMPT_QUERY_KEY",
    "PuzzlesMazeExitReachabilityLabelTask",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
