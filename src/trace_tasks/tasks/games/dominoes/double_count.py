"""Count face-up dominoes that are doubles."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import prepare_domino_count_objective, run_domino_lifecycle
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.rules import CANONICAL_DOMINOES, canonical_tile
from .shared.sampling import build_sampled_scene, build_tableau_instances
from .shared.state import DominoSceneAxes


TASK_ID = "task_games__dominoes__double_count"
QUERY_ID = "double_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _sample_double_scene(rng, *, candidate_count: int, target_answer: int):
    """Construct a tableau with exactly the target number of doubles."""

    double_pool = [tile for tile in CANONICAL_DOMINOES if int(tile[0]) == int(tile[1])]
    filler_pool = [tile for tile in CANONICAL_DOMINOES if int(tile[0]) != int(tile[1])]
    if int(len(double_pool)) < int(target_answer):
        raise ValueError("insufficient double dominoes")
    if int(len(filler_pool)) < int(candidate_count - target_answer):
        raise ValueError("insufficient non-double dominoes")
    annotation_tiles = list(rng.sample(double_pool, int(target_answer)))
    selected_candidates = annotation_tiles + list(rng.sample(filler_pool, int(candidate_count - target_answer)))
    candidate_instances, annotation_tile_ids, reference_tile_id = build_tableau_instances(
        rng=rng,
        candidate_tiles=[canonical_tile(int(tile[0]), int(tile[1])) for tile in selected_candidates],
        annotation_tiles=annotation_tiles,
    )
    return build_sampled_scene(
        chain_instances=(),
        candidate_instances=candidate_instances,
        annotation_tile_ids=annotation_tile_ids,
        answer_value=int(target_answer),
        reference_tile_id=reference_tile_id,
    )


def _prepare_double_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    axes: DominoSceneAxes,
):
    """Resolve count axes and bind loose-double semantics."""

    return prepare_domino_count_objective(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        axes=axes,
        prompt_query_key=QUERY_ID,
        attempt_namespace="games.dominoes.double",
        target_support_key="double_target_answer_support",
        target_fallback_support=DEFAULTS.double_target_answer_support,
        target_namespace="double.target_answer",
        minimum_candidate_count=lambda target: max(7, int(target)),
        candidate_namespace="double",
        sample_scene=_sample_double_scene,
        example_answer=2,
    )


@register_task
class GamesDominoesDoubleCountTask:
    """Count face-up dominoes whose two halves match."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_domino_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            prepare_objective=_prepare_double_objective,
        )


__all__ = ["GamesDominoesDoubleCountTask"]
