"""Count face-up dominoes above a marked reference pip sum."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import DominoObjectivePlan, domino_bbox_set_attempt, resolve_domino_count_axes, run_domino_lifecycle
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.prompts import domino_integer_json_examples, domino_output_slots
from .shared.rules import CANONICAL_DOMINOES, tile_sum
from .shared.sampling import build_sampled_scene, build_tableau_instances
from .shared.state import DominoSceneAxes


TASK_ID = "task_games__dominoes__higher_sum_than_reference_count"
QUERY_ID = "higher_sum_than_reference_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _sample_higher_sum_scene(rng, *, candidate_count: int, target_answer: int):
    """Construct a tableau with one marked reference and larger-sum witnesses."""

    feasible_references = [tile for tile in CANONICAL_DOMINOES if 6 <= tile_sum(tile) <= 8]
    for _ in range(320):
        reference_tile = feasible_references[int(rng.randrange(len(feasible_references)))]
        reference_sum = int(tile_sum(reference_tile))
        candidate_pool = [tile for tile in CANONICAL_DOMINOES if tile != reference_tile]
        annotation_pool = [tile for tile in candidate_pool if int(tile_sum(tile)) > int(reference_sum)]
        filler_pool = [tile for tile in candidate_pool if int(tile_sum(tile)) <= int(reference_sum)]
        if int(len(annotation_pool)) < int(target_answer):
            continue
        if int(len(filler_pool)) < int(candidate_count - target_answer):
            continue

        annotation_tiles = [] if int(target_answer) == 0 else list(rng.sample(annotation_pool, int(target_answer)))
        selected_candidates = annotation_tiles + list(rng.sample(filler_pool, int(candidate_count - target_answer)))
        candidate_instances, annotation_tile_ids, reference_tile_id = build_tableau_instances(
            rng=rng,
            reference_tile=reference_tile,
            reference_role="reference_sum",
            candidate_tiles=selected_candidates,
            annotation_tiles=annotation_tiles,
        )
        return build_sampled_scene(
            chain_instances=(),
            candidate_instances=candidate_instances,
            annotation_tile_ids=annotation_tile_ids,
            answer_value=int(target_answer),
            reference_tile_id=reference_tile_id,
            reference_sum=int(reference_sum),
        )
    raise ValueError("unable to sample higher-sum domino tableau")


def _prepare_higher_sum_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    axes: DominoSceneAxes,
):
    """Resolve count axes and bind larger-than-reference semantics."""

    count_axes = resolve_domino_count_axes(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        axes=axes,
        target_support_key="higher_sum_target_answer_support",
        target_fallback_support=DEFAULTS.higher_sum_target_answer_support,
        target_namespace="higher_sum.target_answer",
        minimum_candidate_count=lambda target: max(7, int(target)),
        candidate_namespace="higher_sum",
    )
    json_example, json_example_answer_only = domino_integer_json_examples(answer_value=3)
    prompt_dynamic_slots = domino_output_slots(
        prompt_query_key=QUERY_ID,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
    )

    def construct_attempt(rng, _axes: DominoSceneAxes):
        sample = _sample_higher_sum_scene(
            rng,
            candidate_count=int(count_axes.candidate_axis.value),
            target_answer=int(count_axes.target_axis.value),
        )
        return domino_bbox_set_attempt(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(sample.answer_value)),
            execution_extra={"target_answer": int(sample.answer_value)},
        )

    return DominoObjectivePlan(
        attempt_namespace="games.dominoes.higher_sum",
        prompt_query_key=QUERY_ID,
        query_params=dict(count_axes.query_params),
        prompt_dynamic_slots=prompt_dynamic_slots,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesDominoesHigherSumThanReferenceCountTask:
    """Count face-up dominoes with a larger pip sum than the reference tile."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'aggregation')
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
            prepare_objective=_prepare_higher_sum_objective,
        )


__all__ = ["GamesDominoesHigherSumThanReferenceCountTask"]
