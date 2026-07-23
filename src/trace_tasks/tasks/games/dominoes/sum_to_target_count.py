"""Count face-up dominoes with a target pip sum."""

from __future__ import annotations

from typing import List

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.support_sampling import resolve_integer_support

from ._lifecycle import DominoObjectivePlan, domino_bbox_set_attempt, resolve_domino_count_axes, run_domino_lifecycle
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.prompts import domino_integer_json_examples, domino_output_slots
from .shared.rules import CANONICAL_DOMINOES, tile_sum
from .shared.sampling import build_sampled_scene, build_tableau_instances
from .shared.state import DominoSceneAxes


TASK_ID = "task_games__dominoes__sum_to_target_count"
QUERY_ID = "sum_to_target_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _sample_sum_to_target_scene(
    rng,
    *,
    candidate_count: int,
    target_answer: int,
    target_total_support: tuple[int, ...],
    explicit_target_total: int | None,
):
    """Construct a scene where exactly the target candidates match a pip total."""

    for _ in range(320):
        candidate_pool = tuple(CANONICAL_DOMINOES)
        feasible_totals: List[int] = []
        for total in target_total_support:
            exact_pool = [tile for tile in candidate_pool if tile_sum(tile) == int(total)]
            filler_pool = [tile for tile in candidate_pool if tile_sum(tile) != int(total)]
            if int(target_answer) == 0:
                if int(len(filler_pool)) >= int(candidate_count):
                    feasible_totals.append(int(total))
            elif int(len(exact_pool)) >= int(target_answer) and int(len(filler_pool)) >= int(candidate_count - target_answer):
                feasible_totals.append(int(total))
        if explicit_target_total is not None:
            target_total = int(explicit_target_total)
            if int(target_total) not in set(feasible_totals):
                raise ValueError(f"unsupported target total: {target_total}")
        else:
            if not feasible_totals:
                continue
            target_total = int(feasible_totals[int(rng.randrange(len(feasible_totals)))])

        exact_pool = [tile for tile in candidate_pool if tile_sum(tile) == int(target_total)]
        filler_pool = [tile for tile in candidate_pool if tile_sum(tile) != int(target_total)]
        annotation_tiles = [] if int(target_answer) == 0 else list(rng.sample(exact_pool, int(target_answer)))
        selected_candidates = list(annotation_tiles) + list(rng.sample(filler_pool, int(candidate_count - target_answer)))
        candidate_instances, annotation_tile_ids, reference_tile_id = build_tableau_instances(
            rng=rng,
            candidate_tiles=selected_candidates,
            annotation_tiles=annotation_tiles,
        )
        return build_sampled_scene(
            chain_instances=(),
            candidate_instances=candidate_instances,
            annotation_tile_ids=annotation_tile_ids,
            answer_value=int(target_answer),
            reference_tile_id=reference_tile_id,
            target_total=int(target_total),
        )
    raise ValueError("unable to sample target-sum domino scene")


def _prepare_sum_to_target_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
    axes: DominoSceneAxes,
):
    """Resolve target count/total axes and bind pip-sum equality semantics."""

    count_axes = resolve_domino_count_axes(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        axes=axes,
        target_support_key="sum_to_target_answer_support",
        target_fallback_support=DEFAULTS.sum_to_target_answer_support,
        target_namespace="sum_to_target.target_answer",
        minimum_candidate_count=lambda target: 7 if int(target) == 0 else max(7, int(target)),
        candidate_namespace="sum_to_target",
    )
    target_total_support = resolve_integer_support(
        task_params,
        gen_defaults=_GEN_DEFAULTS,
        key="sum_target_total_support",
        fallback=DEFAULTS.sum_target_total_support,
    )
    explicit_target_total = None if task_params.get("target_total") is None else int(task_params["target_total"])
    json_example, json_example_answer_only = domino_integer_json_examples(answer_value=3)
    prompt_dynamic_slots = {
        **domino_output_slots(
            prompt_query_key=QUERY_ID,
            json_example=json_example,
            json_example_answer_only=json_example_answer_only,
        ),
    }
    query_params = {
        **dict(count_axes.query_params),
        "target_total_support": [int(value) for value in target_total_support],
        "target_total": explicit_target_total,
    }

    def construct_attempt(rng, _axes: DominoSceneAxes):
        sample = _sample_sum_to_target_scene(
            rng,
            candidate_count=int(count_axes.candidate_axis.value),
            target_answer=int(count_axes.target_axis.value),
            target_total_support=tuple(int(value) for value in target_total_support),
            explicit_target_total=explicit_target_total,
        )
        return domino_bbox_set_attempt(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(sample.answer_value)),
            query_params={"target_total": int(sample.target_total) if sample.target_total is not None else None},
            execution_extra={"target_answer": int(sample.answer_value)},
            prompt_dynamic_slots={"target_total_text": str(sample.target_total)},
        )

    return DominoObjectivePlan(
        attempt_namespace="games.dominoes.sum_to_target",
        prompt_query_key=QUERY_ID,
        query_params=query_params,
        prompt_dynamic_slots=prompt_dynamic_slots,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesDominoesSumToTargetCountTask:
    """Count face-up dominoes whose pip sum equals the sampled target total."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'aggregation')
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
            prepare_objective=_prepare_sum_to_target_objective,
        )


__all__ = ["GamesDominoesSumToTargetCountTask"]
