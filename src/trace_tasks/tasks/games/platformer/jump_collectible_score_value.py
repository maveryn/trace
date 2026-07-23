"""Sum collectible scores along a shown platformer jump path."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import AttemptPlatformerResult, ObjectivePlatformerPlan, run_platformer_lifecycle
from .shared.annotations import point_set_for_entity_ids
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.sampling import (
    PlatformerVisualAxes,
    integer_support_from_defaults,
    sample_scored_collectible_path_scene,
)
from .shared.state import PlatformerSample


TASK_ID = "task_games__platformer__jump_collectible_score_value"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "jump_collectible_score_value"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


@register_task
class GamesPlatformerJumpCollectibleScoreValueTask:
    """Sum collectible scores along the shown jump arc."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_platformer_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_score_objective,
        )


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for platformer score output."""

    return (
        json.dumps(
            {"annotation": [[346, 228], [448, 200], [554, 216]], "answer": 13},
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        json.dumps({"answer": 13}, separators=(",", ":"), ensure_ascii=False),
    )


def _validate_score_sample(sample: PlatformerSample) -> None:
    """Verify the scored path collectibles sum to the integer answer."""

    target_ids = tuple(str(entity_id) for entity_id in sample.target_collectible_ids)
    collectible_by_id = {str(coin.collectible_id): coin for coin in sample.collectibles}
    if not target_ids:
        raise ValueError("score sample requires target collectible ids")
    score = 0
    saw_coin = False
    saw_bonus = False
    for collectible_id in target_ids:
        collectible = collectible_by_id.get(str(collectible_id))
        if collectible is None:
            raise ValueError("score target references a missing collectible")
        if not bool(collectible.on_path):
            raise ValueError("score target must be on the path")
        if collectible.score_value is None:
            score += 1
            saw_coin = True
        else:
            score += int(collectible.score_value)
            saw_bonus = True
    if not saw_coin or not saw_bonus:
        raise ValueError("score sample requires at least one coin and one bonus item")
    if int(sample.answer) != int(score):
        raise ValueError("score sample answer does not match target collectibles")
    if tuple(sample.annotation_entity_ids) != target_ids:
        raise ValueError("score annotation ids must match target collectibles")


def _prepare_score_objective(
    _instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    axes: PlatformerVisualAxes,
) -> ObjectivePlatformerPlan:
    """Resolve score supports and bind the scored-path constructor."""

    on_arc_coin_support = integer_support_from_defaults(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="score_on_arc_coin_count_support",
        fallback=DEFAULTS.score_on_arc_coin_count_support,
    )
    on_arc_bonus_support = integer_support_from_defaults(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="score_on_arc_bonus_count_support",
        fallback=DEFAULTS.score_on_arc_bonus_count_support,
    )
    off_arc_bonus_support = integer_support_from_defaults(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="score_off_arc_bonus_count_support",
        fallback=DEFAULTS.score_off_arc_bonus_count_support,
    )
    bonus_value_support = integer_support_from_defaults(
        params,
        gen_defaults=_GEN_DEFAULTS,
        key="score_bonus_value_support",
        fallback=DEFAULTS.score_bonus_value_support,
    )

    def construct_attempt(rng: Any, attempt_axes: PlatformerVisualAxes) -> AttemptPlatformerResult:
        sample = sample_scored_collectible_path_scene(
            rng=rng,
            axes=attempt_axes,
            on_arc_coin_count_support=on_arc_coin_support,
            on_arc_bonus_count_support=on_arc_bonus_support,
            off_arc_bonus_count_support=off_arc_bonus_support,
            bonus_value_support=bonus_value_support,
            mode=PROMPT_QUERY_KEY,
        )
        _validate_score_sample(sample)
        annotation_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
        return AttemptPlatformerResult(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(sample.answer)),
            annotation_entity_ids=annotation_ids,
            build_annotation=lambda rendered: point_set_for_entity_ids(rendered.rendered_scene, annotation_ids),
            witness_type="object_set",
            relations_extra={"target_collectible_ids": list(annotation_ids)},
            execution_extra={"target_collectible_ids": list(annotation_ids)},
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePlatformerPlan(
        attempt_namespace="games.platformer.path_score_total",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint='set "answer" to the integer total score from collectibles on the dashed jump arc',
        annotation_hint='set "annotation" to [x, y] pixel points at the centers of each coin or printed-value bonus item included in the dashed-arc score',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "score_on_arc_coin_count_support": [int(value) for value in on_arc_coin_support],
            "score_on_arc_bonus_count_support": [int(value) for value in on_arc_bonus_support],
            "score_off_arc_bonus_count_support": [int(value) for value in off_arc_bonus_support],
            "score_bonus_value_support": [int(value) for value in bonus_value_support],
        },
        construct_attempt=construct_attempt,
    )

__all__ = ["GamesPlatformerJumpCollectibleScoreValueTask"]
