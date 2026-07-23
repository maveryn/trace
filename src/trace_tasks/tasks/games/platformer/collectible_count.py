"""Count collectibles on the shown platformer jump path."""

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
    resolve_platformer_integer_axis,
    sample_collectible_path_scene,
)
from .shared.state import PlatformerSample


TASK_ID = "task_games__platformer__collectible_count"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "collectible_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for platformer collectible-count output."""

    return (
        json.dumps(
            {"annotation": [[346, 228], [448, 200], [554, 216], [659, 272]], "answer": 4},
            separators=(",", ":"),
            ensure_ascii=False,
        ),
        json.dumps({"answer": 4}, separators=(",", ":"), ensure_ascii=False),
    )


def _validate_count_sample(sample: PlatformerSample, *, target_count: int) -> None:
    """Verify the collected coin count and annotation ids are consistent."""

    on_path_ids = tuple(str(coin.collectible_id) for coin in sample.collectibles if bool(coin.on_path))
    if int(sample.answer) != int(target_count):
        raise ValueError("path collectible answer does not match target count")
    if len(on_path_ids) != int(target_count):
        raise ValueError("path collectible construction did not preserve target count")
    if tuple(sample.annotation_entity_ids) != on_path_ids:
        raise ValueError("path collectible annotation ids must match all on-path coins")


def _prepare_count_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    axes: PlatformerVisualAxes,
) -> ObjectivePlatformerPlan:
    """Resolve the target collectible count and bind the path constructor."""

    target_axis = resolve_platformer_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="target_collectible_count_support",
        explicit_key="target_collectible_count",
        fallback_support=DEFAULTS.target_collectible_count_support,
        namespace=f"{TASK_ID}.target_collectible_count",
        balanced_flag_key="balanced_target_collectible_count_sampling",
    )

    def construct_attempt(rng: Any, attempt_axes: PlatformerVisualAxes) -> AttemptPlatformerResult:
        sample = sample_collectible_path_scene(
            rng=rng,
            axes=attempt_axes,
            target_collectible_count=int(target_axis.target_value),
            mode=PROMPT_QUERY_KEY,
        )
        _validate_count_sample(sample, target_count=int(target_axis.target_value))
        annotation_ids = tuple(str(entity_id) for entity_id in sample.annotation_entity_ids)
        return AttemptPlatformerResult(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(sample.answer)),
            annotation_entity_ids=annotation_ids,
            build_annotation=lambda rendered: point_set_for_entity_ids(rendered.rendered_scene, annotation_ids),
            witness_type="object_set",
            relations_extra={"target_collectible_count": int(target_axis.target_value)},
            execution_extra={"target_collectible_count": int(target_axis.target_value)},
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePlatformerPlan(
        attempt_namespace="games.platformer.path_collectible_total",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint='set "answer" to the number of coins lying on the dashed jump arc as an integer',
        annotation_hint='set "annotation" to [x, y] pixel points at the centers of each coin lying on the dashed jump arc',
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={
            "query_id_probabilities": dict(query_probabilities),
            "target_collectible_count": int(target_axis.target_value),
            "target_collectible_count_support": [int(value) for value in target_axis.target_value_support],
            "target_collectible_count_probabilities": dict(target_axis.target_value_probabilities),
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesPlatformerCollectibleCountTask:
    """Count collectibles lying on the full shown jump arc."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
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
            prepare_objective=_prepare_count_objective,
        )


__all__ = ["GamesPlatformerCollectibleCountTask"]
