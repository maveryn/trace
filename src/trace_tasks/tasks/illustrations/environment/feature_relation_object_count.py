"""Count foreground objects in a relation to an environment feature."""

from __future__ import annotations

from functools import partial
from typing import Any, Dict, Mapping, Tuple

from ...base import TaskOutput
from ...registry import register_task
from ...shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from ...shared.fixed_query import select_task_query_id
from ._lifecycle import EnvironmentCountPlan, run_environment_count_lifecycle
from .shared.defaults import FEATURE_RELATION_ON_DEFAULTS, FEATURE_RELATION_SIDE_DEFAULTS, render_fallback
from .shared.output import bind_feature_relation_result
from .shared.prompts import prompt_slots_feature_relation
from .shared.rendering import feature_relation_render_overrides
from .shared.sampling import resolve_feature_choice, sample_scene_object_count, sample_target_count_by_keys
from .shared.state import EnvironmentChoice


TASK_ID = "task_illustrations__environment__feature_relation_object_count"
SCENE_ID = "environment"
ABOVE_FEATURE_QUERY_ID = "above_feature"
BELOW_FEATURE_QUERY_ID = "below_feature"
ON_FEATURE_QUERY_ID = "on_feature"
QUERY_IDS: Tuple[str, ...] = (ABOVE_FEATURE_QUERY_ID, BELOW_FEATURE_QUERY_ID, ON_FEATURE_QUERY_ID)
_QUERY_TO_RELATION = {
    ABOVE_FEATURE_QUERY_ID: "above",
    BELOW_FEATURE_QUERY_ID: "below",
    ON_FEATURE_QUERY_ID: "on",
}
PROMPT_QUERY_KEY = "feature_relation_object_count"
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "illustrations",
    SCENE_ID,
    task_id=TASK_ID,
)


def _sample_target_count(
    params: Mapping[str, Any],
    instance_seed: int,
    choice: EnvironmentChoice,
    generation_defaults: Mapping[str, Any],
) -> Tuple[int, Dict[str, float]]:
    """Sample target support for side or on-feature placement."""

    if str(choice.relation) == "on":
        return sample_target_count_by_keys(
            params,
            instance_seed,
            choice,
            generation_defaults,
            low_key="on_feature_target_count_min",
            high_key="on_feature_target_count_max",
            defaults=FEATURE_RELATION_ON_DEFAULTS,
        )
    return sample_target_count_by_keys(
        params,
        instance_seed,
        choice,
        generation_defaults,
        low_key="feature_side_target_count_min",
        high_key="feature_side_target_count_max",
        defaults=FEATURE_RELATION_SIDE_DEFAULTS,
    )


def _build_plan(*, query_id: str) -> EnvironmentCountPlan:
    """Build the public-owned feature-relation count objective plan."""

    return EnvironmentCountPlan(
        public_id=TASK_ID,
        local_query_id=str(query_id),
        seed_namespace=f"{TASK_ID}:scene",
        prompt_query_key=PROMPT_QUERY_KEY,
        resolve_choice=partial(resolve_feature_choice, public_id=TASK_ID, include_relation=True),
        object_count_sampler=partial(sample_scene_object_count, public_id=TASK_ID, defaults=FEATURE_RELATION_SIDE_DEFAULTS),
        target_count_sampler=_sample_target_count,
        render_overrides=feature_relation_render_overrides,
        bind_result=bind_feature_relation_result,
        prompt_slots=partial(prompt_slots_feature_relation, public_id=TASK_ID),
        render_fallback=render_fallback(FEATURE_RELATION_SIDE_DEFAULTS),
    )


@register_task
class IllustrationsEnvironmentFeatureRelationObjectCountTask:
    """Count foreground objects above, below, or on a road/river feature."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "illustrations"
    supported_query_ids = QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        query_id, _query_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=QUERY_IDS,
            default_query_id=ABOVE_FEATURE_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}:query",
        )
        relation = _QUERY_TO_RELATION[str(query_id)]
        explicit_relation = task_params.get("relation")
        if explicit_relation is not None and str(explicit_relation) != str(relation):
            raise ValueError(f"relation must match query_id {query_id!r}: expected {relation!r}")
        task_params["relation"] = str(relation)
        return run_environment_count_lifecycle(
            plan=_build_plan(query_id=str(query_id)),
            domain=self.domain,
            generation_defaults=_GEN_DEFAULTS,
            rendering_defaults=_RENDER_DEFAULTS,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=task_params,
            max_attempts=int(max_attempts),
        )


__all__ = [
    "ABOVE_FEATURE_QUERY_ID",
    "BELOW_FEATURE_QUERY_ID",
    "IllustrationsEnvironmentFeatureRelationObjectCountTask",
    "ON_FEATURE_QUERY_ID",
]
