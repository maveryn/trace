"""Count boats docked on the image-left or image-right side of the main dock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.sampling import support_probability_map
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile

from ._lifecycle import HarborCountPlan, run_harbor_count_lifecycle, sorted_harbor_boats
from .shared.output import isometric_harbor_boat_count_render_map
from .shared.rendering import BOAT_SIDE_VALUES, SCENE_ID, render_isometric_harbor_scene
from .shared.sampling import CountTaskSampleSpec, select_count
from .shared.state import IsoHarborEntity, IsoHarborScene


TASK_ID = "task_illustrations__isometric_harbor__boat_side_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("left_side_boat_count", "right_side_boat_count")
QUERY_TO_SIDE: Mapping[str, str] = {
    "left_side_boat_count": "left",
    "right_side_boat_count": "right",
}
_REQUIRED_PROMPT_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "answer_hint_boat_side_count",
    "annotation_hint_boat_side_count",
    "json_example_boat_side_count",
    "json_example_answer_only_boat_side_count",
)


@dataclass(frozen=True)
class _SampleSpec(CountTaskSampleSpec):
    target_side: str
    target_side_label: str


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Resolve the side-count query, target count, and canvas profile."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="left_side_boat_count",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    target_count, target_count_probabilities, answer_count_support = select_count(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=_GEN_DEFAULTS,
        support_key="answer_count_support",
        explicit_key="target_count",
        fallback=(0, 1, 2, 3, 4, 5),
        namespace=f"{TASK_ID}:target_count",
    )
    profile = resolve_canvas_profile(
        params=task_params,
        defaults=_RENDER_DEFAULTS,
        fallback_width=1200,
        fallback_height=800,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:canvas_profile",
    )
    target_side = str(QUERY_TO_SIDE[str(selected_query)])
    return _SampleSpec(
        selected_key=str(selected_query),
        prompt_query_key=str(selected_query),
        query_probabilities=dict(query_probabilities),
        target_count=int(target_count),
        target_count_probabilities=dict(target_count_probabilities),
        answer_count_support=tuple(int(value) for value in answer_count_support),
        answer_count_probabilities=dict(support_probability_map(answer_count_support, sort_keys=True)),
        canvas_width=int(profile.width),
        canvas_height=int(profile.height),
        canvas_profile=str(profile.profile_id),
        canvas_profile_probabilities=dict(profile.probabilities),
        target_side=target_side,
        target_side_label="image-left side" if target_side == "left" else "image-right side",
    )


def _matching_boats(scene: IsoHarborScene, sample: _SampleSpec) -> tuple[IsoHarborEntity, ...]:
    return sorted_harbor_boats(
        scene,
        predicate=lambda entity: str(entity.metadata.get("dock_side", "")) == str(sample.target_side),
    )


def _prompt_slots(prompt_defaults: Mapping[str, Any], sample: CountTaskSampleSpec) -> dict[str, str]:
    side_sample = sample if isinstance(sample, _SampleSpec) else None
    target_side_label = "" if side_sample is None else str(side_sample.target_side_label)
    return {
        "target_side_label": target_side_label,
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_boat_side_count"]).format(
            target_side_label=target_side_label
        ),
        "annotation_hint": str(prompt_defaults["annotation_hint_boat_side_count"]).format(
            target_side_label=target_side_label
        ),
        "json_example": str(prompt_defaults["json_example_boat_side_count"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_boat_side_count"]),
    }


def _identity_fields(sample: CountTaskSampleSpec) -> dict[str, Any]:
    side_sample = sample if isinstance(sample, _SampleSpec) else None
    return {
        "target_side": "" if side_sample is None else str(side_sample.target_side),
        "target_side_label": "" if side_sample is None else str(side_sample.target_side_label),
    }


def _extra_query_params(sample: CountTaskSampleSpec) -> dict[str, Any]:
    return {"allowed_sides": list(BOAT_SIDE_VALUES)}


def _build_plan() -> HarborCountPlan:
    """Build the public-owned boat-side count objective plan."""

    return HarborCountPlan(
        public_id=TASK_ID,
        operation="count_boats_on_main_dock_side",
        required_prompt_keys=_REQUIRED_PROMPT_KEYS,
        sample_spec=lambda instance_seed, params: _sample_spec(instance_seed=int(instance_seed), params=params),
        prompt_slots=_prompt_slots,
        scene_builder=lambda scene_seed, sample, task_params: render_isometric_harbor_scene(
            scene_seed,
            width=sample.canvas_width,
            height=sample.canvas_height,
            canvas_profile=sample.canvas_profile,
            canvas_profile_probabilities=sample.canvas_profile_probabilities,
            required_boat_counts_by_side={
                str(sample.target_side): int(sample.target_count)
            } if isinstance(sample, _SampleSpec) else {},
            render_style_params=task_params,
            render_style_defaults=_RENDER_DEFAULTS,
        ),
        entity_selector=lambda scene, sample: _matching_boats(scene, sample) if isinstance(sample, _SampleSpec) else (),
        render_map=lambda scene, sample, counted_ids: isometric_harbor_boat_count_render_map(
            scene=scene,
            target_side=str(sample.target_side) if isinstance(sample, _SampleSpec) else "",
            counted_entity_ids=counted_ids,
        ),
        identity_fields=_identity_fields,
        extra_query_params=_extra_query_params,
    )


@register_task
class IllustrationsIsometricHarborBoatSideCountTask:
    """Count boats docked on one side of the main dock."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "illustrations"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_harbor_count_lifecycle(
            plan=_build_plan(),
            domain=self.domain,
            prompt_defaults=_PROMPT_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
        )


__all__ = [
    "IllustrationsIsometricHarborBoatSideCountTask",
    "QUERY_TO_SIDE",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
