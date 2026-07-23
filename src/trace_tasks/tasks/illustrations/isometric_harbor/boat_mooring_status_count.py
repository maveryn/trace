"""Count harbor boats by whether they are tied to the dock or in open water."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import select_task_query_id
from trace_tasks.tasks.illustrations.shared.canvas_profiles import resolve_canvas_profile

from ._lifecycle import HarborCountPlan, run_harbor_count_lifecycle, sorted_harbor_boats
from .shared.output import isometric_harbor_mooring_status_count_render_map
from .shared.rendering import BOAT_MOORING_STATUS_VALUES, SCENE_ID, render_isometric_harbor_scene
from .shared.sampling import CountTaskSampleSpec, select_count, support_ints
from .shared.state import IsoHarborEntity, IsoHarborScene


TASK_ID = "task_illustrations__isometric_harbor__boat_mooring_status_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("moored_boat_count", "open_water_boat_count")
QUERY_TO_STATUS: Mapping[str, str] = {
    "moored_boat_count": "moored",
    "open_water_boat_count": "open_water",
}
STATUS_LABELS: Mapping[str, str] = {
    "moored": "boats tied along the main dock",
    "open_water": "boats in open water that are not tied to the dock",
}
_REQUIRED_PROMPT_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "answer_hint_boat_mooring_status_count",
    "annotation_hint_boat_mooring_status_count",
    "json_example_boat_mooring_status_count",
    "json_example_answer_only_boat_mooring_status_count",
)


@dataclass(frozen=True)
class _SampleSpec(CountTaskSampleSpec):
    target_status: str
    target_status_label: str
    other_status: str
    other_count: int
    other_count_probabilities: dict[str, float]


_SCENE_DEFAULTS = get_scene_defaults("illustrations", SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
    task_id=TASK_ID,
)


def _select_other_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    target_count: int,
) -> tuple[int, dict[str, float]]:
    support = support_ints(
        params,
        _GEN_DEFAULTS,
        support_key="other_count_support",
        fallback=(0, 1, 2, 3, 4, 5),
    )
    valid_support = tuple(value for value in support if int(target_count) > 0 or int(value) > 0)
    if not valid_support:
        raise ValueError("other_count_support must allow a nonzero count when target_count is 0")
    explicit = params.get("other_count")
    if explicit is not None:
        value = int(explicit)
        if value not in set(valid_support):
            raise ValueError(f"other_count must be one of {valid_support}")
        return value, support_probability_map(valid_support, selected=value, sort_keys=True)
    namespace = f"{TASK_ID}:other_count"
    if params.get("_sample_cursor") is not None:
        namespace = f"{namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), namespace)
    value, probabilities = uniform_choice_with_probabilities(rng, valid_support, sort_keys=True)
    return int(value), dict(probabilities)


def _sample_spec(*, instance_seed: int, params: Mapping[str, Any]) -> _SampleSpec:
    """Resolve the mooring-status query, target/contrast counts, and canvas profile."""

    selected_query, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=SUPPORTED_QUERY_IDS,
        default_query_id="moored_boat_count",
        task_id=TASK_ID,
        namespace=f"{TASK_ID}:query",
    )
    target_count, target_count_probabilities, answer_count_support = select_count(
        instance_seed=int(instance_seed),
        params=task_params,
        defaults=_GEN_DEFAULTS,
        support_key="answer_count_support",
        explicit_key="target_count",
        fallback=(1, 2, 3, 4, 5, 6),
        namespace=f"{TASK_ID}:target_count",
    )
    other_count, other_count_probabilities = _select_other_count(
        instance_seed=int(instance_seed),
        params=task_params,
        target_count=int(target_count),
    )
    profile = resolve_canvas_profile(
        params=task_params,
        defaults=_RENDER_DEFAULTS,
        fallback_width=1200,
        fallback_height=800,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}:canvas_profile",
    )
    target_status = str(QUERY_TO_STATUS[str(selected_query)])
    other_status = "open_water" if target_status == "moored" else "moored"
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
        target_status=target_status,
        target_status_label=str(STATUS_LABELS[target_status]),
        other_status=other_status,
        other_count=int(other_count),
        other_count_probabilities=dict(other_count_probabilities),
    )


def _matching_boats(scene: IsoHarborScene, sample: _SampleSpec) -> tuple[IsoHarborEntity, ...]:
    return sorted_harbor_boats(
        scene,
        predicate=lambda entity: str(entity.metadata.get("mooring_status", "")) == str(sample.target_status),
    )


def _counts_for_render(sample: _SampleSpec) -> tuple[int, int]:
    moored_count = int(sample.target_count if sample.target_status == "moored" else sample.other_count)
    open_water_count = int(sample.target_count if sample.target_status == "open_water" else sample.other_count)
    return moored_count, open_water_count


def _validate_scene_counts(scene: IsoHarborScene, *, moored_count: int, open_water_count: int) -> None:
    status_counts = dict(scene.trace.get("boat_counts_by_mooring_status", {}))
    if int(status_counts.get("moored", -1)) != int(moored_count):
        raise ValueError("rendered moored count did not match request")
    if int(status_counts.get("open_water", -1)) != int(open_water_count):
        raise ValueError("rendered open-water count did not match request")


def _prompt_slots(prompt_defaults: Mapping[str, Any], sample: CountTaskSampleSpec) -> dict[str, str]:
    status_sample = sample if isinstance(sample, _SampleSpec) else None
    target_status_label = "" if status_sample is None else str(status_sample.target_status_label)
    return {
        "target_status_label": target_status_label,
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults["answer_hint_boat_mooring_status_count"]).format(
            target_status_label=target_status_label
        ),
        "annotation_hint": str(prompt_defaults["annotation_hint_boat_mooring_status_count"]).format(
            target_status_label=target_status_label
        ),
        "json_example": str(prompt_defaults["json_example_boat_mooring_status_count"]),
        "json_example_answer_only": str(prompt_defaults["json_example_answer_only_boat_mooring_status_count"]),
    }


def _identity_fields(sample: CountTaskSampleSpec) -> dict[str, Any]:
    status_sample = sample if isinstance(sample, _SampleSpec) else None
    return {
        "target_mooring_status": "" if status_sample is None else str(status_sample.target_status),
        "target_status_label": "" if status_sample is None else str(status_sample.target_status_label),
    }


def _extra_query_params(sample: CountTaskSampleSpec) -> dict[str, Any]:
    if not isinstance(sample, _SampleSpec):
        return {"allowed_mooring_statuses": list(BOAT_MOORING_STATUS_VALUES)}
    moored_count, open_water_count = _counts_for_render(sample)
    return {
        "allowed_mooring_statuses": list(BOAT_MOORING_STATUS_VALUES),
        "other_mooring_status": str(sample.other_status),
        "other_count": int(sample.other_count),
        "other_count_probabilities": dict(sample.other_count_probabilities),
        "moored_count": int(moored_count),
        "open_water_count": int(open_water_count),
    }


def _build_plan() -> HarborCountPlan:
    """Build the public-owned mooring-status count objective plan."""

    return HarborCountPlan(
        public_id=TASK_ID,
        operation="count_boats_by_mooring_status",
        required_prompt_keys=_REQUIRED_PROMPT_KEYS,
        sample_spec=lambda instance_seed, params: _sample_spec(instance_seed=int(instance_seed), params=params),
        prompt_slots=_prompt_slots,
        scene_builder=_render_scene_from_sample,
        entity_selector=lambda scene, sample: _matching_boats(scene, sample) if isinstance(sample, _SampleSpec) else (),
        render_map=lambda scene, sample, counted_ids: isometric_harbor_mooring_status_count_render_map(
            scene=scene,
            target_status=str(sample.target_status) if isinstance(sample, _SampleSpec) else "",
            counted_entity_ids=counted_ids,
        ),
        identity_fields=_identity_fields,
        extra_query_params=_extra_query_params,
        scene_validator=_validate_scene_from_sample,
    )


def _render_scene_from_sample(scene_seed: int, sample: CountTaskSampleSpec, params: Mapping[str, Any]) -> IsoHarborScene:
    """Render with exact moored/open-water counts requested by the public task sample."""

    if not isinstance(sample, _SampleSpec):
        raise TypeError("mooring-status sample expected")
    moored_count, open_water_count = _counts_for_render(sample)
    return render_isometric_harbor_scene(
        scene_seed,
        width=sample.canvas_width,
        height=sample.canvas_height,
        canvas_profile=sample.canvas_profile,
        canvas_profile_probabilities=sample.canvas_profile_probabilities,
        required_moored_boat_count=moored_count,
        required_open_water_boat_count=open_water_count,
        render_style_params=params,
        render_style_defaults=_RENDER_DEFAULTS,
    )


def _validate_scene_from_sample(scene: IsoHarborScene, sample: CountTaskSampleSpec) -> None:
    """Validate renderer counts against the sampled status-count request."""

    if not isinstance(sample, _SampleSpec):
        raise TypeError("mooring-status sample expected")
    moored_count, open_water_count = _counts_for_render(sample)
    _validate_scene_counts(scene, moored_count=moored_count, open_water_count=open_water_count)


@register_task
class IllustrationsIsometricHarborBoatMooringStatusCountTask:
    """Count boats by mooring status in the isometric harbor."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
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
    "IllustrationsIsometricHarborBoatMooringStatusCountTask",
    "QUERY_TO_STATUS",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
]
