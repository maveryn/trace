"""Shared sampling helpers for park/playground illustration tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from .....core.sampling import uniform_choice_with_probabilities
from .....core.query_ids import SINGLE_QUERY_ID
from ....shared.config_defaults import group_default
from ....shared.fixed_query import select_task_query_id
from ...shared.object_library import STYLE_IDS
from ...shared.task_support import (
    bounds,
    render_params as _shared_render_params,
    sample_count,
    spawned_task_rng,
    style_weights as _shared_style_weights,
    uniform_string_probability_map,
)
from .defaults import CountDefaults
from .state import (
    EquipmentSampleSpec,
    PARK_EQUIPMENT_TYPES,
    PARK_PERSON_ACTIVITIES,
    PARK_SETTING_IDS,
    PARK_ZONE_TYPES,
    ParkEquipmentSpec,
    ParkPersonSpec,
    PersonCountSampleSpec,
    park_activity_display_name,
    park_equipment_display_name,
    park_zone_display_name,
)


def activity_support(params: Mapping[str, Any], defaults: Mapping[str, Any], *, fallback: Sequence[str]) -> Tuple[str, ...]:
    """Resolve supported park person activity ids."""

    raw = params.get("activity_support", group_default(defaults, "activity_support", fallback))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("activity_support must be a sequence")
    support = tuple(str(value) for value in raw if str(value) in set(PARK_PERSON_ACTIVITIES))
    if len(support) < 2:
        raise ValueError("activity_support must contain at least two activities")
    return tuple(dict.fromkeys(support))


def equipment_support(params: Mapping[str, Any], defaults: Mapping[str, Any], *, fallback: Sequence[str] = PARK_EQUIPMENT_TYPES) -> Tuple[str, ...]:
    """Resolve supported park playground equipment ids."""

    raw = params.get("equipment_type_support", group_default(defaults, "equipment_type_support", fallback))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("equipment_type_support must be a sequence")
    support = tuple(str(value) for value in raw if str(value) in set(PARK_EQUIPMENT_TYPES))
    if len(support) < 2:
        raise ValueError("equipment_type_support must contain at least two equipment types")
    return tuple(dict.fromkeys(support))


def zone_support(params: Mapping[str, Any], defaults: Mapping[str, Any], *, fallback: Sequence[str] = PARK_ZONE_TYPES) -> Tuple[str, ...]:
    """Resolve supported semantic park zone ids."""

    raw = params.get("zone_support", group_default(defaults, "zone_support", fallback))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("zone_support must be a sequence")
    support = tuple(str(value) for value in raw if str(value) in set(PARK_ZONE_TYPES))
    if len(support) < 2:
        raise ValueError("zone_support must contain at least two park zones")
    return tuple(dict.fromkeys(support))


def support_choice(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    support: Sequence[str],
    explicit_key: str,
) -> Tuple[str, Dict[str, float]]:
    """Choose one supported semantic operand with an explicit override or seed."""

    values = tuple(str(value) for value in support if str(value))
    if not values:
        raise ValueError(f"{explicit_key} support must not be empty")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = str(explicit)
        if selected not in set(values):
            raise ValueError(f"{explicit_key} is outside configured support")
        return selected, uniform_string_probability_map(values, selected=selected)
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = uniform_choice_with_probabilities(rng, values, sort_keys=False)
    return str(selected), dict(probabilities)


def render_params(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    *,
    fallback_width: int,
    fallback_height: int,
    fallback_scale: int,
    instance_seed: int | None = None,
    namespace: str = "park_playground:canvas_profile",
) -> Dict[str, Any]:
    """Resolve park canvas render parameters."""

    return _shared_render_params(
        params,
        render_defaults,
        prefix="park",
        fallback_width=fallback_width,
        fallback_height=fallback_height,
        fallback_scale=fallback_scale,
        instance_seed=instance_seed,
        namespace=namespace,
    )


def style_weights(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> Dict[str, float]:
    """Resolve illustration style weights."""

    return _shared_style_weights(params, render_defaults, style_ids=STYLE_IDS)


def setting_weights(params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> Dict[str, float]:
    """Resolve park setting weights."""

    raw = params.get("park_setting_weights", group_default(render_defaults, "park_setting_weights", {setting: 1.0 for setting in PARK_SETTING_IDS}))
    if not isinstance(raw, Mapping):
        raise ValueError("park_setting_weights must be a mapping")
    return {str(key): max(0.0, float(value)) for key, value in raw.items()}


def sample_people_total(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    attempt_index: int,
    namespace: str,
    query_support: Sequence[str],
    generation_defaults: Mapping[str, Any],
    defaults: CountDefaults,
) -> PersonCountSampleSpec:
    """Sample visible people for a total people-count objective."""

    branch_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in query_support),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(namespace),
        namespace=f"{namespace}:query",
    )
    rng = spawned_task_rng(int(instance_seed), str(namespace), int(attempt_index))
    activities = activity_support(task_params, generation_defaults, fallback=PARK_PERSON_ACTIVITIES)
    person_min, person_max = bounds(
        task_params,
        generation_defaults,
        "person_count_min",
        "person_count_max",
        defaults.person_count_min,
        defaults.person_count_max,
    )
    people_count_key = "person" + str("_count")
    person_count, person_probabilities = sample_count(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:{people_count_key}",
        low=int(person_min),
        high=int(person_max),
        explicit_key=people_count_key,
    )
    person_specs = tuple(
        ParkPersonSpec(activity=str(rng.choice(activities)), role="target")
        for _ in range(int(person_count))
    )
    return PersonCountSampleSpec(
        branch_id=str(branch_id),
        person_count=int(person_count),
        person_specs=person_specs,
        query_probabilities=dict(query_probabilities),
        person_count_probabilities=dict(person_probabilities),
    )


def sample_option_answer_index(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    seed_scope: str,
    option_labels: Sequence[str],
    namespace_suffix: str = "answer",
    explicit_index_key: str = "correct_index",
    explicit_label_key: str = "answer_label",
) -> Tuple[int, Dict[str, float]]:
    """Sample a visible option-label index with common explicit overrides."""

    labels = tuple(str(label) for label in option_labels)
    if len(labels) < 2:
        raise ValueError("option_labels must contain at least two labels")
    if params.get(str(explicit_index_key)) is not None:
        value = int(params[str(explicit_index_key)])
        if value < 0 or value >= len(labels):
            raise ValueError(f"{explicit_index_key} outside option label support")
        return int(value), {str(value): 1.0}
    if params.get(str(explicit_label_key)) is not None:
        label = str(params[str(explicit_label_key)])
        if label not in set(labels):
            raise ValueError(f"{explicit_label_key} outside option label support")
        value = int(labels.index(label))
        return int(value), {str(value): 1.0}
    namespace = f"{seed_scope}:{namespace_suffix}"
    if params.get("_sample_cursor") is not None:
        namespace = f"{namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), namespace)
    value, probabilities = uniform_choice_with_probabilities(
        rng,
        tuple(range(len(labels))),
        sort_keys=True,
    )
    return int(value), dict(probabilities)


def sample_equipment_items(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    attempt_index: int,
    namespace: str,
    query_support: Sequence[str],
    generation_defaults: Mapping[str, Any],
    defaults: CountDefaults,
) -> EquipmentSampleSpec:
    """Sample one equipment type with exact target and distractor equipment."""

    branch_id, query_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=tuple(str(value) for value in query_support),
        default_query_id=SINGLE_QUERY_ID,
        task_id=str(namespace),
        namespace=f"{namespace}:query",
    )
    rng = spawned_task_rng(int(instance_seed), str(namespace), int(attempt_index))
    equipment_values = equipment_support(task_params, generation_defaults, fallback=PARK_EQUIPMENT_TYPES)
    target_equipment, equipment_probabilities = support_choice(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:target_equipment_type",
        support=equipment_values,
        explicit_key="target_equipment_type",
    )
    target_min, target_max = bounds(
        task_params,
        generation_defaults,
        "target_count_min",
        "target_count_max",
        defaults.target_count_min,
        defaults.target_count_max,
    )
    target_count, target_probabilities = sample_count(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:target_count",
        low=int(target_min),
        high=int(target_max),
        explicit_key="target_count",
    )
    equipment_min, equipment_max = bounds(
        task_params,
        generation_defaults,
        "equipment_count_min",
        "equipment_count_max",
        defaults.equipment_count_min,
        defaults.equipment_count_max,
    )
    equipment_count, equipment_count_probabilities = sample_count(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:equipment_count",
        low=max(int(equipment_min), int(target_count) + 2),
        high=int(equipment_max),
        explicit_key="equipment_count",
    )
    person_min, person_max = bounds(
        task_params,
        generation_defaults,
        "person_count_min",
        "person_count_max",
        defaults.person_count_min,
        defaults.person_count_max,
    )
    people_count_key = "person" + str("_count")
    person_count, person_probabilities = sample_count(
        params=task_params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}:{people_count_key}",
        low=int(person_min),
        high=int(person_max),
        explicit_key=people_count_key,
    )
    distractor_equipment = [str(value) for value in equipment_values if str(value) != str(target_equipment)]
    if not distractor_equipment:
        raise ValueError("equipment count needs at least one distractor equipment type")
    equipment_specs = [ParkEquipmentSpec(equipment_type=str(target_equipment), role="target") for _ in range(int(target_count))]
    for index in range(int(equipment_count) - int(target_count)):
        equipment_type = str(distractor_equipment[index]) if index < len(distractor_equipment) else str(rng.choice(tuple(distractor_equipment)))
        equipment_specs.append(ParkEquipmentSpec(equipment_type=equipment_type, role="distractor"))
    rng.shuffle(equipment_specs)
    person_specs = tuple(
        ParkPersonSpec(activity=str(rng.choice(PARK_PERSON_ACTIVITIES)), role="decor")
        for _ in range(int(person_count))
    )
    return EquipmentSampleSpec(
        branch_id=str(branch_id),
        target_equipment_type=str(target_equipment),
        equipment_name=park_equipment_display_name(str(target_equipment)),
        target_count=int(target_count),
        equipment_count=int(equipment_count),
        person_count=int(person_count),
        equipment_specs=tuple(equipment_specs),
        person_specs=tuple(person_specs),
        query_probabilities=dict(query_probabilities),
        target_equipment_probabilities=dict(equipment_probabilities),
        target_count_probabilities=dict(target_probabilities),
        equipment_count_probabilities=dict(equipment_count_probabilities),
        person_count_probabilities=dict(person_probabilities),
    )


__all__ = [
    "activity_support",
    "bounds",
    "equipment_support",
    "render_params",
    "sample_equipment_items",
    "sample_count",
    "sample_option_answer_index",
    "sample_people_total",
    "setting_weights",
    "spawned_task_rng",
    "style_weights",
    "support_choice",
    "uniform_string_probability_map",
    "zone_support",
]
