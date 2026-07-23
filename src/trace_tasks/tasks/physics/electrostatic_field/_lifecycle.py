"""Scene-private lifecycle assembly for electrostatic field tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.fixed_query import force_query_id_params
from trace_tasks.tasks.shared.output_metadata import default_task_versions
from trace_tasks.tasks.shared.prompt_variants import build_prompt_query_spec

from .shared.output import point_map_value, visual_trace_sections
from .shared.output import (
    direction_axis_params,
    direction_scenario_payload,
    option_letter_axis_params,
    potential_axis_params,
    potential_scenario_payload,
    scene_axis_params,
    zero_field_scenario_payload,
)
from .shared.prompts import build_electrostatic_prompt_artifacts
from .shared.rendering import render_with_attempts
from .shared.sampling import (
    make_direction_spec,
    make_potential_spec,
    make_zero_field_spec,
    resolve_direction_axes,
    resolve_option_letter_axes,
    resolve_potential_axes,
    resolve_scene_axes,
)
from .shared.state import (
    POINT_LETTERS,
    SCENE_MODE_DIRECTION,
    SCENE_MODE_POTENTIAL,
    SCENE_MODE_ZERO_FIELD,
)


def run_point_map_lifecycle(
    *,
    domain: str,
    scene_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    bundle_fallback: str,
    prompt_task_key: str,
    prompt_query_key: str,
    render_scope: str,
    prepare_axes: Callable[[int, Mapping[str, Any], Mapping[str, Any], str], Mapping[str, Any]],
    construct_scene: Callable[[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Any],
    prompt_slots: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    bind_answer: Callable[[Any], TypedValue],
    scenario_payload: Callable[[Any, Any, TypedValue], Mapping[str, Any]],
    params_payload: Callable[[Mapping[str, Any], TypedValue], Mapping[str, Any]],
    scene_kind: Callable[[Mapping[str, Any]], str],
    scenario_trace_key: str,
    answer_type_name: str,
    query_id: str = SINGLE_QUERY_ID,
    query_id_probabilities: Mapping[str, float] | None = None,
) -> TaskOutput:
    """Assemble one point-map task output from public objective hooks."""

    resolved_query_id = str(query_id)
    task_params = force_query_id_params(params or {}, query_id=resolved_query_id)
    task_params.pop("query_id", None)
    prepared = dict(
        prepare_axes(
            int(instance_seed),
            task_params,
            generation_defaults,
            str(namespace),
        )
    )
    scene_axes = prepared["scene_axes"]
    scene_spec, rendered = render_with_attempts(
        instance_seed=int(instance_seed),
        params=task_params,
        max_attempts=int(max_attempts),
        rendering_defaults=rendering_defaults,
        accent_color_name=str(scene_axes.accent_color_name),
        namespace=str(namespace),
        make_scene_spec=lambda rng: construct_scene(
            rng,
            prepared,
            task_params,
            generation_defaults,
        ),
    )
    resolved_prompt_defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "task_key"),
        context=f"prompt defaults for {scene_id}",
    )
    prompt_artifacts = build_electrostatic_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(resolved_prompt_defaults.get("bundle_id", bundle_fallback)),
        task_key=str(resolved_prompt_defaults.get("task_key", prompt_task_key)),
        prompt_query_key=str(prompt_query_key),
        dynamic_slots={str(key): value for key, value in prompt_slots(prepared).items()},
        instance_seed=int(instance_seed),
    )
    answer_gt = bind_answer(scene_spec)
    annotation_value = point_map_value(rendered)
    annotation_gt = TypedValue(type="point_map", value=annotation_value)
    scenario = dict(scenario_payload(scene_spec, rendered, answer_gt))
    query_params = dict(params_payload(prepared, answer_gt))
    resolved_query_probabilities = {
        str(key): float(value)
        for key, value in dict(query_id_probabilities or {resolved_query_id: 1.0}).items()
    }
    query_params["query_id"] = resolved_query_id
    query_params["query_id_probabilities"] = dict(resolved_query_probabilities)
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_artifacts,
        query_id=resolved_query_id,
        params=query_params,
    )
    trace_payload = {
        "scene_ir": {
            "scene_kind": str(scene_kind(prepared)),
            "entities": [dict(entity) for entity in rendered.scene_entities],
            "relations": {
                "query_id": resolved_query_id,
                "scene_mode": str(query_params.get("scene_mode", "")),
                "target_answer": answer_gt.value,
                str(scenario_trace_key): dict(scenario),
                "annotation_entity_ids": list(rendered.annotation_entity_ids),
                "annotation_key_by_entity_id": dict(rendered.annotation_key_by_entity_id),
            },
        },
        "query_spec": prompt_query_spec,
        "execution_trace": {
            **query_params,
            "answer_type": str(answer_type_name),
            str(scenario_trace_key): dict(scenario),
            "annotation_entity_ids": list(rendered.annotation_entity_ids),
            "annotation_key_by_entity_id": dict(rendered.annotation_key_by_entity_id),
        },
        **visual_trace_sections(
            rendered,
            annotation_value=annotation_value,
            render_scope=str(render_scope),
        ),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=rendered.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=default_task_versions(),
        scene_id=str(scene_id),
        query_id=resolved_query_id,
    )


def _prepare_direction_axes(
    instance_seed: int,
    task_params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> dict[str, Any]:
    """Resolve scene and requested-direction axes for the direction objective."""

    return {
        "scene_axes": resolve_scene_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=defaults,
            namespace=str(namespace),
        ),
        "direction_axes": resolve_direction_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=defaults,
            namespace=str(namespace),
        ),
    }


def _construct_direction_scene(
    rng: Any,
    prepared: Mapping[str, Any],
    _task_params: Mapping[str, Any],
    _defaults: Mapping[str, Any],
):
    """Construct the exact symbolic field-direction scene."""

    return make_direction_spec(
        scene_axes=prepared["scene_axes"],
        direction_axes=prepared["direction_axes"],
        rng=rng,
    )


def _build_direction_prompt_slots(prepared: Mapping[str, Any]) -> dict[str, str]:
    """Return no dynamic prompt slots for field/force direction requests."""

    return {}


def _bind_option_answer(scene_spec: Any) -> TypedValue:
    """Bind an option-letter answer from the symbolic target answer."""

    return TypedValue(type="option_letter", value=str(scene_spec.target_answer))


def _describe_direction_scenario(scene_spec: Any, rendered: Any, _answer: TypedValue) -> dict[str, Any]:
    """Return symbolic direction metadata for verifier tracing."""

    return direction_scenario_payload(scene_spec, rendered)


def _build_direction_query_params(prepared: Mapping[str, Any], answer: TypedValue) -> dict[str, Any]:
    """Build prompt/query params for the direction objective."""

    return {
        "query_id": SINGLE_QUERY_ID,
        "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
        "scene_mode": SCENE_MODE_DIRECTION,
        "target_answer": str(answer.value),
        **scene_axis_params(prepared["scene_axes"]),
        **direction_axis_params(prepared["direction_axes"]),
    }


def _direction_scene_kind(prepared: Mapping[str, Any]) -> str:
    """Name the rendered direction scene variant for trace metadata."""

    return f"physics_electrostatic_field_{prepared['scene_axes'].scene_variant}_direction"


def run_direction_lifecycle(
    *,
    domain: str,
    scene_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    bundle_fallback: str,
    prompt_task_key: str,
    prompt_query_key: str,
    query_id: str,
    query_id_probabilities: Mapping[str, float],
) -> TaskOutput:
    """Run the direction-choice lifecycle with public prompt keys supplied by the task."""

    return run_point_map_lifecycle(
        domain=domain,
        scene_id=scene_id,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        namespace=str(namespace),
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        bundle_fallback=str(bundle_fallback),
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        render_scope="electrostatic_field_map",
        prepare_axes=_prepare_direction_axes,
        construct_scene=_construct_direction_scene,
        prompt_slots=_build_direction_prompt_slots,
        bind_answer=_bind_option_answer,
        scenario_payload=_describe_direction_scenario,
        params_payload=_build_direction_query_params,
        scene_kind=_direction_scene_kind,
        scenario_trace_key="direction_scenario",
        answer_type_name="option_letter",
        query_id=str(query_id),
        query_id_probabilities=query_id_probabilities,
    )


def _prepare_zero_axes(
    instance_seed: int,
    task_params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> dict[str, Any]:
    """Resolve scene style and the candidate letter that is physically correct."""

    return {
        "scene_axes": resolve_scene_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=defaults,
            namespace=str(namespace),
        ),
        "option_axes": resolve_option_letter_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=defaults,
            namespace=str(namespace),
            option_letters=POINT_LETTERS,
            weights_key="point_option_letter_weights",
            balance_key="balanced_point_option_letter_sampling",
        ),
    }


def _construct_zero_scene(
    rng: Any,
    prepared: Mapping[str, Any],
    _task_params: Mapping[str, Any],
    _defaults: Mapping[str, Any],
):
    """Place charges and candidate points so one option has zero net field."""

    return make_zero_field_spec(
        scene_axes=prepared["scene_axes"],
        option_axes=prepared["option_axes"],
        rng=rng,
    )


def _empty_prompt_slots(_prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Return no dynamic prompt slots."""

    return {}


def _describe_zero_scenario(scene_spec: Any, rendered: Any, _answer: TypedValue) -> dict[str, Any]:
    """Return charge and candidate metadata for the selected zero-field option."""

    return zero_field_scenario_payload(scene_spec, rendered)


def _build_zero_query_params(prepared: Mapping[str, Any], answer: TypedValue) -> dict[str, Any]:
    """Build prompt/query params for the zero-field objective."""

    return {
        "query_id": SINGLE_QUERY_ID,
        "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
        "scene_mode": SCENE_MODE_ZERO_FIELD,
        "target_answer": str(answer.value),
        **scene_axis_params(prepared["scene_axes"]),
        **option_letter_axis_params(prepared["option_axes"]),
    }


def _zero_scene_kind(prepared: Mapping[str, Any]) -> str:
    """Name the rendered zero-field scene variant for trace metadata."""

    return f"physics_electrostatic_field_{prepared['scene_axes'].scene_variant}_zero_field"


def run_zero_lifecycle(
    *,
    domain: str,
    scene_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    bundle_fallback: str,
    prompt_task_key: str,
    prompt_query_key: str,
) -> TaskOutput:
    """Run the zero-field lifecycle with public prompt keys supplied by the task."""

    return run_point_map_lifecycle(
        domain=domain,
        scene_id=scene_id,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        namespace=str(namespace),
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        bundle_fallback=str(bundle_fallback),
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        render_scope="electrostatic_field_map",
        prepare_axes=_prepare_zero_axes,
        construct_scene=_construct_zero_scene,
        prompt_slots=_empty_prompt_slots,
        bind_answer=_bind_option_answer,
        scenario_payload=_describe_zero_scenario,
        params_payload=_build_zero_query_params,
        scene_kind=_zero_scene_kind,
        scenario_trace_key="zero_field_scenario",
        answer_type_name="option_letter",
    )


def _prepare_numeric_axes(
    instance_seed: int,
    task_params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    namespace: str,
) -> dict[str, Any]:
    """Resolve scene axes and the exact integer potential answer."""

    return {
        "scene_axes": resolve_scene_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=defaults,
            namespace=str(namespace),
        ),
        "potential_axes": resolve_potential_axes(
            instance_seed=int(instance_seed),
            params=task_params,
            defaults=defaults,
            namespace=str(namespace),
        ),
    }


def _construct_numeric_scene(
    rng: Any,
    prepared: Mapping[str, Any],
    task_params: Mapping[str, Any],
    defaults: Mapping[str, Any],
):
    """Choose charge values and distances whose q/r terms sum exactly."""

    return make_potential_spec(
        scene_axes=prepared["scene_axes"],
        potential_axes=prepared["potential_axes"],
        rng=rng,
        params=task_params,
        defaults=defaults,
    )


def _bind_integer_answer(scene_spec: Any) -> TypedValue:
    """Bind a signed integer answer from the symbolic target answer."""

    return TypedValue(type="integer", value=int(scene_spec.target_answer))


def _describe_numeric_scenario(
    scene_spec: Any,
    rendered: Any,
    answer: TypedValue,
    *,
    value_trace_key: str,
) -> dict[str, Any]:
    """Return charge-distance contribution metadata with the requested value key."""

    payload = dict(potential_scenario_payload(scene_spec, rendered))
    payload[str(value_trace_key)] = int(answer.value)
    return payload


def _build_numeric_query_params(prepared: Mapping[str, Any], answer: TypedValue) -> dict[str, Any]:
    """Build prompt/query params for the numeric potential objective."""

    return {
        "query_id": SINGLE_QUERY_ID,
        "query_id_probabilities": {SINGLE_QUERY_ID: 1.0},
        "scene_mode": SCENE_MODE_POTENTIAL,
        "target_answer": int(answer.value),
        **scene_axis_params(prepared["scene_axes"]),
        **potential_axis_params(prepared["potential_axes"]),
    }


def _numeric_scene_kind(prepared: Mapping[str, Any]) -> str:
    """Name the rendered numeric-potential scene variant for trace metadata."""

    return f"physics_electrostatic_field_{prepared['scene_axes'].scene_variant}_potential"


def run_numeric_lifecycle(
    *,
    domain: str,
    scene_id: str,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    namespace: str,
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    bundle_fallback: str,
    prompt_task_key: str,
    prompt_query_key: str,
    value_trace_key: str,
) -> TaskOutput:
    """Run the numeric-potential lifecycle with public prompt keys supplied by the task."""

    return run_point_map_lifecycle(
        domain=domain,
        scene_id=scene_id,
        instance_seed=int(instance_seed),
        params=params,
        max_attempts=int(max_attempts),
        namespace=str(namespace),
        generation_defaults=generation_defaults,
        rendering_defaults=rendering_defaults,
        prompt_defaults=prompt_defaults,
        bundle_fallback=str(bundle_fallback),
        prompt_task_key=str(prompt_task_key),
        prompt_query_key=str(prompt_query_key),
        render_scope="electrostatic_field_map",
        prepare_axes=_prepare_numeric_axes,
        construct_scene=_construct_numeric_scene,
        prompt_slots=_empty_prompt_slots,
        bind_answer=_bind_integer_answer,
        scenario_payload=lambda scene_spec, rendered, answer: _describe_numeric_scenario(
            scene_spec,
            rendered,
            answer,
            value_trace_key=str(value_trace_key),
        ),
        params_payload=_build_numeric_query_params,
        scene_kind=_numeric_scene_kind,
        scenario_trace_key="potential_scenario",
        answer_type_name="integer",
    )
