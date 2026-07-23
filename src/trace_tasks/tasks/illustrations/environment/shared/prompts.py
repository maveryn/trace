"""Prompt rendering helpers for environment illustration tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants

from .labels import CROSSED_FEATURE_NAMES, CROSSING_NAMES, feature_relation_phrase
from .sampling import environment_setting_name
from .state import BoundCountResult, EnvironmentChoice


def required_environment_prompt_defaults(
    prompt_defaults: Mapping[str, Any],
    keys: Sequence[str],
    *,
    context: str,
) -> dict[str, Any]:
    """Validate prompt defaults for a public environment task."""

    return dict(required_group_defaults(prompt_defaults, list(keys), context=str(context)))


def render_environment_prompt(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
    preferred_mode: str = "answer_and_annotation",
) -> Any:
    """Render environment prompt variants and trace metadata."""

    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        preferred_mode=str(preferred_mode),
    )
    return build_prompt_trace_artifacts(prompt_selection)


def prompt_slots_feature_relation(
    prompt_defaults: Mapping[str, Any],
    choice: EnvironmentChoice,
    _bound: BoundCountResult,
    _scene: Any,
    *,
    public_id: str,
) -> dict[str, Any]:
    """Format prompt slots for a road/river relation-count question."""

    defaults = required_environment_prompt_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "question_text_feature_relation_object_count",
            "answer_hint_feature_relation",
            "annotation_hint_feature_relation",
            "json_example_feature_relation",
            "json_example_answer_only_feature_relation",
        ],
        context=f"prompt defaults for {public_id}",
    )
    phrase = feature_relation_phrase(choice.feature_type, choice.relation)
    return {
        "environment_setting": environment_setting_name(str(choice.theme_id)),
        "question_text": str(defaults["question_text_feature_relation_object_count"]).format(feature_relation_phrase=str(phrase)),
        "json_output_contract": str(defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
        "answer_hint": str(defaults["answer_hint_feature_relation"]).format(feature_relation_phrase=str(phrase)),
        "annotation_hint": str(defaults["annotation_hint_feature_relation"]).format(feature_relation_phrase=str(phrase)),
        "json_example": str(defaults["json_example_feature_relation"]),
        "json_example_answer_only": str(defaults["json_example_answer_only_feature_relation"]),
    }


def prompt_slots_crossing(
    prompt_defaults: Mapping[str, Any],
    choice: EnvironmentChoice,
    _bound: BoundCountResult,
    _scene: Any,
    *,
    public_id: str,
) -> dict[str, Any]:
    """Format prompt slots for bridge/crosswalk feature-count questions."""

    defaults = required_environment_prompt_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "question_text_crossing_feature_count",
            "answer_hint_crossing_feature",
            "annotation_hint_crossing_feature",
            "json_example_crossing_feature",
            "json_example_answer_only_crossing_feature",
        ],
        context=f"prompt defaults for {public_id}",
    )
    crossing_name = CROSSING_NAMES[str(choice.crossing_type)]
    crossed_feature_name = CROSSED_FEATURE_NAMES[str(choice.crossing_type)]
    return {
        "environment_setting": environment_setting_name(str(choice.theme_id)),
        "question_text": str(defaults["question_text_crossing_feature_count"]).format(
            crossing_name=str(crossing_name),
            crossed_feature_name=str(crossed_feature_name),
        ),
        "json_output_contract": str(defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
        "answer_hint": str(defaults["answer_hint_crossing_feature"]).format(crossing_name=str(crossing_name), crossed_feature_name=str(crossed_feature_name)),
        "annotation_hint": str(defaults["annotation_hint_crossing_feature"]).format(crossing_name=str(crossing_name), crossed_feature_name=str(crossed_feature_name)),
        "json_example": str(defaults["json_example_crossing_feature"]),
        "json_example_answer_only": str(defaults["json_example_answer_only_crossing_feature"]),
    }


def prompt_slots_window(
    prompt_defaults: Mapping[str, Any],
    choice: EnvironmentChoice,
    _bound: BoundCountResult,
    _scene: Any,
    *,
    public_id: str,
) -> dict[str, Any]:
    """Format prompt slots for the lit-window building-count contract."""

    defaults = required_environment_prompt_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "question_text_building_window_count",
            "answer_hint_building_window",
            "annotation_hint_building_window",
            "json_example_building_window",
            "json_example_answer_only_building_window",
        ],
        context=f"prompt defaults for {public_id}",
    )
    window_phrase = "lit windows"
    return {
        "environment_setting": environment_setting_name(str(choice.theme_id)),
        "question_text": str(defaults["question_text_building_window_count"]).format(window_phrase=str(window_phrase)),
        "json_output_contract": str(defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
        "answer_hint": str(defaults["answer_hint_building_window"]).format(window_phrase=str(window_phrase)),
        "annotation_hint": str(defaults["annotation_hint_building_window"]).format(window_phrase=str(window_phrase)),
        "json_example": str(defaults["json_example_building_window"]),
        "json_example_answer_only": str(defaults["json_example_answer_only_building_window"]),
    }


__all__ = [
    "prompt_slots_crossing",
    "prompt_slots_feature_relation",
    "prompt_slots_window",
    "render_environment_prompt",
    "required_environment_prompt_defaults",
]
