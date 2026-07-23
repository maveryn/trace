"""Prompt wording regressions for icon-domain tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.core.seed import hash64
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks import create_task


BANNED_PROMPT_PHRASES = (
    "simple named shape",
    "uniquely identifiable",
    "reference icons A and B",
    "candidate icons",
    "named icon",
    "named icons",
    "unlettered",
    "unlabeled",
    "additional distractor icons",
)


def _prompt_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for item in value.values():
            strings.extend(_prompt_strings(item))
        return strings
    if isinstance(value, (list, tuple)):
        strings = []
        for item in value:
            strings.extend(_prompt_strings(item))
        return strings
    return []


def test_icons_prompt_defaults_avoid_generator_facing_scene_phrases() -> None:
    configs = [get_scene_defaults("icons", group) for group in ("counting", "relation", "pattern", "sequence")]
    configs.append(get_scene_defaults("icons", "named_field"))
    configs.append(get_scene_defaults("icons", "named_grid"))
    configs.append(get_scene_defaults("icons", "paired_canvas"))
    configs.append(get_scene_defaults("icons", "pair_grid"))
    configs.append(get_scene_defaults("icons", "single_transform_options"))
    configs.append(get_scene_defaults("icons", "mirror_grid"))
    for cfg in configs:
        prompt_overrides = dict(cfg["prompt"].get("task_overrides", {}))
        if not prompt_overrides:
            prompt_overrides = {"__shared__": dict(cfg["prompt"].get("shared", {}))}
        for task_id, values in prompt_overrides.items():
            prompt_strings = [
                text
                for key, value in dict(values).items()
                if str(key).startswith(("object_description", "question_text"))
                for text in _prompt_strings(value)
            ]
            joined = "\n".join(prompt_strings)
            for phrase in BANNED_PROMPT_PHRASES:
                assert phrase not in joined, f"{task_id} contains banned prompt phrase: {phrase}"


def test_icons_named_feedback_tasks_render_natural_prompt_text() -> None:
    examples = [
        (
            "task_icons__named_field__closer_to_reference_count",
            {
                "queried_reference_label": "A",
                "target_shape_id": "star",
                "reference_a_shape_id": "circle",
                "reference_b_shape_id": "square",
                "target_answer": 2,
                "target_icon_count": 5,
            },
        ),
        (
            "task_icons__named_field__reference_distance_rank_label",
            {"distance_rank_query": "closest_to_named_reference_label", "answer_label": "A"},
        ),
        (
            "task_icons__named_field__scoped_attribute_count",
            {"query_id": "inside_shape_count", "target_shape_id": "bell", "target_count": 2},
        ),
        (
            "task_icons__named_path__path_neighbor_label",
            {
                "query_id": "after_second_shape_label",
                "answer_label": "E",
                "distractor_count": 5,
                "target_occurrence_count": 3,
            },
        ),
        (
            "task_icons__mirror_grid__mirror_symmetry_match_label",
            {"option_count": 6, "answer_label": "C"},
        ),
    ]
    for index, (task_id, params) in enumerate(examples):
        out = create_task(task_id).generate(
            hash64(20260601, "icons-prompt-wording", index),
            params=params,
            max_attempts=300,
        )
        prompt = str(out.prompt)
        for phrase in BANNED_PROMPT_PHRASES:
            assert phrase not in prompt
        assert "icon" in prompt


def test_icons_option_scene_prompts_use_positive_option_wording() -> None:
    examples = [
        (
            "task_icons__named_field__reference_distance_rank_label",
            {"distance_rank_query": "closest_to_named_reference_label", "answer_label": "A"},
            "six option icons labeled A-F",
        ),
        (
            "task_icons__named_path__path_neighbor_label",
            {
                "query_id": "before_last_shape_label",
                "answer_label": "B",
                "distractor_count": 4,
                "target_occurrence_count": 3,
            },
            "six option icons labeled A-F",
        ),
    ]
    for index, (task_id, params, expected_phrase) in enumerate(examples):
        out = create_task(task_id).generate(
            hash64(20260601, "icons-positive-option-wording", index),
            params=params,
            max_attempts=300,
        )
        prompt = str(out.prompt)
        assert expected_phrase in prompt
        assert "other icons" in prompt
        for phrase in BANNED_PROMPT_PHRASES:
            assert phrase not in prompt
