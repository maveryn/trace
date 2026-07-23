"""Prompt assembly helpers for single-series chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


OBJECT_DESCRIPTION_BY_VARIANT: dict[str, str] = {
    "area": "an ordered labeled area chart where point labels identify y-values from left to right",
    "bar": "an ordered labeled bar chart where each bar height is read from the vertical axis from left to right",
    "horizontal_bar": "an ordered labeled horizontal bar chart where each bar length is read from the horizontal axis from top to bottom",
    "line": "an ordered labeled line chart where point labels identify y-values from left to right",
    "scatter": "a labeled scatter plot where point labels identify y-values on the vertical axis",
    "dot_plot": "an ordered labeled dot plot where point labels identify y-values from left to right",
    "lollipop": "an ordered labeled lollipop chart where point labels identify y-values from left to right",
}


def object_description(scene_variant: str) -> str:
    return str(OBJECT_DESCRIPTION_BY_VARIANT.get(str(scene_variant), "an ordered labeled chart"))


def render_prompt_artifacts(
    *,
    bundle_id: str,
    scene_key: str,
    task_key: str,
    prompt_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
        scene_key=str(scene_key),
        task_key=str(task_key),
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)


def quoted_join(labels: Sequence[str]) -> str:
    quoted = [f'"{str(label)}"' for label in labels]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return str(quoted[0])
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return f"{', '.join(quoted[:-1])}, and {quoted[-1]}"


def ordinal(value: int) -> str:
    number = int(value)
    suffix = "th"
    if number % 100 not in {11, 12, 13}:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def statistic_prompt(statistic_kind: str, *, rank_text: str) -> str:
    if str(statistic_kind) == "median":
        return "median value"
    if str(statistic_kind) == "nth_highest":
        return f"{str(rank_text)}-highest distinct displayed value"
    if str(statistic_kind) == "nth_lowest":
        return f"{str(rank_text)}-lowest distinct displayed value"
    raise ValueError(f"unsupported statistic_kind: {statistic_kind}")


def turning_slots(kind: str) -> dict[str, str]:
    if str(kind) == "peak":
        return {
            "turning_point_type": "peak",
            "turning_point_plural": "peaks",
            "turning_point_comparison": "higher than",
        }
    if str(kind) == "trough":
        return {
            "turning_point_type": "trough",
            "turning_point_plural": "troughs",
            "turning_point_comparison": "lower than",
        }
    raise ValueError(f"unsupported turning point kind: {kind}")


def streak_slots(direction: str) -> dict[str, str]:
    if str(direction) == "increasing":
        return {"streak_direction": "increasing", "streak_step_verb": "increases"}
    if str(direction) == "decreasing":
        return {"streak_direction": "decreasing", "streak_step_verb": "decreases"}
    raise ValueError(f"unsupported streak direction: {direction}")


def endpoint_slots(kind: str) -> dict[str, str]:
    if str(kind) == "absolute":
        return {"endpoint_change_kind": "absolute", "endpoint_change_instruction": "absolute change in value"}
    if str(kind) == "signed":
        return {
            "endpoint_change_kind": "signed",
            "endpoint_change_instruction": "signed change in value, keeping decreases negative",
        }
    if str(kind) == "percent":
        return {
            "endpoint_change_kind": "percent",
            "endpoint_change_instruction": (
                "integer percentage change using the start label's value as the base, omitting the percent sign"
            ),
        }
    raise ValueError(f"unsupported endpoint kind: {kind}")


def crossing_slots(direction: str, *, projected: bool) -> dict[str, str]:
    if str(direction) == "above":
        base = {
            "crossing_direction": "above",
            "comparison_phrase": "greater than",
            "crossing_direction_verb": "rises above",
        }
    elif str(direction) == "below":
        base = {
            "crossing_direction": "below",
            "comparison_phrase": "less than",
            "crossing_direction_verb": "falls below",
        }
    else:
        raise ValueError(f"unsupported crossing direction: {direction}")
    instruction = (
        "The plotted labels form one linear series and the final labels are empty future slots; extend the same step into those future labels."
        if bool(projected)
        else "Follow the plotted labels in displayed order."
    )
    return {**base, "crossing_mode_instruction": instruction}


__all__ = [
    "crossing_slots",
    "endpoint_slots",
    "object_description",
    "ordinal",
    "quoted_join",
    "render_prompt_artifacts",
    "statistic_prompt",
    "streak_slots",
    "turning_slots",
]
