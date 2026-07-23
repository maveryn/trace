"""Prompt assembly for multiseries chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_DEFAULTS, SCENE_ID


TASK_PROMPT_KEY = "multiseries_query"

_OBJECT_DESCRIPTIONS = {
    "grouped_bar": (
        "a grouped bar chart with labeled category groups on the horizontal axis and a legend. "
        "Each colored bar height gives that series value for its category"
    ),
    "grouped_horizontal_bar": (
        "a grouped horizontal bar chart with labeled category groups on the vertical axis and a legend. "
        "Each colored bar length gives that series value for its category"
    ),
    "multi_line": (
        "a multi-line chart with labeled categories on the horizontal axis and a legend. "
        "Each colored series has one point per category, and the relevant values are the points' y-values"
    ),
    "grouped_lollipop": (
        "a grouped lollipop chart with labeled categories on the horizontal axis and a legend. "
        "Each colored point gives that series value for its category"
    ),
}


def object_description(scene_variant: str) -> str:
    return str(_OBJECT_DESCRIPTIONS.get(str(scene_variant), "a multiseries chart with labeled categories and a legend"))


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="multiseries_chart",
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


def ordinal(value: int) -> str:
    if 10 <= int(value) % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(int(value) % 10, "th")
    return f"{int(value)}{suffix}"


def ranked_phrase(rank: int, base: str) -> str:
    if int(rank) == 1:
        return str(base)
    return f"{ordinal(int(rank))} {str(base)}"


def comparison_phrase(comparison: str) -> str:
    if str(comparison) == "greater_than":
        return "higher than"
    if str(comparison) == "less_than":
        return "lower than"
    raise ValueError(f"unsupported comparison: {comparison}")


def change_prompt_slots(change_direction: str | None) -> Dict[str, str]:
    if str(change_direction) == "increase":
        return {
            "change_direction": "increase",
            "change_expression": "the second queried series minus the first queried series",
            "change_action": "increase from first to second",
        }
    if str(change_direction) == "decrease":
        return {
            "change_direction": "decrease",
            "change_expression": "the first queried series minus the second queried series",
            "change_action": "decrease from first to second",
        }
    return {"change_direction": "", "change_expression": "", "change_action": ""}


def change_measure_prompt_slots(change_measure: str | None, change_direction: str | None) -> Dict[str, str]:
    if str(change_measure) == "directional_change":
        return {
            "change_measure": "directional change",
            "change_measure_prompt": str(change_prompt_slots(change_direction)["change_action"]),
        }
    if str(change_measure) == "absolute_gap":
        return {
            "change_measure": "absolute gap",
            "change_measure_prompt": "absolute gap",
        }
    return {"change_measure": "", "change_measure_prompt": ""}


def extremum_prompt_slots(extremum_direction: str | None, *, answer_rank: int) -> Dict[str, str]:
    if str(extremum_direction) == "smallest":
        return {
            "extremum_direction": "smallest",
            "ranked_extremum": ranked_phrase(int(answer_rank), "smallest"),
        }
    return {
        "extremum_direction": "largest",
        "ranked_extremum": ranked_phrase(int(answer_rank), "largest"),
    }


def ratio_measure_prompt_slots(
    ratio_measure: str | None,
    *,
    target_series: str,
    numerator_series: str,
    denominator_series: str,
) -> Dict[str, str]:
    if str(ratio_measure) == "series_share":
        return {
            "ratio_measure": "series share",
            "ratio_measure_prompt": f'percentage share for "{str(target_series)}" out of each category total',
        }
    if str(ratio_measure) == "pair_ratio":
        return {
            "ratio_measure": "pair ratio",
            "ratio_measure_prompt": f'percentage ratio of "{str(numerator_series)}" to "{str(denominator_series)}"',
        }
    return {"ratio_measure": "", "ratio_measure_prompt": ""}

__all__ = [
    "TASK_PROMPT_KEY",
    "build_prompt_artifacts",
    "change_measure_prompt_slots",
    "change_prompt_slots",
    "comparison_phrase",
    "extremum_prompt_slots",
    "object_description",
    "ordinal",
    "ranked_phrase",
    "ratio_measure_prompt_slots",
]
