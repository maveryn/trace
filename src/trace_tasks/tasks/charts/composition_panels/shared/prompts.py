"""Prompt slot helpers for composition-panel charts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .annotations import format_annotation_key_list
from .state import DOMAIN, SCENE_ID, CompositionPanelsDataset, CompositionPanelsSelection


BUNDLE_ID = "charts_composition_panels_v1"
SCENE_KEY = "composition_panels_scene"
TASK_KEY = "composition_panels_aggregate_value_query"

JSON_OUTPUT_CONTRACT = 'Use a valid JSON object with keys "annotation" and "answer" in that order for the final answer.'
JSON_OUTPUT_CONTRACT_ANSWER_ONLY = 'Use a valid JSON object with key "answer" for the final answer.'
ANSWER_HINT = 'set "answer" to the requested integer; for percentage-point questions, omit the percent sign'

OBJECT_DESCRIPTION_BY_VARIANT: dict[str, str] = {
    "composition_pie_panels": "several small pie charts. Each panel has the same segment legend, slice numbers are percentages, and each panel title includes its total count",
    "composition_donut_panels": "several small donut charts. Each panel has the same segment legend, slice numbers are percentages, and each panel title includes its total count",
}

def format_quoted(values: Sequence[str]) -> str:
    return ", ".join(f'"{str(value)}"' for value in values)


def object_description(scene_variant: str) -> str:
    return str(OBJECT_DESCRIPTION_BY_VARIANT.get(str(scene_variant), "several small composition charts"))


def prompt_slots(
    *,
    prompt_key: str,
    scene_variant: str,
    dataset: CompositionPanelsDataset,
    selection: CompositionPanelsSelection,
    annotation_points: Mapping[str, Sequence[float]],
    annotation_hint_template: str,
    json_example: str,
    json_example_answer_only: str,
) -> dict[str, Any]:
    extras = dict(selection.trace)
    del prompt_key
    annotation_hint = str(annotation_hint_template).format(
        annotation_key_list=format_annotation_key_list(list(annotation_points.keys())),
        **{str(key): value for key, value in extras.items()},
    )
    answer_hint = str(extras.get("answer_hint", ANSWER_HINT))
    return {
        "object_description": object_description(str(scene_variant)),
        "extremum_word": str(extras.get("extremum_word", "")),
        "top_k": str(extras.get("top_k", "")),
        "rank_segment": str(extras.get("rank_segment", "")),
        "target_segment": str(extras.get("target_segment", "")),
        "target_count": str(extras.get("target_count", "")),
        "segment_a": str(extras.get("segment_a", "")),
        "segment_b": str(extras.get("segment_b", "")),
        "condition_segment": str(extras.get("condition_segment", "")),
        "threshold": str(extras.get("threshold", "")),
        "start_panel": str(extras.get("start_panel", "")),
        "end_panel": str(extras.get("end_panel", "")),
        "segment_list": format_quoted(dataset.segment_labels),
        "json_output_contract": JSON_OUTPUT_CONTRACT,
        "json_output_contract_answer_only": JSON_OUTPUT_CONTRACT_ANSWER_ONLY,
        "annotation_hint": str(annotation_hint),
        "answer_hint": answer_hint,
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }


def render_prompt_artifacts(
    *,
    prompt_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=BUNDLE_ID,
        scene_key=SCENE_KEY,
        task_key=TASK_KEY,
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered)
