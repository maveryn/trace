"""Prompt assembly for matrix chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import SCENE_ID


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_matrix_v1"
_DEFAULTS = get_scene_defaults(DOMAIN, SCENE_ID)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _DEFAULTS if isinstance(_DEFAULTS, Mapping) else {},
)


_OBJECT_DESCRIPTIONS = {
    "confusion_matrix_counts": "a labeled confusion matrix with printed integer counts in each active cell",
    "annotated_heatmap_table": "a labeled annotated heatmap table with printed integer values in each cell",
    "correlation_matrix_signed": "a labeled signed association matrix with printed integer scores in each cell",
    "triangular_pairwise_matrix": "a labeled triangular pairwise matrix where only the filled cells count",
    "clustered_block_matrix": "a labeled matrix with grouped row and column blocks and printed integer values in each cell",
}


def dynamic_slots(
    dataset: Mapping[str, Any],
    *,
    scene_variant: str,
    supports_unanswerable: bool,
) -> dict[str, Any]:
    qparams = dict(dataset.get("question_params", {}))
    return {
        "object_description": str(_OBJECT_DESCRIPTIONS.get(str(scene_variant), "a labeled matrix with printed values")),
        "row_label": str(qparams.get("row_label", "")),
        "query_axis": str(qparams.get("query_axis", "")),
        "axis_label": str(qparams.get("axis_label", "")),
        "answer_axis": str(qparams.get("answer_axis", "")),
        "extremum_phrase": str(qparams.get("extremum_phrase", "")),
        "comparison_phrase": str(qparams.get("comparison_phrase", "")),
        "unanswerable_instruction": (
            'If the requested row or column label is not visible, answer exactly "unanswerable".'
            if bool(supports_unanswerable)
            else ""
        ),
    }


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(_PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="matrix_cell",
        task_key="matrix_cell_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
