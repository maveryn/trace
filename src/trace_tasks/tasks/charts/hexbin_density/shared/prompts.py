"""Prompt assembly for hexbin-density charts."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.shared.prompt_variants import (
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from trace_tasks.tasks.charts.hexbin_density.shared.defaults import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_DEFAULTS, SCENE_ID
from trace_tasks.tasks.charts.hexbin_density.shared.state import HexbinDataset


def build_prompt(
    dataset: HexbinDataset,
    *,
    prompt_query_key: str,
    instance_seed: int,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="hexbin_density_scene",
        task_key="hexbin_density_threshold_query",
        query_key=str(prompt_query_key),
        dynamic_slots={
            "object_description": "a hexbin density chart with numeric axes and a discrete density-level legend",
            "density_threshold_phrase": str(dataset.query.threshold_phrase),
            "density_threshold_operator": str(dataset.query.threshold_operator),
            "density_threshold_level": str(dataset.query.threshold_level),
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return (
        str(prompt_artifacts.prompt),
        dict(prompt_artifacts.prompt_variants),
        {
            "template_id": str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        },
    )


__all__ = [
    "build_prompt",
]
