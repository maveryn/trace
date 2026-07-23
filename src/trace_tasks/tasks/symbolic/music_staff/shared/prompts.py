"""Prompt rendering helpers for symbolic music-staff scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_task_prompt_variants

from .state import DOMAIN, SCENE_ID


def build_music_staff_prompt(
    *,
    prompt_defaults: Mapping[str, Any],
    scene_variant: str,
    prompt_key: str,
    branch_key: str,
    prompt_slots: Mapping[str, str],
    instance_seed: int,
    output_modes: Sequence[str] = PROMPT_OUTPUT_MODES,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Render one prompt from task-owned prompt and branch keys."""

    required_keys = (
        "bundle_id",
        "scene_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        f"object_description_{scene_variant}",
        f"answer_hint_{branch_key}",
        f"annotation_hint_{branch_key}",
        f"json_example_{branch_key}",
        f"json_example_answer_only_{branch_key}",
    )
    values = required_group_defaults(prompt_defaults, required_keys, context=f"music-staff prompt defaults for {prompt_key}")
    selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(values["bundle_id"]),
        scene_key=str(values["scene_key"]),
        task_key=str(prompt_key),
        query_key=str(branch_key),
        answer_or_annotation_keys=tuple(str(mode) for mode in output_modes),
        dynamic_slots={
            "object_description": str(values[f"object_description_{scene_variant}"]),
            "json_output_contract": str(values["json_output_contract"]),
            "json_output_contract_answer_only": str(values["json_output_contract_answer_only"]),
            "annotation_hint": str(values[f"annotation_hint_{branch_key}"]),
            "answer_hint": str(values[f"answer_hint_{branch_key}"]),
            "json_example": str(values[f"json_example_{branch_key}"]),
            "json_example_answer_only": str(values[f"json_example_answer_only_{branch_key}"]),
            "target_key": str(prompt_slots.get("target_key", "")),
            "target_marker": str(prompt_slots.get("target_marker", "A")),
            "interval_name": str(prompt_slots.get("interval_name", "")),
            "target_meter_type": str(prompt_slots.get("target_meter_type", "")),
        },
        instance_seed=int(instance_seed),
    )
    artifacts = build_prompt_trace_artifacts(selection)
    return str(artifacts.prompt), dict(artifacts.prompt_variants), {
        "prompt_variant": dict(artifacts.prompt_variant),
        "prompt_variant_active_key": str(artifacts.prompt_variant_active_key),
        "prompt_variants_for_trace": dict(artifacts.prompt_variants_for_trace),
        "bundle_id": str(values["bundle_id"]),
    }
