"""Shared task-level prompt-variant rendering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ...core.prompts import PromptRenderResult, render_prompt_variants


PROMPT_OUTPUT_MODES: Tuple[str, str] = ("answer_only", "answer_and_annotation")


@dataclass(frozen=True)
class _PromptVariantSelection:
    """Rendered prompt payload with deterministic active-mode selection."""

    prompt: str
    active_mode: str
    active_result: PromptRenderResult
    prompt_variants: Dict[str, str]
    prompt_results: Dict[str, PromptRenderResult]


@dataclass(frozen=True)
class PromptTraceArtifacts:
    """Prompt artifacts consumed by `TaskOutput` and trace `query_spec` fields."""

    prompt: str
    prompt_variants: Dict[str, str]
    prompt_variant_active_key: str
    prompt_variant: Dict[str, Any]
    prompt_variants_for_trace: Dict[str, Dict[str, Any]]


_PromptTraceArtifacts = PromptTraceArtifacts


def render_task_prompt_variants(
    *,
    domain: str,
    scene_id: str | None = None,
    bundle_id: str,
    scene_key: str,
    task_key: str,
    query_key: str | None = None,
    slots: Mapping[str, Any] | None = None,
    dynamic_slots: Mapping[str, Any] | None = None,
    instance_seed: int,
    answer_or_annotation_keys: Sequence[str] = PROMPT_OUTPUT_MODES,
    preferred_mode: str = "answer_and_annotation",
) -> _PromptVariantSelection:
    """Render task prompt variants and choose one active output mode."""
    prompt_results = render_prompt_variants(
        domain=domain,
        scene_id=scene_id,
        bundle_id=bundle_id,
        scene_key=scene_key,
        task_key=task_key,
        query_key=query_key,
        answer_or_annotation_keys=answer_or_annotation_keys,
        slots=slots,
        dynamic_slots=dynamic_slots,
        instance_seed=instance_seed,
    )
    if not prompt_results:
        raise ValueError("render_prompt_variants returned an empty mapping")

    active_mode = str(preferred_mode)
    if active_mode not in prompt_results:
        active_mode = sorted(prompt_results.keys())[0]
    active_result = prompt_results[active_mode]
    prompt_variants = {
        key: result.prompt
        for key, result in sorted(prompt_results.items())
    }
    return _PromptVariantSelection(
        prompt=active_result.prompt,
        active_mode=active_mode,
        active_result=active_result,
        prompt_variants=prompt_variants,
        prompt_results=prompt_results,
    )


def render_scene_prompt_variants(
    *,
    domain: str,
    scene_id: str,
    bundle_id: str,
    scene_key: str,
    task_key: str,
    query_key: str | None = None,
    slots: Mapping[str, Any] | None = None,
    dynamic_slots: Mapping[str, Any] | None = None,
    instance_seed: int,
    answer_or_annotation_keys: Sequence[str] = PROMPT_OUTPUT_MODES,
    preferred_mode: str = "answer_and_annotation",
) -> _PromptVariantSelection:
    """Render prompt variants from a scene-aligned prompt bundle."""

    return render_task_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(bundle_id),
        scene_key=str(scene_key),
        task_key=str(task_key),
        query_key=query_key,
        slots=slots,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
        answer_or_annotation_keys=answer_or_annotation_keys,
        preferred_mode=str(preferred_mode),
    )


def build_prompt_trace_artifacts(selection: _PromptVariantSelection) -> PromptTraceArtifacts:
    """Convert prompt-variant selection into normalized trace/output payload fields."""
    active_mode = str(selection.active_mode)
    active_result = selection.active_result
    prompt = str(selection.prompt)
    prompt_variants = dict(selection.prompt_variants)
    prompt_variant = dict(active_result.metadata)
    prompt_variants_for_trace = {
        key: {"prompt": result.prompt, "metadata": dict(result.metadata)}
        for key, result in sorted(selection.prompt_results.items())
    }
    return PromptTraceArtifacts(
        prompt=prompt,
        prompt_variants=prompt_variants,
        prompt_variant_active_key=active_mode,
        prompt_variant=prompt_variant,
        prompt_variants_for_trace=prompt_variants_for_trace,
    )


def build_prompt_query_spec(
    *,
    prompt_artifacts: PromptTraceArtifacts,
    query_id: str,
    params: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build the canonical prompt-backed `query_spec` trace payload.

    Public task files own the selected query id and task-specific params. This
    helper owns the common trace shape so scene packages do not duplicate prompt
    metadata plumbing.
    """

    query_id_text = str(query_id)
    param_map: Dict[str, Any] = {str(key): value for key, value in dict(params or {}).items()}
    existing_query_id = param_map.get("query_id")
    if existing_query_id is not None and str(existing_query_id) != query_id_text:
        raise ValueError(
            "query_spec params query_id conflicts with selected query_id "
            f"{query_id_text!r}: {existing_query_id!r}"
        )
    param_map["query_id"] = query_id_text

    prompt_variant = dict(prompt_artifacts.prompt_variant)
    prompt_bundle_id = prompt_variant.get("prompt_bundle_id")
    if prompt_bundle_id is None or not str(prompt_bundle_id).strip():
        raise ValueError("prompt artifacts missing prompt_bundle_id")

    return {
        "query_id": query_id_text,
        "template_id": str(prompt_bundle_id),
        "prompt_variant": prompt_variant,
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        "params": param_map,
    }
