"""Private lifecycle plumbing for flow-network public objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import PromptTraceArtifacts
from .shared.output import create_flow_network_scene_bundle, trace_sections
from .shared.prompts import build_flow_network_prompt_artifacts
from .shared.state import FlowNetworkDefaults, FlowNetworkSceneBundle, ResolvedFlowNetworkAxes, SCENE_ID


ObjectiveResolver = Callable[[int, Mapping[str, Any]], Tuple[ResolvedFlowNetworkAxes, int, Dict[str, float]]]
TraceFieldResolver = Callable[[FlowNetworkSceneBundle], Mapping[str, Any]]


def _empty_trace_fields(_bundle: FlowNetworkSceneBundle) -> Mapping[str, Any]:
    """Return no extra public trace fields for objectives that need no aliases."""

    return {}


@dataclass(frozen=True)
class FlowNetworkObjectivePlan:
    """Public-owned objective contract consumed by scene-private plumbing."""

    owner_id: str
    supported_branch_names: Tuple[str, ...]
    default_branch_name: str
    prompt_query_key: str
    sampling_namespace: str
    resolve_objective_axes: ObjectiveResolver
    extra_trace_fields: TraceFieldResolver = _empty_trace_fields


def _trace_payload(
    *,
    plan: FlowNetworkObjectivePlan,
    bundle: FlowNetworkSceneBundle,
    prompt_artifacts: PromptTraceArtifacts,
    public_branch_name: str,
    public_branch_probabilities: Mapping[str, float],
    answer_value: int,
    answer_probabilities: Mapping[str, float],
    prompt_bundle_id: str,
) -> Dict[str, Any]:
    """Attach public objective metadata to neutral flow-network trace sections."""

    sections = trace_sections(bundle)
    scene_payload = sections["scene_ir"]
    scene_payload["task_id"] = str(plan.owner_id)
    extra_fields = dict(plan.extra_trace_fields(bundle))
    scene_payload["relations"] = {
        "query_id": str(public_branch_name),
        "objective": str(plan.prompt_query_key),
        **dict(scene_payload["relations"]),
        **dict(extra_fields),
    }

    execution = sections["execution_trace"]
    execution.update(
        {
            "task_id": str(plan.owner_id),
            "scene_id": SCENE_ID,
            "query_id": str(public_branch_name),
            "objective": str(plan.prompt_query_key),
            "answer": int(answer_value),
            "annotation_edges": [list(edge) for edge in bundle.annotation_edges],
            **dict(extra_fields),
        }
    )

    return {
        "scene_ir": scene_payload,
        "query_spec": {
            "task_id": str(plan.owner_id),
            "query_id": str(public_branch_name),
            "template_id": str(prompt_bundle_id),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
            "params": {
                "query_id": str(public_branch_name),
                "objective": str(plan.prompt_query_key),
                "query_id_probabilities": dict(public_branch_probabilities),
                "target_answer": int(answer_value),
                "target_answer_probabilities": dict(answer_probabilities),
                **dict(sections["axis_parameter_fields"]),
            },
        },
        "render_spec": sections["render_spec"],
        "render_map": sections["render_map"],
        "execution_trace": execution,
        "witness_symbolic": sections["witness_symbolic"],
        "projected_annotation": sections["projected_annotation"],
    }


def run_flow_network_plan(
    *,
    plan: FlowNetworkObjectivePlan,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_bundle_id: str,
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    defaults: FlowNetworkDefaults,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run common flow-network rendering while preserving public objective ownership."""

    public_branch_name, public_branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=plan.supported_branch_names,
        default_query_id=str(plan.default_branch_name),
        task_id=str(plan.owner_id),
        namespace=f"{plan.owner_id}.query",
    )
    axes, answer_value, answer_probabilities = plan.resolve_objective_axes(int(instance_seed), task_params)
    bundle = create_flow_network_scene_bundle(
        instance_seed=int(instance_seed),
        params=task_params,
        render_defaults=render_defaults,
        axes=axes,
        namespace=str(plan.sampling_namespace),
        max_attempts=int(max_attempts),
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        defaults=defaults,
    )
    prompt_artifacts = build_flow_network_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(prompt_bundle_id),
        prompt_key=str(plan.prompt_query_key),
        dynamic_slots={"object_description": "a directed capacity network with source S and sink T"},
        instance_seed=int(instance_seed),
    )
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="integer", value=int(answer_value)),
        annotation_gt=TypedValue(type="segment_set", value=list(bundle.annotation_segments)),
        image=bundle.render.image,
        image_id="img0",
        trace_payload=_trace_payload(
            plan=plan,
            bundle=bundle,
            prompt_artifacts=prompt_artifacts,
            public_branch_name=str(public_branch_name),
            public_branch_probabilities=public_branch_probabilities,
            answer_value=int(answer_value),
            answer_probabilities=answer_probabilities,
            prompt_bundle_id=str(prompt_bundle_id),
        ),
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_branch_name),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = ["FlowNetworkObjectivePlan", "run_flow_network_plan"]
