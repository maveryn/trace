"""Private graph-options lifecycle for task-owned objective plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from ....core.seed import hash64, spawn_rng
from ....core.types import TypedValue
from ...base import TaskOutput
from ...shared.fixed_query import select_task_query_id
from ...shared.output_metadata import default_task_versions
from .shared.annotations import project_option_panel_bbox
from .shared.output import trace_sections
from .shared.prompts import build_graph_options_prompt_artifacts
from .shared.rendering import render_graph_options_scene
from .shared.sampling import option_count_from_params, resolve_correct_option_index, resolve_edge_mode, resolve_scene_variant
from .shared.state import GraphOptionsDataset, GraphOptionsDefaults, SCENE_ID


DatasetBuilder = Callable[..., GraphOptionsDataset]
DescriptionBuilder = Callable[[str], str]


@dataclass(frozen=True)
class GraphOptionsObjectivePlan:
    """Task-owned objective hooks consumed by scene-private lifecycle plumbing."""

    owner_id: str
    supported_branch_names: Tuple[str, ...]
    default_branch_name: str
    prompt_key: str
    prompt_bundle_id: str
    sampling_namespace: str
    dataset_factory: DatasetBuilder
    defaults: GraphOptionsDefaults
    object_description: DescriptionBuilder


def sample_dataset_with_retries(
    *,
    factory: DatasetBuilder,
    instance_seed: int,
    params: Mapping[str, Any],
    edge_mode: str,
    max_attempts: int,
    namespace: str,
    factory_kwargs: Mapping[str, Any],
) -> GraphOptionsDataset:
    """Retry objective-owned dataset sampling with a deterministic attempt namespace."""

    last_error: Exception | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            rng = spawn_rng(
                int(hash64(int(instance_seed), f"{namespace}.dataset_seed", int(attempt))),
                f"{namespace}.dataset",
                int(attempt),
            )
            return factory(
                rng=rng,
                params=params,
                instance_seed=int(instance_seed),
                edge_mode=str(edge_mode),
                namespace=str(namespace),
                **dict(factory_kwargs),
            )
        except Exception as exc:  # pragma: no cover - retry path depends on sampled graph structure
            last_error = exc
            continue
    raise RuntimeError("failed to generate graph-options dataset") from last_error


def run_graph_options_plan(
    *,
    plan: GraphOptionsObjectivePlan,
    domain: str,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    background_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
) -> TaskOutput:
    """Run common option-panel rendering while public plans bind the objective."""

    public_branch, public_branch_probabilities, task_params = select_task_query_id(
        instance_seed=int(instance_seed),
        params=params,
        supported_query_ids=plan.supported_branch_names,
        default_query_id=str(plan.default_branch_name),
        task_id=str(plan.owner_id),
        namespace=f"{plan.owner_id}.query",
    )
    scene_variant, scene_variant_probabilities = resolve_scene_variant(
        task_params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(plan.sampling_namespace),
    )
    edge_mode, edge_mode_probabilities = resolve_edge_mode(
        task_params,
        gen_defaults,
        instance_seed=int(instance_seed),
        namespace=str(plan.sampling_namespace),
    )
    option_count = option_count_from_params(task_params, gen_defaults, defaults=plan.defaults)
    correct_option_index = resolve_correct_option_index(
        task_params,
        instance_seed=int(instance_seed),
        option_count=int(option_count),
        namespace=f"{plan.sampling_namespace}.option",
    )
    dataset = sample_dataset_with_retries(
        factory=plan.dataset_factory,
        instance_seed=int(instance_seed),
        params=task_params,
        edge_mode=str(edge_mode),
        max_attempts=int(max_attempts),
        namespace=str(plan.sampling_namespace),
        factory_kwargs={
            "gen_defaults": gen_defaults,
            "option_count": int(option_count),
            "correct_option_index": int(correct_option_index),
            "defaults": plan.defaults,
        },
    )
    rendered = render_graph_options_scene(
        dataset=dataset,
        scene_variant=str(scene_variant),
        params=task_params,
        render_defaults=render_defaults,
        background_defaults=background_defaults,
        noise_defaults=noise_defaults,
        instance_seed=int(instance_seed),
        namespace=str(plan.sampling_namespace),
    )
    prompt_artifacts = build_graph_options_prompt_artifacts(
        domain=str(domain),
        bundle_id=str(plan.prompt_bundle_id),
        prompt_key=str(plan.prompt_key),
        object_description=str(plan.object_description(str(edge_mode))),
        instance_seed=int(instance_seed),
    )
    annotation_projection = project_option_panel_bbox(rendered.bbox_map, dataset.correct_option_panel_id)
    annotation_bbox = list(annotation_projection["bbox"])
    sections = trace_sections(
        dataset=dataset,
        rendered=rendered,
        prompt_artifacts=prompt_artifacts,
        prompt_bundle_id=str(plan.prompt_bundle_id),
        prompt_key=str(plan.prompt_key),
        edge_mode_probabilities=edge_mode_probabilities,
        scene_variant_probabilities=scene_variant_probabilities,
        annotation_projection=annotation_projection,
    )
    scene_ir = dict(sections["scene_ir"])
    scene_ir["task_id"] = str(plan.owner_id)
    scene_ir["relations"] = {
        "query_id": str(public_branch),
        "prompt_key": str(plan.prompt_key),
        **dict(scene_ir["relations"]),
    }
    query_spec = {
        "task_id": str(plan.owner_id),
        "query_id": str(public_branch),
        "template_id": str(plan.prompt_bundle_id),
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants": dict(prompt_artifacts.prompt_variants_for_trace),
        "params": {
            "query_id": str(public_branch),
            "query_id_probabilities": dict(public_branch_probabilities),
            "prompt_key": str(plan.prompt_key),
            **dict(sections["axis_parameter_fields"]),
        },
    }
    execution_trace = {
        "task_id": str(plan.owner_id),
        "scene_id": SCENE_ID,
        "query_id": str(public_branch),
        "prompt_key": str(plan.prompt_key),
        "answer": str(dataset.answer_option_label),
        "query_structure_spec": dict(dataset.source_structure_spec),
        **dict(sections["execution_trace"]),
    }
    return TaskOutput(
        prompt=str(prompt_artifacts.prompt),
        answer_gt=TypedValue(type="option_letter", value=str(dataset.answer_option_label)),
        annotation_gt=TypedValue(type="bbox", value=annotation_bbox),
        image=rendered.image,
        image_id="img0",
        trace_payload={
            "scene_ir": scene_ir,
            "query_spec": query_spec,
            "render_spec": sections["render_spec"],
            "render_map": sections["render_map"],
            "execution_trace": execution_trace,
            "witness_symbolic": sections["witness_symbolic"],
            "projected_annotation": sections["projected_annotation"],
        },
        task_versions=default_task_versions(),
        scene_id=SCENE_ID,
        query_id=str(public_branch),
        prompt_variants=dict(prompt_artifacts.prompt_variants),
    )


__all__ = ["GraphOptionsObjectivePlan", "run_graph_options_plan", "sample_dataset_with_retries"]
