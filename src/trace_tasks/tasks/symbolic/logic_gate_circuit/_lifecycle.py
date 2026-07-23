"""Shared lifecycle assembly for single-circuit logic-gate count tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from PIL import Image

from ....core.types import TypedValue

from .shared.annotations import projected_bbox_set
from .shared.output import circuit_trace
from .shared.prompts import build_logic_gate_prompt
from .shared.rendering import render_payload_sections, render_single_circuit_bundle, rounded_render_maps
from .shared.state import LogicCircuitSpec, SCENE_ID


@dataclass(frozen=True)
class SingleCircuitCountArtifacts:
    prompt: str
    answer_gt: TypedValue
    annotation_gt: TypedValue
    image: Image.Image
    trace_payload: dict[str, Any]
    prompt_variants: dict[str, str]


def build_single_circuit_count_artifacts(
    *,
    task_id: str,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    instance_seed: int,
    params: Mapping[str, Any],
    circuit: LogicCircuitSpec,
    annotation_item_ids: Sequence[str],
    answer_value: int,
    scene_variant: str,
    prompt_key: str,
    branch_key: str,
    annotation_hint_key: str,
    answer_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    prompt_extra_slots: Mapping[str, Any] | None,
    query_payload: Mapping[str, Any],
    execution_extra: Mapping[str, Any],
    logic_gate_metadata: Mapping[str, Any],
    public_query_value: str,
    show_fixed_input_values: bool,
) -> SingleCircuitCountArtifacts:
    """Assemble rendering, bbox-set annotation, prompt metadata, and trace payload."""

    render_bundle = render_single_circuit_bundle(
        instance_seed=int(instance_seed),
        circuit=circuit,
        params=params,
        render_defaults=render_defaults,
        noise_defaults=noise_defaults,
        show_fixed_input_values=bool(show_fixed_input_values),
    )
    prompt, prompt_variants, prompt_meta, _prompt_artifacts = build_logic_gate_prompt(
        domain=str(domain),
        prompt_defaults=prompt_defaults,
        scene_variant=str(scene_variant),
        prompt_key=str(prompt_key),
        branch_key=str(branch_key),
        instance_seed=int(instance_seed),
        annotation_hint_key=str(annotation_hint_key),
        answer_hint_key=str(answer_hint_key),
        json_example_key=str(json_example_key),
        json_example_answer_only_key=str(json_example_answer_only_key),
        extra_dynamic_slots=prompt_extra_slots,
    )
    item_bboxes, output_points, signal_points = rounded_render_maps(render_bundle)
    annotation_bboxes = [list(item_bboxes[str(item_id)]) for item_id in annotation_item_ids]
    projected_annotation = projected_bbox_set(annotation_bboxes)
    answer_gt = TypedValue(type="integer", value=int(answer_value))
    annotation_gt = TypedValue(type="bbox_set", value=list(annotation_bboxes))
    render_spec, render_map = render_payload_sections(
        render_bundle,
        item_bboxes=item_bboxes,
        output_points=output_points,
        signal_points=signal_points,
        annotation_source="item_bboxes_px",
    )
    render_spec["scene_variant"] = str(scene_variant)
    payload = dict(query_payload)
    trace_payload = {
        "scene_ir": {
            "scene_kind": SCENE_ID,
            "entities": [dict(entity) for entity in render_bundle.rendered.entities],
            "relations": {
                **dict(payload),
                "answer_value": int(answer_value),
            },
        },
        "query_spec": {
            "query_id": str(public_query_value),
            "template_id": str(prompt_meta["bundle_id"]),
            "prompt_variant": dict(prompt_meta["prompt_variant"]),
            "prompt_variant_active_key": str(prompt_meta["prompt_variant_active_key"]),
            "prompt_variants": dict(prompt_meta["prompt_variants_for_trace"]),
            "params": dict(payload),
        },
        "render_spec": dict(render_spec),
        "render_map": dict(render_map),
        "execution_trace": {
            **dict(payload),
            "task_id": str(task_id),
            "answer_value": int(answer_value),
            "answer_type": "integer",
            "annotation_item_ids": [str(item_id) for item_id in annotation_item_ids],
            "logic_gate_metadata": dict(logic_gate_metadata),
            "source_circuit": circuit_trace(circuit),
            **dict(execution_extra),
        },
        "witness_symbolic": {"type": "bbox_set", "value": list(annotation_bboxes)},
        "projected_annotation": dict(projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_gt.to_dict(),
    }
    return SingleCircuitCountArtifacts(
        prompt=str(prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_gt,
        image=render_bundle.image,
        trace_payload=trace_payload,
        prompt_variants=dict(prompt_variants),
    )
