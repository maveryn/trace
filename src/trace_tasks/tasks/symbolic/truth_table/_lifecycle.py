"""Scene-private lifecycle helpers for symbolic truth-table tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from ....core.types import TypedValue
from ....core.visual.noise import apply_post_image_noise
from ...base import TaskOutput
from ...shared.output_metadata import default_task_versions
from ...shared.prompt_variants import build_prompt_query_spec
from ..shared.scene_style import (
    make_symbolic_scene_background,
    resolve_symbolic_scene_style,
)
from .shared.defaults import POST_IMAGE_NOISE_DEFAULTS
from .shared.prompts import render_truth_prompt
from .shared.rendering import (
    render_count_scene,
    render_expression_from_rows_scene,
    render_pattern_scene,
)
from .shared.sampling import (
    build_with_retries,
    resolve_truth_table_scene_variant,
)
from .shared.state import RenderedTruthTableScene
from .shared.styles import resolve_render_params

SCENE_ID = "truth_table"
DOMAIN = "symbolic"


@dataclass(frozen=True)
class TruthTableTaskBinding:
    """Task-owned prompt and output binding for one truth-table objective."""

    seed_namespace: str
    internal_query_key: str
    task_prompt_key: str
    object_description_prefix: str
    annotation_hint_key: str
    answer_hint_key: str
    json_example_key: str
    json_example_answer_only_key: str
    answer_type: str
    failure_message: str


@dataclass(frozen=True)
class TruthRenderArtifacts:
    image: Image.Image
    rendered_scene: RenderedTruthTableScene
    render_spec: dict[str, Any]
    render_map: dict[str, Any]
    task_versions: dict[str, str]


def _make_canvas(
    *,
    instance_seed: int,
    namespace: str,
    canvas_width: int,
    canvas_height: int,
) -> tuple[Image.Image, Any, dict[str, Any], dict[str, Any]]:
    scene_style, scene_style_meta = resolve_symbolic_scene_style(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    image, background_meta = make_symbolic_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=scene_style,
    )
    return image, scene_style, dict(scene_style_meta), dict(background_meta)


def _draw_truth_artifacts(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str,
    scene_variant: str,
    render_scene: Callable[..., RenderedTruthTableScene],
    render_kwargs: Mapping[str, Any],
    annotation_source: str,
) -> TruthRenderArtifacts:
    """Render one finalized truth scene while preserving projection coordinates."""

    render_params = resolve_render_params(params, render_defaults)
    canvas, style, scene_style_meta, background_meta = _make_canvas(
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
    )
    rendered_scene = render_scene(
        canvas,
        params=render_params,
        style=style,
        scene_variant=str(scene_variant),
        **dict(render_kwargs),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_spec = {
        "scene_id": SCENE_ID,
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(scene_variant),
        "scene_style": dict(scene_style_meta),
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "truth_table_style": dict(rendered_scene.style_metadata),
    }
    render_map = {
        "image_id": "img0",
        "scene_bbox_px": list(rendered_scene.scene_bbox_px),
        "item_bboxes_px": dict(rendered_scene.item_bboxes),
        "cell_bboxes_px": dict(rendered_scene.cell_bboxes),
        "column_bboxes_px": dict(rendered_scene.column_bboxes),
        "row_bboxes_px": dict(rendered_scene.row_bboxes),
        "annotation_source": str(annotation_source),
    }
    return TruthRenderArtifacts(
        image=image,
        rendered_scene=rendered_scene,
        render_spec=dict(render_spec),
        render_map=dict(render_map),
        task_versions=default_task_versions(),
    )


def _truth_entity_trace_records(
    entities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entity in entities:
        record = {
            "item_id": str(entity.get("item_id", "")),
            "entity_type": str(entity.get("entity_type", "")),
            "role": str(entity.get("role", "")),
            "label": str(entity.get("label", "")),
        }
        if "value" in entity:
            record["value"] = str(entity["value"])
        if "expression" in entity:
            record["expression"] = str(entity["expression"])
        if "pattern" in entity:
            record["pattern"] = str(entity["pattern"])
        if "bbox_px" in entity:
            record["bbox_px"] = list(entity["bbox_px"])
        records.append(record)
    return records


def run_truth_table_instance(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    max_attempts: int,
    gen_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    prompt_defaults: Mapping[str, Any],
    binding: TruthTableTaskBinding,
    dataset_factory: Callable[[int, str, Mapping[str, float]], Any],
    render_scene: Callable[..., RenderedTruthTableScene],
    render_kwargs_factory: Callable[[Any], Mapping[str, Any]],
    annotation_factory: Callable[[RenderedTruthTableScene, Any], Any],
    answer_value_factory: Callable[[Any], Any],
    answer_support_factory: Callable[[Any], Sequence[Any]],
    annotation_item_ids_factory: Callable[[Any], Sequence[str]],
    metadata_factory: Callable[[Any], Mapping[str, Any]],
    annotation_source: str,
) -> TaskOutput:
    """Run the common truth-table render/prompt/output lifecycle."""

    scene_variant, scene_variant_probabilities = resolve_truth_table_scene_variant(
        params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{binding.seed_namespace}.scene_variant",
    )
    dataset = build_with_retries(
        lambda retry_seed: dataset_factory(
            int(retry_seed),
            str(scene_variant),
            scene_variant_probabilities,
        ),
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        failure_message=str(binding.failure_message),
    )
    render_artifacts = _draw_truth_artifacts(
        instance_seed=int(instance_seed),
        params=params,
        render_defaults=render_defaults,
        namespace=f"{binding.seed_namespace}.background",
        scene_variant=str(scene_variant),
        render_scene=render_scene,
        render_kwargs=render_kwargs_factory(dataset),
        annotation_source=str(annotation_source),
    )
    prompt_runtime = render_truth_prompt(
        prompt_defaults,
        domain=DOMAIN,
        scene_id=SCENE_ID,
        scene_variant=str(scene_variant),
        task_key=str(binding.task_prompt_key),
        object_description_key=f"{binding.object_description_prefix}_{scene_variant}",
        annotation_hint_key=str(binding.annotation_hint_key),
        answer_hint_key=str(binding.answer_hint_key),
        json_example_key=str(binding.json_example_key),
        json_example_answer_only_key=str(binding.json_example_answer_only_key),
        instance_seed=int(instance_seed),
        context=f"prompt defaults for {binding.seed_namespace}",
    )
    annotation_artifacts = annotation_factory(render_artifacts.rendered_scene, dataset)
    answer_gt = TypedValue(
        type=str(binding.answer_type), value=answer_value_factory(dataset)
    )
    query_params = {
        "query_id": str(binding.internal_query_key),
        "internal_query_id": str(binding.internal_query_key),
        "internal_query_id_probabilities": {str(binding.internal_query_key): 1.0},
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": {
            str(key): float(value) for key, value in scene_variant_probabilities.items()
        },
        "target_answer_support": [value for value in answer_support_factory(dataset)],
        "question_format": str(binding.internal_query_key),
    }
    prompt_query_spec = build_prompt_query_spec(
        prompt_artifacts=prompt_runtime.artifacts,
        query_id=str(binding.internal_query_key),
        params=query_params,
    )
    metadata = dict(metadata_factory(dataset))
    annotation_item_ids = [str(item) for item in annotation_item_ids_factory(dataset)]
    trace_payload = {
        "scene_ir": {
            "scene_kind": "symbolic_truth_table",
            "entities": _truth_entity_trace_records(
                render_artifacts.rendered_scene.entities
            ),
            "relations": {
                "query_id": str(binding.internal_query_key),
                "internal_query_id": str(binding.internal_query_key),
                "scene_id": SCENE_ID,
                "scene_variant": str(scene_variant),
                "answer_value": answer_gt.value,
            },
        },
        "query_spec": {
            **dict(prompt_query_spec),
            "template_id": str(prompt_runtime.metadata["bundle_id"]),
        },
        "render_spec": dict(render_artifacts.render_spec),
        "render_map": dict(render_artifacts.render_map),
        "execution_trace": {
            **dict(query_params),
            "answer_value": answer_gt.value,
            "answer_type": str(answer_gt.type),
            "annotation_item_ids": list(annotation_item_ids),
            "truth_table_metadata": dict(metadata),
        },
        "witness_symbolic": {
            "type": str(annotation_artifacts.annotation_gt.type),
            "ids": list(annotation_item_ids),
        },
        "projected_annotation": dict(annotation_artifacts.projected_annotation),
        "answer_gt": answer_gt.to_dict(),
        "annotation_gt": annotation_artifacts.annotation_gt.to_dict(),
    }
    return TaskOutput(
        prompt=str(prompt_runtime.prompt),
        answer_gt=answer_gt,
        annotation_gt=annotation_artifacts.annotation_gt,
        image=render_artifacts.image,
        image_id="img0",
        trace_payload=trace_payload,
        task_versions=dict(render_artifacts.task_versions),
        scene_id=SCENE_ID,
        query_id=str(binding.internal_query_key),
        prompt_variants=dict(prompt_runtime.prompt_variants),
    )


__all__ = [
    "TruthTableTaskBinding",
    "render_count_scene",
    "render_expression_from_rows_scene",
    "render_pattern_scene",
    "run_truth_table_instance",
]
